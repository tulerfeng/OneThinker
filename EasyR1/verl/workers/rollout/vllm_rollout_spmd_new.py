# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import itertools
from contextlib import contextmanager
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from vllm import LLM, RequestOutput, SamplingParams

from ...protocol import DataProto
from ...utils import torch_functional as VF
# 关键：继续使用项目内的数据处理函数
from ...utils.dataset import process_image, process_video
from ...utils.torch_dtypes import PrecisionType
from .base import BaseRollout
from .config import RolloutConfig

# ---------- 日志工具：每行唯一，避免 Ray 折叠 ----------
_LOG_COUNTER = itertools.count()

def _log_unique(msg: str) -> None:
    """
    每行日志带 rank/pid/时间 + 全局单调计数器 + 纳秒时间戳，保证跨进程/跨时刻唯一，杜绝 Ray 聚合折叠。
    """
    rank = os.getenv("RANK", "0")
    pid = os.getpid()
    ts = time.strftime("%H:%M:%S")
    nsec = time.time_ns()  # 纳秒时间
    cnt = next(_LOG_COUNTER)  # 进程内单调递增
    # uid 放在行尾，既唯一又不影响阅读
    print(f"[rank={rank} pid={pid} t={ts}] {msg}  #uid={rank}-{pid}-{cnt}-{nsec}", flush=True)

def _pp_list_head(name: str, items: list, n: int = 10) -> None:
    """
    只打印列表前 n 条，逐条带索引，行行唯一；如果列表更长，追加一条“… and X more”。
    """
    if items is None:
        _log_unique(f"{name}: None")
        return
    _log_unique(f"{name}: len={len(items)} (showing first {min(n, len(items))})")
    for i, it in enumerate(items[:n]):
        _log_unique(f"{name}[{i}]: {it}")
    if len(items) > n:
        _log_unique(f"{name}: ... and {len(items) - n} more")

def _repeat_interleave(value: Union[torch.Tensor, np.ndarray, list], repeats: int):
    """repeat the elements, supports tensor / ndarray / list"""
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    elif isinstance(value, np.ndarray):
        return np.repeat(value, repeats, axis=0)
    elif isinstance(value, list):
        out = []
        for v in value:
            out.extend([v] * repeats)
        return out
    else:
        raise TypeError(f"Unsupported type for repeat_interleave: {type(value)}")


def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[dict[int, float]]:
    # enforce vllm to not output image token
    # TODO: add video token
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}
    else:
        return None


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
    ):
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        engine_kwargs = {}
        if processor is not None:  # only VLMs have processor
            engine_kwargs["disable_mm_preprocessor_cache"] = True
            if config.limit_images:
                # 同时限制 image/video 每样本个数
                engine_kwargs["limit_mm_per_prompt"] = {"image": 1, "video": 1}

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format="dummy",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": False,
            "logit_bias": _get_logit_bias(processor),
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        # 保留原有打印 -> 替换为带唯一前缀
        _log_unique(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)

        # 轻量 DEBUG，确认基础维度
        _log_unique(
            f"[DEBUG] bsz: {batch_size} "
            f"len(raw_prompt_ids): {len(batch_raw_prompt_ids)} "
            f"len(multi_modal_data): {(None if batch_multi_modal_data is None else len(batch_multi_modal_data))}"
        )

        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        # 逐样本构造 vllm_inputs，并记录对齐后的非张量输出
        vllm_inputs = []
        mm_types = [None] * batch_size
        images_aligned = [None] * batch_size
        videos_aligned = [None] * batch_size
        mm_kwargs_aligned = [None] * batch_size  # 每个样本的 mm_processor_kwargs（仅 video 用）

        # ==== 新增 DEBUG：打印 multi_modal_data 的“声明类型分布” ====
        if batch_multi_modal_data is not None:
            img_decl_idx, vid_decl_idx, none_decl_idx = [], [], []
            for i, m in enumerate(batch_multi_modal_data):
                if isinstance(m, dict):
                    if ("images" in m and m["images"]) or ("image" in m and m.get("image")):
                        img_decl_idx.append(i)
                    elif ("videos" in m and m["videos"]) or ("video" in m and m.get("video")):
                        vid_decl_idx.append(i)
                    else:
                        none_decl_idx.append(i)
                else:
                    none_decl_idx.append(i)

            _log_unique(
                f"[DEBUG] declared types in multi_modal_data: "
                f"{{'image': {len(img_decl_idx)}, 'video': {len(vid_decl_idx)}, 'none': {len(none_decl_idx)}, 'bsz': {batch_size}}}"
            )
            # 逐条打印前若干条目，避免一次性大列表触发折叠
            head_n = 10
            sample_decl = []
            for i in range(min(head_n, batch_size)):
                kind = (
                    "image" if i in img_decl_idx else
                    "video" if i in vid_decl_idx else
                    "none"
                )
                sample_decl.append({"idx": i, "kind": kind, "raw": batch_multi_modal_data[i]})
            _pp_list_head("multi_modal_data.head", sample_decl, n=head_n)

        if batch_multi_modal_data is not None:
            img_cnt, vid_cnt, txt_cnt = 0, 0, 0
            for i, (raw_prompt_ids, single_mm) in enumerate(zip(batch_raw_prompt_ids, batch_multi_modal_data)):
                # 判定样本类型 & 调用各自处理函数
                mm_type = None
                media = None
                mm_kwargs = {}

                if isinstance(single_mm, dict):
                    # image 分支
                    if ("images" in single_mm and single_mm["images"]) or ("image" in single_mm and single_mm.get("image")):
                        img_path = single_mm["images"][0] if "images" in single_mm else single_mm["image"]
                        media = process_image(
                            img_path,
                            prompts.meta_info.get("min_pixels"),
                            prompts.meta_info.get("max_pixels"),
                        )
                        mm_type = "image"
                        mm_kwargs = {}  # 图像无需 kwargs
                        img_cnt += 1

                    # video 分支
                    elif ("videos" in single_mm and single_mm["videos"]) or ("video" in single_mm and single_mm.get("video")):
                        vid_path = single_mm["videos"][0] if "videos" in single_mm else single_mm["video"]
                        # 关键：按样本取回 (processed_video, sample_fps)
                        processed_video, sample_fps = process_video(
                            vid_path,
                            prompts.meta_info.get("min_pixels"),
                            prompts.meta_info.get("max_pixels"),
                            prompts.meta_info.get("video_fps", 2.0),
                            return_fps=True,
                        )
                        media = processed_video
                        # 兼容 float 或 [float] 两种返回
                        if isinstance(sample_fps, (list, tuple, np.ndarray)):
                            if len(sample_fps) > 0:
                                sample_fps = float(sample_fps[0])
                            else:
                                sample_fps = float(prompts.meta_info.get("video_fps", 2.0))
                        else:
                            sample_fps = float(sample_fps)

                        # 逐样本 mm_processor_kwargs（至少带 fps；常见再加 do_sample_frames）
                        mm_kwargs = {
                            "fps": sample_fps,
                            "do_sample_frames": False,
                        }
                        mm_type = "video"
                        vid_cnt += 1
                    else:
                        # 理论上你的数据没有纯文本，但保持健壮
                        mm_type = None
                        txt_cnt += 1

                mm_types[i] = mm_type

                if mm_type == "image":
                    images_aligned[i] = media
                    mm_kwargs_aligned[i] = {}
                    vllm_inputs.append({
                        "prompt_token_ids": list(raw_prompt_ids),
                        "multi_modal_data": {"image": media},
                    })
                elif mm_type == "video":
                    videos_aligned[i] = media
                    mm_kwargs_aligned[i] = mm_kwargs or {}
                    vllm_inputs.append({
                        "prompt_token_ids": list(raw_prompt_ids),
                        "multi_modal_data": {"video": media},
                        "mm_processor_kwargs": mm_kwargs_aligned[i],
                    })
                else:
                    vllm_inputs.append({"prompt_token_ids": list(raw_prompt_ids)})

            # 轻量统计（原有）
            _log_unique(f"[DEBUG] counts: {{'image': {img_cnt}, 'video': {vid_cnt}, 'text_only': {txt_cnt}, 'bsz': {batch_size}}}")

            # ==== 新增 DEBUG：逐样本映射（前 16 个），更直观 ====
            head_n = min(batch_size, 16)
            _log_unique("[DEBUG] sample-wise head mapping (idx, type, has_img, has_vid, fps_or_None):")
            for i in range(head_n):
                t = mm_types[i]
                has_img = images_aligned[i] is not None
                has_vid = videos_aligned[i] is not None
                fps_i = None
                if isinstance(mm_kwargs_aligned[i], dict):
                    fps_i = mm_kwargs_aligned[i].get("fps", None)
                _log_unique(f"  idx={i:02d} type={t} img={has_img} vid={has_vid} fps={fps_i}")

            # ==== 新增 DEBUG：全量索引核对 ====
            vid_idx = [i for i, t in enumerate(mm_types) if t == "video"]
            img_idx = [i for i, t in enumerate(mm_types) if t == "image"]
            none_idx = [i for i, t in enumerate(mm_types) if t is None]
            _log_unique(
                f"[DEBUG] processed types (after process_*): "
                f"{{'image': {len(img_idx)}, 'video': {len(vid_idx)}, 'none': {len(none_idx)}, 'bsz': {batch_size}}}"
            )
            if len(vid_idx) <= 32:
                _pp_list_head("video_indices.processed", vid_idx, n=len(vid_idx))
            if len(img_idx) <= 32:
                _pp_list_head("image_indices.processed", img_idx, n=len(img_idx))
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_ids)} for raw_ids in batch_raw_prompt_ids]

        # 原来的两条大列表打印 -> 替换为 head 打印，避免折叠
        _pp_list_head("multi_modal_data.debug_head", batch_multi_modal_data, n=10)
        _pp_list_head("mm_processor_kwargs.debug_head", mm_kwargs_aligned, n=10)

        # 运行 vLLM
        with self.update_sampling_params(**prompts.meta_info):
            completions: list[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=self.use_tqdm
            )
            response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            if self.sampling_params.n > 1:
                # 注意：这里必须同时 repeat multi_modal_data，否则下游会拿到 None
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                mm_types = _repeat_interleave(mm_types, self.sampling_params.n)
                images_aligned = _repeat_interleave(images_aligned, self.sampling_params.n)
                videos_aligned = _repeat_interleave(videos_aligned, self.sampling_params.n)
                mm_kwargs_aligned = _repeat_interleave(mm_kwargs_aligned, self.sampling_params.n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n)
                    _log_unique(
                        f"[DEBUG] after repeat n>1: new_bsz: {batch_size} "
                        f"len(multi_modal_data): {len(batch_multi_modal_data)}"
                    )

        # 拼接响应 & 更新 masks/pos_ids
        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.ndim == 3:  # qwen2vl mrope: (batch_size, 4, seq_length)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # 对齐并输出 non-tensor
        if batch_multi_modal_data is not None:
            non_tensor_batch_out = {
                "multi_modal_data": batch_multi_modal_data,  # 原样返回，兼容下游
                "mm_types": mm_types,
                "images": images_aligned,
                "videos": videos_aligned,
                "mm_processor_kwargs": mm_kwargs_aligned,
            }
        else:
            non_tensor_batch_out = {
                "mm_types": mm_types,
                "images": images_aligned,
                "videos": videos_aligned,
                "mm_processor_kwargs": mm_kwargs_aligned,
            }

        # 兜底对齐到 bsz（以防万一）
        for k, v in list(non_tensor_batch_out.items()):
            if isinstance(v, list):
                if len(v) < batch_size:
                    non_tensor_batch_out[k] = v + [None] * (batch_size - len(v))
                elif len(v) > batch_size:
                    non_tensor_batch_out[k] = v[:batch_size]

        # 额外 DEBUG：确认 multi_modal_data 与 bsz 对齐
        _log_unique(
            f"[DEBUG] final shapes: bsz: {batch_size} "
            f"len(non_tensor.multi_modal_data): "
            f"{(None if 'multi_modal_data' not in non_tensor_batch_out else len(non_tensor_batch_out['multi_modal_data']))} "
            f"len(mm_processor_kwargs): {len(non_tensor_batch_out['mm_processor_kwargs'])}"
        )

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch_out, meta_info=prompts.meta_info)
