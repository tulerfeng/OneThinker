# 

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

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from . import torch_functional as VF


QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please answer this question based on the visual content."
    "Provide your thinking process between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
    "At the end, you must output the final answer in the format:\n"
    "<answer><your_answer_here></answer>\n"
)

TYPE_TEMPLATE = {
    "multiple choice": (
        "Please provide only the single option letter (e.g., A, B, C, D, etc.) "
        "within the <answer>...</answer> tags.\n"
        "Example:\n<answer>A</answer>"
    ),
    "numerical": (
        "Please provide only the numerical value within the <answer>...</answer> tags.\n"
        "Example:\n<answer>3.14</answer>"
    ),
    "OCR": (
        "Please provide only the transcribed text within the <answer>...</answer> tags.\n"
        "Example:\n<answer>Hello World</answer>"
    ),
    "open-ended": (
        "Please provide only your text answer within the <answer>...</answer> tags.\n"
        "Example:\n<answer>The capital of France is Paris.</answer>"
    ),
    "regression": (
        "Please provide only the numerical value within the <answer>...</answer> tags.\n"
        "Example:\n<answer>42.7</answer>"
    ),
    "math": (
        "Please provide only the final result (a number or LaTeX formula) within the <answer>...</answer> tags.\n"
        "Example:\n<answer>$$-\\dfrac{3}{2}$$</answer>"
    ),
    "temporal grounding": (
        "Please provide only the time span in seconds as JSON within the <answer>...</answer> tags.\n"
        "Example:\n<answer>{\"time\": [12.3, 25.7]}</answer>"
    ),
    "spatial grounding": (
        "Please provide only the bounding box as JSON with key 'boxes' within the <answer>...</answer> tags.\n"
        "Example:\n<answer>{\"boxes\": [35, 227, 437, 932]}</answer>"
    ),
    "spatial-temporal grounding": (
        "Please provide only the time span in seconds and bounding boxes as JSON within the <answer>...</answer> tags.\n"
        "You MUST output one bounding box for every integer second within the given time span (inclusive).\n"
        "Example:\n"
        "<answer>{\"time\": [8.125, 13.483], \"boxes\": {\"9\": [317, 422, 582, 997], "
        "\"10\": [332, 175, 442, 369], \"11\": [340, 180, 450, 370]}}</answer>\n"
        "Note: Each key in 'boxes' must be an integer second within the span, and its value must be a 4-number bounding box [x1, y1, x2, y2]."
    ),
    "tracking": (
        "Please track the target object throughout the video and provide one bounding box per second, "
        "ONLY up to 32 seconds, within the <answer>...</answer> tags.\n"
        "Example:\n"
        "<answer>{\"boxes\": {\"1\": [405, 230, 654, 463], \"2\": [435, 223, 678, 446], ..., "
        "\"32\": [415, 203, 691, 487]}}</answer>\n"
        "Note: Each key in 'boxes' must correspond to a second (1, 2, 3, ..., 32) and contain a 4-number bounding box [x1, y1, x2, y2]."
    ),
    "segmentation_image": (
        "This task prepares inputs for image object segmentation with a specialized model (e.g., SAM2).\n"
        "Please provide ONE bounding box, 3 positive points (clearly INSIDE the object), and 3 negative points "
        "(clearly OUTSIDE the object) within the <answer>...</answer> tags.\n"
        "Choose informative points that help distinguish object vs. background. Prefer negatives on clear non-object "
        "pixels INSIDE the box when safe; otherwise place them just outside on obvious background. "
        "Negatives must NEVER be on the object or on its boundary.\n"
        "Example:\n"
        "<answer>{\"boxes\": [x1, y1, x2, y2], \"positive_points\": [[x,y],[x,y],[x,y]], "
        "\"negative_points\": [[x,y],[x,y],[x,y]]}</answer>"
    ),
    "segmentation_video": (
        "This task prepares inputs for video object segmentation with a specialized model (e.g., SAM2).\n"
        "Please select ONE representative time (in seconds), and provide ONE bounding box, "
        "3 positive points (clearly INSIDE the object), and 3 negative points (clearly OUTSIDE the object) "
        "within the <answer>...</answer> tags.\n"
        "Choose informative points that help distinguish object vs. background. Prefer negatives on clear non-object "
        "pixels INSIDE the box when safe; otherwise place them just outside on obvious background. "
        "Negatives must NEVER be on the object or on its boundary.\n"
        "Example:\n"
        "<answer>{\"time\": <time_in_seconds>, \"boxes\": [x1, y1, x2, y2], "
        "\"positive_points\": [[x,y],[x,y],[x,y]], \"negative_points\": [[x,y],[x,y],[x,y]]}</answer>"
    )
}


def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    # print(max_pixels)

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def process_video(
    video: str, min_pixels: int = 4*32*32, max_pixels: int = 64*32*32, max_frames: int = 128, video_fps: float = 2, return_fps: bool = False
):
    vision_info = {"video": video, "min_pixels": min_pixels, "max_pixels": max_pixels, "max_frames": max_frames, "fps": video_fps}
    return fetch_video(vision_info, image_patch_size=16, return_video_sample_fps=return_fps, return_video_metadata=return_fps)


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )


    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        data_type = (example.get("data_type") or "").strip().lower()
        pt = example.get("problem_type") or ""
        question = prompt_str  

        if (pt == "multiple choice") and isinstance(example.get("options"), list) and example["options"]:
            opts = "\n".join(example["options"])
            question = f"{question}\nOptions:\n{opts}"

        if pt == "segmentation":
            type_key = "segmentation_video" if data_type == "video" else "segmentation_image"
        else:
            type_key = pt

        tail = TYPE_TEMPLATE.get(type_key, "")
        prompt_str = QUESTION_TEMPLATE.format(Question=question) + tail

        if self.image_key in example and isinstance(example.get(self.image_key), list) and len(example.get(self.image_key)) > 0:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            # print(content_list)

            return [{"role": "user", "content": content_list}]
        elif self.video_key in example and isinstance(example.get(self.video_key), list) and len(example.get(self.video_key)) > 0:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            # print(content_list)

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]


    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        if self.image_key in example and isinstance(example.get(self.image_key), list) and len(example.get(self.image_key)) > 0:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]
            try:
                if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                    images = [os.path.join(self.image_dir, image) for image in images]

            except Exception as e:
                print(f"images type: {type(images)} | value: {images}")
                print("full example:", example)



            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            
            print(images, model_inputs["input_ids"].size(-1))
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.video_key in example and isinstance(example.get(self.video_key), list) and len(example.get(self.video_key)) > 0:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            for video in videos:
                processed_videos.append(process_video(video))

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            # print(videos, model_inputs["input_ids"].size(-1))
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)
        example.pop(self.prompt_key, None)

        if self.image_key in example and isinstance(example.get(self.image_key), list) and len(example.get(self.image_key)) > 0:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images}
        elif self.video_key in example and isinstance(example.get(self.video_key), list) and len(example.get(self.video_key)) > 0:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video, return_fps=True
                )
                video_kwargs = {"do_sample_frames": False}
                processed_videos.append(processed_video)


                video_fps_list.append(video_fps)

            # print([prompt])
            if processed_video is not None:
                # print(processed_video)
                # print(video_kwargs)
                # print(processed_video[0].shape)
                processed_video, video_metadatas = processed_video
                processed_video, video_metadatas = [processed_video], [video_metadatas]
            else:
                video_metadatas = None
            model_inputs= self.processor(text=[prompt], videos=processed_video, add_special_tokens=False, video_metadata=video_metadatas, return_tensors="pt", do_resize=False, **video_kwargs)

            # print(videos, model_inputs["input_ids"].size(-1))


            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos}
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]


        if "images" in example: 
            example.pop("images")
        elif "videos" in example:
            example.pop("videos")

        # print(example)

        # print(example.keys())


        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index



            else:
                from ..models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example.pop(self.answer_key)

        # print(example)
        # print(input_ids.shape)


        # print(example)
        return example
