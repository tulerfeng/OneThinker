#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import torch
from tqdm import tqdm
from rouge_score import rouge_scorer

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

# ====== Optional: math equivalence dependencies (automatically degrade if unavailable)======
try:
    from math_verify import parse as math_parse, verify as math_verify
except Exception:
    math_parse = math_verify = None

try:
    from mathruler.grader import grade_answer as math_grade_answer
except Exception:
    math_grade_answer = None
# ====================================================

# =========================
# Default parameters (overridable by command line)
# =========================
DEFAULT_BASE_PREFIX = ""
DEFAULT_BSZ = 64
DEFAULT_SEED = 0
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_K = -1

# Video/Image preprocessing parameters
DEFAULT_MAX_PIXELS_VIDEO = 256 * 32 * 32
DEFAULT_MAX_FRAMES = 128
DEFAULT_FPS = 2
DEFAULT_MAX_PIXELS_IMAGE = 1024 * 32 * 32

# =========================
# PROMPT (keep exactly as provided)
# =========================
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

# =========================
# Utility functions (parsing and metrics)
# =========================
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)

def extract_answer(text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = ANSWER_RE.search(text)
    return m.group(1).strip() if m else text.strip()

def normalize_number(num_str: str) -> Optional[float]:
    try:
        return float((num_str or "").replace(",", ""))
    except Exception:
        return None

def _is_list_of_numbers(x, n=None) -> bool:
    if not isinstance(x, list):
        return False
    if n is not None and len(x) != n:
        return False
    try:
        for v in x:
            float(v)
        return True
    except Exception:
        return False

def iou_1d(pred: List[float], gt: List[float]) -> float:
    if not _is_list_of_numbers(pred, 2) or not _is_list_of_numbers(gt, 2):
        return 0.0
    s1, e1 = float(pred[0]), float(pred[1])
    s2, e2 = float(gt[0]), float(gt[1])
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 1e-12 else 0.0

def iou_2d(box1: List[float], box2: List[float]) -> float:
    if not _is_list_of_numbers(box1, 4) or not _is_list_of_numbers(box2, 4):
        return 0.0
    x1, y1, x2, y2 = map(float, box1)
    X1, Y1, X2, Y2 = map(float, box2)
    inter_x1, inter_y1 = max(x1, X1), max(y1, Y1)
    inter_x2, inter_y2 = min(x2, X2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area1 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area2 = max(0.0, X2 - X1) * max(0.0, Y2 - Y1)
    union = area1 + area2 - inter_area
    return inter_area / union if union > 1e-12 else 0.0

def mean_iou_over_gt_frames(pred_boxes: Dict[str, List[float]], gt_boxes: Dict[str, List[float]]) -> float:
    if not isinstance(gt_boxes, dict) or not gt_boxes:
        return 0.0
    total, n = 0.0, 0
    for k, gbox in gt_boxes.items():
        total += iou_2d(pred_boxes.get(k, []), gbox)
        n += 1
    return total / n if n > 0 else 0.0

def mean_iou_over_intersection(pred_boxes: Dict[str, List[float]], gt_boxes: Dict[str, List[float]]) -> float:
    if not isinstance(pred_boxes, dict) or not isinstance(gt_boxes, dict):
        return 0.0
    common = [k for k in pred_boxes.keys() if k in gt_boxes]
    if not common:
        return 0.0
    vals = [iou_2d(pred_boxes[k], gt_boxes[k]) for k in common]
    return sum(vals) / len(vals) if vals else 0.0

def wer(reference: str, hypothesis: str) -> float:
    ref_words, hyp_words = (reference or "").split(), (hypothesis or "").split()
    m, n = len(ref_words), len(hyp_words)
    d = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): d[i][0] = i
    for j in range(n+1): d[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return d[m][n] / max(1, m)

def compute_rouge_score(reference: str, hypothesis: str) -> float:
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    scores = scorer.score(reference or "", hypothesis or "")
    return (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3.0

# =============== Math equivalence (math) ===============
def _math_equivalent(gt: str, pred: str) -> bool:
    # Prefer math_verify; if it fails, fall back to mathruler.grader; if that also fails, use exact string match
    try:
        if math_parse and math_verify:
            return bool(math_verify(math_parse(gt), math_parse(pred)))
    except Exception:
        pass
    try:
        if math_grade_answer:
            return bool(math_grade_answer(pred, gt))
    except Exception:
        pass
    return gt.strip() == pred.strip()

# =============== Strict MRA (Regression) ===============
def mean_relative_accuracy_strict(pred: float, target: float) -> float:
    """
    MRA = (1/10) * sum_{θ in {0.5,0.55,...,0.95}}  1[ |y_hat - y| / |y| < 1 - θ ]
    If target≈0 then return 0 (avoid division by zero; this metric assumes y!=0).
    """
    try:
        p = float(pred); t = float(target)
        if abs(t) < 1e-12:
            return 0.0
    except Exception:
        return 0.0

    rel = abs(p - t) / abs(t)
    count = 0
    for k in range(10):  # 0.5, 0.55, ..., 0.95
        theta = 0.5 + 0.05 * k
        if rel < (1.0 - theta):
            count += 1
    return count / 10.0

# =============== accuracy (return value + component details) ===============
def accuracy_only(
    response: str,
    ground_truth: str,
    data_type: str,
    problem_type: str
) -> Tuple[float, Dict[str, float]]:
    """
    Returns:
      - accuracy ∈ [0,1]
      - components: raw score of each sub-metric (for composite tasks only sub-metrics are recorded, not the weighted sum)
        e.g.:
          temporal grounding: {"tiou": x}
          spatial grounding : {"iou": x}
          spatial-temporal  : {"tiou": a, "miou_inter": b}
          tracking          : {"miou_gt": x}
          seg_image         : {"iou": i, "pos_sim": p, "neg_sim": n}
          seg_video         : {"iou": i, "time_sim": t, "pos_sim": p, "neg_sim": n}
    """
    ans = extract_answer(response) or response.strip()
    gt  = extract_answer(ground_truth) or ground_truth or ""
    ptype = (problem_type or "").strip()
    ptype_l = ptype.lower()
    dtype = (data_type or "").lower()

    # multiple choice
    if ptype_l == "multiple choice":
        a = (ans.strip()[:1] if ans else "").upper()
        g = (gt.strip()[:1]  if gt  else "").upper()
        return (1.0 if a == g and a in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" else 0.0, {})

    # numerical (strict to two decimals)
    if ptype_l == "numerical":
        a_num, g_num = normalize_number(ans), normalize_number(gt)
        ok = (a_num is not None and g_num is not None and round(a_num,2) == round(g_num,2))
        return (1.0 if ok else 0.0, {})

    # regression (strict MRA)
    if ptype_l == "regression":
        a_num, g_num = normalize_number(ans), normalize_number(gt)
        if a_num is None or g_num is None:
            return (0.0, {})
        mra = mean_relative_accuracy_strict(a_num, g_num)
        return (float(mra), {})   # single metric, not split into components

    # ocr
    if ptype_l == "ocr" or ptype == "OCR":
        return (max(0.0, min(1.0, 1.0 - wer(gt, ans))), {})

    # open-ended (ROUGE)
    if ptype_l == "open-ended":
        return (max(0.0, min(1.0, compute_rouge_score(gt, ans))), {})

    # math (symbolic equivalence)
    if ptype_l == "math":
        return (1.0 if _math_equivalent(gt, ans) else 0.0, {})

    # JSON types
    def _json(s: str):
        try:
            return json.loads(s)
        except Exception:
            return None

    # temporal grounding: tIoU
    if ptype_l == "temporal grounding":
        pred = _json(ans); gtj = _json(gt)
        tiou = iou_1d(pred.get("time") if isinstance(pred, dict) else None,
                      gtj.get("time")  if isinstance(gtj, dict)  else None)
        # accuracy = tiou; components only record tiou
        return (float(tiou), {"tiou": float(tiou)})

    # spatial grounding: IoU
    if ptype_l == "spatial grounding":
        pred = _json(ans); gtj = _json(gt)
        iou = iou_2d(pred.get("boxes") if isinstance(pred, dict) else None,
                     gtj.get("boxes")  if isinstance(gtj, dict)  else None)
        return (float(iou), {"iou": float(iou)})

    # spatial-temporal grounding: record components separately, do not put the combined value into components
    if ptype_l == "spatial-temporal grounding":
        pred = _json(ans); gtj = _json(gt)
        if not isinstance(pred, dict) or not isinstance(gtj, dict):
            return (0.0, {"tiou": 0.0, "miou_inter": 0.0})
        tiou = iou_1d(pred.get("time"), gtj.get("time"))
        pboxes, gboxes = pred.get("boxes"), gtj.get("boxes")
        miou = mean_iou_over_intersection(pboxes if isinstance(pboxes, dict) else {},
                                          gboxes if isinstance(gboxes, dict) else {})
        # accuracy is still 0.5*tiou + 0.5*miou for overall; components only record tiou and miou
        acc = 0.5 * tiou + 0.5 * miou
        return (float(acc), {"tiou": float(tiou), "miou_inter": float(miou)})

    # tracking: mean mIoU over GT frames (missing=0)
    if ptype_l == "tracking":
        pred = _json(ans); gtj = _json(gt)
        if not isinstance(pred, dict) or not isinstance(gtj, dict):
            return (0.0, {"miou_gt": 0.0})
        pboxes, gboxes = pred.get("boxes"), gtj.get("boxes")
        miou_gt = mean_iou_over_gt_frames(pboxes if isinstance(pboxes, dict) else {},
                                          gboxes if isinstance(gboxes, dict) else {})
        return (float(miou_gt), {"miou_gt": float(miou_gt)})

    # segmentation —— do not compute metrics, directly return 0.0 and empty components (beyond the added logic, keep unchanged)
    if ptype_l == "segmentation":
        return (0.0, {})

    # unknown type
    return (0.0, {})

# =============== Component-level R@t statistics ===============
RECALL_THRESHOLDS = [0.3, 0.5, 0.7]

def init_aggregator() -> Dict[str, Any]:
    return {
        "overall_acc_sum": 0.0,
        "overall_count": 0,
        "per_type_acc_sum": defaultdict(float),
        "per_type_count": defaultdict(int),

        # Component-level (by problem_type, then by component name)
        "comp_sum": defaultdict(lambda: defaultdict(float)),     # comp_sum[ptype][comp]
        "comp_count": defaultdict(lambda: defaultdict(int)),     # comp_count[ptype][comp]
        "comp_recall_hits": defaultdict(lambda: defaultdict(lambda: defaultdict(int))),  # hits[ptype][comp][t]
    }

def accumulate_metrics(
    agg: Dict[str, Any],
    ptype: str,
    acc: float,
    comp: Dict[str, float],
):
    pkey = (ptype or "").strip()

    # overall + per_type accuracy
    agg["overall_acc_sum"] += float(acc)
    agg["overall_count"]   += 1
    agg["per_type_acc_sum"][pkey] += float(acc)
    agg["per_type_count"][pkey]   += 1

    # components
    for cname, cval in comp.items():
        val = float(cval)
        agg["comp_sum"][pkey][cname]   += val
        agg["comp_count"][pkey][cname] += 1
        for t in RECALL_THRESHOLDS:
            if val >= t:
                agg["comp_recall_hits"][pkey][cname][t] += 1

def finalize_metrics(agg: Dict[str, Any]) -> Dict[str, Any]:
    out = {}

    # overall
    out["overall/acc"] = agg["overall_acc_sum"] / max(1, agg["overall_count"])

    # per_type accuracy
    for pkey, cnt in agg["per_type_count"].items():
        out[f"{pkey}/acc"] = agg["per_type_acc_sum"][pkey] / max(1, cnt)

    # per_type component means + recalls
    for pkey, comp_sums in agg["comp_sum"].items():
        for cname, s in comp_sums.items():
            c = agg["comp_count"][pkey][cname]
            if c > 0:
                out[f"{pkey}/{cname}/mean"] = s / c
                for t in RECALL_THRESHOLDS:
                    hits = agg["comp_recall_hits"][pkey][cname][t]
                    out[f"{pkey}/{cname}/R@{t}"] = hits / c
    return out

# =========================
# vLLM input packing
# =========================
def build_user_content_item(
    data_type: str,
    full_path: str,
    add_image_path: Optional[str],
    max_pixels_video: int,
    max_frames: int,
    fps: int,
    max_pixels_image: int
) -> List[Dict[str, Any]]:
    content = []
    if data_type == "video":
        content.append({
            "type": "video",
            "video": full_path,
            "max_pixels": max_pixels_video,
            "max_frames": max_frames,
            "fps": fps
        })
    elif data_type == "image":
        content.append({
            "type": "image",
            "image": full_path,
            "max_pixels": max_pixels_image
        })
    elif data_type == "video-image":
        content.append({
            "type": "video",
            "video": full_path,
            "max_pixels": max_pixels_video,
            "max_frames": max_frames,
            "fps": fps
        })
        if not add_image_path:
            raise ValueError("data_type=video-image requires additional_path (image path)")
        content.append({
            "type": "image",
            "image": add_image_path,
            "max_pixels": max_pixels_image
        })
    return content

def prepare_inputs_for_vllm_single(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs
    return {
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs
    }

# =========================
# Main pipeline
# =========================
def main():
    parser = argparse.ArgumentParser(description="Multimodal Evaluation (accuracy + component-wise recalls)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_json", type=str, required=True)

    # Output name: out_dir / (basename(input_json) + suffix + .json)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--suffix", type=str, default="_eval")

    parser.add_argument("--base_prefix", type=str, default=DEFAULT_BASE_PREFIX)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BSZ)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)

    parser.add_argument("--max_pixels_video", type=int, default=DEFAULT_MAX_PIXELS_VIDEO)
    parser.add_argument("--max_frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--max_pixels_image", type=int, default=DEFAULT_MAX_PIXELS_IMAGE)
    args = parser.parse_args()

    in_base = Path(args.input_json).stem
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_json_path = out_dir / f"{in_base}{args.suffix}.json"

    # Read data
    if args.input_json.endswith(".jsonl"):
        data = []
        with open(args.input_json, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
    else:
        with open(args.input_json, "r", encoding="utf-8") as f:
            data = json.load(f)

    # Resume from checkpoint
    final_output: List[Dict[str, Any]] = []
    agg = init_aggregator()
    start_idx = 0
    if output_json_path.exists():
        try:
            with open(output_json_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            final_output = existing.get("results", [])
            # Replay once to restore aggregation (ensure consistency)
            agg = init_aggregator()
            for sample in final_output:
                acc = float(sample.get("accuracy", 0.0))
                ptype = sample.get("problem_type", "")
                comps = sample.get("components", {}) if isinstance(sample.get("components"), dict) else {}
                accumulate_metrics(agg, ptype, acc, comps)
            start_idx = len(final_output)
            print(f"[Resume] Found {start_idx} processed samples, resume from {start_idx}.")
        except Exception as e:
            print(f"[Warn] Failed to read existing output: {e}")

    # Initialize model
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    torch.manual_seed(args.seed)

    processor = AutoProcessor.from_pretrained(args.model_path)
    llm = LLM(
        model=args.model_path,
        max_model_len = 81920,
        gpu_memory_utilization=0.8,
        mm_encoder_tp_mode="data",
        tensor_parallel_size=torch.cuda.device_count(),
        seed=args.seed
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_k=args.top_k,
        stop_token_ids=[],
    )

    # Build prompt
    def build_prompt_text(example: Dict[str, Any]) -> str:
        prompt_str = example.get("problem") or ""
        data_type = (example.get("data_type") or "").strip().lower()
        pt = example.get("problem_type") or ""

        # multiple choice: append options
        question = prompt_str
        if (pt == "multiple choice") and isinstance(example.get("options"), list) and example["options"]:
            opts = "\n".join(example["options"])
            question = f"{question}\nOptions:\n{opts}"

        pt_lower = pt.strip().lower()
        if pt_lower == "segmentation":
            type_key = "segmentation_video" if data_type == "video" else "segmentation_image"
        else:
            type_key = pt if pt in TYPE_TEMPLATE else pt_lower

        tail = TYPE_TEMPLATE.get(type_key, "")
        return QUESTION_TEMPLATE.format(Question=question) + tail

    # Main loop
    BSZ = args.batch_size
    for i in tqdm(range(start_idx, len(data), BSZ), desc="Batches"):
        batch = data[i:i+BSZ]

        inputs_for_vllm = []
        for example in batch:
            # Prefix path with base directory
            raw_path = example.get("path") or ""
            full_path = os.path.join(args.base_prefix, raw_path.lstrip("./").lstrip("/"))
            add_path = None
            if (example.get("data_type") or "").strip().lower() == "video-image":
                add_raw = example.get("additional_path") or ""
                add_path = os.path.join(args.base_prefix, add_raw.lstrip("./").lstrip("/"))

            # content
            content = build_user_content_item(
                data_type=(example.get("data_type") or "").strip().lower(),
                full_path=full_path,
                add_image_path=add_path,
                max_pixels_video=args.max_pixels_video,
                max_frames=args.max_frames,
                fps=args.fps,
                max_pixels_image=args.max_pixels_image
            )
            content.append({"type": "text", "text": build_prompt_text(example)})

            messages = [{"role": "user", "content": content}]
            inputs_for_vllm.append(prepare_inputs_for_vllm_single([messages[0]], processor))

        # Generation
        try:
            outputs = llm.generate(inputs_for_vllm, sampling_params=sampling_params)
            texts = [o.outputs[0].text for o in outputs]
        except Exception as e:
            print(f"[Error] vLLM generate failed at batch start_idx={i}: {e}")
            texts = ["<answer>ERROR</answer>"] * len(inputs_for_vllm)

        # Evaluation + accumulation + write results
        for example, out_text in zip(batch, texts):
            pred_ans = extract_answer(out_text)
            gt_ans   = example.get("solution", "")

            acc, components = accuracy_only(
                response=out_text,
                ground_truth=gt_ans,
                data_type=(example.get("data_type") or "").strip().lower(),
                problem_type=example.get("problem_type","")
            )

            sample_out = dict(example)
            sample_out["output"] = out_text
            sample_out["prediction"] = pred_ans
            # —— New: for Segmentation, only extract <answer> content into predicted_answer_norm (no metric calculation)
            if (example.get("problem_type","").strip().lower() == "segmentation"):
                sample_out["predicted_answer_norm"] = pred_ans
            # —— Others remain unchanged
            sample_out["accuracy"] = float(acc)
            if components:
                sample_out["components"] = {k: float(v) for k, v in components.items()}

            final_output.append(sample_out)
            accumulate_metrics(
                agg,
                example.get("problem_type",""),
                float(acc),
                components
            )

        # Write to disk per batch (including accumulated metrics)
        metrics_dict = finalize_metrics(agg)
        try:
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "results": final_output,
                        "metrics": metrics_dict,
                        "meta": {
                            "batch_size": BSZ,
                            "model_path": args.model_path,
                            "input_json": args.input_json
                        }
                    },
                    f, indent=2, ensure_ascii=False
                )
        except Exception as e:
            print(f"[Warn] Failed to write output json at batch end (i={i}): {e}")

    # Final write to disk
    metrics_dict = finalize_metrics(agg)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": final_output,
                "metrics": metrics_dict,
                "meta": {
                    "batch_size": BSZ,
                    "model_path": args.model_path,
                    "input_json": args.input_json
                }
            },
            f, indent=2, ensure_ascii=False
        )
    print(f"[Done] Saved {len(final_output)} samples to {output_json_path}")
    print("[Metrics]")
    for k, v in metrics_dict.items():
        try:
            print(f"  {k}: {v:.4f}")
        except Exception:
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
