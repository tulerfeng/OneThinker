# -*- coding: utf-8 -*-
# Rewards for multimodal tasks with <think>...</think><answer>...</answer> outputs.
import re
import json
import math
import itertools
from typing import Any, Dict, List, Optional
import random

import torch
from rouge_score import rouge_scorer
from math_verify import parse as math_parse, verify as math_verify
from mathruler.grader import grade_answer

# ===================== Model-based reward configuration =====================
# Whether to use external Reward Model to compute accuracy for open-ended type
USE_MODEL_FOR_OPEN_ENDED: bool = False

# External RM model and service address (kept consistent with example)
RM_MODEL_PATH = "internlm/POLAR-7B"
RM_SERVER_ADDRESS = "xx.xx.xx.xx:xxxx"
# ==========================================================

# ===================== External RM evaluation dependencies =====================
from verl.workers.reward.model_reward import RewardModelClient
import numpy as np
# =========================================================


# -------------------------
# Patterns for format check
# -------------------------
THINK_ANSWER_PATTERN = re.compile(
    r"\A\s*<think>.*?</think>\s*<answer>.*?</answer>\s*\Z",
    re.DOTALL
)

ANSWER_CAPTURE_PATTERN = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    re.DOTALL
)


# -------------------------
# Utilities
# -------------------------
def extract_answer(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    m = ANSWER_CAPTURE_PATTERN.search(text)
    return m.group(1).strip() if m else None


def normalize_number(num_str: str) -> Optional[float]:
    try:
        return float((num_str or "").replace(",", ""))
    except Exception:
        return None


def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05) -> float:
    pred_t = torch.tensor(pred, dtype=torch.float32)
    tgt_t  = torch.tensor(target, dtype=torch.float32)
    rel_error = torch.abs(pred_t - tgt_t) / (torch.abs(tgt_t) + 1e-8)
    thresholds = torch.arange(start, end + interval/2, interval, dtype=torch.float32)
    return (rel_error < (1 - thresholds)).float().mean().item()


def wer(reference: str, hypothesis: str) -> float:
    ref_words, hyp_words = (reference or "").split(), (hypothesis or "").split()
    m, n = len(ref_words), len(hyp_words)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): d[i][0] = i
    for j in range(n + 1): d[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            d[i][j] = d[i - 1][j - 1] if ref_words[i - 1] == hyp_words[j - 1] else 1 + min(
                d[i - 1][j], d[i][j - 1], d[i - 1][j - 1]
            )
    return d[m][n] / max(1, m)


def compute_rouge_score(reference: str, hypothesis: str) -> float:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference or "", hypothesis or "")
    return (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3.0


# ---------- IoU helpers (strict format: must be numeric lists) ----------
def _is_list_of_numbers(x, n=None):
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
    # Strict: must be numeric lists with length 2; otherwise return 0
    if not _is_list_of_numbers(pred, 2) or not _is_list_of_numbers(gt, 2):
        return 0.0
    try:
        s1, e1 = float(pred[0]), float(pred[1])
        s2, e2 = float(gt[0]), float(gt[1])
    except Exception:
        return 0.0
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 1e-12 else 0.0


def iou_2d(box1: List[float], box2: List[float]) -> float:
    # Strict: must be numeric lists with length 4; otherwise return 0
    if not _is_list_of_numbers(box1, 4) or not _is_list_of_numbers(box2, 4):
        return 0.0
    try:
        x1, y1, x2, y2 = map(float, box1)
        X1, Y1, X2, Y2 = map(float, box2)
    except Exception:
        return 0.0
    inter_x1, inter_y1 = max(x1, X1), max(y1, Y1)
    inter_x2, inter_y2 = min(x2, X2), min(y2, Y2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area1 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area2 = max(0.0, X2 - X1) * max(0.0, Y2 - Y1)
    union = area1 + area2 - inter_area
    return inter_area / union if union > 1e-12 else 0.0


def mean_iou_over_gt_frames(pred_boxes: Dict[str, List[float]], gt_boxes: Dict[str, List[float]]) -> float:
    """
    For tracking: average over all GT frames (missing predicted frames are counted as 0).
    pred_boxes, gt_boxes: {frame_str: [x1,y1,x2,y2]}
    """
    if not isinstance(gt_boxes, dict) or not gt_boxes:
        return 0.0
    total, n = 0.0, 0
    for k, gbox in gt_boxes.items():
        total += iou_2d(pred_boxes.get(k, []), gbox)
        n += 1
    return total / n if n > 0 else 0.0


def mean_iou_over_intersection(pred_boxes: Dict[str, List[float]], gt_boxes: Dict[str, List[float]]) -> float:
    """
    For spatial-temporal tasks: IoU is averaged only over the intersection of frame keys
    (missing frames are not penalized).
    """
    if not isinstance(pred_boxes, dict) or not isinstance(gt_boxes, dict):
        return 0.0
    common = [k for k in pred_boxes.keys() if k in gt_boxes]
    if not common:
        return 0.0
    vals = [iou_2d(pred_boxes[k], gt_boxes[k]) for k in common]
    return sum(vals) / len(vals) if vals else 0.0


# -------------------------
# Segmentation point matching: strict 3↔3 optimal assignment
# -------------------------
def _pairwise_l2(p, q) -> float:
    try:
        dx = float(p[0]) - float(q[0])
        dy = float(p[1]) - float(q[1])
        return math.hypot(dx, dy)
    except Exception:
        return float("inf")


def assignment_similarity_3(pred_pts: List[List[float]],
                            gt_pts: List[List[float]],
                            sigma: float = 50.0) -> float:
    """
    3↔3 optimal matching (minimal total distance), returns Gaussian kernel similarity:
        sim = exp(- (avg_dist^2) / (2 * sigma^2)) ∈ [0,1]

    Explanation of parameters:
      - sigma: controls sensitivity range (smaller → more sensitive, sharper)
      - pred_pts, gt_pts: [[x,y], [x,y], [x,y]] exactly 3 points each
      - If the number or format of points is invalid, return 0.0
    """
    # ----------- Check input validity -----------
    if not isinstance(pred_pts, list) or not isinstance(gt_pts, list) or len(pred_pts) != 3 or len(gt_pts) != 3:
        return 0.0
    for p in pred_pts + gt_pts:
        if not _is_list_of_numbers(p, 2):
            return 0.0

    # ----------- Compute optimal matching distance -----------
    best_sum = float('inf')
    for perm in itertools.permutations(range(3)):
        s = 0.0
        good = True
        for i in range(3):
            d = _pairwise_l2(pred_pts[perm[i]], gt_pts[i])
            if math.isinf(d):
                good = False
                break
            s += d
        if good:
            best_sum = min(best_sum, s)

    if math.isinf(best_sum):
        return 0.0

    # ----------- Average distance & Gaussian similarity -----------
    avg_d = best_sum / 3.0
    sim = math.exp(- (avg_d ** 2) / (2 * sigma ** 2))
    return max(0.0, min(1.0, sim))


# -------------------------
# Format reward + structure reward
# -------------------------
def tag_format_reward(response: str) -> float:
    """
    Format requirement (format reward):
      Must strictly be: <think>...</think><answer>...</answer>
      Arbitrary newlines/whitespaces are allowed in the middle, but tag order and closures must be correct.
      Returns 1.0 if satisfied; otherwise 0.0.
    """
    return 1.0 if THINK_ANSWER_PATTERN.fullmatch(response or "") else 0.0


def answer_structure_bonus(answer: str, ground_truth: str, data_type: str, problem_type: str) -> float:
    """
    Structure requirements (structure reward):
      - spatial-temporal grounding:
          JSON structure must satisfy:
            {"time": [s, e], "boxes": {"frame_id": [x1, y1, x2, y2], ...}}
          +0.25 if the structure is valid;
          plus bbox key overlap ratio * 0.25 (overlap ratio = |pred.keys ∩ gt.keys| / |gt.keys|).
      - tracking:
          JSON structure must satisfy:
            {"boxes": {"frame_id": [x1, y1, x2, y2], ...}}
          +0.25 if the structure is valid;
          plus bbox key overlap ratio * 0.25.
      - temporal grounding:
          {"time": [s, e]} gets +0.5 if valid, otherwise 0.
      - spatial grounding:
          {"boxes": [x1, y1, x2, y2]} gets +0.5 if valid, otherwise 0.
      - segmentation:
          * image:
              {"boxes": [..], "positive_points": [[x,y],[x,y],[x,y]], "negative_points": [[x,y],[x,y],[x,y]]}
          * video:
              {"time": t, "boxes": [..], "positive_points": [[x,y],[x,y],[x,y]], "negative_points": [[x,y],[x,y],[x,y]]}
          +0.5 if the corresponding structure is satisfied, otherwise 0.
      - Other non-structured tasks: default +0.5.
    """
    ptype = (problem_type or "").lower()
    dtype = (data_type or "").lower()

    def _json(s):
        try:
            return json.loads(s)
        except Exception:
            return None

    if ptype in {"spatial-temporal grounding", "tracking"}:
        obj_pred = _json(answer)
        obj_gt   = _json(ground_truth)
        part_json = 0.0
        part_overlap = 0.0

        if ptype == "spatial-temporal grounding":
            json_ok = (
                isinstance(obj_pred, dict)
                and isinstance(obj_pred.get("time"), list) and len(obj_pred["time"]) == 2
                and isinstance(obj_pred.get("boxes"), dict)
                and all(_is_list_of_numbers(v, 4) for v in obj_pred["boxes"].values())
            )
        else:  # tracking
            json_ok = (
                isinstance(obj_pred, dict)
                and isinstance(obj_pred.get("boxes"), dict)
                and all(_is_list_of_numbers(v, 4) for v in obj_pred["boxes"].values())
            )
        if json_ok:
            part_json = 0.25

        if isinstance(obj_pred, dict) and isinstance(obj_gt, dict):
            pboxes = obj_pred.get("boxes", {})
            gboxes = obj_gt.get("boxes", {})
            if isinstance(pboxes, dict) and isinstance(gboxes, dict) and len(gboxes) > 0:
                inter = set(pboxes.keys()) & set(gboxes.keys())
                overlap_ratio = len(inter) / float(len(gboxes))
                part_overlap = 0.25 * max(0.0, min(1.0, float(overlap_ratio)))

        return part_json + part_overlap

    needs_check = {"temporal grounding", "spatial grounding", "segmentation"}
    if ptype in needs_check:
        obj = _json(answer)
        if ptype == "temporal grounding":
            ok = isinstance(obj, dict) and _is_list_of_numbers(obj.get("time"), 2)
            return 0.5 if ok else 0.0

        if ptype == "spatial grounding":
            ok = isinstance(obj, dict) and _is_list_of_numbers(obj.get("boxes"), 4)
            return 0.5 if ok else 0.0

        if ptype == "segmentation":
            if dtype == "image":
                ok = (
                    isinstance(obj, dict)
                    and _is_list_of_numbers(obj.get("boxes"), 4)
                    and isinstance(obj.get("positive_points"), list) and len(obj["positive_points"]) == 3
                    and isinstance(obj.get("negative_points"), list) and len(obj["negative_points"]) == 3
                    and all(_is_list_of_numbers(p, 2) for p in obj["positive_points"])
                    and all(_is_list_of_numbers(p, 2) for p in obj["negative_points"])
                )
                return 0.5 if ok else 0.0
            elif dtype == "video":
                ok = (
                    isinstance(obj, dict)
                    and isinstance(obj.get("time"), (int, float))  # time must be numeric
                    and _is_list_of_numbers(obj.get("boxes"), 4)
                    and isinstance(obj.get("positive_points"), list) and len(obj["positive_points"]) == 3
                    and isinstance(obj.get("negative_points"), list) and len(obj["negative_points"]) == 3
                    and all(_is_list_of_numbers(p, 2) for p in obj["positive_points"])
                    and all(_is_list_of_numbers(p, 2) for p in obj["negative_points"])
                )
                return 0.5 if ok else 0.0
            else:
                return 0.0

    # Non-structured tasks: default +0.5
    return 0.5


# -------------------------
# Math equivalence helper
# -------------------------
def _math_equivalent(gt: str, pred: str) -> bool:
    """
    Use math_verify to perform symbolic equivalence checking; if it fails (exceptions, etc.),
    fall back to grade_answer.
    """
    try:
        return bool(math_verify(math_parse(gt), math_parse(pred)))
    except Exception:
        return grade_answer(pred, gt)


# -------------------------
# Accuracy reward (normalized to [0,1])
# -------------------------
def accuracy_reward(response: str,
                    ground_truth: str,
                    data_type: str,
                    problem_type: str) -> float:
    """
    Normalized accuracy ∈ [0,1]. Strict format requirement: if the format is invalid, always return 0.
    Wrapped with try/except: any exception → 0.0.
    """
    try:
        ans = extract_answer(response) or response.strip()
        ptype = (problem_type or "").lower()
        dtype = (data_type or "").lower()
        gt = ground_truth or ""

        # ------ Pure QA type ------
        if ptype == "multiple choice":
            return 1.0 if grade_answer(ans.strip(), gt.strip()) else 0.0

        if ptype == "numerical":
            gt_num, pr_num = normalize_number(gt), normalize_number(ans)
            return 1.0 if (gt_num is not None and pr_num is not None and round(gt_num, 2) == round(pr_num, 2)) else 0.0

        if ptype == "regression":
            gt_num, pr_num = normalize_number(gt), normalize_number(ans)
            if gt_num is None or pr_num is None:
                return 0.0
            return mean_relative_accuracy(pr_num, gt_num)

        if ptype == "ocr":
            return max(0.0, min(1.0, 1.0 - wer(gt, ans)))

        if ptype == "open-ended":
            return max(0.0, min(1.0, compute_rouge_score(gt, ans)))

        if ptype == "math":
            return 1.0 if _math_equivalent(gt, ans) else 0.0

        # ------ JSON type (strict format)------
        def _load_json(s: str):
            try:
                return json.loads(s)
            except Exception:
                return None

        # temporal grounding: tIoU ∈ [0,1]
        if ptype == "temporal grounding":
            pred = _load_json(ans)
            gtj  = _load_json(gt)
            if not isinstance(pred, dict) or not isinstance(gtj, dict):
                return 0.0
            return iou_1d(pred.get("time"), gtj.get("time"))

        # spatial grounding: box IoU ∈ [0,1]
        if ptype == "spatial grounding":
            pred = _load_json(ans)
            gtj  = _load_json(gt)
            if not isinstance(pred, dict) or not isinstance(gtj, dict):
                return 0.0
            return iou_2d(pred.get("boxes"), gtj.get("boxes"))

        # spatial-temporal grounding: 0.5*tIoU + 0.5*mIoU(intersection)
        if ptype == "spatial-temporal grounding":
            pred = _load_json(ans)
            gtj  = _load_json(gt)
            if not isinstance(pred, dict) or not isinstance(gtj, dict):
                return 0.0
            tiou = iou_1d(pred.get("time"), gtj.get("time"))
            pboxes = pred.get("boxes")
            gboxes = gtj.get("boxes")
            if not isinstance(pboxes, dict) or not isinstance(gboxes, dict):
                miou_inter = 0.0
            else:
                miou_inter = mean_iou_over_intersection(pboxes, gboxes)
            return 0.5 * tiou + 0.5 * miou_inter

        # tracking: mean mIoU over GT frames (missing frames = 0)
        if ptype == "tracking":
            pred = _load_json(ans)
            gtj  = _load_json(gt)
            if not isinstance(pred, dict) or not isinstance(gtj, dict):
                return 0.0
            pboxes = pred.get("boxes")
            gboxes = gtj.get("boxes")
            if not isinstance(pboxes, dict) or not isinstance(gboxes, dict):
                return 0.0
            return mean_iou_over_gt_frames(pboxes, gboxes)

        # segmentation（image/video）
        if ptype == "segmentation":
            pred = _load_json(ans)
            gtj  = _load_json(gt)
            if not isinstance(pred, dict) or not isinstance(gtj, dict):
                return 0.0

            iou = iou_2d(pred.get("boxes"), gtj.get("boxes"))

            pos_pred = pred.get("positive_points")
            pos_gt   = gtj.get("positive_points")
            neg_pred = pred.get("negative_points")
            neg_gt   = gtj.get("negative_points")

            # Must be strict 3↔3
            if not (isinstance(pos_pred, list) and len(pos_pred) == 3 and
                    isinstance(pos_gt, list)   and len(pos_gt)   == 3 and
                    all(_is_list_of_numbers(p, 2) for p in pos_pred) and
                    all(_is_list_of_numbers(p, 2) for p in pos_gt)):
                return 0.0
            if not (isinstance(neg_pred, list) and len(neg_pred) == 3 and
                    isinstance(neg_gt, list)   and len(neg_gt)   == 3 and
                    all(_is_list_of_numbers(p, 2) for p in neg_pred) and
                    all(_is_list_of_numbers(p, 2) for p in neg_gt)):
                return 0.0

            pos_sim = assignment_similarity_3(pos_pred, pos_gt)
            neg_sim = assignment_similarity_3(neg_pred, neg_gt)

            if dtype == "image":
                return 0.5 * iou + 0.25 * pos_sim + 0.25 * neg_sim

            if dtype == "video":
                # time must be numeric (strict)
                t_pred = pred.get("time")
                t_gt   = gtj.get("time")
                if not isinstance(t_pred, (int, float)) or not isinstance(t_gt, (int, float)):
                    return 0.0
                time_sim = math.exp(-abs(float(t_pred) - float(t_gt)) / 2.0)  # τ=2s
                return 0.3 * iou + 0.3 * time_sim + 0.2 * pos_sim + 0.2 * neg_sim

            return 0.0

        # Unknown type
        return 0.0
    except Exception:
        # Outer fallback: any exception will be scored as 0
        return 0.0


# ===================== Wrapper: batch call external model for open-ended =====================
def evaluate_open_ended_with_rm(
    open_ended_queue: List[Dict[str, Any]],
    results: List[Dict[str, float]],
    format_weight: float,
    rm_server_type: str,
    rm_batch_size: int,
    normalize_model_reward_by_problem_id: bool
) -> None:
    """
    Take open-ended samples in open_ended_queue, and call external RM in batches to evaluate accuracy.
    Failed batches fall back to ROUGE. Optionally apply mean-std → min-max normalization within
    each problem_id group.
    After evaluation, this function will fill results[idx]['accuracy'] in-place and recompute
    results[idx]['overall'].
    """
    if not USE_MODEL_FOR_OPEN_ENDED or not open_ended_queue:
        return

    client = RewardModelClient(
        RM_MODEL_PATH,
        server_type=rm_server_type,
        server_address=RM_SERVER_ADDRESS
    )

    def _chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    model_scores: List[float] = [0.0] * len(open_ended_queue)

    for batch_id, batch in enumerate(_chunks(open_ended_queue, rm_batch_size)):
        data = [{"prompt": b["prompt"], "reference": b["reference"], "output": b["output"]} for b in batch]
        try:
            rewards = client(data)  # expected to return list[float]
            for j, sc in enumerate(rewards):
                model_scores[(batch_id * rm_batch_size) + j] = float(sc)
        except Exception:
            # Fallback: use ROUGE to compute scores for this batch
            for j, b in enumerate(batch):
                ref = b["reference"]
                out = b["output"]
                model_scores[(batch_id * rm_batch_size) + j] = float(max(0.0, min(1.0, compute_rouge_score(ref, out))))

    if normalize_model_reward_by_problem_id:
        groups: Dict[Any, List[int]] = {}
        for k, b in enumerate(open_ended_queue):
            gid = b.get("problem_id", None)
            groups.setdefault(gid, []).append(k)

        for gid, indices in groups.items():
            vals = np.array([model_scores[k] for k in indices], dtype=np.float32)
            mean, std = vals.mean(), vals.std()
            if std == 0:
                norm_vals = np.ones_like(vals)
            else:
                z = (vals - mean) / (std + 1e-6)
                norm_vals = (z - z.min()) / (z.max() - z.min() + 1e-12)
            for t, k in enumerate(indices):
                model_scores[k] = float(norm_vals[t])

    # Fill back accuracy, and recompute overall
    for k, b in enumerate(open_ended_queue):
        idx = b["idx"]
        results[idx]["accuracy"] = float(max(0.0, min(1.0, model_scores[k])))
        results[idx]["overall"] = (
            (1.0 - format_weight) * results[idx]["accuracy"]
            + format_weight * results[idx]["format"]
            + results[idx]["structure_reward"]
        )
# ==================================================================


# -------------------------
# Public API
# -------------------------
def compute_score(
    reward_inputs: List[Dict[str, Any]],
    format_weight: float = 0.1,
    # ===== Still kept as configurable parameters =====
    rm_server_type: str = "vllm",
    rm_batch_size: int = 64,
    normalize_model_reward_by_problem_id: bool = True,
) -> List[Dict[str, float]]:
    """
    Batch interface.
    Each item:
        {
            "response": str,
            "response_length": int,
            "ground_truth": str,   # may also contain <answer>...</answer>, here we extract it first
            "data_type": str,      # "image" | "video" | ...
            "problem_type": str    # see branches above
            # Optional additional fields:
            # "problem": str        # used as prompt for external RM in open-ended tasks
            # "problem_id": Any     # grouping key for normalization
        }
    Returns: list of dict with keys {overall, format, accuracy, structure_reward}
    overall = (1 - format_weight) * accuracy + format_weight * format + structure_reward
      - format: 1.0 if <think>...</think><answer>...</answer>, otherwise 0.0
      - structure_reward:
          * spatial-temporal / tracking: 0.25 (JSON valid) + 0.25 (key overlap ratio)
          * other structured tasks: +0.5 if valid
          * non-structured tasks: default +0.5
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for this reward function.")

    results: List[Dict[str, float]] = []
    # ===================== Collect open-ended samples to be evaluated =====================
    open_ended_queue = []  # Each item: {idx, prompt, reference, output, problem_id}
    # ================================================================

    for idx, item in enumerate(reward_inputs):
        try:
            # Normalize tag whitespaces, e.g. < / think > → </think>
            raw_response = item.get("response", "") or ""
            response = re.sub(r"\s*(<|>|/)\s*", r"\1", raw_response)

            # print(response)

            data_type = item.get("data_type", "") or ""
            problem_type = item.get("problem_type", "") or ""

            # ground_truth may also be wrapped in <answer>...</answer>; extract it first here
            raw_gt = item.get("ground_truth", "") or ""
            gt_extracted = extract_answer(raw_gt) or raw_gt

            # 1) format reward —— requires strict tag structure: <think>...</think><answer>...</answer>
            f_score = tag_format_reward(response)

            # 2) structure reward —— according to JSON structure requirements by task type (see function doc)
            ans = extract_answer(response) or ""
            s_reward = answer_structure_bonus(ans, gt_extracted, data_type, problem_type)

            # 3) accuracy (all normalized to [0,1])
            if USE_MODEL_FOR_OPEN_ENDED and (problem_type or "").lower() == "open-ended":
                # First set to 0, and finally compute with external model and fill back
                a_score = 0.0
                open_ended_queue.append({
                    "idx": idx,
                    "prompt": item.get("problem", "") or "",
                    "reference": gt_extracted or "",
                    "output": ans or "",
                    "problem_id": item.get("problem_id", None),
                })
            else:
                a_score = accuracy_reward(response, gt_extracted, data_type, problem_type)

            if f_score == 0:
                s_reward = 0

            overall = (1.0 - format_weight) * a_score + format_weight * f_score + s_reward

            results.append({
                "overall": float(overall),
                "format": float(f_score),
                "accuracy": float(a_score),
                "structure_reward": float(s_reward),
            })
        except Exception:
            # Fallback for the entire sample: any exception, all four fields are set to 0
            results.append({
                "overall": 0.0,
                "format": 0.0,
                "accuracy": 0.0,
                "structure_reward": 0.0,
            })



    # ===================== Call wrapper for batch external evaluation and fill back =====================
    evaluate_open_ended_with_rm(
        open_ended_queue=open_ended_queue,
        results=results,
        format_weight=format_weight,
        rm_server_type=rm_server_type,
        rm_batch_size=rm_batch_size,
        normalize_model_reward_by_problem_id=normalize_model_reward_by_problem_id
    )
    # ======================================================================

    if random.random() < 0.01:

        for idx, item in enumerate(reward_inputs):

            print('type', item.get("problem_type", ""))
            print('gt', extract_answer(item.get("ground_truth", "")))
            print('ans', extract_answer(item.get("response", "")))
            print({
                "overall": results[idx]["overall"],
                "format": results[idx]["format"],
                "accuracy": results[idx]["accuracy"],
                "structure_reward": results[idx]["structure_reward"],
            })

    return results
