import os
import sys
import json
import math
import shutil
import tempfile
import subprocess
import traceback
import warnings
from contextlib import contextmanager
from typing import Dict, Any, Tuple, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import multiprocessing as mp
from threading import Thread
import queue as thread_queue  # used in single-process mode
import gc

# === NEW: CLI & random visualization ratio ===
import argparse
import random

# =========================
# Fixed "parameters" (modify as needed)
# =========================
PRED_JSON = ""
DATA_ROOT = "<Evaluation-data-root>"
SAM2_CKPT = "/sam2/sam2.1_hiera_large.pt"
SAM2_CFG  = "//sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
OUT_DIR   = "./sam2_viz-full"
OUT_JSON  = ""

# Multi-GPU parallel config
NUM_GPUS = 1
WORKERS_PER_GPU = 1

# Pre-extraction and visualization control
PRE_EXTRACT_THREADS = 1
VISUALIZE = False
FRAME_IMG_EXT = ".jpg"
VIZ_RATIO = 0.0

# Epoch size
EPOCH_SIZE = 320

# =========================
# Suppress SAM2 noisy warnings & tqdm internal outputs
# =========================
warnings.filterwarnings(
    "ignore",
    message="cannot import name '_C' from 'sam2'",
    category=UserWarning,
    module=r".*sam2.*",
)

@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# -------- GPU memory cleanup helper --------
def free_cuda():
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

# =========================
# Dependency check & RLE decoding
# =========================
try:
    from pycocotools import mask as cocomask
except Exception as e:
    raise RuntimeError(
        "缺少依赖 pycocotools（用于 RLE 解码）。请先安装：\n"
        "pip install pycocotools\n"
        f"原始错误：{repr(e)}"
    )

def _check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except FileNotFoundError:
        raise RuntimeError("未检测到 ffmpeg。请先安装并确保其在 PATH 中。")
_check_ffmpeg()

# =========================
# SAM2
# =========================
from sam2.build_sam import build_sam2_video_predictor

# ============ Utility functions / visualization / metrics ============

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def _squeeze_mask(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m)
    if m.ndim == 3:
        if m.shape[0] == 1:
            m = m[0]
        elif m.shape[-1] == 1:
            m = m[..., 0]
    return (m > 0).astype(np.uint8)

def overlay_mask(img: np.ndarray, mask: np.ndarray, rgba: Tuple[float,float,float,float]) -> np.ndarray:
    h, w = img.shape[:2]
    mask = _squeeze_mask(mask)
    if mask.shape != (h, w):
        mask = np.array(Image.fromarray(mask).resize((w, h), Image.NEAREST))
    out = img.copy().astype(np.float32) / 255.0
    color = np.array(rgba[:3], dtype=np.float32)
    alpha = float(rgba[3])
    m = mask.astype(np.float32)[..., None]
    out = out * (1 - m * alpha) + color * (m * alpha)
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return out

def show_points_on_ax(coords: Optional[np.ndarray], labels: Optional[np.ndarray], ax,
                      pos_size=50, neg_size=50, edge_linewidth=0.33):
    if coords is None or labels is None or len(coords) == 0:
        return
    pos = coords[labels == 1]
    neg = coords[labels == 0]
    if len(pos) > 0:
        ax.scatter(pos[:, 0], pos[:, 1], color='green', marker='*', s=pos_size,
                   edgecolor='white', linewidth=edge_linewidth)
    if len(neg) > 0:
        ax.scatter(neg[:, 0], neg[:, 1], color='red', marker='*', s=neg_size,
                   edgecolor='white', linewidth=edge_linewidth)

def show_box_on_ax(box: Optional[np.ndarray], ax):
    if box is None: return
    if not (isinstance(box, np.ndarray) and box.size == 4): return
    x0, y0, x1, y1 = box.tolist()
    ax.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, edgecolor='yellow', facecolor=(0,0,0,0), lw=2))

def ffmpeg_extract_frame(video_path: str, time_s: float, save_path: str, target_wh: Tuple[int, int]):
    W, H = target_wh
    ensure_dir(os.path.dirname(save_path))
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-y",
        "-ss", f"{max(0.0, float(time_s))}",
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        "-vf", f"scale={W}:{H}",
        save_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not os.path.exists(save_path):
        raise RuntimeError(f"ffmpeg 抽帧失败: {video_path} @ {time_s}s -> {save_path}\n{proc.stderr.decode('utf-8', errors='ignore')}")

def ffmpeg_images_to_mp4(img_pattern: str, out_mp4: str, fps: float):
    vf = "pad=ceil(iw/2)*2:ceil(ih/2)*2"
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-y",
        "-framerate", f"{max(1e-6, float(fps))}",
        "-i", img_pattern,
        "-vf", vf,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        out_mp4,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not os.path.exists(out_mp4):
        raise RuntimeError(f"ffmpeg 视频编码失败: {out_mp4}\n{proc.stderr.decode('utf-8', errors='ignore')}")

def video_to_frame_dir(video_abs_path: str) -> str:
    parent = os.path.dirname(video_abs_path)
    stem = os.path.splitext(os.path.basename(video_abs_path))[0]
    return os.path.join(parent, stem)

def _extract_single_frame_safe(video_abs_path: str, t: float, out_img: str, target_wh: Tuple[int, int]) -> bool:
    try:
        ffmpeg_extract_frame(video_abs_path, t, out_img, target_wh)
        return True
    except Exception as e:
        sys.stderr.write(f"[FFMPEG] fail @ {video_abs_path} t={t:.4f}s -> {out_img} err={repr(e)}\n")
        return False

def ensure_video_frames_extracted(
    video_abs_path: str,
    frame_dir: str,
    fps: float,
    num_frames: int,
    target_wh: Tuple[int, int],
    threads: int = PRE_EXTRACT_THREADS,
    desc: str = "extract",
) -> None:
    ensure_dir(frame_dir)
    missing = []
    for order in range(num_frames):
        out_img = os.path.join(frame_dir, f"{order:06d}{FRAME_IMG_EXT}")
        if not os.path.exists(out_img):
            missing.append(order)

    if not missing:
        return

    with ThreadPoolExecutor(max_workers=max(1, int(threads))) as ex:
        futures = []
        for order in missing:
            t = order / float(fps)
            out_img = os.path.join(frame_dir, f"{order:06d}{FRAME_IMG_EXT}")
            futures.append(ex.submit(_extract_single_frame_safe, video_abs_path, t, out_img, target_wh))
        for _ in tqdm(as_completed(futures), total=len(futures), desc=desc, dynamic_ncols=True):
            pass

    still_missing = [
        o for o in range(num_frames)
        if not os.path.exists(os.path.join(frame_dir, f"{o:06d}{FRAME_IMG_EXT}"))
    ]
    if still_missing:
        raise RuntimeError(
            f"仍有帧缺失（{len(still_missing)}）: {video_abs_path} -> {frame_dir} 例如 {still_missing[:5]}"
        )

def decode_coco_rle_to_mask(rle_obj: Dict[str, Any]) -> np.ndarray:
    counts = rle_obj["counts"]
    size = rle_obj["size"]
    if isinstance(counts, list):
        rle = {"counts": counts, "size": size}
    else:
        rle = {"counts": counts.encode("utf-8"), "size": size}
    m = cocomask.decode(rle)
    return _squeeze_mask(m)

def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter) / float(union) if union > 0 else 0.0

# --- NEW: return intersection/union for accumulating cIoU ---
def compute_inter_union(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[int, int]:
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    return inter, union

def _neighbors_sum(binary: np.ndarray) -> np.ndarray:
    h, w = binary.shape
    s = np.zeros_like(binary, dtype=np.uint16)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            y0 = max(0, dy); y1 = h + min(0, dy)
            x0 = max(0, dx); x1 = w + min(0, dx)
            s[y0:y1, x0:x1] += binary[y0-dy:y1-dy, x0-dx:x1-dx]
    return s

def _extract_boundary(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    neigh = _neighbors_sum(m)
    boundary = (m == 1) & (neigh < 8)
    return boundary.astype(np.uint8)

def _dilate(binary: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return (binary > 0).astype(np.uint8)
    h, w = binary.shape
    out = np.zeros_like(binary, dtype=np.uint8)
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            y0 = max(0, dy); y1 = h + min(0, dy)
            x0 = max(0, dx); x1 = w + min(0, dx)
            out[y0:y1, x0:x1] |= binary[y0-dy:y1-dy, x0-dx:x1-dx]
    return out

def boundary_fscore(pred_mask: np.ndarray, gt_mask: np.ndarray, tau_ratio: float = 0.0075) -> float:
    pm = (pred_mask > 0).astype(np.uint8)
    gm = (gt_mask   > 0).astype(np.uint8)
    if pm.sum() == 0 and gm.sum() == 0:
        return 1.0
    if pm.sum() == 0 or gm.sum() == 0:
        return 0.0
    h, w = pm.shape
    r = max(1, int(round(tau_ratio * math.hypot(h, w))))
    pb = _extract_boundary(pm)
    gb = _extract_boundary(gm)
    if pb.sum() == 0 and gb.sum() == 0:
        return 1.0
    if pb.sum() == 0 or gb.sum() == 0:
        return 0.0
    pb_d = _dilate(pb, r)
    gb_d = _dilate(gb, r)
    tp_p = (pb & gb_d).sum()
    tp_g = (gb & pb_d).sum()
    prec = float(tp_p) / float(pb.sum()) if pb.sum() > 0 else 0.0
    rec  = float(tp_g) / float(gb.sum()) if gb.sum() > 0 else 0.0
    if (prec + rec) == 0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)

# =========================
# Coordinate de-normalization (0~1000 -> pixels)
# =========================
def _clip_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(float(v)))))

def _denorm_xy(x: float, y: float, W: int, H: int) -> Tuple[int, int]:
    return _clip_int(x * W / 1000.0, 0, W), _clip_int(y * H / 1000.0, 0, H)

def _denorm_box(box: List[float], W: int, H: int) -> Optional[np.ndarray]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    x1, y1 = _denorm_xy(box[0], box[1], W, H)
    x2, y2 = _denorm_xy(box[2], box[3], W, H)
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def _denorm_points(pts: List[List[float]], W: int, H: int) -> np.ndarray:
    if not isinstance(pts, list):
        return np.zeros((0, 2), dtype=np.float32)
    out = []
    for p in pts:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            px, py = _denorm_xy(p[0], p[1], W, H)
            out.append([px, py])
    if not out:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array(out, dtype=np.float32)

def parse_pred_from_norm(sample: Dict[str, Any], W: int, H: int):
    """
    Return: (box_np, pos_np|None, neg_np|None, time_sec, pts, labels)
    """
    raw = sample.get("predicted_answer_norm", "")
    try:
        obj = json.loads(raw) if isinstance(raw, str) else (raw if isinstance(raw, dict) else {})
    except Exception:
        obj = {}

    time_sec = 0.0
    try:
        if "time" in obj and obj["time"] is not None:
            time_sec = float(obj["time"])
    except Exception:
        time_sec = 0.0

    box_np = None
    if isinstance(obj.get("boxes"), (list, tuple)) and len(obj["boxes"]) == 4:
        box_np = _denorm_box(obj["boxes"], W, H)

    pos_np = _denorm_points(obj.get("positive_points", []), W, H)
    neg_np = _denorm_points(obj.get("negative_points", []), W, H)

    if pos_np.shape[0] == 0 and neg_np.shape[0] == 0:
        pts = None
        labels = None
    else:
        pts = np.concatenate([pos_np, neg_np], axis=0) if pos_np.shape[0] + neg_np.shape[0] > 0 else None
        labels = None
        if pts is not None:
            labels = np.concatenate([
                np.ones(len(pos_np), dtype=np.int32),
                np.zeros(len(neg_np), dtype=np.int32)
            ], axis=0)

    return box_np, (pos_np if pos_np.size > 0 else None), (neg_np if neg_np.size > 0 else None), time_sec, pts, labels

# =========================
# Propagation (try bidirectional) — with cleanup
# =========================
def _collect_into(video_masks: Dict[int, np.ndarray], run_iter):
    for out_frame_idx, out_obj_ids, out_logits in run_iter:
        m = _squeeze_mask((out_logits[0] > 0).cpu().numpy())
        video_masks[int(out_frame_idx)] = m

def propagate_bidir(predictor, frame_dir: str, inference_state, ann_order: int,
                    boxes: Optional[np.ndarray], points: Optional[np.ndarray], labels: Optional[np.ndarray]) -> Dict[int, np.ndarray]:
    video_masks: Dict[int, np.ndarray] = {}
    state_b = None
    try:
        try:
            _collect_into(video_masks, predictor.propagate_in_video(
                inference_state, start_frame_idx=ann_order, reverse=False))
            forward_ok = True
        except TypeError:
            _collect_into(video_masks, predictor.propagate_in_video(inference_state))
            forward_ok = True
        except Exception as e:
            print(f"[WARN] forward propagation failed: {repr(e)}")
            forward_ok = False

        backward_done = False
        if forward_ok:
            try:
                _collect_into(video_masks, predictor.propagate_in_video(
                    inference_state, start_frame_idx=ann_order, reverse=True))
                backward_done = True
            except TypeError:
                backward_done = False
            except Exception as e:
                print(f"[WARN] backward(reverse=True) failed: {repr(e)}")
                backward_done = False

        if not backward_done:
            try:
                with suppress_stdout_stderr():
                    state_b = predictor.init_state(video_path=frame_dir)
                    box_np = boxes.astype(np.float32) if (boxes is not None and len(boxes) == 4) else None
                    if box_np is not None:
                        predictor.add_new_points_or_box(state_b, frame_idx=ann_order, obj_id=1, box=box_np)
                    if points is not None and len(points) > 0:
                        predictor.add_new_points_or_box(state_b, frame_idx=ann_order, obj_id=1,
                                                        points=points.astype(np.float32),
                                                        labels=labels.astype(np.int32),
                                                        box=box_np)

                if hasattr(predictor, "propagate_in_video_with_order"):
                    order = list(range(ann_order - 1, -1, -1))
                    try:
                        _collect_into(video_masks, predictor.propagate_in_video_with_order(state_b, order))
                        backward_done = True
                    except Exception as e:
                        print(f"[WARN] propagate_in_video_with_order failed: {repr(e)}")

                if not backward_done and hasattr(predictor, "propagate_in_video_to_frame"):
                    for idx in range(ann_order - 1, -1, -1):
                        try:
                            out = predictor.propagate_in_video_to_frame(state_b, idx)
                            if isinstance(out, tuple) and len(out) == 3:
                                out_frame_idx, out_obj_ids, out_logits = out
                                m = _squeeze_mask((out_logits[0] > 0).cpu().numpy())
                                video_masks[int(out_frame_idx)] = m
                            else:
                                if out is not None:
                                    m = _squeeze_mask((out[0] > 0).cpu().numpy())
                                    video_masks[int(idx)] = m
                        except Exception as e:
                            print(f"[WARN] to_frame({idx}) failed: {repr(e)}")
            except Exception as e:
                print(f"[WARN] backward fallback init_state failed: {repr(e)}")
    finally:
        try:
            if hasattr(predictor, "reset_state") and inference_state is not None:
                predictor.reset_state(inference_state)
        except Exception:
            pass
        try:
            if hasattr(predictor, "reset_state") and state_b is not None:
                predictor.reset_state(state_b)
        except Exception:
            pass
        del state_b
        free_cuda()
    return video_masks

# =========================
# —— Single-sample visualization & evaluation (using pre-extracted frames) — with finally cleanup
# =========================
def visualize_and_eval_one(predictor, sample: Dict[str, Any], device: torch.device, do_viz: bool) -> Dict[str, Any]:
    pid = sample.get("problem_id")
    data_type = sample.get("data_type")
    reso = sample.get("resolution", {})
    W, H = int(reso.get("width")), int(reso.get("height"))
    abs_path = os.path.join(DATA_ROOT, sample.get("path").lstrip("./"))

    inference_state = None
    out_mask_logits = None
    video_masks = None
    tmp_dir = None

    try:
        box_np, pos_np, neg_np, ann_time, points, labels = parse_pred_from_norm(sample, W, H)

        if data_type == "image":
            try:
                frame = Image.open(abs_path).convert("RGB")
                if frame.size != (W, H):
                    frame = frame.resize((W, H), Image.BILINEAR)
            except Exception as e:
                return {"problem_id": pid, "status": f"image-open-fail: {repr(e)}", "reward": None,
                        "viz_key_pred": None, "viz_key_gt": None, "viz_pred_video": None, "viz_gt_video": None}

            tmp_dir = tempfile.mkdtemp(prefix=f"sam2_{pid}_")
            frames_dir = os.path.join(tmp_dir, "frames")
            ensure_dir(frames_dir)
            key_jpg = os.path.join(frames_dir, f"000000{FRAME_IMG_EXT}")
            frame.save(key_jpg)

            with suppress_stdout_stderr():
                inference_state = predictor.init_state(video_path=frames_dir)
                out_mask_logits = None
                if box_np is not None:
                    _, _, out_mask_logits = predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=0, obj_id=1, box=box_np,
                    )
                if points is not None and len(points) > 0:
                    _, _, out_mask_logits = predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=0, obj_id=1,
                        points=points.astype(np.float32),
                        labels=labels.astype(np.int32) if labels is not None else None,
                        box=box_np,
                    )

            if out_mask_logits is None:
                return {"problem_id": pid, "status": "no-prompts", "reward": None,
                        "viz_key_pred": None, "viz_key_gt": None, "viz_pred_video": None, "viz_gt_video": None}

            pred_mask = _squeeze_mask((out_mask_logits[0] > 0).cpu().numpy())

            seg_out = sample.get("segmentation_output", {})
            gt_mask = None
            rle = seg_out.get("segmentation_rle", None)
            if rle:
                gt_mask = decode_coco_rle_to_mask(rle)
                if gt_mask.shape != (H, W):
                    gt_mask = np.array(Image.fromarray(gt_mask).resize((W, H), Image.NEAREST))

            # --- Image: record per-image IoU and inter/union (for accumulating cIoU) ---
            if gt_mask is not None:
                inter, union = compute_inter_union(pred_mask, gt_mask)
                iou_val = float(inter) / float(union) if union > 0 else 0.0
                reward = {"IoU": iou_val, "inter": int(inter), "union": int(union)}
            else:
                reward = {"IoU": 0.0, "inter": 0, "union": 0}

            key_pred_png, key_gt_png = None, None
            if do_viz:
                img_np = np.array(frame)
                ensure_dir(OUT_DIR)
                key_pred_png = os.path.join(OUT_DIR, f"viz_{pid}_key_pred.png")
                key_gt_png   = os.path.join(OUT_DIR, f"viz_{pid}_key_gt.png")

                vis_pred = overlay_mask(img_np, pred_mask, (1.0, 0.0, 1.0, 0.5))
                plt.figure(figsize=(8, 6))
                plt.title(f"problem_id={pid} (image) - Pred")
                plt.imshow(vis_pred)
                if box_np is not None: show_box_on_ax(box_np, plt.gca())
                show_points_on_ax(points, labels if labels is not None else None, plt.gca())
                plt.axis("off")
                plt.savefig(key_pred_png, bbox_inches="tight", dpi=150)
                plt.close("all")

                if gt_mask is not None:
                    vis_gt = overlay_mask(img_np, gt_mask, (0.0, 0.0, 1.0, 0.5))
                    plt.figure(figsize=(8, 6))
                    plt.title(f"problem_id={pid} (image) - GT")
                    plt.imshow(vis_gt)
                    if box_np is not None: show_box_on_ax(box_np, plt.gca())
                    show_points_on_ax(points, labels if labels is not None else None, plt.gca())
                    plt.axis("off")
                    plt.savefig(key_gt_png, bbox_inches="tight", dpi=150)
                    plt.close("all")
                else:
                    key_gt_png = None

            return {"problem_id": pid, "status": "ok", "reward": reward,
                    "viz_key_pred": key_pred_png, "viz_key_gt": key_gt_png,
                    "viz_pred_video": None, "viz_gt_video": None}

        elif data_type == "video":
            fps = float(sample.get("fps", 0) or 0)
            if fps <= 0:
                return {"problem_id": pid, "status": "bad-fps", "reward": None,
                        "viz_key_pred": None, "viz_key_gt": None, "viz_pred_video": None, "viz_gt_video": None}

            seg_out = sample.get("segmentation_output", {}) or {}
            frames_list: List[str] = seg_out.get("frames", []) or []
            rle_map: Dict[str, Any] = seg_out.get("segmentation_rle", {}) or {}

            if not frames_list:
                return {"problem_id": pid, "status": "no-frames-list", "reward": None,
                        "viz_key_pred": None, "viz_key_gt": None, "viz_pred_video": None, "viz_gt_video": None}

            frame_dir = video_to_frame_dir(abs_path)

            try:
                ann_time = float(ann_time or 0.0)
            except Exception:
                ann_time = 0.0
            ann_order = int(round(ann_time * fps))
            ann_order = max(0, min(ann_order, len(frames_list) - 1))

            try:
                jpgs = [p for p in os.listdir(frame_dir) if p.endswith(FRAME_IMG_EXT)]
            except FileNotFoundError:
                jpgs = []
            if not jpgs:
                return {"problem_id": pid, "status": "no-jpg-in-frame-dir", "reward": None,
                        "viz_key_pred": None, "viz_key_gt": None, "viz_pred_video": None, "viz_gt_video": None}

            with suppress_stdout_stderr():
                inference_state = predictor.init_state(video_path=frame_dir)
                if box_np is not None:
                    predictor.add_new_points_or_box(inference_state, frame_idx=ann_order, obj_id=1, box=box_np)
                if points is not None and len(points) > 0:
                    predictor.add_new_points_or_box(inference_state, frame_idx=ann_order, obj_id=1,
                                                    points=points.astype(np.float32),
                                                    labels=labels.astype(np.int32) if labels is not None else None,
                                                    box=box_np)
                video_masks = propagate_bidir(predictor, frame_dir, inference_state, ann_order, box_np, points, labels)

            Js, Fs = [], []
            for order, frame_key in enumerate(frames_list):
                rle = rle_map.get(frame_key, None)
                if not rle:
                    continue

                gt_mask = decode_coco_rle_to_mask(rle)
                if gt_mask.shape != (H, W):
                    gt_mask = np.array(Image.fromarray(gt_mask).resize((W, H), Image.NEAREST))

                pred_mask = video_masks.get(order, None)
                if pred_mask is None:
                    Js.append(0.0)
                    Fs.append(0.0)
                    continue

                if pred_mask.shape != (H, W):
                    pred_mask = np.array(Image.fromarray(pred_mask).resize((W, H), Image.NEAREST))

                j = compute_iou(pred_mask, gt_mask)
                f = boundary_fscore(pred_mask, gt_mask, tau_ratio=0.0075)
                Js.append(j); Fs.append(f)

            if Js:
                J_mean = float(np.mean(Js))
                F_mean = float(np.mean(Fs))
                JF_mean = float((J_mean + F_mean) / 2.0)
                reward = {"J": J_mean, "F": F_mean, "J&F": JF_mean}
            else:
                reward = {"J": 0.0, "F": 0.0, "J&F": 0.0}

            # Only export key-frame prediction image; MP4 does not overlay key-frame box/points
            key_pred_png = None
            pred_mp4, gt_mp4 = None, None
            if do_viz:
                tmp_dir_viz = tempfile.mkdtemp(prefix=f"sam2_{pid}_viz_")
                pred_frames_dir = os.path.join(tmp_dir_viz, "pred_frames")
                gt_frames_dir   = os.path.join(tmp_dir_viz, "gt_frames")
                ensure_dir(pred_frames_dir); ensure_dir(gt_frames_dir)

                for order, frame_key in enumerate(frames_list):
                    src_img_path = os.path.join(frame_dir, f"{order:06d}{FRAME_IMG_EXT}")
                    if not os.path.exists(src_img_path):
                        continue
                    img = np.array(Image.open(src_img_path).convert("RGB"))

                    pm = video_masks.get(order, None)
                    if pm is not None:
                        img_pred = overlay_mask(img, pm, (1.0, 0.0, 1.0, 0.5))
                    else:
                        img_pred = img

                    # —— Key-frame static image (only export PNG, do not modify frames used for MP4) ——
                    if order == ann_order:
                        ensure_dir(OUT_DIR)
                        # Overlay box/points on the static image
                        fig = plt.figure(figsize=(8, 6))
                        ax = plt.gca()
                        ax.imshow(img_pred)
                        if box_np is not None: show_box_on_ax(box_np, ax)
                        show_points_on_ax(points, labels if labels is not None else None, ax)
                        ax.axis("off")
                        key_pred_png = os.path.join(OUT_DIR, f"viz_{pid}_key_pred.png")
                        plt.savefig(key_pred_png, bbox_inches="tight", dpi=150)
                        plt.close(fig)

                    # Frames for MP4: do NOT additionally overlay box/points (keep consistent with other frames)
                    Image.fromarray(img_pred).save(os.path.join(pred_frames_dir, f"{order:06d}.png"))

                    # GT visualization video (only overlay GT mask; still no key-frame box/points)
                    rle = rle_map.get(frame_key, None)
                    if rle:
                        gm = decode_coco_rle_to_mask(rle)
                        img_gt = overlay_mask(img, gm, (0.0, 0.0, 1.0, 0.5))
                    else:
                        img_gt = img
                    Image.fromarray(img_gt).save(os.path.join(gt_frames_dir, f"{order:06d}.png"))

                ensure_dir(OUT_DIR)
                pred_mp4 = os.path.join(OUT_DIR, f"viz_{pid}_pred.mp4")
                gt_mp4   = os.path.join(OUT_DIR, f"viz_{pid}_gt.mp4")
                try:
                    ffmpeg_images_to_mp4(os.path.join(pred_frames_dir, "%06d.png"), pred_mp4, fps)
                except Exception as e:
                    print(f"[PID={pid}] MAKE_PRED_VIDEO_FAIL: {repr(e)}")
                    pred_mp4 = None
                try:
                    ffmpeg_images_to_mp4(os.path.join(gt_frames_dir,   "%06d.png"), gt_mp4, fps)
                except Exception as e:
                    print(f"[PID={pid}] MAKE_GT_VIDEO_FAIL: {repr(e)}")
                    gt_mp4 = None

                shutil.rmtree(tmp_dir_viz, ignore_errors=True)

            # Note: viz_key_gt is always None (no GT key-frame static image for video)
            return {"problem_id": pid, "status": "ok", "reward": reward,
                    "viz_key_pred": key_pred_png, "viz_key_gt": None,
                    "viz_pred_video": pred_mp4, "viz_gt_video": gt_mp4}

        else:
            return {"problem_id": pid, "status": "skip-unknown-data_type", "reward": None,
                    "viz_key_pred": None, "viz_key_gt": None, "viz_pred_video": None, "viz_gt_video": None}

    finally:
        try:
            if inference_state is not None and hasattr(predictor, "reset_state"):
                predictor.reset_state(inference_state)
        except Exception:
            pass

        try: del out_mask_logits
        except Exception: pass
        try: del video_masks
        except Exception: pass

        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        free_cuda()

# =========================
# —— worker
# =========================
def worker_run(args):
    worker_rank, gpu_id, samples_slice, prog_q = args
    world_size_info = os.environ.get("WORLD_SIZE_INFO", "")
    print(f"[WORKER {worker_rank}] start on GPU {gpu_id}  slice={len(samples_slice)} {world_size_info}")

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    try:
        torch.backends.cudnn.benchmark = False
        if device.type == "cuda" and torch.cuda.get_device_properties(device).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    torch.set_grad_enabled(False)

    def build_predictor_on_device():
        with suppress_stdout_stderr():
            predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=device)
        return predictor

    predictor = build_predictor_on_device()

    out_rows = []
    for global_idx, sample in samples_slice:
        pid = sample.get("problem_id")
        print(f"[WORKER {worker_rank} | PID={pid}] START data_type={sample.get('data_type')} "
              f"reso=({sample.get('resolution',{}).get('width')}, {sample.get('resolution',{}).get('height')}) "
              f"path={sample.get('path')}")

        try:
            if device.type == "cuda":
                torch.cuda.synchronize()

            do_viz = (VIZ_RATIO > 0.0) and (random.random() < VIZ_RATIO)

            if device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    row = visualize_and_eval_one(predictor, sample, device, do_viz)
            else:
                row = visualize_and_eval_one(predictor, sample, device, do_viz)

            if device.type == "cuda":
                torch.cuda.synchronize()

        except RuntimeError as e:
            if "CUDA" in str(e).upper():
                print(f"[WORKER {worker_rank} | CUDA ERROR pid={pid}] {repr(e)}")
                free_cuda()
                try:
                    predictor = build_predictor_on_device()
                    print(f"[WORKER {worker_rank}] predictor rebuilt after CUDA error @ pid={pid}")
                except Exception as e2:
                    print(f"[WORKER {worker_rank}] [WARN] predictor rebuild failed: {repr(e2)}")
                row = {
                    "problem_id": pid,
                    "status": f"error: {repr(e)}",
                    "reward": None,
                    "viz_key_pred": None,
                    "viz_key_gt": None,
                    "viz_pred_video": None,
                    "viz_gt_video": None,
                }
            else:
                row = {
                    "problem_id": pid,
                    "status": f"error: {repr(e)}",
                    "reward": None,
                    "viz_key_pred": None,
                    "viz_key_gt": None,
                    "viz_pred_video": None,
                    "viz_gt_video": None,
                }
                print(f"[WORKER {worker_rank} | ERROR pid={pid}] {repr(e)}")
                traceback.print_exc()
        except Exception as e:
            row = {
                "problem_id": pid,
                "status": f"error: {repr(e)}",
                "reward": None,
                "viz_key_pred": None,
                "viz_key_gt": None,
                "viz_pred_video": None,
                "viz_gt_video": None,
            }
            print(f"[WORKER {worker_rank} | ERROR pid={pid}] {repr(e)}")
            traceback.print_exc()

        out_rows.append((global_idx, row))

        try:
            if prog_q is not None:
                prog_q.put(1)
        except Exception:
            pass

        print(f"[WORKER {worker_rank} | PID={pid}] status={row.get('status')} reward={row.get('reward')}")
        free_cuda()

    try:
        del predictor
    except Exception:
        pass
    free_cuda()

    return out_rows

# =========================
# Preprocessing: scan JSON and pre-extract frames for all videos
# =========================
def pre_extract_all_frames(results: List[Dict[str, Any]]):
    uniq_videos = []
    seen = set()
    for sample in results:
        if sample.get("data_type") != "video":
            continue
        path = sample.get("path")
        if not path:
            continue
        abs_path = os.path.join(DATA_ROOT, path.lstrip("./"))
        if abs_path in seen:
            continue
        seen.add(abs_path)
        reso = sample.get("resolution", {}) or {}
        W, H = int(reso.get("width")), int(reso.get("height"))
        fps = float(sample.get("fps", 0) or 0)
        seg_out = sample.get("segmentation_output", {}) or {}
        frames_list = seg_out.get("frames", []) or []
        if fps > 0 and frames_list:
            uniq_videos.append( (abs_path, (W,H), fps, len(frames_list)) )

    if not uniq_videos:
        return

    for video_abs_path, (W,H), fps, n_frames in tqdm(uniq_videos, desc="Pre-extract videos", dynamic_ncols=True):
        frame_dir = video_to_frame_dir(video_abs_path)
        try:
            ensure_video_frames_extracted(
                video_abs_path=video_abs_path,
                frame_dir=frame_dir,
                fps=fps,
                num_frames=n_frames,
                target_wh=(W, H),
                threads=PRE_EXTRACT_THREADS,
                desc=f"extract:{os.path.basename(frame_dir)}"
            )
        except Exception as e:
            sys.stderr.write(f"[WARN] pre-extract fail: {video_abs_path} -> {repr(e)}\n")

# =========================
# Main entry (parallel + aggregation; write back JSON)
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True, help="输入预测 JSON 路径")
    parser.add_argument("--viz_ratio", type=float, default=0.0, help="可视化比例（0~1），0 表示不做可视化")
    args = parser.parse_args()

    in_dir  = os.path.dirname(os.path.abspath(args.input_json))
    stem    = os.path.splitext(os.path.basename(args.input_json))[0]
    out_json_name = f"{stem}_sam2.json"
    out_json_path = os.path.join(in_dir, out_json_name)
    out_viz_dir   = os.path.join(in_dir, f"{stem}_sam2")

    global PRED_JSON, OUT_JSON, OUT_DIR, VISUALIZE, VIZ_RATIO
    PRED_JSON = args.input_json
    OUT_JSON  = out_json_path
    OUT_DIR   = out_viz_dir
    VIZ_RATIO = max(0.0, min(1.0, float(args.viz_ratio)))
    VISUALIZE = VIZ_RATIO > 0.0

    if VISUALIZE:
        ensure_dir(OUT_DIR)

    with open(PRED_JSON, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "results" in payload:
        results = payload["results"]
        wrapper_dict = True
    elif isinstance(payload, list):
        results = payload
        wrapper_dict = False
    else:
        raise ValueError("输入 JSON 格式不正确：应为数组或包含 'results' 字段的对象。")

    pre_extract_all_frames(results)

    world_size = NUM_GPUS * WORKERS_PER_GPU if torch.cuda.is_available() else 1
    os.environ["WORLD_SIZE_INFO"] = f"(WORLD_SIZE={world_size}, NUM_GPUS={NUM_GPUS}, WPG={WORKERS_PER_GPU})"

    indexed = list(enumerate(results))
    total_samples = len(indexed)

    if world_size == 1:
        print("[INFO] CUDA 不可用，退化为单进程 CPU/MPS 串行。")
        prog_q = thread_queue.SimpleQueue()

        def consume_progress_single(q, total):
            with tqdm(total=total, desc="Processing", dynamic_ncols=True) as pbar:
                for _ in range(total):
                    try:
                        q.get()
                        pbar.update(1)
                    except Exception:
                        break

        prog_thread = Thread(target=consume_progress_single, args=(prog_q, total_samples), daemon=True)
        prog_thread.start()

        rows = worker_run((0, 0, indexed, prog_q))
        prog_thread.join(timeout=5)
        gathered = rows
    else:
        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        prog_q = manager.Queue()

        def consume_progress(q, total):
            with tqdm(total=total, desc="Processing", dynamic_ncols=True) as pbar:
                for _ in range(total):
                    try:
                        q.get()
                        pbar.update(1)
                    except Exception:
                        break

        prog_thread = Thread(target=consume_progress, args=(prog_q, total_samples), daemon=True)
        prog_thread.start()

        gathered = []
        for start in range(0, total_samples, EPOCH_SIZE):
            epoch_slice = indexed[start: start + EPOCH_SIZE]

            slices = [[] for _ in range(world_size)]
            for i, s in epoch_slice:
                w = i % world_size
                slices[w].append((i, s))

            worker_args = []
            for w in range(world_size):
                gpu_id = w // WORKERS_PER_GPU
                worker_args.append((w, gpu_id, slices[w], prog_q))

            with ctx.Pool(processes=world_size, maxtasksperchild=1) as pool:
                results_list = pool.map(worker_run, worker_args)

            for lst in results_list:
                gathered.extend(lst)

        prog_thread.join(timeout=5)

    gathered.sort(key=lambda x: x[0])
    outputs_only = [r for _, r in gathered]

    pid2row = {r["problem_id"]: r for r in outputs_only if "problem_id" in r}

    if wrapper_dict:
        new_results = []
        for sample in results:
            pid = sample.get("problem_id")
            row = pid2row.get(pid)
            if row is not None:
                s = dict(sample)
                s["reward"] = row.get("reward")
                s["status"] = row.get("status")
                s["viz_key_pred"] = row.get("viz_key_pred")
                s["viz_key_gt"] = row.get("viz_key_gt")
                s["viz_pred_video"] = row.get("viz_pred_video")
                s["viz_gt_video"] = row.get("viz_gt_video")
                new_results.append(s)
            else:
                new_results.append(sample)
        new_payload = dict(payload)
        new_payload["results"] = new_results
    else:
        new_results = []
        for sample in payload:
            pid = sample.get("problem_id")
            row = pid2row.get(pid)
            if row is not None:
                s = dict(sample)
                s["reward"] = row.get("reward")
                s["status"] = row.get("status")
                s["viz_key_pred"] = row.get("viz_key_pred")
                s["viz_key_gt"] = row.get("viz_key_gt")
                s["viz_pred_video"] = row.get("viz_pred_video")
                s["viz_gt_video"] = row.get("viz_gt_video")
                new_results.append(s)
            else:
                new_results.append(sample)
        new_payload = {"results": new_results}

    ok_items = [r for r in new_payload["results"] if r.get("status") == "ok" and isinstance(r.get("reward"), dict)]
    video_J, video_F, video_JF, image_IoU = [], [], [], []

    # --- NEW: accumulate total inter/union for cIoU ---
    total_inter, total_union = 0, 0

    for r in ok_items:
        rew = r["reward"]
        if "J" in rew and "F" in rew and "J&F" in rew:
            video_J.append(rew["J"]); video_F.append(rew["F"]); video_JF.append(rew["J&F"])
        if "IoU" in rew:
            image_IoU.append(rew["IoU"])
            # If inter/union exists (image samples will write it), accumulate for global cIoU
            if "inter" in rew and "union" in rew:
                total_inter += int(rew.get("inter", 0))
                total_union += int(rew.get("union", 0))

    avg_rewards = {}
    if video_J:
        avg_rewards["video/J"]   = float(np.mean(video_J))
        avg_rewards["video/F"]   = float(np.mean(video_F))
        avg_rewards["video/J&F"] = float(np.mean(video_JF))
    if image_IoU:
        # gIoU = mean of per-image IoU; keep original image/IoU key for compatibility
        giou_val = float(np.mean(image_IoU))
        avg_rewards["image/IoU"]  = giou_val
        avg_rewards["image/gIoU"] = giou_val
        # cIoU = total intersection / total union (more biased towards large objects)
        avg_rewards["image/cIoU"] = (float(total_inter) / float(total_union)) if total_union > 0 else 0.0

    new_payload["avg_rewards"] = avg_rewards

    ensure_dir(os.path.dirname(OUT_JSON) or ".")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(new_payload, f, ensure_ascii=False, indent=2)

    print(f"[DONE] saved: {OUT_JSON}")
    print(f"[INFO] ok={len(ok_items)} / total={len(outputs_only)}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.8")
    main()
