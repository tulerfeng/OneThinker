# -*- coding: utf-8 -*-
import os
import re
import json
import tempfile
import subprocess
from typing import Optional, Tuple, List, Dict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch

from sam2.build_sam import build_sam2_video_predictor

# ======= MANUAL CONFIG: SAM2 path & output dir =======

SAM2_CKPT = "/sam2/sam2.1_hiera_large.pt"
SAM2_CFG  = "//sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
OUT_DIR   = "./sam2_viz-full"


# Limit extracted visualization fps (for long videos)
MAX_FPS = 16.0

# Answer is expected between <answer>...</answer> as a JSON string
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


# =========================
# Small utilities
# =========================
def _ensure_dir(d: str):
    if d:
        os.makedirs(d, exist_ok=True)


def _clip_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(float(v)))))


def _denorm_xy(x: float, y: float, W: int, H: int) -> Tuple[int, int]:
    """
    Convert 0~1000 normalized coordinates back to pixel coordinates.
    """
    return _clip_int(x * W / 1000.0, 0, W), _clip_int(y * H / 1000.0, 0, H)


def _denorm_box(box: List[float], W: int, H: int) -> Optional[np.ndarray]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    x1, y1 = _denorm_xy(box[0], box[1], W, H)
    x2, y2 = _denorm_xy(box[2], box[3], W, H)
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _denorm_points(pts, W: int, H: int) -> np.ndarray:
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


def _overlay_mask(img: np.ndarray, mask: np.ndarray,
                  rgba=(1.0, 0.0, 1.0, 0.25)) -> np.ndarray:
    """
    Overlay a binary mask onto an RGB image with given RGBA color.
    """
    h, w = img.shape[:2]
    m = np.asarray(mask)
    if m.ndim == 3:
        if m.shape[0] == 1:
            m = m[0]
        elif m.shape[-1] == 1:
            m = m[..., 0]
    if m.shape != (h, w):
        m = np.array(Image.fromarray(m.astype(np.uint8)).resize((w, h), Image.NEAREST))

    out = img.copy().astype(np.float32) / 255.0
    color = np.array(rgba[:3], dtype=np.float32)
    alpha = float(rgba[3])

    mm = (m > 0).astype(np.float32)[..., None]
    out = out * (1 - mm * alpha) + color * (mm * alpha)
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return out


def _show_points_and_box(img: np.ndarray,
                         box: Optional[np.ndarray],
                         pos: Optional[np.ndarray],
                         neg: Optional[np.ndarray],
                         title: str,
                         save_path: str):
    """
    Draw box + positive/negative points on top of an image and save.
    """
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.imshow(img)

    if box is not None and box.size == 4:
        x0, y0, x1, y1 = box.tolist()
        ax.add_patch(plt.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            edgecolor="yellow", facecolor=(0, 0, 0, 0), lw=2
        ))

    if pos is not None and pos.size > 0:
        ax.scatter(pos[:, 0], pos[:, 1],
                   color="green", marker="*", s=50,
                   edgecolor="white", linewidth=0.33)

    if neg is not None and neg.size > 0:
        ax.scatter(neg[:, 0], neg[:, 1],
                   color="red", marker="*", s=50,
                   edgecolor="white", linewidth=0.33)

    ax.set_title(title)
    ax.axis("off")
    _ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close("all")


def _extract_answer_json(answer_text: str) -> Optional[dict]:
    """
    Extract JSON object from <answer>...</answer> section.
    """
    if not isinstance(answer_text, str):
        return None
    m = ANSWER_RE.search(answer_text)
    if not m:
        return None
    raw = m.group(1).strip()
    try:
        return json.loads(raw)
    except Exception as e:
        print(f"[SAM2-Vis] Failed to parse JSON from <answer>: {e} raw={raw!r}")
        return None


# =========================
# ffmpeg / ffprobe helpers
# =========================
def _ffprobe_get_fps_and_duration(video_path: str) -> Tuple[float, float]:
    """
    Use ffprobe to get (fps, duration_in_seconds) for the first video stream.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"[SAM2-Vis] ffprobe failed for {video_path}\n"
            + proc.stderr.decode("utf-8", errors="ignore")
        )

    lines = proc.stdout.decode("utf-8", errors="ignore").strip().splitlines()
    if len(lines) < 2:
        raise RuntimeError(f"[SAM2-Vis] Unexpected ffprobe output for {video_path}: {lines}")

    fps_line = lines[0].strip()
    dur_line = lines[1].strip()

    # Parse fps
    fps_native = 0.0
    if "/" in fps_line:
        num, den = fps_line.split("/", 1)
        try:
            num_f = float(num)
            den_f = float(den)
            fps_native = num_f / max(den_f, 1e-8)
        except Exception:
            fps_native = 0.0
    else:
        try:
            fps_native = float(fps_line)
        except Exception:
            fps_native = 0.0

    # Parse duration
    try:
        duration = float(dur_line)
    except Exception:
        duration = 0.0

    if fps_native <= 0:
        fps_native = 25.0  # rough fallback
    if duration <= 0:
        duration = 1.0      # avoid division by zero

    return fps_native, duration


def _ffmpeg_extract_all_frames(video_path: str, frames_dir: str, fps: float):
    """
    Extract frames at (downsampled) fps <= MAX_FPS.
    One extracted frame per (1 / fps) seconds.
    """
    _ensure_dir(frames_dir)
    pattern = os.path.join(frames_dir, "%06d.jpg")
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-y",
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-vsync", "0",
        pattern,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"[SAM2-Vis] ffmpeg frame extraction failed for {video_path}\n"
            + proc.stderr.decode("utf-8", errors="ignore")
        )

    files = sorted(
        f for f in os.listdir(frames_dir)
        if f.lower().endswith(".jpg")
    )
    if not files:
        raise RuntimeError(f"[SAM2-Vis] No frames extracted from {video_path} into {frames_dir}")


def _ffmpeg_images_to_mp4(img_pattern: str, out_mp4: str, fps: float):
    """
    Encode a sequence of images into an MP4 (H.264) at the given fps.
    """
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
        raise RuntimeError(
            f"[SAM2-Vis] ffmpeg video encoding failed: {out_mp4}\n"
            + proc.stderr.decode("utf-8", errors="ignore")
        )


def _propagate_bidir_simple(
    predictor,
    inference_state,
    key_idx: int,
) -> Dict[int, np.ndarray]:
    """
    Bidirectional propagation around key_idx.

    - First try forward propagation starting from key_idx.
    - Then try backward propagation (reverse=True) starting from key_idx.
    - If some APIs are not supported, we gracefully fall back.
    """
    video_masks: Dict[int, np.ndarray] = {}

    # ---------- forward propagation ----------
    try:
        iterator_fwd = predictor.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=key_idx,
            reverse=False,
        )
    except TypeError:
        # Older API without start_frame_idx/reverse:
        iterator_fwd = predictor.propagate_in_video(inference_state)

    for out_frame_idx, out_obj_ids, out_logits in iterator_fwd:
        m = (out_logits[0] > 0).cpu().numpy()
        video_masks[int(out_frame_idx)] = m

    # ---------- backward propagation (best effort) ----------
    try:
        iterator_bwd = predictor.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=key_idx,
            reverse=True,
        )
        for out_frame_idx, out_obj_ids, out_logits in iterator_bwd:
            m = (out_logits[0] > 0).cpu().numpy()
            video_masks[int(out_frame_idx)] = m
    except TypeError:
        # API does not support reverse / start_frame_idx; skip backward.
        pass
    except Exception as e:
        print(f"[SAM2-Vis] Backward propagation failed: {repr(e)}")

    return video_masks


# =========================
# Main visualization entry
# =========================
def visualize_segmentation(
    media_path: str,
    answer_text: str,
    is_video: bool = False,
) -> Optional[str]:
    """
    Simple SAM2-based segmentation visualization.

    For images:
      - Parse JSON prompts from <answer>...</answer>.
      - Run SAM2 on a single-frame "video".
      - Overlay mask + box + points, save PNG to OUT_DIR.
      - Return PNG path.

    For videos:
      - Use ffprobe to get native fps.
      - Use fps_vis = min(native_fps, MAX_FPS).
      - Extract frames at fps_vis.
      - Use JSON "time" (seconds) and fps_vis to pick the key frame index.
      - Run SAM2 with box + points on that key frame.
      - Bidirectionally propagate masks to all frames.
      - Overlay masks on every frame, save a temporary PNG sequence.
      - Encode PNGs into an MP4 via ffmpeg (using fps_vis).
      - Also export one key-frame PNG with box + points.
      - Return MP4 path if successful, otherwise key-frame PNG.
    """
    obj = _extract_answer_json(answer_text)
    if obj is None:
        print("[SAM2-Vis] No valid JSON inside <answer>...</answer>, skip.")
        return None

    _ensure_dir(OUT_DIR)

    # Decide device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(f"[SAM2-Vis] Build SAM2 predictor on {device}")

    predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=device)

    tmp_dir = tempfile.mkdtemp(prefix="sam2_vis_one_")
    frames_dir = os.path.join(tmp_dir, "frames")
    _ensure_dir(frames_dir)

    inference_state = None

    try:
        if not is_video:
            # ==================== IMAGE MODE ====================
            img = Image.open(media_path).convert("RGB")
            W, H = img.size

            # Save as "single-frame video"
            key_frame_path = os.path.join(frames_dir, "000000.jpg")
            img.save(key_frame_path)

            # Denormalize box/points
            box_np = None
            if isinstance(obj.get("boxes"), (list, tuple)) and len(obj["boxes"]) == 4:
                box_np = _denorm_box(obj["boxes"], W, H)

            pos_np = _denorm_points(obj.get("positive_points", []), W, H)
            neg_np = _denorm_points(obj.get("negative_points", []), W, H)
            if pos_np.size == 0:
                pos_np = None
            if neg_np.size == 0:
                neg_np = None

            points_all = None
            labels_all = None
            pts_list = []
            labels_list = []
            if pos_np is not None:
                pts_list.append(pos_np.astype(np.float32))
                labels_list.append(np.ones(len(pos_np), dtype=np.int32))
            if neg_np is not None:
                pts_list.append(neg_np.astype(np.float32))
                labels_list.append(np.zeros(len(neg_np), dtype=np.int32))
            if pts_list:
                points_all = np.concatenate(pts_list, axis=0)
                labels_all = np.concatenate(labels_list, axis=0)

            box_for_sam = box_np.astype(np.float32) if (box_np is not None and box_np.size == 4) else None

            # SAM2 inference on the single frame
            inference_state = predictor.init_state(video_path=frames_dir)
            _, _, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=points_all,
                labels=labels_all,
                box=box_for_sam,
            )

            if out_mask_logits is None:
                print("[SAM2-Vis] SAM2 returned no mask logits, skip.")
                return None

            mask = (out_mask_logits[0] > 0).cpu().numpy()
            img_np = _overlay_mask(np.array(img), mask, rgba=(1.0, 0.0, 1.0, 0.25))

            stem = os.path.splitext(os.path.basename(media_path))[0]
            out_png = os.path.join(OUT_DIR, f"seg_vis_img_{stem}.png")
            _show_points_and_box(
                img_np,
                box_np,
                pos_np,
                neg_np,
                title="segmentation (image)",
                save_path=out_png,
            )

            print(f"[SAM2-Vis] Image visualization saved to: {out_png}")
            return out_png

        else:
            # ==================== VIDEO MODE ====================
            print("[SAM2-Vis] Video mode: probing native fps and duration ...")
            fps_native, duration = _ffprobe_get_fps_and_duration(media_path)
            fps = min(fps_native, MAX_FPS)
            print(f"[SAM2-Vis] native_fps={fps_native:.4f}, duration={duration:.4f}s, used_fps={fps:.4f}")

            print("[SAM2-Vis] Extracting frames at used_fps ...")
            _ffmpeg_extract_all_frames(media_path, frames_dir, fps=fps)

            frame_files = sorted(
                f for f in os.listdir(frames_dir)
                if f.lower().endswith(".jpg")
            )
            if not frame_files:
                raise RuntimeError("[SAM2-Vis] No frames extracted for video visualization.")

            # Use the first frame to get width/height
            first_frame = Image.open(os.path.join(frames_dir, frame_files[0])).convert("RGB")
            W, H = first_frame.size

            # Denormalize prompts
            box_np = None
            if isinstance(obj.get("boxes"), (list, tuple)) and len(obj["boxes"]) == 4:
                box_np = _denorm_box(obj["boxes"], W, H)

            pos_np = _denorm_points(obj.get("positive_points", []), W, H)
            neg_np = _denorm_points(obj.get("negative_points", []), W, H)
            if pos_np.size == 0:
                pos_np = None
            if neg_np.size == 0:
                neg_np = None

            points_all = None
            labels_all = None
            pts_list = []
            labels_list = []
            if pos_np is not None:
                pts_list.append(pos_np.astype(np.float32))
                labels_list.append(np.ones(len(pos_np), dtype=np.int32))
            if neg_np is not None:
                pts_list.append(neg_np.astype(np.float32))
                labels_list.append(np.zeros(len(neg_np), dtype=np.int32))
            if pts_list:
                points_all = np.concatenate(pts_list, axis=0)
                labels_all = np.concatenate(labels_list, axis=0)

            # time in seconds; frame index ≈ round(time * used_fps)
            key_time = 0.0
            if "time" in obj and obj["time"] is not None:
                try:
                    key_time = float(obj["time"])
                except Exception:
                    key_time = 0.0

            key_idx = int(round(max(0.0, key_time) * fps))
            n_frames = len(frame_files)
            key_idx = max(0, min(key_idx, n_frames - 1))

            print(
                f"[SAM2-Vis] Video mode, using key frame index={key_idx} "
                f"(time≈{key_time:.3f}s, n_frames={n_frames}, used_fps={fps:.4f})"
            )

            # Build SAM2 state
            inference_state = predictor.init_state(video_path=frames_dir)
            box_for_sam = box_np.astype(np.float32) if (box_np is not None and box_np.size == 4) else None

            _, _, _ = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=key_idx,
                obj_id=1,
                points=points_all,
                labels=labels_all,
                box=box_for_sam,
            )

            print("[SAM2-Vis] Propagating masks bidirectionally ...")
            video_masks = _propagate_bidir_simple(
                predictor,
                inference_state=inference_state,
                key_idx=key_idx,
            )

            # Build visualization frames
            viz_frames_dir = os.path.join(tmp_dir, "viz_frames")
            _ensure_dir(viz_frames_dir)

            stem = os.path.splitext(os.path.basename(media_path))[0]
            key_png = os.path.join(OUT_DIR, f"seg_vis_vid_{stem}_key.png")

            for idx, fname in enumerate(sorted(frame_files)):
                img_path = os.path.join(frames_dir, fname)
                img = np.array(Image.open(img_path).convert("RGB"))

                mask = video_masks.get(idx, None)
                if mask is not None:
                    img_pred = _overlay_mask(img, mask, rgba=(1.0, 0.0, 1.0, 0.25))
                else:
                    img_pred = img

                out_frame_path = os.path.join(viz_frames_dir, f"{idx:06d}.png")
                Image.fromarray(img_pred).save(out_frame_path)

                # Save a key-frame PNG with box/points overlay for quick inspection
                if idx == key_idx:
                    _show_points_and_box(
                        img_pred,
                        box_np,
                        pos_np,
                        neg_np,
                        title=f"segmentation (video keyframe idx={key_idx})",
                        save_path=key_png,
                    )

            # Encode video
            mp4_out = os.path.join(OUT_DIR, f"seg_vis_vid_{stem}.mp4")
            try:
                _ffmpeg_images_to_mp4(
                    img_pattern=os.path.join(viz_frames_dir, "%06d.png"),
                    out_mp4=mp4_out,
                    fps=fps,
                )
                print(f"[SAM2-Vis] Video visualization saved to: {mp4_out}")
                return mp4_out
            except Exception as e:
                print(f"[SAM2-Vis] Failed to encode MP4: {repr(e)}, fallback to key PNG.")
                if os.path.exists(key_png):
                    return key_png
                return None

    finally:
        # Try to reset internal state, then delete predictor & temp dir
        try:
            if inference_state is not None and hasattr(predictor, "reset_state"):
                predictor.reset_state(inference_state)
        except Exception:
            pass
        try:
            del predictor
        except Exception:
            pass
        try:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
