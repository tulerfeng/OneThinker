# -*- coding: utf-8 -*-
import os
import torch
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

# Import SAM2 visualization (used only for segmentation)
from simple_sam2_vis import visualize_segmentation

# vLLM multiprocessing mode
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# ================== MANUAL CONFIGURATION ==================
# Model checkpoint
CHECKPOINT_PATH = "OneThink/OneThinker-8B"

# Media path (image or video)
MEDIA_PATH = ""
# Whether the media is a video
IS_VIDEO = False
# Question text
QUESTION_TEXT = ""

# Problem type (must be a key in TYPE_TEMPLATE)
# Examples:
# "open-ended", "multiple choice", "math",
# "temporal grounding", "spatial grounding",
# "spatial-temporal grounding", "tracking",
# "segmentation_image", "segmentation_video"
PROBLEM_TYPE = "segmentation_video"

# ==========================================================

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
        "Example:\n"
        "<answer>{\"boxes\": [x1, y1, x2, y2], \"positive_points\": [[x,y],[x,y],[x,y]], "
        "\"negative_points\": [[x,y],[x,y],[x,y]]}</answer>"
    ),
    "segmentation_video": (
        "This task prepares inputs for video object segmentation with a specialized model (e.g., SAM2).\n"
        "Please select ONE representative time (in seconds), and provide ONE bounding box, "
        "3 positive points (clearly INSIDE the object), and 3 negative points (clearly OUTSIDE the object) "
        "within the <answer>...</answer> tags.\n"
        "Example:\n"
        "<answer>{\"time\": <time_in_seconds>, \"boxes\": [x1, y1, x2, y2], "
        "\"positive_points\": [[x,y],[x,y],[x,y]], \"negative_points\": [[x,y],[x,y],[x,y]]}</answer>"
    )
}


def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    print(f"video_kwargs: {video_kwargs}")

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    return {
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }


if __name__ == "__main__":
    # Build final prompt text
    type_hint = TYPE_TEMPLATE.get(PROBLEM_TYPE, "")
    full_text = QUESTION_TEMPLATE.format(Question=QUESTION_TEXT) + type_hint

    # Build message in plain form
    if IS_VIDEO:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": MEDIA_PATH,
                        "max_pixels": 256 * 32 * 32,
                        "max_frames": 128,
                        "fps": 2,
                    },
                    {"type": "text", "text": full_text},
                ],
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": MEDIA_PATH,
                        "max_pixels": 1024 * 32 * 32,
                    },
                    {"type": "text", "text": full_text},
                ],
            }
        ]

    # vLLM inference
    processor = AutoProcessor.from_pretrained(CHECKPOINT_PATH)
    inputs = [prepare_inputs_for_vllm(messages, processor)]

    llm = LLM(
        model=CHECKPOINT_PATH,
        mm_encoder_tp_mode="data",
        tensor_parallel_size=1,
        max_model_len = 81920,
        gpu_memory_utilization=0.7,
        seed=0,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        top_k=-1,
        stop_token_ids=[],
    )

    print("\n========== PROMPT ==========")
    print(inputs[0]["prompt"])
    print("============================\n")

    outputs = llm.generate(inputs, sampling_params=sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text
        print("\n========== MODEL OUTPUT ==========")
        print(generated_text)

        # Automatically visualize if it is a segmentation task
        if PROBLEM_TYPE in ["segmentation_image", "segmentation_video"]:
            print("\n[Segmentation] Running SAM2 visualization...")
            vis_path = visualize_segmentation(
                media_path=MEDIA_PATH,
                answer_text=generated_text,
                is_video=IS_VIDEO,
            )
            print(f"[Segmentation] Visualization saved to: {vis_path}")
