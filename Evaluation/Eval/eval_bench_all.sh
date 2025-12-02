#!/usr/bin/env bash

model_paths=(
"model/OneThinker-8B" 
)

datasets=(
  "eval_longvideoreason.json"
  "eval_mmvu.json"
  "eval_videomathqa.json"
  "eval_videomme.json"
  "eval_videommmu.json"

  "eval_charades.json"
  "eval_anet_rtl.json"
  "eval_activitynet.json"

  "eval_refcoco_testA.json"
  "eval_refcoco_testB.json"
  "eval_refcoco_val.json"
  "eval_refcocop_testA.json"
  "eval_refcocop_testB.json"
  "eval_refcocop_val.json"
  "eval_refcocog_test.json"
  "eval_refcocog_val.json"

  "eval_stvg.json"

  "eval_got10k.json"

  "eval_seg_refcoco.json"
  "eval_seg_refcocog.json"
  "eval_seg_refcocop.json"

  "eval_seg_mevis.json"
  "eval_seg_reasonvos.json"
)


DATASET_PREFIX=<Evaluation-data-root>
OUT_ROOT_BASE=<Results-save-root>


DIR_SUFFIX="qwen3vl-256frame"

OUT_DIR="${OUT_ROOT_BASE%/}/${DIR_SUFFIX}"

SUFFIX="_cot_greedy_128"

MAX_PIXELS_VIDEO=$((256*32*32))  # = 262144
MAX_FRAMES=128
FPS=2


export CUDA_VISIBLE_DEVICES=0
export DECORD_EOF_RETRY_MAX=2048001


[ -d "$OUT_DIR" ] || mkdir -p "$OUT_DIR"

for model in "${model_paths[@]}"; do
  for ds_name in "${datasets[@]}"; do
    ds_path="${DATASET_PREFIX%/}/${ds_name}"
    python -u Evaluation/Eval/eval_bench.py \
      --model_path "$model" \
      --input_json "$ds_path" \
      --out_dir "$OUT_DIR" \
      --suffix "$SUFFIX" \
      --max_pixels_video "$MAX_PIXELS_VIDEO" \
      --max_frames "$MAX_FRAMES" \
      --fps "$FPS"
  done
done

# ====== SAM2 segmentation post-processing for each evaluation output ======
# Datasets that require SAM2 post-processing
seg_datasets=(
  "eval_seg_refcoco.json"
  "eval_seg_refcocog.json"
  "eval_seg_refcocop.json"
  "eval_seg_mevis.json"
  "eval_seg_reasonvos.json"
)

POST_SCRIPT="seg_post_sam2.py"
VIZ_RATIO=1.0  # 0 = no visualization; e.g., 0.1 = randomly visualize 10% of samples

for model in "${model_paths[@]}"; do
  for ds_name in "${datasets[@]}"; do

    # Only apply SAM2 to specified segmentation datasets
    [[ ! " ${seg_datasets[@]} " =~ " ${ds_name} " ]] && continue

    eval_json="${OUT_DIR}/$(basename "${ds_name%.*}")${SUFFIX}.json"
    if [[ -f "$eval_json" ]]; then
      echo "[POST] seg_post_sam2 -> $eval_json (viz_ratio=${VIZ_RATIO})"
      python "$POST_SCRIPT" --input_json "$eval_json" --viz_ratio "$VIZ_RATIO"
    else
      echo "[WARN] eval json not found, skip post: $eval_json"
    fi
  done
done
