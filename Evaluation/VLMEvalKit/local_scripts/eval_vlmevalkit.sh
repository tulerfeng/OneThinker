


export LMUData=<TSV_Filefolder>




MODEL=<Model_Path>

WORK_BASE=<Results_Save_Path>


DATASETS=(

  "MathVista_MINI"
  "MathVerse_MINI"
  "MMBench_DEV_EN"
  "MMMU_DEV_VAL"
  "MMStar"
  "AI2D_TEST"
  "ScienceQA_TEST"
  "MMT-Bench_VAL"
  "MMSci_DEV_Captioning_image_only"
  "MMVet"

  "Video_Holmes_128frame"
  'LongVideoBench_128frame'
  'Video_MMLU_CAP_128frame'


)


SUFFIX="results"

export CUDA_VISIBLE_DEVICES=0

for DATA in "${DATASETS[@]}"; do

  WORK_DIR="${WORK_BASE}/${SUFFIX}"

  python run.py \
    --data "${DATA}" \
    --model "${MODEL}" \
    --work-dir "${WORK_DIR}" \
    --verbose \
    --reuse
done
