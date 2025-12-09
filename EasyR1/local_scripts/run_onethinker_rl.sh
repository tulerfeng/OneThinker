
#!/usr/bin/env bash
set -x

export DECORD_EOF_RETRY_MAX=2048001
export WANDB_API_KEY=<YOUR_KEY>


project_name='EasyR1-onethinker-rl'
exp_name='qwen3_vl_onethinker-rl'

MODEL_PATH=<SFT_Model>
TRAIN_FILE="onethinker_rl_train.json"
TEST_FILE="onethinker_rl_train.json"
IMAGE_DIR=<Train_Data_Path>

ROLLOUT_BS=128
GLOBAL_BS=32
MB_PER_UPDATE=1
MB_PER_EXP=1
TP_SIZE=4
N_GPUS_PER_NODE=8
NNODES=4



python3 -m verl.trainer.main \
    config=EasyR1/examples/config_ema_grpo_64.yaml \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.image_dir="${IMAGE_DIR}" \
    data.rollout_batch_size="${ROLLOUT_BS}" \
    worker.actor.global_batch_size="${GLOBAL_BS}" \
    worker.actor.micro_batch_size_per_device_for_update="${MB_PER_UPDATE}" \
    worker.actor.micro_batch_size_per_device_for_experience="${MB_PER_EXP}" \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.optim.lr=2e-6 \
    worker.rollout.tensor_parallel_size="${TP_SIZE}" \
    algorithm.filter_low=0.01 \
    algorithm.filter_high=0.99 \
    algorithm.online_filtering=true \
    algorithm.filter_key=accuracy \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq=100 \
    trainer.save_checkpoint_path=EasyR1/checkpoints


