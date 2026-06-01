#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/workspace/data/cifar10-imagefolder}"
WORK_DIR="${WORK_DIR:-/workspace/runs}"
NUM_CLASSES="${NUM_CLASSES:-10}"
NUM_PROCESSES="${NUM_PROCESSES:-$(python - <<'PY'
import torch
print(max(torch.cuda.device_count(), 1))
PY
)}"

ACCELERATE_ARGS=(launch --num_processes "${NUM_PROCESSES}")
if [ "${NUM_PROCESSES}" -gt 1 ]; then
  ACCELERATE_ARGS+=(--multi_gpu)
fi

accelerate "${ACCELERATE_ARGS[@]}" train.py \
  --experiment-name "${EXPERIMENT_NAME:-resnet50-runpod-ddp}" \
  --data-dir "${DATA_DIR}" \
  --work-dir "${WORK_DIR}" \
  --num-classes "${NUM_CLASSES}" \
  --epochs "${EPOCHS:-10}" \
  --save-checkpoint-interval "${SAVE_CHECKPOINT_INTERVAL:-1}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --learning-rate "${LEARNING_RATE:-0.1}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" \
  --momentum "${MOMENTUM:-0.9}" \
  --step-lr-decay "${STEP_LR_DECAY:-0.1}" \
  --lr-step-size "${LR_STEP_SIZE:-30}" \
  --lr-warmup-epochs "${LR_WARMUP_EPOCHS:-0}" \
  --max-grad-norm "${MAX_GRAD_NORM:-1.0}" \
  --img-size "${IMG_SIZE:-224}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --mixed-precision "${MIXED_PRECISION:-bf16}" \
  --log-with "${LOG_WITH:-none}"
