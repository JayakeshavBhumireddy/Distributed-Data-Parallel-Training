#!/usr/bin/env bash
set -euo pipefail

accelerate launch train.py \
  --experiment-name "resnet50-local-demo" \
  --data-dir "./data/cifar10-imagefolder" \
  --work-dir "./runs" \
  --num-classes 10 \
  --epochs 2 \
  --save-checkpoint-interval 1 \
  --batch-size 64 \
  --gradient-accumulation-steps 1 \
  --learning-rate 0.01 \
  --weight-decay 1e-4 \
  --momentum 0.9 \
  --step-lr-decay 0.1 \
  --lr-step-size 30 \
  --lr-warmup-epochs 0 \
  --max-grad-norm 1.0 \
  --img-size 224 \
  --num-workers 4 \
  --log-with none
