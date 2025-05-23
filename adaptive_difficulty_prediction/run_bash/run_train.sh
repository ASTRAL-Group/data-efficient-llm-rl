#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Set specific parameters instead of looping
model_name="Qwen/Qwen2.5-Math-1.5B-Instruct"
ref_size=256
loss_type="binary_cross_entropy"
num_layers=3
scaling="group_logit_temp"
batch_size=256

# Run a single training job
accelerate launch \
  --num_processes 1 \
  train.py \
  --loss_type "$loss_type" \
  --model_name "$model_name" \
  --batch_size_per_gpu $batch_size \
  --ref_size $ref_size \
  --data_path datasets/ \
  --use_embeddings \
  --freeze_encoder \
  --epochs 20 \
  --seed 1 \
  --lr 1e-3 \
  --num_layers $num_layers \
  --use_scheduler \
  --save_predictions \
  --use_layernorm \
  --output_dir "outputs" \
  --scaling "$scaling" \
  --left_padding \
  --residual