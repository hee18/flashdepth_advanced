#!/bin/bash
# Train Gear3 with 2 GPUs (GPU 0 and 1)

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh  # Or miniconda3
conda activate flashdepth

# Set environment
export CUDA_VISIBLE_DEVICES=0,1

# Run with torchrun
torchrun \
    --nproc_per_node=2 \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    train_gear3.py \
    "$@"
