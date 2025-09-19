# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

This project requires specific setup due to Mamba2 dependencies and torch version constraints:

```bash
conda create -n flashdepth python=3.11 --yes
conda activate flashdepth
bash setup_env.sh
```

**Important**: Torch version must be <= 2.4 for Mamba2 compatibility. The setup installs a local Mamba package from `./mamba/` directory.

## Common Commands

### Training
```bash
# First stage training (FlashDepth-L and FlashDepth-S at 518x518)
torchrun --nproc_per_node=8 train.py --config-path configs/flashdepth-l/ load=checkpoints/depth_anything_v2_vitl.pth dataset.data_root=<path_to_data>
torchrun --nproc_per_node=8 train.py --config-path configs/flashdepth-s/ load=checkpoints/depth_anything_v2_vits.pth dataset.data_root=<path_to_data>

# Second stage training (FlashDepth Full at higher resolution)
torchrun --nproc_per_node=8 train.py --config-path configs/flashdepth load=configs/flashdepth-s/<checkpoint.pth> hybrid_configs.teacher_model_path=configs/flashdepth-l/<checkpoint.pth> dataset.data_root=<path_to_data>
```

### Inference
```bash
# Run inference on video
torchrun train.py --config-path configs/flashdepth inference=true eval.random_input=<path_to_video> eval.outfolder=output

# Performance timing test
torchrun train.py --config-path configs/flashdepth inference=true eval.dummy_timing=true

# If encountering NaN comparison errors, add:
eval.compile=false
```

### Model Variants
- **FlashDepth (Full)**: Fastest, best for high resolution (configs/flashdepth)
- **FlashDepth-L**: Most accurate, recommended for low resolution <518 (configs/flashdepth-l)
- **FlashDepth-S**: Smallest model (configs/flashdepth-s)

## Architecture

### Core Components

**FlashDepth Model** (`flashdepth/model.py`):
- Vision Transformer backbone (DINOv2) with variants: ViT-S, ViT-L
- Mamba2 temporal modules for video sequence processing
- DPT (Dense Prediction Transformer) head for depth estimation
- Hybrid fusion system combining teacher-student models

**Temporal Processing**:
- Primary: Mamba2 layers (`flashdepth/mamba.py`)
- Alternative modules: Hydra (bi-directional Mamba), xLSTM, Transformer RNN
- Configurable placement in DPT layers via `mamba_in_dpt_layer`

**Training Pipeline**:
- Two-stage training: low-res first (518x518), then high-res (2K)
- Hybrid model uses teacher-student distillation
- Multi-dataset training with configurable dataset combinations

### Data Loading

**Dataset Structure** (`dataloaders/`):
- `CombinedDataset`: Handles multiple datasets simultaneously
- Individual dataset loaders: MVS-Synth, Spring, TartanAir, PointOdyssey, DynamicReplica
- Video sequence loading with temporal consistency
- Resolution-aware preprocessing for different aspect ratios

**Key Configuration**:
- `dataset.train_datasets`: List of training datasets
- `dataset.val_datasets`: List of validation datasets
- `dataset.video_length`: Temporal sequence length
- `dataset.resolution`: Target resolution ('2k' or specific values)

### Model Configuration

**Critical Settings** (`configs/*/config.yaml`):
- `model.use_mamba`: Enable/disable Mamba temporal processing
- `model.mamba_in_dpt_layer`: Which DPT layer to insert Mamba (moved to layer 1 in current implementation)
- `hybrid_configs.use_hybrid`: Enable teacher-student fusion
- `training.gradient_checkpointing`: Memory optimization for large models

## Development Notes

- **No traditional test suite**: Validation happens during training via `val_freq`
- **Checkpoints**: Saved automatically every `save_freq` iterations to config directory
- **Logging**: Uses Hydra for configuration management and optional Wandb integration
- **Distributed Training**: Built for multi-GPU training with torchrun
- **Memory**: Large models require gradient checkpointing and careful batch sizing

## Data Format

Training data should follow specific directory structures as documented in `dataloaders/README.md`. Each dataset has its own format requirements for images and depth files.