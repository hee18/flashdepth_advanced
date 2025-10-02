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

#### Original FlashDepth Training
```bash
# First stage training (FlashDepth-L and FlashDepth-S at 518x518)
torchrun --nproc_per_node=8 train.py --config-path configs/flashdepth-l/ load=checkpoints/depth_anything_v2_vitl.pth dataset.data_root=<path_to_data>
torchrun --nproc_per_node=8 train.py --config-path configs/flashdepth-s/ load=checkpoints/depth_anything_v2_vits.pth dataset.data_root=<path_to_data>

# Second stage training (FlashDepth Full at higher resolution)
torchrun --nproc_per_node=8 train.py --config-path configs/flashdepth load=configs/flashdepth-s/<checkpoint.pth> hybrid_configs.teacher_model_path=configs/flashdepth-l/<checkpoint.pth> dataset.data_root=<path_to_data>
```

#### Metric Depth Training (GSP Head)
```bash
# Train Global Scale Predictor head (single GPU)
python train_metric_head.py --config-path configs/flashdepth \
  dataset.data_root=<path_to_data> \
  dataset.train_datasets=[tartanair] \
  dataset.val_datasets=[tartanair] \
  training.batch_size=12 \
  training.workers=4 \
  gpu=0

# With Docker
./run_docker.sh train --batch-size 12 --workers 4 --gpu 0
```

### Testing

```bash
# Test trained GSP model on TartanAir sequences
python test_metric_head.py \
  --config-path configs/flashdepth \
  --gsp-checkpoint train_results/results_1/best_metric_head_step_21000.pth \
  --results-dir test_results/results_1 \
  --frame-interval 2 \
  --gpu 0

# With Docker
./run_docker.sh test --frame-interval 2 --gpu 1
```

### Inference

```bash
# Run inference on video (relative depth)
torchrun train.py --config-path configs/flashdepth inference=true eval.random_input=<path_to_video> eval.outfolder=output

# Performance timing test
torchrun train.py --config-path configs/flashdepth inference=true eval.dummy_timing=true

# If encountering NaN comparison errors, add:
eval.compile=false
```

### Docker Commands

```bash
# Build Docker image
./run_docker.sh build

# Interactive shell for debugging
./run_docker.sh shell

# View logs
./run_docker.sh logs

# Clean up containers and volumes
./run_docker.sh clean
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
- Optional Global Scale Predictor (GSP) head for metric depth conversion

**Temporal Processing**:
- Primary: Mamba2 layers (`flashdepth/mamba.py`)
- Alternative modules: Hydra (bi-directional Mamba), xLSTM, Transformer RNN
- Configurable placement in DPT layers via `mamba_in_dpt_layer`

**Training Pipeline**:
- Two-stage training: low-res first (518x518), then high-res (2K)
- Hybrid model uses teacher-student distillation
- Multi-dataset training with configurable dataset combinations

**Metric Depth System** (`train_metric_head.py`, `test_metric_head.py`):
- Global Scale Predictor (GSP) head converts relative depth to metric depth
- Uses CLS token from DINOv2 encoder (1024-dim) to predict scale and shift
- Formula: `D_metric = Scale × D_relative + Shift`
- Freezes base FlashDepth model during GSP training (only ~262K trainable parameters)
- Training: BFloat16 mixed precision, cosine annealing with warmup, gradient clipping
- Testing: Comprehensive metrics including Temporal Alignment Error (TAE) for video consistency

### Data Loading

**Dataset Structure** (`dataloaders/`):
- `CombinedDataset`: Handles multiple datasets simultaneously
- Individual dataset loaders: MVS-Synth, Spring, TartanAir, PointOdyssey, DynamicReplica
- Video sequence loading with temporal consistency
- Resolution-aware preprocessing for different aspect ratios
- Custom collate function filters None values for robust batch processing

**Key Configuration**:
- `dataset.train_datasets`: List of training datasets
- `dataset.val_datasets`: List of validation datasets
- `dataset.video_length`: Temporal sequence length (5 for training, 50 for testing)
- `dataset.resolution`: Target resolution ('2k' or specific values like 518)

### Model Configuration

**Critical Settings** (`configs/*/config.yaml`):
- `model.use_mamba`: Enable/disable Mamba temporal processing
- `model.mamba_in_dpt_layer`: Which DPT layer to insert Mamba (moved to layer 1 in current implementation)
- `hybrid_configs.use_hybrid`: Enable teacher-student fusion
- `training.gradient_checkpointing`: Memory optimization for large models

**Metric Training Settings** (`train_metric_head.py`):
- Single GPU training (CUDA_VISIBLE_DEVICES controlled)
- Batch size: 12, Workers: 4 (optimized for GSP training)
- Learning rate: 1e-4 with cosine annealing (warmup → stable → decay)
- Loss: Log L1 loss between predicted and ground truth metric depth
- Gradient clipping: max_norm=1.0
- Checkpointing: Saves best model based on validation loss

## Development Notes

### Testing and Validation

**Training Validation**: Happens during training via `val_freq` (every 1000 steps)

**Metric Depth Testing** (`test_metric_head.py`):
- Tests on multiple TartanAir sequences (up to 4 sequences)
- Comprehensive metrics: MAE, RMSE, AbsRel, δ1/δ2/δ3, and TAE
- Temporal Alignment Error (TAE): Measures frame-to-frame consistency
- Frame-wise analysis: Identifies best/worst frames and optimal 5-frame sequences
- Generates extensive visualizations:
  - Depth sequence visualization (input, prediction, GT)
  - Best/worst frame comparisons with metrics overlay
  - TAE 5-frame analysis with temporal consistency metrics
  - Individual image exports (seq3 only)
- Outputs JSON results: per-sequence and averaged statistics

**Important Testing Details**:
- Resolution consistency: 518×518 for all stages (training, validation, testing)
- Valid mask handling: Combines GT validity (>0) and prediction range (0-1000m) to filter outliers
- BFloat16 compatibility: All tensors converted to Float32 for numpy operations
- GPU memory management: Cache cleared after each sequence
- Frame interval control: Adjustable visualization density (--frame-interval flag)

### Checkpoints and Logging

**Relative Depth Training**:
- Saved automatically every `save_freq` iterations to config directory
- Uses Hydra for configuration management and optional Wandb integration
- Distributed training built for multi-GPU with torchrun

**Metric Depth Training**:
- Single GPU setup (no DDP)
- Saves best model to `results_dir/best_metric_head_step_XXXX.pth`
- Also saves `latest_metric_head.pth` for resumption
- Visualizations saved at strategic intervals: steps 1, 10, 50, 100, then every 250 steps
- Optional Wandb logging: set `training.wandb=true` in config

### Memory Optimization

- Large models require gradient checkpointing and careful batch sizing
- Metric training uses BFloat16 mixed precision
- GPU cache cleared between video sequences during testing
- Custom collate function in dataloaders handles None values without crashes

### Docker Environment

**Files**: `Dockerfile`, `docker-compose.yml`, `run_docker.sh`, `test_docker_setup.py`

**Key Features**:
- Ubuntu 22.04 base with Python 3.11
- PyTorch 2.4.0 (CPU by default, GPU requires CUDA base image)
- Volume mounting: datasets, results, checkpoints
- Easy-to-use run scripts with GPU selection
- Dataset path handling: expects lowercase directory names (tartanair, not Tartanair)

**GPU Setup**:
- Install NVIDIA Container Runtime on host
- Update Dockerfile to use `nvidia/cuda:11.8-devel-ubuntu22.04`
- Install CUDA-enabled PyTorch and enable Mamba2/Flash Attention compilation

## Data Format

Training data should follow specific directory structures as documented in `dataloaders/README.md`. Each dataset has its own format requirements for images and depth files.

**Key Points**:
- TartanAir: Depth files in `.npy` format, already in metric depth (meters)
- MVS-Synth: Depth in `.exr` format
- Spring: Disparity in `.dsp5` format (requires conversion to depth)
- PointOdyssey: Depth in `.png` format
- DynamicReplica: Depth in `.geometric.png` format

## Metric Depth System Architecture

### Conversion Pipeline
```
Input Video → FlashDepth (Frozen) → Relative Depth (0-1 normalized)
                     ↓
              CLS Token → GSP Head (Trainable) → Scale, Shift
                                                      ↓
                           Metric Depth = Scale × Relative Depth + Shift
```

### GSP Head Structure
- Input: CLS token from DINOv2 (1024-dim)
- Architecture: Linear(1024→256) → ReLU → Linear(256→2)
- Output: Scale (positive, Softplus activation) and Shift (real number)

### Training Strategy
- Freeze entire FlashDepth backbone (ViT, DPT, Mamba modules)
- Train only GSP head (~262K parameters vs ~340M total)
- Use datasets with ground truth metric depth (TartanAir, etc.)
- Optimize with Log L1 loss for scale-invariant learning

### Evaluation Metrics
- **Standard**: MAE, RMSE, AbsRel, δ1/δ2/δ3 (threshold accuracy)
- **Temporal**: TAE (Temporal Alignment Error) - measures frame-to-frame consistency
- **Analysis**: Best/worst frame identification, optimal 5-frame sequence detection

See `flashdepth_advanced.md` for complete technical specifications of the metric depth system.
