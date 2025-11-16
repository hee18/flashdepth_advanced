# GEAR5: Metric Depth Enhancement System

This document covers three variants for enhancing FlashDepth with metric depth capabilities:
- **Gear5 (GRU)**: GRU-based temporal scale/shift prediction (default)
- **Gear5 (Mamba)**: Mamba2-based temporal scale/shift prediction (lightweight alternative)
- **Gear5 FiLM**: FiLM-style channel-wise feature modulation (trainable Mamba)

All variants share common components (CLS token extraction, importance map generation) but differ in their temporal modeling and modulation strategies.

---

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Gear5 GRU: GRU-Based Temporal Modulation](#gear5-gru-gru-based-temporal-modulation)
- [Gear5 Mamba: Mamba2-Based Temporal Modulation](#gear5-mamba-mamba2-based-temporal-modulation)
- [Gear5 FiLM: Channel-Wise Feature Modulation](#gear5-film-channel-wise-feature-modulation)
- [Comparison](#comparison)
- [Loss Functions](#loss-functions)
- [Training & Testing](#training--testing)
- [Configuration](#configuration)
- [Recent Updates](#recent-updates)

---

## Architecture Overview

All Gear5 variants extend FlashDepth with metric depth prediction capabilities by:
1. Extracting semantic features from multi-layer CLS tokens
2. Generating importance maps from attention weights for loss weighting
3. Applying learned transformations to enhance depth predictions

**Key Shared Components**:
- **Multi-layer CLS Token Extraction**: Layers [11, 23] for ViT-L, [5, 11] for ViT-S
- **ImportanceMapGenerator**: CLS-to-patch attention → importance map for loss weighting
- **Loss Functions**: All support `log_l1` (standard) and `importance` (weighted) loss types
- **Canonical Space**: All training/inference uses canonical focal length (500.0 for 518×518, configurable via `canonical_focal_length`; on/off via `use_canonical_space`)

**Key Differences**:
| Component | Gear5 (GRU) | Gear5 (Mamba) | Gear5 FiLM |
|-----------|-------------|---------------|------------|
| **Modulation Target** | Final relative depth | Final relative depth | DPT path_1 features |
| **Modulation Method** | GRU-based scale/shift | Mamba2-based scale/shift | FiLM gamma/beta |
| **Temporal Backend** | Bi-GRU (in head) | Mamba2 (in head) | Mamba (existing) |
| **Trainable Params** | ~132K | ~147K | ~1.03M |
| **Frozen Components** | All except Gear5Head | All except Gear5Head | ViT + DPT + conv1 |
| **Training Speed** | Fast | Fast | Slower |
| **Memory Usage** | Low | Low | Higher |

---

## Gear5 (GRU): GRU-Based Temporal Modulation

### Architecture

Gear5 (GRU) applies **temporal scale and shift** to the final relative depth output using a GRU-based predictor.

```
Video Input [B, T, 3, H, W]
    ↓
 ViT Encoder (Frozen)
    ↓
CLS Tokens [Layers 11, 23] → ImportanceMapGenerator → Importance Map
    ↓                              ↓
TemporalScalePredictor          Loss Weighting
 (Bi-GRU, 2 layers)
    ↓
Scale [B, T, 1, 1, 1], Shift [B, T, 1, 1, 1]
    ↓
Relative Depth × Scale + Shift = Metric Depth
```

### Key Components

#### 1. TemporalScalePredictor (GRU)
Predicts scale and shift parameters using a bidirectional GRU for temporal consistency.

```python
class TemporalScalePredictor(nn.Module):
    """
    CLS token [B, T, 1024] → Linear → Bi-GRU → Linear → Scale/Shift

    Architecture:
        - Input projection: 1024 → 256
        - Bi-directional GRU: hidden_dim=256, 2 layers
        - Output projection: 512 (bidirectional) → 2 (scale, shift)

    Trainable params: ~132K
    """
```

**Forward Flow**:
```python
cls_token: [B, T, 1024]  # Last layer CLS token (Layer 23)
    ↓ Linear(1024 → 256)
cls_features: [B, T, 256]
    ↓ Bi-GRU(2 layers, 256 hidden)
gru_output: [B, T, 512]  # 256 × 2 (bidirectional)
    ↓ Linear(512 → 2)
scale_shift: [B, T, 2]
    ↓ Split
scale: [B, T, 1, 1, 1] = exp(scale_raw)  # Ensures positive
shift: [B, T, 1, 1, 1] = shift_raw        # Real number (can be negative)
```

**Temporal Processing**: GRU processes sequence bidirectionally, ensuring each frame's scale/shift considers past and future context.

### Training Command

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=1 python train_gear5.py \
  --config-path configs/gear5 \
  training.iterations=60001 \
  dataset.data_root=/path/to/datasets \
  +loss_type=log_l1

# Multi-GPU (DDP)
./train_gear5_ddp.sh \
  --config-path configs/gear5 \
  --results-dir train_results/results_X/gear_5_gru/large/ \
  --batch-size 20 \
  --loss importance
```

**Key Parameters**:
- `training.gear5_lr`: 1.0e-4 (TemporalScalePredictor learning rate)
- `training.batch_size`: 20 per GPU (effective 40 with 2 GPUs)
- Trainable params: ~132K

---

## Gear5 (Mamba): Mamba2-Based Temporal Modulation

### Architecture

Gear5 (Mamba) replaces the GRU with **Mamba2** for temporal scale/shift prediction, offering a lightweight alternative with similar parameter count.

```
Video Input [B, T, 3, H, W]
    ↓
 ViT Encoder (Frozen)
    ↓
CLS Tokens [Layers 11, 23] → ImportanceMapGenerator → Importance Map
    ↓                              ↓
TemporalScalePredictor          Loss Weighting
 (Mamba2, 1 layer)
    ↓
Scale [B, T, 1, 1, 1], Shift [B, T, 1, 1, 1]
    ↓
Relative Depth × Scale + Shift = Metric Depth
```

### Key Components

#### 1. TemporalScalePredictor (Mamba2)
Uses Mamba2 for efficient temporal modeling with state-space architecture.

```python
class TemporalScalePredictor(nn.Module):
    """
    CLS token [B, T, 1024] → Linear → Mamba2 → Linear → Scale/Shift

    Architecture:
        - Input projection: 1024 → 256
        - Mamba2: d_model=256, d_state=64, d_conv=4, expand=2
        - Output projection: 256 → 2 (scale, shift)

    Trainable params: ~147K
    """
```

**Forward Flow**:
```python
cls_token: [B, T, 1024]  # Last layer CLS token (Layer 23)
    ↓ Linear(1024 → 256)
cls_features: [B, T, 256]
    ↓ Mamba2(d_state=64, d_conv=4)
mamba_output: [B, T, 256]
    ↓ Linear(256 → 2)
scale_shift: [B, T, 2]
    ↓ Split
scale: [B, T, 1, 1, 1] = exp(scale_raw)  # Ensures positive
shift: [B, T, 1, 1, 1] = shift_raw        # Real number (can be negative)
```

**Advantages over GRU**:
- More efficient temporal modeling with state-space mechanism
- Better long-range dependency capture
- Slightly higher parameter count (~147K vs ~132K)

### Training Command

```bash
# Single GPU with Mamba2 backend
CUDA_VISIBLE_DEVICES=1 python train_gear5.py \
  --config-path configs/gear5 \
  training.iterations=60001 \
  dataset.data_root=/path/to/datasets \
  +loss_type=log_l1 \
  +use_mamba_predictor=true

# Multi-GPU (DDP) - using Docker script
./train_gear5_ddp.sh \
  --config-path configs/gear5 \
  --results-dir train_results/results_X/gear_5_mamba/large/ \
  --batch-size 20 \
  --loss importance \
  --mamba  # Enable Mamba2 instead of GRU
```

**Key Parameters**:
- `use_mamba_predictor`: true (use Mamba2 instead of GRU)
- `training.gear5_lr`: 1.0e-4
- Trainable params: ~147K

---

## Gear5 FiLM: Channel-Wise Feature Modulation

### Architecture

Gear5 FiLM applies **channel-wise FiLM modulation** to DPT features before Mamba temporal modeling.

```
Video Input [B, T, 3, H, W]
    ↓
 ViT Encoder (Frozen)
    ↓
CLS Tokens [Layers 11, 23] → GlobalFeatureNetwork → ModulationNetwork
    ↓                              ↓                      ↓
DPT (Frozen)                  Global Feature         Gamma, Beta
    ↓                              ↓                      ↓
path_1 features          SimpleFeatureModulator (Channel-wise)
    ↓
Modulated path_1 → Mamba (Trainable) → output_conv2 (Trainable) → Metric Depth
    ↓
ImportanceMapGenerator → Importance Map (Loss Weighting)
```

### Key Components

#### 1. GlobalFeatureNetwork
Extracts global semantic features from CLS token.

```python
class GlobalFeatureNetwork(nn.Module):
    """
    CLS token → Global semantic feature

    Architecture:
        - Linear(1024 → 512) → ReLU
        - Linear(512 → 256) → ReLU
    """
```

#### 2. ModulationNetwork
Generates channel-wise gamma and beta for FiLM modulation.

```python
class ModulationNetwork(nn.Module):
    """
    Global feature → Gamma, Beta (channel-wise)

    Architecture:
        - Linear(256 → 512) → ReLU
        - Linear(512 → 512)  # First 256: gamma, Last 256: beta
    """
```

**Key Concept**: Each of 256 DPT channels gets its own gamma/beta pair, applied uniformly across spatial locations.

#### 3. SimpleFeatureModulator
Applies FiLM modulation: `feature * gamma + beta`

### Training Command

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=1 python train_gear5_film.py \
  --config-path configs/gear5_film \
  training.iterations=60001 \
  dataset.data_root=/path/to/datasets \
  +loss_type=log_l1

# Multi-GPU (DDP)
./train_gear5_film_ddp.sh \
  --config-path configs/gear5_film \
  --results-dir train_results/results_X/gear_5_film/large/ \
  --batch-size 20 \
  --loss importance
```

**Key Parameters**:
- `training.film_lr`: 1.0e-4 (Gear5FilmHead learning rate)
- `training.mamba_lr`: 1.0e-5 (Mamba temporal modules)
- `training.output_lr`: 1.0e-5 (output_conv2)
- Trainable params: ~1.03M (head + Mamba + conv2)

---

## Comparison

### Feature Comparison

| Feature | Gear5 (GRU) | Gear5 (Mamba) | Gear5 FiLM |
|---------|-------------|---------------|------------|
| **Modulation Target** | Final depth map | Final depth map | DPT features |
| **Modulation Timing** | After all processing | After all processing | Before Mamba |
| **Modulation Type** | Scalar scale/shift | Scalar scale/shift | Channel-wise (256×) |
| **Temporal Backend** | Bi-GRU (2 layers) | Mamba2 (1 layer) | Mamba (4 layers) |
| **Trainable Params** | ~132K | ~147K | ~1.03M |
| **Frozen Components** | All except head | All except head | ViT + DPT + conv1 |
| **Training Speed** | Fast | Fast | Slower |
| **Memory Usage** | Low | Low | Higher |
| **Long-range Dependency** | Good (bidirectional) | Better (state-space) | Best (4-layer Mamba) |
| **Modulation Granularity** | Coarse (1 scale/shift) | Coarse (1 scale/shift) | Fine (256 gamma/beta) |

### When to Use Which?

**Use Gear5 (GRU)** when:
- You want proven, stable temporal modeling
- Minimal parameters (~132K) with bidirectional context
- Standard training setup without special dependencies

**Use Gear5 (Mamba)** when:
- You want efficient state-space temporal modeling
- Better long-range dependency capture than GRU
- Similar parameter count (~147K) but better scaling

**Use Gear5 FiLM** when:
- You want fine-grained channel-wise modulation
- You can afford more trainable parameters (~1.03M)
- You want modulation integrated early in pipeline
- Best temporal modeling with 4-layer Mamba

---

## Loss Functions

All Gear5 variants support two loss types with **NaN-safe** implementations.

### Log L1 Loss (Standard)

```python
def log_l1_loss(pred, target, valid_mask):
    """
    Log L1 loss in inverse depth space (100/m)

    Formula: L = |log(pred_inv) - log(target_inv)|
    where pred_inv = 100/pred, target_inv = 100/target

    NaN-safe: Clamps values to [epsilon, +inf] before log
    """
    epsilon = 1e-8

    # Clamp to positive values BEFORE log (critical for NaN prevention)
    pred = torch.clamp(pred, min=epsilon)
    target = torch.clamp(target, min=epsilon)

    pred_inv = 100.0 / (pred + epsilon)
    target_inv = 100.0 / (target + epsilon)

    loss = torch.abs(
        torch.log(pred_inv + epsilon) -
        torch.log(target_inv + epsilon)
    )

    return loss[valid_mask].mean()
```

**Usage**:
```bash
python train_gear5.py --config-path configs/gear5 +loss_type=log_l1
```

### Importance-Weighted Loss

```python
def importance_weighted_loss(pred, target, valid_mask, importance_map):
    """
    Importance-weighted Log L1 loss

    Formula: L_weighted = L × (1 + fg_ratio × importance)

    Where:
        - L: Standard Log L1 loss per pixel
        - importance: Spatial map from CLS attention [0, 1]
        - fg_ratio: Fraction of high-attention pixels
    """
    # Compute base loss (NaN-safe)
    epsilon = 1e-8
    pred = torch.clamp(pred, min=epsilon)
    target = torch.clamp(target, min=epsilon)

    pred_inv = 100.0 / (pred + epsilon)
    target_inv = 100.0 / (target + epsilon)
    loss = torch.abs(torch.log(pred_inv + epsilon) - torch.log(target_inv + epsilon))

    # Resize importance map to depth resolution
    importance_resized = F.interpolate(
        importance_map.view(B*T, 1, patch_h, patch_w),
        size=(H, W), mode='bilinear', align_corners=True
    )

    # Compute foreground ratio
    importance_threshold = importance_resized.mean()
    fg_ratio = (importance_resized > importance_threshold).float().mean()

    # Apply importance weighting
    weighted_loss = loss * (1.0 + fg_ratio * importance_resized)

    return weighted_loss[valid_mask].mean()
```

**Usage**:
```bash
python train_gear5.py --config-path configs/gear5 +loss_type=importance
```

**Effect**: Higher loss weights on semantically important regions (high CLS attention), lower weights on background.

---

## Valid Mask Criteria

**CRITICAL**: All Gear5 variants follow consistent valid mask criteria:

### Training & Validation
```python
# Valid mask: GT > 0 (all valid pixels, no distance threshold)
valid_mask = (gt_depth_inverse > 0) & (pred_depth_inverse > 0)
```

**No 70m threshold** in training/validation to use all available GT data.

### Testing
```python
# Valid mask: 70m threshold (canonical space evaluation)
MIN_INVERSE_DEPTH = 100.0 / 70.0  # 1.4286 (inverse of 70m)
valid_mask = (gt_depth_inverse >= MIN_INVERSE_DEPTH) & (pred_depth_inverse > 0)
```

**Why 70m threshold in test?**
- Canonical space (fx=500 at 518×518) evaluation standard
- Ensures fair comparison across datasets
- Filters out unreliable far-depth regions

---

## Training & Testing

### Training

#### Gear5 (GRU) - Default

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=1 python train_gear5.py \
  --config-path configs/gear5 \
  training.iterations=60001 \
  dataset.data_root=/path/to/datasets \
  +loss_type=log_l1

# Multi-GPU (DDP)
./train_gear5_ddp.sh \
  --config-path configs/gear5 \
  --results-dir train_results/results_X/gear_5_gru/large/ \
  --batch-size 20 \
  --loss importance
```

#### Gear5 (Mamba) - With Mamba2 Backend

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=1 python train_gear5.py \
  --config-path configs/gear5 \
  training.iterations=60001 \
  dataset.data_root=/path/to/datasets \
  +loss_type=log_l1 \
  +use_mamba_predictor=true

# Multi-GPU (DDP) - Docker
./train_gear5_ddp.sh \
  --config-path configs/gear5 \
  --results-dir train_results/results_X/gear_5_mamba/large/ \
  --batch-size 20 \
  --loss importance \
  --mamba  # Enable Mamba2 backend
```

#### Gear5 FiLM

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=1 python train_gear5_film.py \
  --config-path configs/gear5_film \
  training.iterations=60001 \
  dataset.data_root=/path/to/datasets \
  +loss_type=log_l1

# Multi-GPU (DDP)
./train_gear5_film_ddp.sh \
  --config-path configs/gear5_film \
  --results-dir train_results/results_X/gear_5_film/large/ \
  --batch-size 20 \
  --loss importance
```

### Testing

#### Gear5 (GRU/Mamba)

```bash
CUDA_VISIBLE_DEVICES=1 python test_gear5.py \
  --config-path configs/gear5 \
  --checkpoint train_results/results_X/gear_5/phase_1/checkpoint_step60000.pth \
  --results-dir test_results/gear5_results \
  --gpu 1
```

#### Gear5 FiLM

```bash
CUDA_VISIBLE_DEVICES=1 python test_gear5_film.py \
  --config-path configs/gear5_film \
  --checkpoint train_results/results_X/gear_5_film/phase_1/checkpoint_step60000.pth \
  --results-dir test_results/gear5_film_results \
  --gpu 1
```

**Output**:
- Depth predictions (npy/png)
- Visualization grids (input, prediction, GT, valid masks)
- Metrics JSON (MAE, RMSE, AbsRel, δ1/δ2/δ3, TAE)
- Scale/shift visualizations (GRU/Mamba variants)
- Gamma/beta visualizations (FiLM variant)
- Importance maps

### Docker Commands

**Build**:
```bash
./run_docker.sh build
```

**Training**:
```bash
# Gear5 (GRU)
./run_docker.sh train_gear5_ddp --loss importance --batch-size 20 --gpu 0,1

# Gear5 (Mamba)
./run_docker.sh train_gear5_ddp --loss importance --batch-size 20 --mamba --gpu 0,1

# Gear5 FiLM
./run_docker.sh train_gear5_film_ddp --loss log_l1 --batch-size 20 --gpu 0,1
```

**Testing**:
```bash
./run_docker.sh test_gear5 --gpu 1
./run_docker.sh test_gear5_film --gpu 1
```

---

## Configuration

### Gear5 Config (`configs/gear5/config.yaml`)

```yaml
# General settings
config_dir: null
inference: false
load: configs/flashdepth-l/iter_10001.pth  # FlashDepth-L weights

# Canonical space
canonical_focal_length: 500.0  # For 518×518 resolution
use_canonical_space: true

# Loss function ('log_l1' or 'importance')
loss_type: "log_l1"

# Temporal predictor backend
use_mamba_predictor: false  # false = GRU (default), true = Mamba2

# Dataset
dataset:
  data_root: null
  resolution: 'base'  # 518×518
  video_length: 5
  train_datasets: [mvs-synth, dynamicreplica, tartanair, pointodyssey, spring]
  val_datasets: [sintel, waymo_seg]

# Training
training:
  batch_size: 20  # Per GPU
  workers: 8
  iterations: 60001
  save_freq: 5000
  val_freq: 1000
  log_freq: 100
  wandb: false

  # Learning rate
  gear5_lr: 1.0e-4     # TemporalScalePredictor
  weight_decay: 1.0e-6

# Model
model:
  vit_size: "vitl"
  patch_size: 14
  attn_class: "MemEffAttention"

  # Attention layers for CLS extraction
  target_blocks: [11, 23]      # ViT-L
  target_blocks_s: [5, 11]     # ViT-S

  # Mamba (frozen in Gear5)
  use_mamba: true
  mamba_type: "add"
  num_mamba_layers: 4
  mamba_in_dpt_layer: [1]
  mamba_d_conv: 4
  mamba_d_state: 256

# Evaluation
eval:
  compile: false
  metrics: true
  save_grid: true
  test_datasets: [sintel]
  save_vis_map: true
```

### Gear5 FiLM Config (`configs/gear5_film/config.yaml`)

```yaml
# (Similar structure to Gear5)

# Loss function
loss_type: "log_l1"

# Training
training:
  batch_size: 20
  workers: 8
  iterations: 60001

  # Learning rates (3 separate rates)
  film_lr: 1.0e-4      # FiLM modules
  mamba_lr: 1.0e-5     # Mamba (trainable)
  output_lr: 1.0e-5    # output_conv2
  weight_decay: 1.0e-6

# Model
model:
  use_mamba: true  # Trainable in FiLM variant
  # ... (same Mamba config)
```

---

## Recent Updates

### NaN Loss Fix (2025-11-16)

**Problem**: Occasional `loss=nan` during training due to:
1. `log(negative_value)` or `log(0)` when shift is negative
2. Invalid GT pixels (0 or negative) not filtered before log

**Solution**: Added `torch.clamp(min=epsilon)` before all log operations:

```python
# train_gear5.py & train_gear5_film.py
epsilon = 1e-8
pred_depth_flat = torch.clamp(pred_depth_flat, min=epsilon)
gt_depth_flat = torch.clamp(gt_depth_flat, min=epsilon)

loss = torch.abs(
    torch.log(pred_depth_flat + epsilon) -
    torch.log(gt_depth_flat + epsilon)
)
```

```python
# utils/gear_losses.py (LogL1Loss)
epsilon = 1e-8
pred_valid = torch.clamp(pred_valid, min=epsilon)
gt_valid = torch.clamp(gt_valid, min=epsilon)

loss = F.l1_loss(
    torch.log(pred_valid + epsilon),
    torch.log(gt_valid + epsilon),
    reduction='mean'
)
```

**Files Modified**:
- `train_gear5.py`: Lines 1073-1083
- `train_gear5_film.py`: Lines 1013-1022
- `utils/gear_losses.py`: Lines 48-69

**Impact**: All Gear variants (2, 3, 4, 5, 5_film) now NaN-safe.

### Valid Mask Standardization (2025-11-16)

**Changes**:
- Train/Validation: `GT > 0` (no 70m threshold)
- Test: `GT >= 100/70` (70m threshold in canonical space)
- Consistent across all Gear variants

**Files Modified**:
- `train_gear2.py`, `train_gear3.py`, `train_gear4.py`
- `train_gear5.py`, `train_gear5_film.py`

---

## Summary

**Gear5 (GRU)**: Proven, stable GRU-based temporal scale/shift prediction (~132K params)

**Gear5 (Mamba)**: Efficient Mamba2-based temporal modeling (~147K params) with better long-range dependencies

**Gear5 FiLM**: Fine-grained channel-wise modulation with trainable Mamba (~1.03M params)

All variants:
- Support canonical space normalization (fx=500 @ 518×518)
- Generate importance maps for loss weighting
- Support `log_l1` and `importance` loss types
- Use multi-layer CLS tokens [11, 23] from ViT-L
- **NaN-safe** loss computation with clamping
- Consistent valid mask criteria (train/val: GT>0, test: 70m threshold)

Choose based on:
- **Parameter budget**: GRU/Mamba (~132-147K) vs FiLM (~1.03M)
- **Temporal modeling**: GRU (bidirectional) vs Mamba2 (state-space) vs 4-layer Mamba
- **Modulation granularity**: Scalar (GRU/Mamba) vs Channel-wise (FiLM)
- **Training time**: Fast (GRU/Mamba) vs Slower (FiLM)
