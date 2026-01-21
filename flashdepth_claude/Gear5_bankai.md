# Gear5 Bankai: Unified Mamba for Metric Depth Estimation

## 1. Overview

Gear5 Bankai is an advanced metric depth estimation system that unifies temporal processing and metric scale/shift prediction into a single Mamba2-based architecture.

### Key Features

1. **Unified Architecture**: Single Mamba2 replaces both F-Mamba (FlashDepth temporal) and T-Mamba (Gear5 scale predictor)
2. **Temporal Consistency**: Enhanced TAE (Temporal Alignment Error) through unified temporal processing
3. **Relative Depth Quality**: Preserves DAv2's DINOv2-DPT structure strength
4. **TGM Loss**: Temporal Gradient Matching loss from Video Depth Anything for improved temporal consistency
5. **Head-First Tuning**: Two-phase training strategy for stable optimization

---

## 2. Architecture

### 2.1 Overall Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        DINOv2 (Frozen)                          │
│  ViT-L: CLS tokens [17, 23] → avg → Fused CLS-L [1, 1024]      │
│  ViT-S: CLS tokens [8, 11]  → avg → Fused CLS-S [1, 384]       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────┐
        │  [Hybrid only] CrossAttn(Q:CLS-S, KV:CLS-L) │
        └─────────────────────────────────────────────┘
                              ↓
              Fused CLS → Linear + ReLU → [1, C]

┌─────────────────────────────────────────────────────────────────┐
│                         DPT (Frozen)                            │
│  path_1 (148×148, C) → Downsample → Flatten → [h×w, C]         │
│                                                                 │
│  Downsample ratio:                                              │
│    - Large: ×0.1 (148→14, 196 tokens)                          │
│    - Small/Hybrid: ×0.05 (148→7, 49 tokens)                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    Concat → [h×w + 1, C]
                    (Spatial tokens + CLS at END)
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Unified Mamba (Trainable)                    │
│  - Per-frame processing + hidden state propagation              │
│  - Temporal consistency via hidden state                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
            ┌─────────────────┴─────────────────┐
            ↓                                   ↓
    Spatial [h×w, C]                     CLS [1, C]
            ↓                                   ↓
    Unflatten + Upsample                 MLP Head
    (Large: ×10, Small/Hybrid: ×20)          ↓
            ↓                           Scale (Softplus)
    + original_path_1                   Shift (Clamped)
            ↓
    output_conv (Head-First Tuning)
            ↓
      Relative Depth
            ↓
    Metric Depth = Scale × Relative + Shift
```

### 2.2 Key Components

#### UnifiedMamba

- Processes spatial tokens + CLS token through Mamba2
- CLS at END position to aggregate spatial information
- Hidden state propagation for temporal consistency
- Zero-init output projection for training stability

```python
# Key parameters (Large model)
- dpt_dim: 256
- cls_embed_dim: 1024
- num_mamba_layers: 4
- downsample_factor: 0.1
- Parameters: ~1.76M
```

#### BankaiMetricHead

- Wrapper for UnifiedMamba + Importance Map Generator
- Optional CLS CrossAttention for Hybrid mode
- Outputs: spatial_out, scale, shift, importance_map

### 2.3 Model Variants

| Variant | DPT Dim | CLS Dim | Downsample | Mamba Tokens |
|---------|---------|---------|------------|--------------|
| Large | 256 | 1024 | 0.1 | 196 + 1 |
| Small | 64 | 384 | 0.05 | 49 + 1 |
| Hybrid | 64 | 384 (fused) | 0.05 | 49 + 1 |

---

## 3. Training Strategy

### 3.1 Loss Function

```
L_total = L_depth + α × L_TGM

Where:
- L_depth: Log L1 Loss on inverse depth
- L_TGM: Temporal Gradient Matching Loss (Video Depth Anything)
- α: TGM weight (default: 0.3)
```

#### TGM Loss Details

- Multi-scale temporal gradients (stride=1, 2, 4, 8)
- Validity masking for stable regions
- Trimmed MAE for outlier robustness
- Exponential decay for longer temporal distances

### 3.2 Head-First Tuning (Two-Phase Training)

| Component | Phase 1 | Phase 2 |
|-----------|---------|---------|
| DINOv2 (encoder) | Frozen | Frozen |
| DPT (decoder) | Frozen | Frozen |
| UnifiedMamba | **Frozen** | **Trainable** |
| Metric Head (MLP) | **Trainable** | **Trainable** |
| output_conv | Frozen | **Trainable** |

**Phase 1**: ~5,000 steps
- Train only scale/shift MLPs
- UnifiedMamba uses FlashDepth pretrained weights
- Goal: Initialize scale/shift to reasonable range

**Phase 2**: ~35,000 steps
- Train UnifiedMamba + Metric Head + output_conv
- Learning rate: 0.1× of Phase 1
- Goal: Joint optimization for temporal consistency

### 3.3 Pretrained Weights

| Component | Source |
|-----------|--------|
| DINOv2 | FlashDepth pretrained |
| DPT | FlashDepth pretrained |
| UnifiedMamba | FlashDepth Mamba (F-Mamba) |
| Metric Head | Random init |
| output_conv | FlashDepth pretrained |

---

## 4. Usage

### 4.1 Training

```bash
# Phase 1: Train Metric Head only
torchrun --nproc_per_node=2 train_gear5.py \
    --config-path configs/gear5 \
    use_bankai=true \
    bankai_phase=1 \
    tgm_weight=0.3 \
    dataset.data_root=<path_to_data>

# Phase 2: Train UnifiedMamba + output_conv
torchrun --nproc_per_node=2 train_gear5.py \
    --config-path configs/gear5 \
    use_bankai=true \
    bankai_phase=2 \
    tgm_weight=0.3 \
    load=<phase1_checkpoint> \
    dataset.data_root=<path_to_data>
```

### 4.2 Testing

```bash
python test_gear5.py \
    --config-path configs/gear5 \
    use_bankai=true \
    load=<checkpoint_path> \
    results_dir=test_results/bankai
```

### 4.3 Configuration Options

```yaml
# configs/gear5/config_l.yaml (or config_s.yaml)

# Enable Bankai mode
use_bankai: true

# Training phase (1 or 2)
bankai_phase: 1

# TGM loss weight (0 to disable)
tgm_weight: 0.3

# Architecture settings
bankai_downsample: 0.1  # Large: 0.1, Small: 0.05
bankai_num_mamba_layers: 4

# CLS layers (3rd and 4th intermediate)
cls_layers: [3, 4]
```

---

## 5. Evaluation

### 5.1 Metrics

| Category | Metric | Target |
|----------|--------|--------|
| Depth Quality | MAE, RMSE | Maintain or improve |
| Accuracy | δ1, δ2, δ3 | Maintain or improve |
| Temporal | **TAE** | **20%+ improvement** |
| Efficiency | FPS | <10% drop |

### 5.2 Datasets

- **Training**: TartanAir, MVS-Synth, DynamicReplica, PointOdyssey, Spring
- **Validation**: Sintel, Waymo_seg
- **Testing**: ETH3D, UrbanSyn, Unreal4K, Bonn

---

## 6. Implementation Files

### Modified Files

```
flashdepth/gear5_modules.py     # UnifiedMamba, BankaiMetricHead
utils/gear_losses.py            # TGMTemporalLoss, CombinedBankaiLoss
train_gear5.py                  # Bankai mode training
test_gear5.py                   # Bankai mode testing
configs/gear5/config_l.yaml     # Bankai config options
configs/gear5/config_s.yaml     # Bankai config options
```

### Key Classes

- `UnifiedMamba`: Unified temporal processing for spatial + CLS tokens
- `BankaiMetricHead`: Full metric head with UnifiedMamba
- `TGMTemporalLoss`: Temporal Gradient Matching loss
- `CombinedBankaiLoss`: Depth loss + TGM loss

---

## 7. References

- [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) - TGM Loss
- [FlashDepth](https://github.com/xxx/FlashDepth) - Base architecture
- [Mamba2](https://github.com/state-spaces/mamba) - Temporal modeling

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2025-01-20 | v1.0 | Initial implementation |
