# Gear5 Bankai: Unified Mamba for Metric Depth Estimation

## 1. Overview

Gear5 Bankai is an advanced metric depth estimation system that unifies temporal processing and metric scale/shift prediction into a single Mamba2-based architecture.

### Key Features

1. **Unified Architecture**: Single Mamba2 replaces both F-Mamba (FlashDepth temporal) and T-Mamba (Gear5 scale predictor)
2. **Weight Compatibility**: UnifiedMamba structure matches FlashDepth MambaModel for pretrained weight loading
3. **Memory Efficiency**: Original Mamba removed after weight copy (~4.3M params freed)
4. **Temporal Consistency**: Enhanced TAE through unified temporal processing
5. **TGM Loss**: Temporal Gradient Matching loss from Video Depth Anything
6. **Head-First Tuning**: Two-phase training strategy for stable optimization

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
        │  [Hybrid only] CLS Fusion (FiLM/Bilinear)  │
        │  FiLM: γ(CLS-L) × CLS-S + β(CLS-L)         │
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
│  - Structure matches FlashDepth MambaModel (weight compatible)  │
│  - Per-frame processing + hidden state propagation              │
│  - Causal: CLS receives spatial info, but does NOT affect it   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
            ┌─────────────────┴─────────────────┐
            ↓                                   ↓
    Spatial [h×w, C]                     CLS [1, C]
            ↓                                   ↓
    final_layer (from FlashDepth)        MLP Head (new)
            ↓                                   ↓
    Unflatten + Upsample                 Scale (Softplus)
    (Large: ×10, Small/Hybrid: ×20)      Shift
            ↓
    + original_path_1 (residual)
            ↓
    output_conv (Head-First Tuning)
            ↓
      Relative Depth
            ↓
    Metric Depth = Scale × Relative + Shift
```

### 2.2 Mamba2 Causal Property

**Important**: Mamba2 is a **causal** model (SSM part).

```
Token order: [spatial_0, spatial_1, ..., spatial_N, CLS]
                                                    ↑
                                              CLS at END

Information flow:
  spatial tokens → CLS: ✓ (CLS receives full spatial context)
  CLS → spatial tokens: ✗ (Causal - CLS does NOT affect spatial output)
```

This design ensures:
- CLS aggregates all spatial information for scale/shift prediction
- Spatial output quality is preserved (not contaminated by CLS)

### 2.3 UnifiedMamba Structure (FlashDepth Compatible)

```python
class UnifiedMamba(nn.Module):
    def __init__(self, dpt_dim=256, cls_embed_dim=1024, num_layers=4,
                 d_state=256, d_conv=4, ...):

        # NEW: CLS projection (random init)
        self.cls_proj = nn.Sequential(
            nn.Linear(cls_embed_dim, dpt_dim),
            nn.ReLU()
        )

        # FROM FLASHDEPTH: Mamba2 blocks (weight loaded)
        # Structure: blocks[0][0..3] - matches FlashDepth MambaModel
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                MambaBlock(d_model=dpt_dim, d_state=d_state, d_conv=d_conv, ...)
                for layer_idx in range(num_layers)
            ])
        ])

        # FROM FLASHDEPTH: final_layer (weight loaded)
        self.final_layer = nn.Sequential(
            nn.GELU(),
            nn.Linear(dpt_dim, dpt_dim)
        )

        # NEW: Scale/Shift heads (random init)
        self.scale_head = nn.Sequential(...)
        self.shift_head = nn.Sequential(...)
```

### 2.4 Mamba2 Hyperparameters

| Parameter | Description | Value |
|-----------|-------------|-------|
| `d_state` | SSM hidden state dimension (temporal memory capacity) | 256 |
| `d_conv` | Local convolution kernel size | 4 |
| `expand` | MLP expansion factor | 2 (ViT-L) / 4 (ViT-S) |
| `headdim` | Attention head dimension | 64 (ViT-L) / 32 (ViT-S) |
| `num_layers` | Number of MambaBlock layers | 4 |

### 2.5 Model Variants

| Variant | DPT Dim | CLS Dim | d_state | Downsample | Mamba Tokens |
|---------|---------|---------|---------|------------|--------------|
| Large | 256 | 1024 | 256 | 0.1 | 196 + 1 |
| Small | 64 | 384 | 256 | 0.05 | 49 + 1 |
| Hybrid | 64 | 384 (fused) | 256 | 0.05 | 49 + 1 |

---

## 3. Weight Loading Strategy

### 3.1 Weight Mapping

| Source (FlashDepth) | Target (UnifiedMamba) | Status |
|---------------------|----------------------|--------|
| `mamba.blocks.0.X.*` | `gear5_metric_head.unified_mamba.blocks.0.X.*` | **Copied** |
| `mamba.final_layer.*` | `gear5_metric_head.unified_mamba.final_layer.*` | **Copied** |
| - | `gear5_metric_head.unified_mamba.cls_proj.*` | Random init |
| - | `gear5_metric_head.unified_mamba.scale_head.*` | Random init |
| - | `gear5_metric_head.unified_mamba.shift_head.*` | Random init |

### 3.2 Parameter Count

```
FlashDepth Original Mamba: ~4.3M params
  └─ blocks.0.[0-3]: ~4.0M (4 MambaBlock layers)
  └─ final_layer: ~0.3M

UnifiedMamba Total: ~4.7M params
  ├─ FROM FlashDepth (~4.3M):
  │   └─ blocks.0.[0-3]: ~4.0M (copied from FlashDepth)
  │   └─ final_layer: ~0.3M (copied from FlashDepth)
  │
  └─ NEW Components (~0.4M):
      └─ cls_proj: ~262K (1024→256 projection)
      └─ scale_head: ~33K (256→128→1)
      └─ shift_head: ~33K (256→128→1)

Memory Savings:
  - Original Mamba REMOVED after weight copy
  - Net change: +0.4M new params (scale/shift heads)
```

### 3.3 Loading Process

```python
# 1. Create BankaiMetricHead with UnifiedMamba
model.gear5_metric_head = BankaiMetricHead(
    dpt_dim=dpt_dim,
    d_state=config.model.mamba_d_state,  # Must match FlashDepth (256)
    d_conv=config.model.mamba_d_conv,    # Must match FlashDepth (4)
    ...
)

# 2. Copy FlashDepth Mamba weights to UnifiedMamba
_copy_mamba_weights_to_unified(model)
# - blocks.0.X.* → unified_mamba.blocks.0.X.*
# - final_layer.* → unified_mamba.final_layer.*

# 3. Remove Original Mamba to free memory
del model.mamba
model.use_mamba = False
```

---

## 4. Training Strategy

### 4.1 Loss Function

```
L_total = L_depth + α × L_TGM

Where:
- L_depth: Log L1 Loss on inverse depth
- L_TGM: Temporal Gradient Matching Loss (Video Depth Anything)
- α: TGM weight (default: 0.3)
```

### 4.2 Head-First Tuning (Two-Phase Training)

| Component | Phase 1 | Phase 2 |
|-----------|---------|---------|
| DINOv2 (encoder) | Frozen | Frozen |
| DPT (decoder) | Frozen | Frozen |
| UnifiedMamba (blocks + final_layer) | **Frozen** | **Trainable** |
| Scale/Shift heads | **Trainable** | **Trainable** |
| output_conv | Frozen | **Trainable** |

**Phase 1** (~5,000 steps):
- Train only scale/shift MLPs (~66K params)
- UnifiedMamba uses FlashDepth pretrained weights (frozen)
- Goal: Initialize scale/shift to reasonable metric range
- CLS receives spatial info through frozen Mamba → meaningful scale/shift learning

**Phase 2** (~35,000 steps):
- Train UnifiedMamba + Scale/Shift + output_conv (~4.6M params)
- Learning rate: 0.1× of Phase 1
- Goal: Joint optimization for temporal consistency + metric accuracy

### 4.3 Phase 1 Expected Log

```
=== BANKAI Phase 1 ===
Frozen:
  - ViT + DPT: 334,983,680
  - UnifiedMamba (frozen in P1): ~4,200,000
  - output_conv: 331,969
Trainable:
  - UnifiedMamba (incl. scale/shift): 0
  - Other (scale/shift heads): ~66,000
  - output_conv: 0
Total frozen: ~339,500,000
Total trainable: ~66,000
```

### 4.4 Phase 2 Expected Log

```
=== BANKAI Phase 2 ===
Frozen:
  - ViT + DPT: 334,983,680
  - Original Mamba: REMOVED (not loaded)
  - output_conv: 0
Trainable:
  - UnifiedMamba (incl. scale/shift): ~4,300,000
  - Other (CLS fusion, etc.): 0
  - output_conv: 331,969
Total frozen: 334,983,680
Total trainable: ~4,600,000
```

---

## 5. CLS Fusion Module (Hybrid Mode)

### 5.1 FiLM Fusion (Recommended)

**Feature-wise Linear Modulation**: Teacher가 Student를 channel-wise로 modulate

```python
class FiLMFusion(nn.Module):
    def __init__(self, student_dim=384, teacher_dim=1024, hidden_dim=512):
        self.film_generator = nn.Sequential(
            nn.Linear(teacher_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, student_dim * 2)  # γ, β
        )

    def forward(self, cls_s, cls_l):
        film_params = self.film_generator(cls_l)
        gamma, beta = film_params.chunk(2, dim=-1)
        return gamma * cls_s + beta
```

### 5.2 Bilinear Fusion (Alternative)

**Low-rank Bilinear Pooling**: 두 feature 간 second-order interaction

```python
class BilinearFusion(nn.Module):
    def __init__(self, student_dim=384, teacher_dim=1024, rank=64):
        self.U = nn.Linear(student_dim, rank, bias=False)
        self.V = nn.Linear(teacher_dim, rank, bias=False)
        self.P = nn.Linear(rank, student_dim)

    def forward(self, cls_s, cls_l):
        z = self.U(cls_s) * self.V(cls_l)  # Hadamard product
        return self.P(z)
```

---

## 6. Usage

### 6.1 Training

```bash
# Phase 1: Train Scale/Shift heads only
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

### 6.2 Testing

```bash
python test_gear5.py \
    --config-path configs/gear5 \
    use_bankai=true \
    load=<checkpoint_path> \
    results_dir=test_results/bankai
```

### 6.3 Configuration Options

```yaml
# configs/gear5/config_l.yaml

# Enable Bankai mode
use_bankai: true

# Training phase (1 or 2)
bankai_phase: 1

# TGM loss weight (0 to disable)
tgm_weight: 0.3

# Architecture settings
bankai_downsample: 0.1  # Large: 0.1, Small: 0.05
bankai_num_mamba_layers: 4

# Mamba hyperparameters (MUST match FlashDepth for weight loading!)
mamba_d_state: 256
mamba_d_conv: 4

# CLS layers (3rd and 4th intermediate)
cls_layers: [3, 4]

# CLS Fusion settings (Hybrid mode only)
cls_fusion_type: "film"
cls_fusion_hidden: 512
```

---

## 7. Implementation Files

### Modified Files

```
flashdepth/gear5_modules.py     # UnifiedMamba, BankaiMetricHead
flashdepth/mamba.py             # MambaBlock, MambaModel (reference)
utils/gear_losses.py            # TGMTemporalLoss, CombinedBankaiLoss
train_gear5.py                  # Bankai mode training, weight copy logic
test_gear5.py                   # Bankai mode testing
configs/gear5/config_l.yaml     # Bankai config options
configs/gear5/config_s.yaml     # Bankai config options
```

### Key Classes

- `UnifiedMamba`: FlashDepth-compatible Mamba with CLS + scale/shift heads
- `BankaiMetricHead`: Full metric head wrapper
- `FiLMFusion` / `BilinearFusion`: CLS fusion modules (Hybrid mode)
- `TGMTemporalLoss`: Temporal Gradient Matching loss
- `_copy_mamba_weights_to_unified()`: Weight transfer utility

---

## 8. Evaluation

### 8.1 Metrics

| Category | Metric | Target |
|----------|--------|--------|
| Depth Quality | MAE, RMSE | Maintain or improve |
| Accuracy | δ1, δ2, δ3 | Maintain or improve |
| Temporal | **TAE** | **20%+ improvement** |
| Efficiency | FPS | <10% drop |

### 8.2 Datasets

- **Training**: TartanAir, MVS-Synth, DynamicReplica, PointOdyssey, Spring
- **Validation**: Sintel, Waymo_seg
- **Testing**: ETH3D, UrbanSyn, Unreal4K, Bonn

---

## 9. References

- [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) - TGM Loss
- [FlashDepth](https://github.com/xxx/FlashDepth) - Base architecture
- [Mamba2](https://github.com/state-spaces/mamba) - Temporal modeling

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2025-01-26 | v1.2 | UnifiedMamba structure aligned with FlashDepth MambaModel for weight loading; Original Mamba removed for memory efficiency; Added weight copy logic; Clarified Mamba2 causal property |
| 2025-01-22 | v1.1 | CLS Fusion Module added (FiLM primary, Bilinear alternative) |
| 2025-01-20 | v1.0 | Initial implementation |
