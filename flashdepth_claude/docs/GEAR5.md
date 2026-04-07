# GEAR5: Metric Depth Enhancement System

This document covers the Gear5 architecture for enhancing FlashDepth with metric depth capabilities using **Mamba2-based Temporal Scale Predictor (TSP)**.

---

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [TSP (Temporal Scale Predictor) Deep Dive](#tsp-temporal-scale-predictor-deep-dive)
- [MambaBlock Architecture](#mambablock-architecture)
- [Dimension Flow](#dimension-flow)
- [Loss Functions](#loss-functions)
- [Training & Testing](#training--testing)
- [Configuration](#configuration)

---

## Architecture Overview

Gear5 extends FlashDepth with metric depth prediction by:
1. Extracting semantic features from multi-layer CLS tokens
2. Generating importance maps from attention weights for loss weighting
3. Using TSP (Temporal Scale Predictor) to predict per-frame scale/shift for metric conversion

```
Video Input [B, T, 3, H, W]
    ↓
 ViT Encoder (Frozen)
    ↓
CLS Tokens [Layers 11, 23] → ImportanceMapGenerator → Importance Map
    ↓                              ↓
TemporalScalePredictor          Loss Weighting
 (Mamba2-based TSP)
    ↓
Scale [B, T, 1, 1, 1], Shift [B, T, 1, 1, 1]
    ↓
Relative Depth × Scale + Shift = Metric Depth
```

**Key Components:**
- **Multi-layer CLS Token Extraction**: Layers [11, 23] for ViT-L, [5, 11] for ViT-S
- **ImportanceMapGenerator**: CLS-to-patch attention → importance map for loss weighting
- **TSP (Temporal Scale Predictor)**: Mamba2-based temporal modeling for scale/shift prediction
- **Canonical Space**: All training/inference uses canonical focal length (500.0 for 518×518)

---

## TSP (Temporal Scale Predictor) Deep Dive

### What is TSP?

TSP는 FlashDepth의 상대적 깊이(relative depth)를 절대적 메트릭 깊이(metric depth)로 변환하기 위한 **프레임별 scale과 shift를 예측**하는 모듈입니다.

```python
metric_depth = scale × relative_depth + shift
```

### TSP vs FlashDepth의 MambaModel 차이점

**중요**: TSP는 FlashDepth의 기존 `MambaModel`과 **완전히 별개**입니다!

| 구분 | FlashDepth MambaModel | Gear5 TSP |
|------|----------------------|-----------|
| **역할** | DPT feature의 temporal modeling | Scale/Shift prediction |
| **입력** | DPT path features [B, 256, H, W] | CLS tokens [B, T, 1024] |
| **출력** | Modulated DPT features | Scale [B,T], Shift [B,T] |
| **위치** | DPT 파이프라인 내부 | DPT 파이프라인 외부 (별도) |
| **학습** | **Frozen** (동결) | **Trainable** (학습) |
| **MambaBlock 수** | 4 layers | 1 layer |

```
┌─────────────────────────────────────────────────────────────────┐
│                     FlashDepth Pipeline                         │
│  ┌──────────┐    ┌─────────────────────────────────────────┐   │
│  │   ViT    │───▶│  DPT with MambaModel (FROZEN)           │   │
│  │ Encoder  │    │  - 4 MambaBlocks at path_3              │   │
│  │ (Frozen) │    │  - Temporal feature modulation          │   │
│  └──────────┘    └─────────────────────────────────────────┘   │
│       │                         │                               │
│       │                         ▼                               │
│       │                  Relative Depth                         │
│       │                         │                               │
│       ▼                         │                               │
│  CLS Tokens                     │                               │
│  [B, T, 1024]                   │                               │
│       │                         │                               │
└───────┼─────────────────────────┼───────────────────────────────┘
        │                         │
        ▼                         │
┌───────────────────────┐         │
│  TSP (TRAINABLE)      │         │
│  - 1 MambaBlock       │         │
│  - Scale/Shift pred   │         │
└───────────────────────┘         │
        │                         │
        │ Scale, Shift            │
        │ [B, T]                  │
        ▼                         ▼
┌─────────────────────────────────────────────────────────────────┐
│     Metric Depth = Scale × Relative Depth + Shift               │
└─────────────────────────────────────────────────────────────────┘
```

### TSP의 MambaBlock 사용

TSP는 `flashdepth/mamba.py`의 `MambaBlock` 클래스를 **그대로 재사용**합니다:

```python
# flashdepth/gear5_modules.py - TemporalScalePredictor
from .mamba import MambaBlock

self.temporal_mamba = MambaBlock(
    d_model=feature_dim,  # 256
    layer_idx=0,          # Single layer
    expand=2,
    d_state=64,
    d_conv=4,
    headdim=64,
    use_hydra=False
)
```

**핵심 포인트:**
- TSP가 사용하는 MambaBlock은 FlashDepth 원본과 **구조적으로 동일**
- 다만 **새로운 인스턴스**로 초기화되어 TSP 전용으로 학습됨
- FlashDepth의 기존 MambaModel은 동결(frozen)된 상태로 유지

---

## MambaBlock Architecture

### MambaBlock의 출처와 구조

`MambaBlock`은 FlashDepth 저자가 `mamba_ssm` 라이브러리의 표준 패턴을 따라 구현한 것입니다:

```python
class MambaBlock(nn.Module):
    def __init__(self, d_model, layer_idx, expand, d_state=64, d_conv=4, headdim=64, use_hydra=False):
        super().__init__()
        from mamba_ssm import Mamba2

        # 1. Pre-normalization
        self.norm1 = nn.LayerNorm(d_model)

        # 2. Core Mamba2 (from mamba_ssm library)
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            layer_idx=layer_idx,
            headdim=headdim
        )

        # 3. MLP block (Transformer-style)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, inference_params=None):
        # Transformer-style block with residual connections

        # Sub-block 1: Mamba2 with residual
        residual = x
        x = self.norm1(x)
        x = self.mamba(x, inference_params=inference_params)
        x = residual + x

        # Sub-block 2: MLP with residual
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x

        return x
```

### MambaBlock 구조 분석

```
MambaBlock = Transformer 스타일 블록

┌────────────────────────────────────────────────────────────────┐
│                        MambaBlock                              │
├────────────────────────────────────────────────────────────────┤
│  Input [B, T, d_model]                                         │
│       │                                                        │
│       ├──────────────────┐                                     │
│       │                  │                                     │
│       ▼                  │                                     │
│  LayerNorm (norm1)       │ (residual)                          │
│       │                  │                                     │
│       ▼                  │                                     │
│  ┌──────────────┐        │                                     │
│  │   Mamba2     │        │                                     │
│  │ (mamba_ssm)  │        │                                     │
│  │  ~432K params│        │                                     │
│  └──────────────┘        │                                     │
│       │                  │                                     │
│       ▼                  │                                     │
│       + ◄────────────────┘                                     │
│       │                                                        │
│       ├──────────────────┐                                     │
│       │                  │                                     │
│       ▼                  │                                     │
│  LayerNorm (norm2)       │ (residual)                          │
│       │                  │                                     │
│       ▼                  │                                     │
│  ┌──────────────┐        │                                     │
│  │     MLP      │        │                                     │
│  │ (4x expand)  │        │                                     │
│  │  ~526K params│        │                                     │
│  └──────────────┘        │                                     │
│       │                  │                                     │
│       ▼                  │                                     │
│       + ◄────────────────┘                                     │
│       │                                                        │
│       ▼                                                        │
│  Output [B, T, d_model]                                        │
└────────────────────────────────────────────────────────────────┘
```

### 파라미터 분석 (d_model=256, expand=2)

| 컴포넌트 | 출처 | 파라미터 | 비율 |
|---------|------|---------|------|
| Mamba2 core | mamba_ssm 라이브러리 | 431,768 | 45.1% |
| LayerNorm (×2) | FlashDepth 추가 | 1,024 | 0.1% |
| MLP (FFN) | FlashDepth 추가 | 525,568 | 54.8% |
| **Total** | | **958,360** | 100% |

**이 구조는 mamba_ssm 라이브러리의 표준 패턴입니다:**
- Mamba 논문에서 "H3 블록과 MLP 블록을 결합"하는 구조 권장
- `mamba_ssm.modules.block.Block` 클래스에서도 동일한 패턴 제공
- FlashDepth는 공식 Block 대신 자체 구현했지만 구조는 동일

### MambaBlock 사용 위치 비교

FlashDepth와 gear5 모두 **동일한 MambaBlock 클래스**를 사용하며, **동일한 위치 전략**을 따릅니다:

```
DPT Layer 구조와 Mamba 삽입 위치:

┌─────────────────────────────────────────────────────────────────┐
│                          DPT Head                               │
├─────────────────────────────────────────────────────────────────┤
│  encoder → layer_4_rn                                           │
│               ↓                                                 │
│           path_4 (refinenet4) ─── [0] Mamba 삽입 가능           │
│               ↓                                                 │
│           path_3 (refinenet3) ─── [1] Small/Hybrid 삽입 위치    │
│               ↓                                                 │
│           path_2 (refinenet2) ─── [2] Mamba 삽입 가능           │
│               ↓                                                 │
│           path_1 (refinenet1) ─── [3] Large 삽입 위치           │
│               ↓                                                 │
│           output_conv                                           │
│               ↓                                                 │
│           depth output                                          │
└─────────────────────────────────────────────────────────────────┘
```

**모델별 Mamba 배치 설정 (원본 FlashDepth = gear5):**

| Config | `mamba_in_dpt_layer` | 삽입 위치 | Feature 해상도 | d_model |
|--------|---------------------|----------|---------------|---------|
| **Large (ViT-L)** | `[3]` | path_1 이후 | 148×148 | 256 |
| **Small (ViT-S)** | `[1]` | path_3 이후 | 74×74 | 64 |
| **Hybrid** | `[1]` | path_3 이후 | 74×74 | 64 |

**왜 Large와 Small/Hybrid가 다른 위치를 사용하나?**
- **Large (path_1)**: 고해상도 feature에서 temporal modeling → 더 세밀한 시간적 일관성
- **Small/Hybrid (path_3)**: 저해상도 feature에서 temporal modeling → 계산 효율성

**Mamba 입력 차원:**
```
Mamba 입력: [B, L, d_model]
├─ d_model: 256 (ViT-L) 또는 64 (ViT-S) → config에 따라 자동 설정
└─ L = h × w: 해상도에 따라 동적으로 변함 (Mamba2가 가변 길이 지원)

예시 (518×518 입력, patch_size=14 → 37×37 patches):
- path_1 (Large): L = 148 × 148 = 21,904
- path_3 (Small/Hybrid): L = 74 × 74 = 5,476
```

**결론: "MambaBlock을 어디에, 몇 개 사용할지"의 의미:**
- **어디에**: `mamba_in_dpt_layer` 설정으로 DPT 파이프라인의 삽입 위치 결정
- **몇 개**: `num_mamba_layers` 설정으로 연속된 MambaBlock 개수 결정 (기본: 4)
- MambaBlock 클래스 자체는 변경 없이 그대로 사용
- **시퀀스 길이(L)는 동적**이므로 Mamba 코드 수정 불필요

---

## Dimension Flow

### TSP 차원 흐름 (TemporalScalePredictor)

```
┌─────────────────────────────────────────────────────────────────┐
│              TSP (Temporal Scale Predictor) Flow                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CLS Token (from ViT Layer 23)                                  │
│  Shape: [B, T, 1024]                                            │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Feature Extractor                                       │   │
│  │  Linear(1024 → 256) + ReLU                               │   │
│  │  Params: 1024×256 + 256 = 262,400                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│  Features: [B, T, 256]                                          │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  MambaBlock (temporal modeling)                          │   │
│  │  d_model=256, expand=2, d_state=64                       │   │
│  │                                                          │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │ norm1: LayerNorm(256)           →      512 params│    │   │
│  │  │ mamba: Mamba2(256, expand=2)    → ~431,768 params│    │   │
│  │  │ norm2: LayerNorm(256)           →      512 params│    │   │
│  │  │ mlp: 256→1024→256               → ~525,568 params│    │   │
│  │  │ ─────────────────────────────────────────────────│    │   │
│  │  │ Total MambaBlock:               → ~958,360 params│    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│  Mamba Output: [B, T, 256]                                      │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Projection Layer                                        │   │
│  │  Linear(256 → 128)                                       │   │
│  │  Params: 256×128 + 128 = 32,896                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│  Hidden States: [B, T, 128]                                     │
│       │                                                         │
│       ├────────────────────────┬───────────────────────────┐   │
│       ▼                        ▼                           │   │
│  ┌──────────────┐      ┌──────────────┐                    │   │
│  │ Scale Head   │      │ Shift Head   │                    │   │
│  │ Linear(128→1)│      │ Linear(128→1)│                    │   │
│  │ 129 params   │      │ 129 params   │                    │   │
│  └──────────────┘      └──────────────┘                    │   │
│       │                        │                                │
│       ▼                        ▼                                │
│  Softplus()               Identity()                            │
│       │                        │                                │
│       ▼                        ▼                                │
│  Scale: [B, T]            Shift: [B, T]                         │
│  (positive)               (any value)                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

TSP 총 파라미터:
├─ Feature Extractor:  262,400
├─ MambaBlock:         958,360
├─ Projection:          32,896
├─ Scale Head:             129
└─ Shift Head:             129
─────────────────────────────────
Total:              ~1,253,914 params (~1.25M)
```

### 전체 Gear5 파이프라인 차원 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│                    Complete Gear5 Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Video                                                    │
│  Shape: [B, T, 3, 518, 518]                                     │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  ViT Encoder (DINOv2 ViT-L, FROZEN)                      │   │
│  │  Patch size: 14×14 → 37×37 patches                       │   │
│  │  Params: ~304M (frozen)                                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ├─────────────────────────────────────────────┐          │
│       │                                             │          │
│       ▼                                             ▼          │
│  Patch Tokens                              CLS Tokens          │
│  [B×T, 1369, 1024]                         [B, T, 1024]        │
│       │                                             │          │
│       ▼                                             │          │
│  ┌─────────────────────────────────┐               │          │
│  │  DPT Head (FROZEN)              │               │          │
│  │  with MambaModel at path_3      │               │          │
│  │  (4 × MambaBlock = ~3.83M)      │               │          │
│  └─────────────────────────────────┘               │          │
│       │                                             │          │
│       ▼                                             │          │
│  Relative Depth                                     │          │
│  [B, T, 1, 518, 518]                               │          │
│  (normalized 0~1)                                   │          │
│       │                                             │          │
│       │                                             ▼          │
│       │                           ┌─────────────────────────┐  │
│       │                           │  TSP (TRAINABLE)        │  │
│       │                           │  1 × MambaBlock         │  │
│       │                           │  ~1.25M params          │  │
│       │                           └─────────────────────────┘  │
│       │                                             │          │
│       │                                  Scale, Shift          │
│       │                                  [B, T]    [B, T]      │
│       │                                             │          │
│       └──────────────────┬──────────────────────────┘          │
│                          │                                      │
│                          ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Metric Depth Conversion                                 │   │
│  │  metric_depth = scale × relative_depth + shift           │   │
│  │  scale: [B,T,1,1,1], shift: [B,T,1,1,1]                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          │                                      │
│                          ▼                                      │
│  Metric Depth                                                   │
│  [B, T, 1, 518, 518]                                           │
│  (meters, absolute scale)                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

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
```

**Effect**: Higher loss weights on semantically important regions (high CLS attention), lower weights on background.

---

## Valid Mask Criteria

### Training & Validation
```python
# Valid mask: GT > 0 (all valid pixels, no distance threshold)
valid_mask = (gt_depth_inverse > 0) & (pred_depth_inverse > 0)
```

### Testing
```python
# Valid mask: 70m threshold (canonical space evaluation)
MIN_INVERSE_DEPTH = 100.0 / 70.0  # 1.4286 (inverse of 70m)
valid_mask = (gt_depth_inverse >= MIN_INVERSE_DEPTH) & (pred_depth_inverse > 0)
```

---

## Training & Testing

### Training

```bash
# Single GPU with Mamba2 backend
CUDA_VISIBLE_DEVICES=1 python train_gear5.py \
  --config-path configs/gear5 \
  training.iterations=60001 \
  dataset.data_root=/path/to/datasets \
  +loss_type=log_l1 \
  +use_mamba_predictor=true

# Multi-GPU (DDP)
./run_docker.sh train_gear5_ddp \
  --mamba \
  --config-variant l \
  --results-dir train_results/results_X/gear_5_mamba/large/ \
  --batch-size 20 \
  --loss importance

# With custom CLS layer selection (single layer - only 4th)
./run_docker.sh train_gear5_ddp \
  --mamba \
  --config-variant hybrid \
  --gear-checkpoint train_results/results_X/gear_5_mamba/small/best.pth \
  --results-dir train_results/results_X/gear_5_mamba/hybrid/ \
  --batch-size 1 \
  --epochs 60001 \
  --cls-layer 4
```

**Key Parameters:**
- `--mamba`: Use Mamba2-based TSP (default: GRU)
- `--cls-layer`: CLS token extraction layers (1-4). Examples:
  - `4` → 4번째 레이어만 사용 (ViT-L: block 23, ViT-S: block 11)
  - `2,4` → 2번째, 4번째 레이어 사용 (기본값)
  - `1,2,3,4` → 모든 4개 레이어 사용
- `training.gear5_lr`: 1.0e-4
- Trainable params: ~1.25M (TSP only, rest frozen)

### CLS Layer Selection

ViT의 4개 intermediate layer 중 어느 레이어에서 CLS token을 추출할지 선택할 수 있습니다:

```
ViT-L: intermediate_layer_idx = [4, 11, 17, 23]
       --cls-layer 1 → block 4
       --cls-layer 2 → block 11 (default)
       --cls-layer 3 → block 17
       --cls-layer 4 → block 23 (default)

ViT-S: intermediate_layer_idx = [2, 5, 8, 11]
       --cls-layer 1 → block 2
       --cls-layer 2 → block 5 (default)
       --cls-layer 3 → block 8
       --cls-layer 4 → block 11 (default)
```

**사용 예시:**
- `--cls-layer 4`: 마지막 레이어만 사용 (가장 고수준 semantic)
- `--cls-layer 2,4`: 중간 + 마지막 레이어 (기본값, 다양한 수준의 feature)
- `--cls-layer 1,2,3,4`: 모든 레이어 사용 (실험용)

### Testing

```bash
CUDA_VISIBLE_DEVICES=1 python test_gear5.py \
  --config-path configs/gear5 \
  --checkpoint train_results/results_X/gear_5/phase_1/checkpoint_step60000.pth \
  --results-dir test_results/gear5_results \
  --gpu 1
```

**Output:**
- Depth predictions (npy/png)
- Visualization grids (input, prediction, GT, valid masks)
- Metrics JSON (MAE, RMSE, AbsRel, δ1/δ2/δ3, TAE)
- Scale/shift visualizations
- Importance maps

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
use_mamba_predictor: true  # Mamba2-based TSP

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
  gear5_lr: 1.0e-4     # TSP learning rate
  weight_decay: 1.0e-6

# Model
model:
  vit_size: "vitl"
  patch_size: 14
  attn_class: "MemEffAttention"

  # Attention layers for CLS extraction
  target_blocks: [11, 23]      # ViT-L
  target_blocks_s: [5, 11]     # ViT-S

  # Mamba (frozen in Gear5 - original FlashDepth temporal modules)
  use_mamba: true
  mamba_type: "add"
  num_mamba_layers: 4
  mamba_in_dpt_layer: [3]  # ViT-L uses path_1 (layer 3), ViT-S/Hybrid uses [1] (path_3)
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

---

## Parameter Summary

| Component | Parameters | Status |
|-----------|------------|--------|
| ViT-L Encoder | ~304M | Frozen |
| DPT Head | ~9.4M | Frozen |
| DPT MambaModel (4 layers) | ~3.83M | Frozen |
| **TSP (Trainable)** | **~1.25M** | **Trainable** |
| - Feature Extractor | 262K | |
| - MambaBlock | 958K | |
| - Projection + Heads | 33K | |

**Total Trainable: ~1.25M parameters** (전체 모델의 ~0.4%)

---

## Summary

**Gear5 핵심 아키텍처:**

1. **FlashDepth 파이프라인 전체 동결** (ViT + DPT + MambaModel)
2. **TSP만 학습** (~1.25M params)
3. TSP는 CLS token → Scale/Shift 예측
4. `metric_depth = scale × relative_depth + shift`

**MambaBlock 사용:**
- FlashDepth의 MambaModel: DPT path_3에서 4개 MambaBlock (동결)
- TSP: 별도 1개 MambaBlock (학습)
- 두 MambaBlock은 **같은 클래스**이지만 **다른 용도**

**Canonical Space:**
- 학습/추론 모두 fx=500 (518×518) 기준
- 테스트 시 70m threshold 적용
