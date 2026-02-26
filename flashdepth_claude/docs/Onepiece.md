# Onepiece V2: Unified Global Mamba for Metric Depth Estimation

## Overview

Onepiece V2는 FlashDepth의 relative depth를 metric depth로 변환하기 위한 통합 아키텍처. V1 대비 핵심 변경: **CLS token 제거**, GAP+GStdP 기반 512-dim 입력, FiLM/MetricHead 분리, 3-loss full graph, SceneCutDetector 삭제.

### V1 → V2 핵심 변경

| 항목 | V1 | V2 |
|------|----|----|
| 입력 토큰 | CLS(1024) + GAP(256) = 1280-dim | **GAP(256) + GStdP(256) = 512-dim** |
| Mamba 구조 | 2 layers, d=1280, 46.36M params | **4 layers, d=512, ~10.7M params** |
| Head 구조 | Combined MLP (scale/shift + FiLM) | **Separated**: FiLMGenerator + Conv MetricHead |
| Scene cut | SceneCutDetector (CLS cosine) | **삭제** |
| Loss 체계 | LogL1 + TGM + WFC (3-loss) | **LogL1 + TGM + OFC (3-loss, full graph)** |
| Phase 2 unfreeze | DPT + output_conv | **FiLMGenerator + output_conv (DPT always frozen)** |
| Scale 초기값 | ~1.0 (softplus(0.54)) | **~100 (softplus(100))** — 100x factor 흡수 |
| Shift 범위 | [0, 0.1] | **[0, 1.0]** |
| Depth 변환 | `100 / relative_depth` | **`1 / relative_depth`** (scale이 100 흡수) |

---

## Architecture

### Full Pipeline

```
Input Video [B, T, 3, H, W]
    |
    v
[Frozen] DINOv2 ViT-L Encoder
    |
    v
Intermediate features (4 layers)   ← CLS token 사용하지 않음
    |
    v
[Always Frozen] DPT Head
    |
    v
DPT features [B*T, 256, h, w]
    |
    +-------------------+
    |                   |
    v                   v
GAP [B*T, 256]     GStdP [B*T, 256]     (preserve dpt_features for FiLM)
    |                   |
    +-------+-----------+
            |
    concat [B*T, 512]
            |
            v  (reshape to [B, T, 512])
    [Trainable] UnifiedGlobalMamba (4 layers, d=512)
            |
            v  [B, T, 512]
            |
            v  (reshape to [B*T, 512])
    [Phase 1: Frozen / Phase 2: Trainable] FiLMGenerator
            |
            v
    gamma [B*T, 256], beta [B*T, 256]
            |
            v
    FiLM modulation: modulated = gamma * dpt_features + beta
            |
            +----------------------------+
            |                            |
            v                            v
    [Phase 1: Frozen / Phase 2:     [Trainable] MetricHead (Conv)
     Trainable] Relative Head       Conv(256→64→2) → spatial mean
    (output_conv)                        |
            |                            v
            v                       scale (softplus, init≈100)
    relative_depth [B*T, H, W]     shift (sigmoid, range [0, 1.0])
            |                            |
            +----------------------------+
            |
            v
    depth_from_rel = 1.0 / relative_depth
    metric_depth = scale * depth_from_rel + shift
```

---

## Core Modules

### 1. UnifiedGlobalMamba (`flashdepth/onepiece_modules.py`)

GAP + GStdP concatenated token을 temporal processing하는 Mamba2 모듈.

```python
UnifiedGlobalMamba(
    d_input=512,        # GAP(256) + GStdP(256) for ViT-L
    num_layers=4,       # 4-layer Mamba2 stack
    d_state=64,         # SSM state dimension
    d_conv=4,           # 1D convolution kernel size
    expand=2,           # Mamba expansion factor
    headdim=64          # Head dimension
)
```

**Mamba2 Dimension Constraint**: `d_model * expand / headdim` must be a multiple of 8.

| Variant | GAP dim | GStdP dim | d_input | d_model (Mamba2-valid) | nheads | Projection |
|---------|---------|-----------|---------|------------------------|--------|------------|
| ViT-L | 256 | 256 | 512 | 512 | 16 (=512*2/64) | None |

**Auto-projection**: `_find_valid_mamba_dim(d_input)` finds nearest `d_model >= d_input` satisfying the constraint. When `d_input != d_model`, input/output linear projection layers are automatically added.

**Forward modes**:
- `forward(x)`: Training batch mode. `[B, T, 512]` 입력, parallel scan으로 전체 시퀀스 처리
- `forward_single_frame(x)`: Inference streaming mode. `[B, 1, 512]` 입력, hidden state 유지하며 frame-by-frame 처리

### 2. OnepieceFiLMGenerator (`flashdepth/onepiece_modules.py`)

Mamba output에서 FiLM parameters를 생성하는 경량 MLP.

```python
OnepieceFiLMGenerator(
    mamba_dim=512,      # Mamba output dimension
    dpt_dim=256         # DPT feature dimension (= target FiLM dimension)
)
```

**Architecture**:
```
refined_global [B*T, 512]
    |
    v
Linear(512, 256) → ReLU → Linear(256, 512)
    |
    v
chunk(2) → gamma_raw [256], beta [256]
    |
    v
gamma = 1 + gamma_raw     (residual: identity at init)
beta = beta_raw            (zero at init)
```

**Weight initialization**: Last layer zero-init → `gamma=1, beta=0` (identity transform).

### 3. OnepieceMetricHead (`flashdepth/onepiece_modules.py`)

Modulated spatial features에서 scale/shift를 예측하는 Conv-based head.

```python
OnepieceMetricHead(
    dpt_dim=256,         # Input feature channels
    hidden_dim=64        # Hidden layer channels
)
```

**Architecture**:
```
modulated_features [B*T, 256, h, w]
    |
    v
Conv2d(256, 64, 1) → ReLU → Conv2d(64, 2, 1)
    |
    v
spatial_mean → [B*T, 2]
    |                           |
    v                           v
softplus(raw_scale).clamp(max=1000)   1.0 * sigmoid(raw_shift)
= scale (positive, max 1000)          = shift (range [0, 1.0])
```

**Initialization**:
- **Scale**: `softplus(100.0) ≈ 100.0` — scale이 old 100x factor를 흡수
- **Shift**: `1.0 * sigmoid(-5) ≈ 0.007 ≈ 0`

### 4. FiLM Spatial Modulation

DPT features에 대한 channel-wise affine transform:

```python
# gamma: [B*T, 256, 1, 1], beta: [B*T, 256, 1, 1]
modulated_features = gamma * dpt_features + beta
```

- `gamma = 1 + gamma_raw`: residual 구조, 초기 identity (원본 feature 유지)
- `beta = beta_raw`: 초기 0
- Spatial broadcast: 모든 (h, w) 위치에 같은 gamma, beta 적용 (channel-wise)
- **FiLM 목적**: scale/shift가 depth map에 uniform 적용되는 반면, FiLM은 DPT feature 자체를 modulate하여 output_conv가 더 나은 relative depth를 생성하도록 유도

---

## Loss Functions (`utils/onepiece_losses.py`)

### 3-Loss System (Full Graph)

```
Phase 1: L_total = w1 * L_log_l1 + w2 * L_tgm
Phase 2: L_total = w1 * L_log_l1 + w2 * L_tgm + w3 * L_ofc
```

**Default weights**: `1:1:0.01` (OFC raw L2 스케일이 크므로 0.01로 축소)

### Gradient Flow

| Loss | Target | Gradient Scope | Phase |
|------|--------|---------------|-------|
| **Log L1** | metric_depth (full graph) | Mamba + FiLM + MetricHead + RelativeHead | 1+2 |
| **TGM** | metric_depth (full graph) | Mamba + FiLM + MetricHead + RelativeHead | 1+2 |
| **OFC** | modulated_features | FiLM → Mamba | 2 only |

**Note**: Phase 1에서 FiLMGenerator는 zero-init (identity)이며 frozen이므로, LogL1/TGM gradient가 FiLM의 zero weight를 통과하지 못해 Mamba에 도달하지 않음. Phase 1은 MetricHead range 안정화 목적이며 1500 step으로 짧게 유지.

### 1. Log L1 Loss (Full graph)

Per-frame metric depth 정확도 loss.

```
L_log_l1 = |log(pred_inverse) - log(gt_inverse)|
```

- `pred_inverse = 1.0 / metric_depth`, `gt_inverse = gt_depth` (stored as 1/m)
- Valid mask 적용: GT > 0 & pred > 0 & pred < 1000m
- Full graph: gradient flows to all trainable modules

### 2. TGM Temporal Loss (Full graph)

Temporal gradient matching: frame간 depth 변화가 GT와 일치하도록.

```
L_tgm = |Delta_pred(t, t-1) - Delta_gt(t, t-1)|
```

- Full gradient: Mamba + FiLM + MetricHead + RelativeHead (Phase 2)

### 3. Optical Flow Consistency Loss (OFC, FiLM→Mamba, Phase 2 only)

DPT feature의 temporal consistency를 optical flow 기반으로 강제.

```
1. Sea-RAFT로 optical flow 추정: flow [B, T-1, 2, H, W], confidence [B, T-1, 1, H, W]
2. Flow를 feature resolution으로 resize + scale 조정
3. Grid sample로 feat_{t-1}을 feat_t 위치로 warp
4. Confidence-weighted L2 loss:
   L_feat = mean(confidence * ||feat_t - warp(feat_{t-1})||^2)
```

- Feature downsample 제거 (full resolution 사용)
- Flow confidence는 OFC 내부에서만 사용 (다른 loss에 전달 안 함)

### Loss Shape Convention

| Loss | 기대 Shape | 처리 위치 | 비고 |
|------|-----------|----------|------|
| **Log L1** | 아무 shape (element-wise) | 내부 `mask.bool()` flat indexing | `[B, T, H, W]` 직접 전달 OK |
| **TGM** | `[B, T, H, W]` (4D) | 내부 `B, T, H, W = shape` 파싱 | temporal gradient 계산에 T축 필요 |
| **OFC** | `[B, T, C, h, w]` (5D) | 내부 `B, T, C, h, w = shape` 파싱 | features + images 모두 5D |

`train_onepiece.py`의 `train_step`에서 model output → loss 전달 시:
- `metric_depth`, `gt_depth`, `valid_mask` → `[B, T, H, W]` 그대로 전달 (LogL1, TGM이 처리)

### Depth Valid Range

| Stage | GT Valid | Pred Valid | 비고 |
|-------|---------|-----------|------|
| **Training** | `gt_inverse > 0` | `metric_depth > 0` & `< 1000m` | + `actual_valid_masks` AND |
| **Validation** | `gt > 0` & `< 70m` | `pred > 0` & `< 70m` | test와 동일 기준 |
| **Test (메트릭)** | `gt > 0` & `< 70m` | `pred > 0` & `< 70m` | 미터 공간 기준 |

---

## Flow Estimator (`utils/flow_estimator.py`)

**Sea-RAFT** (https://github.com/princeton-vl/SEA-RAFT) wrapper.

- **Phase 2 훈련 시에만 사용** (OFC loss), Phase 1 및 추론/테스트에서는 불필요
- 완전 frozen (eval mode, `requires_grad=False`)
- 입력: RGB 0-1 normalized → 내부에서 0-255 변환
- 출력: flow `[B, 2, H, W]`, confidence `[B, 1, H, W]`
- 8의 배수로 padding 후 처리, 이후 crop
- `estimate_flow_batch`: T 프레임에 대해 T-1개 flow 쌍 생성

**설치 요구사항**:
```bash
git clone https://github.com/princeton-vl/SEA-RAFT.git third_party/SEA-RAFT/
mkdir -p third_party/SEA-RAFT/models/
# Pretrained weights → third_party/SEA-RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth
```

**Sea-RAFT 없이도 Phase 1 훈련 가능** (OFC는 Phase 2에서만 활성화). 테스트는 항상 가능.

---

## Training (`train_onepiece.py`)

### Two-Phase Training Strategy

#### Phase 1 (Step 0 ~ `auto_transition_step`, default 1500)
**목표**: Metric alignment (scale/shift 수렴)

| Component | Status |
|-----------|--------|
| DINOv2 ViT-L | **Frozen** |
| DPT Head | **Frozen** |
| output_conv (RelativeHead) | **Frozen** |
| OnepieceFiLMGenerator | **Frozen** (zero-init, identity) |
| UnifiedGlobalMamba | **Trainable** (requires_grad=True, but no gradient due to FiLM zero-init barrier) |
| OnepieceMetricHead | **Trainable** |

- Losses: **Log L1 + TGM** (full graph, but effectively MetricHead only due to FiLM zero-init)
- LR: 1e-4 (onepiece params)
- Warmup: 500 steps (0.1 → 1.0 linear)
- **Note**: FiLM zero-init이 gradient barrier 역할 → Mamba에 gradient 도달 불가. MetricHead의 scale/shift range 안정화가 목적이므로 1500 step이면 충분.

#### Phase 2 (Step 1500+)
**목표**: Full video optimization (depth quality 향상)

| Component | Status |
|-----------|--------|
| DINOv2 ViT-L | **Frozen** (항상) |
| DPT Head | **Frozen** (항상, V1과 다름!) |
| output_conv (RelativeHead) | **Trainable** (LR 1/10) |
| OnepieceFiLMGenerator | **Trainable** (LR 1/10) |
| UnifiedGlobalMamba | **Trainable** |
| OnepieceMetricHead | **Trainable** |

- Losses: **Log L1 + TGM + OFC** (full graph)
- FiLM/RelativeHead에 500-step warmup 적용 (LR 0 → 1/10 of base)
- Optimizer 재생성 (2 param groups: mamba_metric, film_relhead)
- OFC가 FiLM weights를 non-zero로 update → 이후 LogL1/TGM gradient가 Mamba에 도달

### Phase Transition 로직

```python
if self.current_phase == 1 and step == self.auto_transition_step:  # default 1500
    self._transition_to_phase2()
```

Phase 2 전환 시:
1. `_configure_parameters_phase2()`: FiLM + output_conv unfreeze
2. Optimizer 재생성 (param groups 분리)
3. Scheduler 재생성 (phase2-specific warmup lambda)
4. `_set_train_mode()`: FiLM, output_conv를 train mode로 전환

### Optimizer Parameter Groups

**Phase 1**: 1 group
```python
[{'params': mamba + metric_head, 'lr': 1e-4}]
```

**Phase 2**: 2 groups
```python
[{'params': mamba + metric_head, 'lr': 1e-4},
 {'params': film_generator + output_conv, 'lr': 1e-5}]
```

### Training Details

| 항목 | 값 |
|------|-----|
| Resolution | 518x518 (base) |
| Video length | 8 frames |
| Batch size | 3 per GPU |
| Optimizer | AdamW (betas=[0.9, 0.95], weight_decay=1e-6) |
| Mixed precision | BFloat16 autocast (forward), Float32 (loss) |
| Gradient clipping | max_norm=1.0 |
| Gradient checkpointing | Enabled (ViT + DPT) |
| Train datasets | tartanair (metric GT required) |
| Val datasets | tartanair |
| Validation freq | 1000 steps |
| Save freq | 5000 steps |
| Total iterations | 40001 |

### Checkpoint Format

```python
{
    'global_step': int,
    'model': state_dict,
    'optimizer': optimizer_state,
    'scheduler': scheduler_state,
    'best_val_loss': float,
    'best_step': int,
    'current_val_loss': float,
    'dataset_losses': dict,
    'num_sequences': dict,
    'config': dict,
    'current_phase': int,
}
```

저장 파일:
- `best.pth`: Best validation loss 기준
- `checkpoint_step{N}.pth`: 주기적 저장 (5000 step마다)
- `last.pth`: 훈련 종료 시

Checkpoint loading: `unified_global_mamba`, `onepiece_metric_head`, `onepiece_film_generator` 키는 로딩 제외 (새로 학습)

---

## Testing (`test_onepiece.py`)

### Test Datasets (Default)

`sintel`, `waymo_seg`, `eth3d`, `urbansyn`, `unreal4k`, `bonn`

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| AbsRel | \|pred - gt\| / gt (상대 오차) |
| MAE | Mean Absolute Error (m) |
| RMSE | Root Mean Squared Error (m) |
| delta_1 | max(pred/gt, gt/pred) < 1.25 |
| delta_2 | < 1.25^2 |
| delta_3 | < 1.25^3 |
| TAE (reproj) | Temporal Alignment Error (reprojection 기반) |
| FPS | Frames per second |

### De-canonicalization

```python
# fx_ratio = canonical_fx / actual_fx
# resize_ratio = resized_fx / actual_fx
de_canonical_ratio_metric = resize_ratio / fx_ratio
pred_depths_actual = metric_depth_canonical * de_canonical_ratio_metric
```

### Depth Conversion Formula

```python
# Model output
relative_depth = output_conv(modulated_features)    # relative depth
depth_from_rel = 1.0 / (relative_depth + 1e-8)     # scale absorbs old 100x
metric_depth = scale * depth_from_rel + shift       # metric depth (meters)
```

### JSON Outputs

| File | Content |
|------|---------|
| `test_results.json` | Aggregated metrics (전체 평균) |
| `per_sequence_results.json` | Per-sequence metrics |
| `best_sequence.json` | AbsRel 최소 sequence |
| `worst_sequence.json` | AbsRel 최대 sequence |
| `scale_shift_comparison.json` | Pred vs Optimal scale/shift, drift 분석 |
| `depth_range_analysis.json` | Depth range별 AbsRel, delta_1 |
| `temporal_analysis.json` | Per-frame TAE, spike 감지 |

### Visualizations

| Type | Location |
|------|----------|
| Per-frame PNG | `frames/seq{N}/frame_{T}.png` (Image, GT, Pred) |
| Error heatmaps | `error_heatmaps/seq{N}/error_{T}.png` (scale/shift overlay) |
| Best/worst frame | 3x3 grid with depth distribution |
| Video/GIF | Sequence animation (skip for urbansyn, unreal4k) |

---

## Model Integration (`flashdepth/model.py`)

### Initialization (lines 108-140)

```python
if self.use_onepiece:
    gap_dim = dpt_dim                     # 256 (ViT-L)
    pooled_dim = gap_dim * 2              # 512 (GAP + GStdP)

    self.unified_global_mamba = UnifiedGlobalMamba(d_input=pooled_dim, num_layers=4, ...)
    self.onepiece_film_generator = OnepieceFiLMGenerator(mamba_dim=pooled_dim, dpt_dim=gap_dim)
    self.onepiece_metric_head = OnepieceMetricHead(dpt_dim=gap_dim)
```

### `forward_with_onepiece` (line 460)

7-step pipeline:

1. **DINOv2 Encoder** (frozen): `get_intermediate_layers(video_flat)` → `encoder_features`
2. **DPT Head** (always frozen): `encoder_features → dpt_features [B*T, 256, h, w]`
3. **GAP + GStdP**: `adaptive_avg_pool2d + std → concat [B*T, 512]`
4. **UnifiedGlobalMamba** (trainable): `[B, T, 512] → refined_global`
5. **FiLMGenerator**: `refined_global → gamma, beta → modulated = gamma * dpt + beta`
6a. **Relative Head**: `modulated → relative_depth`
6b. **MetricHead**: `modulated → scale (max 1000), shift`
7. **Metric conversion**: `metric_depth = scale * (1/relative_depth) + shift`

All outputs use full graph (no gradient isolation).

### Return Dict

```python
{
    'metric_depth': [B, T, H, W],            # Full graph (for LogL1 + TGM)
    'relative_depth': [B, T, H, W],          # Full graph
    'modulated_features': [B, T, 256, h, w], # For OFC
    'scale': [B, T],                          # Per-frame scale (init ≈ 100, max 1000)
    'shift': [B, T],                          # Per-frame shift (range [0, 1.0])
    'dpt_features': [B, T, 256, h, w],       # Raw DPT features
}
```

### `forward_with_onepiece_streaming` (line 576)

Frame-by-frame streaming inference. Mamba hidden state를 유지하며 1 frame씩 처리. Training forward와 수학적으로 동일한 결과 (Mamba2 parallel scan == sequential recurrence).

---

## Configuration (`configs/onepiece/config.yaml`)

```yaml
# Onepiece V2 Configuration
# GAP+GStdP(512-dim) → Unified Global Mamba (4 layers) → FiLM + MetricHead

load: null
no_shift: false                  # Scale-only mode (zero out shift)

model:
  vit_size: "vitl"
  use_onepiece: true
  unified_mamba_layers: 4
  unified_mamba_d_state: 64
  unified_mamba_d_conv: 4

dataset:
  resolution: 'base'             # 518x518
  video_length: 8
  train_datasets: [tartanair]    # Metric GT required
  val_datasets: [tartanair]

training:
  batch_size: 3                  # Per GPU
  lr:
    onepiece: 1.0e-4             # Mamba + MetricHead
    dpt: 1.0e-5                  # FiLM + RelativeHead (Phase 2)
    warmup_steps: 500

loss:
  log_l1_weight: 1.0
  tgm_weight: 1.0
  ofc_weight: 0.01              # Raw L2 스케일 보정
  use_log_space: true

phase:
  auto_transition_step: 1500     # Phase 1 → Phase 2
  phase2_warmup_steps: 500

flow:
  model: "sea_raft"
  checkpoint: "third_party/SEA-RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth"
```

---

## Parameter Count (ViT-L)

| Module | Parameters | Phase 1 | Phase 2 |
|--------|-----------|---------|---------|
| DINOv2 ViT-L | ~304M | Frozen | Frozen |
| DPT Head (dpt_dim=256) | ~30.6M | Frozen | **Frozen** |
| UnifiedGlobalMamba (512-dim, 4L) | ~10.7M | **Trainable** | **Trainable** |
| OnepieceMetricHead (Conv) | ~16.5K | **Trainable** | **Trainable** |
| OnepieceFiLMGenerator (512→256) | ~0.26M | Frozen | **Trainable** |
| output_conv (RelativeHead) | ~0.33M | Frozen | **Trainable** |
| **Phase 1 trainable** | **~10.9M** | | |
| **Phase 2 trainable** | **~11.3M** | | |

V1 대비 Phase 1 trainable: 47.2M → **10.9M** (77% 감소). 주 원인: d_model 1280→512 (params ∝ d_model²).

### Per-layer Parameter Breakdown (d_model=512, 4 layers)

| Component | Per-layer | Total (×4) |
|-----------|:-:|:-:|
| `in_proj` (512→~2048) | ~1.05M | ~4.2M |
| `conv1d` (1024, d_conv=4) | ~4K | ~16K |
| `out_proj` (1024→512) | ~524K | ~2.1M |
| `mlp` (512↔2048) | ~2.1M | ~8.4M |
| norm, bias | ~3K | ~12K |
| **Total** | **~3.68M** | **~10.7M** |

---

## File Structure

```
flashdepth_claude/
├── flashdepth/
│   ├── model.py                    # forward_with_onepiece (line 460), streaming (line 576)
│   ├── onepiece_modules.py         # UnifiedGlobalMamba, OnepieceFiLMGenerator, OnepieceMetricHead
│   └── mamba.py                    # MambaBlock, InferenceParams
├── utils/
│   ├── onepiece_losses.py          # OpticalFlowConsistencyLoss, OnepieceCombinedLoss (3-loss)
│   ├── onepiece_visualization.py   # OnepieceVisualizer (3x3 grid)
│   ├── flow_estimator.py           # Sea-RAFT wrapper (frozen, Phase 2 only)
│   └── gear_losses.py              # LogL1Loss, TGMTemporalLoss (reused)
├── configs/
│   └── onepiece/
│       └── config.yaml             # Onepiece V2 config (GAP+GStdP, 4-loss)
├── third_party/
│   └── SEA-RAFT/                   # Sea-RAFT optical flow (training Phase 2 only)
├── train_onepiece.py               # OnepieceTrainer (Phase 1/2 auto-transition)
├── test_onepiece.py                # OnepieceTester (streaming inference + evaluation)
└── Onepiece.md                     # This document
```

---

## Removed in V2

- **CLS token**: No longer extracted or used. `_get_intermediate_layers_with_cls` no longer called by onepiece.
- **SceneCutDetector**: Entirely deleted from `onepiece_modules.py`. No `scene_cut_weights` in loss or training.
- **Feature downsample**: OFC uses full-resolution modulated features (no 1/4 downsampling).
- **Combined MetricHead**: Old MLP-based head that predicted scale/shift AND FiLM params together. Replaced by separate `OnepieceFiLMGenerator` + Conv-based `OnepieceMetricHead`.
- **`cls_layers` config**: No CLS layer selection needed.
- **DPT unfreeze in Phase 2**: DPT Head stays frozen in both phases. Only FiLM + RelativeHead (output_conv) are unfrozen in Phase 2.
- **SSIL loss**: Removed — output_conv (~330K params) too small to benefit, and SSIL diverged causing scale explosion.
- **Gradient isolation for LogL1**: Removed — LogL1 now uses full graph `metric_depth` instead of `metric_depth_isolated`.
- **WFC naming**: Renamed to OFC (Optical Flow Consistency) for clarity.
