# Onepiece V3: Spatial Mamba + Dual-Stream Architecture for Metric Depth

## Overview

Onepiece V3는 FlashDepth의 relative depth를 metric depth로 변환하기 위한 **Dual-Stream Architecture**. V1의 global token approach (CLS+GAP→UnifiedGlobalMamba→MetricHead+FiLM)를 **Spatial Mamba** 기반으로 대체. DPT feature를 1/10 downsample → Mamba temporal processing → dual output으로 relative + metric depth를 동시 생성.

### Version History

| Version | Architecture | Key Feature |
|---------|-------------|-------------|
| V1 | CLS+GAP → UnifiedGlobalMamba → FiLM+MetricHead | Global token Mamba, FiLM spatial modulation |
| V2 | GAP+GStdP → shared modulated features | 최적화 충돌로 V1보다 성능 하락 |
| **V3** | **DPT → SpatialMamba → Dual-Stream (Relative+Metric)** | FlashDepth 패턴 재사용, FiLM 제거 |

### V3 핵심 변경점 (vs V1)

| 항목 | V1 | V3 |
|------|-----|-----|
| Temporal 모델 | UnifiedGlobalMamba (CLS+GAP, 1280-dim) | **SpatialMamba** (DPT features, 256-dim) |
| Mamba 입력 | 1 global token/frame | **196 spatial tokens/frame** (1/10 downsampled) |
| Metric Head | MLP(1280→512→2) | **ConvMetricHead** (Conv1x1 on low-res Mamba output + GAP) |
| FiLM | gamma/beta modulation on DPT features | **없음** (Mamba output upsample+add) |
| Scene cut | 훈련+추론 모두 사용 (TGM gating) | **추론 only** (Mamba state reset) |
| Phase 1 trainable | UnifiedGlobalMamba + MetricHead (47.2M) | **ConvMetricHead only** (~33K) |
| Phase 1 → 2 전환 | 5000 steps | **1500 steps** |
| Depth valid range | 70m | **80m** (configurable via --max-depth) |

---

## Architecture

### Full Pipeline (Training)

```
Input Video [B, T, 3, H, W]
    |
    v
[Frozen] DINOv2 Encoder (ViT-L or ViT-S)
    |
    v
Intermediate features (4 layers)
    |
    v
[Frozen/Phase2] DPT Head
    |
    v
DPT features [B*T, 256, h, w]
    |
    v
[Trainable] SpatialMamba
    |
    +------- downsample (1/10) ------+
    |                                |
    |   [B*T, 256, h', w'] → reshape → [B, T, h'*w', 256]
    |                                |
    |   Per-frame Mamba blocks (4 layers, hidden state)
    |                                |
    |   final_layer (GELU + Linear, ZERO-INIT)
    |                                |
    +------- upsample + ADD ---------+
    |                                |
    v                                v
post_mamba [B*T, 256, h, w]    mamba_raw [B*T, 256, h', w']
(= DPT + upsample(final_layer(mamba_out)))
    |                                |
    v                                v
[Frozen/Phase2] output_conv    [Trainable] ConvMetricHead
    |                                |
    v                                v
relative_depth [B*T, H, W]    scale [B*T, 1], shift [B*T, 1]
    |                                |
    +--------------------------------+
    |
    v
metric_depth = scale * 100/(rel+eps) + shift   (metric mode)
         OR  = scale * rel + shift              (inverse mode)
```

### Scene Cut Detection (Inference Only)

```
CLS token (last layer) [B, embed_dim]
    |
    v
D_cls = 1 - cosine_similarity(CLS_t, CLS_{t-1})
    |
    v
if D_cls > tau (0.05):
    → SpatialMamba.start_new_sequence()  (Mamba hidden state reset)
    → Record frame index in reset_frames list
```

V3에서 Scene Cut Detection은 **추론 시에만** 사용. 훈련에서는 CLS를 추출하지 않으며 TGM loss에 scene cut weight를 적용하지 않음.

---

## Core Modules

### 1. SpatialMamba (`flashdepth/onepiece_modules.py`)

FlashDepth의 `dpt_features_to_mamba` 패턴을 재사용한 spatial temporal Mamba.

```python
SpatialMamba(
    dpt_dim=256,              # DPT feature dimension
    num_layers=4,             # Mamba block depth
    d_state=256,              # SSM state dimension
    d_conv=4,                 # 1D convolution kernel
    expand=2,                 # Mamba expansion factor (ViT-L)
    headdim=64,               # Head dimension (ViT-L)
    downsample_factor=0.1,    # Spatial downsample (1/10)
    max_batch_size=8          # For InferenceParams allocation
)
```

**ViT-S Override**: `expand=4, headdim=32` (matching FlashDepth-S architecture).

**Forward modes**:
- `forward(dpt_features, B, T)`: 훈련용 batch mode. Returns `(post_mamba, mamba_raw_spatial)`
- `forward_single_frame(dpt_features_single)`: 추론용 streaming mode. Returns `(post_mamba, mamba_raw_spatial)`
- `start_new_sequence()`: Mamba hidden state reset (scene cut 시)

**Zero-init final_layer**: `nn.Sequential(GELU(), Linear(dpt_dim, dpt_dim))` with weights/bias zeroed.
→ 훈련 시작 시 `post_mamba = original DPT features` (identity, Mamba output = 0).

**Internal**: 각 layer는 `MambaBlock` (`flashdepth/mamba.py`). `InferenceParams`로 streaming hidden state 관리.

### 2. ConvMetricHead (`flashdepth/onepiece_modules.py`)

Low-resolution Mamba output으로부터 scale/shift를 예측하는 Conv-based head.

```python
ConvMetricHead(
    dpt_dim=256,              # Input channels (Mamba output)
    hidden_dim=64,            # Hidden channels
    train_mode="metric"       # "metric" or "inverse"
)
```

**Architecture**:
```
mamba_raw [B*T, 256, h', w']
    |
    v
Conv2d(256, 64, 1) → ReLU → Conv2d(64, 2, 1)
    |
    v
Global Average Pooling → [B*T, 2]
    |
    v
raw[:, 0] → scale,  raw[:, 1] → shift
```

**Metric mode**:
- scale = softplus(raw_scale)  (항상 양수)
- shift = sigmoid(raw_shift)   (범위 [0, 1])

**Inverse mode**:
- scale = softplus(raw_scale)  (항상 양수)
- shift = raw_shift             (unconstrained)

**Weight 초기화**:
- Scale bias: `softplus(0.5413) ≈ 1.0` (identity scale)
- Shift bias: `-5.0` → `sigmoid(-5) ≈ 0.0067 ≈ 0` (metric mode)

### 3. SceneCutDetector (`flashdepth/onepiece_modules.py`)

V1과 동일. CLS cosine distance 기반 scene cut 감지. **V3에서는 추론 only**.

```python
SceneCutDetector(tau=0.05, k=80)
```

---

## Loss Functions (`utils/onepiece_losses.py`)

### Combined Loss

```
L_total = w1 * L_log_l1 + w2 * L_tgm + w3 * L_ofc
```

**Default weights**: 1:1:0.01

### 1. Log L1 Loss (Reuses `gear_losses.LogL1Loss`)

Per-frame metric depth 정확도 loss.

```
L_log_l1 = |log(100/pred_meters) - log(gt_inverse)|
```

### 2. TGM Temporal Loss (Reuses `gear_losses.TGMTemporalLoss`)

Temporal gradient matching. V3에서는 **scene cut weight 미적용** (훈련에서 SCD 제거).

### 3. Optical Flow Consistency Loss (OFC, renamed from WFC)

**post_mamba_features**의 temporal consistency를 optical flow 기반으로 강제.

```
1. Sea-RAFT로 optical flow 추정
2. post_mamba features를 1/4로 downsample
3. Flow warp + confidence-weighted L2 loss
```

V1에서는 `dpt_features`에 적용했으나, V3에서는 `post_mamba_features`에 적용.
→ Gradient가 DPT와 SpatialMamba 양쪽으로 전파됨 (Phase 2).

---

## Training (`train_onepiece.py`)

### Two-Phase Training Strategy

#### Phase 1 (Step 0 ~ 1500)
**목표**: Metric alignment (scale/shift 수렴)

| Component | Status |
|-----------|--------|
| DINOv2 | **Frozen** |
| DPT Head | **Frozen** |
| output_conv | **Frozen** |
| SpatialMamba | **Frozen** (zero-init → no-op) |
| ConvMetricHead | **Trainable** |

- Trainable params: ~33K (ConvMetricHead only)
- SpatialMamba의 zero-init final_layer로 인해 post_mamba = 원본 DPT features
- **OFC skip**: DPT frozen → gradient 전파 불가 → Sea-RAFT 연산 건너뜀

#### Phase 2 (Step 1500+)
**목표**: Full video optimization

| Component | Status |
|-----------|--------|
| DINOv2 | **Frozen** (항상) |
| DPT Head | **Trainable** (LR 1/10) |
| output_conv | **Trainable** (LR 1/10) |
| SpatialMamba | **Trainable** |
| ConvMetricHead | **Trainable** |

- Phase 2 warmup: 500 steps (DPT LR 0 → 1/10 of base)
- OFC 활성화: post_mamba → DPT + Mamba 양쪽 gradient

### Training Modes

| Mode | Depth conversion | Scale | Shift |
|------|-----------------|-------|-------|
| `metric` (default) | `scale * 100/(rel+eps) + shift` | softplus | sigmoid |
| `inverse` | `scale * rel + shift` | softplus | unconstrained |

Config: `train_mode: metric` or `train_mode: inverse`

### Training Configuration

| 항목 | 값 |
|------|-----|
| Resolution | 518x518 (base) |
| Video length | 8 frames |
| Batch size | 3 (ViT-L) / 8 (ViT-S) per GPU |
| Optimizer | AdamW (betas=[0.9, 0.95], weight_decay=1e-6) |
| Base LR | 1e-4 (onepiece), 1e-5 (dpt, 1/10) |
| Warmup | 500 steps |
| Phase transition | 1500 steps |
| Total iterations | 40001 |
| Validation | Every 1000 steps |
| Val max depth | **80m** |
| Loss | LogL1 + TGM + OFC (1:1:0.01) |

---

## Testing (`test_onepiece.py`)

### Forward Mode

Streaming inference: frame-by-frame with Mamba state, CLS-based scene cut detection.

```python
model.forward_with_onepiece_streaming(images)
# Returns: metric_depth, relative_depth, scale, shift, reset_frames
```

Or per-frame helper:
```python
model.forward_onepiece_single_frame(frame, prev_cls=prev_cls)
# Returns: dict with metric_depth, relative_depth, scale, shift, cls_token, is_reset
```

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| AbsRel | \|pred - gt\| / gt |
| MAE | Mean Absolute Error (m) |
| RMSE | Root Mean Squared Error (m) |
| delta_1/2/3 | Threshold accuracy |
| TAE (reproj) | Temporal Alignment Error |
| rTC | Flow-based temporal consistency |
| FPS | Frames per second |

### Max Depth Configuration

```bash
# CLI (Hydra-style for test_onepiece.py, test_gear5.py)
python test_onepiece.py --max-depth 80

# CLI (argparse for test_comparison.py, test_video_comparison.py)
python test_comparison.py --max-depth 80

# Shell scripts
./run_comparison.sh metric3d --max-depth 80
./run_video_comparison.sh vda --max-depth 80

# Docker
./run_docker.sh test_onepiece --max-depth 80
```

Default: 80.0m (changed from V1's 70.0m)

### Depth Range Analysis

3개 depth range별 metrics:
- **0-10m**: 근거리
- **10-30m**: 중거리
- **30-{max_depth}m**: 원거리

### Visualizations

| Type | Location |
|------|----------|
| Per-frame PNG | `frames/seq{N}/frame_{T}.png` |
| Error heatmaps | `error_heatmaps/seq{N}/error_{T}.png` |
| Best/Worst frame | `{best,worst}_frames/` (3x3 grid with Scene Cut Resets panel) |
| Video/GIF | Sequence animation |

---

## Configuration

### Config Variants (`configs/onepiece/`)

| File | Variant | ViT | DPT dim | Mamba expand | Mamba headdim | Batch |
|------|---------|-----|---------|-------------|---------------|-------|
| `config_l.yaml` | l | ViT-L | 256 | 2 | 64 | 3 |
| `config_s.yaml` | s | ViT-S | 64 | 4 | 32 | 8 |

### Config Sample (`config_l.yaml`)

```yaml
config_variant: l
load: configs/flashdepth-l/iter_10001.pth
train_mode: metric               # "metric" or "inverse"

model:
  vit_size: "vitl"
  use_onepiece: true
  spatial_mamba_layers: 4
  spatial_mamba_d_state: 256
  spatial_mamba_d_conv: 4
  spatial_mamba_downsample: 0.1

loss:
  log_l1_weight: 1.0
  tgm_weight: 1.0
  ofc_weight: 0.01
  use_log_space: true

phase:
  auto_transition_step: 1500
  phase2_warmup_steps: 500

scene_cut:
  tau: 0.05
  k: 80
```

---

## Docker Commands

### Training

```bash
# Single GPU - ViT-L
./run_docker.sh train_onepiece --config-variant l --batch-size 3 --gpu 0 \
    --results-dir train_results/onepiece/large/

# Single GPU - ViT-S
./run_docker.sh train_onepiece --config-variant s --batch-size 8 --gpu 0 \
    --results-dir train_results/onepiece/small/

# DDP (2 GPUs)
./run_docker.sh train_onepiece_ddp --config-variant l --batch-size 3 --ddp-gpus 1,2 \
    --results-dir train_results/onepiece/large/
```

### Testing

```bash
# Test ViT-L
./run_docker.sh test_onepiece --config-variant l \
    --gear-checkpoint train_results/onepiece/large/best.pth --gpu 0

# Test with custom max depth
./run_docker.sh test_onepiece --config-variant l \
    --gear-checkpoint train_results/onepiece/large/best.pth --gpu 0 --max-depth 100
```

---

## Flow Estimator (`utils/flow_estimator.py`)

**Sea-RAFT** wrapper. **훈련 시에만 사용**, 추론/테스트에서는 불필요.

설치:
```bash
git clone https://github.com/princeton-vl/SEA-RAFT.git third_party/SEA-RAFT/
# Checkpoint: third_party/SEA-RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth
```

---

## File Structure

```
flashdepth_claude/
├── flashdepth/
│   ├── model.py                    # forward_with_onepiece(), forward_with_onepiece_streaming(),
│   │                               #   forward_onepiece_single_frame()
│   ├── onepiece_modules.py         # SpatialMamba, ConvMetricHead, SceneCutDetector
│   │                               #   (V1: UnifiedGlobalMamba, OnepieceMetricHead commented out)
│   └── mamba.py                    # MambaBlock, InferenceParams (shared with FlashDepth)
├── utils/
│   ├── onepiece_losses.py          # OpticalFlowConsistencyLoss (OFC), OnepieceCombinedLoss
│   ├── onepiece_visualization.py   # OnepieceVisualizer
│   ├── flow_estimator.py           # Sea-RAFT wrapper (training only)
│   └── gear_losses.py              # LogL1Loss, TGMTemporalLoss (reused)
├── configs/
│   └── onepiece/
│       ├── config.yaml             # Default config (ViT-L)
│       ├── config_l.yaml           # ViT-L variant
│       └── config_s.yaml           # ViT-S variant
├── train_onepiece.py               # OnepieceTrainer (Phase 1→2 auto-transition at 1500 steps)
├── test_onepiece.py                # OnepieceTester (streaming, SCD, reset_frames)
├── test_gear5.py                   # Gear5 tester (copied from onepiece2)
├── test_comparison.py              # Image-model comparison tester
├── test_video_comparison.py        # Video-model comparison tester
├── run_comparison.sh               # Shell script for image model evaluation
├── run_video_comparison.sh         # Shell script for video model evaluation
└── run_docker.sh                   # Docker launcher
```

---

## Known Limitations / Notes

1. **Sea-RAFT 의존성**: OFC Loss 계산에 필수. 테스트는 불필요.
2. **Zero-init 의존**: SpatialMamba의 final_layer가 zero-init이므로 Phase 1에서 frozen 상태로 identity. Phase 2에서 학습 시작.
3. **Scene Cut Detection은 추론 only**: 훈련에서는 CLS 추출 안 함. TGM에 scene cut weight 미적용.
4. **Inverse mode**: `train_mode: inverse`로 inverse depth regression 가능. Scale/shift activation 다름.
5. **ViT-S auto-detect**: `dpt_dim=64`이면 SpatialMamba가 자동으로 `expand=4, headdim=32` 사용.
