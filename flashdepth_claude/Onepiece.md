# Onepiece V3: CLS-guided Spatial Mamba + Dual-Stream Architecture for Metric Depth

## Overview

Onepiece V3는 FlashDepth의 relative depth를 metric depth로 변환하기 위한 **CLS-guided Dual-Stream Architecture**. DPT feature를 1/10 downsample한 뒤 DINOv2 fused CLS token을 앞에 prepend하여 Mamba에 통과 → CLS는 CLSMetricHead로 (scale/shift 예측), DPT는 temporal alignment으로 역할 분리.

### Version History

| Version | Architecture | Key Feature |
|---------|-------------|-------------|
| V1 | CLS+GAP → UnifiedGlobalMamba → FiLM+MetricHead | Global token Mamba, FiLM spatial modulation |
| V2 | GAP+GStdP → shared modulated features | 최적화 충돌로 V1보다 성능 하락 |
| V3-prev | DPT → SpatialMamba → ConvMetricHead(spatial) | DPT dual-role 문제 (alignment + metric) |
| **V3** | **CLS prepend + SpatialMamba → CLSMetricHead + DPT alignment** | CLS-DPT 역할 분리, gradient conflict 해소 |

### V3 핵심 변경점 (vs V3-prev)

| 항목 | V3-prev | V3 (current) |
|------|---------|-----|
| Mamba 입력 | 196 spatial tokens/frame | **1 CLS + 196 spatial tokens/frame** (CLS prepend) |
| Metric Head | ConvMetricHead (Conv1x1 on spatial + GAP) | **CLSMetricHead** (MLP on Mamba-processed CLS) |
| CLS 사용 | 추론 SCD에만 (last layer) | **fused CLS (layers 17,23 평균)**: SCD + Mamba input + MetricHead |
| CLS projection | 없음 | **Linear(1024→256)** (FlashDepth에 별도 모듈) |
| Phase 1 trainable | ConvMetricHead only (~33K) | **CLSMetricHead + CLS projection** (~263K) |
| Phase 1 CLS flow | N/A | CLS bypass (Mamba frozen이라 의미 없음) |
| Phase 2 CLS flow | N/A | CLS → Mamba 통과 → temporal context 획득 |

---

## Architecture

### Full Pipeline (Training)

```
Input Video [B, T, 3, H, W]
    |
    v
[Frozen] DINOv2 Encoder (ViT-L or ViT-S)
    |
    +---> Intermediate features (4 layers)
    +---> Fused CLS token (layers 17,23 averaged) [B*T, 1024]
    |
    v
[Trainable] CLS projection: Linear(1024→256)  →  cls_projected [B*T, 256]
    |
[Frozen/Phase2] DPT Head  →  DPT features [B*T, 256, h, w]
    |
    v
[Trainable] SpatialMamba
    |
    +------- downsample (1/10) ------+
    |                                |
    |   DPT: [B, T, h'*w', 256]     |
    |   CLS: [B, T, 1, 256]         |
    |                                |
    |   concat → [B, T, 1+h'*w', 256]  (CLS prepend)
    |                                |
    |   Per-frame Mamba blocks (4 layers, hidden state)
    |                                |
    |   split → CLS [B*T, 256]  +  DPT [B*T, h'*w', 256]
    |                |                |
    |                |          final_layer (GELU + Linear, ZERO-INIT)
    |                |                |
    |                |          upsample + ADD (residual with original DPT)
    |                |                |
    |                v                v
    |       cls_output [B*T, 256]   post_mamba [B*T, 256, h, w]
    |                |                |
    |                v                v
    |      [Trainable] CLSMetricHead  [Frozen/Phase2] output_conv
    |                |                |
    |                v                v
    |      scale, shift [B*T, 1]    relative_depth [B*T, H, W]
    |                |                |
    +----------------+----------------+
                     |
                     v
    metric_depth = scale * 100/(rel+eps) + shift   (metric mode)
             OR  = scale * rel + shift              (inverse mode)
```

**Phase 1 특이사항**: SpatialMamba가 frozen이므로 CLS는 Mamba를 bypass하고 cls_projected가 직접 CLSMetricHead로 전달됨.

### Scene Cut Detection (Inference Only)

```
Fused CLS token (layers 17,23 avg) [B, embed_dim]
    |
    v
D_cls = 1 - cosine_similarity(CLS_t, CLS_{t-1})
    |
    v
if D_cls > tau (0.05):
    → SpatialMamba.start_new_sequence()  (Mamba hidden state reset)
    → Record frame index in reset_frames list
```

V3에서 Scene Cut Detection은 **추론 시에만** 사용. 훈련에서는 SCD를 적용하지 않으며 TGM loss에 scene cut weight를 적용하지 않음. SCD와 Mamba input 모두 동일한 **fused CLS (layers 17,23 평균)**를 사용.

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
- `forward(dpt_features, B, T, cls_projected=None)`: 훈련용 batch mode. CLS 전달 시 prepend→split→`(post_mamba, cls_output)` 반환. CLS 없으면 legacy `(post_mamba, mamba_raw_spatial)` 반환.
- `forward_single_frame(dpt_features_single, cls_projected=None)`: 추론용 streaming mode. 동일 CLS 옵션.
- `start_new_sequence()`: Mamba hidden state reset (scene cut 시)

**CLS prepend 동작**:
```
Per frame: [CLS(1), DPT(196)] → Mamba → split → CLS(1), DPT(196)
```
- CLS는 Mamba의 causal 특성상 같은 프레임 DPT에 오염되지 않음
- 이전 프레임 hidden state를 통해 temporal context만 수신
- MambaBlock의 pre-norm residual로 원본 CLS 정보 보존

**Zero-init final_layer**: `nn.Sequential(GELU(), Linear(dpt_dim, dpt_dim))` with weights/bias zeroed.
→ 훈련 시작 시 `post_mamba = original DPT features` (identity, Mamba output = 0).

**Internal**: 각 layer는 `MambaBlock` (`flashdepth/mamba.py`). `InferenceParams`로 streaming hidden state 관리.

### 2. CLSMetricHead (`flashdepth/onepiece_modules.py`)

Mamba-processed CLS token으로부터 scale/shift를 예측하는 MLP-based head.

```python
CLSMetricHead(
    dpt_dim=256,              # Input dimension (Mamba CLS output)
    hidden_dim=64,            # Hidden dimension
    train_mode="metric"       # "metric" or "inverse"
)
```

**Architecture**:
```
cls_output [B*T, 256]
    |
    v
Linear(256, 64) → ReLU → Linear(64, 2)
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

### 2b. CLS Projection (`model.py`)

DINOv2 fused CLS (1024-dim)를 DPT feature space (256-dim)로 변환.

```python
self.cls_projection = nn.Linear(embed_dim, dpt_dim)  # 1024 → 256
self.cls_layer_indices = [2, 3]  # ViT layers 17, 23 평균
```

FlashDepth 모듈에 별도 배치 (SpatialMamba 내부가 아님).
→ Phase 1에서 SpatialMamba가 `torch.no_grad()`로 감싸져도 CLS projection의 gradient flow 보장.

### 2c. ConvMetricHead (Legacy, `flashdepth/onepiece_modules.py`)

V3-prev에서 사용하던 Conv-based head. 코드 유지되나 현재 사용하지 않음.

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
| CLS projection | **Trainable** |
| CLSMetricHead | **Trainable** |

- Trainable params: ~263K (CLSMetricHead + CLS projection)
- CLS는 Mamba를 bypass → cls_projected가 직접 CLSMetricHead로 전달
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
| CLS projection | **Trainable** |
| CLSMetricHead | **Trainable** |

- Phase 2 warmup: 500 steps (DPT LR 0 → 1/10 of base)
- CLS가 Mamba 통과 → temporal context 획득 후 CLSMetricHead로
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
│   ├── onepiece_modules.py         # SpatialMamba, CLSMetricHead, ConvMetricHead(legacy), SceneCutDetector
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
3. **Scene Cut Detection은 추론 only**: 훈련에서는 SCD 미적용. TGM에 scene cut weight 미적용. SCD와 Mamba 모두 fused CLS (layers 17,23) 사용.
4. **Phase 1 CLS bypass**: Phase 1에서 SpatialMamba가 frozen이므로 CLS는 Mamba를 거치지 않고 직접 CLSMetricHead로 전달. Phase 2부터 Mamba 통과.
5. **Inverse mode**: `train_mode: inverse`로 inverse depth regression 가능. Scale/shift activation 다름.
6. **ViT-S auto-detect**: `dpt_dim=64`이면 SpatialMamba가 자동으로 `expand=4, headdim=32` 사용.
7. **V3-prev 체크포인트 비호환**: ConvMetricHead → CLSMetricHead, cls_projection 추가로 기존 V3 체크포인트 재학습 필요.
