# Onepiece: Unified Global Mamba for Metric Depth Estimation

## Overview

Onepiece는 FlashDepth의 relative depth를 metric depth로 변환하기 위한 통합 아키텍처. 기존 Gear5의 분리된 CLS-based scale/shift 예측을 **하나의 Mamba2 시퀀스 모델**로 통합하여, CLS token과 DPT Global Average Pooling (GAP) feature를 함께 temporal processing한 뒤 scale/shift 예측과 FiLM spatial modulation을 동시에 수행한다.

### 핵심 차별점 (vs Gear5)

| 항목 | Gear5 | Onepiece |
|------|-------|----------|
| Temporal 모델 | GRU or Mamba2 (CLS-only) | Unified Global Mamba (CLS+GAP) |
| 입력 차원 | CLS 1024-dim | CLS + GAP = **1280-dim** (ViT-L) / **448-dim** (ViT-S) |
| Spatial modulation | 없음 (scale/shift만) | **FiLM** (gamma, beta로 DPT feature 변조) |
| Scene cut | 없음 | **SceneCutDetector** (CLS cosine distance) |
| Loss | Log L1 + TGM | Log L1 + TGM + **Feature Consistency** |
| Flow estimator | 불필요 | **Sea-RAFT** (training only, frozen) |
| Importance map | 있음 (DINOv2 attention) | **없음** (단순화) |

---

## Architecture

### Full Pipeline

```
Input Video [B, T, 3, H, W]
    |
    v
[Frozen] DINOv2 ViT-L Encoder
    |                         \
    v                          v
CLS token [B*T, 1024]    Intermediate features (4 layers)
                               |
                               v
                    [Frozen/Phase2] DPT Head
                               |
                               v
                    DPT features [B*T, 256, h, w]
                               |
                        +------+------+
                        |             |
                        v             v
                GAP [B*T, 256]   (preserve for FiLM)
                        |
    CLS + GAP concat    |
    [B*T, 1280]  <------+
        |
        v  (reshape to [B, T, 1280])
    [Trainable] UnifiedGlobalMamba (2 layers)
        |
        v  [B, T, 1280]
        +---------------------------+
        |                           |
        v                           v
    Path A: Scale/Shift         Path B: FiLM
    MLP(1280->512->2)          MLP(256->256->512)
        |                           |
        v                           v
    scale (softplus)            gamma (1 + raw)
    shift (0.1*sigmoid)         beta (raw)
                                    |
                                    v
                          FiLM modulation:
                          feat' = gamma * feat + beta
                                    |
                                    v
                          [Frozen/Phase2] output_conv
                                    |
                                    v
                          relative_depth [B*T, H, W]
                                    |
                                    v
                          depth_from_relative = 100 / relative_depth
                                    |
                                    v
                          metric_depth = scale * depth_from_relative + shift
```

### Scene Cut Detection (병렬 실행)

```
CLS tokens [B, T, 1024]
    |
    v
D_cls = 1 - cosine_similarity(CLS_t, CLS_{t-1})    [B, T-1]
    |
    v
W_temporal = 1 - sigmoid(k * (D_cls - tau))          [B, T-1]
    |
    v
TGM loss *= W_temporal
Feature Consistency loss *= W_temporal
(Log L1 loss에는 적용 안 함 -- per-frame loss이므로)
```

---

## Core Modules

### 1. UnifiedGlobalMamba (`flashdepth/onepiece_modules.py`)

CLS + GAP = unified-dim token sequence를 temporal processing하는 Mamba2 모듈.

```python
UnifiedGlobalMamba(
    d_input=1280,       # CLS(1024) + GAP(256) for ViT-L; CLS(384) + GAP(64) = 448 for ViT-S
    num_layers=2,       # 2-layer Mamba2 stack
    d_state=64,         # SSM state dimension
    d_conv=4,           # 1D convolution kernel size
    expand=2,           # Mamba expansion factor
    headdim=64          # Head dimension
)
```

**Mamba2 Dimension Constraint**: `d_model * expand / headdim` must be a multiple of 8.

| Variant | CLS dim | GAP dim | d_input | d_model (Mamba2-valid) | nheads | Projection |
|---------|---------|---------|---------|------------------------|--------|------------|
| ViT-L | 1024 | 256 | 1280 | 1280 | 40 (=1280*2/64) | None |
| ViT-S | 384 | 64 | 448 | 512 | 16 (=512*2/64) | Linear(448→512) + Linear(512→448) |

**Auto-projection**: `_find_valid_mamba_dim(d_input)` finds nearest `d_model >= d_input` satisfying the constraint. When `d_input != d_model`, input/output linear projection layers are automatically added.

**Forward modes**:
- `forward(x)`: 훈련용 batch mode. `[B, T, 1280]` 입력, parallel scan으로 전체 시퀀스 처리
- `forward_single_frame(x)`: 추론용 streaming mode. `[B, 1, 1280]` 입력, hidden state 유지하며 frame-by-frame 처리

**Internal**: 각 layer는 `MambaBlock` (`flashdepth/mamba.py`). `InferenceParams`로 hidden state 관리.

### Spatial Mamba vs UnifiedGlobalMamba 상세 비교

두 Mamba 모두 동일한 `MambaBlock` (`flashdepth/mamba.py`)을 사용한다. MambaBlock 내부 구조:

```
Input x [B, L, d_model]
    |
    v
[Pre-Norm] LayerNorm(d_model)     ← norm1
    |
    v
[Mamba2 SSM]                       ← mamba (핵심 SSM 블록)
    |   in_proj:  Linear(d_model → ~4×d_model)   ... 입력 → 내부 확장
    |   conv1d:   Conv1d(d_inner, kernel=d_conv)  ... 로컬 컨텍스트
    |   SSM scan: A, B, C, D matrices             ... 시퀀스 모델링
    |   norm:     RMSNorm(d_inner)
    |   out_proj: Linear(d_inner → d_model)       ... 내부 → 출력 축소
    |
    v
[Residual] x = residual + mamba_out
    |
    v
[Pre-Norm] LayerNorm(d_model)     ← norm2
    |
    v
[MLP] Linear(d_model → 4×d_model) → GELU → Linear(4×d_model → d_model)
    |
    v
[Residual] x = residual + mlp_out
    |
    v
Output x [B, L, d_model]
```

#### 설정 비교

| 설정 | Spatial Mamba | UnifiedGlobalMamba |
|------|-------------|-------------------|
| **역할** | DPT feature의 temporal consistency | CLS+GAP의 temporal reasoning → scale/shift/FiLM |
| **입력** | `[B, 196, 256]` (spatial tokens) | `[B, T, 1280]` (1 global token/frame) |
| d_model | 256 | 1280 |
| d_inner (=d_model×expand) | 512 | 2560 |
| expand | 2 | 2 |
| headdim | 64 | 64 |
| nheads | 8 | 40 |
| d_state | 256 | 64 |
| **d_conv** | **256** | **4** |
| layers | 4 | 2 |
| 후처리 | `final_layer` (GELU→Linear, zero-init) | 없음 (MetricHead가 후처리) |

#### Per-layer 파라미터 비교 (실측)

| Component | Spatial (d=256) | Unified (d=1280) | 비율 | 역할 |
|-----------|:-:|:-:|:-:|------|
| `in_proj` | [1544, 256] = **395K** | [5288, 1280] = **6,769K** | 17x | 입력→내부 확장 |
| `conv1d` | [1024, 1, **256**] = **262K** | [2688, 1, **4**] = **11K** | 0.04x | 로컬 컨텍스트 |
| `out_proj` | [256, 512] = **131K** | [1280, 2560] = **3,277K** | 25x | 내부→출력 축소 |
| `mlp` | 256↔1024 = **525K** | 1280↔5120 = **13,107K** | 25x | 비선형 변환 |
| norm, bias 등 | ~3K | ~13K | - | - |
| **Per-layer 합계** | **1.32M** | **23.18M** | **17.6x** | |

| 집계 | Spatial | Unified |
|------|:-:|:-:|
| Per-layer | 1.32M | 23.18M |
| × Layers | ×4 | ×2 |
| + final_layer | +66K | - |
| **Total** | **5.33M** | **46.36M** |

#### conv1d 차이가 큰 이유

- **Spatial Mamba**: `d_conv=256` → DPT feature의 196개 spatial token 간 넓은 로컬 컨텍스트 필요
- **Unified Mamba**: `d_conv=4` → 프레임 시퀀스에서 인접 4프레임만 보면 충분

conv1d의 weight shape은 `[d_inner, 1, d_conv]`이므로, d_conv가 256 vs 4로 **64배 차이**. 하지만 전체 파라미터에서 conv1d 비중이 작아서(Spatial에서도 20%), 총 파라미터 차이의 주된 원인은 **d_model² 스케일링**인 in_proj, out_proj, mlp.

#### 파라미터가 d_model²에 비례하는 이유

Mamba 파라미터는 sequence length와 무관하고, d_model에 의해 결정된다:
- `in_proj`: d_model × ~4×d_model ≈ **4 × d_model²**
- `out_proj`: d_inner × d_model = expand × **d_model²**
- `mlp`: 2 × d_model × 4×d_model = **8 × d_model²**
- d_model 5배 (256→1280) → 파라미터 약 **25배/layer**
- Layer 수 반감 (4→2) 보정 → 최종 **약 8.7배** (5.33M → 46.36M)

### 2. OnepieceMetricHead (`flashdepth/onepiece_modules.py`)

Refined global token(1280-dim)으로부터 scale/shift와 FiLM parameters를 예측하는 dual-path head.

#### Path A: Scale/Shift 예측
```
refined_global [B*T, 1280]
    |
    v
Linear(1280, 512) -> ReLU -> Linear(512, 2)
    |                           |
    v                           v
raw_scale                    raw_shift
    |                           |
    v                           v
softplus(raw_scale)          0.1 * sigmoid(raw_shift)
= scale (항상 양수)          = shift (범위 [0, 0.1])
```

#### Path B: FiLM 생성
```
refined_global의 마지막 256-dim (GAP position) 추출
    |
    v
Linear(256, 256) -> ReLU -> Linear(256, 512)
    |
    v
chunk(2) -> gamma_raw [256], beta [256]
    |
    v
gamma = 1 + gamma_raw     (residual: 초기값 identity)
beta = beta                (초기값 0)
```

#### Weight 초기화 전략
- **Scale**: `softplus(0.5413) ≈ 1.0`으로 시작 (identity scale)
- **Shift**: `0.1 * sigmoid(-5) ≈ 0.0007 ≈ 0`으로 시작 (shift 거의 없음)
- **FiLM**: 마지막 layer zero-init → `gamma=1, beta=0` (identity transform)
- 이 초기화로 훈련 시작 시 모델이 **원본 FlashDepth의 relative depth를 그대로 유지**

### 3. SceneCutDetector (`flashdepth/onepiece_modules.py`)

CLS token의 cosine distance로 scene cut을 감지하여 temporal loss를 gating.

```python
SceneCutDetector(
    tau=0.05,    # Scene cut threshold
    k=80         # Sigmoid steepness
)
```

**동작 원리**:
```
D_cls = 1 - cos_sim(CLS_t, CLS_{t-1})      # Cosine distance [0, 2]
W_temporal = 1 - sigmoid(80 * (D_cls - 0.05))

When D_cls < 0.05 (같은 장면):  W_temporal ≈ 1.0 → temporal loss 유지
When D_cls > 0.05 (장면 전환):  W_temporal ≈ 0.0 → temporal loss 억제
```

**tau/k 선정 근거** (`measure_cls_distance.py` 실험):
- TartanAir 6개 scene에서 300 frames씩 CLS token 추출
- Adjacent frames (dt=1): D_cls mean ≈ 0.001~0.005
- Far-apart frames (dt=200): D_cls max ≈ 0.03~0.04
- Cross-sequence pairs: D_cls min ≈ 0.06~0.10
- **tau=0.05는 same-scene max와 cross-scene min의 중간점**

**NOTE**: `@torch.no_grad()` 데코레이터 적용. Scene cut detection은 gradient를 받지 않음 (CLS token은 frozen DINOv2에서 추출).

### 4. FiLM Spatial Modulation

DPT features에 대한 channel-wise affine transform:

```python
# gamma: [B*T, 256, 1, 1], beta: [B*T, 256, 1, 1]
modulated_features = gamma * dpt_features + beta
```

- `gamma = 1 + gamma_raw`: residual 구조로, 초기에는 identity (원본 feature 유지)
- `beta = beta_raw`: 초기에는 0
- Spatial broadcast: 모든 (h, w) 위치에 같은 gamma, beta 적용 (channel-wise, not spatial-wise)
- **FiLM의 목적**: scale/shift가 전체 depth map에 uniform하게 적용되는 반면, FiLM은 DPT feature 자체를 modulate하여 output_conv가 더 나은 relative depth를 생성하도록 유도

---

## Loss Functions (`utils/onepiece_losses.py`)

### Combined Loss

```
L_total = w1 * L_log_l1 + w2 * L_tgm + w3 * L_feat_cons
```

**Default weights**: 1:1:0.01 (feat_cons는 raw L2 스케일이 ~14로 크므로 0.01로 축소)

### 1. Log L1 Loss (Reuses `gear_losses.LogL1Loss`)

Per-frame metric depth 정확도 loss.

```
L_log_l1 = |log(pred_inverse) - log(gt_inverse)|
```

- `pred_inverse = 100 / metric_depth`, `gt_inverse = gt_depth * 100`
- Valid mask 적용: GT > 0 & pred > 0 & pred < 1000m
- **Scene cut weight 미적용** (per-frame loss이므로)

### 2. TGM Temporal Loss (Reuses `gear_losses.TGMTemporalLoss`)

Temporal gradient matching: frame간 depth 변화가 GT와 일치하도록.

```
L_tgm = |Delta_pred(t, t-1) - Delta_gt(t, t-1)|
```

- Scene cut weight 적용: **per-pair weighting** (각 pair에 해당 W_temporal 개별 적용)
- Multi-stride (stride=2,4,8) 시 pair (t, t+stride) → `min(W_temporal[t:t+stride])` 사용
- Scene cut 시 해당 pair만 억제 (다른 pair에 영향 없음)

### 3. Warp Feature Consistency Loss (NEW)

DPT feature의 temporal consistency를 optical flow 기반으로 강제.

```
1. Sea-RAFT로 optical flow 추정: flow [B, T-1, 2, H, W], confidence [B, T-1, 1, H, W]
2. DPT features를 1/4로 downsample (효율성)
3. Flow를 feature 해상도로 resize + scale 조정
4. Grid sample로 feat_{t-1}을 feat_t 위치로 warp
5. Confidence-weighted L2 loss:
   L_feat = mean(confidence * ||feat_t - warp(feat_{t-1})||^2)
6. Scene cut weight 적용: L_feat *= W_temporal
```

**Feature downsample**: DPT feature (h, w)를 (h/4, w/4)로 줄여서 flow 연산 효율화

### Depth Valid Range

| Stage | GT Valid | Pred Valid | 비고 |
|-------|---------|-----------|------|
| **Training** | `gt_inverse > 0` | `metric_depth > 0` & `< 1000m` | + `actual_valid_masks` AND |
| **Validation** | `gt > 0` & `< 70m` | `pred > 0` & `< 70m` | test와 동일 기준 |
| **Test (메트릭)** | `gt > 0` & `< 70m` | `pred > 0` & `< 70m` | 미터 공간 기준 |
| **Train vis only** | `inverse > 100/70` (70m) | `inverse > 100/200` (200m) | canonical valid mask |

- Training은 상한 **1000m**으로 느슨하게 설정 (원거리 anchor 유지, log L1이 자연 감쇠)
- Validation/Test는 상한 **70m**으로 통일 (best model 선정과 평가 기준 일치)
- Training visualization은 별도 canonical threshold 적용 (GT 70m, pred 200m)

---

## Flow Estimator (`utils/flow_estimator.py`)

**Sea-RAFT** (https://github.com/princeton-vl/SEA-RAFT) wrapper.

- **훈련 시에만 사용**, 추론/테스트에서는 불필요
- 완전 frozen (eval mode, `requires_grad=False`)
- 입력: RGB 0-1 normalized → 내부에서 0-255 변환
- 출력: flow `[B, 2, H, W]`, confidence `[B, 1, H, W]`
- 8의 배수로 padding 후 처리, 이후 crop
- `estimate_flow_batch`: T 프레임에 대해 T-1개 flow 쌍 생성

**설치 요구사항**:
```bash
# 1. Clone
git clone https://github.com/princeton-vl/SEA-RAFT.git third_party/SEA-RAFT/

# 2. Pretrained weights 다운로드
mkdir -p third_party/SEA-RAFT/models/
# 아래 중 하나로 다운로드 후 third_party/SEA-RAFT/models/ 에 저장:
#   - HuggingFace: MemorySlices/Tartan-C-T-TSKH-spring540x960-M
#   - Google Drive: https://drive.google.com/drive/folders/1YLovlvUW94vciWvTyLf-p3uWscbOQRWW

# 최종 경로:
# third_party/SEA-RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth
```

Docker는 `.:/app` volume mount를 사용하므로, **호스트에 설치하면 컨테이너에서 자동으로 접근 가능**.

**Sea-RAFT가 없으면 Onepiece 훈련 불가** (RuntimeError 발생). 테스트는 가능.

---

## Training (`train_onepiece.py`)

### Two-Phase Training Strategy

#### Phase 1 (Step 0 ~ `auto_transition_step`, default 5000)
**목표**: Metric alignment (scale/shift 수렴)

| Component | Status |
|-----------|--------|
| DINOv2 ViT-L | **Frozen** |
| DPT Head | **Frozen** |
| output_conv | **Frozen** |
| UnifiedGlobalMamba | **Trainable** |
| OnepieceMetricHead | **Trainable** |

- Trainable params: ~6.2M (UnifiedGlobalMamba + OnepieceMetricHead)
- LR: 1e-4 (onepiece params)
- Warmup: 500 steps (0.1 → 1.0 linear)
- Schedule: Warmup → Constant → Cosine decay (30% 시점부터)
- **Feature Consistency Loss skip**: DPT frozen → gradient 전파 불가 → Sea-RAFT 연산 자체를 건너뜀

#### Phase 2 (Step 5000+)
**목표**: Full video optimization (depth quality 향상)

| Component | Status |
|-----------|--------|
| DINOv2 ViT-L | **Frozen** (항상) |
| DPT Head | **Trainable** (LR 1/10) |
| output_conv | **Trainable** (LR 1/10) |
| UnifiedGlobalMamba | **Trainable** |
| OnepieceMetricHead | **Trainable** |

- DPT/output_conv params에 500-step warmup 적용 (LR 0 → 1/10 of base)
- Optimizer 재생성 (3개 param group: onepiece, dpt, output_conv)

### Phase Transition 로직

```python
# train_onepiece.py:608
if self.current_phase == 1 and step == self.auto_transition_step:
    self._transition_to_phase2()
```

Phase 2 전환 시:
1. `_configure_parameters_phase2()`: DPT + output_conv 파라미터 unfreeze
2. Optimizer 재생성 (param groups 분리)
3. Scheduler 재생성 (phase2-specific warmup lambda)
4. `_set_train_mode()`: DPT, output_conv를 train mode로 전환

### Data Format

Gear5 8-element batch format 사용:
```python
images, gt_depth, focal_lengths_canonical, focal_lengths_actual,
actual_valid_masks, fx_ratio, resize_ratio, dataset_idx = batch
```

- `images`: [B, T, 3, 518, 518] (video_length=8)
- `gt_depth`: inverse depth (1/m scale, normalized)
- GT → metric depth 변환: `gt_meters = 1.0 / gt_depth.clamp(min=1e-8)`

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
| Train datasets | mvs-synth, dynamicreplica, tartanair, pointodyssey, spring (Gear5 동일) |
| Val datasets | sintel, waymo_seg (Gear5 동일) |
| Validation freq | 1000 steps |
| Save freq | 5000 steps |
| Total iterations | 60001 |

### Validation Configuration (Gear5와 동일)

| Phase | Val vis sequences | max_val_batches | dataset_max_sequences |
|-------|-------------------|-----------------|----------------------|
| Phase 1 | sintel: [0,4,7], waymo_seg: [0-7] | unlimited | 없음 |
| Phase 2 | sintel: [0], waymo_seg: [0] | 16 | sintel: 8, waymo_seg: 8 |

### Visualization Schedule

| Step | Type |
|------|------|
| 0, 10, 50, 100 | Training vis |
| Every 250 steps | Training vis |
| Every 1000 steps | Validation vis |

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

테스트 시 canonical space에서 actual space로 변환:
```python
# fx_ratio = canonical_fx / actual_fx
# resize_ratio = resized_fx / actual_fx
de_canonical_ratio_metric = resize_ratio / fx_ratio
pred_depths_actual = metric_depth_canonical * de_canonical_ratio_metric
```

### Optimal Scale/Shift (Oracle)

각 frame에 대해 Least Squares로 최적 scale/shift 계산:
```python
# gt = opt_scale * pred + opt_shift
A = np.stack([pred_valid, np.ones_like(pred_valid)], axis=1)
result = np.linalg.lstsq(A, gt_valid, rcond=None)
opt_scale, opt_shift = result[0]
```

이를 통해 모델의 predicted scale/shift와 optimal scale/shift 간의 gap 분석 가능.

### Depth Range Analysis

3개 depth range별 metrics 분석:
- **0-10m**: 근거리
- **10-30m**: 중거리
- **30-70m**: 원거리

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
| Best frame | `best_frames/best_frame_seq{N}_{T}_absrel_{V}.png` (3x3 grid) |
| Worst frame | `worst_frames/worst_frame_seq{N}_{T}_absrel_{V}.png` (3x3 grid) |
| Video/GIF | Sequence animation (skip for urbansyn, unreal4k) |

### Best/Worst Frame Visualization Layout (3x3)

```
Row 1: Input Image | GT Depth | Pred Depth
Row 2: Scale/Shift Plot (over time) | Error Map | Metrics Panel
Row 3: Depth Distribution (colspan=2) | D_cls Plot
```

---

## Training Visualization (`utils/onepiece_visualization.py`)

### OnepieceVisualizer

Training/validation 중 3x3 grid 시각화 생성:

```
Row 1: Input Image | GT Depth (m) | Pred Depth (m)
Row 2: Valid Mask | Absolute Error | Metrics & Training Info
Row 3: Depth Distribution (colspan=2) | (empty/info)
```

**Sparse dataset 처리**: waymo, waymo_seg, nuscenes는 sparse depth GT를 가짐.
- GT density < 50%이면 sparse 모드 활성화
- Height mask: valid pixel이 10개 이상인 row만 포함
- Pred 시각화 시 GT-missing 영역도 표시 (pred valid 조건 하에)

---

## Configuration

### Config Variants (`configs/onepiece/`)

Gear5와 동일한 `--config-variant` / `--cls-layer` 패턴 지원.

| File | Variant | ViT | CLS dim | GAP dim | Unified dim | Batch |
|------|---------|-----|---------|---------|-------------|-------|
| `config_l.yaml` | l | ViT-L | 1024 | 256 | 1280 | 3 |
| `config_s.yaml` | s | ViT-S | 384 | 64 | 448→512 (projected) | 8 |

### CLS Layer Selection

`cls_layers` config로 multi-layer CLS token averaging 지원 (1-indexed, Gear5와 동일 패턴).

```
ViT-L: intermediate_layer_idx = [4, 11, 17, 23]
       Layer 1→block 4, Layer 2→block 11, Layer 3→block 17, Layer 4→block 23

ViT-S: intermediate_layer_idx = [2, 5, 8, 11]
       Layer 1→block 2, Layer 2→block 5, Layer 3→block 8, Layer 4→block 11
```

**Example**: `cls_layers: [2, 4]` (default)
- ViT-L: encoder_indices=[1,3] → blocks [11, 23] → average CLS from layer 11 and 23
- ViT-S: encoder_indices=[1,3] → blocks [5, 11] → average CLS from layer 5 and 11

**Example**: `--cls-layer 3,4`
- ViT-L: encoder_indices=[2,3] → blocks [17, 23]
- ViT-S: encoder_indices=[2,3] → blocks [8, 11]

### Config Sample (`config_l.yaml`)

```yaml
config_variant: l
load: configs/flashdepth-l/iter_10001.pth
cls_layers: [2, 4]              # Multi-layer CLS averaging

# Model
model:
  vit_size: "vitl"              # ViT-L (1024-dim CLS token)
  use_onepiece: true            # Enable Onepiece modules
  unified_mamba_layers: 2       # UnifiedGlobalMamba depth
  unified_mamba_d_state: 64     # SSM state dimension
  unified_mamba_d_conv: 4       # Conv kernel size

# Dataset
dataset:
  resolution: 'base'            # 518x518
  video_length: 8               # 8 frames
  train_datasets: [mvs-synth, dynamicreplica, tartanair, pointodyssey, spring]
  val_datasets: [sintel, waymo_seg]

# Training
training:
  batch_size: 3                 # Per GPU (ViT-L: 3, ViT-S: 8)
  workers: 4
  gradient_checkpointing: true
  total_iters: 40001
  val_freq: 1000
  save_freq: 5000
  lr:
    onepiece: 1.0e-4            # Base LR
    dpt: 1.0e-5                 # 1/10 of base (Phase 2)
    warmup_steps: 500

# Loss (1:1:0.01)
loss:
  log_l1_weight: 1.0
  tgm_weight: 1.0
  feat_cons_weight: 0.01       # Raw L2 스케일(~14) 보정
  use_log_space: true

# Scene Cut
scene_cut:
  tau: 0.05                     # Threshold
  k: 80                         # Sigmoid steepness

# Phase Transition
phase:
  auto_transition_step: 5000    # Phase 1 → Phase 2
  phase2_warmup_steps: 500      # DPT warmup

# Flow (training only)
flow:
  checkpoint: "third_party/SEA-RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth"
```

---

## Docker Commands

### Training

```bash
# Single GPU - ViT-L (default)
./run_docker.sh train_onepiece --config-variant l --batch-size 3 --gpu 0 \
    --results-dir train_results/onepiece/large/

# Single GPU - ViT-S
./run_docker.sh train_onepiece --config-variant s --batch-size 8 --gpu 0 \
    --results-dir train_results/onepiece/small/

# DDP (2 GPUs) - ViT-L with custom CLS layers
./run_docker.sh train_onepiece_ddp --config-variant l --batch-size 3 --ddp-gpus 1,2 \
    --cls-layer 3,4 --results-dir train_results/onepiece/large_cls34/

# DDP - ViT-S with scale-only mode
./run_docker.sh train_onepiece_ddp --config-variant s --batch-size 8 --ddp-gpus 1,2 \
    --no-shift --results-dir train_results/onepiece/small_noshfit/

# With WandB
WANDB_API_KEY=xxx ./run_docker.sh train_onepiece_ddp --config-variant l --batch-size 3 \
    --ddp-gpus 1,2 --results-dir train_results/onepiece/large/
```

### Testing

```bash
# Test ViT-L
./run_docker.sh test_onepiece --config-variant l \
    --gear-checkpoint train_results/onepiece/large/best.pth \
    --cls-layer 2,4 --gpu 0

# Test ViT-S
./run_docker.sh test_onepiece --config-variant s \
    --gear-checkpoint train_results/onepiece/small/best.pth \
    --cls-layer 2,4 --gpu 0
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--config-variant V` | Config variant: `l` (ViT-L) or `s` (ViT-S) | l |
| `--cls-layer L` | CLS layer selection (1-4, comma-separated) | 2,4 |
| `--batch-size N` | Batch size per GPU | 3 |
| `--workers N` | DataLoader workers | 8 |
| `--gpu ID` | Single GPU ID | 0 |
| `--ddp-gpus IDs` | DDP GPU IDs (e.g., "1,2") | "0,1" |
| `--results-dir PATH` | Results directory | train_results/results_1 |
| `--epochs N` | Total training iterations | 60001 |
| `--wandb BOOL` | Enable WandB logging | true |
| `--wandb-name NAME` | WandB experiment name | auto |
| `--no-shift` | Scale-only mode (shift=0) | false |
| `--frame-interval N` | Test visualization interval | 1 |
| `--vid-len N` | Test video length | 50 |

---

## Model Integration (`flashdepth/model.py`)

### Initialization (lines 107-140)

```python
if self.use_onepiece:
    cls_dim = self.pretrained.embed_dim  # 1024 (ViT-L) or 384 (ViT-S)
    gap_dim = dpt_dim                     # 256 (ViT-L) or 64 (ViT-S)
    unified_dim = cls_dim + gap_dim       # 1280 (ViT-L) or 448 (ViT-S)

    self.unified_global_mamba = UnifiedGlobalMamba(d_input=unified_dim, ...)
    # d_input=1280 → d_model=1280 (ViT-L, no projection)
    # d_input=448 → d_model=512 (ViT-S, auto-projected)
    self.onepiece_metric_head = OnepieceMetricHead(input_dim=unified_dim, dpt_dim=gap_dim)
    self.scene_cut_detector = SceneCutDetector(tau=0.05, k=80)
```

### `_get_intermediate_layers_with_cls` (single-pass CLS + features 추출)

DINOv2 encoder를 **1회만 호출**하여 intermediate features와 CLS token을 동시 추출.
(Gear5에서는 `get_intermediate_layers` + `forward_features` 2회 호출했던 문제 해결)

```python
def _get_intermediate_layers_with_cls(self, x, layer_indices, cls_layer_indices=None):
    """
    Returns:
        intermediate_features: DPT용 patch tokens (CLS stripped)
        cls_token: CLS token [B*T, embed_dim]
                   - cls_layer_indices가 주어지면 → 해당 layer들의 CLS 평균
                   - None이면 → 마지막 layer의 CLS만 사용
    """
```

CLS layer 선택 로직이 함수 내부에서 처리되므로, caller는 반환값 2개만 받으면 됨.

### Forward Pass (`forward_with_onepiece`)

7-step pipeline:

1. **DINOv2 Encoder** (frozen, single-pass): `_get_intermediate_layers_with_cls(video_flat, layer_indices, cls_layer_indices)` → `encoder_features` + `cls_tokens_flat`
2. **DPT Head** (frozen/Phase2): `encoder_features → dpt_features [B*T, dpt_dim, h, w]`
3. **GAP**: `dpt_features → gap_features [B*T, dpt_dim]`
4. **UnifiedGlobalMamba** (trainable): `[CLS;GAP] [B, T, unified_dim] → refined_global`
5a. **MetricHead** (trainable): `refined_global → scale, shift, gamma, beta`
5b. **FiLM**: `modulated = gamma * dpt_features + beta`
6. **Final Head** (frozen/Phase2): `modulated_features → relative_depth`
7. **Metric conversion**: `metric_depth = scale * (100/relative_depth) + shift`

### Return Dict

```python
{
    'relative_depth': [B, T, H, W],         # FiLM-modulated relative depth
    'metric_depth': [B, T, H, W],           # Final metric depth (meters)
    'scale': [B, T],                         # Per-frame scale
    'shift': [B, T],                         # Per-frame shift
    'dpt_features': [B, T, 256, h, w],      # For feature consistency loss
    'cls_tokens': [B, T, 1024],             # For scene cut detection
    'scene_cut_weights': [B, T-1],          # Temporal gating weights
    'd_cls': [B, T-1],                       # Cosine distances (logging)
}
```

---

## Parameter Count 요약

### ViT-L Variant

실측값 (training log 기준):
- Frozen: 335,315,649 (ViT + DPT + output_conv)
- **Phase 1 Trainable: 47,214,834** (UnifiedGlobalMamba + OnepieceMetricHead)

| Module | Parameters | Phase 1 | Phase 2 |
|--------|-----------|---------|---------|
| DINOv2 ViT-L | 304.37M | Frozen | Frozen |
| DPT Head (dpt_dim=256) | 30.62M | Frozen | **Trainable** |
| output_conv | 0.33M | Frozen | **Trainable** |
| UnifiedGlobalMamba (1280→1280) | 46.36M | **Trainable** | **Trainable** |
| OnepieceMetricHead (1280, 256) | 0.85M | **Trainable** | **Trainable** |
| SceneCutDetector | 0 | - | - |
| **Total** | **382.53M** | | |
| **Phase 1 trainable** | **47.21M** | | |
| **Phase 2 trainable** | **78.16M** | | |

### ViT-S Variant

| Module | Parameters | Phase 1 | Phase 2 |
|--------|-----------|---------|---------|
| DINOv2 ViT-S | ~22M | Frozen | Frozen |
| DPT Head (dpt_dim=64) | ~1.5M | Frozen | **Trainable** |
| output_conv | ~0.5M | Frozen | **Trainable** |
| UnifiedGlobalMamba (448→512, projected) | ~2.1M | **Trainable** | **Trainable** |
| OnepieceMetricHead (448, 64) | ~0.3M | **Trainable** | **Trainable** |
| SceneCutDetector | 0 | - | - |
| **Phase 1 total trainable** | **~2.4M** | | |
| **Phase 2 total trainable** | **~4.4M** | | |

### Architectural Option: CLS Dimension Reduction

현재 ViT-L에서 UnifiedGlobalMamba가 46.36M으로 전체 trainable params의 대부분을 차지.
근본 원인: Mamba 파라미터 수는 d_model²에 비례하며, CLS(1024)+GAP(256)=1280-dim이 d_model로 직접 들어감.

그런데 Mamba의 최종 출력은:
- Scale/Shift: 2개 scalar
- FiLM gamma/beta: 각 256개 = 512개 값
- **총 514개 출력값에 46M params는 과잉일 수 있음**

**변경 옵션:**

```
현재:   CLS(1024) + GAP(256) → [1280] → Mamba(d=1280, 2L, 46.36M) → 514 outputs
옵션 A: CLS(1024) → Linear(256) + GAP(256) → [512] → Mamba(d=512, 2L, ~5.4M) → 514 outputs
옵션 B: CLS(1024) → Linear(256) + GAP(256) → [512] → Mamba(d=512, 4L, ~10.8M) → 514 outputs
```

**Trade-offs:**
- CLS 1024→256 projection: semantic 정보 일부 손실 가능. 단, 최종 목적이 scale/shift 2개 + FiLM 512개 값 추출이므로 256-dim으로도 충분할 수 있음
- d_model 512: d_model²에 비례하므로 파라미터 ~6배 감소 (1280²/512² ≈ 6.25)
- Layer 수 4로 늘려도 ~10.8M으로 현재 46.36M 대비 ~77% 절감
- `_find_valid_mamba_dim(512)` = 512 (Mamba2-valid, projection 불필요)

**구현 시 변경점:**
1. `model.py`: `cls_proj = nn.Linear(cls_dim, gap_dim)` 추가, `unified_dim = gap_dim * 2`
2. `onepiece_modules.py`: `UnifiedGlobalMamba(d_input=512)`, `OnepieceMetricHead(input_dim=512)`
3. `config_l.yaml`: `cls_projection: true`, `unified_mamba_layers: 4` 등 옵션 추가

**미적용 상태** — 현재 구조(1280-dim)로 학습 결과 확인 후 필요 시 적용 예정.

---

## CLS Distance 실험 (`measure_cls_distance.py`)

SceneCutDetector의 tau/k 하이퍼파라미터 선정을 위한 사전 실험 스크립트.

### 실험 구성

1. **Adjacent frames (dt=1)**: 같은 시퀀스의 연속 프레임
2. **Far-apart frames (dt=10,25,50,100,200)**: 같은 시퀀스 내 먼 프레임
3. **Cross-sequence pairs**: 다른 scene 간 프레임 쌍

### 사용 방법

```bash
python measure_cls_distance.py
```

DINOv2 ViT-L encoder로 TartanAir 6개 scene (abandonedfactory, hospital, ocean, office, seasidetown, japanesealley)에서 CLS token을 추출하고 cosine distance 통계를 출력.

### 기대 결과

- Adjacent (dt=1): mean D_cls ≈ 0.001~0.005
- Far-apart (dt=200): max D_cls ≈ 0.03~0.04
- Cross-sequence: min D_cls ≈ 0.06~0.10
- **Gap 존재**: same-scene max < 0.05 < cross-scene min → tau=0.05 적합

---

## File Structure

```
flashdepth_claude/
├── flashdepth/
│   ├── model.py                    # forward_with_onepiece() (line 458)
│   ├── onepiece_modules.py         # UnifiedGlobalMamba, OnepieceMetricHead, SceneCutDetector
│   └── mamba.py                    # MambaBlock, InferenceParams
├── utils/
│   ├── onepiece_losses.py          # WarpFeatureConsistencyLoss, OnepieceCombinedLoss
│   ├── onepiece_visualization.py   # OnepieceVisualizer (3x3 grid)
│   ├── flow_estimator.py           # Sea-RAFT wrapper (frozen)
│   └── gear_losses.py              # LogL1Loss, TGMTemporalLoss (reused)
├── configs/
│   └── onepiece/
│       ├── config.yaml             # Hydra config (default, legacy)
│       ├── config_l.yaml           # ViT-L variant (CLS=1024, GAP=256, unified=1280)
│       └── config_s.yaml           # ViT-S variant (CLS=384, GAP=64, unified=448→512)
├── third_party/
│   └── SEA-RAFT/                   # Sea-RAFT optical flow (git clone, training only)
│       └── models/
│           └── Tartan-C-T-TSKH-spring540x960-M.pth  # Pretrained weights (수동 다운로드 필요)
├── train_onepiece.py               # OnepieceTrainer (Phase 1/2 auto-transition)
├── test_onepiece.py                # OnepieceTester (full evaluation + visualization)
├── measure_cls_distance.py         # CLS cosine distance 실험
└── run_docker.sh                   # Docker 실행 (train_onepiece, train_onepiece_ddp, test_onepiece)
```

---

## Hydra Override 주의사항

onepiece config (`config_l.yaml`, `config_s.yaml`)에 이미 정의된 키는 `+` prefix 없이 override:

```bash
# config에 이미 있는 키 → prefix 없이 (no_shift, cls_layers 등)
no_shift=false
cls_layers='[3,4]'

# config에 없는 키 → + prefix 필요 (results_dir, frame_interval 등)
+results_dir=train_results/onepiece/
+frame_interval=2
```

`+key=value`를 config에 이미 있는 키에 쓰면 Hydra 에러 발생:
`Could not append to config. An item is already at 'key'.`

---

## Known Limitations / Notes

1. **Sea-RAFT 의존성**: Feature Consistency Loss 계산에 Sea-RAFT 필수. 테스트는 필요 없음. Docker에서는 `.:/app` mount로 호스트의 `third_party/SEA-RAFT`가 자동 접근됨.
2. **FiLM은 channel-wise**: spatial-wise modulation이 아닌 channel-wise affine transform. 모든 (h,w) 위치에 동일한 gamma, beta 적용.
3. **Shift 범위 제한**: `0.1 * sigmoid()` = [0, 0.1]. 큰 shift가 필요한 scene에서 한계.
4. **Scene cut detection은 no_grad**: W_temporal은 gradient를 받지 않아, scene cut 판정 자체는 학습되지 않음.
5. **ViT-S Mamba2 dimension**: 448 (=384+64) is not Mamba2-valid → auto-projected to 512 via linear layers. 약간의 parameter 증가.
