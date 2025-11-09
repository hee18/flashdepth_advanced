# FlashDepth Gear: Feature-level Metric Depth Learning

**작성일**: 2025-10-02
**최종 업데이트**: 2025-10-27
**브랜치**: gear3
**목적**: Feature-level FiLM modulation을 통한 metric depth 학습

**학습 가능 파라미터**: **9.2M / 329M (2.81%)** ⭐
- Gear modules: 4.6M
- Mamba: 4.3M
- output_conv: 0.3M

## 주요 변경사항 (2025-10-27) ⭐⭐⭐⭐⭐

### 1. 통합 Config 구조 (L/S/Hybrid)
모든 Gear 모델(gear2/3/3_upgrade)에 대해 **3가지 variant config** 제공:

```
configs/
├── gear2/
│   ├── config_l.yaml       # Stage 1: ViT-L + Mamba (path_1 사용)
│   ├── config_s.yaml       # Stage 1: ViT-S + Mamba (path_3 사용)
│   └── config_hybrid.yaml  # Stage 2: ViT-S student + ViT-L teacher (2K resolution)
├── gear3/
│   ├── config_l.yaml
│   ├── config_s.yaml
│   └── config_hybrid.yaml
└── gear3_upgrade/
    ├── config_l.yaml
    ├── config_s.yaml
    └── config_hybrid.yaml
```

**Key Features**:
- **config_l**: `vit_size: "vitl"`, `mamba_in_dpt_layer: [3]` (path_1 사용)
- **config_s**: `vit_size: "vits"`, `mamba_in_dpt_layer: [1]` (path_3 사용)
- **config_hybrid**: Student (ViT-S) + Teacher (ViT-L), 2K resolution (mvs-synth, spring only)

### 2. Forward Pass 버그 수정 (Mamba 미사용 문제 해결) ⭐⭐⭐⭐⭐

**Critical Bug**: 모든 Gear 모델이 Mamba를 사용하지 않고 있었음!

**문제**:
```python
# WRONG (이전)
dpt_features = model.depth_head.get_forward_features(encoder_features, patch_h, patch_w)
# → Mamba 없이 DPT features만 반환
```

**해결**:
```python
# CORRECT (현재)
# 1. Initialize Mamba sequence
if hasattr(model, 'mamba'):
    model.mamba.start_new_sequence()  # ← 원본 FlashDepth에도 있음!

# 2. Use forward_with_mamba instead
dpt_output = model.depth_head.forward_with_mamba(
    encoder_features, patch_h, patch_w,
    temporal_layer=model.mamba_in_dpt_layer,
    mamba_fn=model.dpt_features_to_mamba,
    shape_placeholder=(B, T, None, h, w)
)  # Returns path_1 (or path_3 for ViT-S) with Mamba applied

# 3. Wrap in list for Gear module compatibility
path_1_modulated, importance_map, ... = model.gear3_head(
    patch_tokens, attention_weights, [dpt_output], patch_h, patch_w
)
```

**영향을 받은 파일**:
- ✅ train_gear2.py (training loop + validation loop + training viz)
- ✅ train_gear3.py (training loop + validation loop + training viz)
- ✅ train_gear3_upgrade.py (training loop + validation loop + training viz)
- ✅ test_gear2.py (warmup + test loop)
- ✅ test_gear3.py (warmup + test loop)
- ✅ test_gear3_upgrade.py (warmup + test loop)

### 3. run_docker.sh 업데이트

**새로운 옵션**:
```bash
--config-variant l|s|hybrid  # Gear config variant 선택
--nuscenes                   # nuScenes fine-tuning 모드 (Stage 3)
```

**사용 예시**:
```bash
# Stage 1: ViT-L로 학습
./run_docker.sh train_gear2_ddp --config-variant l

# Stage 1: ViT-S로 학습
./run_docker.sh train_gear2_ddp --config-variant s

# Stage 2: Hybrid 학습 (ViT-S student + ViT-L teacher, 2K)
./run_docker.sh train_gear2_ddp --config-variant hybrid

# Stage 3: nuScenes fine-tuning
./run_docker.sh train_gear2 --nuscenes
```

**자동 조정**:
- Hybrid variant: batch_size=1, workers=2, iterations=40001 (2K resolution)
- Stage 1 (L/S): batch_size=20, workers=8, iterations=40001 (518×518)

**DEPRECATED**:
- ~~`--phase 1|2|3`~~ → 대신 `--config-variant l|s|hybrid` + `--nuscenes` 사용

### 4. 원본 FlashDepth와의 차이

**변경 없음 (100% 동일)**:
- ✅ DINOv2 encoder (frozen)
- ✅ DPT decoder (frozen, refinenet + projects/resize)
- ✅ Mamba temporal processing (trainable)
- ✅ `forward_with_mamba()` 로직
- ✅ `model.mamba.start_new_sequence()` 호출

**추가된 것만**:
- ✅ Gear modules (gear2/gear3/gear3_upgrade heads)
- ✅ Config 파일 구조 (L/S/Hybrid variants)
- ✅ run_docker.sh 옵션

**핵심**: 원본 FlashDepth의 core 로직은 전혀 건드리지 않았음. Gear modules만 추가됨.

---

## 이전 변경사항 (2025-10-13)

- ❌ **Canonicalization 완전 제거** (불필요, raw inverse depth로 충분)
- ✅ **Importance map: Raw attention weights 직접 사용** (학습 불필요)
- ❌ **모든 regularization losses 제거** (importance map 학습 안함)
- ✅ **DPT output_conv만 학습**: Modulated features → depth 변환

**이전 변경사항 (2025-10-09)**:
- ✅ DepthVariancePseudoLabelLoss 추가 (GT depth variance → importance map pseudo-label)
- ❌ BimodalLoss 완전 제거 (continuous modulation 방해: binary 강제 → variance supervision과 충돌)
- ⚠️ EdgeAwareLoss 기본 비활성화 (variance supervision으로 충분)
- ✅ ContrastiveFGBGLoss weight 증가: 0.1 → 0.3 (FG/BG feature separation 강화)

---

## 목차

1. [개요](#개요)
2. [핵심 아이디어](#핵심-아이디어)
3. [아키텍처 설계](#아키텍처-설계)
4. [학습 전략](#학습-전략)
5. [손실 함수 및 Valid Depth Range](#손실-함수-및-valid-depth-range)
6. [사용 방법](#사용-방법)
7. [최적화 설정](#최적화-설정)
8. [기대 효과](#기대-효과)

---

## 개요

Gear는 FlashDepth에 **Feature-level Metric Injection**을 적용하여 metric depth 추정 성능을 향상시킵니다.
기존 GSP(Global Scale Predictor)와 달리, **DPT feature에 직접 FiLM-style modulation**을 적용합니다.

### 기존 방식 vs Gear

| 방식            | GSP (기존)              | Gear (현재)                  |
|----------------|------------------------|-------------------------------|
| **Metric 주입** | Depth map에 scale/shift | **DPT features에 modulation** |
|  **공간 변화**   |  전역 균일               | **Importance map 기반 spatial modulation** |
| **FG/BG 구분**  |  없음                   | **Separate FG/BG modulation** (gear3/upgrade만) |
| **학습 파라미터** |  ~0.5M                 | **~9.2M** (Gear + Mamba + output_conv) |

### Gear Variants

| Model | 특징 | Mamba Layer | Params |
|-------|-----|-------------|--------|
| **Gear2** | Simple modulation (no FG/BG) | path_1/3 | ~4.6M |
| **Gear3** | FG/BG separation (attention-based) | path_1/3 | ~4.6M |
| **Gear3 Upgrade** | Advanced FG/BG (3 methods) | path_1/3 | ~4.6M |

---

## 핵심 아이디어

### 1. FiLM-style Feature Modulation

**현재 구현**: DPT의 **path_1 (ViT-L) 또는 path_3 (ViT-S)만 modulate**

```python
# Spatial-adaptive modulation (Gear3)
gamma[x,y] = importance[x,y] × γ_fg + (1 - importance[x,y]) × γ_bg
beta[x,y] = importance[x,y] × β_fg + (1 - importance[x,y]) × β_bg
modulated_feature[x,y] = gamma[x,y] ⊙ feature[x,y] + beta[x,y]
```

**Mamba Layer 선택**:
- **ViT-L**: `mamba_in_dpt_layer: [3]` → path_1 (Layer 23+17+11+4, ALL fused)
- **ViT-S**: `mamba_in_dpt_layer: [1]` → path_3 (Layer 11+5+2, smaller features)

**왜 이 layer들인가?**:
- **path_1 (ViT-L)**: ALL layers fused → 가장 완전한 multi-scale feature
- **path_3 (ViT-S)**: ViT-S는 더 얕은 네트워크 → path_3가 충분한 fusion 제공

### 2. Attention-based Importance Map (학습 불필요!)

**핵심 변경**: ImportancePredictor 제거, **DINOv2 attention weights를 직접 importance map으로 사용**

```python
# gear3_modules.py - process_attention_to_importance()
def process_attention_to_importance(attention_weights, patch_h, patch_w):
    # 1. CLS→patch attention (semantic importance)
    cls_to_patch = attention_weights[:, :, 0, 1:]  # [B, num_heads, num_patches]

    # 2. Average over heads
    attn_scores = cls_to_patch.mean(dim=1)  # [B, num_patches]
    attn_map = attn_scores.reshape(B, 1, patch_h, patch_w)

    # 3. Remove register token (highest attention patch)
    # DINOv2 has 1 register patch with extreme attention - remove via 3×3 inpainting
    max_val = attn_map.max()
    outlier_mask = (attn_map == max_val)
    attn_smoothed = F.conv2d(attn_map, kernel_3x3, padding=1)  # Local average
    attn_map = torch.where(outlier_mask, attn_smoothed, attn_map)

    # 4. Percentile normalization (1-99) to [0, 1]
    # More robust than min-max: reduces sensitivity to outliers
    attn_p1 = torch.quantile(attn_map, 0.01)
    attn_p99 = torch.quantile(attn_map, 0.99)
    importance_map = (attn_map - attn_p1) / (attn_p99 - attn_p1 + 1e-8)
    importance_map = torch.clamp(importance_map, 0.0, 1.0)

    return importance_map  # [B, 1, H, W], range [0, 1]
```

**장점**:
- ❌ **ImportancePredictor 제거**: ~1.2M params 절약
- ✅ **DINOv2의 검증된 semantic attention 활용**: 추가 학습 불필요
- ✅ **Register token 제거**: Outlier 영향 최소화 (3×3 local inpainting)
- ✅ **Percentile normalization**: Min-max보다 robust (1-99 percentile)
- ✅ **Gradient flow 자동 보장**: No zero init tricks needed
- ✅ **메모리 효율**: Last block만 저장 (~11GB 절약)

---

## 아키텍처 설계

### Forward Pass 로직 (Mamba 사용)

**전체 파이프라인**:

```
Video Frame → DINOv2 (frozen) → Patch Tokens + Last Block Attention
                                      ↓
                    ┌─────────────────┴─────────────────┐
                    ↓                                   ↓
         Raw Attention → Importance Map    ForegroundBackgroundNetworks (Gear3)
         (No learning!)    [B,1,H,W]      (Attention-based Pooling)
                    ↓                                   ↓
                    ↓                      FG/BG Features [B,256]
                    ↓                   (Top attn → FG, Low attn → BG)
                    └─────────────────┬─────────────────┘
                                      ↓
                            ModulationNetworks (×4 layers)
                                      ↓
                          γ_fg, β_fg, γ_bg, β_bg [B,256]
                                      ↓
         DPT Features → forward_with_mamba() → Mamba-processed Features
                                      ↓
                            FeatureModulator → Modulated Features
                                      ↓
                     output_conv1/2 (trainable from scratch)
                                      ↓
                            Inverse Depth (100/m, 직접 출력)
```

**Critical: `forward_with_mamba()` 호출 순서**:

```python
# train_gear*.py, test_gear*.py 공통 패턴
# 1. Initialize Mamba sequence (매 video sequence 시작 시)
if hasattr(model, 'mamba'):
    model.mamba.start_new_sequence()  # ← 원본 FlashDepth에도 있음!

# 2. Frame-by-frame processing
for t in range(T):
    img_t = images[:, t]

    # 3. DINOv2 encoder (frozen)
    encoder_features = model.pretrained.get_intermediate_layers(
        img_t, model.intermediate_layer_idx[model.encoder]
    )

    # 4. DPT + Mamba (forward_with_mamba)
    dpt_output = model.depth_head.forward_with_mamba(
        encoder_features, patch_h, patch_w,
        temporal_layer=model.mamba_in_dpt_layer,  # [3] for ViT-L, [1] for ViT-S
        mamba_fn=model.dpt_features_to_mamba,
        shape_placeholder=(B, T, None, h, w)
    )  # Returns path_1 (ViT-L) or path_3 (ViT-S) with Mamba applied

    # 5. Gear modulation
    path_1_modulated, importance_map, ... = model.gear3_head(
        patch_tokens, attention_weights, [dpt_output], patch_h, patch_w
    )  # Note: [dpt_output] wrapping for list compatibility

    # 6. Output head
    out = model.depth_head.scratch.output_conv1(path_1_modulated)
    out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
    out = model.depth_head.scratch.output_conv2(out)
```

**주요 포인트**:
1. **`start_new_sequence()`**: 매 video sequence 시작 시 Mamba state 초기화
2. **`forward_with_mamba()`**: DPT refinement + Mamba temporal processing
3. **`[dpt_output]` wrapping**: Gear head는 list of features를 기대 (원래 [path_4, path_3, path_2, path_1])

---

## 학습 전략

### Training Stages (3-stage workflow)

#### Stage 1: ViT-L or ViT-S (Base Resolution, 518×518)

**목적**: FlashDepth 가중치로부터 Gear module + Mamba + output_conv 학습

**Dataset**: 5개 (mvs-synth, dynamicreplica, tartanair, pointodyssey, spring)

**Config**:
```bash
# ViT-L
./run_docker.sh train_gear2_ddp --config-variant l \
  --batch-size 20 --workers 8 --results-dir train_results/gear2_l

# ViT-S
./run_docker.sh train_gear2_ddp --config-variant s \
  --batch-size 20 --workers 8 --results-dir train_results/gear2_s
```

**학습 설정**:
- Batch size: 20 per GPU (effective 40 with DDP)
- Workers: 8
- Iterations: 40001
- Resolution: 518×518

#### Stage 2: Hybrid (High Resolution, 2K)

**목적**: Student (ViT-S) + Teacher (ViT-L) fusion, 고해상도 학습

**Dataset**: 2개 (mvs-synth, spring) - 2K 해상도만

**Config**:
```bash
./run_docker.sh train_gear2_ddp --config-variant hybrid \
  --results-dir train_results/gear2_hybrid
```

**학습 설정**:
- Batch size: 1 per GPU (자동 조정, 2K resolution)
- Workers: 2 (자동 조정)
- Iterations: 40001
- Resolution: 2K
- Teacher: Gear-L Stage 1 checkpoint (frozen)
- Student: Gear-S Stage 1 checkpoint → fine-tune

#### Stage 3: nuScenes Fine-tuning (Optional)

**목적**: Autonomous driving dataset에 fine-tuning

**Config**:
```bash
./run_docker.sh train_gear2 --nuscenes \
  --results-dir train_results/gear2_nuscenes
```

**학습 설정**:
- 이전 checkpoint 로드 (Stage 1 또는 Stage 2)
- nuScenes dataset
- Gear + Mamba + output_conv만 학습

### 파라미터 설정

| 모듈 | 학습 여부 | LR | 파라미터 수 | 비고 |
|------|----------|-----|------------|------|
| DINOv2 Encoder | ❌ Frozen | - | ~300M | FlashDepth weights |
| DPT projects/resize | ❌ Frozen | - | ~5M | FlashDepth weights |
| DPT refinenet | ❌ Frozen | - | ~15M | FlashDepth weights |
| **DPT output_conv** | ✅ **Train from scratch** | **5e-5** | ~0.3M | **로드 안함** |
| **Mamba** | ✅ **Train from scratch** | **1e-4** | ~4.3M | **로드 안함** |
| **Gear modules** | ✅ Train | **1e-4** | ~4.6M | 신규 |

**총 학습 가능**: ~9.2M (2.81%)
**이유**: Modulated features를 받는 모듈은 사전학습 가중치 불가

### 학습 설정 (최적화됨)

**Hardware**: 2× RTX A6000 (48GB each), 96 CPU cores, 503GB RAM

**Scheduler**: Cosine Annealing with Warmup
```
Warmup (0-10%):    1e-5 → 1e-4
Stable (10-30%):   1e-4
Decay (30-100%):   1e-4 → 1e-6
```

---

## 손실 함수 및 Valid Depth Range

### Primary Loss: Inverse Depth Loss (100/m scale)

```python
# Training loop (NO canonicalization!)
gt_depth_inverse = gt_depth * 100.0  # Dataloader gives 1/m, scale to 100/m

# Valid mask: 1.43 < inverse_depth (i.e., depth < 70m)
MIN_INVERSE_DEPTH = 100.0 / 70.0  # ≈ 1.43 (same as original FlashDepth)
valid_mask = (gt_inverse > MIN_INVERSE_DEPTH)

depth_loss = LogL1Loss(pred_inverse, gt_inverse, valid_mask)
```

### Regularization Losses: **모두 제거됨!** ❌

**핵심 변경 (2025-10-13)**:
- ✅ **Importance map = Raw attention weights** (학습 불필요)
- ❌ **모든 regularization losses 제거**:
  - DepthVariancePseudoLabelLoss ❌
  - EdgeAwareLoss ❌
  - ContrastiveFGBGLoss ❌

**이유**:
- Importance map은 DINOv2 attention weights를 직접 사용 → **학습 파라미터 없음**
- Regularization은 importance map 학습을 위한 것 → **불필요**
- FG/BG networks와 modulation networks만 학습

### Total Loss (현재)

```python
# 현재 (2025-10-13): Depth loss만 사용!
total_loss = depth_loss  # Log L1 loss on inverse depth

# Regularization losses 모두 제거됨
```

### Valid Depth Range: **70m 이하** (원본 FlashDepth와 동일)

**배경**:
- 원본 FlashDepth: 70m (KITTI/NYUv2 기준)
- Gear: **70m** (원본과 일관성 유지)

**적용**:
1. **Training loss**: `gt_inverse > 100/70 ≈ 1.43` (depth < 70m)
2. **Validation loss**: 동일 (70m threshold)
3. **Test metrics**: `0 < depth < 70m` 필터링 (지표 계산용)
4. **Visualization**: 제한 없음 (모든 depth 범위 표시, 단 지표는 70m 이하만)

### Valid Mask Logic (2025-11-07 업데이트) ⭐

**핵심 원칙**: Loss는 **GT valid 영역에서만** 계산하되, **극단적인 pred outlier만 제외**

**이전 방식 (잘못됨)**:
```python
# WRONG: Pred도 70m 이하여야 loss 계산
gt_valid_mask = (gt_depth_inverse_flat > MIN_INVERSE_DEPTH)  # GT < 70m
pred_valid_mask = (pred_depth_inverse_flat > MIN_INVERSE_DEPTH)  # Pred < 70m
valid_mask = gt_valid_mask & pred_valid_mask  # ← 모델이 loss 회피 가능! ❌
```

**문제점**:
- 모델이 70m 이상을 예측하면 해당 픽셀에서 loss가 계산되지 않음
- 모델이 나쁜 예측을 하면 오히려 loss가 줄어드는 역효과

**현재 방식 (올바름)**:
```python
# CORRECT: GT valid + Pred outlier filtering only
gt_valid_mask = (gt_depth_inverse_flat > MIN_INVERSE_DEPTH)  # GT < 70m

# Outlier mask: 극단적인 예측만 제외 (>200m는 학습 불안정)
MAX_DEPTH_OUTLIER = 200.0
MIN_INVERSE_OUTLIER = 100.0 / MAX_DEPTH_OUTLIER  # ≈ 0.5
pred_outlier_mask = (pred_depth_inverse_flat > MIN_INVERSE_OUTLIER)  # Pred < 200m

# Final mask: GT valid AND pred not extreme outlier
valid_mask = gt_valid_mask & pred_outlier_mask  # ✅
```

**장점**:
1. ✅ **GT valid 영역 모두 학습**: GT가 있는 모든 픽셀에서 loss 계산
2. ✅ **모델이 loss 회피 불가**: 나쁜 예측도 loss에 반영됨
3. ✅ **극단값만 제외**: >200m 예측만 필터링 (학습 안정성)
4. ✅ **합리적인 학습**: GT 70~200m 예측에도 패널티 (잘못된 예측)

**적용 위치**:
- ✅ `train_gear2.py`: train_step() line 940-948, validate() line 1118-1130
- ✅ `train_gear3.py`: train_step() line 912-920, validate() line 1086-1098
- ✅ `train_gear3_upgrade.py`: train_step() line 1031-1039, validate() line 1247-1259

**시각화 지표도 동일**:
- Validation 시 표시되는 MAE, AbsRel, δ1 등도 동일한 mask 사용
- `gt_valid_mask & pred_outlier_mask` 영역에서만 계산

---

## Canonical Space (선택사항)

### 개념 및 동기

**문제**: 서로 다른 초점거리(fx)를 가진 카메라로 촬영된 이미지들을 학습할 때, 같은 크기의 물체라도 fx에 따라 다른 metric depth로 해석되어야 합니다.

**해결책**: 모든 이미지를 **고정된 canonical fx 기준**으로 변환하여 학습합니다. 이를 통해 모델은 "canonical fx로 촬영했다면 몇 m로 보일까?"를 학습하게 됩니다.

### 핵심 원리

**Pinhole Camera 모델**:
```
pixel_size = fx × (object_size / depth)
```

**같은 이미지** (같은 pixel_size) 조건:
```
fx₁ / depth₁ = fx₂ / depth₂
depth₂ = depth₁ × (fx₂ / fx₁)
```

**예시**:
- fx=500으로 촬영한 3m 거리의 자동차
- 만약 fx=1000으로 촬영했다면? → **6m여야 같은 크기!**
- 계산: `depth_canonical = 3 × (1000/500) = 6m` ✅

**물리적 직관**:
- fx↓ (광각) → 물체가 작게 보임
- Canonical fx↑로 해석 → 같은 크기 얻으려면 물체가 더 멀어야 함
- **depth_canonical > depth_actual** (when fx_actual < CANONICAL_FX)

### 올바른 수식 (2025-11-07 수정)

**Critical Bug Fix**: 이전 구현에서 inverse depth canonicalization 수식이 **정반대**로 되어 있었습니다!

#### Metric Depth Canonicalization (올바름)
```python
depth_canonical = depth_actual * (CANONICAL_FX / fx_actual)
```

#### Inverse Depth Canonicalization (수정됨!)

**수학적 유도**:
```
inverse_canonical = 1 / depth_canonical
                  = 1 / [depth_actual × (CANONICAL_FX / fx_actual)]
                  = (fx_actual / CANONICAL_FX) × (1 / depth_actual)
                  = inverse_actual × (fx_actual / CANONICAL_FX)  ← 핵심!
```

**올바른 코드**:
```python
# GT depth from dataloader is inverse depth (1/m)
gt_depth_inverse_100 = gt_depth * 100.0  # Scale to 100/m

# Apply canonical space transformation
if self.config.get('use_canonical_space', False):
    CANONICAL_FX = 1000.0  # or from config
    fx_actual = focal_lengths.view(B, T, 1, 1, 1)

    # CORRECT: inverse_canonical = inverse_actual × (fx_actual / CANONICAL_FX)
    gt_depth_inverse_100 = gt_depth_inverse_100 * (fx_actual / CANONICAL_FX)  ✅
```

**틀린 코드 (이전)**:
```python
# WRONG (이전 구현 - 2025-11-07 이전)
gt_depth_inverse_100 = gt_depth_inverse_100 * (CANONICAL_FX / fx_actual)  ❌
# → 수식이 반대로 되어 학습이 불가능했음!
```

### 검증 예시

**Scenario**: fx=500, depth=4.5m → canonical fx=1000에서는?

| Step | 올바른 값 | 틀린 코드 (이전) |
|------|-----------|-----------------|
| 1. depth_canonical | 4.5 × (1000/500) = **9m** ✅ | - |
| 2. inverse_actual | 1/4.5 = 0.2222 | 0.2222 |
| 3. inverse_canonical | 0.2222 × (500/1000) = **0.1111** ✅ | 0.2222 × (1000/500) = 0.4444 ❌ |
| 4. 역변환 확인 | 1/0.1111 = **9m** ✅ | 1/0.4444 = **2.25m** ❌ |

**결과**: 틀린 코드는 9m여야 할 값을 2.25m로 계산 → 학습 불가능!

### 일반적인 실수

❌ **실수 1**: Metric depth 수식을 inverse depth에 그대로 적용
```python
# WRONG: Metric depth 수식을 inverse depth에 사용
inverse_canonical = inverse_actual * (CANONICAL_FX / fx_actual)  ❌
```

✅ **올바름**: Inverse depth는 역수이므로 비율도 역수
```python
# CORRECT: fx 비율을 inverse로
inverse_canonical = inverse_actual * (fx_actual / CANONICAL_FX)  ✅
```

❌ **실수 2**: "depth가 커지면 inverse depth도 커진다"
- ❌ **틀림**: depth와 inverse depth는 **반비례** 관계!
- ✅ **맞음**: depth↑ → inverse depth↓

❌ **실수 3**: Canonical fx가 크면 GT depth도 커진다고 착각
- ❌ **틀림**: fx는 카메라 파라미터, 실제 거리와 무관
- ✅ **맞음**: fx↑로 **해석**하면 같은 이미지에서 거리가 더 멀어 보임

### Visualization: resized_max 의미

**이전**: GT depth의 최대값만 표시
```
resized_fx: 500, resized_max: 4.5
```

**현재**: Canonical 70m threshold가 actual fx에서 몇 m인지 표시
```
resized_fx: 500, resized_max: 4.5, canon_70m→actual: 35.0m
```

**의미**:
- Canonical space에서 70m threshold (valid range)
- fx=500에서는 35m까지만 valid
- fx=2000에서는 140m까지 valid
- 공식: `actual_70m = 70 × (fx_actual / CANONICAL_FX)`

### 사용 방법

**Config 설정**:
```yaml
# Enable canonical space transformation
use_canonical_space: true
canonical_focal_length: 1000.0  # or resolution-dependent dict
```

**Resolution-dependent config**:
```yaml
canonical_focal_length:
  base: 500.0    # 518×518
  '2k': 1000.0   # 2K resolution
```

**학습 효과**:
- ✅ 다양한 fx 카메라 데이터를 일관되게 학습
- ✅ Test 시 임의의 fx에 대해 de-canonicalization 수행
- ✅ Metric depth 정확도 향상 (fx-invariant learning)

**주의사항**:
- ⚠️ 2025-11-07 이전 checkpoint는 **틀린 수식**으로 학습됨 → 재학습 필요!
- ⚠️ De-canonicalization은 올바르게 구현되어 있었음 (test 코드)
- ⚠️ Config에서 `use_canonical_space: true` 설정 필요 (기본값: false)

---

## 사용 방법

### Stage 1 학습 (ViT-L or ViT-S)

```bash
# ViT-L (추천)
./run_docker.sh train_gear2_ddp --config-variant l \
  --batch-size 20 --workers 8 \
  --results-dir train_results/gear2_l

# ViT-S (smaller/faster)
./run_docker.sh train_gear2_ddp --config-variant s \
  --batch-size 20 --workers 8 \
  --results-dir train_results/gear2_s
```

**Datasets** (순서: 원본 FlashDepth와 동일):
1. mvs-synth
2. dynamicreplica
3. tartanair
4. pointodyssey
5. spring

**Validation**: sintel, waymo

### Stage 2 학습 (Hybrid, 2K)

```bash
./run_docker.sh train_gear2_ddp --config-variant hybrid \
  --results-dir train_results/gear2_hybrid \
  --flashdepth-checkpoint train_results/gear2_s/best.pth
```

**Requirements**:
- Student checkpoint: Gear-S Stage 1 best model
- Teacher checkpoint: Gear-L Stage 1 best model (config에서 지정)

**Auto-adjustments**:
- Batch size: 1 per GPU (2K resolution)
- Workers: 2
- Datasets: mvs-synth, spring only

### Stage 3 학습 (nuScenes fine-tuning)

```bash
./run_docker.sh train_gear2 --nuscenes \
  --results-dir train_results/gear2_nuscenes \
  --flashdepth-checkpoint train_results/gear2_l/best.pth
```

### 테스트

```bash
# ViT-L
./run_docker.sh test_gear2 --config-variant l \
  --flashdepth-checkpoint train_results/gear2_l/best.pth \
  --results-dir test_results/gear2_l

# Hybrid
./run_docker.sh test_gear2 --config-variant hybrid \
  --flashdepth-checkpoint train_results/gear2_hybrid/best.pth \
  --results-dir test_results/gear2_hybrid
```

---

## 최적화 설정

### GPU 메모리 사용량

**Current (batch_size=20 per GPU, ViT-L)**:
- GPU 0: ~41GB / 48GB (**7GB 여유**)
- GPU 1: ~33GB / 48GB (15GB 여유)
- GPU utilization: **100%** ✅

**Hybrid (batch_size=1 per GPU, 2K)**:
- GPU 0/1: ~45GB / 48GB (OOM 방지)

### RAM 사용량

- Total: 503GB
- Used: ~36GB
- Shared memory: 24GB / 252GB (10%)
- **여유 충분** ✅

### CPU 사용량

- 사용률: ~14.5% (96 cores)
- Workers: 22 processes (2 GPUs × 8 + overhead)
- **병목 없음** ✅

---

## Config 파일 구조

### config_l.yaml (ViT-L, Stage 1)

```yaml
model:
  vit_size: "vitl"
  patch_size: 14
  mamba_in_dpt_layer: [3]  # path_1 사용 (ALL layers fused)
  use_mamba: true

training:
  batch_size: 20  # Per GPU
  workers: 8
  iterations: 40001
  gear2_lr: 1.0e-4  # or gear3_lr, gear3_upgrade_lr
  mamba_lr: 1.0e-4
  final_head_lr: 5.0e-5

dataset:
  resolution: 'base'  # 518×518
  train_datasets: [mvs-synth, dynamicreplica, tartanair, pointodyssey, spring]

hybrid_configs:
  use_hybrid: false

load: configs/flashdepth-l/iter_10001.pth
```

### config_s.yaml (ViT-S, Stage 1)

```yaml
model:
  vit_size: "vits"
  patch_size: 14
  mamba_in_dpt_layer: [1]  # path_3 사용 (smaller features)
  use_mamba: true

training:
  batch_size: 20
  workers: 8
  iterations: 40001
  gear2_lr: 1.0e-4
  mamba_lr: 1.0e-4
  final_head_lr: 5.0e-5

dataset:
  resolution: 'base'  # 518×518
  train_datasets: [mvs-synth, dynamicreplica, tartanair, pointodyssey, spring]

hybrid_configs:
  use_hybrid: false

load: configs/flashdepth-s/iter_10001.pth
```

### config_hybrid.yaml (ViT-S student + ViT-L teacher, Stage 2)

```yaml
model:
  vit_size: "vits"  # Student uses ViT-S
  patch_size: 14
  mamba_in_dpt_layer: [1]  # path_3
  use_mamba: true

training:
  batch_size: 4   # Smaller batch for 2K resolution
  workers: 8
  iterations: 40001
  gear2_lr: 1.0e-4  # Fine-tune
  mamba_lr: 1.0e-4
  final_head_lr: 5.0e-5
  fusion_lr: 1.0e-4  # Hybrid fusion (from scratch)

dataset:
  resolution: '2k'  # 2K resolution
  train_datasets: [mvs-synth, spring]  # Only high-res datasets

hybrid_configs:
  use_hybrid: true
  teacher_model_path: train_results/results_XX/gear2_l/best.pth  # Gear-L Stage 1
  teacher_resolution: 490
  layers_to_skip: [1,2,3]  # Only path_4 for fusion
  num_blocks: 4
  mlp_expand: 2
  num_heads: 2

load: train_results/results_XX/gear2_s/best.pth  # Gear-S Stage 1
```

---

## 파일 구조

```
flashdepth_claude/
├── flashdepth/
│   ├── gear2_modules.py          # Gear2 모듈
│   ├── gear3_modules.py          # Gear3 모듈
│   ├── gear3_upgrade_modules.py  # Gear3 Upgrade 모듈
│   ├── dinov2_layers/
│   │   └── attention.py          # Attention weights 최적화
│   └── original_dpt.py           # forward_with_mamba()
├── configs/
│   ├── gear2/
│   │   ├── config_l.yaml         # ViT-L Stage 1
│   │   ├── config_s.yaml         # ViT-S Stage 1
│   │   └── config_hybrid.yaml    # Hybrid Stage 2
│   ├── gear3/
│   │   ├── config_l.yaml
│   │   ├── config_s.yaml
│   │   └── config_hybrid.yaml
│   └── gear3_upgrade/
│       ├── config_l.yaml
│       ├── config_s.yaml
│       └── config_hybrid.yaml
├── train_gear2.py                # Gear2 training (Mamba 사용)
├── train_gear3.py                # Gear3 training (Mamba 사용)
├── train_gear3_upgrade.py        # Gear3 Upgrade training (Mamba 사용)
├── test_gear2.py                 # Gear2 testing (Mamba 사용)
├── test_gear3.py                 # Gear3 testing (Mamba 사용)
├── test_gear3_upgrade.py         # Gear3 Upgrade testing (Mamba 사용)
├── run_docker.sh                 # Docker runner (--config-variant, --nuscenes)
└── flashdepth_gear3.md           # 이 문서
```

---

## 요약

### Mamba 사용 버그 수정 (2025-10-27)

**Critical Bug**: 모든 Gear 모델이 `get_forward_features()`를 사용하여 **Mamba를 실행하지 않고 있었음**

**해결**:
1. ✅ `forward_with_mamba()` 사용 (DPT + Mamba)
2. ✅ `start_new_sequence()` 호출 (원본 FlashDepth에도 있음)
3. ✅ `[dpt_output]` wrapping (Gear head는 list를 기대)
4. ✅ 모든 train/test 파일 수정 완료

### Config 구조 (2025-10-27)

**3-stage workflow**:
1. **Stage 1**: ViT-L or ViT-S (518×518, 5 datasets)
2. **Stage 2**: Hybrid (2K, 2 datasets, student+teacher)
3. **Stage 3**: nuScenes fine-tuning (optional)

**사용법**:
```bash
# Stage 1
./run_docker.sh train_gear2_ddp --config-variant l

# Stage 2
./run_docker.sh train_gear2_ddp --config-variant hybrid

# Stage 3
./run_docker.sh train_gear2 --nuscenes
```

### 핵심 아키텍처 (2025-10-13)

```
DINOv2 (frozen) → Layer 4, 11, 17, 23
                           ↓
              DPT Progressive Fusion (frozen):
                path_4 = Layer 23 ONLY
                path_3 = Layer 23 + 17
                path_2 = Layer 23 + 17 + 11
                path_1 = Layer 23 + 17 + 11 + 4 (ALL FUSED) ⭐
                           ↓
              Last Block Attention → Raw Importance Map (no learning!)
                           ↓
                   FG/BG Networks (trainable, Gear3만)
                           ↓
               Modulation Networks (trainable)
                           ↓
         path_1 or path_3 → Mamba (trainable) → Feature Modulator (FiLM)
                           ↓
           output_conv1/2 (trainable from scratch)
                           ↓
              Inverse Depth (100/m scale, no canonicalization)
```

**핵심**:
- ✅ **ViT-L: path_1**, **ViT-S: path_3** modulate
- ✅ **Importance map = Raw attention** (학습 불필요, register token 제거)
- ✅ **FG/BG 분리 = Disjoint masks** (mean 기준, adaptive, Gear3만)
- ✅ **Mamba 사용 확인됨** (`forward_with_mamba()` + `start_new_sequence()`)
- ✅ **Depth loss만 사용** (regularization 제거)
- ✅ **학습 파라미터: 9.2M / 329M (2.81%)**

---

**Last Update**: 2025-10-27
**Branch**: gear3
**Developer**: hsy
