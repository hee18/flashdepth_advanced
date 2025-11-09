# train_gear3_upgrade.py 중요 체크리스트

**이 문서는 실수 방지를 위한 핵심 사항 정리입니다. 코드 수정 전 반드시 읽고 확인하세요!**

---

## 🚨🚨🚨 CRITICAL BUG - Canonicalization Formula (2025-11-07 수정) 🚨🚨🚨

### ⚠️ 치명적 버그: Inverse Depth Canonicalization 수식이 정반대! ⚠️

**영향**: 2025-11-07 이전에 `use_canonical_space: true`로 학습된 모든 모델은 **틀린 수식**으로 학습되었으며 **재학습 필요**합니다!

**버그 내용**:
```python
# ❌ 틀린 코드 (2025-11-07 이전) - 수식이 정반대!
gt_depth_inverse_100 = gt_depth_inverse_100 * (CANONICAL_FX / fx_actual)  ❌
# 결과: 학습이 불가능함! (GT가 엉뚱한 값으로 변환됨)

# ✅ 올바른 코드 (2025-11-07 수정)
gt_depth_inverse_100 = gt_depth_inverse_100 * (fx_actual / CANONICAL_FX)  ✅
```

**수학적 유도**:
```
Metric depth 변환:
  depth_canonical = depth_actual × (CANONICAL_FX / fx_actual)  ← 맞음

Inverse depth 변환 (역수!):
  inverse_canonical = 1 / depth_canonical
                    = 1 / [depth_actual × (CANONICAL_FX / fx_actual)]
                    = (fx_actual / CANONICAL_FX) × (1 / depth_actual)
                    = inverse_actual × (fx_actual / CANONICAL_FX)  ← 비율이 반대!
```

**예시로 검증** (fx=500, depth=4.5m → canonical fx=1000):

| Step | 올바른 값 | 틀린 코드 결과 |
|------|-----------|---------------|
| depth_canonical | 4.5 × (1000/500) = **9m** ✅ | - |
| inverse_actual | 1/4.5 = 0.2222 | 0.2222 |
| inverse_canonical | 0.2222 × (500/1000) = **0.1111** ✅ | 0.2222 × (1000/500) = 0.4444 ❌ |
| 역변환 | 1/0.1111 = **9m** ✅ | 1/0.4444 = **2.25m** ❌ |

**수정된 파일** (12 locations):
- ✅ `train_gear2.py` (lines 693, 850)
- ✅ `train_gear3.py` (lines 679, 827)
- ✅ `train_gear3_upgrade.py` (lines 727, 924)
- ✅ `test_gear2.py` (line 548)
- ✅ `test_gear3.py` (line 545)
- ✅ `test_gear3_upgrade.py` (line 614)

**주의사항**:
1. **De-canonicalization은 올바르게 구현되어 있었음** (test 코드의 prediction → metric 변환)
2. Config에서 `use_canonical_space: false`로 학습한 모델은 영향 없음
3. `use_canonical_space: true`로 학습한 모델만 재학습 필요

**자세한 설명**: `flashdepth_gear3.md`의 "Canonical Space" 섹션 참고

---

## 🚨🚨 CRITICAL FIX - Valid Mask Logic (2025-11-07 수정) 🚨🚨

### ⚠️ 중요한 수정: Training/Validation Loss 계산 시 Valid Mask 로직 변경 ⚠️

**핵심 원칙**: Loss는 **GT valid 영역에서만** 계산하되, **극단적인 pred outlier만 제외**

**문제**: 이전에는 `gt_valid_mask & pred_valid_mask`를 사용하여 **모델이 loss를 회피**할 수 있었습니다!

### ❌ 잘못된 방식 (2025-11-07 이전)

```python
# GT < 70m 영역
gt_valid_mask = (gt_depth_inverse_flat > MIN_INVERSE_DEPTH)  # GT < 70m

# Pred < 70m 영역
pred_valid_mask = (pred_depth_inverse_flat > MIN_INVERSE_DEPTH)  # Pred < 70m

# 둘 다 만족해야 loss 계산
valid_mask = gt_valid_mask & pred_valid_mask  # ← 문제!
```

**왜 문제인가?**
1. 모델이 70m 이상을 예측하면 해당 픽셀에서 **loss가 계산되지 않음**
2. 모델이 **나쁜 예측을 하면 오히려 loss가 줄어듦** (역효과!)
3. 모델이 학습 회피 가능 (70m 이상 예측으로 loss 무시)

**예시**:
- GT: 50m (valid)
- Pred: 80m (잘못된 예측)
- 결과: `pred_valid_mask = False` → **loss 계산 안됨** ❌
- 기대: 50m vs 80m 차이에 대해 패널티를 주어야 함! ✅

### ✅ 올바른 방식 (2025-11-07 수정)

```python
# GT < 70m 영역 (학습 대상 영역)
gt_valid_mask = (gt_depth_inverse_flat > MIN_INVERSE_DEPTH)  # GT < 70m

# Pred outlier 필터링: 극단적인 예측만 제외 (>200m는 학습 불안정)
MAX_DEPTH_OUTLIER = 200.0
MIN_INVERSE_OUTLIER = 100.0 / MAX_DEPTH_OUTLIER  # ≈ 0.5 (100/m 단위)
pred_outlier_mask = (pred_depth_inverse_flat > MIN_INVERSE_OUTLIER)  # Pred < 200m

# Final mask: GT valid AND pred not extreme outlier
valid_mask = gt_valid_mask & pred_outlier_mask  # ✅
```

**왜 올바른가?**
1. ✅ **GT valid 영역 모두 학습**: GT가 있는 모든 픽셀에서 loss 계산
2. ✅ **모델이 loss 회피 불가**: 나쁜 예측(70~200m)도 loss에 반영됨
3. ✅ **극단값만 제외**: >200m 예측만 필터링 (학습 안정성 확보)
4. ✅ **합리적인 학습**: GT 50m vs Pred 80m → 패널티 부여 ✅

**예시 (수정 후)**:
- GT: 50m (valid)
- Pred: 80m (잘못된 예측, 하지만 <200m)
- 결과: `gt_valid_mask = True`, `pred_outlier_mask = True` → **loss 계산됨** ✅
- 효과: 모델이 80m 예측에 대해 패널티를 받고 개선됨!

### 수정된 파일 (6 locations)

**Training 함수**:
- ✅ `train_gear2.py`: train_step() lines 940-948, validate() lines 1118-1130
- ✅ `train_gear3.py`: train_step() lines 912-920, validate() lines 1086-1098
- ✅ `train_gear3_upgrade.py`: train_step() lines 1031-1039, validate() lines 1247-1259

### 수정 전/후 비교

| Scenario | GT | Pred | 이전: gt&pred valid | 현재: gt valid & pred outlier | 비고 |
|----------|-----|------|---------------------|------------------------------|------|
| Normal | 50m | 45m | ✅ Loss 계산 | ✅ Loss 계산 | 정상 |
| Bad pred | 50m | 80m | ❌ Loss 안함 | ✅ Loss 계산 | **핵심 차이!** |
| Very bad | 50m | 250m | ❌ Loss 안함 | ❌ Loss 안함 (outlier) | 극단값 제외 |
| Out of range | 90m | 45m | ❌ Loss 안함 (GT invalid) | ❌ Loss 안함 (GT invalid) | GT 범위 밖 |

### 추가 정보

**Threshold 선택 이유**:
- **70m (GT valid)**: 원본 FlashDepth와 일치, 학습 데이터 범위
- **200m (Pred outlier)**: 극단적으로 큰 예측만 제외 (학습 안정성)
  - 70~200m 예측: 나쁜 예측이지만 loss 계산 (학습 개선)
  - >200m 예측: 극단적 outlier, gradient explosion 방지

**Validation metrics도 동일**:
- Validation 시 표시되는 MAE, AbsRel, δ1 등도 동일한 mask 사용
- `gt_valid_mask & pred_outlier_mask` 영역에서만 계산
- 시각화 지표도 합리적으로 계산됨

**자세한 설명**: `flashdepth_gear3.md`의 "Valid Mask Logic" 섹션 참고

---

## 🚨 가장 흔한 실수들

### 1. Dataset 필터링 로직 반대로 이해 ❌
```python
# ❌ 잘못된 코드 (사용할 씬을 반환하면 안됨!)
def get_filter_scenes(self, split):
    val_scenes = ['segment-A', 'segment-B', ...]  # 사용하고 싶은 8개
    return val_scenes  # ← 이것들이 제외됨! 나머지 38개가 사용됨!

# ✅ 올바른 코드 (제외할 씬을 반환)
def get_filter_scenes(self, split):
    val_scenes_to_use = ['segment-A', 'segment-B', ...]  # 사용하고 싶은 8개
    scenes_to_exclude = [s for s in all_scenes if s not in val_scenes_to_use]
    return scenes_to_exclude  # 이것들을 제외 → 8개만 사용!
```

**이유:**
- `base_dataset_pairs.py`의 `_build_pairs()`는 `if item in filter_scenes: continue`로 동작
- filter_scenes = "필터링(제외)할 씬들"
- **사용할 씬이 아니라 제외할 씬을 반환해야 함!**

### 2. 모델 Output 단위 착각 ❌
```python
# ❌ 잘못된 코드 (절대 하지 마세요!)
pred_depth_inverse = out * 100.0  # 이미 100/m 단위인데 또 곱함!

# ✅ 올바른 코드
pred_depth_inverse = out  # 모델 output은 이미 100/m 단위!
```

**이유:**
- 모델은 GT (100/m 단위)로 학습되므로 output도 자동으로 100/m 단위
- GT: `gt_depth_inverse_100 = gt_depth * 100.0` (1/m → 100/m)
- 모델은 이 GT에 맞춰 학습 → output도 100/m 단위로 나옴
- **Training과 Validation 모두 동일하게 적용!**

### 3. Validation에서 't' 변수 미정의 ❌
```python
# ❌ 잘못된 코드
focal_lengths[:, t:t+1]  # t가 validation context에 없음!

# ✅ 올바른 코드
focal_lengths[:, 0:1]  # 첫 번째 프레임 명시적으로 지정
```

**이유:**
- Validation은 전체 sequence를 한번에 처리 (loop 없음)
- Visualization은 첫 번째 프레임(t=0)만 사용
- 변수 `t`는 존재하지 않음!

### 4. GT Depth 단위 혼동 ❌
```python
# Dataloader → Training 흐름
gt_depth (dataloader)          # 1/m 단위 (inverse depth)
    ↓
gt_depth_inverse_100 = gt_depth * 100.0  # 100/m 단위로 변환
    ↓
gt_depth_inverse_100 *= (CANONICAL_FX / fx_actual)  # Canonical space 변환
    ↓
Loss(pred, gt_depth_inverse_100)  # 둘 다 100/m, canonical space
```

---

## 📋 핵심 처리 플로우

### Training Step (train_step)

```python
# 1. GT Depth 준비
gt_depth_inverse_100 = gt_depth * 100.0  # 1/m → 100/m

# 2. Canonical Space 변환 (use_canonical_space=True)
if self.config.get('use_canonical_space', False):
    CANONICAL_FX = self._get_canonical_focal_length()  # base: 500.0, 2k: 1500.0
    fx_actual = focal_lengths.view(B, T, 1, 1, 1)
    gt_depth_inverse_100 = gt_depth_inverse_100 * (CANONICAL_FX / fx_actual)

# 3. Forward Pass
images_flat = rearrange(images, 'b t c h w -> (b t) c h w')  # B*T 로 flatten
encoder_features = model.pretrained.get_intermediate_layers(...)
dpt_features = model.depth_head.get_forward_features(...)
path_1_modulated, ... = model.gear3_upgrade_head(...)  # Gear3 modulation
path_1_temporal = model.dpt_features_to_mamba(...)  # Mamba temporal modeling
out = model.depth_head.scratch.output_conv2(...)

# 4. Prediction (이미 100/m 단위!)
pred_depth_inverse = out  # ← 100 곱하지 않음!

# 5. Loss 계산
MIN_INVERSE_DEPTH = 100.0 / 70.0  # >70m 필터링 (warmup: 200m)
valid_mask = (gt > MIN_INVERSE_DEPTH) & (pred > MIN_INVERSE_DEPTH)
loss = loss_fn(pred_depth_inverse_flat, gt_depth_inverse_flat, valid_mask)
```

### Validation (validate)

```python
# Training과 동일한 forward pass
# 차이점:
# 1. 전체 sequence 한번에 처리 (loop 없음)
# 2. 첫 프레임만 visualization
# 3. @torch.no_grad() 데코레이터

# Visualization용 첫 프레임 추출
pred_depth_inverse_seq = rearrange(pred_depth_inverse, '(b t) 1 h w -> b t 1 h w', b=B, t=T)
pred_depth_inverse_vis = pred_depth_inverse_seq[:, 0]  # 첫 프레임
focal_lengths_vis = focal_lengths[:, 0:1]  # ← t 아님!
```

---

## 🔧 모델 구조 & 학습 전략

### Frozen vs Trainable

| 모듈 | 상태 | 이유 |
|------|------|------|
| DINOv2 Encoder | ❄️ Frozen | Pre-trained 유지 |
| DPT Refinement | ❄️ Frozen | Pre-trained 유지 |
| Mamba | 🔥 Trainable | Modulated input 받음 |
| output_conv1/2 | 🔥 Trainable | Modulated features 받음 |
| gear3_upgrade_head | 🔥 Trainable | Metric head (새로 학습) |

### Phase별 학습 전략

**Phase 1 (518×518):**
- 5개 dataset 학습
- DINOv2 + DPT만 checkpoint에서 로드
- Mamba, output_conv, gear3_head: **scratch부터 학습**

**Phase 2 (2K resolution):**
- MVS-Synth, Spring 학습
- Phase 1 checkpoint 로드 (Gear modules + Mamba 유지)
- **ViT-DPT만 Hybrid weights로 덮어쓰기**
- Gear modules는 계속 fine-tune

**Phase 3 (2K resolution):**
- nuScenes fine-tune
- Phase 2 checkpoint 전체 로드

---

## 🎯 Canonical Space 정규화 (2025-11-07 수정)

### 목적
다양한 카메라 intrinsics를 하나의 기준으로 통일

### 올바른 수식 ✅
```python
# Inverse depth canonical 변환 (수정됨!)
inverse_canonical = inverse_actual × (fx_actual / CANONICAL_FX)  ✅

# Metric depth 관계 (참고용)
depth_canonical = depth_actual × (CANONICAL_FX / fx_actual)  ✅
```

**주의**: Inverse depth는 metric depth의 역수이므로 **비율도 역수**입니다!

### 물리적 의미
```
같은 이미지 크기 조건:
  fx₁ / depth₁ = fx₂ / depth₂
  depth₂ = depth₁ × (fx₂ / fx₁)

예시: fx=500, depth=3m → fx=1000으로 본다면?
  depth_canonical = 3 × (1000/500) = 6m  ← 더 멀어야 같은 크기!
```

### Config 설정
```yaml
use_canonical_space: true
canonical_focal_length:
  base: 500.0   # 518×518 (또는 1000.0)
  '2k': 1500.0  # 2K resolution
```

### 예시 (올바른 계산)
```python
# Spring dataset
fx_actual = 588.6  # 518×518 기준
CANONICAL_FX = 500.0
scaling = 588.6 / 500.0 = 1.1772  ✅  (이전: 0.8495 ❌)

# 10m 물체
gt_inverse_100 = 10.0  # 100/m
gt_canonical = 10.0 × 1.1772 = 11.772  ✅  (이전: 8.495 ❌)
```

### 검증 방법
```python
# fx=500, depth=4.5m를 canonical fx=1000으로 변환하면?
# Expected: 9m (4.5 × 1000/500)

# Inverse depth로 계산:
inverse_actual = 1/4.5 = 0.2222
inverse_canonical = 0.2222 × (500/1000) = 0.1111  ✅
depth_back = 1/0.1111 = 9m  ✅  (정확!)

# 틀린 수식으로 계산:
inverse_canonical_wrong = 0.2222 × (1000/500) = 0.4444  ❌
depth_back_wrong = 1/0.4444 = 2.25m  ❌  (완전히 틀림!)
```

---

## 🎨 Visualization 주의사항

### Training Visualization
```python
# 첫 번째 샘플, 첫 번째 프레임 사용
sample_batch = (
    images[:1, :1].float().cpu(),        # [1, 1, C, H, W]
    gt_depth_metric[:1].float().cpu(),   # [1, 1, H, W]
    dataset_idx,
    focal_lengths[:1, :1].float().cpu()  # [1, 1]
)
```

### Validation Visualization
```python
# 전체 sequence 처리 후 첫 프레임만 사용
sample_batch = (
    img_t_resized.unsqueeze(1).float().cpu(),  # [B, 1, C, H, W]
    gt_depth_metric.unsqueeze(1),              # [B, 1, H, W]
    dataset_idx,
    focal_lengths[:, 0:1].float().cpu()  # ← [:, t:t+1] 절대 금지!
)
```

### Metric Depth 변환 (Visualization용)
```python
# Inverse depth (100/m) → Metric depth (m)
pred_depth_metric = 100.0 / (pred_depth_inverse.float() + 1e-8)
gt_depth_metric = 100.0 / (gt_depth_inverse_100.float() + 1e-8)

# Float32로 변환 (BFloat16 → Float32) 필수!
pred_depth_metric = pred_depth_metric.float().cpu()
gt_depth_metric = gt_depth_metric.float().cpu()
```

### 🔥 Validation Metrics & Valid Mask 처리 (2025-01-06 수정)

**핵심 원칙**: 원본 FlashDepth는 **GT valid만** 사용하여 metrics 계산!

**❌ 과거의 잘못된 구현 (gear3_upgrade_visualization.py:157-160)**
```python
if is_sparse:
    valid_mask_metrics = canonical_pred_valid  # 완전히 거꾸로!
```

**✅ 올바른 구현 (모든 visualization 파일 통일)**
```python
# 1. Metrics 계산용 마스크: GT valid + Pred outlier filtering
MAX_DEPTH_OUTLIER = 200.0  # 200m 이상 예측값은 아웃라이어로 필터링
pred_outlier_mask = (pred_depth_frame > 0) & (pred_depth_frame < MAX_DEPTH_OUTLIER)
valid_mask_metrics = canonical_gt_valid & pred_outlier_mask

# 2. GT Depth 시각화용 마스크: GT valid만 사용
valid_mask_gt_vis = canonical_gt_valid

# 3. Pred Depth 시각화용 마스크: Pred valid 사용 (모델이 예측한 범위)
valid_mask_pred_vis = canonical_pred_valid

# 4. Valid Mask 시각화: GT valid만 표시 (실제 평가 영역)
valid_mask_vis = canonical_gt_valid
```

**핵심 개념**:
1. **Metrics 계산**: GT valid 영역에서 계산하되, pred outlier(>200m)는 제외
   - 이유: 학습에 아웃라이어가 영향 주면 안됨
   - Sparse 데이터셋은 자연스럽게 GT valid 영역이 작음 (예: Waymo_seg)

2. **GT Depth 시각화**: `canonical_gt_valid` 사용
   - Sparse 데이터셋: sparse points만 표시
   - Dense 데이터셋: 전체 valid 영역 표시

3. **Pred Depth 시각화**: `canonical_pred_valid` 사용
   - 모델이 예측한 전체 범위 표시 (sparse/dense 관계없이)

4. **Valid Mask 시각화**: `canonical_gt_valid` 사용
   - 실제 metrics 계산되는 영역 표시 (GT가 있는 곳)
   - 모든 데이터셋 동일하게 적용 (sparse/dense 구분 없음)

**적용된 파일**:
- `utils/gear2_visualization.py` (lines 98-125, 243-250)
- `utils/gear3_visualization.py` (lines 110-139, 173-217, 283-288)
- `utils/gear3_upgrade_visualization.py` (lines 154-176, 211-253)

**참고**: 원본 FlashDepth `/FlashDepth/utils/eval_metrics/metrics.py:62`
```python
valid_mask = (gt_depth > 0) & gt_valid_pixel_mask  # GT valid만 사용!
```

---

## 🧮 Loss 계산 세부사항

### Valid Mask 생성
```python
# Warmup: 처음 100 step
if self.global_step < 100:
    MIN_INVERSE_DEPTH = 100.0 / 200.0  # 200m threshold
else:
    MIN_INVERSE_DEPTH = 100.0 / 70.0   # 70m threshold

# GT와 Pred 모두 유효해야 함
gt_valid_mask = (gt_depth_inverse_flat > MIN_INVERSE_DEPTH)
pred_valid_mask = (pred_depth_inverse_flat > MIN_INVERSE_DEPTH)
valid_mask = gt_valid_mask & pred_valid_mask
```

### Loss Function
```python
# LogL1Loss 사용
loss = self.loss_fn(
    pred_depth_inverse_flat.float(),  # [B*T, H, W]
    gt_depth_inverse_flat.float(),    # [B*T, H, W]
    valid_mask.float()                # [B*T, H, W]
)
```

---

## 🔄 Optimizer & Scheduler

### Optimizer
```python
# 모든 trainable modules 동일한 LR
base_lr = 1e-4  # config: gear3_lr

param_groups = [
    {'params': gear3_params, 'lr': base_lr},
    {'params': mamba_params, 'lr': base_lr},
    {'params': output_conv_params, 'lr': base_lr}
]

optimizer = torch.optim.AdamW(
    param_groups,
    betas=[0.9, 0.95],
    weight_decay=1e-6
)
```

### Scheduler
```python
# Warmup → Stable → Cosine Decay
warmup_steps = 1000  # 0.1x → 1x
decay_start = int(total_steps * 0.3)  # 30% 지점부터 decay
# Cosine decay: 1.0 → 0.01
```

### Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
```

---

## 📂 Dataset 필터링 로직 (매우 중요!)

### ⚠️ `get_filter_scenes()` 함수의 반대 로직

**절대 착각하지 마세요!** `get_filter_scenes()`는 사용할 씬이 아니라 **제외할 씬**을 반환합니다!

```python
# base_dataset_pairs.py의 _build_pairs() 함수
def _build_pairs(self):
    all_scenes = self.get_all_scenes(scenes_path)
    filter_scenes = self.get_filter_scenes(self.split)  # 제외할 씬들

    for item in all_scenes:
        if item in filter_scenes:
            continue  # ← filter_scenes에 있으면 건너뜀(제외함)!
```

### ❌ 잘못된 구현 (과거 버전)

```python
def get_filter_scenes(self, split):
    if split == 'val':
        # 이렇게 하면 안됨! 이 8개가 제외됨!
        val_scenes = ['segment-A', 'segment-B', ...]  # 사용하고 싶은 8개
        return val_scenes  # ← 이것들이 제외됨!
```

**결과**: 8개 씬은 건너뛰고, 나머지 38개를 사용 😱

### ✅ 올바른 구현

```python
def get_filter_scenes(self, split):
    all_scenes = self.get_all_scenes(self.get_scenes_path())

    if split == 'val':
        # 사용하고 싶은 8개 정의
        val_scenes_to_use = [
            'segment-10017090168044687777_6380_000_6400_000',
            'segment-10023947602400723454_1120_000_1140_000',
            'segment-1005081002024129653_5313_150_5333_150',
            'segment-10061305430875486848_1080_000_1100_000',
            'segment-10072140764565668044_4060_000_4080_000',
            'segment-10072231702153043603_5725_000_5745_000',
            'segment-10075870402459732738_1060_000_1080_000',
            'segment-10094743350625019937_3420_000_3440_000',
        ]
        # 나머지를 제외 목록으로 반환
        scenes_to_exclude = [s for s in all_scenes if s not in val_scenes_to_use]
        return scenes_to_exclude  # 이것들을 제외 → 8개만 사용됨!
```

### 💡 핵심 이해

| 함수 반환 | 실제 동작 | 결과 |
|----------|----------|------|
| A, B, C | A, B, C를 **제외** | D, E, F만 사용 |
| D, E, F | D, E, F를 **제외** | A, B, C만 사용 |

**기억하세요**: `get_filter_scenes()` = "이것들을 **걸러내라**(제외하라)"

### 🔧 디버깅 방법

캐시 파일 문제로 인해 씬 개수가 이상할 때:

```bash
# 1. 캐시 삭제
rm dataloaders/pairs_cache/waymo_seg_pairs.pkl

# 2. 디버그 로그 확인
# - "Total scenes available: 46"
# - "Scenes to filter out: 38"  ← 38개를 제외
# - "Will process 8 scenes"     ← 8개만 사용
```

### 📝 Waymo Validation 시퀀스 고정

**Waymo/Waymo_seg validation은 항상 정확히 이 8개만 사용:**

```python
segment-10017090168044687777_6380_000_6400_000
segment-10023947602400723454_1120_000_1140_000
segment-1005081002024129653_5313_150_5333_150
segment-10061305430875486848_1080_000_1100_000
segment-10072140764565668044_4060_000_4080_000
segment-10072231702153043603_5725_000_5745_000
segment-10075870402459732738_1060_000_1080_000
segment-10094743350625019937_3420_000_3440_000
```

- `waymo_dataset.py`: `get_filter_scenes()` 수정 필요
- `waymo_segmentation_dataset.py`: `_load_sequences()` 직접 필터링 (다른 방식)

---

## 🔍 디버깅 팁

### Step 0-4: Debug Logging
```python
if self.global_step < 5:
    self.logger.info(f"DEBUG - Raw GT: min={gt_depth.min():.4f}")
    self.logger.info(f"DEBUG - After 100x: min={gt_depth_inverse_100.min():.4f}")
    self.logger.info(f"DEBUG - Pred: min={pred.min():.4f}, mean={pred.mean():.4f}")
```

### Validation Loss가 inf인 경우
1. **'t' 변수 에러 확인** → focal_lengths 접근 실패
2. **Valid pixels 확인** → "Total batches with valid pixels: 0"
3. **Prediction 단위 확인** → 100 곱했는지 확인 (절대 곱하면 안됨!)

### NaN 발생시
1. Prediction clamping 확인: `torch.clamp(pred, min=1e-3, max=1e4)`
2. Division by zero 방지: `100.0 / (pred + 1e-8)`
3. Valid mask가 비어있지 않은지 확인

---

## 📝 코드 수정시 체크리스트

### 새 기능 추가시
- [ ] Training과 Validation 모두 수정했는가?
- [ ] 단위 변환 (1/m, 100/m, m) 올바른가?
- [ ] Visualization용 변수 추출시 `[:, 0]` 사용했는가?
- [ ] BFloat16 → Float32 변환 (CPU 이동시)했는가?
- [ ] DDP wrapping 고려했는가? (`model.module`)

### Bug Fix시
- [ ] Training과 Validation 동일한 수정했는가?
- [ ] 다른 train_gear*.py 파일도 수정했는가?
- [ ] Validation loss inf 문제 해결했는가?

### Validation 수정시
- [ ] `t` 변수 사용하지 않았는가?
- [ ] 첫 프레임 명시적으로 `[:, 0]` 또는 `[:, 0:1]` 사용했는가?
- [ ] 전체 sequence 처리 (loop 없음) 유지했는가?

### Dataset 필터링 수정시
- [ ] `get_filter_scenes()`가 **제외할** 씬을 반환하는지 확인했는가?
- [ ] 사용하고 싶은 씬이 아니라 **제외할 씬** 목록을 반환했는가?
- [ ] 캐시 파일을 삭제했는가? (`rm dataloaders/pairs_cache/*.pkl`)
- [ ] 디버그 로그에서 씬 개수가 예상과 일치하는지 확인했는가?

---

## 🚀 마지막 확인사항

**코드 수정 후 반드시 확인:**

```bash
# 1. Training 시작 로그 확인
# - GT min/max 값이 합리적인가? (100/m 단위)
# - Pred min/max 값이 GT와 비슷한 범위인가?

# 2. Validation 로그 확인
# - "name 't' is not defined" 에러 없는가?
# - "Total batches with valid pixels: 0" 아닌가?
# - Validation loss가 inf 아닌가?

# 3. Visualization 확인
# - 이미지가 제대로 저장되는가?
# - resized_fx 값이 표시되는가?
# - Depth map이 합리적으로 보이는가?
```

---

**⚠️ 이 문서의 내용을 위반하면 큰 벌을 받을 것입니다!**

**✅ 코드 수정 전 이 문서를 다시 읽으세요!**
