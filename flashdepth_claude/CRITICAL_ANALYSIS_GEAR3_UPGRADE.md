# 🚨 Gear3 Upgrade 학습 구조의 근본적인 문제점 분석

**분석일**: 2025-11-09
**분석자**: Claude
**목적**: Gear3 Upgrade의 metric feature modulation 학습 가능성 검증

---

## 📋 Executive Summary

**결론**: ⚠️ **현재 학습 구조는 metric feature modulation을 제대로 학습할 수 없습니다.**

**핵심 문제**:
1. Training 시 Canonical Space transformation이 metric scale 정보를 파괴
2. Test 시 canonical space 역변환이 누락되어 잘못된 metric depth 예측
3. FG/BG modulation이 metric scale signal을 받지 못함
4. 근본적으로 relative depth 학습 구조

---

## 🔍 상세 분석

### 1️⃣ Training Pipeline 분석

#### 1.1 GT Depth 처리 과정

```python
# train_gear3_upgrade.py:908-910
# GT depth from dataloader is inverse depth (1/m), scale to 100/m for training
# This matches FlashDepth's relative depth scale (≈ 100/metric_depth)
gt_depth_inverse_100 = gt_depth * 100.0  # (1/m) * 100 = 100/m
```

**분석**:
- Dataloader는 **metric inverse depth** (1/m) 제공
- 100을 곱해서 100/m으로 스케일링
- 주석에 "relative depth scale"이라 적혀있지만, **실제로는 여전히 metric inverse depth**

#### 1.2 Canonical Space Transformation

```python
# train_gear3_upgrade.py:918-924
if self.config.get('use_canonical_space', False):
    CANONICAL_FX = self._get_canonical_focal_length()

    # Transform inverse depth directly to canonical space
    # inverse_canonical = inverse_actual * (fx_actual / CANONICAL_FX)
    fx_actual = focal_lengths.view(B, T, 1, 1, 1)
    gt_depth_inverse_100 = gt_depth_inverse_100 * (fx_actual / CANONICAL_FX)
```

**분석**:
- Focal length에 따라 inverse depth 스케일 조정
- 서로 다른 카메라의 depth를 정규화하는 목적
- **문제**: 이 과정에서 **metric scale 정보가 파괴됨**

**수학적 설명**:
```
원래 metric inverse depth: inv_metric = 1/D_metric
Canonical transformation: inv_canonical = inv_metric * (fx_actual / CANONICAL_FX)

→ inv_canonical은 더 이상 metric이 아님!
→ focal length에 따라 스케일이 달라짐
→ Relative/Affine-invariant depth처럼 동작
```

#### 1.3 Loss Computation

```python
# train_gear3_upgrade.py:1047-1050
loss = self.loss_fn(
    pred_depth_inverse_flat.float(),  # Predicted inverse depth (canonical space)
    gt_depth_inverse_flat.float(),     # GT inverse depth (canonical space)
    valid_mask.float()
)
```

**분석**:
- LogL1Loss 사용
- 두 입력 모두 **canonical space**의 inverse depth
- Metric scale 정보 없음!

**핵심 문제**:
```
Loss는 canonical space에서 계산됨
→ 서로 다른 focal length의 이미지들이 동일한 기준으로 학습됨
→ 모델은 "canonical space에서의 depth"를 학습
→ Metric depth 정보를 배울 수 없음
```

---

### 2️⃣ Test Pipeline 분석

#### 2.1 Metric Depth 변환 (문제!)

```python
# test_gear3_upgrade.py:727 근처
# Convert to metric depth: 100/m -> m
pred_depth_metric = 100.0 / (pred_depth_inverse_100[0] + 1e-8)  # [1, H, W]
```

**분석**:
- 단순히 역수를 취해서 metric depth 계산
- **Canonical space 역변환이 누락됨!**

**올바른 변환**:
```python
# 현재 (틀림)
pred_depth_metric = 100.0 / pred_depth_inverse_100

# 올바른 변환
pred_depth_canonical = 100.0 / pred_depth_inverse_100
pred_depth_metric = pred_depth_canonical * (CANONICAL_FX / fx_actual)  # ← 이 단계 누락!
```

**결과**:
- Test 시 예측한 depth는 **canonical space depth**
- 이것은 **잘못된 metric depth**
- Focal length가 다르면 스케일이 틀림

---

### 3️⃣ Feature Modulation 학습 가능성 분석

#### 3.1 FG/BG Modulation의 입력

```python
# train_gear3_upgrade.py:982-986
path_1_modulated, importance_map, fg_features, bg_features, fg_mask, bg_mask = model.gear3_upgrade_head(
    patch_tokens,              # From DINOv2 encoder
    attention_weights,         # Attention maps
    [path_1],                  # DPT features (before Mamba)
    patch_h, patch_w,
    attention_weights_multi_layer=attention_weights_multi_layer,
    cls_token=cls_token
)
```

**분석**:
- 입력: Visual features, attention weights
- **Metric depth 정보 없음!**
- **Focal length 정보 없음!**

#### 3.2 Supervision Signal

**Training supervision**:
```
Loss: LogL1(pred_canonical, gt_canonical)
     ↓
Gradient flows back to Gear3 Upgrade Head
     ↓
What can it learn?
→ Relative depth structure (FG/BG separation based on depth order)
→ NOT metric scale information!
```

**핵심 문제**:
1. Canonical space loss는 **scale-invariant** 성격을 가짐
2. 서로 다른 focal length의 이미지들이 동일하게 정규화됨
3. **Metric scale을 학습할 supervision signal이 없음**

**비유**:
```
목표: "실제 거리(meter)를 구분하는 feature modulation 학습"
현실: "정규화된 상대적 깊이 순서를 학습"
      → 1m, 2m, 3m를 구분할 수 없음
      → "가까움", "중간", "멈" 정도만 구분 가능
```

---

### 4️⃣ 원본 FlashDepth와의 비교

#### FlashDepth (원본)

```python
# train.py (원본 FlashDepth)
# NO canonical space transformation!
# Loss directly on inverse depth
```

**특징**:
- Canonical space 없음
- Relative depth 학습 (affine-invariant)
- Test 시 scale/shift alignment 필요

#### Gear2 (GSP Head)

```python
# Gear2 approach:
# 1. Base model: Relative depth 예측 (FlashDepth와 동일)
# 2. GSP Head: CLS token → (scale, shift) 예측
# 3. Metric depth = scale * relative_depth + shift
```

**특징**:
- Relative depth는 base model이 담당
- **Metric scale은 GSP head가 별도로 학습**
- Explicit metric supervision (metric depth GT 필요)

#### Gear3 Upgrade (현재)

```python
# Gear3 Upgrade approach:
# 1. Canonical space로 정규화
# 2. Feature modulation 적용
# 3. Loss on canonical space
# 4. Test 시 단순 역수 변환
```

**문제점**:
- Canonical space가 metric scale 정보 파괴
- Feature modulation이 metric을 학습할 수 없음
- Test 변환이 잘못됨

---

## 🎯 근본 원인 요약

### Problem 1: Metric Scale 정보 파괴

```
GT metric inverse depth (1/m)
     ↓ × 100
GT inverse depth (100/m)
     ↓ × (fx_actual / CANONICAL_FX)  ← 여기서 metric 정보 파괴!
GT canonical inverse depth (정규화됨, metric 아님)
     ↓ Loss
Feature modulation은 canonical space만 학습
```

### Problem 2: Test 변환 오류

```
Predicted canonical inverse depth
     ↓ ÷ 100  (단순 역수)
Predicted "metric" depth  ← 잘못됨! Canonical space depth임
```

**올바른 변환**:
```
Predicted canonical inverse depth
     ↓ ÷ 100
Predicted canonical depth
     ↓ × (CANONICAL_FX / fx_actual)  ← 이 단계 필요!
Predicted metric depth
```

### Problem 3: Supervision Signal 부재

**현재 학습 구조**:
```
Input: Images (different focal lengths)
   ↓ Canonical normalization
Canonical space depth
   ↓ Loss (scale-invariant)
Feature modulation
```

**문제**: Focal length 정보가 loss에 반영되지 않음
→ Metric scale을 학습할 수 없음

---

## 💡 해결 방안

### Option 1: Canonical Space 제거 + Metric Loss (권장) ⭐⭐⭐⭐⭐

**방법**:
```python
# Remove canonical space transformation
gt_depth_inverse_100 = gt_depth * 100.0  # Keep as metric inverse depth

# Train directly on metric inverse depth
loss = self.loss_fn(pred_inverse, gt_inverse, valid_mask)

# Test: simple conversion (now correct!)
pred_depth_metric = 100.0 / pred_inverse
```

**장점**:
- ✅ Metric scale 정보 보존
- ✅ Feature modulation이 metric을 학습 가능
- ✅ Test 변환 간단하고 정확
- ✅ Focal length에 무관하게 정확한 metric depth

**단점**:
- ⚠️ 서로 다른 focal length의 이미지들을 함께 학습하기 어려울 수 있음
  (하지만 metric depth GT가 있으면 문제없음)

### Option 2: Canonical Space 역변환 추가

**방법**:
```python
# Training: Keep current canonical space approach
gt_canonical = gt_metric * (fx_actual / CANONICAL_FX)

# Test: Add inverse transformation
pred_canonical_depth = 100.0 / pred_inverse_100
pred_metric_depth = pred_canonical_depth * (CANONICAL_FX / fx_actual)
```

**장점**:
- ✅ Training code 변경 최소화
- ✅ Test 변환 정확해짐

**단점**:
- ❌ Feature modulation은 여전히 canonical space만 학습
- ❌ Metric-aware modulation이 아님
- ❌ 근본적인 문제 미해결

### Option 3: Focal Length를 모델 입력으로 제공 ⭐⭐⭐⭐

**방법**:
```python
# Gear3 Upgrade Head에 focal length 정보 전달
path_1_modulated, ... = model.gear3_upgrade_head(
    patch_tokens,
    attention_weights,
    [path_1],
    patch_h, patch_w,
    focal_lengths=focal_lengths,  # ← 추가!
    ...
)

# Head 내부에서 focal length 기반 adaptive modulation
class Gear3UpgradeHead:
    def forward(self, ..., focal_lengths=None):
        if focal_lengths is not None:
            # Use focal length to modulate features
            # E.g., focal_embedding = self.focal_encoder(focal_lengths)
            # features = features * focal_embedding
```

**장점**:
- ✅ Metric scale 정보를 명시적으로 제공
- ✅ Feature modulation이 focal length 인지 가능
- ✅ Canonical space 유지 가능

**단점**:
- ⚠️ Architecture 변경 필요
- ⚠️ Focal embedding 설계 필요

### Option 4: GSP Head 추가 (Gear2 방식) ⭐⭐⭐

**방법**:
```python
# Gear3 Upgrade for relative depth
# + Separate GSP head for metric scale
pred_relative = gear3_upgrade_model(images)
scale, shift = gsp_head(cls_token)
pred_metric = scale * pred_relative + shift
```

**장점**:
- ✅ Proven approach (Gear2)
- ✅ Metric scale 명시적으로 학습
- ✅ 두 단계 분리 (relative structure + metric scale)

**단점**:
- ⚠️ Two-stage approach (feature modulation은 여전히 relative)
- ⚠️ 추가 파라미터

---

## 🎪 검증 실험 제안

### Experiment 1: Canonical Space 효과 검증

**목적**: Canonical space가 성능에 미치는 영향 확인

**방법**:
```bash
# Baseline: use_canonical_space=false
python train_gear3_upgrade.py use_canonical_space=false

# Current: use_canonical_space=true
python train_gear3_upgrade.py use_canonical_space=true
```

**예상 결과**:
- 만약 canonical=false가 더 나은 metric depth 예측
  → Canonical space가 문제임을 확인

### Experiment 2: Test 변환 수정 효과

**목적**: Canonical 역변환의 중요성 확인

**방법**:
```python
# Modify test_gear3_upgrade.py
# Add canonical inverse transformation
pred_canonical_depth = 100.0 / pred_inverse_100
pred_metric_depth = pred_canonical_depth * (CANONICAL_FX / fx_actual)
```

**예상 결과**:
- Metric depth 정확도 향상 (특히 다른 focal length 데이터셋)

### Experiment 3: Focal Length Input

**목적**: Focal length 정보 제공의 효과

**방법**:
- Gear3 Upgrade Head에 focal length embedding 추가
- Metric depth loss 사용

**예상 결과**:
- Feature modulation이 metric-aware해짐

---

## 📊 최종 권고사항

### 즉시 수정 필요 (Critical)

1. **Test 변환 수정** (test_gear3_upgrade.py)
   ```python
   # 현재 (틀림)
   pred_depth_metric = 100.0 / pred_inverse_100

   # 수정
   if use_canonical_space:
       pred_canonical_depth = 100.0 / pred_inverse_100
       pred_depth_metric = pred_canonical_depth * (CANONICAL_FX / fx_actual)
   else:
       pred_depth_metric = 100.0 / pred_inverse_100
   ```

### 단기 개선 (High Priority)

2. **Canonical Space 제거** (train_gear3_upgrade.py)
   - `use_canonical_space=False`로 학습
   - Metric inverse depth로 직접 학습
   - 다양한 focal length 데이터셋에서 성능 확인

### 장기 개선 (Medium Priority)

3. **Focal Length Conditioning**
   - Gear3 Upgrade Head에 focal length 입력 추가
   - Adaptive metric-aware modulation 설계

4. **Hybrid Approach**
   - Feature modulation for structure
   - GSP head for metric scale
   - Best of both worlds

---

## 🔬 결론

**현재 상태**:
- ❌ Metric feature modulation 학습 **불가능**
- ❌ Test metric depth 예측 **부정확**
- ✅ Relative depth structure는 학습 가능

**권장 조치**:
1. **즉시**: Test 변환 수정 (canonical 역변환 추가)
2. **단기**: Canonical space 제거하고 metric loss로 재학습
3. **장기**: Focal length conditioning 또는 GSP head 추가

**핵심 메시지**:
> **Canonical space transformation은 multi-focal-length training을 위한 것이지만,
> metric scale 정보를 파괴하여 metric-aware feature modulation을 불가능하게 만듭니다.**
