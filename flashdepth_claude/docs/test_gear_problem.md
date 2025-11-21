# Test Gear GT Canonicalization 문제

## 문제 요약

Test/Val 시 GT가 불필요하게 canonical space로 변환되었다가 다시 actual space로 복원되는 문제.

## 현재 상황

### Training 시 (올바름):
```
GT: actual space → canonical space (depth correction 포함)
Model: canonical space에서 학습
```

### Test/Val 시 (불필요한 왕복):
```
GT: actual → canonical (dataloader) → actual (test_gear5)  ← 불필요한 왕복!
Pred: canonical (model) → actual (test_gear5)
```

### 왜 문제인가:
1. `_apply_canonical_transform`의 **depth_correction**이 GT 값을 변경함
2. 다시 de-canonicalization으로 복원해야 함
3. **불필요한 연산 + 부동소수점 정밀도 손실 가능성**
4. 코드 복잡도 증가

### 올바른 Test/Val 방식:
```
GT: actual space (원본 그대로, 변환 없음)
Pred: canonical space → actual space (de-canonicalization만 수행)
```

---

## 구체적인 수정 방안

### 옵션 1: CombinedDataset 수정 (추천 ✅)

**장점:**
- 근본적 해결
- 모든 test_gear* 파일에 자동 적용
- Training/Test/Val의 명확한 분리

**단점:**
- 영향 범위가 큼 (CombinedDataset 사용하는 모든 코드)
- 신중한 테스트 필요

#### 수정 위치 1: `dataloaders/combined_dataset.py` - `__init__` 메서드

**파일:** `dataloaders/combined_dataset.py`
**라인:** ~60-90 (`__init__` 메서드)

**수정 전:**
```python
class CombinedDataset(Dataset):
    def __init__(self, datasets, root_dir, split='train',
                 video_length=5, cache_dir=None, resolution='518'):
        self.split = split
        self.video_length = video_length
        # ... 기타 초기화
```

**수정 후:**
```python
class CombinedDataset(Dataset):
    def __init__(self, datasets, root_dir, split='train',
                 video_length=5, cache_dir=None, resolution='518',
                 apply_gt_canonical=None):  # ← NEW parameter
        """
        Args:
            apply_gt_canonical (bool, optional):
                - None (default): auto-detect based on split (train=True, test/val=False)
                - True: apply canonical transform to GT (for training)
                - False: keep GT in actual space (for test/val)
        """
        self.split = split
        self.video_length = video_length

        # Auto-detect GT canonicalization based on split
        if apply_gt_canonical is None:
            self.apply_gt_canonical = (split == 'train')
        else:
            self.apply_gt_canonical = apply_gt_canonical

        logging.info(f"CombinedDataset: split={split}, apply_gt_canonical={self.apply_gt_canonical}")
        # ... 기타 초기화
```

#### 수정 위치 2: `dataloaders/combined_dataset.py` - `__getitem__` 메서드 (train split)

**파일:** `dataloaders/combined_dataset.py`
**라인:** ~540-560 (train split의 depth 처리 부분)

**수정 전:**
```python
# Apply Metric3D-style canonical transformation (now returns 6 values)
# This corrects GT depth based on actual vs theoretical resize ratios
depth_inverse_canonical, fx_canonical, fx_actual_returned, actual_valid_mask, fx_ratio, resize_ratio = self._apply_canonical_transform(
    depth_inverse_actual, fx_actual, original_h, original_w, target_resolution, resize_factor
)
```

**수정 후:**
```python
# Apply canonical transformation based on self.apply_gt_canonical
if self.apply_gt_canonical:
    # Training: Apply canonical transform to GT
    depth_inverse_canonical, fx_canonical, fx_actual_returned, actual_valid_mask, fx_ratio, resize_ratio = self._apply_canonical_transform(
        depth_inverse_actual, fx_actual, original_h, original_w, target_resolution, resize_factor
    )
else:
    # Test/Val: Keep GT in actual space (no canonicalization)
    depth_inverse_canonical = depth_inverse_actual  # No transformation
    fx_canonical = fx_actual  # Use actual focal length
    fx_actual_returned = fx_actual

    # Compute actual_valid_mask (<70m) in actual space
    with np.errstate(divide='ignore', invalid='ignore'):
        depth_actual = np.where(depth_inverse_actual > 1e-8, 1.0 / depth_inverse_actual, 0.0)
    actual_valid_mask = (depth_actual > 0) & (depth_actual < 70.0)

    # No correction needed
    fx_ratio = 1.0  # No focal length change
    resize_ratio = 1.0  # No resize correction
```

#### 수정 위치 3: `dataloaders/combined_dataset.py` - `__getitem__` 메서드 (val split)

**파일:** `dataloaders/combined_dataset.py`
**라인:** ~405-420 (val split의 depth 처리 부분)

**동일한 수정 패턴 적용** (위의 train split 수정과 동일)

#### 수정 위치 4: `dataloaders/combined_dataset.py` - `__getitem__` 메서드 (test split)

**파일:** `dataloaders/combined_dataset.py`
**라인:** ~475-490 (test split의 depth 처리 부분)

**동일한 수정 패턴 적용** (위의 train split 수정과 동일)

#### 수정 위치 5: `test_gear5.py` - GT de-canonicalization 제거

**파일:** `test_gear5.py`
**라인:** ~1108-1113

**수정 전:**
```python
# GT is in canonical space (fx=500), de-canonicalize to actual space for visualization
gt_depth_inverse_100_cpu = gt_depth_inverse_100[0]  # [T, 1, H, W] to CPU
gt_depth_canonical = 100.0 / (gt_depth_inverse_100_cpu + 1e-8)  # [T, 1, H, W] in canonical meters (CPU)

# De-canonicalize: depth_actual = depth_canonical × (fx_actual / fx_canonical)
de_canonical_ratio = fx_actual_tensor[0].cpu() / CANONICAL_FX  # [T] (CPU)
gt_depth_metric = gt_depth_canonical * de_canonical_ratio.view(T, 1, 1, 1)  # [T, 1, H, W] in actual meters (CPU)
```

**수정 후:**
```python
# GT is already in actual space (no canonicalization applied in dataloader for test/val)
gt_depth_inverse_100_cpu = gt_depth_inverse_100[0]  # [T, 1, H, W] to CPU
gt_depth_metric = 100.0 / (gt_depth_inverse_100_cpu + 1e-8)  # [T, 1, H, W] in actual meters (CPU)
# No de-canonicalization needed - GT is already in actual space
```

#### 수정 위치 6: 기타 test_gear*.py 파일들

**파일:** `test_gear4.py`, `test_gear5_film.py` 등
**동일한 패턴으로 수정**

**검색 패턴:**
```bash
grep -n "de_canonical_ratio.*fx_actual.*CANONICAL_FX" test_gear*.py
grep -n "gt_depth_metric = gt_depth_canonical \*" test_gear*.py
```

---

### 옵션 2: test_gear*.py만 수정 (비추천 ❌)

**이 방법은 잘못되었습니다!**

이유:
- GT가 dataloader에서 이미 canonical로 변환됨
- De-canonicalization을 제거하면 GT는 canonical space에 머무름
- Pred는 actual space로 변환됨
- **공간 불일치 발생!**

```python
# 잘못된 수정 예시 (하지 말 것!)
gt_depth_metric = 100.0 / (gt_depth_inverse_100 + 1e-8)  # ← canonical space
# Pred는 actual space로 변환됨 → 공간 불일치!
```

---

## 영향 범위

### 수정이 필요한 파일:

1. **`dataloaders/combined_dataset.py`** (핵심)
   - `__init__`: apply_gt_canonical 파라미터 추가
   - `__getitem__` (train/val/test split 모두): 조건부 canonicalization

2. **`test_gear5.py`**
   - Line ~1108-1113: GT de-canonicalization 제거

3. **`test_gear4.py`**
   - 동일한 패턴 확인 및 수정

4. **`test_gear5_film.py`**
   - 동일한 패턴 확인 및 수정

5. **기타 test_gear*.py 파일들**
   - 동일한 패턴이 있는지 확인

### 테스트 필요:

1. **Training 영향 없음 확인:**
   ```bash
   # Training 시 apply_gt_canonical=True (기본값)
   python train_gear5.py --config-variant hybrid
   ```

2. **Test 결과 변화 확인:**
   ```bash
   # Before/After 메트릭 비교
   python test_gear5.py --dataset unreal4k --vid-len 100
   ```

3. **모든 데이터셋 테스트:**
   - TartanAir
   - Unreal4K
   - NuScenes
   - Spring
   - PointOdyssey

---

## 기대 효과

1. **정확성 향상:** 불필요한 변환 제거로 부동소수점 정밀도 손실 방지
2. **코드 명확성:** Training과 Test/Val의 GT 처리 방식 명확히 분리
3. **성능 개선:** 불필요한 연산 제거 (미미하지만)
4. **유지보수성:** 향후 canonicalization 관련 버그 방지

---

## 참고: Canonical Space vs Actual Space

### Canonical Space (fx=500):
- 모델이 학습하는 공간
- 다양한 데이터셋을 통일된 focal length로 정규화
- Depth는 canonical fx 기준으로 측정됨

### Actual Space (dataset의 실제 fx):
- 원본 데이터의 공간
- 각 데이터셋/프레임마다 실제 focal length 사용
- Depth는 실제 fx 기준으로 측정됨

### De-canonicalization:
```
depth_actual = depth_canonical × (fx_actual / fx_canonical)
```

이 변환은 **Prediction에만 필요**하며, **GT는 원본 actual space 그대로 유지**해야 함.
