# Canonical Space Normalization Implementation Plan

## 목차
1. [이론적 배경](#1-이론적-배경)
2. [수학적 분석](#2-수학적-분석)
3. [Canonical Space 개념](#3-canonical-space-개념)
4. [데이터셋별 Focal Length](#4-데이터셋별-focal-length)
5. [구현 계획](#5-구현-계획)
6. [검증 방법](#6-검증-방법)
7. [주의사항](#7-주의사항)
8. [사용법](#8-사용법)

---

## 1. 이론적 배경

### 1.1 문제 정의: Focal Length Ambiguity

단안 metric depth 추정은 본질적으로 **ill-posed** 문제입니다:

```
같은 이미지 크기 → 다른 실제 깊이 가능

[Wide-angle: fx=500]     [Telephoto: fx=2000]
10px @ 5m 물체          10px @ 20m 물체
```

**예시:**
- 같은 10m 거리 물체
- Wide-angle (fx=500): 작게 보임
- Telephoto (fx=2000): 크게 보임 (4배!)

**문제:** 이미지만으로는 실제 거리를 알 수 없음!

### 1.2 왜 Intrinsic 정보가 필요한가?

**Projection Equation:**
```
pixel_size = fx × object_size / depth

→ depth = fx × object_size / pixel_size
```

**Focal length 없이는 scale이 모호함:**
- DINOv2의 prior (물체 크기 등)로 어느 정도 추정 가능
- 하지만 fx 변화에 취약
- 학습 데이터와 다른 카메라에서 성능 저하

### 1.3 현재 문제점

**현재 train_gear 모델:**
```python
# train_gear3.py:768
focal_length = 1000.0  # 선언만 하고 사용 안 함!

# 모델 입력
- DINOv2 features (patch tokens, attention)
- DPT features (path_1)
→ Focal length 정보 없음!
```

**학습 데이터셋 fx 분포:**
- TartanAir: fx ≈ 320
- Spring: fx ≈ 450-600
- MVS-Synth: fx ≈ 1000-1500
- Waymo: fx ≈ 1000-2000

→ 모델이 "평균적인" fx ≈ 700-1000을 암묵적으로 학습
→ 이 범위를 벗어나면 성능 저하

---

## 2. 수학적 분석

### 2.1 Loss Space 선택: Inverse vs Metric

현재 코드는 **Inverse depth space**에서 loss를 계산합니다:

```python
# train_gear3.py:781, 841, 874-878
gt_depth_inverse_100 = gt_depth * 100.0  # (1/m) * 100 = 100/m
pred_depth_inverse = out  # 모델이 100/m 스케일로 출력
loss = LogL1Loss(pred_inverse, gt_inverse)  # inverse space에서 계산
```

**질문:** 왜 metric space가 아닌 inverse space인가?

### 2.2 방법 A: Metric Space Loss

```python
pred_inverse = model(x)  # ~100/m 스케일
pred_metric = 100.0 / pred_inverse  # m 스케일로 변환
gt_metric = 100.0 / gt_inverse  # m 스케일

loss = L1(pred_metric, gt_metric)  # metric space에서 loss
```

**Gradient 계산:**
```
Loss_A = |pred_metric - gt_metric|
       = |100/pred_inverse - 100/gt_inverse|

∂Loss_A/∂pred_inverse = ∂Loss_A/∂pred_metric × ∂pred_metric/∂pred_inverse
                       = sign(pred_metric - gt_metric) × (-100/pred_inverse²)
                       = -100 × sign(...) / pred_inverse²
```

**핵심:** Gradient가 `1/pred_inverse²`에 비례!

**실전 영향:**
- **가까운 물체** (pred_inverse=10, depth=10m):
  - Gradient ∝ 1/100 = **0.01**
  - **학습 느림** → 자율주행에서 치명적!

- **먼 물체** (pred_inverse=1, depth=100m):
  - Gradient ∝ 1/1 = **1.0**
  - **학습 빠름** → sky 등 덜 중요한 영역에 편향

**문제점:**
- 가까운 물체(중요!)에 둔감
- 먼 물체(덜 중요)에 민감
- **Inverse gradient weighting** 효과

### 2.3 방법 B: Inverse Space Loss (현재 방법) ✅

```python
pred_inverse = model(x)  # 100/m 스케일
gt_inverse = gt * 100.0  # 100/m 스케일로 변환

loss = L1(pred_inverse, gt_inverse)  # inverse space에서 loss
```

**Gradient 계산:**
```
Loss_B = |pred_inverse - gt_inverse|

∂Loss_B/∂pred_inverse = sign(pred_inverse - gt_inverse) × 1
                       = sign(...)
```

**핵심:** Gradient가 **균일** (거리에 무관)!

**장점:**
- 모든 거리에서 **동일한 attention**
- 가까운 물체와 먼 물체 균등하게 학습
- 수치적으로 안정 (1/pred² 폭발 없음)
- **일반적으로 depth estimation에서 inverse space가 더 나음** (논문들 consensus)

### 2.4 결론: 두 방법은 완전히 다름!

| 항목 | 방법 A (Metric) | 방법 B (Inverse) ✅ |
|------|----------------|-------------------|
| Gradient | ∝ 1/pred² | ∝ 1 (균일) |
| 가까운 물체 | 학습 느림 ❌ | 균등 ✅ |
| 먼 물체 | 학습 빠름 (편향) | 균등 ✅ |
| 수치 안정성 | 불안정 (폭발 가능) | 안정 ✅ |
| 자율주행 | 부적합 | 적합 ✅ |

**→ 현재 방법 (Inverse space)이 올바름!**

---

## 3. Canonical Space 개념

### 3.1 Inverse vs Canonical: 직교하는 개념

**중요한 깨달음:**
- **Loss space 선택** (Inverse vs Metric): Gradient 특성 결정
- **Focal length normalization** (Canonical vs Non-canonical): fx 영향 제거

**둘은 독립적!**
- Inverse space loss는 유지 (균일한 gradient)
- Canonical space는 fx 정규화만 담당

### 3.2 Canonical Space란?

**정의:** 모든 데이터를 "표준 카메라"로 본 것처럼 변환

**핵심 아이디어:**
```python
# 물리 법칙
pixel_size = fx × object_size / depth_metric

# fx=500인 카메라에서 10m 물체
pixel_500 = 500 × obj / 10

# 이것을 fx=1000 카메라로 본 것처럼 만들려면?
# pixel_500 = 1000 × obj / depth_canonical
# → depth_canonical = 1000 × obj / pixel_500
#                   = 1000 × obj / (500 × obj / 10)
#                   = 1000 / 500 × 10
#                   = 2 × 10 = 20m

# 일반화
depth_canonical = depth_metric × (fx_actual / CANONICAL_FX)
```

### 3.3 학습/테스트 변환

**학습 시:**
```python
CANONICAL_FX = 1000.0

# GT를 canonical space로 변환
gt_metric = 1.0 / gt_depth  # dataloader에서 inverse로 옴
gt_metric_canonical = gt_metric × (fx_actual / CANONICAL_FX)

# Inverse depth로 변환 (100/m 스케일)
gt_inverse_canonical = 100.0 / gt_metric_canonical
                     = 100.0 × CANONICAL_FX / (gt_metric × fx_actual)
                     = 100.0 × CANONICAL_FX × gt_depth / fx_actual

# 모델은 canonical inverse 출력
pred_inverse_canonical = model(img)

# Loss (inverse space, 균일한 gradient!)
loss = L1(pred_inverse_canonical, gt_inverse_canonical)
```

**테스트 시:**
```python
# 모델 예측 (canonical inverse)
pred_inverse_canonical = model(img_test)

# Metric depth로 역변환
pred_metric_canonical = 100.0 / pred_inverse_canonical
pred_metric = pred_metric_canonical / (fx_test / CANONICAL_FX)
            = pred_metric_canonical × CANONICAL_FX / fx_test
            = (100.0 / pred_inverse_canonical) × CANONICAL_FX / fx_test
```

### 3.4 장점

**1. 이미지 품질 보존:**
- 이미지를 리사이즈하지 않음 (대안: 이미지 리사이즈)
- DINOv2 feature 품질 100% 유지
- 계산 효율적 (단순 곱셈)

**2. Focal length robust:**
- 학습 데이터의 fx 분포에 덜 민감
- 다양한 카메라에 일반화

**3. 검증된 방법:**
- **Metric3D v2 (CVPR 2024)**: 동일한 방식 사용
- SOTA 성능 입증

---

## 4. 데이터셋별 Focal Length

### 4.1 확인된 값

| Dataset | Focal Length (fx) | Resolution | 출처 | 파일 |
|---------|------------------|------------|------|------|
| **TartanAir** | 320 | 640×480 | 공식 문서 | tartanair_dataset.py |
| **Spring** | 450-600 (가변) | 1920×1080 | intrinsics.txt | spring_dataset.py |
| **Waymo** | 1000-2000 (가변) | 1920×1280 | intrinsic matrix | waymo_dataset.py |
| **Dynamic Replica** | width/2 (근사) | 가변 | Pinhole 표준 | dynamicreplica_dataset.py |

### 4.2 조사 필요

| Dataset | 추정 방법 | 파일 |
|---------|----------|------|
| **MVS-Synth** | 데이터셋 camera params 파일 | mvssynth_dataset.py |
| **PointOdyssey** | 데이터셋 조사 | pointodyssey_dataset.py |
| **nuScenes** | 데이터셋 조사 | nuscenes_dataset.py |
| **Sintel** | 데이터셋 조사 | sintel_dataset.py |

### 4.3 Fallback 전략

**fx 정보 없는 경우:**
1. **이미지 width 기반 추정**: fx ≈ width × 0.7 (일반적 FOV 60-70도)
2. **평균값 사용**: 학습 데이터 fx 중간값 (≈ 500-800)
3. **Warning 출력**:
   ```python
   logger.warning(f"Focal length unavailable for {dataset}, using default {default_fx}")
   ```

---

## 5. 구현 계획

### 5.1 수정 파일 목록

```
flashdepth_claude/
├── dataloaders/
│   ├── combined_dataset.py           [수정] fx 반환 로직 추가
│   ├── tartanair_dataset.py          [수정] fx=320 반환
│   ├── spring_dataset.py             [수정] fx 반환 추가 (이미 로드 중)
│   ├── waymo_dataset.py              [조사] intrinsic에서 fx 추출
│   ├── mvssynth_dataset.py           [조사] camera params 로드
│   ├── dynamicreplica_dataset.py     [조사] fx=width/2 계산
│   ├── pointodyssey_dataset.py       [조사] 데이터셋 확인
│   ├── nuscenes_dataset.py           [조사] 데이터셋 확인
│   └── sintel_dataset.py             [조사] 데이터셋 확인
├── train_gear3_upgrade.py            [수정] Canonical space 적용 (메인!)
├── test_gear3_upgrade.py             [수정] Inverse transform 적용 (메인!)
├── train_gear3.py                    [수정] 동일하게 적용
├── test_gear3.py                     [수정] 동일하게 적용
├── train_gear2.py                    [수정] 동일하게 적용
├── test_gear2.py                     [수정] 동일하게 적용
└── configs/
    ├── gear3_upgrade/config.yaml     [수정] canonical_fx 추가
    ├── gear3/config.yaml             [수정] canonical_fx 추가
    └── gear2/config.yaml             [수정] canonical_fx 추가
```

### 5.2 A. 데이터로더 수정

#### **A-1. CombinedDataset (핵심!)**

**파일:** `dataloaders/combined_dataset.py`

**현재 반환값:**
```python
return images, depths, dataset_idx
```

**수정 후:**
```python
def __getitem__(self, idx):
    images = []
    depths = []
    focal_lengths = []  # ← 추가

    for seq_idx in sequence_indices:
        pair = dataset_list[seq_idx]

        # 이미지, depth 로드
        image = _load_and_process_image(pair['image'], ...)
        depth = self.depth_read_list[dataset_idx](pair['depth'], is_inverse=True)

        # Focal length 로드 (새로 추가!)
        fx = self.get_focal_length_list[dataset_idx](pair, image.shape)

        images.append(image)
        depths.append(depth)
        focal_lengths.append(fx)

    images = torch.stack(images, dim=0)  # [T, C, H, W]
    depths = torch.stack(depths, dim=0)  # [T, H, W]
    focal_lengths = torch.tensor(focal_lengths, dtype=torch.float32)  # [T]

    return images, depths, focal_lengths, dataset_idx  # ← fx 추가
```

**주의:** Validation/Test split도 동일하게 수정 필요!

#### **A-2. TartanAir (간단!)**

**파일:** `dataloaders/tartanair_dataset.py`

**추가:**
```python
class TartanairDepth(BaseDatasetPairs):
    def __init__(self, root_dir, split, load_cache=None):
        # 기존 코드...
        self.focal_length = 320.0  # TartanAir V1/V2 모두 320

    def get_focal_length(self, pair, image_shape):
        """
        TartanAir focal length는 고정값 320

        Args:
            pair: 이미지/depth pair dict (사용 안 함)
            image_shape: 이미지 shape (사용 안 함)

        Returns:
            float: focal length (320.0)
        """
        return self.focal_length
```

#### **A-3. Spring (거의 완료!)**

**파일:** `dataloaders/spring_dataset.py`

**현재 상태:**
```python
def depth_read(self, path, is_inverse=True, resize_factor=1.0, ...):
    # 이미 fx 로드 중!
    index = int(os.path.basename(path).split('left_')[1].split('.')[0]) - 1
    intrinsics_path = os.path.dirname(path.replace('disp1_left', 'cam_data')) + '/intrinsics.txt'
    fx = np.loadtxt(intrinsics_path)[index][0]
    fx = fx * resize_factor

    # depth 계산에 사용
    inverse_depth = disparity / (fx * SPRING_BASELINE)

    return inverse_depth  # fx는 반환 안 함!
```

**수정 방안 1 (추천):** 별도 메서드 추가
```python
def get_focal_length(self, pair, image_shape):
    """
    Spring focal length를 intrinsics.txt에서 로드

    Args:
        pair: {'depth': depth_path, ...}
        image_shape: (H, W) for resize_factor calculation
    """
    depth_path = pair['depth']
    index = int(os.path.basename(depth_path).split('left_')[1].split('.')[0]) - 1
    intrinsics_path = os.path.dirname(depth_path.replace('disp1_left', 'cam_data')) + '/intrinsics.txt'
    fx = np.loadtxt(intrinsics_path)[index][0]

    # Resize factor 계산 (원본 해상도 1920x1080)
    original_width = 1920
    current_width = image_shape[1]  # (H, W)
    resize_factor = current_width / original_width

    fx = fx * resize_factor
    return float(fx)
```

**수정 방안 2:** depth_read() 반환값 변경
```python
def depth_read(self, path, is_inverse=True, resize_factor=1.0, return_fx=False, ...):
    # 기존 코드...
    fx = np.loadtxt(intrinsics_path)[index][0]
    fx = fx * resize_factor

    inverse_depth = disparity / (fx * SPRING_BASELINE)

    if return_fx:
        return inverse_depth, fx
    return inverse_depth
```

**추천:** 방안 1 (인터페이스 일관성 유지)

#### **A-4. Waymo (조사 필요)**

**파일:** `dataloaders/waymo_dataset.py`

**WebSearch 결과:**
- Intrinsic matrix 제공 (9개 float32)
- `calibration.intrinsic[0, 0]` = fx

**구현 예시:**
```python
def get_focal_length(self, pair, image_shape):
    """
    Waymo focal length를 intrinsic matrix에서 추출

    Waymo provides 3x3 intrinsic matrix:
    [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
    """
    # 데이터셋 구조에 따라 intrinsic 로드
    scene_path = pair.get('scene_path') or pair.get('depth')
    calib_path = ...  # 데이터셋 구조 확인 필요

    intrinsic_matrix = load_intrinsic(calib_path)  # [3, 3] numpy array
    fx = float(intrinsic_matrix[0, 0])

    return fx
```

#### **A-5. MVS-Synth (조사 필요)**

**파일:** `dataloaders/mvssynth_dataset.py`

**WebSearch 결과:**
- Camera parameters 제공
- 데이터셋 다운로드 후 파일 구조 확인 필요

**Fallback:**
```python
def get_focal_length(self, pair, image_shape):
    """
    MVS-Synth focal length

    TODO: 데이터셋에서 camera params 파일 찾기
    Fallback: 이미지 width 기반 추정
    """
    # Fallback: width × 0.7 (일반적 FOV 60-70도)
    width = image_shape[1]
    fx_estimated = width * 0.7

    logger.warning(f"MVS-Synth: Using estimated focal length {fx_estimated:.1f} (width={width})")
    return float(fx_estimated)
```

#### **A-6. Dynamic Replica**

**파일:** `dataloaders/dynamicreplica_dataset.py`

**WebSearch 결과:**
- Pinhole camera: fx = width / 2
- Principal point = (width-1)/2, (height-1)/2

**구현:**
```python
def get_focal_length(self, pair, image_shape):
    """
    Dynamic Replica focal length (Pinhole approximation)

    Standard pinhole: fx = width / 2
    """
    width = image_shape[1]
    fx = width / 2.0
    return float(fx)
```

#### **A-7~9. PointOdyssey, nuScenes, Sintel (조사 필요)**

**각 파일:** `pointodyssey_dataset.py`, `nuscenes_dataset.py`, `sintel_dataset.py`

**구현 템플릿:**
```python
def get_focal_length(self, pair, image_shape):
    """
    [Dataset name] focal length

    TODO: 데이터셋 조사 후 구현
    """
    # Option 1: 데이터셋에서 로드
    # fx = load_from_dataset(...)

    # Option 2: Fallback
    width = image_shape[1]
    fx_default = width * 0.7  # 또는 학습 데이터 평균 (500-800)

    logger.warning(f"{self.dataset_name}: Using default focal length {fx_default:.1f}")
    return float(fx_default)
```

### 5.3 B. 학습 스크립트 수정

#### **B-1. train_gear3_upgrade.py (메인!)**

**Config 추가:**
```yaml
# configs/gear3_upgrade/config.yaml
canonical_fx: 1000.0  # 기본값 (또는 518.8579 - Metric3D)
```

**train_step() 수정:**
```python
def train_step(self, batch):
    # Unpack batch with focal lengths
    images, gt_depth, focal_lengths, dataset_idx = batch  # ← fx 추가
    images = images.to(self.device)
    gt_depth = gt_depth.to(self.device)
    focal_lengths = focal_lengths.to(self.device)  # [B, T] or [B*T]

    # Add channel dimension if needed
    if gt_depth.ndim == 3:
        gt_depth = gt_depth.unsqueeze(1)
    elif gt_depth.ndim == 4 and gt_depth.shape[1] != 1:
        gt_depth = gt_depth.unsqueeze(2)

    B, T = images.shape[:2]

    # Get canonical focal length from config
    CANONICAL_FX = self.config.get('canonical_fx', 1000.0)

    # ===== Canonical Space Transformation =====
    # GT depth from dataloader is inverse depth (1/m)
    # 1. Convert to metric depth
    gt_depth_metric = 1.0 / (gt_depth + 1e-8)  # [B, T, 1, H, W], meters

    # 2. Transform to canonical space
    # Expand focal_lengths to match gt shape: [B, T] → [B, T, 1, 1, 1]
    fx_expanded = focal_lengths.view(B, T, 1, 1, 1)
    gt_metric_canonical = gt_depth_metric * (fx_expanded / CANONICAL_FX)

    # 3. Convert to inverse depth (100/m scale for training)
    gt_inverse_canonical = 100.0 / (gt_metric_canonical + 1e-8)
    # Simplified formula:
    # = 100.0 × CANONICAL_FX / (gt_metric × fx)
    # = 100.0 × CANONICAL_FX × gt_depth / fx

    # ===== Forward Pass =====
    # ... (기존 코드와 동일, encoder/DPT/Gear3/Mamba) ...

    # Prediction is in canonical inverse depth space
    pred_inverse_canonical = out  # [B*T, 1, H, W]

    # ===== Loss Computation =====
    # Reshape GT from (B, T, 1, H, W) to (B*T, H, W)
    gt_inverse_canonical_flat = rearrange(gt_inverse_canonical, 'b t 1 h w -> (b t) h w')

    # Remove channel dimension from prediction
    pred_inverse_canonical_flat = pred_inverse_canonical.squeeze(1)  # [B*T, H, W]

    # Compute valid mask
    MIN_INVERSE_DEPTH = 100.0 / 70.0  # 70m threshold
    gt_valid_mask = (gt_inverse_canonical_flat > MIN_INVERSE_DEPTH)
    pred_valid_mask = (pred_inverse_canonical_flat > MIN_INVERSE_DEPTH)
    valid_mask = gt_valid_mask & pred_valid_mask

    if valid_mask.sum() == 0:
        self.logger.error("No valid GT & Pred pixels in batch!")
        return {'loss': 0.0}

    # Compute loss in canonical inverse space
    with torch.amp.autocast('cuda', enabled=False):
        loss = self.loss_fn(
            pred_inverse_canonical_flat.float(),
            gt_inverse_canonical_flat.float(),
            valid_mask.float()
        )

    # Backward pass
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    self.optimizer.step()
    self.scheduler.step()

    return {'loss': loss.item()}
```

**validate() 수정:**
- train_step()과 동일한 canonical transformation 적용

**Visualization 수정:**
```python
# Visualization (metric depth로 변환!)
with torch.no_grad():
    # Model prediction (canonical inverse)
    pred_inverse_canonical = model(...)  # [B, 1, H, W]

    # Convert to metric depth for visualization
    pred_metric_canonical = 100.0 / (pred_inverse_canonical + 1e-8)
    pred_metric = pred_metric_canonical * (CANONICAL_FX / fx_actual)

    # GT metric depth
    gt_metric = 1.0 / (gt_depth + 1e-8)

    # Visualize
    self.visualizer.create_validation_summary(
        ...,
        pred_depth=pred_metric,  # metric space!
        gt_depth=gt_metric
    )
```

#### **B-2. train_gear3.py**
- train_gear3_upgrade.py와 **완전히 동일**하게 수정

#### **B-3. train_gear2.py**
- Gear2 구조 확인 후 동일 로직 적용

### 5.4 C. 테스트 스크립트 수정

#### **C-1. test_gear3_upgrade.py (메인!)**

**테스트 루프 수정:**
```python
def test(self):
    CANONICAL_FX = self.config.get('canonical_fx', 1000.0)

    for batch in self.test_loader:
        images, gt_depth, focal_lengths, dataset_idx = batch  # ← fx 추가
        images = images.to(self.device)
        gt_depth = gt_depth.to(self.device)
        focal_lengths = focal_lengths.to(self.device)

        B, T = images.shape[:2]

        # Initialize Mamba
        if hasattr(self.model, 'mamba'):
            self.model.mamba.start_new_sequence()

        # Process all frames
        for t in range(T):
            img_t = images[:, t]
            gt_t = gt_depth[:, t]
            fx_t = focal_lengths[:, t]  # [B]

            # Forward pass (canonical inverse output)
            pred_inverse_canonical = self.model(img_t)  # [B, 1, H, W]

            # ===== Convert to Metric Depth =====
            # 1. Canonical inverse → canonical metric
            pred_metric_canonical = 100.0 / (pred_inverse_canonical + 1e-8)

            # 2. Canonical metric → actual metric
            fx_expanded = fx_t.view(B, 1, 1, 1)
            pred_metric = pred_metric_canonical * (CANONICAL_FX / fx_expanded)

            # GT metric depth
            gt_metric = 1.0 / (gt_t + 1e-8)

            # ===== Compute Metrics =====
            # Now in metric space!
            self.metrics.update(pred_metric, gt_metric)

            # ===== Visualization =====
            # Save as metric depth
            save_depth_visualization(pred_metric, gt_metric, ...)

    # Final results
    results = self.metrics.compute()
    return results
```

#### **C-2. test_gear3.py**
- test_gear3_upgrade.py와 동일하게 수정

#### **C-3. test_gear2.py**
- 동일 로직 적용

### 5.5 D. Config 파일 수정

**모든 gear config에 추가:**
- `configs/gear3_upgrade/config.yaml`
- `configs/gear3/config.yaml`
- `configs/gear2/config.yaml`

```yaml
# Canonical Space Configuration
canonical_fx: 1000.0  # 기본값

# Alternative options:
# canonical_fx: 518.8579  # Metric3D 값
# canonical_fx: 500.0     # 학습 데이터 중간값
# canonical_fx: 800.0     # 학습 데이터 평균
```

---

## 6. 검증 방법

### 6.1 Unit Test: Sanity Check

**테스트 시나리오:**
```python
# test_canonical_space.py
import torch
import torch.nn.functional as F

def test_canonical_invariance():
    """
    동일 장면을 다른 fx로 변환했을 때
    최종 metric depth 예측이 같아야 함
    """
    CANONICAL_FX = 1000.0

    # 원본 이미지, GT depth
    img_original = load_test_image()  # [1, 3, 518, 518]
    gt_metric = 10.0  # 10m 거리 물체
    fx_original = 500.0

    # ===== Scenario 1: fx=500 =====
    # GT canonical space 변환
    gt_metric_canonical_1 = gt_metric * (fx_original / CANONICAL_FX)
    # = 10 × 0.5 = 5.0m (canonical)
    gt_inverse_canonical_1 = 100.0 / gt_metric_canonical_1
    # = 100 / 5 = 20.0 (100/m)

    # 모델 예측
    pred_inverse_canonical_1 = model(img_original)  # ~20.0

    # Metric으로 변환
    pred_metric_canonical_1 = 100.0 / pred_inverse_canonical_1
    pred_metric_1 = pred_metric_canonical_1 * (CANONICAL_FX / fx_original)
    # = (100/20) × (1000/500) = 5 × 2 = 10.0m ✅

    # ===== Scenario 2: fx=2000 (4배 zoom) =====
    fx_zoom = 2000.0

    # 이미지를 4배 확대 (fx=2000처럼 보이도록)
    img_zoom = F.interpolate(img_original, scale_factor=4.0)

    # GT canonical space 변환
    gt_metric_canonical_2 = gt_metric * (fx_zoom / CANONICAL_FX)
    # = 10 × 2.0 = 20.0m (canonical)
    gt_inverse_canonical_2 = 100.0 / gt_metric_canonical_2
    # = 100 / 20 = 5.0 (100/m)

    # 모델 예측
    pred_inverse_canonical_2 = model(img_zoom)  # ~5.0

    # Metric으로 변환
    pred_metric_canonical_2 = 100.0 / pred_inverse_canonical_2
    pred_metric_2 = pred_metric_canonical_2 * (CANONICAL_FX / fx_zoom)
    # = (100/5) × (1000/2000) = 20 × 0.5 = 10.0m ✅

    # Assertion
    assert torch.allclose(pred_metric_1, pred_metric_2, atol=0.5), \
        f"Predictions should be similar: {pred_metric_1} vs {pred_metric_2}"

    print("✅ Canonical space invariance test passed!")
```

### 6.2 Integration Test: Cross-Dataset 성능

**비교 실험:**
```bash
# Before: fx 정보 없이 학습
python train_gear3_upgrade.py canonical_fx=null

# After: Canonical space 적용
python train_gear3_upgrade.py canonical_fx=1000.0

# 테스트: 다양한 fx 데이터셋
python test_gear3_upgrade.py \
    --checkpoint results/canonical/best.pth \
    --test-datasets tartanair,spring,waymo
```

**예상 결과:**
- Before: TartanAir (fx=320)에서 성능 저하
- After: 모든 데이터셋에서 균일한 성능

### 6.3 Ablation Study: Canonical FX 선택

**실험:**
```bash
# Variant 1: CANONICAL_FX=500
python train_gear3_upgrade.py canonical_fx=500.0

# Variant 2: CANONICAL_FX=1000 (Metric3D)
python train_gear3_upgrade.py canonical_fx=1000.0

# Variant 3: CANONICAL_FX=1500
python train_gear3_upgrade.py canonical_fx=1500.0
```

**분석:**
- 학습 데이터 fx 분포에 따라 최적값 결정
- 중간값 vs 평균값 비교

---

## 7. 주의사항

### 7.1 ⚠️ 학습 Target 변화

**Before (fx 무관):**
```python
gt_inverse = gt_depth × 100  # (1/m) × 100 = 100/m
```

**After (fx 의존):**
```python
gt_inverse_canonical = 100 × CANONICAL_FX × gt_depth / fx
```

**결과:**
- 학습 target이 완전히 바뀜!
- **기존 checkpoint와 호환 불가능**
- **처음부터 재학습 필수!**

### 7.2 ⚠️ Visualization 주의

**Canonical inverse depth는 물리적 의미 없음:**
```python
# ❌ 잘못된 시각화
pred_inverse_canonical = model(img)
visualize(pred_inverse_canonical)  # 숫자가 이상하게 나옴!

# ✅ 올바른 시각화
pred_inverse_canonical = model(img)
pred_metric = CANONICAL_FX / fx × (100.0 / pred_inverse_canonical)
visualize(pred_metric)  # metric depth (m)
```

**이유:**
- Canonical space는 "가상의 카메라" 기준
- 실제 거리와 다름
- 항상 metric으로 변환 후 시각화!

### 7.3 ⚠️ Focal Length 범위 검증

**비정상적인 fx 값 필터링:**
```python
def validate_focal_length(fx, width):
    """
    Focal length가 합리적인 범위인지 검증

    일반적 범위: 0.5 × width < fx < 2.0 × width
    """
    min_fx = width * 0.3  # 초광각 (FOV ~120도)
    max_fx = width * 2.0  # 망원 (FOV ~30도)

    if fx < min_fx or fx > max_fx:
        logger.warning(
            f"Abnormal focal length detected: fx={fx:.1f}, width={width}. "
            f"Expected range: [{min_fx:.1f}, {max_fx:.1f}]. "
            f"Using default: {width * 0.7:.1f}"
        )
        return width * 0.7  # Default: FOV ~60도

    return fx
```

### 7.4 ⚠️ Dataloader 변경사항

**모든 코드에서 batch unpacking 수정 필요:**

**Before:**
```python
images, gt_depth, dataset_idx = batch
```

**After:**
```python
images, gt_depth, focal_lengths, dataset_idx = batch  # ← fx 추가!
```

**영향 받는 코드:**
- train_gear*.py의 train_step(), validate()
- test_gear*.py의 test()
- 모든 visualization 코드

### 7.5 ⚠️ 역호환성

**기존 코드와의 호환:**
```python
# Config에 canonical_fx가 없으면 기존 방식 사용
CANONICAL_FX = self.config.get('canonical_fx', None)

if CANONICAL_FX is None:
    # Legacy mode: fx 정보 무시
    gt_inverse = gt_depth * 100.0
else:
    # Canonical mode: fx 정규화
    gt_inverse_canonical = 100.0 * CANONICAL_FX * gt_depth / fx
```

---

## 8. 사용법

### 8.1 학습

**기본 사용:**
```bash
# Canonical space 적용 (fx=1000)
torchrun --nproc_per_node=8 train_gear3_upgrade.py \
    --config-path configs/gear3_upgrade/ \
    canonical_fx=1000.0
```

**Canonical FX 변경:**
```bash
# Metric3D 값 사용
torchrun --nproc_per_node=8 train_gear3_upgrade.py \
    canonical_fx=518.8579

# 학습 데이터 중간값
torchrun --nproc_per_node=8 train_gear3_upgrade.py \
    canonical_fx=500.0
```

**Legacy 모드 (fx 무시):**
```bash
# canonical_fx=null로 설정하면 기존 방식
torchrun --nproc_per_node=8 train_gear3_upgrade.py \
    canonical_fx=null
```

### 8.2 테스트

**기본 사용:**
```bash
python test_gear3_upgrade.py \
    --config-path configs/gear3_upgrade/ \
    --checkpoint results/canonical/best.pth \
    canonical_fx=1000.0
```

**주의:** 학습 시와 동일한 `canonical_fx` 사용 필수!

### 8.3 Config 파일 설정

**configs/gear3_upgrade/config.yaml:**
```yaml
# Model configuration
model:
  encoder: vitl
  use_mamba: true
  # ...

# Canonical space configuration
canonical_fx: 1000.0  # 학습 시 사용한 값

# Training configuration
training:
  batch_size: 4
  iterations: 50000
  # ...
```

---

## 9. Troubleshooting

### Q1. "No valid GT & Pred pixels in batch!" 에러

**원인:**
- Focal length 변환으로 GT 범위가 변경됨
- Threshold (70m) 벗어날 수 있음

**해결:**
```python
# train_gear3.py에서 threshold 조정
if self.global_step < 100:
    MIN_INVERSE_DEPTH = 100.0 / 200.0  # Warmup: 200m
else:
    MIN_INVERSE_DEPTH = 100.0 / 100.0  # Normal: 100m (기존 70m에서 완화)
```

### Q2. Visualization이 이상함 (depth 값이 너무 크거나 작음)

**원인:**
- Canonical inverse depth를 직접 시각화
- Metric으로 변환 안 함

**해결:**
```python
# ❌ 잘못
visualize(pred_inverse_canonical)

# ✅ 올바름
pred_metric = CANONICAL_FX / fx × (100.0 / pred_inverse_canonical)
visualize(pred_metric)
```

### Q3. "focal_length not found in batch" 에러

**원인:**
- CombinedDataset에서 fx 반환 안 함
- Collate function 수정 안 됨

**해결:**
1. `combined_dataset.py` 수정 확인
2. 모든 dataset의 `get_focal_length()` 구현 확인
3. Collate function에서 fx 처리 확인

### Q4. 기존 checkpoint 사용 가능?

**답변:** 불가능

**이유:**
- 학습 target이 바뀜 (fx 의존)
- 모델 출력도 canonical space로 변경
- Weight 자체는 호환되지만, 예측값 해석이 달라짐

**대안:**
1. 처음부터 재학습
2. 또는 legacy mode (canonical_fx=null)로 기존 checkpoint 사용

### Q5. 어떤 CANONICAL_FX 값을 선택해야 하나?

**권장 사항:**

1. **Metric3D 따라가기:** 518.8579
   - 검증된 값
   - 논문에서 사용

2. **학습 데이터 중간값:** 500-800
   - TartanAir (320) + Spring (500) + Waymo (1500) → median ≈ 500
   - 데이터셋 분포에 최적화

3. **Round number:** 1000.0
   - 계산 단순
   - 해석 용이

**실험 권장:**
```bash
# 여러 값으로 ablation study
for fx in 500 518 800 1000 1500; do
    torchrun train_gear3_upgrade.py canonical_fx=$fx
done
```

---

## 10. 참고 자료

### 논문
1. **Metric3D v2 (CVPR 2024)**
   - "A Versatile Monocular Geometric Foundation Model"
   - Canonical space normalization 사용
   - https://arxiv.org/abs/2404.15506

2. **Depth Pro (Apple, 2024)**
   - "Sharp Monocular Metric Depth in Less Than a Second"
   - FX를 입력으로 받지 않고 FOV 추정
   - https://arxiv.org/abs/2410.02073

3. **DMD (DeepMind, 2023)**
   - "Diffusion for Metric Depth"
   - FOV conditioning
   - Logarithmic scale depth parameterization

### 관련 이슈
- Inverse vs Metric space loss: https://github.com/isl-org/MiDaS/issues/23
- Camera calibration for depth estimation: https://github.com/facebookresearch/Replica-Dataset/issues/43

### 데이터셋 문서
- TartanAir: https://tartanair.org/modalities.html
- Spring: https://spring-benchmark.org/faq
- Waymo: https://github.com/waymo-research/waymo-open-dataset/issues/361
- Dynamic Replica: https://dynamic-stereo.github.io/

---

## 변경 이력

- **2025-01-XX**: 초안 작성
- **2025-01-XX**: 수학적 분석 추가 (Inverse vs Metric loss)
- **2025-01-XX**: 구현 계획 상세화

---

## TODO

- [ ] PointOdyssey focal length 조사
- [ ] nuScenes focal length 조사
- [ ] Sintel focal length 조사
- [ ] MVS-Synth camera params 파일 확인
- [ ] Unit test 작성 (test_canonical_space.py)
- [ ] Ablation study 실행 (다양한 CANONICAL_FX)
- [ ] Cross-dataset 성능 비교 (Before/After)

---

**끝.**
