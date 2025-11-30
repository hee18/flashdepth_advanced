# FGWise Evaluation Implementation Plan

## 개요
fg_mask를 활용한 foreground-wise depth evaluation 구현 계획

## 선행 작업 (완료 필요)
```bash
# fg_mask 생성 스크립트 실행
python scripts/generate_fg_masks.py \
    --data-root /path/to/datasets \
    --checkpoint configs/flashdepth-l/iter_10001.pth \
    --datasets eth3d,sintel,waymo_seg,vkitti,unreal4k \
    --gpu 0
```

## 생성된 fg_mask 경로
```
data_root/
├── eth3d/{Scene}/fg_masks/{frame}.png
├── sintel/fg_masks/training/clean/{scene}/{frame}.png
├── waymo_seg/val/{segment}/FRONT/fg_masks/{frame}.png
├── vkitti/{Scene}/clone/frames/fg_masks/Camera_0/fg_{frame}.png
└── unreal4k/UnrealStereo4K_{seq}/fg_masks/{frame}.png
```

---

## Phase 1: 데이터로더 수정

### 1.1 BaseDatasetPairs 수정
**파일**: `dataloaders/base_dataset_pairs.py`

```python
# __getitem__에 fg_mask 로딩 추가
def __getitem__(self, idx):
    ...
    # 기존 코드
    images, depths, intrinsics = self._load_sequence(...)

    # NEW: fg_mask 로딩 (옵션)
    if self.load_fg_mask:
        fg_masks = self._load_fg_masks(seq_path, frame_indices)
        return images, depths, intrinsics, fg_masks

    return images, depths, intrinsics
```

### 1.2 각 데이터셋 로더 수정
- `eth3d_dataset.py`
- `sintel_dataset.py`
- `waymo_segmentation_dataset.py`
- `vkitti_segmentation_dataset.py`
- `unreal4k_dataset.py`

각 로더에 `_get_fg_mask_path()` 메서드 추가

---

## Phase 2: test_gear5.py FGWise 평가 추가

### 2.1 FGWise Metrics 클래스
**위치**: `test_gear5.py` 또는 새 파일 `flashdepth/fgwise_metrics.py`

```python
class FGWiseMetrics:
    """Foreground-wise depth evaluation metrics."""

    def compute(self, pred_depth, gt_depth, fg_mask, valid_mask):
        """
        Args:
            pred_depth: [H, W] predicted depth
            gt_depth: [H, W] ground truth depth
            fg_mask: [H, W] binary mask (255=foreground)
            valid_mask: [H, W] valid depth mask

        Returns:
            dict with fg_mae, fg_rmse, fg_absrel, fg_delta1, etc.
        """
        # Combine masks
        fg_valid = (fg_mask > 0) & valid_mask
        bg_valid = (fg_mask == 0) & valid_mask

        # Compute metrics for FG and BG separately
        fg_metrics = self._compute_metrics(pred_depth, gt_depth, fg_valid)
        bg_metrics = self._compute_metrics(pred_depth, gt_depth, bg_valid)

        return {
            'fg_mae': fg_metrics['mae'],
            'fg_absrel': fg_metrics['absrel'],
            'bg_mae': bg_metrics['mae'],
            'bg_absrel': bg_metrics['absrel'],
            ...
        }
```

### 2.2 test_gear5.py 수정
```python
# __init__에 추가
self.fgwise_enabled = config.get('fg_wise', {}).get('enabled', False)
if self.fgwise_enabled:
    self.fgwise_metrics = FGWiseMetrics()

# test_sequence에서 fg_mask 로딩 및 평가
if self.fgwise_enabled:
    fg_mask = self._load_fg_mask(dataset_name, seq_name, frame_idx)
    fgwise_results = self.fgwise_metrics.compute(
        pred_depth, gt_depth, fg_mask, valid_mask
    )
```

### 2.3 CLI 옵션 추가
```bash
./run_docker.sh test_gear5 --fgwise --dataset waymo_seg
```

---

## Phase 3: test_comparison.py FGWise 평가 추가

### 3.1 구조
`test_comparison.py`는 여러 depth estimation 방법을 비교하므로, 각 방법에 대해 동일한 fg_mask 적용

```python
# 각 프레임 평가 시
for method in methods:
    pred_depth = method_outputs[method]

    # Global metrics
    global_metrics = compute_metrics(pred_depth, gt_depth, valid_mask)

    # FGWise metrics (NEW)
    if fgwise_enabled:
        fg_metrics = compute_fgwise_metrics(pred_depth, gt_depth, fg_mask, valid_mask)
```

### 3.2 결과 포맷
```json
{
  "method_A": {
    "mae": 0.123,
    "absrel": 0.045,
    "fg_mae": 0.089,
    "fg_absrel": 0.032,
    "bg_mae": 0.156,
    "bg_absrel": 0.058
  },
  ...
}
```

---

## Phase 4: test_video_comparison.py FGWise 평가 추가

유사하게 temporal consistency metrics에도 FG/BG 분리 적용

---

## Phase 5: Visualization

### 5.1 FG Overlay 시각화
```python
def visualize_fg_overlay(image, fg_mask, pred_depth, gt_depth):
    """FG 영역을 빨간 테두리로 표시한 depth 비교 이미지"""
    # FG contour 추출
    contours = cv2.findContours(fg_mask, ...)

    # Depth 이미지에 contour 오버레이
    vis = create_depth_comparison(pred_depth, gt_depth)
    cv2.drawContours(vis, contours, -1, (255, 0, 0), 2)

    return vis
```

### 5.2 FG/BG 분리 메트릭 차트
Bar chart로 FG vs BG 메트릭 비교 시각화

---

## 구현 우선순위

| 순서 | 작업 | 예상 시간 |
|-----|-----|----------|
| 1 | fg_mask 생성 스크립트 실행 | 수 시간 (자동) |
| 2 | FGWiseMetrics 클래스 작성 | 30분 |
| 3 | test_gear5.py에 fgwise 옵션 추가 | 1시간 |
| 4 | test_comparison.py에 fgwise 추가 | 1시간 |
| 5 | test_video_comparison.py에 fgwise 추가 | 1시간 |
| 6 | Visualization 추가 | 1시간 |

---

## 참고: objwise vs fgwise 차이

| 항목 | objwise | fgwise |
|-----|---------|--------|
| 마스크 소스 | 데이터셋 제공 segmentation | ViT attention 기반 생성 |
| 의미 | 특정 클래스 (차량, 사람 등) | 모델이 attention을 주는 영역 |
| 데이터셋 | waymo_seg, vkitti만 | 모든 데이터셋 가능 |
| 목적 | 객체별 depth 정확도 | attention-depth 상관관계 분석 |

---

## 실행 명령어 예시

```bash
# Step 1: fg_mask 생성 (한 번만)
python scripts/generate_fg_masks.py \
    --data-root /data/datasets \
    --checkpoint configs/flashdepth-l/iter_43002.pth \
    --gpu 0

# Step 2: fgwise 테스트
./run_docker.sh test_gear5 --fgwise --dataset eth3d --gpu 0
./run_docker.sh test_gear5 --fgwise --dataset sintel --gpu 0
./run_docker.sh test_gear5 --fgwise --dataset waymo_seg --gpu 0

# Step 3: comparison 테스트
python test_comparison.py --fgwise --datasets eth3d,sintel
```
