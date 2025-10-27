# Object-wise Depth Evaluation (객체별 깊이 추정 평가)

이 문서는 특정 객체 타입(차량, 보행자, 자전거 등)에 대한 깊이 추정 정확도 개선을 입증하기 위한 객체별 깊이 평가 시스템을 설명합니다.

## 개요

객체별 평가 시스템은 각 세그멘테이션 클래스별로 깊이 메트릭을 개별 계산하여 다음을 가능하게 합니다:

1. **어떤 객체가 가장 큰 이득을 얻는지 식별** - Gear3의 attention 메커니즘으로부터
2. **객체 타입별 모델 비교** - 예: Gear3 vs Baseline의 차량, 보행자 등에 대한 성능
3. **상세 리포트 생성** - 클래스별 성능 개선 표시
4. **객체별 개선 시각화** - 깊이 추정 품질 향상 시각화

## 구성 요소

### 1. `utils/object_wise_evaluation.py`

세그멘테이션 클래스별로 깊이 정확도를 계산하는 핵심 메트릭 계산 모듈.

**기능**:
- 다중 데이터셋 지원: KITTI, Cityscapes, NYU Depth V2, ScanNet, VKITTI2, Waymo, Sintel
- 클래스별 표준 메트릭 계산: MAE, RMSE, AbsRel, δ1/δ2/δ3
- 비디오 시퀀스 전체에 대한 메트릭 집계
- 모델 간 비교하여 개선율(%) 표시
- JSON 파일로 결과 저장

**사용 예시**:
```python
from utils.object_wise_evaluation import ObjectWiseMetrics

# Waymo용 초기화
evaluator = ObjectWiseMetrics(dataset_type='waymo')

# 한 프레임에 대한 메트릭 계산
class_metrics = evaluator.compute_metrics_per_class(
    pred_depth,   # (H, W) 예측 깊이
    gt_depth,     # (H, W) ground truth 깊이
    seg_mask,     # (H, W) 세그멘테이션 마스크
    min_pixels=100  # 픽셀 수가 적은 클래스 스킵
)

# 여러 프레임에 대해 집계
all_metrics = [class_metrics_frame1, class_metrics_frame2, ...]
aggregated = evaluator.aggregate_metrics(all_metrics)

# 두 모델 비교
comparison = evaluator.compare_models(
    baseline_metrics, gear3_metrics,
    model_a_name="Baseline", model_b_name="Gear3"
)

# 출력 및 저장
evaluator.print_summary(aggregated, comparison=comparison)
evaluator.save_results(aggregated, output_path, comparison=comparison)
```

### 2. `test_object_wise.py`

데이터셋에서 객체별 메트릭을 실행하기 위한 종단간(end-to-end) 평가 스크립트.

**명령줄 사용법**:
```bash
# 단일 모델 평가
python test_object_wise.py \
    --model-checkpoint train_results/results_14/gear_3/phase_1/best.pth \
    --config-path configs/gear3 \
    --dataset waymo \
    --data-root /home/cvlab/hsy/Datasets/waymo_segmentation \
    --results-dir test_results/object_wise/gear3_waymo \
    --gpu 0

# 두 모델 비교
python test_object_wise.py \
    --model-checkpoint train_results/results_14/gear_3/phase_1/best.pth \
    --baseline-checkpoint train_results/results_14/gear_2/phase_1/best.pth \
    --config-path configs/gear3 \
    --dataset waymo \
    --data-root /home/cvlab/hsy/Datasets/waymo_segmentation \
    --results-dir test_results/object_wise/gear3_vs_gear2 \
    --gpu 0 \
    --max-sequences 50
```

**인자**:
- `--model-checkpoint`: Gear3 체크포인트 경로
- `--baseline-checkpoint`: (선택) 비교용 baseline 체크포인트 경로
- `--config-path`: 모델 설정 디렉토리
- `--dataset`: 데이터셋 타입 (`waymo`, `sintel`, `kitti`, `cityscapes`, `nyu`, `vkitti2`)
- `--data-root`: 데이터셋 루트 디렉토리
- `--results-dir`: 결과 출력 디렉토리
- `--gpu`: GPU 장치 ID
- `--max-sequences`: 평가할 최대 시퀀스 수 (기본: 전체)
- `--video-length`: 비디오 시퀀스 길이 (기본: 5)
- `--batch-size`: 배치 크기 (기본: 1)

### 3. Gear2/3/3_upgrade 테스트 스크립트의 Object-wise 모드

`test_gear2.py`, `test_gear3.py`, `test_gear3_upgrade.py`는 모두 object-wise 평가를 지원합니다.

**Docker 명령어로 실행**:
```bash
# Gear2 객체별 평가 (Waymo)
OBJWISE_DATASET=waymo ./run_docker.sh test_gear2_objwise

# Gear3 객체별 평가 (Sintel)
OBJWISE_DATASET=sintel ./run_docker.sh test_gear3_objwise

# Gear3 Upgrade 객체별 평가 (Waymo, multi_layer 분리)
OBJWISE_DATASET=waymo ./run_docker.sh test_gear3_upgrade_objwise --separation multi_layer
```

**데이터셋 설정 방법**:
```bash
# 환경변수로 데이터셋 지정
OBJWISE_DATASET=waymo ./run_docker.sh test_gear3_objwise

# 또는 기본값 사용 (waymo)
./run_docker.sh test_gear3_objwise

# 지원 데이터셋: waymo, sintel
```

### 4. 데이터셋 로더

#### `dataloaders/waymo_segmentation_dataset.py`

Waymo Open Dataset용 세그멘테이션 데이터 로더 (현재 구현됨).

**기능**:
- RGB 이미지, 깊이 맵, 세그멘테이션 마스크 로드
- 시간적 시퀀스 처리 (비디오 프레임)
- 일관된 해상도로 리사이즈 (518x518)
- 인스턴스 ID를 클래스 ID로 매핑
- 배치 처리를 위한 커스텀 collate 함수

**데이터셋 구조**:
```
waymo_segmentation/
├── train/
│   ├── sequence_0/
│   │   ├── images/
│   │   │   ├── 000000.jpg
│   │   │   ├── 000001.jpg
│   │   │   └── ...
│   │   ├── depth/
│   │   │   ├── 000000.npy
│   │   │   ├── 000001.npy
│   │   │   └── ...
│   │   └── segmentation/
│   │       ├── 000000.png
│   │       ├── 000001.png
│   │       └── ...
│   └── sequence_1/
│       └── ...
└── val/
    └── ...
```

**세그멘테이션 클래스 (Waymo)**:
- 0: 배경 (Background)
- 1: 차량 (Vehicle)
- 2: 보행자 (Pedestrian)
- 3: 사이클리스트 (Cyclist)
- 4: 표지판 (Sign)

#### `dataloaders/sintel_segmentation_dataset.py`

MPI Sintel Dataset용 세그멘테이션 데이터 로더 (현재 구현됨).

**데이터셋 구조**:
```
sintel_segmentation/
├── training/
│   ├── final/           # RGB images
│   │   ├── alley_1/
│   │   │   ├── frame_0001.png
│   │   │   └── ...
│   │   └── ...
│   ├── depth/           # Depth maps
│   │   ├── alley_1/
│   │   │   ├── frame_0001.dpt
│   │   │   └── ...
│   │   └── ...
│   └── segmentation/    # Segmentation masks
│       ├── alley_1/
│       │   ├── frame_0001.png
│       │   └── ...
│       └── ...
└── test/
    └── ...
```

**세그멘테이션 클래스 (Sintel)**:
- 0: 배경 (Background)
- 1: 전경 객체 (Foreground Object)

#### `dataloaders/kitti_segmentation_dataset.py`

KITTI용 세그멘테이션 데이터 로더 (참고 구현).

**데이터셋 구조**:
```
KITTI/
├── raw/                              # RGB 이미지
│   └── 2011_09_26/
│       └── 2011_09_26_drive_0001_sync/
│           └── image_02/data/
│               ├── 0000000000.png
│               └── ...
├── depth/                            # LiDAR 깊이 맵
│   └── 2011_09_26_drive_0001_sync/
│       └── proj_depth/groundtruth/image_02/
│           ├── 0000000000.png
│           └── ...
└── segmentation/                     # 세그멘테이션 마스크
    └── 2011_09_26_drive_0001_sync/
        └── image_02/
            ├── 0000000000.png
            └── ...
```

## 지원 데이터셋

### 1. Waymo Open Dataset (구현 완료 ✅)

**클래스**: 배경, 차량, 보행자, 사이클리스트, 표지판 (5개 클래스)

**세그멘테이션 획득**:
- Waymo Open Dataset의 공식 세그멘테이션 레이블 사용
- 또는 SAM (Segment Anything Model)으로 생성

**데이터**: LiDAR 깊이 (최대 75m, 밀집)

**사용법**:
```bash
OBJWISE_DATASET=waymo ./run_docker.sh test_gear3_objwise
```

### 2. MPI Sintel (구현 완료 ✅)

**클래스**: 배경, 전경 객체 (2개 클래스)

**세그멘테이션 획득**:
- Sintel의 공식 optical flow와 occlusion 마스크 활용
- 또는 SAM으로 생성

**데이터**: 렌더링 깊이 (완벽한 ground truth, 밀집)

**사용법**:
```bash
OBJWISE_DATASET=sintel ./run_docker.sh test_gear3_objwise
```

### 3. KITTI (TODO)

**클래스**: 배경, 차량, 보행자, 사이클리스트 (인스턴스 세그멘테이션)

**세그멘테이션 획득**:
- Option A: KITTI 인스턴스 세그멘테이션 레이블 다운로드
- Option B: Semantic KITTI 레이블 사용
- Option C: SAM으로 생성

**데이터**: LiDAR 깊이 (희소, 최대 80m)

### 4. Cityscapes (TODO)

**클래스**: 19개 시맨틱 클래스 (도로, 인도, 건물, 차량, 사람 등)

**세그멘테이션 획득**: Cityscapes 시맨틱/인스턴스 세그멘테이션 다운로드

**데이터**: 스테레오 깊이 (밀집, 최대 50m)

### 5. NYU Depth V2 (TODO)

**클래스**: 40개 실내 클래스 (침대, 의자, 테이블, 벽, 바닥 등)

**세그멘테이션 획득**: NYU Depth V2 데이터셋에 포함

**데이터**: Kinect RGB-D (밀집, 실내 장면)

### 6. VKITTI2 (TODO)

**클래스**: 13개 클래스 (지형, 나무, 건물, 도로, 차량, 밴, 트럭 등)

**세그멘테이션 획득**: 완벽한 합성 세그멘테이션 포함

**데이터**: 완벽한 합성 깊이 (밀집, 최대 100m)

## Docker로 실행하기

### run_docker.sh에서 데이터셋 설정하는 방법

**환경변수 `OBJWISE_DATASET` 사용**:

```bash
# Waymo 데이터셋으로 Gear3 평가
OBJWISE_DATASET=waymo ./run_docker.sh test_gear3_objwise

# Sintel 데이터셋으로 Gear2 평가
OBJWISE_DATASET=sintel ./run_docker.sh test_gear2_objwise

# Waymo 데이터셋으로 Gear3 Upgrade 평가 (multi_layer 분리 방법)
OBJWISE_DATASET=waymo ./run_docker.sh test_gear3_upgrade_objwise --separation multi_layer

# 기본값 사용 (waymo)
./run_docker.sh test_gear3_objwise
```

**지원 데이터셋 값**:
- `waymo` - Waymo Open Dataset (기본값)
- `sintel` - MPI Sintel Dataset

**데이터 경로 자동 설정**:

스크립트는 자동으로 데이터 경로를 다음과 같이 설정합니다:
```
/data/datasets/{OBJWISE_DATASET}_seg
```

예시:
- `waymo` → `/data/datasets/waymo_seg`
- `sintel` → `/data/datasets/sintel_seg`

Docker Compose 설정에서 호스트 경로를 마운트해야 합니다:
```yaml
# docker-compose.yml
volumes:
  - /home/cvlab/hsy/Datasets:/data/datasets
```

### 전체 명령어 예시

```bash
# 1. Gear2 - Waymo 객체별 평가
OBJWISE_DATASET=waymo ./run_docker.sh test_gear2_objwise \
  --results-dir test_results/gear2_waymo_objwise \
  --gpu 0

# 2. Gear3 - Sintel 객체별 평가
OBJWISE_DATASET=sintel ./run_docker.sh test_gear3_objwise \
  --results-dir test_results/gear3_sintel_objwise \
  --gpu 1

# 3. Gear3 Upgrade - Waymo 객체별 평가 (CLS 분리)
OBJWISE_DATASET=waymo ./run_docker.sh test_gear3_upgrade_objwise \
  --separation cls_seg \
  --results-dir test_results/gear3_upgrade_waymo_cls \
  --gpu 0

# 4. 커스텀 체크포인트 지정
OBJWISE_DATASET=waymo ./run_docker.sh test_gear3_objwise \
  --flashdepth-checkpoint train_results/results_14/gear_3/phase_1/checkpoint_step_25000.pth \
  --results-dir test_results/gear3_waymo_step25k \
  --gpu 0
```

## 데이터셋 준비하기

### Waymo Open Dataset 준비

1. **Waymo Open Dataset 다운로드**:
   - https://waymo.com/open/download/
   - Perception dataset (카메라 이미지, LiDAR)
   - Segmentation labels (optional, 없으면 SAM 사용)

2. **데이터 추출 및 구성**:
   ```bash
   # TFRecord에서 이미지/깊이/세그멘테이션 추출
   python scripts/extract_waymo_data.py \
       --input waymo_open_dataset_v_1_4_0/ \
       --output /home/cvlab/hsy/Datasets/waymo_segmentation/
   ```

3. **디렉토리 구조 확인**:
   ```
   waymo_segmentation/
   ├── train/
   │   ├── sequence_0/
   │   │   ├── images/
   │   │   ├── depth/
   │   │   └── segmentation/
   │   └── ...
   └── val/
       └── ...
   ```

### Sintel 준비

1. **MPI Sintel 다운로드**:
   - http://sintel.is.tue.mpg.de/downloads
   - Final pass (clean RGB images)
   - Depth maps
   - Optional: Occlusion masks for segmentation

2. **세그멘테이션 생성** (없는 경우):
   ```bash
   # SAM으로 세그멘테이션 생성
   python scripts/generate_sam_masks.py \
       --images sintel/training/final/ \
       --output sintel_segmentation/training/segmentation/ \
       --model-type vit_h
   ```

3. **디렉토리 구조 확인**:
   ```
   sintel_segmentation/
   ├── training/
   │   ├── final/
   │   ├── depth/
   │   └── segmentation/
   └── test/
       └── ...
   ```

### KITTI 준비 (TODO)

1. **KITTI Raw Data 다운로드** (RGB 이미지):
   - http://www.cvlibs.net/datasets/kitti/raw_data.php
   - `KITTI/raw/`에 추출

2. **KITTI Depth 다운로드** (LiDAR ground truth):
   - http://www.cvlibs.net/datasets/kitti/eval_depth.php
   - `KITTI/depth/`에 추출

3. **세그멘테이션 마스크 획득** (3가지 옵션):
   - **Option A**: KITTI instance segmentation benchmark
   - **Option B**: Semantic KITTI labels
   - **Option C**: SAM으로 생성

## 평가 실행하기

### Step 1: 데이터셋 준비

위의 데이터셋 준비 지침에 따라 RGB, 깊이, 세그멘테이션 데이터를 준비합니다.

### Step 2: Docker로 평가 실행

```bash
# Waymo에서 Gear3 평가
OBJWISE_DATASET=waymo ./run_docker.sh test_gear3_objwise \
    --results-dir test_results/object_wise/gear3_waymo \
    --gpu 0
```

### Step 3: 결과 분석

결과는 `{results_dir}/object_wise_results.json`에 저장됩니다:

```json
{
  "dataset_type": "waymo",
  "per_class_metrics": {
    "vehicle": {
      "mae": 2.345,
      "rmse": 3.456,
      "abs_rel": 0.123,
      "a1": 0.875,
      "num_pixels": 125000,
      "num_frames": 50
    },
    "pedestrian": {
      "mae": 1.234,
      "rmse": 2.345,
      "abs_rel": 0.098,
      "a1": 0.912,
      "num_pixels": 45000,
      "num_frames": 50
    },
    "cyclist": {
      "mae": 1.567,
      "rmse": 2.678,
      "abs_rel": 0.105,
      "a1": 0.898,
      "num_pixels": 15000,
      "num_frames": 35
    }
  },
  "overall_metrics": {
    "mae": 1.982,
    "rmse": 2.876,
    "abs_rel": 0.109,
    "a1": 0.895
  }
}
```

## Gear3에서 예상되는 개선

Gear3의 attention 기반 전경/배경 분리를 기반으로 한 예상 개선:

### 높은 개선 예상:
- **차량 (Vehicle)** - 공간적 attention을 받는 움직이는 객체
- **보행자 (Pedestrian)** - 전경의 두드러진 객체
- **사이클리스트 (Cyclist)** - 움직이는 전경 객체

### 중간 개선 예상:
- **건물 (Building)** - 정적 배경, 장면에 따라 다름
- **표지판 (Sign)** - 크기가 작지만 중요한 객체

### 낮은 개선 예상:
- **배경 (Background)** - 일반적인 정적 영역
- **하늘 (Sky)** - 깊이 없음, 종종 마스킹됨

## SAM 마스크 생성 (선택사항)

데이터셋에 세그멘테이션 레이블이 없는 경우, Segment Anything Model로 생성할 수 있습니다:

```python
# scripts/generate_sam_masks.py
import numpy as np
from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
import torch

# SAM 모델 로드
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
sam.to(device="cuda")
mask_generator = SamAutomaticMaskGenerator(sam)

# 이미지 처리
for image_path in image_paths:
    image = np.array(Image.open(image_path))
    masks = mask_generator.generate(image)

    # 마스크 저장 (인스턴스 ID)
    seg_mask = np.zeros(image.shape[:2], dtype=np.uint16)
    for i, mask in enumerate(masks):
        seg_mask[mask['segmentation']] = i + 1

    # 저장
    output_path = output_dir / image_path.name
    Image.fromarray(seg_mask).save(output_path)
```

**실행**:
```bash
python scripts/generate_sam_masks.py \
    --images /data/datasets/waymo/images/ \
    --output /data/datasets/waymo_segmentation/segmentation/ \
    --model-type vit_h \
    --checkpoint sam_vit_h.pth
```

## 문제 해결

### 이슈: "Segmentation not found for sequence"

**해결책**: 세그멘테이션 디렉토리 구조가 데이터셋 형식과 일치하는지 확인:
```
{dataset}_segmentation/{split}/{sequence_name}/segmentation/{frame_id}.png
```

### 이슈: "Too few pixels for class"

**해결책**: `compute_metrics_per_class()`에서 `min_pixels` 임계값을 낮추거나 시퀀스 수를 늘림.

### 이슈: "No sequences found"

**해결책**: 데이터셋 경로와 구조를 확인. 동일한 시퀀스에 대해 깊이와 세그멘테이션이 존재하는지 확인.

### 이슈: 평가 중 메모리 오류

**해결책**: `--batch-size`를 1로 줄이거나 `--max-sequences`를 사용하여 평가 크기 제한.

### 이슈: Docker에서 데이터셋을 찾을 수 없음

**해결책**: `docker-compose.yml`에서 볼륨 마운트 확인:
```yaml
volumes:
  - /home/cvlab/hsy/Datasets:/data/datasets
```

그리고 호스트 경로에 데이터가 있는지 확인:
```bash
ls /home/cvlab/hsy/Datasets/waymo_seg/
ls /home/cvlab/hsy/Datasets/sintel_seg/
```

## 향후 개선사항

- [ ] Cityscapes 데이터셋 로더 구현
- [ ] NYU Depth V2 데이터셋 로더 구현
- [ ] ScanNet 데이터셋 로더 구현
- [ ] VKITTI2 데이터셋 로더 구현
- [ ] SAM 마스크 생성 스크립트 추가
- [ ] 클래스별 개선 시각화 추가
- [ ] 개선이 발생하는 위치를 보여주는 공간 히트맵
- [ ] Multi-GPU 평가 지원
- [ ] 객체 클래스별 시간적 일관성 메트릭 추가

## 인용

이 객체별 평가 시스템을 사용하는 경우 다음을 인용해주세요:

```bibtex
@article{flashdepth2024,
  title={FlashDepth: Real-time Monocular Depth Estimation with Temporal Processing},
  author={...},
  year={2024}
}
```
