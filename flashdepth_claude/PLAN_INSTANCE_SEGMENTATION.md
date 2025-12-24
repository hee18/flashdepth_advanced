# Instance Segmentation + Depth Estimation Test Plan

## Overview
YOLOv11 인스턴스 세그멘테이션 + 트래킹을 다양한 depth 모델과 결합하여 각 객체의 depth를 프레임별로 추적하는 테스트 기능 추가.

## Docker 환경 분리
두 개의 Docker 이미지 사용:
- **flashdepth** (26.9GB): Gear5/FlashDepth 모델용
- **flashdepth_comparison** (58.1GB): Comparison 모델들용 (metric3d, unidepth, zoedepth, depthpro, vda, depthcrafter)

## User Choices
- **Segmentation Model**: yolo11x-seg (highest accuracy)
- **Tracker**: BoTSORT (built-in to YOLOv11)
- **Target Classes**: Person only (COCO class 0)
- **Camera Intrinsics**: NuScenes defaults (fx=1266.4, cx=816.3)
- **Depth Models**: Gear5 + 모든 Comparison 모델 지원

## Why YOLOv11 over SAM2
- Built-in tracking (BoTSORT/ByteTrack) - 추가 설정 불필요
- 간단한 설치 (`pip install ultralytics`)
- 실시간 성능 (30+ FPS on consumer GPUs)
- 낮은 메모리 사용량 (~20M params for yolo11x-seg)

---

## 구현 구조

### Docker 환경별 테스트 스크립트
```
flashdepth Docker (Gear5용):
├── test_instance_depth.py          # Gear5 모델 + YOLOv11
└── run_instance_test.sh            # Shell wrapper

flashdepth_comparison Docker (Comparison 모델용):
├── test_instance_comparison.py     # IMAGE 모델들 (metric3d, unidepth, zoedepth, depthpro)
├── test_instance_video_comparison.py # VIDEO 모델들 (vda, depthcrafter)
├── run_instance_comparison.sh      # IMAGE 모델용 shell wrapper
└── run_instance_video_comparison.sh # VIDEO 모델용 shell wrapper
```

### 공유 유틸리티
```
utils/
├── instance_depth_utils.py         # 마스크 처리, depth 계산 함수
└── instance_visualization.py       # 시각화 함수 (trajectory plot, video export)
```

---

## Files to Create

### 1. `test_instance_depth.py` (Gear5용 - flashdepth Docker)
**Location**: `/home/cvlab/hsy/flashdepth_advanced/flashdepth_claude/test_instance_depth.py`

**Structure** (follows test_gear5.py patterns):
```python
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flashdepth.model import FlashDepth
from flashdepth.gear5_modules import Gear5MetricHead
from utils.instance_depth_utils import (
    get_eroded_mask_and_center,
    calculate_mask_depth,
    calculate_lateral_position
)
from utils.instance_visualization import (
    create_frame_visualization,
    create_trajectory_plot,
    save_video_result
)

class InstanceDepthTester:
    def __init__(self, config):
        self.config = config
        self.device = "cuda:0"

        # Results directory (like test_gear5.py)
        self.save_dir = Path(config.get('results_dir', 'test_results/instance_depth'))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # NuScenes camera intrinsics (default)
        self.fx = config.get('fx', 1266.4)
        self.cx = config.get('cx', 816.3)  # half of 1600
        self.cy = config.get('cy', 450.0)  # half of 900

        # YOLOv11 setup
        seg_model_name = config.get('seg_model', 'yolo11x-seg.pt')
        self.yolo = YOLO(seg_model_name)
        self.tracker_config = config.get('tracker', 'botsort.yaml')
        self.person_only = config.get('person_only', True)
        self.center_mask = config.get('center_mask', True)

        # Video path
        self.video_path = config.get('video_path', '/data/datasets/videos_mfdepth')
        self.frame_interval = config.get('frame_interval', 1)

        # Setup Gear5 model (like test_gear5.py)
        self.model = self._setup_model()

    def _setup_model(self):
        """Load Gear5 model (same pattern as test_gear5.py)"""
        model_config = dict(self.config.model)
        model_config['batch_size'] = 1
        model_config['use_metric_head'] = False

        model = FlashDepth(**model_config)

        # Add Gear5 metric head
        model_embed_dim = 1024 if model.encoder == 'vitl' else 384
        use_mamba_temporal = self.config.model.get('use_mamba_temporal', False)

        model.gear5_metric_head = Gear5MetricHead(
            embed_dim=model_embed_dim,
            feature_dim=256,
            hidden_dim=128,
            use_mamba=use_mamba_temporal
        )

        # Load checkpoint (same as test_gear5.py)
        checkpoint_path = self.config.get('load')
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            # Extract state dict handling...
            model.load_state_dict(state_dict, strict=False)

        return model.to(self.device).eval()

    def process_video(self, video_path):
        """Process single video with segmentation + depth"""
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        track_trajectories = defaultdict(list)
        result_frames = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # YOLOv11 segmentation + tracking
            classes = [0] if self.person_only else None  # 0 = person
            seg_results = self.yolo.track(
                frame,
                persist=True,
                classes=classes,
                tracker=self.tracker_config
            )

            # Depth estimation with Gear5
            depth_map = self._estimate_depth(frame)

            # Process each instance
            instances_info = self._process_instances(
                seg_results[0], depth_map, frame.shape
            )

            # Update trajectories
            for inst in instances_info:
                track_trajectories[inst['track_id']].append({
                    'frame': frame_idx,
                    'depth_m': inst['depth'],
                    'lateral_m': inst['lateral_pos'],
                    'center_x': inst['center_x'],
                    'center_y': inst['center_y']
                })

            # Create visualization
            vis_frame = create_frame_visualization(frame, depth_map, instances_info)
            result_frames.append(vis_frame)

            frame_idx += 1

        cap.release()
        return track_trajectories, result_frames, fps

    def _estimate_depth(self, frame):
        """Run Gear5 inference on single frame"""
        # Preprocess frame (normalize, resize to 518x518)
        # Run through FlashDepth + Gear5 metric head
        # Return metric depth map at original resolution
        pass

    def _process_instances(self, seg_result, depth_map, frame_shape):
        """Extract depth info for each tracked instance"""
        instances_info = []

        if seg_result.masks is None or seg_result.boxes.id is None:
            return instances_info

        for mask_xy, track_id, box in zip(
            seg_result.masks.xy,
            seg_result.boxes.id.int().cpu().tolist(),
            seg_result.boxes.xyxy.cpu().numpy()
        ):
            # Create binary mask from polygon
            mask = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.uint8)
            polygon = mask_xy.astype(np.int32)
            cv2.fillPoly(mask, [polygon], 1)

            # Get eroded center mask
            if self.center_mask:
                depth_mask, center_x = get_eroded_mask_and_center(mask)
            else:
                depth_mask = mask
                M = cv2.moments(mask, binaryImage=True)
                center_x = int(M['m10'] / (M['m00'] + 1e-6))

            # Calculate depth from mask
            depth = calculate_mask_depth(depth_mask, depth_map)

            # Calculate lateral position
            lateral_pos = calculate_lateral_position(
                depth, center_x, self.fx, self.cx
            )

            M = cv2.moments(mask, binaryImage=True)
            center_y = int(M['m01'] / (M['m00'] + 1e-6))

            instances_info.append({
                'track_id': track_id,
                'depth': depth,
                'lateral_pos': lateral_pos,
                'center_x': center_x,
                'center_y': center_y,
                'box': box,
                'mask': depth_mask
            })

        return instances_info

    def test(self):
        """Main test loop"""
        video_path = Path(self.video_path)

        if video_path.is_file():
            videos = [video_path]
        else:
            videos = list(video_path.glob('*.mp4'))

        all_results = {}

        for video in videos:
            trajectories, frames, fps = self.process_video(video)

            # Save results
            video_name = video.stem
            video_save_dir = self.save_dir / video_name
            video_save_dir.mkdir(exist_ok=True)

            # Save JSON
            self._save_json_results(trajectories, video, video_save_dir)

            # Save trajectory plot
            create_trajectory_plot(
                trajectories,
                video_save_dir / 'trajectory_plot.png'
            )

            # Save video
            save_video_result(
                frames,
                video_save_dir / 'result_video.mp4',
                fps,
                self.frame_interval
            )

            all_results[video_name] = trajectories

        return all_results

@hydra.main(version_base=None, config_path="configs/gear5", config_name="config")
def main(config: DictConfig):
    config.inference = True
    tester = InstanceDepthTester(config)
    tester.test()

if __name__ == "__main__":
    main()
```

---

### 2. `test_instance_comparison.py` (IMAGE 모델용 - flashdepth_comparison Docker)
**Location**: `/home/cvlab/hsy/flashdepth_advanced/flashdepth_claude/test_instance_comparison.py`

**지원 모델**: metric3d, unidepth, zoedepth, depthpro, depthanythingv2, cut3r

**Structure** (follows test_comparison.py patterns):
```python
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
import json

# 기존 adapter 패턴 재사용
from adapters.metric3d_adapter import Metric3DAdapter
from adapters.unidepth_adapter import UniDepthAdapter
from adapters.zoedepth_adapter import ZoeDepthAdapter
from adapters.depthpro_adapter import DepthProAdapter
# ...

from utils.instance_depth_utils import (
    get_eroded_mask_and_center,
    calculate_mask_depth,
    calculate_lateral_position
)
from utils.instance_visualization import (
    create_frame_visualization,
    create_trajectory_plot,
    save_video_result
)

class InstanceComparisonTester:
    """Frame-by-frame depth model + YOLOv11 instance segmentation"""

    def __init__(self, method_name, config, adapter):
        self.method_name = method_name
        self.config = config
        self.adapter = adapter
        self.device = f"cuda:{config.get('gpu', 0)}"

        # YOLOv11 setup
        self.yolo = YOLO('yolo11x-seg.pt')
        self.tracker_config = 'botsort.yaml'
        self.person_only = config.get('person_only', True)
        self.center_mask = config.get('center_mask', True)

        # Camera intrinsics (NuScenes defaults)
        self.fx = config.get('fx', 1266.4)
        self.cx = config.get('cx', 816.3)

        # Load depth model via adapter
        self.adapter.load_model()

    def process_video(self, video_path):
        """Process video: YOLOv11 segmentation + adapter depth estimation"""
        cap = cv2.VideoCapture(str(video_path))
        track_trajectories = defaultdict(list)
        result_frames = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # YOLOv11 segmentation + tracking
            classes = [0] if self.person_only else None
            seg_results = self.yolo.track(frame, persist=True, classes=classes,
                                          tracker=self.tracker_config)

            # Depth estimation via adapter (frame-by-frame)
            frame_tensor = self._preprocess_frame(frame)
            with torch.no_grad():
                depth_map = self.adapter.inference(frame_tensor)
            depth_map = depth_map.squeeze().cpu().numpy()

            # Process instances and update trajectories
            instances_info = self._process_instances(seg_results[0], depth_map, frame.shape)
            for inst in instances_info:
                track_trajectories[inst['track_id']].append({
                    'frame': frame_idx,
                    'depth_m': inst['depth'],
                    'lateral_m': inst['lateral_pos'],
                    'center_x': inst['center_x'],
                    'center_y': inst['center_y']
                })

            # Visualization
            vis_frame = create_frame_visualization(frame, depth_map, instances_info)
            result_frames.append(vis_frame)
            frame_idx += 1

        cap.release()
        return track_trajectories, result_frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True, choices=[
        'metric3d', 'unidepth', 'zoedepth', 'depthpro', 'depthanythingv2', 'cut3r'
    ])
    parser.add_argument('--version', default=None)  # v1 or v2 for metric3d, unidepth
    parser.add_argument('--video-path', required=True)
    parser.add_argument('--results-dir', default='test_results/instance_comparison')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--person-only', action='store_true', default=True)
    parser.add_argument('--center-mask', action='store_true', default=True)
    parser.add_argument('--frame-interval', type=int, default=1)
    args = parser.parse_args()

    # Create adapter based on method
    if args.method == 'metric3d':
        adapter = Metric3DAdapter(version=args.version or 'v2')
    elif args.method == 'unidepth':
        adapter = UniDepthAdapter(version=args.version or 'v2')
    elif args.method == 'zoedepth':
        adapter = ZoeDepthAdapter()
    elif args.method == 'depthpro':
        adapter = DepthProAdapter()
    # ... etc

    config = vars(args)
    tester = InstanceComparisonTester(args.method, config, adapter)
    tester.test()

if __name__ == "__main__":
    main()
```

---

### 3. `test_instance_video_comparison.py` (VIDEO 모델용 - flashdepth_comparison Docker)
**Location**: `/home/cvlab/hsy/flashdepth_advanced/flashdepth_claude/test_instance_video_comparison.py`

**지원 모델**: vda (Video-Depth-Anything), depthcrafter

**핵심 차이점**: VIDEO 모델은 전체 시퀀스를 한번에 처리하므로, depth 추정 후 YOLOv11 트래킹 적용

```python
class InstanceVideoComparisonTester:
    """Video depth model + YOLOv11 instance segmentation"""

    def process_video(self, video_path):
        # 1. 먼저 전체 프레임 로드
        frames = self._load_all_frames(video_path)

        # 2. VIDEO 모델로 전체 시퀀스 depth 추정 (한번에)
        frames_tensor = self._preprocess_sequence(frames)  # [1, T, 3, H, W]
        with torch.no_grad():
            depth_maps = self.adapter.inference(frames_tensor)  # [1, T, H, W]

        # 3. 각 프레임에 YOLOv11 segmentation + tracking 적용
        track_trajectories = defaultdict(list)
        result_frames = []

        for frame_idx, (frame, depth_map) in enumerate(zip(frames, depth_maps)):
            # YOLOv11 tracking
            seg_results = self.yolo.track(frame, persist=True, classes=[0],
                                          tracker='botsort.yaml')

            # Process instances
            instances_info = self._process_instances(seg_results[0], depth_map, frame.shape)
            # ... (동일한 로직)

        return track_trajectories, result_frames
```

---

### 4. `utils/instance_depth_utils.py`
**Location**: `/home/cvlab/hsy/flashdepth_advanced/flashdepth_claude/utils/instance_depth_utils.py`

```python
import cv2
import numpy as np

def get_eroded_mask_and_center(mask, kernel_size=5):
    """
    Erode mask for robust depth extraction and get centroid.

    Args:
        mask: Binary mask (H, W) with 0/1 values
        kernel_size: Erosion kernel size

    Returns:
        eroded_mask: Eroded binary mask
        center_x: X coordinate of centroid
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)

    # If erosion makes mask empty, use original
    if eroded_mask.sum() == 0:
        eroded_mask = mask

    # Calculate centroid
    M = cv2.moments(mask, binaryImage=True)
    if M['m00'] == 0:
        ys, xs = np.where(mask == 1)
        center_x = int(np.mean(xs)) if len(xs) > 0 else 0
    else:
        center_x = int(M['m10'] / M['m00'])

    return eroded_mask, center_x


def calculate_mask_depth(mask, depth_map):
    """
    Calculate mean depth within masked region.

    Args:
        mask: Binary mask (H, W)
        depth_map: Depth map (H, W) in meters

    Returns:
        Mean depth in meters (or 1000.0 if invalid)
    """
    ys, xs = np.where(mask == 1)
    if len(xs) == 0:
        return 1000.0

    valid_depths = depth_map[ys, xs]
    valid_depths = valid_depths[valid_depths > 0]

    if valid_depths.size == 0:
        return 1000.0

    return float(np.mean(valid_depths))


def calculate_lateral_position(depth, center_x, fx, cx):
    """
    Calculate lateral (X) position from depth and pixel coordinates.

    Formula: x_metric = (x_px - cx) * depth / fx

    Args:
        depth: Depth in meters
        center_x: Pixel X coordinate of object center
        fx: Focal length X
        cx: Principal point X

    Returns:
        Lateral position in meters
    """
    return (center_x - cx) * depth / fx


def create_mask_from_polygons(polygons, image_shape):
    """Create binary mask from polygon points."""
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    for polygon in polygons:
        polygon = polygon.astype(np.int32)
        cv2.fillPoly(mask, [polygon], 1)
    return mask
```

---

### 3. `utils/instance_visualization.py`
**Location**: `/home/cvlab/hsy/flashdepth_advanced/flashdepth_claude/utils/instance_visualization.py`

```python
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def colorize_depth(depth, cmap_name='Spectral', apply_mask=None):
    """Colorize depth map using matplotlib colormap."""
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    cmap = matplotlib.colormaps[cmap_name]
    depth_colored = (cmap(depth_normalized)[:, :, :3] * 255).astype(np.uint8)

    if apply_mask is not None:
        depth_colored[apply_mask == 0] = 0

    return depth_colored


def create_frame_visualization(frame, depth_map, instances_info):
    """
    Create visualization with depth overlay and instance info.

    Args:
        frame: Original BGR frame
        depth_map: Depth map (H, W)
        instances_info: List of dicts with track_id, depth, lateral_pos, box, mask

    Returns:
        Visualization frame (BGR)
    """
    vis_frame = frame.copy()

    # Create combined mask
    combined_mask = np.zeros(depth_map.shape, dtype=np.uint8)
    for inst in instances_info:
        combined_mask = np.maximum(combined_mask, inst['mask'])

    # Colorize depth and blend
    if combined_mask.sum() > 0:
        depth_colored = colorize_depth(depth_map, apply_mask=combined_mask)
        alpha = 0.7
        vis_frame = cv2.addWeighted(
            vis_frame, 1.0,
            cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR), alpha,
            0
        )

    # Draw instance info
    for inst in instances_info:
        box = inst['box']
        track_id = inst['track_id']
        depth = inst['depth']
        lat_pos = inst['lateral_pos']

        # Draw bounding box
        cv2.rectangle(
            vis_frame,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            (0, 255, 0), 2
        )

        # Draw ID and depth info
        label = f"ID:{track_id}, Z:{depth:.2f}m, X:{lat_pos:.2f}m"
        cv2.putText(
            vis_frame, label,
            (int(box[0]), int(box[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
        )

    return vis_frame


def create_trajectory_plot(track_trajectories, output_path):
    """
    Create matplotlib plot of instance trajectories (depth vs lateral).

    Args:
        track_trajectories: Dict[track_id -> list of trajectory points]
        output_path: Path to save plot
    """
    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap('tab10')

    for idx, (track_id, traj) in enumerate(sorted(track_trajectories.items())):
        if len(traj) < 2:
            continue

        depths = [p['depth_m'] for p in traj]
        laterals = [p['lateral_m'] for p in traj]

        plt.plot(
            laterals, depths,
            'o-', label=f'Person {track_id}',
            color=cmap(idx % 10),
            markersize=3
        )

    plt.xlabel('Lateral Position (m)')
    plt.ylabel('Depth (m)')
    plt.title('Instance Depth Trajectories')
    plt.legend()
    plt.grid(True)
    plt.gca().invert_yaxis()  # Closer objects at top

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_video_result(frames, output_path, fps, frame_interval=1):
    """
    Save result frames as MP4 video.

    Args:
        frames: List of BGR frames
        output_path: Output video path
        fps: Frames per second
        frame_interval: Save every Nth frame (for visualization)
    """
    if len(frames) == 0:
        return False

    output_path = Path(output_path)

    # Try MP4 first
    try:
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (w, h)
        )

        for i, frame in enumerate(frames):
            if i % frame_interval == 0:
                writer.write(frame)

        writer.release()
        return True

    except Exception as e:
        print(f"MP4 export failed: {e}")

        # Fallback: save as individual frames
        frames_dir = output_path.parent / 'frames'
        frames_dir.mkdir(exist_ok=True)

        for i, frame in enumerate(frames):
            if i % frame_interval == 0:
                cv2.imwrite(str(frames_dir / f'frame_{i:04d}.png'), frame)

        return False
```

---

## Shell Scripts

### 7. `run_instance_test.sh` (Gear5용 - flashdepth Docker)
**Location**: `/home/cvlab/hsy/flashdepth_advanced/flashdepth_claude/run_instance_test.sh`

```bash
#!/bin/bash
# Instance Segmentation + Gear5 Depth Testing
# Usage: ./run_instance_test.sh [options]

VIDEO_PATH="/home/cvlab/hsy/Datasets/videos_mfdepth"
RESULTS_DIR="test_results/instance_depth"
GPU_ID=0
CHECKPOINT=""
FRAME_INTERVAL=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --video-path) VIDEO_PATH="$2"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --gpu) GPU_ID="$2"; shift 2 ;;
        --flashdepth-checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --frame-interval) FRAME_INTERVAL="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: ./run_instance_test.sh [options]"
            echo "  --video-path PATH           Video file/directory"
            echo "  --results-dir PATH          Output directory"
            echo "  --gpu ID                    GPU device"
            echo "  --flashdepth-checkpoint PATH  Gear5 checkpoint"
            echo "  --frame-interval N          Visualization interval"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# flashdepth Docker 사용
CUDA_VISIBLE_DEVICES=$GPU_ID docker compose run --rm flashdepth \
    python test_instance_depth.py \
    --config-path configs/gear5 \
    --config-name config \
    +video_path="$VIDEO_PATH" \
    +results_dir="$RESULTS_DIR" \
    +frame_interval=$FRAME_INTERVAL \
    ${CHECKPOINT:+load="$CHECKPOINT"}
```

---

### 8. `run_instance_comparison.sh` (IMAGE 모델용 - flashdepth_comparison Docker)
**Location**: `/home/cvlab/hsy/flashdepth_advanced/flashdepth_claude/run_instance_comparison.sh`

```bash
#!/bin/bash
# Instance Segmentation + Comparison Depth Models (IMAGE)
# Usage: ./run_instance_comparison.sh <method> [options]

# 지원 모델: metric3d, unidepth, zoedepth, depthpro, depthanythingv2, cut3r
METHOD=$1
shift

VIDEO_PATH="/home/cvlab/hsy/Datasets/videos_mfdepth"
RESULTS_DIR="test_results/instance_comparison"
GPU_ID=0
VERSION=""
FRAME_INTERVAL=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --video-path) VIDEO_PATH="$2"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --gpu) GPU_ID="$2"; shift 2 ;;
        --version) VERSION="$2"; shift 2 ;;
        --frame-interval) FRAME_INTERVAL="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: ./run_instance_comparison.sh <method> [options]"
            echo "Methods: metric3d, unidepth, zoedepth, depthpro, depthanythingv2, cut3r"
            echo "Options:"
            echo "  --video-path PATH     Video file/directory"
            echo "  --results-dir PATH    Output directory"
            echo "  --gpu ID              GPU device"
            echo "  --version v1|v2       Model version (metric3d, unidepth)"
            echo "  --frame-interval N    Visualization interval"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Conda 환경 매핑
case $METHOD in
    metric3d) CONDA_ENV="metric3d" ;;
    unidepth) CONDA_ENV="unidepth" ;;
    zoedepth) CONDA_ENV="zoedepth" ;;
    depthpro) CONDA_ENV="depthpro" ;;
    depthanythingv2) CONDA_ENV="depthanythingv2" ;;
    cut3r) CONDA_ENV="cut3r" ;;
    *) echo "Unknown method: $METHOD"; exit 1 ;;
esac

# flashdepth_comparison Docker 사용
CUDA_VISIBLE_DEVICES=$GPU_ID docker compose run --rm comparison bash -c "
    source /opt/miniforge/etc/profile.d/conda.sh && \
    conda activate $CONDA_ENV && \
    python test_instance_comparison.py \
        --method $METHOD \
        --video-path '$VIDEO_PATH' \
        --results-dir '$RESULTS_DIR' \
        --gpu $GPU_ID \
        --frame-interval $FRAME_INTERVAL \
        ${VERSION:+--version $VERSION}
"
```

---

### 9. `run_instance_video_comparison.sh` (VIDEO 모델용 - flashdepth_comparison Docker)
**Location**: `/home/cvlab/hsy/flashdepth_advanced/flashdepth_claude/run_instance_video_comparison.sh`

```bash
#!/bin/bash
# Instance Segmentation + Video Depth Models
# Usage: ./run_instance_video_comparison.sh <method> [options]

# 지원 모델: vda, depthcrafter
METHOD=$1
shift

VIDEO_PATH="/home/cvlab/hsy/Datasets/videos_mfdepth"
RESULTS_DIR="test_results/instance_video_comparison"
GPU_ID=0
FRAME_INTERVAL=1
METRIC_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --video-path) VIDEO_PATH="$2"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --gpu) GPU_ID="$2"; shift 2 ;;
        --frame-interval) FRAME_INTERVAL="$2"; shift 2 ;;
        --metric) METRIC_FLAG="--metric"; shift ;;  # VDA metric mode
        -h|--help)
            echo "Usage: ./run_instance_video_comparison.sh <method> [options]"
            echo "Methods: vda, depthcrafter"
            echo "Options:"
            echo "  --video-path PATH     Video file/directory"
            echo "  --results-dir PATH    Output directory"
            echo "  --gpu ID              GPU device"
            echo "  --frame-interval N    Visualization interval"
            echo "  --metric              VDA metric depth mode"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Conda 환경 매핑
case $METHOD in
    vda) CONDA_ENV="vda" ;;
    depthcrafter) CONDA_ENV="depthcrafter" ;;
    *) echo "Unknown method: $METHOD"; exit 1 ;;
esac

# flashdepth_comparison Docker 사용
CUDA_VISIBLE_DEVICES=$GPU_ID docker compose run --rm comparison bash -c "
    source /opt/miniforge/etc/profile.d/conda.sh && \
    conda activate $CONDA_ENV && \
    python test_instance_video_comparison.py \
        --method $METHOD \
        --video-path '$VIDEO_PATH' \
        --results-dir '$RESULTS_DIR' \
        --gpu $GPU_ID \
        --frame-interval $FRAME_INTERVAL \
        $METRIC_FLAG
"
```

---

## Docker 수정사항

### 10. `Dockerfile` 수정 (flashdepth 이미지)
```dockerfile
# Gear5용 - ultralytics 추가
RUN pip install ultralytics>=8.0.0
```

### 11. `Dockerfile.comparison` 수정 (flashdepth_comparison 이미지)
**모든 conda 환경에 ultralytics 추가:**
```dockerfile
# 각 환경에 ultralytics 설치 추가
# metric3d 환경
RUN conda run -n metric3d pip install ultralytics>=8.0.0

# unidepth 환경
RUN conda run -n unidepth pip install ultralytics>=8.0.0

# zoedepth 환경
RUN conda run -n zoedepth pip install ultralytics>=8.0.0

# ... (모든 환경에 동일하게 적용)
```

---

## Output Format

### JSON Results (`instance_tracking_results.json`)
```json
{
  "video_name": "nusc_peds6.mp4",
  "total_frames": 196,
  "fps": 9,
  "resolution": [1600, 900],
  "camera_intrinsics": {"fx": 1266.4, "cx": 816.3, "cy": 450.0},
  "segmentation_model": "yolo11x-seg.pt",
  "tracker": "botsort",
  "depth_model": "gear5",
  "instances": {
    "1": {
      "class": "person",
      "first_frame": 0,
      "last_frame": 150,
      "trajectory": [
        {"frame": 0, "depth_m": 8.47, "lateral_m": 0.52, "center_x": 850, "center_y": 450},
        {"frame": 1, "depth_m": 8.60, "lateral_m": -0.24, "center_x": 845, "center_y": 452}
      ],
      "statistics": {
        "min_depth": 8.05,
        "max_depth": 9.29,
        "avg_depth": 8.67,
        "depth_std": 0.35
      }
    }
  },
  "processing_time_sec": 45.2
}
```

### Visualization Outputs
```
{results_dir}/{video_name}/
├── instance_tracking_results.json   # Per-instance trajectories
├── trajectory_plot.png              # Matplotlib depth vs lateral plot
├── result_video.mp4                 # Video with overlays
└── frames/                          # Fallback if MP4 fails
    ├── frame_0000.png
    ├── frame_0010.png
    └── ...
```

---

## 사용 예시

### Gear5 모델 테스트 (flashdepth Docker)
```bash
# 단일 비디오 테스트
./run_instance_test.sh \
    --video-path /home/cvlab/hsy/Datasets/videos_mfdepth/nusc_peds6.mp4 \
    --results-dir test_results/instance_gear5 \
    --gpu 0 \
    --flashdepth-checkpoint train_results/gear5/best.pth

# 전체 비디오 디렉토리 테스트
./run_instance_test.sh \
    --video-path /home/cvlab/hsy/Datasets/videos_mfdepth \
    --results-dir test_results/instance_gear5_all \
    --gpu 0
```

### IMAGE 모델 테스트 (flashdepth_comparison Docker)
```bash
# Metric3D v2
./run_instance_comparison.sh metric3d \
    --video-path /home/cvlab/hsy/Datasets/videos_mfdepth \
    --results-dir test_results/instance_metric3d \
    --gpu 0 \
    --version v2

# UniDepth v2
./run_instance_comparison.sh unidepth \
    --video-path /home/cvlab/hsy/Datasets/videos_mfdepth \
    --results-dir test_results/instance_unidepth \
    --gpu 0 \
    --version v2

# ZoeDepth
./run_instance_comparison.sh zoedepth \
    --video-path /home/cvlab/hsy/Datasets/videos_mfdepth \
    --results-dir test_results/instance_zoedepth \
    --gpu 0

# DepthPro
./run_instance_comparison.sh depthpro \
    --video-path /home/cvlab/hsy/Datasets/videos_mfdepth \
    --results-dir test_results/instance_depthpro \
    --gpu 0
```

### VIDEO 모델 테스트 (flashdepth_comparison Docker)
```bash
# Video-Depth-Anything (relative depth)
./run_instance_video_comparison.sh vda \
    --video-path /home/cvlab/hsy/Datasets/videos_mfdepth \
    --results-dir test_results/instance_vda \
    --gpu 0

# Video-Depth-Anything (metric depth)
./run_instance_video_comparison.sh vda \
    --video-path /home/cvlab/hsy/Datasets/videos_mfdepth \
    --results-dir test_results/instance_vda_metric \
    --gpu 0 \
    --metric

# DepthCrafter
./run_instance_video_comparison.sh depthcrafter \
    --video-path /home/cvlab/hsy/Datasets/videos_mfdepth \
    --results-dir test_results/instance_depthcrafter \
    --gpu 0
```

---

## 구현 노트

1. **EMA 후처리 없음** - Raw depth 값 직접 사용
2. **Center-mask erosion** - 5×5 타원형 커널로 robust depth 추출
3. **BoTSORT tracking** - 프레임간 re-identification에 최적
4. **Video fallback** - MP4 내보내기 실패시 개별 PNG 프레임 저장
5. **메모리 관리** - 비디오간 GPU 캐시 클리어
6. **해상도 처리** - Depth map을 원본 프레임 크기로 upsampling

---

## 파일 요약

### 새로 생성할 파일 (8개)
| 파일 | Docker 환경 | 용도 |
|------|------------|------|
| `test_instance_depth.py` | flashdepth | Gear5 + YOLOv11 |
| `test_instance_comparison.py` | flashdepth_comparison | IMAGE 모델 + YOLOv11 |
| `test_instance_video_comparison.py` | flashdepth_comparison | VIDEO 모델 + YOLOv11 |
| `run_instance_test.sh` | flashdepth | Gear5 runner |
| `run_instance_comparison.sh` | flashdepth_comparison | IMAGE 모델 runner |
| `run_instance_video_comparison.sh` | flashdepth_comparison | VIDEO 모델 runner |
| `utils/instance_depth_utils.py` | 공유 | 마스크/depth 유틸 |
| `utils/instance_visualization.py` | 공유 | 시각화 유틸 |

### 수정할 파일 (2개)
| 파일 | 수정 내용 |
|------|----------|
| `Dockerfile` | `ultralytics>=8.0.0` 추가 |
| `Dockerfile.comparison` | 모든 conda 환경에 `ultralytics` 추가 |
