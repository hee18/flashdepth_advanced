"""
Generate Instance LiDAR Ground Truth

NuScenes crosswalk_samples의 카메라 프레임과 LiDAR 스캔을 사용하여
인스턴스별 depth trajectory GT를 생성합니다.

YOLOv11 세그멘테이션 + BoTSORT 트래킹과 동일한 방식으로 인스턴스를 검출하고,
해당 마스크 영역의 LiDAR depth를 추출합니다.

Usage:
    python scripts/generate_instance_lidar_gt.py \
        --data-dir /home/cvlab/hsy/Datasets/v1.0-mini \
        --output-dir test_results/crosswalk_gt \
        --gpu 0
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.instance_depth_utils import (
    get_center_mask,
    get_mask_center,
    calculate_lateral_position,
    create_mask_from_yolo_result,
    compute_instance_statistics,
    resize_depth_to_frame,
)
from utils.lidar_projection_utils import (
    load_lidar_pcd_bin,
    project_lidar_to_camera,
    extract_depth_from_lidar_with_knearest,
    calculate_lateral_from_lidar,
    quaternion_to_rotation_matrix,
)
from utils.instance_visualization import (
    create_trajectory_plot,
    create_depth_timeline_plot,
    create_frame_visualization,
    save_video_result,
    export_frame_images,
)
from utils.lidar_projection_utils import create_sparse_depth_map

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NuScenesCalibration:
    """NuScenes 캘리브레이션 데이터 로더"""

    def __init__(self, metadata_dir: Path):
        self.metadata_dir = metadata_dir

        # Load metadata
        with open(metadata_dir / 'calibrated_sensor.json') as f:
            self.calibrated_sensors = json.load(f)
        with open(metadata_dir / 'sensor.json') as f:
            self.sensors = json.load(f)
        with open(metadata_dir / 'ego_pose.json') as f:
            self.ego_poses = json.load(f)
        with open(metadata_dir / 'sample_data.json') as f:
            self.sample_data = json.load(f)

        # Build mappings
        self.sensor_token_to_channel = {s['token']: s['channel'] for s in self.sensors}
        self.calibrated_sensor_map = {cs['token']: cs for cs in self.calibrated_sensors}
        self.ego_pose_map = {ep['token']: ep for ep in self.ego_poses}
        self.sample_data_map = {sd['token']: sd for sd in self.sample_data}

        # Find CAM_FRONT and LIDAR_TOP calibration (first matching)
        self.cam_calib = None
        self.lidar_calib = None
        for cs in self.calibrated_sensors:
            channel = self.sensor_token_to_channel.get(cs['sensor_token'], '')
            if channel == 'CAM_FRONT' and self.cam_calib is None:
                self.cam_calib = cs
            elif channel == 'LIDAR_TOP' and self.lidar_calib is None:
                self.lidar_calib = cs

        if self.cam_calib is None:
            raise ValueError("CAM_FRONT calibration not found")
        if self.lidar_calib is None:
            raise ValueError("LIDAR_TOP calibration not found")

        logger.info(f"CAM_FRONT calibration token: {self.cam_calib['token']}")
        logger.info(f"LIDAR_TOP calibration token: {self.lidar_calib['token']}")

    def get_camera_intrinsics(self) -> np.ndarray:
        """3x3 camera intrinsic matrix"""
        return np.array(self.cam_calib['camera_intrinsic'])

    def get_camera_params(self) -> Dict[str, float]:
        """fx, fy, cx, cy"""
        K = self.get_camera_intrinsics()
        return {
            'fx': K[0, 0],
            'fy': K[1, 1],
            'cx': K[0, 2],
            'cy': K[1, 2]
        }

    def get_lidar_calibration(self) -> Tuple[np.ndarray, np.ndarray]:
        """LiDAR translation and rotation (quaternion)"""
        return (
            np.array(self.lidar_calib['translation']),
            np.array(self.lidar_calib['rotation'])
        )

    def get_camera_calibration(self) -> Tuple[np.ndarray, np.ndarray]:
        """Camera translation and rotation (quaternion)"""
        return (
            np.array(self.cam_calib['translation']),
            np.array(self.cam_calib['rotation'])
        )


class InstanceLidarGTGenerator:
    """인스턴스 LiDAR GT 생성기"""

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        seg_model: str = 'yolo11x-seg.pt',
        tracker: str = 'custom_botsort.yaml',
        person_only: bool = True,
        center_mask: bool = True,
        k_nearest: int = 5,
        gpu: int = 0
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.person_only = person_only
        self.center_mask = center_mask
        self.k_nearest = k_nearest
        self.device = f'cuda:{gpu}' if gpu >= 0 else 'cpu'

        # Paths
        self.cam_dir = self.data_dir / 'crosswalk_samples' / 'CAM_FRONT'
        self.lidar_dir = self.data_dir / 'crosswalk_samples' / 'LIDAR_TOP'
        self.metadata_dir = self.data_dir / 'v1.0-mini'

        # Verify paths exist
        if not self.cam_dir.exists():
            raise FileNotFoundError(f"Camera directory not found: {self.cam_dir}")
        if not self.lidar_dir.exists():
            raise FileNotFoundError(f"LiDAR directory not found: {self.lidar_dir}")
        if not self.metadata_dir.exists():
            raise FileNotFoundError(f"Metadata directory not found: {self.metadata_dir}")

        # Load calibration
        self.calibration = NuScenesCalibration(self.metadata_dir)
        self.cam_params = self.calibration.get_camera_params()
        self.cam_intrinsics = self.calibration.get_camera_intrinsics()
        self.lidar_translation, self.lidar_rotation = self.calibration.get_lidar_calibration()
        self.cam_translation, self.cam_rotation = self.calibration.get_camera_calibration()

        logger.info(f"Camera params: fx={self.cam_params['fx']:.2f}, cx={self.cam_params['cx']:.2f}")

        # Initialize YOLOv11
        self._setup_yolo(seg_model, tracker)

    def _setup_yolo(self, seg_model: str, tracker: str):
        """YOLOv11 초기화"""
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLOv11: {seg_model}")
            self.yolo = YOLO(seg_model)
            self.tracker_config = tracker
            logger.info("YOLOv11 loaded successfully")
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            raise

    def get_frame_pairs(self) -> List[Tuple[Path, Path]]:
        """
        카메라 프레임과 LiDAR 스캔 쌍을 반환 (정렬된 순서로 1:1 매칭)
        """
        cam_files = sorted(self.cam_dir.glob('*.jpg'))
        lidar_files = sorted(self.lidar_dir.glob('*.pcd.bin'))

        if len(cam_files) != len(lidar_files):
            logger.warning(f"Camera ({len(cam_files)}) and LiDAR ({len(lidar_files)}) file count mismatch!")
            min_count = min(len(cam_files), len(lidar_files))
            cam_files = cam_files[:min_count]
            lidar_files = lidar_files[:min_count]

        return list(zip(cam_files, lidar_files))

    def project_lidar_to_image(
        self,
        lidar_path: Path,
        img_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        LiDAR 포인트를 이미지 좌표로 투영

        Args:
            lidar_path: .pcd.bin 파일 경로
            img_size: (width, height)

        Returns:
            uv_coords: (N, 2) pixel coordinates
            depths: (N,) depths in meters
        """
        # Load LiDAR points
        lidar_points = load_lidar_pcd_bin(str(lidar_path))

        # Project to camera
        uv_coords, depths, _ = project_lidar_to_camera(
            lidar_points=lidar_points,
            lidar_translation=self.lidar_translation,
            lidar_rotation=self.lidar_rotation,
            cam_translation=self.cam_translation,
            cam_rotation=self.cam_rotation,
            cam_intrinsics=self.cam_intrinsics,
            img_size=img_size
        )

        return uv_coords, depths

    def process_instances(
        self,
        seg_result,
        uv_coords: np.ndarray,
        depths: np.ndarray,
        frame_shape: Tuple[int, int, int]
    ) -> List[Dict[str, Any]]:
        """
        세그멘테이션 결과에서 인스턴스별 LiDAR depth 추출

        Args:
            seg_result: YOLO segmentation result
            uv_coords: (N, 2) projected LiDAR coordinates
            depths: (N,) LiDAR depths
            frame_shape: (H, W, C)

        Returns:
            List of instance info dicts
        """
        instances = []

        if seg_result.masks is None or seg_result.boxes is None:
            return instances

        boxes = seg_result.boxes
        masks = seg_result.masks

        for i in range(len(boxes)):
            # Track ID (from BoTSORT)
            if boxes.id is None:
                continue
            track_id = int(boxes.id[i].item())

            # Class
            cls_id = int(boxes.cls[i].item())
            cls_name = seg_result.names[cls_id]

            # Create mask from polygon
            mask_xy = masks.xy[i]
            mask = create_mask_from_yolo_result(mask_xy, frame_shape)

            if mask.sum() == 0:
                continue

            # Get center mask or full mask
            if self.center_mask:
                center_mask_arr, center_x = get_center_mask(mask)
                center_x_full, center_y = get_mask_center(mask)
            else:
                center_mask_arr = mask
                center_x, center_y = get_mask_center(mask)
                center_x_full = center_x

            # Extract LiDAR depth from mask region
            depth_m, mean_u, num_points = extract_depth_from_lidar_with_knearest(
                center_mask=center_mask_arr,
                full_mask=mask,
                uv_coords=uv_coords,
                depths=depths,
                center_x=center_x_full,
                center_y=center_y,
                k_nearest=self.k_nearest
            )

            if depth_m <= 0:
                # No valid LiDAR points found
                continue

            # Calculate lateral position
            lateral_m = calculate_lateral_from_lidar(
                depth_m=depth_m,
                mean_u=mean_u,
                fx=self.cam_params['fx'],
                cx=self.cam_params['cx']
            )

            instances.append({
                'track_id': track_id,
                'class': cls_name,
                'depth': depth_m,
                'lateral_pos': lateral_m,
                'center_x': center_x_full,
                'center_y': center_y,
                'num_lidar_points': num_points
            })

        return instances

    def generate(self) -> Tuple[Dict[str, Any], List[np.ndarray], List[np.ndarray]]:
        """GT 생성 메인 함수

        Returns:
            results: GT 결과 딕셔너리
            result_frames: 시각화된 프레임 리스트
            original_frames: 원본 프레임 리스트
        """
        logger.info("Starting LiDAR GT generation...")

        frame_pairs = self.get_frame_pairs()
        total_frames = len(frame_pairs)
        logger.info(f"Found {total_frames} frame pairs")

        if total_frames == 0:
            raise ValueError("No frame pairs found!")

        # Get image size from first frame
        first_frame = cv2.imread(str(frame_pairs[0][0]))
        height, width = first_frame.shape[:2]
        img_size = (width, height)
        logger.info(f"Image size: {width}x{height}")

        track_trajectories = defaultdict(list)
        track_classes = {}
        result_frames = []
        original_frames = []

        # Reset YOLO tracker
        if hasattr(self.yolo, 'predictor') and self.yolo.predictor is not None:
            if hasattr(self.yolo.predictor, 'trackers'):
                for tracker in self.yolo.predictor.trackers:
                    tracker.reset()

        # Process frames
        for frame_idx, (cam_path, lidar_path) in enumerate(tqdm(frame_pairs, desc="Processing frames")):
            # Load camera frame
            frame = cv2.imread(str(cam_path))
            if frame is None:
                logger.warning(f"Failed to load frame: {cam_path}")
                continue

            original_frames.append(frame.copy())

            # Project LiDAR to image
            uv_coords, depths = self.project_lidar_to_image(lidar_path, img_size)

            # Create sparse depth map for visualization
            sparse_depth_map = create_sparse_depth_map(uv_coords, depths, img_size)

            if len(uv_coords) == 0:
                logger.warning(f"No valid LiDAR points for frame {frame_idx}")
                result_frames.append(frame.copy())
                continue

            # YOLOv11 segmentation + tracking
            classes = [0] if self.person_only else None  # 0 = person
            seg_results = self.yolo.track(
                frame,
                persist=True,
                classes=classes,
                tracker=self.tracker_config,
                verbose=False
            )

            # Process instances and collect visualization info
            instances = self.process_instances(
                seg_results[0], uv_coords, depths, frame.shape
            )

            # Build instances_info for visualization (with mask)
            instances_info_for_vis = []
            seg_result = seg_results[0]
            if seg_result.masks is not None and seg_result.boxes is not None:
                boxes = seg_result.boxes
                masks = seg_result.masks
                for i in range(len(boxes)):
                    if boxes.id is None:
                        continue
                    track_id = int(boxes.id[i].item())

                    # Find corresponding instance info
                    inst_info = next((inst for inst in instances if inst['track_id'] == track_id), None)
                    if inst_info is None:
                        continue

                    # Create mask
                    mask_xy = masks.xy[i]
                    mask = create_mask_from_yolo_result(mask_xy, frame.shape)

                    # Bounding box
                    box = boxes.xyxy[i].cpu().numpy().astype(int)

                    instances_info_for_vis.append({
                        'track_id': track_id,
                        'class': inst_info['class'],
                        'depth': inst_info['depth'],
                        'lateral_pos': inst_info['lateral_pos'],
                        'box': box,
                        'mask': mask,
                        'center_x': inst_info['center_x'],
                        'center_y': inst_info['center_y']
                    })

            # Create visualization frame
            vis_frame = create_frame_visualization(frame, sparse_depth_map, instances_info_for_vis)
            result_frames.append(vis_frame)

            # Update trajectories
            for inst in instances:
                track_id = inst['track_id']
                track_trajectories[track_id].append({
                    'frame': frame_idx,
                    'depth_m': inst['depth'],
                    'lateral_m': inst['lateral_pos'],
                    'center_x': inst['center_x'],
                    'center_y': inst['center_y'],
                    'num_lidar_points': inst['num_lidar_points']
                })
                track_classes[track_id] = inst['class']

        # Build results
        results = {
            'video_name': 'crosswalk_sample',
            'total_frames': total_frames,
            'source': 'lidar_gt',
            'fps': 2,  # NuScenes keyframe rate
            'resolution': [width, height],
            'camera_intrinsics': self.cam_params,
            'segmentation_model': 'yolo11x-seg.pt',
            'tracker': self.tracker_config,
            'k_nearest': self.k_nearest,
            'instances': {},
            'num_instances': len(track_trajectories)
        }

        for track_id, trajectory in track_trajectories.items():
            results['instances'][str(track_id)] = {
                'class': track_classes.get(track_id, 'unknown'),
                'first_frame': trajectory[0]['frame'] if trajectory else 0,
                'last_frame': trajectory[-1]['frame'] if trajectory else 0,
                'trajectory': trajectory,
                'statistics': compute_instance_statistics(trajectory)
            }

        return results, result_frames, original_frames

    def save_results(
        self,
        results: Dict[str, Any],
        result_frames: List[np.ndarray],
        original_frames: List[np.ndarray],
        frame_interval: int = 1
    ):
        """결과 저장

        Args:
            results: GT 결과 딕셔너리
            result_frames: 시각화된 프레임 리스트
            original_frames: 원본 프레임 리스트
            frame_interval: 비디오 저장 시 프레임 간격
        """
        output_subdir = self.output_dir / 'crosswalk_sample'
        output_subdir.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = output_subdir / 'instance_tracking_results.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to: {json_path}")

        # Create trajectory plots
        trajectories = {
            int(k): v['trajectory']
            for k, v in results['instances'].items()
        }

        if trajectories:
            # Trajectory plot
            try:
                traj_plot_path = output_subdir / 'trajectory_plot.png'
                create_trajectory_plot(trajectories, traj_plot_path, is_metric=True)
                logger.info(f"Saved trajectory plot: {traj_plot_path}")
            except Exception as e:
                logger.warning(f"Failed to create trajectory plot: {e}")

            # Depth timeline plot
            try:
                timeline_path = output_subdir / 'depth_timeline.png'
                create_depth_timeline_plot(trajectories, timeline_path, is_metric=True)
                logger.info(f"Saved depth timeline: {timeline_path}")
            except Exception as e:
                logger.warning(f"Failed to create timeline plot: {e}")

        # Save frame images to frames folder
        if len(result_frames) > 0:
            frames_dir = output_subdir / 'frames'
            export_frame_images(result_frames, frames_dir, frame_interval=frame_interval, prefix='frame')
            logger.info(f"Saved frame images to: {frames_dir}")

        logger.info(f"GT generation complete! Results saved to: {output_subdir}")


def main():
    parser = argparse.ArgumentParser(description='Generate Instance LiDAR GT')
    parser.add_argument('--data-dir', type=str,
                        default='/home/cvlab/hsy/Datasets/v1.0-mini',
                        help='NuScenes v1.0-mini directory')
    parser.add_argument('--output-dir', type=str,
                        default='test_results/crosswalk_gt',
                        help='Output directory')
    parser.add_argument('--seg-model', type=str, default='yolo11x-seg.pt',
                        help='YOLOv11 segmentation model')
    parser.add_argument('--tracker', type=str, default='botsort.yaml',
                        help='Tracker config')
    parser.add_argument('--no-person-only', action='store_true',
                        help='Track all classes, not just person')
    parser.add_argument('--no-center-mask', action='store_true',
                        help='Use full mask instead of center mask')
    parser.add_argument('--k-nearest', type=int, default=5,
                        help='K for K-nearest fallback')
    parser.add_argument('--frame-interval', type=int, default=1,
                        help='Save every Nth frame to video')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (-1 for CPU)')

    args = parser.parse_args()

    generator = InstanceLidarGTGenerator(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        seg_model=args.seg_model,
        tracker=args.tracker,
        person_only=not args.no_person_only,
        center_mask=not args.no_center_mask,
        k_nearest=args.k_nearest,
        gpu=args.gpu
    )

    results, result_frames, original_frames = generator.generate()
    generator.save_results(results, result_frames, original_frames, frame_interval=args.frame_interval)


if __name__ == '__main__':
    main()
