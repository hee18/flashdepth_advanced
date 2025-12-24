"""
Generate Pedestrian Depth GT from NuScenes Annotations

nuScenes 메타데이터의 3D annotation을 사용하여 pedestrian depth GT를 생성합니다.
YOLO/LiDAR 대신 공식 annotation의 3D bbox를 camera 좌표계로 변환합니다.

Usage:
    python scripts/generate_nuscenes_annotation_gt.py \
        --data-dir /home/cvlab/hsy/Datasets/v1.0-mini \
        --output-dir test_results/crosswalk_gt
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

from utils.lidar_projection_utils import quaternion_to_rotation_matrix
from utils.instance_visualization import (
    create_trajectory_plot,
    create_depth_timeline_plot,
    export_frame_images,
)
from utils.instance_depth_utils import compute_instance_statistics

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_timestamp_from_filename(filename: str) -> int:
    """파일명에서 타임스탬프 추출

    Format: n008-2018-08-28-16-43-51-0400__CAM_FRONT__1535489296012404.jpg
    """
    return int(filename.split('__')[-1].replace('.jpg', '').replace('.pcd.bin', ''))


class NuScenesAnnotationGTGenerator:
    """nuScenes annotation 기반 GT 생성기"""

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        person_only: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.person_only = person_only

        # Paths
        self.samples_cam_dir = self.data_dir / 'crosswalk_samples' / 'CAM_FRONT'
        self.sweeps_cam_dir = self.data_dir / 'crosswalk_sweeps' / 'CAM_FRONT'
        self.metadata_dir = self.data_dir / 'v1.0-mini'

        # Verify paths
        if not self.samples_cam_dir.exists():
            raise FileNotFoundError(f"Samples camera dir not found: {self.samples_cam_dir}")
        if not self.sweeps_cam_dir.exists():
            raise FileNotFoundError(f"Sweeps camera dir not found: {self.sweeps_cam_dir}")
        if not self.metadata_dir.exists():
            raise FileNotFoundError(f"Metadata dir not found: {self.metadata_dir}")

        # Load metadata
        self._load_metadata()

        # Build sample → sweep mapping
        self.sample_to_sweep_idx = self._build_sample_sweep_mapping()

        logger.info(f"Loaded {len(self.sample_files)} samples, {len(self.sweep_files)} sweeps")
        logger.info(f"Sample → Sweep mapping: {len(self.sample_to_sweep_idx)} entries")

    def _load_metadata(self):
        """NuScenes 메타데이터 로드"""
        logger.info("Loading NuScenes metadata...")

        with open(self.metadata_dir / 'sample.json') as f:
            self.samples = json.load(f)
        with open(self.metadata_dir / 'sample_data.json') as f:
            self.sample_data_list = json.load(f)
        with open(self.metadata_dir / 'sample_annotation.json') as f:
            self.annotations = json.load(f)
        with open(self.metadata_dir / 'instance.json') as f:
            self.instances = json.load(f)
        with open(self.metadata_dir / 'category.json') as f:
            self.categories = json.load(f)
        with open(self.metadata_dir / 'ego_pose.json') as f:
            self.ego_poses = json.load(f)
        with open(self.metadata_dir / 'calibrated_sensor.json') as f:
            self.calibrated_sensors = json.load(f)
        with open(self.metadata_dir / 'sensor.json') as f:
            self.sensors = json.load(f)

        # Build mappings
        self.category_map = {c['token']: c['name'] for c in self.categories}
        self.instance_to_category = {
            i['token']: self.category_map.get(i['category_token'], 'unknown')
            for i in self.instances
        }
        self.ego_pose_map = {ep['token']: ep for ep in self.ego_poses}
        self.sample_data_map = {sd['token']: sd for sd in self.sample_data_list}
        self.sensor_map = {s['token']: s['channel'] for s in self.sensors}

        # Find CAM_FRONT calibration
        self.cam_calib = None
        for cs in self.calibrated_sensors:
            channel = self.sensor_map.get(cs['sensor_token'], '')
            if channel == 'CAM_FRONT':
                self.cam_calib = cs
                break

        if self.cam_calib is None:
            raise ValueError("CAM_FRONT calibration not found")

        self.cam_intrinsics = np.array(self.cam_calib['camera_intrinsic'])
        self.cam_translation = np.array(self.cam_calib['translation'])
        self.cam_rotation = np.array(self.cam_calib['rotation'])

        logger.info(f"Camera intrinsics:\n{self.cam_intrinsics}")

        # Build sample_data by filename
        self.sample_data_by_filename = {}
        for sd in self.sample_data_list:
            if 'CAM_FRONT' in sd['filename'] and sd['is_key_frame']:
                fname = os.path.basename(sd['filename'])
                self.sample_data_by_filename[fname] = sd

        # Build annotations by sample_token
        self.annotations_by_sample = defaultdict(list)
        for ann in self.annotations:
            self.annotations_by_sample[ann['sample_token']].append(ann)

        # Get sample/sweep file lists
        self.sample_files = sorted(os.listdir(self.samples_cam_dir))
        self.sweep_files = sorted(os.listdir(self.sweeps_cam_dir))

    def _build_sample_sweep_mapping(self) -> Dict[int, int]:
        """Sample 인덱스 → 가장 가까운 Sweep 인덱스 매핑"""
        sample_ts = [extract_timestamp_from_filename(f) for f in self.sample_files]
        sweep_ts = [extract_timestamp_from_filename(f) for f in self.sweep_files]

        mapping = {}
        for i, s_ts in enumerate(sample_ts):
            distances = [abs(sw_ts - s_ts) for sw_ts in sweep_ts]
            nearest_idx = np.argmin(distances)
            mapping[i] = int(nearest_idx)

            # Log first few mappings
            if i < 5:
                logger.info(f"Sample {i} (ts={s_ts}) → Sweep {nearest_idx} (ts={sweep_ts[nearest_idx]}, diff={distances[nearest_idx]})")

        return mapping

    def transform_annotation_to_camera(
        self,
        ann: Dict[str, Any],
        ego_pose: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Annotation의 global coord를 camera coord로 변환

        Pipeline:
        1. Global → Ego (ego_pose inverse)
        2. Ego → Camera (calibrated_sensor inverse)

        Returns:
            dict with 'depth', 'pos_cam', 'pos_2d' (pixel coords)
        """
        translation = np.array(ann['translation'])
        size = np.array(ann['size'])  # [width, length, height]

        # 1. Global → Ego
        R_ego = quaternion_to_rotation_matrix(np.array(ego_pose['rotation']))
        t_ego = np.array(ego_pose['translation'])
        pos_ego = R_ego.T @ (translation - t_ego)

        # 2. Ego → Camera
        R_cam = quaternion_to_rotation_matrix(self.cam_rotation)
        pos_cam = R_cam.T @ (pos_ego - self.cam_translation)

        # Check if in front of camera
        if pos_cam[2] <= 0:
            return None

        # 3. Depth calculation (nearest face)
        # size[1] = length (depth direction in vehicle frame)
        center_depth = pos_cam[2]
        nearest_face_depth = center_depth - size[1] / 2

        # 4. Project to 2D
        fx = self.cam_intrinsics[0, 0]
        fy = self.cam_intrinsics[1, 1]
        cx = self.cam_intrinsics[0, 2]
        cy = self.cam_intrinsics[1, 2]

        u = fx * pos_cam[0] / pos_cam[2] + cx
        v = fy * pos_cam[1] / pos_cam[2] + cy

        # Calculate approximate 2D bbox size
        # Using height for vertical, width for horizontal
        bbox_height_px = fy * size[2] / pos_cam[2]
        bbox_width_px = fx * size[0] / pos_cam[2]

        return {
            'depth_center': float(center_depth),
            'depth': float(max(nearest_face_depth, 0.1)),  # nearest face depth
            'pos_cam': pos_cam.tolist(),
            'pos_2d': [float(u), float(v)],
            'bbox_2d': [
                float(u - bbox_width_px / 2),
                float(v - bbox_height_px / 2),
                float(u + bbox_width_px / 2),
                float(v + bbox_height_px / 2),
            ],
            'lateral_m': float(pos_cam[0]),  # X in camera coord = lateral
        }

    def generate(self) -> Tuple[Dict[str, Any], List[np.ndarray]]:
        """GT 생성

        Returns:
            results: GT 결과 딕셔너리
            result_frames: 시각화된 프레임 리스트
        """
        logger.info("Starting annotation-based GT generation...")

        # Get first frame size
        first_frame = cv2.imread(str(self.samples_cam_dir / self.sample_files[0]))
        height, width = first_frame.shape[:2]
        logger.info(f"Frame size: {width}x{height}")

        track_trajectories = defaultdict(list)
        track_classes = {}
        result_frames = []
        frame_mapping = {}

        for sample_idx, sample_file in enumerate(tqdm(self.sample_files, desc="Processing samples")):
            # Get sweep index for this sample
            sweep_idx = self.sample_to_sweep_idx[sample_idx]
            frame_mapping[str(sample_idx)] = sweep_idx

            # Load frame
            frame = cv2.imread(str(self.samples_cam_dir / sample_file))
            if frame is None:
                logger.warning(f"Failed to load frame: {sample_file}")
                result_frames.append(np.zeros((height, width, 3), dtype=np.uint8))
                continue

            # Get sample_data for this file
            sample_data = self.sample_data_by_filename.get(sample_file)
            if sample_data is None:
                logger.warning(f"No sample_data found for: {sample_file}")
                result_frames.append(frame.copy())
                continue

            sample_token = sample_data['sample_token']
            ego_pose_token = sample_data['ego_pose_token']
            ego_pose = self.ego_pose_map.get(ego_pose_token)

            if ego_pose is None:
                logger.warning(f"No ego_pose found for token: {ego_pose_token}")
                result_frames.append(frame.copy())
                continue

            # Get annotations for this sample
            annotations = self.annotations_by_sample.get(sample_token, [])

            # Process pedestrian annotations
            frame_instances = []
            for ann in annotations:
                # Check category
                category = self.instance_to_category.get(ann['instance_token'], '')
                if self.person_only and 'human.pedestrian' not in category:
                    continue

                # Transform to camera coord
                result = self.transform_annotation_to_camera(ann, ego_pose)
                if result is None:
                    continue

                # Check if within image bounds
                u, v = result['pos_2d']
                if not (0 <= u < width and 0 <= v < height):
                    continue

                # Use instance_token as track ID (consistent across frames)
                track_id = ann['instance_token'][:8]  # Shortened for display

                frame_instances.append({
                    'track_id': track_id,
                    'category': category,
                    'depth': result['depth'],
                    'lateral_m': result['lateral_m'],
                    'pos_2d': result['pos_2d'],
                    'bbox_2d': result['bbox_2d'],
                    'num_lidar_pts': ann.get('num_lidar_pts', 0),
                })

                # Update trajectory (using sweep_idx as frame number!)
                track_trajectories[track_id].append({
                    'frame': sweep_idx,  # Sweep 기준 프레임 번호
                    'sample_idx': sample_idx,  # Original sample index
                    'depth_m': result['depth'],
                    'lateral_m': result['lateral_m'],
                    'center_x': int(u),
                    'center_y': int(v),
                    'num_lidar_pts': ann.get('num_lidar_pts', 0),
                })
                track_classes[track_id] = category

            # Create visualization frame
            vis_frame = self._create_visualization(frame, frame_instances, sweep_idx)
            result_frames.append(vis_frame)

        # Build results
        results = {
            'video_name': 'crosswalk_sample',
            'total_samples': len(self.sample_files),
            'total_sweeps': len(self.sweep_files),
            'source': 'nuscenes_annotation',
            'frame_index_type': 'sweep',  # 프레임 인덱스가 sweep 기준임을 표시
            'depth_type': 'nearest_face',  # depth 계산 방식
            'fps': 12,  # Sweep rate (approximate)
            'resolution': [width, height],
            'camera_intrinsics': {
                'fx': self.cam_intrinsics[0, 0],
                'fy': self.cam_intrinsics[1, 1],
                'cx': self.cam_intrinsics[0, 2],
                'cy': self.cam_intrinsics[1, 2],
            },
            'frame_mapping': frame_mapping,  # sample_idx → sweep_idx
            'instances': {},
            'num_instances': len(track_trajectories),
        }

        for track_id, trajectory in track_trajectories.items():
            # Sort by frame number
            trajectory = sorted(trajectory, key=lambda x: x['frame'])
            results['instances'][track_id] = {
                'class': track_classes.get(track_id, 'unknown'),
                'first_frame': trajectory[0]['frame'] if trajectory else 0,
                'last_frame': trajectory[-1]['frame'] if trajectory else 0,
                'trajectory': trajectory,
                'statistics': compute_instance_statistics(trajectory),
            }

        return results, result_frames

    def _create_visualization(
        self,
        frame: np.ndarray,
        instances: List[Dict[str, Any]],
        sweep_idx: int,
    ) -> np.ndarray:
        """프레임 시각화 생성"""
        vis = frame.copy()

        # Draw sweep index
        cv2.putText(
            vis,
            f"Sweep idx: {sweep_idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        # Color palette for instances
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 255, 0), (255, 128, 0), (128, 0, 255),
        ]

        for i, inst in enumerate(instances):
            color = colors[i % len(colors)]

            # Draw bbox
            bbox = inst['bbox_2d']
            x1, y1, x2, y2 = [int(c) for c in bbox]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Draw center point
            cx, cy = int(inst['pos_2d'][0]), int(inst['pos_2d'][1])
            cv2.circle(vis, (cx, cy), 5, color, -1)

            # Draw label
            label = f"ID:{inst['track_id']} D:{inst['depth']:.1f}m"
            cv2.putText(
                vis,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        return vis

    def save_results(
        self,
        results: Dict[str, Any],
        result_frames: List[np.ndarray],
        frame_interval: int = 1,
    ):
        """결과 저장"""
        output_subdir = self.output_dir / 'crosswalk_sample'
        output_subdir.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = output_subdir / 'instance_tracking_results.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to: {json_path}")

        # Create trajectory plots
        trajectories = {
            k: v['trajectory']
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

        # Save frame images with sweep index naming
        if len(result_frames) > 0:
            frames_dir = output_subdir / 'frames'
            frames_dir.mkdir(parents=True, exist_ok=True)

            for sample_idx, frame in enumerate(result_frames):
                if sample_idx % frame_interval != 0:
                    continue

                # Use sweep index for filename
                sweep_idx = self.sample_to_sweep_idx[sample_idx]
                frame_path = frames_dir / f'frame_{sweep_idx:04d}.png'
                cv2.imwrite(str(frame_path), frame)

            logger.info(f"Saved frame images to: {frames_dir}")

        logger.info(f"GT generation complete! Results saved to: {output_subdir}")


def main():
    parser = argparse.ArgumentParser(description='Generate Pedestrian Depth GT from NuScenes Annotations')
    parser.add_argument('--data-dir', type=str,
                        default='/home/cvlab/hsy/Datasets/v1.0-mini',
                        help='NuScenes data directory')
    parser.add_argument('--output-dir', type=str,
                        default='test_results/crosswalk_gt',
                        help='Output directory')
    parser.add_argument('--no-person-only', action='store_true',
                        help='Include all annotated objects, not just pedestrians')
    parser.add_argument('--frame-interval', type=int, default=1,
                        help='Save every Nth frame')

    args = parser.parse_args()

    generator = NuScenesAnnotationGTGenerator(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        person_only=not args.no_person_only,
    )

    results, result_frames = generator.generate()
    generator.save_results(results, result_frames, frame_interval=args.frame_interval)


if __name__ == '__main__':
    main()
