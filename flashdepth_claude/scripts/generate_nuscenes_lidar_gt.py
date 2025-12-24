"""
Generate Sparse Depth GT from NuScenes LiDAR

nuScenes LiDAR point cloud를 camera 좌표계로 projection하여 sparse depth GT를 생성합니다.
samples (keyframes)에서만 GT가 생성되고, sweeps 인덱스로 매핑됩니다.

Usage:
    python scripts/generate_nuscenes_lidar_gt.py \
        --data-dir /home/cvlab/hsy/Datasets/v1.0-mini \
        --output-dir test_results/crosswalk_lidar_gt

Output:
    - sparse_depth_gt.json: Sample → Sweep 매핑 + sparse depth 정보
    - depth_maps/: 각 sample의 sparse depth map (.npy)
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.lidar_projection_utils import quaternion_to_rotation_matrix

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_timestamp_from_filename(filename: str) -> int:
    """파일명에서 타임스탬프 추출"""
    return int(filename.split('__')[-1].replace('.jpg', '').replace('.pcd.bin', ''))


def load_lidar_points(lidar_path: Path) -> np.ndarray:
    """Load LiDAR points from .pcd.bin file.

    nuScenes format: float32, 5 elements per point (x, y, z, intensity, ring_index)

    Returns:
        points: (N, 3) array of XYZ coordinates in LiDAR frame
    """
    points = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 5)
    return points[:, :3]  # Return only XYZ


class NuScenesLiDARGTGenerator:
    """nuScenes LiDAR 기반 sparse depth GT 생성기"""

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        max_depth: float = 80.0,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_depth = max_depth

        # Paths
        self.samples_cam_dir = self.data_dir / 'crosswalk_samples' / 'CAM_FRONT'
        self.samples_lidar_dir = self.data_dir / 'crosswalk_samples' / 'LIDAR_TOP'
        self.sweeps_cam_dir = self.data_dir / 'crosswalk_sweeps' / 'CAM_FRONT'
        self.metadata_dir = self.data_dir / 'v1.0-mini'

        # Verify paths
        if not self.samples_cam_dir.exists():
            raise FileNotFoundError(f"Samples camera dir not found: {self.samples_cam_dir}")
        if not self.samples_lidar_dir.exists():
            raise FileNotFoundError(f"Samples LiDAR dir not found: {self.samples_lidar_dir}")
        if not self.metadata_dir.exists():
            raise FileNotFoundError(f"Metadata dir not found: {self.metadata_dir}")

        # Load metadata
        self._load_metadata()

        # Build sample → sweep mapping
        self.sample_to_sweep_idx = self._build_sample_sweep_mapping()

        # Get image size from first sample
        first_img = cv2.imread(str(self.samples_cam_dir / self.sample_files[0]))
        self.img_height, self.img_width = first_img.shape[:2]
        logger.info(f"Image size: {self.img_width}x{self.img_height}")

        logger.info(f"Loaded {len(self.sample_files)} samples, {len(self.sweep_files)} sweeps")

    def _load_metadata(self):
        """NuScenes 메타데이터 로드"""
        logger.info("Loading NuScenes metadata...")

        with open(self.metadata_dir / 'sample_data.json') as f:
            self.sample_data_list = json.load(f)
        with open(self.metadata_dir / 'ego_pose.json') as f:
            self.ego_poses = json.load(f)
        with open(self.metadata_dir / 'calibrated_sensor.json') as f:
            self.calibrated_sensors = json.load(f)
        with open(self.metadata_dir / 'sensor.json') as f:
            self.sensors = json.load(f)

        # Build mappings
        self.ego_pose_map = {ep['token']: ep for ep in self.ego_poses}
        self.sample_data_map = {sd['token']: sd for sd in self.sample_data_list}
        self.sensor_map = {s['token']: s['channel'] for s in self.sensors}

        # Build calibrated_sensor by channel
        self.calib_by_channel = {}
        for cs in self.calibrated_sensors:
            channel = self.sensor_map.get(cs['sensor_token'], '')
            if channel:
                self.calib_by_channel[channel] = cs

        # Camera calibration
        self.cam_calib = self.calib_by_channel.get('CAM_FRONT')
        if self.cam_calib is None:
            raise ValueError("CAM_FRONT calibration not found")

        self.cam_intrinsics = np.array(self.cam_calib['camera_intrinsic'])
        self.cam_translation = np.array(self.cam_calib['translation'])
        self.cam_rotation = np.array(self.cam_calib['rotation'])

        # LiDAR calibration
        self.lidar_calib = self.calib_by_channel.get('LIDAR_TOP')
        if self.lidar_calib is None:
            raise ValueError("LIDAR_TOP calibration not found")

        self.lidar_translation = np.array(self.lidar_calib['translation'])
        self.lidar_rotation = np.array(self.lidar_calib['rotation'])

        logger.info(f"Camera intrinsics: fx={self.cam_intrinsics[0,0]:.1f}, fy={self.cam_intrinsics[1,1]:.1f}")
        logger.info(f"Camera translation: {self.cam_translation}")
        logger.info(f"LiDAR translation: {self.lidar_translation}")

        # Build sample_data by filename (for finding ego_pose)
        self.cam_sd_by_filename = {}
        self.lidar_sd_by_filename = {}
        for sd in self.sample_data_list:
            fname = os.path.basename(sd['filename'])
            if 'CAM_FRONT' in sd['filename']:
                self.cam_sd_by_filename[fname] = sd
            elif 'LIDAR_TOP' in sd['filename']:
                self.lidar_sd_by_filename[fname] = sd

        # Get sample/sweep file lists
        self.sample_files = sorted(os.listdir(self.samples_cam_dir))
        self.lidar_files = sorted(os.listdir(self.samples_lidar_dir))
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

            if i < 3:
                logger.info(f"Sample {i} (ts={s_ts}) → Sweep {nearest_idx} (diff={distances[nearest_idx]})")

        return mapping

    def _find_matching_lidar(self, cam_filename: str) -> Optional[str]:
        """Find matching LiDAR file for a camera sample by timestamp."""
        cam_ts = extract_timestamp_from_filename(cam_filename)

        best_match = None
        min_diff = float('inf')

        for lidar_file in self.lidar_files:
            lidar_ts = extract_timestamp_from_filename(lidar_file)
            diff = abs(lidar_ts - cam_ts)
            if diff < min_diff:
                min_diff = diff
                best_match = lidar_file

        return best_match

    def project_lidar_to_camera(
        self,
        points: np.ndarray,
        cam_ego_pose: Dict,
        lidar_ego_pose: Dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project LiDAR points to camera image plane.

        Transformation pipeline:
        1. LiDAR → Ego (lidar calibration)
        2. Ego(lidar) → Global (lidar ego_pose)
        3. Global → Ego(cam) (camera ego_pose inverse)
        4. Ego(cam) → Camera (camera calibration inverse)
        5. Camera 3D → 2D projection

        Args:
            points: (N, 3) LiDAR points in LiDAR frame
            cam_ego_pose: Ego pose at camera timestamp
            lidar_ego_pose: Ego pose at LiDAR timestamp

        Returns:
            u, v: (M,) pixel coordinates
            depths: (M,) depth values in meters
        """
        N = points.shape[0]

        # 1. LiDAR → Ego
        R_lidar = quaternion_to_rotation_matrix(self.lidar_rotation)
        points_ego_lidar = (R_lidar @ points.T).T + self.lidar_translation

        # 2. Ego(lidar) → Global
        R_ego_lidar = quaternion_to_rotation_matrix(np.array(lidar_ego_pose['rotation']))
        t_ego_lidar = np.array(lidar_ego_pose['translation'])
        points_global = (R_ego_lidar @ points_ego_lidar.T).T + t_ego_lidar

        # 3. Global → Ego(cam)
        R_ego_cam = quaternion_to_rotation_matrix(np.array(cam_ego_pose['rotation']))
        t_ego_cam = np.array(cam_ego_pose['translation'])
        points_ego_cam = (R_ego_cam.T @ (points_global - t_ego_cam).T).T

        # 4. Ego(cam) → Camera
        R_cam = quaternion_to_rotation_matrix(self.cam_rotation)
        points_cam = (R_cam.T @ (points_ego_cam - self.cam_translation).T).T

        # Filter points in front of camera
        valid_mask = points_cam[:, 2] > 0.1  # At least 10cm in front
        points_cam = points_cam[valid_mask]

        if len(points_cam) == 0:
            return np.array([]), np.array([]), np.array([])

        # 5. Project to image plane
        depths = points_cam[:, 2]

        # Perspective projection
        fx, fy = self.cam_intrinsics[0, 0], self.cam_intrinsics[1, 1]
        cx, cy = self.cam_intrinsics[0, 2], self.cam_intrinsics[1, 2]

        u = (fx * points_cam[:, 0] / depths + cx).astype(np.int32)
        v = (fy * points_cam[:, 1] / depths + cy).astype(np.int32)

        # Filter points within image bounds and depth range
        valid = (
            (u >= 0) & (u < self.img_width) &
            (v >= 0) & (v < self.img_height) &
            (depths > 0) & (depths < self.max_depth)
        )

        return u[valid], v[valid], depths[valid]

    def create_sparse_depth_map(
        self,
        u: np.ndarray,
        v: np.ndarray,
        depths: np.ndarray
    ) -> np.ndarray:
        """Create sparse depth map from projected points.

        Args:
            u, v: Pixel coordinates
            depths: Depth values

        Returns:
            depth_map: (H, W) sparse depth map, 0 for invalid pixels
        """
        depth_map = np.zeros((self.img_height, self.img_width), dtype=np.float32)

        if len(u) == 0:
            return depth_map

        # For overlapping points, keep the nearest (smallest depth)
        # Sort by depth descending so smaller depths overwrite
        sorted_idx = np.argsort(-depths)
        u, v, depths = u[sorted_idx], v[sorted_idx], depths[sorted_idx]

        depth_map[v, u] = depths

        return depth_map

    def generate(self):
        """Generate sparse depth GT for all samples."""
        results = {
            'data_dir': str(self.data_dir),
            'num_samples': len(self.sample_files),
            'num_sweeps': len(self.sweep_files),
            'image_size': [self.img_width, self.img_height],
            'max_depth': self.max_depth,
            'sample_to_sweep': self.sample_to_sweep_idx,
            'samples': []
        }

        # Create output directories
        depth_maps_dir = self.output_dir / 'depth_maps'
        depth_maps_dir.mkdir(exist_ok=True)

        logger.info(f"\nGenerating sparse depth GT for {len(self.sample_files)} samples...")

        for sample_idx, cam_file in enumerate(tqdm(self.sample_files, desc="Processing samples")):
            sweep_idx = self.sample_to_sweep_idx[sample_idx]

            # Find matching LiDAR file
            lidar_file = self._find_matching_lidar(cam_file)
            if lidar_file is None:
                logger.warning(f"No matching LiDAR for {cam_file}")
                continue

            # Get ego poses
            cam_sd = self.cam_sd_by_filename.get(cam_file)
            lidar_sd = self.lidar_sd_by_filename.get(lidar_file)

            if cam_sd is None or lidar_sd is None:
                logger.warning(f"Missing sample_data for {cam_file} or {lidar_file}")
                continue

            cam_ego_pose = self.ego_pose_map.get(cam_sd['ego_pose_token'])
            lidar_ego_pose = self.ego_pose_map.get(lidar_sd['ego_pose_token'])

            if cam_ego_pose is None or lidar_ego_pose is None:
                logger.warning(f"Missing ego_pose for sample {sample_idx}")
                continue

            # Load LiDAR points
            lidar_path = self.samples_lidar_dir / lidar_file
            points = load_lidar_points(lidar_path)

            # Project to camera
            u, v, depths = self.project_lidar_to_camera(points, cam_ego_pose, lidar_ego_pose)

            # Create sparse depth map
            depth_map = self.create_sparse_depth_map(u, v, depths)

            # Save depth map
            depth_map_path = depth_maps_dir / f'frame_{sweep_idx:04d}.npy'
            np.save(depth_map_path, depth_map)

            # Statistics
            valid_pixels = np.sum(depth_map > 0)
            mean_depth = np.mean(depths) if len(depths) > 0 else 0
            min_depth = np.min(depths) if len(depths) > 0 else 0
            max_depth = np.max(depths) if len(depths) > 0 else 0

            results['samples'].append({
                'sample_idx': sample_idx,
                'sweep_idx': sweep_idx,
                'cam_file': cam_file,
                'lidar_file': lidar_file,
                'valid_pixels': int(valid_pixels),
                'mean_depth': float(mean_depth),
                'min_depth': float(min_depth),
                'max_depth': float(max_depth),
                'depth_map_file': f'frame_{sweep_idx:04d}.npy'
            })

        # Save results JSON
        results_path = self.output_dir / 'sparse_depth_gt.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nSaved results to {results_path}")
        logger.info(f"Saved {len(results['samples'])} depth maps to {depth_maps_dir}")

        # Print summary
        if results['samples']:
            avg_pixels = np.mean([s['valid_pixels'] for s in results['samples']])
            avg_depth = np.mean([s['mean_depth'] for s in results['samples']])
            logger.info(f"Average valid pixels per frame: {avg_pixels:.0f}")
            logger.info(f"Average depth: {avg_depth:.2f}m")

        return results


def main():
    parser = argparse.ArgumentParser(description='Generate sparse depth GT from nuScenes LiDAR')
    parser.add_argument('--data-dir', type=str, default='/home/cvlab/hsy/Datasets/v1.0-mini',
                       help='Path to nuScenes data directory')
    parser.add_argument('--output-dir', type=str, default='test_results/crosswalk_lidar_gt',
                       help='Output directory for sparse depth GT')
    parser.add_argument('--max-depth', type=float, default=80.0,
                       help='Maximum depth threshold (meters)')

    args = parser.parse_args()

    generator = NuScenesLiDARGTGenerator(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        max_depth=args.max_depth
    )

    generator.generate()


if __name__ == '__main__':
    main()
