"""
Reprojection-based Temporal Alignment Error (TAE) calculation.

Based on Video Depth Anything's implementation.
TAE measures temporal consistency by projecting depth from one frame to another
using camera poses and comparing with the actual depth at the projected location.

Required data per frame:
- depth: [H, W] depth map in meters
- K: [3, 3] camera intrinsic matrix
- pose: [4, 4] camera-to-world transformation matrix (extrinsic)
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import struct

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

logger = logging.getLogger(__name__)


def tae_torch(
    depth1: torch.Tensor,
    depth2: torch.Tensor,
    R_2_1: torch.Tensor,
    t_2_1: torch.Tensor,
    K: torch.Tensor,
    valid_mask2: Optional[torch.Tensor] = None,
    min_depth: float = 0.1,
    max_depth: float = 70.0
) -> float:
    """
    Compute TAE between two frames using 3D reprojection.

    Steps:
    1. Backproject depth1 to 3D points in frame1's coordinate system
    2. Transform 3D points from frame1 to frame2 using relative pose
    3. Project 3D points to frame2's image plane
    4. Compare projected depth with depth2 using AbsRel metric

    Args:
        depth1: [H, W] depth map of frame 1 (meters)
        depth2: [H, W] depth map of frame 2 (meters)
        R_2_1: [3, 3] rotation matrix from frame1 to frame2
        t_2_1: [3, 1] translation vector from frame1 to frame2
        K: [3, 3] camera intrinsic matrix
        valid_mask2: [H, W] valid mask for frame 2
        min_depth: minimum valid depth (meters)
        max_depth: maximum valid depth (meters)

    Returns:
        float: AbsRel error for this frame pair
    """
    H, W = depth1.shape
    device = depth1.device
    dtype = depth1.dtype

    # Create pixel grid
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )

    # Get camera intrinsics
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Backproject to 3D (frame1 coordinates)
    X = (x_coords - cx) * depth1 / fx
    Y = (y_coords - cy) * depth1 / fy
    Z = depth1

    # Stack as [3, H*W]
    points3d = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=0)  # [3, H*W]

    # Transform to frame2 coordinates
    # points3d_transformed = R_2_1 @ points3d + t_2_1
    points3d_transformed = R_2_1 @ points3d + t_2_1  # [3, H*W]

    # Project to frame2 image plane
    X_proj = points3d_transformed[0]
    Y_proj = points3d_transformed[1]
    Z_proj = points3d_transformed[2]

    # Projected pixel coordinates
    x_proj = (X_proj * fx) / Z_proj + cx
    y_proj = (Y_proj * fy) / Z_proj + cy

    # Projected depth
    depth_proj = Z_proj.reshape(H, W)
    x_proj = x_proj.reshape(H, W)
    y_proj = y_proj.reshape(H, W)

    # Create validity mask
    # 1. Projected points within image bounds
    valid_proj = (x_proj >= 0) & (x_proj < W) & (y_proj >= 0) & (y_proj < H)
    # 2. Positive projected depth
    valid_proj = valid_proj & (depth_proj > min_depth) & (depth_proj < max_depth)
    # 3. Valid depth in frame1
    valid_depth1 = (depth1 > min_depth) & (depth1 < max_depth)
    # 4. Valid depth in frame2
    valid_depth2 = (depth2 > min_depth) & (depth2 < max_depth)

    # Combined mask
    valid_mask = valid_proj & valid_depth1 & valid_depth2
    if valid_mask2 is not None:
        valid_mask = valid_mask & valid_mask2

    if valid_mask.sum() == 0:
        return float('nan')

    # Sample depth2 at projected locations using bilinear interpolation
    # Normalize coordinates to [-1, 1] for grid_sample
    x_norm = 2.0 * x_proj / (W - 1) - 1.0
    y_norm = 2.0 * y_proj / (H - 1) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)  # [1, H, W, 2]

    depth2_sampled = torch.nn.functional.grid_sample(
        depth2.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    ).squeeze()  # [H, W]

    # Compute AbsRel error
    abs_rel = torch.abs(depth2_sampled - depth_proj) / depth2_sampled

    # Apply mask and compute mean
    abs_rel_masked = abs_rel[valid_mask]

    return abs_rel_masked.mean().item()


def compute_reprojection_tae(
    pred_depths: torch.Tensor,
    gt_depths: torch.Tensor,
    intrinsics: List[torch.Tensor],
    poses: List[torch.Tensor],
    valid_masks: Optional[List[torch.Tensor]] = None,
    min_depth: float = 0.1,
    max_depth: float = 70.0
) -> Dict[str, float]:
    """
    Compute reprojection-based TAE for a sequence.

    Follows Video Depth Anything's approach:
    1. Scale-shift align predictions to GT (in disparity space)
    2. For each consecutive frame pair:
       - Forward: project frame t to frame t+1, compare
       - Backward: project frame t+1 to frame t, compare
    3. Average all errors

    Args:
        pred_depths: [T, H, W] predicted depth sequence (meters)
        gt_depths: [T, H, W] ground truth depth sequence (meters)
        intrinsics: List of [3, 3] intrinsic matrices (one per frame or single for all)
        poses: List of [4, 4] camera-to-world poses (one per frame)
        valid_masks: Optional list of [H, W] valid masks
        min_depth: minimum valid depth
        max_depth: maximum valid depth

    Returns:
        Dict with 'tae_reproj' (reprojection TAE) and 'tae_reproj_pred' (pred-only TAE)
    """
    T = pred_depths.shape[0]
    device = pred_depths.device

    if T < 2:
        return {'tae_reproj': 0.0, 'tae_reproj_pred': 0.0}

    # Ensure poses are tensors
    poses = [p.to(device) if isinstance(p, torch.Tensor) else torch.tensor(p, device=device, dtype=torch.float32) for p in poses]

    # Handle single intrinsic matrix for all frames
    if len(intrinsics) == 1:
        intrinsics = intrinsics * T
    intrinsics = [K.to(device) if isinstance(K, torch.Tensor) else torch.tensor(K, device=device, dtype=torch.float32) for K in intrinsics]

    # Scale-shift alignment in disparity space (like Video Depth Anything)
    # Convert to disparity
    pred_disp = 1.0 / (pred_depths.clamp(min=1e-3) + 1e-8)
    gt_disp = 1.0 / (gt_depths.clamp(min=1e-3) + 1e-8)

    # Create valid mask for alignment
    valid_for_align = (gt_depths > min_depth) & (gt_depths < max_depth) & (pred_depths > min_depth) & (pred_depths < max_depth)

    # Least squares alignment: gt_disp ≈ scale * pred_disp + shift
    pred_disp_flat = pred_disp[valid_for_align].flatten()
    gt_disp_flat = gt_disp[valid_for_align].flatten()

    if len(pred_disp_flat) > 100:
        # Solve least squares: [pred_disp, 1] @ [scale, shift]^T = gt_disp
        A = torch.stack([pred_disp_flat, torch.ones_like(pred_disp_flat)], dim=1)
        b = gt_disp_flat
        # Normal equations: (A^T A) x = A^T b
        ATA = A.T @ A
        ATb = A.T @ b
        try:
            params = torch.linalg.solve(ATA, ATb)
            scale, shift = params[0].item(), params[1].item()
        except:
            scale, shift = 1.0, 0.0
    else:
        scale, shift = 1.0, 0.0

    # Apply alignment
    pred_disp_aligned = scale * pred_disp + shift
    pred_depths_aligned = 1.0 / (pred_disp_aligned.clamp(min=1e-8))

    # Compute TAE for consecutive frame pairs
    tae_errors_gt = []
    tae_errors_pred = []

    for t in range(T - 1):
        # Get intrinsics (use frame t's intrinsics)
        K = intrinsics[t]

        # Compute relative pose: T_2_1 = T_2^{-1} @ T_1
        # Where T_i is camera-to-world pose for frame i
        T_1 = poses[t]  # [4, 4]
        T_2 = poses[t + 1]  # [4, 4]

        # T_2_1: transforms points from frame1 coords to frame2 coords
        T_2_inv = torch.linalg.inv(T_2)
        T_2_1 = T_2_inv @ T_1  # [4, 4]

        R_2_1 = T_2_1[:3, :3]  # [3, 3]
        t_2_1 = T_2_1[:3, 3:4]  # [3, 1]

        # Inverse: T_1_2 = T_1^{-1} @ T_2
        T_1_inv = torch.linalg.inv(T_1)
        T_1_2 = T_1_inv @ T_2
        R_1_2 = T_1_2[:3, :3]
        t_1_2 = T_1_2[:3, 3:4]

        # Get valid masks
        mask_t = valid_masks[t] if valid_masks else None
        mask_t1 = valid_masks[t + 1] if valid_masks else None

        # Forward TAE (frame t → frame t+1) using GT depth
        error_fwd_gt = tae_torch(
            gt_depths[t], gt_depths[t + 1],
            R_2_1, t_2_1, K, mask_t1,
            min_depth, max_depth
        )

        # Backward TAE (frame t+1 → frame t) using GT depth
        error_bwd_gt = tae_torch(
            gt_depths[t + 1], gt_depths[t],
            R_1_2, t_1_2, K, mask_t,
            min_depth, max_depth
        )

        # Forward TAE using aligned predictions
        error_fwd_pred = tae_torch(
            pred_depths_aligned[t], pred_depths_aligned[t + 1],
            R_2_1, t_2_1, K, mask_t1,
            min_depth, max_depth
        )

        # Backward TAE using aligned predictions
        error_bwd_pred = tae_torch(
            pred_depths_aligned[t + 1], pred_depths_aligned[t],
            R_1_2, t_1_2, K, mask_t,
            min_depth, max_depth
        )

        # Collect errors
        if not np.isnan(error_fwd_gt):
            tae_errors_gt.append(error_fwd_gt)
        if not np.isnan(error_bwd_gt):
            tae_errors_gt.append(error_bwd_gt)
        if not np.isnan(error_fwd_pred):
            tae_errors_pred.append(error_fwd_pred)
        if not np.isnan(error_bwd_pred):
            tae_errors_pred.append(error_bwd_pred)

    # Compute final TAE (×100 for percentage)
    tae_gt = np.mean(tae_errors_gt) * 100 if tae_errors_gt else 0.0
    tae_pred = np.mean(tae_errors_pred) * 100 if tae_errors_pred else 0.0

    return {
        'tae_reproj': tae_pred,  # Main TAE metric (on predictions)
        'tae_reproj_gt': tae_gt,  # Reference TAE on GT (should be ~0 for perfect poses)
    }


# ============================================================================
# Dataset-specific camera pose loaders
# ============================================================================

def load_sintel_camera(cam_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Sintel camera intrinsics and extrinsics from .cam file.

    Binary format:
    - TAG_FLOAT (float32): validation value (202021.25)
    - Intrinsic matrix M: 9 float64 values (3×3)
    - Extrinsic matrix N: 12 float64 values (3×4) [R|t]

    Args:
        cam_path: Path to .cam file

    Returns:
        K: [3, 3] intrinsic matrix
        extrinsic: [4, 4] world-to-camera matrix
    """
    TAG_FLOAT = 202021.25

    with open(cam_path, 'rb') as f:
        # Read TAG_FLOAT
        tag = np.fromfile(f, dtype=np.float32, count=1)[0]
        if abs(tag - TAG_FLOAT) > 0.01:
            logger.warning(f"Unexpected tag in {cam_path}: {tag}")

        # Read intrinsic matrix (3×3)
        K = np.fromfile(f, dtype=np.float64, count=9).reshape(3, 3)

        # Read extrinsic matrix (3×4) [R|t]
        extrinsic_3x4 = np.fromfile(f, dtype=np.float64, count=12).reshape(3, 4)

    # Convert to 4×4 world-to-camera matrix
    extrinsic = np.eye(4)
    extrinsic[:3, :] = extrinsic_3x4

    return K, extrinsic


def load_eth3d_cameras(scene_dir: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load ETH3D camera intrinsics and poses from COLMAP format files.

    Files:
    - cameras.txt: Camera intrinsics (CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[])
    - images.txt: Image poses (IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME)

    Args:
        scene_dir: Path to scene directory containing dslr_calibration_undistorted/

    Returns:
        Dict mapping image_name to (K, pose) where pose is camera-to-world [4, 4]
    """
    calib_dir = Path(scene_dir) / 'dslr_calibration_undistorted'
    cameras_path = calib_dir / 'cameras.txt'
    images_path = calib_dir / 'images.txt'

    # Parse cameras.txt
    cameras = {}
    with open(cameras_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])

            if model == 'PINHOLE':
                fx, fy, cx, cy = map(float, parts[4:8])
            elif model == 'SIMPLE_PINHOLE':
                f_val = float(parts[4])
                fx = fy = f_val
                cx, cy = map(float, parts[5:7])
            else:
                # Default: assume first 4 params are fx, fy, cx, cy
                fx, fy, cx, cy = map(float, parts[4:8])

            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
            cameras[camera_id] = (K, width, height)

    # Parse images.txt
    result = {}
    with open(images_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#') or not line:
            i += 1
            continue

        parts = line.split()
        if len(parts) < 10:
            i += 1
            continue

        # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        image_name = parts[9]

        # Skip POINTS2D line
        i += 2

        # Get intrinsics for this camera
        if camera_id not in cameras:
            continue
        K, _, _ = cameras[camera_id]

        # Convert quaternion to rotation matrix
        # COLMAP uses (qw, qx, qy, qz) convention
        R = quat_to_rotation_matrix(qw, qx, qy, qz)

        # COLMAP stores world-to-camera transform
        # t_wc = -R^T @ t
        t = np.array([[tx], [ty], [tz]])

        # World-to-camera matrix
        T_wc = np.eye(4)
        T_wc[:3, :3] = R
        T_wc[:3, 3:4] = t

        # Camera-to-world matrix (what we need)
        T_cw = np.linalg.inv(T_wc)

        result[image_name] = (K.copy(), T_cw)

    return result


def load_bonn_poses(scene_dir: str) -> Tuple[np.ndarray, Dict[float, np.ndarray]]:
    """
    Load Bonn dataset camera intrinsics and poses.

    Bonn uses TUM RGB-D format:
    - Fixed intrinsics (Kinect sensor): fx=525, fy=525, cx=319.5, cy=239.5
    - groundtruth.txt: timestamp tx ty tz qx qy qz qw

    Args:
        scene_dir: Path to scene directory (e.g., rgbd_bonn_balloon)

    Returns:
        K: [3, 3] intrinsic matrix
        poses: Dict mapping timestamp to [4, 4] camera-to-world pose
    """
    # Fixed Kinect intrinsics for 640x480 resolution
    K = np.array([
        [525.0, 0, 319.5],
        [0, 525.0, 239.5],
        [0, 0, 1]
    ])

    gt_path = Path(scene_dir) / 'groundtruth.txt'
    poses = {}

    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            timestamp = float(parts[0])
            tx, ty, tz = map(float, parts[1:4])
            qx, qy, qz, qw = map(float, parts[4:8])

            # Convert to rotation matrix
            R = quat_to_rotation_matrix(qw, qx, qy, qz)

            # Camera-to-world pose
            T_cw = np.eye(4)
            T_cw[:3, :3] = R
            T_cw[:3, 3] = [tx, ty, tz]

            poses[timestamp] = T_cw

    return K, poses


def load_vkitti_cameras(scene_dir: str, variation: str = 'clone') -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Load VKitti2 camera intrinsics and extrinsics.

    Files:
    - intrinsic.txt: frame cameraID K[0,0] K[1,1] K[0,2] K[1,2]
    - extrinsic.txt: frame cameraID r1,1 r1,2 ... t3 0 0 0 1

    Args:
        scene_dir: Path to scene directory (e.g., Scene01)
        variation: Variation name (e.g., 'clone', 'fog', etc.)

    Returns:
        intrinsics: Dict mapping frame_id to [3, 3] intrinsic matrix
        extrinsics: Dict mapping frame_id to [4, 4] camera-to-world pose
    """
    var_dir = Path(scene_dir) / variation
    intrinsic_path = var_dir / 'intrinsic.txt'
    extrinsic_path = var_dir / 'extrinsic.txt'

    # Parse intrinsics
    intrinsics = {}
    with open(intrinsic_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('frame') or not line:  # Skip header
                continue
            parts = line.split()
            frame_id = int(parts[0])
            camera_id = int(parts[1])

            # Use camera 0 (left camera)
            if camera_id != 0:
                continue

            fx, fy, cx, cy = map(float, parts[2:6])
            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
            intrinsics[frame_id] = K

    # Parse extrinsics
    extrinsics = {}
    with open(extrinsic_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('frame') or not line:  # Skip header
                continue
            parts = line.split()
            frame_id = int(parts[0])
            camera_id = int(parts[1])

            # Use camera 0 (left camera)
            if camera_id != 0:
                continue

            # Read 4×4 matrix (r1,1 r1,2 r1,3 t1 r2,1 r2,2 r2,3 t2 r3,1 r3,2 r3,3 t3 0 0 0 1)
            values = list(map(float, parts[2:]))
            T = np.array(values).reshape(4, 4)

            # VKitti extrinsic is world-to-camera, we need camera-to-world
            T_cw = np.linalg.inv(T)
            extrinsics[frame_id] = T_cw

    return intrinsics, extrinsics


def quat_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """
    Convert quaternion to 3×3 rotation matrix.

    Args:
        qw, qx, qy, qz: Quaternion components (scalar-first convention)

    Returns:
        R: [3, 3] rotation matrix
    """
    # Normalize quaternion
    norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm

    # Rotation matrix from quaternion
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
    ])

    return R


def load_waymo_seg_cameras(
    segment_name: str,
    camera_name: str = 'FRONT',
    waymo_seg_root: str = None
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    Load Waymo Segmentation dataset camera intrinsics and poses.

    Waymo data sources:
    - camera_image/*.parquet: Frame-by-frame pose (vehicle_to_world)
    - camera_calibration/*.parquet: Camera intrinsics and extrinsic (camera_to_vehicle)

    Camera-to-world pose = vehicle_to_world @ camera_to_vehicle

    Args:
        segment_name: Segment name (e.g., '10017090168044687777_6380_000_6400_000')
                     Can include 'segment-' prefix
        camera_name: Camera name ('FRONT' = 1, 'FRONT_LEFT' = 2, etc.)
        waymo_seg_root: Root directory containing waymo_seg/

    Returns:
        K: [3, 3] intrinsic matrix (for original 1920×1280 resolution)
        poses: Dict mapping frame_index to [4, 4] camera-to-world pose
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for Waymo dataset support. Install with: pip install pandas pyarrow")

    # Remove 'segment-' prefix if present
    if segment_name.startswith('segment-'):
        segment_name = segment_name[len('segment-'):]

    # Camera name to ID mapping
    camera_name_to_id = {
        'FRONT': 1,
        'FRONT_LEFT': 2,
        'FRONT_RIGHT': 3,
        'SIDE_LEFT': 4,
        'SIDE_RIGHT': 5
    }
    camera_id = camera_name_to_id.get(camera_name, 1)

    # Find parquet files
    waymo_seg_root = Path(waymo_seg_root) if waymo_seg_root else Path('.')
    camera_image_path = waymo_seg_root / 'waymo_seg' / 'camera_image' / f'{segment_name}.parquet'
    camera_calib_path = waymo_seg_root / 'waymo_seg' / 'camera_calibration' / f'{segment_name}.parquet'

    if not camera_image_path.exists():
        raise FileNotFoundError(f"Camera image parquet not found: {camera_image_path}")
    if not camera_calib_path.exists():
        raise FileNotFoundError(f"Camera calibration parquet not found: {camera_calib_path}")

    # Load camera calibration
    calib_df = pd.read_parquet(camera_calib_path)
    camera_calib = calib_df[calib_df['key.camera_name'] == camera_id].iloc[0]

    # Extract intrinsics
    fx = camera_calib['[CameraCalibrationComponent].intrinsic.f_u']
    fy = camera_calib['[CameraCalibrationComponent].intrinsic.f_v']
    cx = camera_calib['[CameraCalibrationComponent].intrinsic.c_u']
    cy = camera_calib['[CameraCalibrationComponent].intrinsic.c_v']

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    # Extract camera-to-vehicle extrinsic
    camera_to_vehicle = np.array(
        camera_calib['[CameraCalibrationComponent].extrinsic.transform']
    ).reshape(4, 4)

    # Load camera images (for poses)
    image_df = pd.read_parquet(camera_image_path)
    camera_df = image_df[image_df['key.camera_name'] == camera_id].sort_values('key.frame_timestamp_micros')

    # Build frame_index -> pose mapping
    # Frame index corresponds to sorted timestamp order (0, 1, 2, ...)
    poses = {}
    for frame_idx, (_, row) in enumerate(camera_df.iterrows()):
        # Vehicle-to-world pose
        vehicle_to_world = np.array(row['[CameraImageComponent].pose.transform']).reshape(4, 4)

        # Camera-to-world = vehicle_to_world @ camera_to_vehicle
        camera_to_world = vehicle_to_world @ camera_to_vehicle
        poses[frame_idx] = camera_to_world

    return K, poses


# ============================================================================
# High-level interface for test_gear5.py
# ============================================================================

class ReprojectionTAECalculator:
    """
    Calculator for reprojection-based TAE that handles different datasets.
    """

    SUPPORTED_DATASETS = ['sintel', 'eth3d', 'bonn', 'vkitti', 'waymo_seg']

    def __init__(self, data_root: str):
        """
        Initialize calculator.

        Args:
            data_root: Root directory containing datasets
        """
        self.data_root = Path(data_root)
        self._cache = {}  # Cache for loaded camera data

    def is_supported(self, dataset_name: str) -> bool:
        """Check if dataset supports reprojection TAE."""
        return dataset_name.lower() in self.SUPPORTED_DATASETS

    def compute_tae(
        self,
        pred_depths: torch.Tensor,
        gt_depths: torch.Tensor,
        dataset_name: str,
        image_paths: List[str],
        valid_masks: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, float]:
        """
        Compute reprojection TAE for a sequence.

        Args:
            pred_depths: [T, H, W] predicted depths (meters)
            gt_depths: [T, H, W] GT depths (meters)
            dataset_name: Name of dataset
            image_paths: List of image paths for each frame
            valid_masks: Optional valid masks

        Returns:
            Dict with TAE metrics
        """
        dataset_name = dataset_name.lower()

        if not self.is_supported(dataset_name):
            return {'tae_reproj': 0.0, 'tae_reproj_gt': 0.0, 'tae_reproj_supported': False}

        try:
            # Load camera data for each frame
            intrinsics = []
            poses = []

            for img_path in image_paths:
                K, pose = self._get_camera_data(dataset_name, img_path)
                if K is None or pose is None:
                    logger.warning(f"Failed to get camera data for {img_path}")
                    return {'tae_reproj': 0.0, 'tae_reproj_gt': 0.0, 'tae_reproj_supported': False}

                intrinsics.append(torch.tensor(K, dtype=torch.float32))
                poses.append(torch.tensor(pose, dtype=torch.float32))

            # Scale intrinsics to match current image resolution
            H, W = pred_depths.shape[1:]
            intrinsics = self._scale_intrinsics(intrinsics, dataset_name, image_paths[0], (H, W))

            # Compute TAE
            result = compute_reprojection_tae(
                pred_depths, gt_depths,
                intrinsics, poses,
                valid_masks
            )
            result['tae_reproj_supported'] = True

            return result

        except Exception as e:
            logger.error(f"Error computing reprojection TAE: {e}")
            import traceback
            traceback.print_exc()
            return {'tae_reproj': 0.0, 'tae_reproj_gt': 0.0, 'tae_reproj_supported': False}

    def _get_camera_data(self, dataset_name: str, img_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get intrinsics and pose for a single frame."""

        if dataset_name == 'sintel':
            # Convert image path to camera path
            cam_path = img_path.replace('images/training/clean', 'cam_data/training/camdata_left').replace('.png', '.cam')

            if not Path(cam_path).exists():
                return None, None

            K, extrinsic = load_sintel_camera(cam_path)
            # Sintel extrinsic is world-to-camera, convert to camera-to-world
            pose = np.linalg.inv(extrinsic)
            return K, pose

        elif dataset_name == 'eth3d':
            # Get scene directory and image name
            # Path format: .../eth3d/{scene}/images/{image}.jpg
            parts = Path(img_path).parts
            scene_idx = parts.index('eth3d') + 1 if 'eth3d' in parts else -1
            if scene_idx < 0 or scene_idx >= len(parts):
                return None, None

            scene_name = parts[scene_idx]
            scene_dir = self.data_root / 'eth3d' / scene_name
            image_name = Path(img_path).name

            # Load all cameras for this scene (cached)
            cache_key = f"eth3d_{scene_name}"
            if cache_key not in self._cache:
                self._cache[cache_key] = load_eth3d_cameras(str(scene_dir))

            cameras = self._cache[cache_key]

            # Find matching camera
            # Try exact match first
            if image_name in cameras:
                return cameras[image_name]

            # Try with different path format
            for cam_name, (K, pose) in cameras.items():
                if Path(cam_name).stem == Path(image_name).stem:
                    return K, pose

            return None, None

        elif dataset_name == 'bonn':
            # Get scene directory
            # Path format: .../bonn/rgbd_bonn_{name}/rgb/{timestamp}.png
            parts = Path(img_path).parts
            scene_name = None
            for part in parts:
                if part.startswith('rgbd_bonn_'):
                    scene_name = part
                    break

            if scene_name is None:
                return None, None

            scene_dir = self.data_root / 'bonn' / scene_name

            # Load poses (cached)
            cache_key = f"bonn_{scene_name}"
            if cache_key not in self._cache:
                self._cache[cache_key] = load_bonn_poses(str(scene_dir))

            K, poses_dict = self._cache[cache_key]

            # Get timestamp from image filename
            timestamp = float(Path(img_path).stem)

            # Find closest pose
            closest_ts = min(poses_dict.keys(), key=lambda t: abs(t - timestamp))
            pose = poses_dict[closest_ts]

            return K, pose

        elif dataset_name == 'vkitti':
            # Get scene and variation
            # Path format: .../vkitti/{scene}/{variation}/frames/rgb/Camera_0/rgb_{frame}.jpg
            parts = Path(img_path).parts

            # Find vkitti index
            vkitti_idx = -1
            for i, part in enumerate(parts):
                if part == 'vkitti':
                    vkitti_idx = i
                    break

            if vkitti_idx < 0:
                return None, None

            scene_name = parts[vkitti_idx + 1]  # e.g., Scene01
            variation = parts[vkitti_idx + 2]   # e.g., clone

            scene_dir = self.data_root / 'vkitti' / scene_name

            # Load cameras (cached)
            cache_key = f"vkitti_{scene_name}_{variation}"
            if cache_key not in self._cache:
                self._cache[cache_key] = load_vkitti_cameras(str(scene_dir), variation)

            intrinsics, extrinsics = self._cache[cache_key]

            # Get frame number from filename (e.g., rgb_00001.jpg -> 1)
            frame_name = Path(img_path).stem
            frame_num = int(frame_name.split('_')[1])

            if frame_num not in intrinsics or frame_num not in extrinsics:
                return None, None

            return intrinsics[frame_num], extrinsics[frame_num]

        elif dataset_name == 'waymo_seg':
            # Path format: .../waymo_seg/val/segment-{segment_name}/{camera}/rgb/original/{frame:04d}.jpg
            # Or: .../waymo_seg/val/segment-{segment_name}/{camera}/rgb/{frame:04d}.jpg
            path = Path(img_path)
            parts = path.parts

            # Find segment name (starts with 'segment-')
            segment_name = None
            camera_name = None
            for i, part in enumerate(parts):
                if part.startswith('segment-'):
                    segment_name = part
                    # Camera name is next directory
                    if i + 1 < len(parts):
                        camera_name = parts[i + 1]
                    break

            if segment_name is None:
                logger.warning(f"Could not find segment name in path: {img_path}")
                return None, None

            if camera_name is None:
                camera_name = 'FRONT'

            # Get frame index from filename (e.g., 0000.jpg -> 0)
            frame_idx = int(path.stem)

            # Load camera data (cached per segment)
            cache_key = f"waymo_seg_{segment_name}_{camera_name}"
            if cache_key not in self._cache:
                try:
                    K, poses_dict = load_waymo_seg_cameras(
                        segment_name, camera_name, str(self.data_root)
                    )
                    self._cache[cache_key] = (K, poses_dict)
                except Exception as e:
                    logger.error(f"Failed to load Waymo cameras for {segment_name}: {e}")
                    return None, None

            K, poses_dict = self._cache[cache_key]

            if frame_idx not in poses_dict:
                logger.warning(f"Frame {frame_idx} not found in poses for {segment_name}")
                return None, None

            return K.copy(), poses_dict[frame_idx].copy()

        return None, None

    def _scale_intrinsics(
        self,
        intrinsics: List[torch.Tensor],
        dataset_name: str,
        sample_path: str,
        current_shape: Tuple[int, int]
    ) -> List[torch.Tensor]:
        """Scale intrinsics to match current image resolution."""
        H_curr, W_curr = current_shape

        # Get original resolution for each dataset
        if dataset_name == 'sintel':
            W_orig, H_orig = 1024, 436
        elif dataset_name == 'eth3d':
            # ETH3D has variable resolution, get from intrinsics
            # Assume intrinsics are already for original resolution
            W_orig, H_orig = 6048, 4032  # Typical ETH3D resolution
        elif dataset_name == 'bonn':
            W_orig, H_orig = 640, 480
        elif dataset_name == 'vkitti':
            W_orig, H_orig = 1242, 375
        elif dataset_name == 'waymo_seg':
            W_orig, H_orig = 1920, 1280  # Waymo original resolution
        else:
            return intrinsics

        scale_x = W_curr / W_orig
        scale_y = H_curr / H_orig

        scaled = []
        for K in intrinsics:
            K_scaled = K.clone()
            K_scaled[0, 0] *= scale_x  # fx
            K_scaled[1, 1] *= scale_y  # fy
            K_scaled[0, 2] *= scale_x  # cx
            K_scaled[1, 2] *= scale_y  # cy
            scaled.append(K_scaled)

        return scaled

    def clear_cache(self):
        """Clear cached camera data."""
        self._cache.clear()
