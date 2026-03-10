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
    valid_mask1: Optional[torch.Tensor] = None,
    valid_mask2: Optional[torch.Tensor] = None,
    min_depth: float = 0.1,
    max_depth: float = 70.0
) -> float:
    """
    Compute TAE between two frames using 3D reprojection.

    Follows DepthAnyVideo's sparse projection approach:
    1. Only backproject VALID pixels from frame1
    2. Create sparse projected depth map in frame2
    3. Compare only where projection exists AND frame2 depth is valid

    This naturally handles occlusion by not comparing occluded regions.

    Args:
        depth1: [H, W] depth map of frame 1 (meters)
        depth2: [H, W] depth map of frame 2 (meters)
        R_2_1: [3, 3] rotation matrix from frame1 to frame2
        t_2_1: [3, 1] translation vector from frame1 to frame2
        K: [3, 3] camera intrinsic matrix
        valid_mask1: [H, W] valid mask for frame 1 (optional)
        valid_mask2: [H, W] valid mask for frame 2 (optional)
        min_depth: minimum valid depth (meters)
        max_depth: maximum valid depth (meters)

    Returns:
        float: AbsRel error for this frame pair
    """
    H, W = depth1.shape
    device = depth1.device
    dtype = depth1.dtype

    # Create pixel grid with 0.5 offset (pixel center convention, like DepthAnyVideo)
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(0.5, H - 0.5, H, device=device, dtype=dtype),
        torch.linspace(0.5, W - 0.5, W, device=device, dtype=dtype),
        indexing='ij'
    )

    # Get camera intrinsics
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Create valid mask for source frame
    mask1 = (depth1 > min_depth) & (depth1 < max_depth)
    if valid_mask1 is not None:
        mask1 = mask1 & valid_mask1

    # Extract only valid pixels (sparse)
    valid_indices = mask1.flatten()
    x_valid = x_coords.flatten()[valid_indices]
    y_valid = y_coords.flatten()[valid_indices]
    d_valid = depth1.flatten()[valid_indices]

    if d_valid.numel() == 0:
        return float('nan')

    # Backproject valid pixels to 3D (frame1 coordinates)
    X = (x_valid - cx) * d_valid / fx
    Y = (y_valid - cy) * d_valid / fy
    Z = d_valid
    points3d = torch.stack([X, Y, Z], dim=0)  # [3, N_valid]

    # Transform to frame2 coordinates
    points3d_transformed = R_2_1 @ points3d + t_2_1  # [3, N_valid]

    X_proj = points3d_transformed[0]
    Y_proj = points3d_transformed[1]
    Z_proj = points3d_transformed[2]

    # Project to frame2 image plane
    eps = 1e-6
    valid_z = Z_proj > eps
    x_proj = torch.zeros_like(X_proj)
    y_proj = torch.zeros_like(Y_proj)
    x_proj[valid_z] = (X_proj[valid_z] * fx) / Z_proj[valid_z] + cx
    y_proj[valid_z] = (Y_proj[valid_z] * fy) / Z_proj[valid_z] + cy

    # Round to integer pixel coordinates (nearest neighbor)
    x_int = torch.round(x_proj).long()
    y_int = torch.round(y_proj).long()

    # Create validity mask for projected points
    valid_proj = (
        valid_z &
        (x_int >= 0) & (x_int < W) &
        (y_int >= 0) & (y_int < H) &
        (Z_proj > min_depth) & (Z_proj < max_depth)
    )

    if valid_proj.sum() == 0:
        return float('nan')

    # Extract valid projections
    x_int_valid = x_int[valid_proj]
    y_int_valid = y_int[valid_proj]
    depth_proj_valid = Z_proj[valid_proj]

    # Create sparse projected depth map (like DepthAnyVideo's point2depth)
    # Note: If multiple points project to same pixel, last one wins (same as DepthAnyVideo)
    projected_depth = torch.zeros((H, W), device=device, dtype=dtype)
    projected_depth[y_int_valid, x_int_valid] = depth_proj_valid

    # Create mask for valid frame2 depth
    mask2 = (depth2 > min_depth) & (depth2 < max_depth)
    if valid_mask2 is not None:
        mask2 = mask2 & valid_mask2

    # Final comparison mask: where projection exists AND frame2 depth is valid
    compare_mask = (projected_depth > eps) & mask2

    if compare_mask.sum() == 0:
        return float('nan')

    # Compute AbsRel error
    abs_rel = torch.abs(depth2[compare_mask] - projected_depth[compare_mask]) / depth2[compare_mask]

    return abs_rel.mean().item()


def compute_reprojection_tae(
    pred_depths: torch.Tensor,
    gt_depths: torch.Tensor,
    intrinsics: List[torch.Tensor],
    poses: List[torch.Tensor],
    valid_masks: Optional[List[torch.Tensor]] = None,
    min_depth: float = 0.1,
    max_depth: float = 70.0,
    align_to_gt: bool = False
) -> Dict[str, float]:
    """
    Compute reprojection-based TAE for a sequence.

    For each consecutive frame pair:
       - Forward: project frame t to frame t+1, compare
       - Backward: project frame t+1 to frame t, compare
    Average all errors.

    Args:
        pred_depths: [T, H, W] predicted depth sequence (meters)
        gt_depths: [T, H, W] ground truth depth sequence (meters)
        intrinsics: List of [3, 3] intrinsic matrices (one per frame or single for all)
        poses: List of [4, 4] camera-to-world poses (one per frame)
        valid_masks: Optional list of [H, W] valid masks
        min_depth: minimum valid depth
        max_depth: maximum valid depth
        align_to_gt: If True, align predictions to GT in disparity space (for relative depth models).
                     If False, use predictions directly (for metric depth models like Gear5).

    Returns:
        Dict with 'tae_reproj' (reprojection TAE) and 'tae_reproj_gt' (GT reference TAE)
    """
    T = pred_depths.shape[0]
    device = pred_depths.device

    if T < 2:
        return {'tae_reproj': 0.0, 'tae_reproj_gt': 0.0}

    # Ensure poses are tensors
    poses = [p.to(device) if isinstance(p, torch.Tensor) else torch.tensor(p, device=device, dtype=torch.float32) for p in poses]

    # Handle single intrinsic matrix for all frames
    if len(intrinsics) == 1:
        intrinsics = intrinsics * T
    intrinsics = [K.to(device) if isinstance(K, torch.Tensor) else torch.tensor(K, device=device, dtype=torch.float32) for K in intrinsics]

    # Optionally align predictions to GT (for relative depth models)
    if align_to_gt:
        # Scale-shift alignment in disparity space (like Video Depth Anything)
        pred_disp = 1.0 / (pred_depths.clamp(min=1e-3) + 1e-8)
        gt_disp = 1.0 / (gt_depths.clamp(min=1e-3) + 1e-8)

        valid_for_align = (gt_depths > min_depth) & (gt_depths < max_depth) & (pred_depths > min_depth) & (pred_depths < max_depth)
        pred_disp_flat = pred_disp[valid_for_align].flatten()
        gt_disp_flat = gt_disp[valid_for_align].flatten()

        if len(pred_disp_flat) > 100:
            A = torch.stack([pred_disp_flat, torch.ones_like(pred_disp_flat)], dim=1)
            b = gt_disp_flat
            ATA = A.T @ A
            ATb = A.T @ b
            try:
                params = torch.linalg.solve(ATA, ATb)
                scale, shift = params[0].item(), params[1].item()
            except:
                scale, shift = 1.0, 0.0
        else:
            scale, shift = 1.0, 0.0

        pred_disp_aligned = scale * pred_disp + shift
        pred_depths_aligned = 1.0 / (pred_disp_aligned.clamp(min=1e-8))
    else:
        # For metric depth models: use predictions directly without alignment
        pred_depths_aligned = pred_depths

    # Compute TAE for consecutive frame pairs
    tae_errors_gt = []
    tae_errors_pred = []
    nan_count_gt = 0
    nan_count_pred = 0
    small_translation_count = 0

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

        # Check for small translation (pure rotation case)
        translation_magnitude = torch.norm(t_2_1).item()
        if translation_magnitude < 0.01:  # Less than 1cm
            small_translation_count += 1

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
            R_2_1, t_2_1, K,
            valid_mask1=mask_t, valid_mask2=mask_t1,
            min_depth=min_depth, max_depth=max_depth
        )

        # Backward TAE (frame t+1 → frame t) using GT depth
        error_bwd_gt = tae_torch(
            gt_depths[t + 1], gt_depths[t],
            R_1_2, t_1_2, K,
            valid_mask1=mask_t1, valid_mask2=mask_t,
            min_depth=min_depth, max_depth=max_depth
        )

        # Forward TAE using predictions (no alignment for metric depth models)
        error_fwd_pred = tae_torch(
            pred_depths_aligned[t], pred_depths_aligned[t + 1],
            R_2_1, t_2_1, K,
            valid_mask1=mask_t, valid_mask2=mask_t1,
            min_depth=min_depth, max_depth=max_depth
        )

        # Backward TAE using predictions
        error_bwd_pred = tae_torch(
            pred_depths_aligned[t + 1], pred_depths_aligned[t],
            R_1_2, t_1_2, K,
            valid_mask1=mask_t1, valid_mask2=mask_t,
            min_depth=min_depth, max_depth=max_depth
        )

        # Compute per-frame-pair TAE (average of forward and backward)
        # GT TAE for this frame pair
        if not np.isnan(error_fwd_gt) and not np.isnan(error_bwd_gt):
            pair_tae_gt = 0.5 * (error_fwd_gt + error_bwd_gt)
        elif not np.isnan(error_fwd_gt):
            pair_tae_gt = error_fwd_gt
        elif not np.isnan(error_bwd_gt):
            pair_tae_gt = error_bwd_gt
        else:
            pair_tae_gt = float('nan')
            nan_count_gt += 1

        # Pred TAE for this frame pair
        if not np.isnan(error_fwd_pred) and not np.isnan(error_bwd_pred):
            pair_tae_pred = 0.5 * (error_fwd_pred + error_bwd_pred)
        elif not np.isnan(error_fwd_pred):
            pair_tae_pred = error_fwd_pred
        elif not np.isnan(error_bwd_pred):
            pair_tae_pred = error_bwd_pred
        else:
            pair_tae_pred = float('nan')
            nan_count_pred += 1

        tae_errors_gt.append(pair_tae_gt)
        tae_errors_pred.append(pair_tae_pred)

    # Log statistics
    valid_gt = sum(1 for x in tae_errors_gt if not np.isnan(x))
    valid_pred = sum(1 for x in tae_errors_pred if not np.isnan(x))
    logger.debug(f"Reprojection TAE stats: {valid_gt}/{T-1} valid GT pairs, "
                 f"{valid_pred}/{T-1} valid pred pairs")

    # Warn about pure rotation (small translation) sequences
    if small_translation_count > 0:
        logger.debug(f"TAE warning: {small_translation_count}/{T-1} frame pairs have small translation (<1cm). "
                     f"TAE may be inflated due to pure rotation sensitivity.")

    # Convert to percentage (×100)
    per_frame_tae_gt = [x * 100 if not np.isnan(x) else float('nan') for x in tae_errors_gt]
    per_frame_tae_pred = [x * 100 if not np.isnan(x) else float('nan') for x in tae_errors_pred]

    # Compute per-frame TAE difference (pred - gt) = pure prediction error
    per_frame_tae_diff = []
    for pred_val, gt_val in zip(per_frame_tae_pred, per_frame_tae_gt):
        if not np.isnan(pred_val) and not np.isnan(gt_val):
            per_frame_tae_diff.append(pred_val - gt_val)
        else:
            per_frame_tae_diff.append(float('nan'))

    # Compute mean TAE (excluding NaN)
    valid_gt_vals = [x for x in per_frame_tae_gt if not np.isnan(x)]
    valid_pred_vals = [x for x in per_frame_tae_pred if not np.isnan(x)]
    valid_diff_vals = [x for x in per_frame_tae_diff if not np.isnan(x)]

    tae_gt = np.mean(valid_gt_vals) if valid_gt_vals else 0.0
    tae_pred = np.mean(valid_pred_vals) if valid_pred_vals else 0.0
    tae_diff = np.mean(valid_diff_vals) if valid_diff_vals else 0.0

    # Warn if all errors are NaN
    if len(valid_gt_vals) == 0 and len(valid_pred_vals) == 0:
        logger.warning(f"All {T-1} TAE computations returned NaN - check camera poses and depth ranges")

    return {
        'tae_reproj': tae_pred,  # TAE on predictions (includes occlusion baseline)
        'tae_reproj_gt': tae_gt,  # TAE on GT (occlusion baseline)
        'tae': tae_diff,  # Pure prediction error = pred - gt
        'per_frame_tae': per_frame_tae_diff,  # Per-frame-pair TAE difference
        'per_frame_tae_pred': per_frame_tae_pred,  # Per-frame-pair pred TAE (for reference)
        'per_frame_tae_gt': per_frame_tae_gt,  # Per-frame-pair GT TAE (for reference)
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
    # Note: waymo_seg has nested structure: waymo_seg/waymo_seg/camera_image/
    waymo_seg_root = Path(waymo_seg_root) if waymo_seg_root else Path('.')
    camera_image_path = waymo_seg_root / 'waymo_seg' / 'waymo_seg' / 'camera_image' / f'{segment_name}.parquet'
    camera_calib_path = waymo_seg_root / 'waymo_seg' / 'waymo_seg' / 'camera_calibration' / f'{segment_name}.parquet'

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
        """Check if dataset supports reprojection TAE.

        Handles both simple names (e.g., 'sintel') and path-style names
        (e.g., 'sintel/alley_1') by extracting the base dataset name.
        """
        # Extract base dataset name (first part before '/')
        base_name = dataset_name.lower().split('/')[0]
        return base_name in self.SUPPORTED_DATASETS

    def compute_tae(
        self,
        pred_depths: torch.Tensor,
        gt_depths: torch.Tensor,
        dataset_name: str,
        image_paths: List[str],
        valid_masks: Optional[List[torch.Tensor]] = None,
        max_depth: float = 70.0
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
            logger.debug(f"Dataset {dataset_name} not supported for reprojection TAE")
            return {'tae_reproj': 0.0, 'tae_reproj_gt': 0.0, 'tae_reproj_supported': False}

        try:
            # Load camera data for each frame
            intrinsics = []
            poses = []

            for idx, img_path in enumerate(image_paths):
                K, pose = self._get_camera_data(dataset_name, img_path)
                if K is None or pose is None:
                    logger.warning(f"Failed to get camera data for frame {idx}: {img_path}")
                    return {'tae_reproj': 0.0, 'tae_reproj_gt': 0.0, 'tae_reproj_supported': False}

                intrinsics.append(torch.tensor(K, dtype=torch.float32))
                poses.append(torch.tensor(pose, dtype=torch.float32))

            # Scale intrinsics to match current image resolution
            H, W = pred_depths.shape[1:]
            intrinsics = self._scale_intrinsics(intrinsics, dataset_name, image_paths[0], (H, W))

            logger.debug(f"Reprojection TAE: loaded {len(intrinsics)} cameras, depth shape {pred_depths.shape}")

            # Compute TAE
            result = compute_reprojection_tae(
                pred_depths, gt_depths,
                intrinsics, poses,
                valid_masks,
                max_depth=max_depth
            )
            result['tae_reproj_supported'] = True

            # Log if result is zero (indicates potential issue)
            if result['tae_reproj'] == 0.0 and result['tae_reproj_gt'] == 0.0:
                logger.warning(f"Reprojection TAE returned 0.0 for {dataset_name} - all frame pairs may have invalid projections")

            return result

        except Exception as e:
            logger.error(f"Error computing reprojection TAE: {e}")
            import traceback
            traceback.print_exc()
            return {'tae_reproj': 0.0, 'tae_reproj_gt': 0.0, 'tae_reproj_supported': False}

    def _get_camera_data(self, dataset_name: str, img_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get intrinsics and pose for a single frame."""
        # Extract base dataset name (first part before '/')
        # e.g., 'sintel/alley_1' -> 'sintel'
        base_name = dataset_name.lower().split('/')[0]

        if base_name == 'sintel':
            # Convert image path to camera path
            cam_path = img_path.replace('images/training/clean', 'cam_data/training/camdata_left').replace('.png', '.cam')

            if not Path(cam_path).exists():
                return None, None

            K, extrinsic = load_sintel_camera(cam_path)
            # Sintel extrinsic is world-to-camera, convert to camera-to-world
            pose = np.linalg.inv(extrinsic)
            return K, pose

        elif base_name == 'eth3d':
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

        elif base_name == 'bonn':
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

        elif base_name == 'vkitti':
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

        elif base_name == 'waymo_seg':
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
        # Extract base dataset name (first part before '/')
        base_name = dataset_name.lower().split('/')[0]

        # Get original resolution for each dataset
        # Note: These are the ORIGINAL dataset resolutions before any resizing
        if base_name == 'sintel':
            W_orig, H_orig = 1024, 436  # Original Sintel resolution
        elif base_name == 'eth3d':
            # ETH3D has variable resolution, get from intrinsics
            # Assume intrinsics are already for original resolution
            W_orig, H_orig = 6048, 4032  # Typical ETH3D resolution
        elif base_name == 'bonn':
            W_orig, H_orig = 640, 480
        elif base_name == 'vkitti':
            W_orig, H_orig = 1242, 375
        elif base_name == 'waymo_seg':
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
