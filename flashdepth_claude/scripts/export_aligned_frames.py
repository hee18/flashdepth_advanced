#!/usr/bin/env python3
"""
Export aligned depth frames for original FlashDepth predictions.
Similar to test_gear5's frame export functionality.

Usage:
    python scripts/export_aligned_frames.py \
        --pred-dir test_results/original/hybrid/flash_unreal4k/unreal4k/UnrealStereo4K_00000 \
        --gt-dir /home/cvlab/hsy/Datasets/unreal4k/UnrealStereo4K_00000 \
        --frame 83 \
        --output-dir test_results/original/hybrid/flash_unreal4k/unreal4k/UnrealStereo4K_00000/figures
"""

import argparse
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_prediction(pred_dir: Path, frame_idx: int) -> Optional[np.ndarray]:
    """
    Load prediction depth from npy file.

    FlashDepth outputs inverse depth * 100: pred = 100 / depth
    We convert to depth: depth = 100 / pred
    """
    npy_path = pred_dir / "depth_npy_files" / f"frame_{frame_idx}.npy"
    if npy_path.exists():
        pred_raw = np.load(npy_path)
        # FlashDepth outputs: pred = 100 * (1/depth) = 100/depth
        # Convert to depth: depth = 100 / pred
        depth = 100.0 / (pred_raw + 1e-8)
        return depth
    return None


def load_gt_depth(gt_dir: Path, frame_idx: int) -> Optional[np.ndarray]:
    """
    Load GT depth from UnrealStereo4K disparity.

    UnrealStereo4K stores DISPARITY maps in .npy files.
    Convert disparity to metric depth using:
        depth (m) = (baseline × focal_length) / disparity

    Baselines:
    - Indoor scenes (seq 4, 6): 0.2m
    - Outdoor scenes (seq 0, 1, 2, 3, 5, 7, 8): 0.5m

    Focal length (downsampled resolution 2112×1188): fx = 1056

    Args:
        gt_dir: Path to sequence directory (e.g., .../UnrealStereo4K_00000)
        frame_idx: Frame index
    """
    disp_path = gt_dir / "Disp0" / f"{frame_idx:05d}.npy"
    if not disp_path.exists():
        return None

    disp = np.load(disp_path)

    # Extract sequence ID from path
    seq_name = gt_dir.name  # e.g., "UnrealStereo4K_00000"
    seq_id = int(seq_name.split('_')[1][-1])  # Get last digit

    # Determine baseline based on sequence ID
    INDOOR_SEQS = [4, 6]
    baseline = 0.2 if seq_id in INDOOR_SEQS else 0.5
    fx = 1056.0  # Focal length for downsampled resolution

    # Convert disparity to depth
    valid = disp > 0
    depth = np.zeros_like(disp)
    depth[valid] = (baseline * fx) / disp[valid]

    # Handle invalid values
    MAX_VALID_DEPTH = 1000.0
    invalid_mask = (depth <= 0) | (depth > MAX_VALID_DEPTH) | np.isinf(depth) | np.isnan(depth)
    depth[invalid_mask] = 0

    return depth


def load_image(gt_dir: Path, frame_idx: int) -> Optional[np.ndarray]:
    """Load original image."""
    img_path = gt_dir / "Image0" / f"{frame_idx:05d}.png"
    if img_path.exists():
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    return None


def align_in_disparity_space(pred: np.ndarray, gt: np.ndarray, max_depth: float = 70.0) -> Tuple[np.ndarray, float, float]:
    """
    Align predicted depth to GT using least squares in DISPARITY space.

    Same convention as FlashDepth eval:
    - Convert both to disparity (1/depth)
    - Align: gt_disp = s * pred_disp + t
    - Convert back to depth

    Args:
        pred: Predicted depth [T, H, W] (meters or inverse depth)
        gt: Ground truth depth [T, H, W] (meters)
        max_depth: Maximum valid depth threshold

    Returns:
        aligned_pred: Aligned prediction in depth space (meters)
        s: Scale factor (in disparity space)
        t: Shift factor (in disparity space)
    """
    # Create valid mask
    valid_mask = (gt > 0) & (gt < max_depth) & (pred > 0) & (pred < max_depth) & np.isfinite(pred) & np.isfinite(gt)

    if valid_mask.sum() < 100:
        logger.warning(f"Too few valid pixels ({valid_mask.sum()}), returning unaligned")
        return pred, 1.0, 0.0

    # Convert to disparity (inverse depth) for alignment
    pred_disp = 1.0 / pred
    gt_disp = 1.0 / gt

    pred_disp_valid = pred_disp[valid_mask].reshape(-1, 1)
    gt_disp_valid = gt_disp[valid_mask].reshape(-1, 1)

    # Solve least squares in disparity space: gt_disp = s * pred_disp + t
    A = np.hstack([pred_disp_valid, np.ones_like(pred_disp_valid)])
    result, _, _, _ = np.linalg.lstsq(A, gt_disp_valid, rcond=None)
    s, t = result.flatten()

    logger.info(f"Disparity space alignment: s={s:.4f}, t={t:.4f}")

    # Apply alignment in disparity space
    aligned_disp = s * pred_disp + t

    # Convert back to depth, handling edge cases
    aligned_disp = np.clip(aligned_disp, 1e-8, None)  # Prevent division by zero
    aligned_pred = 1.0 / aligned_disp

    # Clamp to valid range
    aligned_pred = np.clip(aligned_pred, 0, max_depth)

    return aligned_pred, float(s), float(t)


def depth_to_colormap(depth: np.ndarray, vmin: float, vmax: float, invalid_mask: np.ndarray = None) -> np.ndarray:
    """
    Convert depth to colormap visualization.

    Args:
        depth: Depth array [H, W]
        vmin: Minimum depth for colormap
        vmax: Maximum depth for colormap
        invalid_mask: If provided, set these pixels to black

    Returns:
        RGB image [H, W, 3] as uint8
    """
    # Normalize to [0, 1]
    depth_norm = (depth - vmin) / (vmax - vmin + 1e-8)
    depth_norm = np.clip(depth_norm, 0, 1)

    # Apply colormap (plasma_r - closer is brighter/yellow, farther is darker/purple)
    cmap = plt.cm.plasma_r
    colored = cmap(depth_norm)[:, :, :3]  # Remove alpha
    colored = (colored * 255).astype(np.uint8)

    # Set invalid pixels to black
    if invalid_mask is not None:
        colored[invalid_mask] = 0

    return colored


def main():
    parser = argparse.ArgumentParser(description='Export aligned depth frames')
    parser.add_argument('--pred-dir', type=str, required=True,
                        help='Directory containing depth_npy_files/')
    parser.add_argument('--gt-dir', type=str, required=True,
                        help='GT directory (e.g., .../UnrealStereo4K_00000)')
    parser.add_argument('--frame', type=int, required=True,
                        help='Center frame index')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: pred-dir/figures)')
    parser.add_argument('--max-depth', type=float, default=70.0,
                        help='Maximum valid depth in meters')

    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    output_dir = Path(args.output_dir) if args.output_dir else pred_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    center_frame = args.frame
    max_depth = args.max_depth

    logger.info(f"Pred dir: {pred_dir}")
    logger.info(f"GT dir: {gt_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Center frame: {center_frame}")

    # Determine baseline/fx for this sequence
    seq_name = gt_dir.name
    seq_id = int(seq_name.split('_')[1][-1])
    INDOOR_SEQS = [4, 6]
    baseline = 0.2 if seq_id in INDOOR_SEQS else 0.5
    fx = 1056.0
    logger.info(f"Sequence {seq_id}: baseline={baseline}m, fx={fx} ({'indoor' if seq_id in INDOOR_SEQS else 'outdoor'})")

    # Find total number of frames
    npy_files = list((pred_dir / "depth_npy_files").glob("frame_*.npy"))
    total_frames = len(npy_files)
    logger.info(f"Total prediction frames: {total_frames}")

    # Load ALL predictions and GTs for global scale/shift alignment
    logger.info("Loading all frames for global scale/shift computation...")
    all_preds = []
    all_gts = []

    for i in range(total_frames):
        pred = load_prediction(pred_dir, i)
        gt = load_gt_depth(gt_dir, i)

        if pred is not None and gt is not None:
            # Resize GT to match prediction if needed
            if pred.shape != gt.shape:
                gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
            all_preds.append(pred)
            all_gts.append(gt)

    all_preds = np.stack(all_preds, axis=0)  # [T, H, W]
    all_gts = np.stack(all_gts, axis=0)  # [T, H, W]
    logger.info(f"Loaded {len(all_preds)} frames, shape: {all_preds.shape}")

    # Compute global scale and shift in DISPARITY space (like FlashDepth eval)
    logger.info("Computing global scale/shift alignment in disparity space...")
    aligned_preds, scale, shift = align_in_disparity_space(all_preds, all_gts, max_depth)
    logger.info(f"Disparity alignment: s={scale:.4f}, t={shift:.4f}")

    # Determine frame range (center ±4)
    start_frame = max(0, center_frame - 4)
    end_frame = min(total_frames - 1, center_frame + 4)
    frame_indices = list(range(start_frame, end_frame + 1))
    logger.info(f"Exporting frames: {frame_indices}")

    # Export each frame with per-frame colormap normalization
    for t in frame_indices:
        # Compute per-frame vmin/vmax from this frame's GT
        gt_depth = all_gts[t]
        valid_gt = gt_depth[(gt_depth > 0) & (gt_depth < max_depth)]
        if len(valid_gt) > 0:
            vmin = np.percentile(valid_gt, 2)
            vmax = np.percentile(valid_gt, 98)
        else:
            vmin, vmax = 0, max_depth
        logger.info(f"Processing frame {t}: vmin={vmin:.2f}, vmax={vmax:.2f}")

        # Load original image
        image = load_image(gt_dir, t)
        if image is not None:
            # Resize to match prediction if needed
            if image.shape[:2] != all_preds[t].shape:
                image = cv2.resize(image, (all_preds[t].shape[1], all_preds[t].shape[0]))
            img_path = output_dir / f"frame_{t:04d}_image.png"
            cv2.imwrite(str(img_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # GT depth colormap (per-frame normalization)
        gt_invalid = (gt_depth <= 0) | (gt_depth >= max_depth)
        gt_colored = depth_to_colormap(gt_depth, vmin, vmax, invalid_mask=gt_invalid)
        gt_path = output_dir / f"frame_{t:04d}_gt_depth.png"
        cv2.imwrite(str(gt_path), cv2.cvtColor(gt_colored, cv2.COLOR_RGB2BGR))

        # Aligned prediction colormap (same per-frame normalization as GT)
        aligned_pred = aligned_preds[t]
        pred_invalid = (aligned_pred <= 0) | (aligned_pred >= max_depth)
        pred_colored = depth_to_colormap(aligned_pred, vmin, vmax, invalid_mask=pred_invalid)
        pred_path = output_dir / f"frame_{t:04d}_pred_aligned.png"
        cv2.imwrite(str(pred_path), cv2.cvtColor(pred_colored, cv2.COLOR_RGB2BGR))

    logger.info(f"Exported {len(frame_indices)} frames to {output_dir}")
    logger.info(f"Disparity Scale: {scale:.4f}, Shift: {shift:.4f}")

    # Save scale/shift info
    info_path = output_dir / "alignment_info.txt"
    with open(info_path, 'w') as f:
        f.write(f"Alignment Method: Disparity (inverse depth) space\n")
        f.write(f"Scale (s): {scale:.6f}\n")
        f.write(f"Shift (t): {shift:.6f}\n")
        f.write(f"Formula: gt_disp = s * pred_disp + t, then depth = 1/disp\n")
        f.write(f"Colormap: Per-frame GT 2%-98% percentile (plasma_r)\n")
        f.write(f"Center frame: {center_frame}\n")
        f.write(f"Frame range: {start_frame}-{end_frame}\n")
        f.write(f"Total frames used for alignment: {total_frames}\n")

    logger.info(f"Done! Check {output_dir}")


if __name__ == "__main__":
    main()
