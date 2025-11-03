"""
Video saving utilities for Gear testing.

This module provides unified video generation functions used across test_gear2.py,
test_gear3.py, and test_gear3_upgrade.py to reduce code duplication.
"""

import logging
from pathlib import Path
import numpy as np
from utils.helpers import torch_batch_to_np_arr, save_gifs_as_grid, save_grid_to_mp4
from .gear_common_helpers import depth_to_colored_frame

logger = logging.getLogger(__name__)


def save_video(
    images,
    pred_depths,
    gt_depths,
    valid_mask,
    sequence_id,
    save_dir,
    config,
    depth_colorizer=None
):
    """
    Create video files (GIF/MP4) similar to FlashDepth validation.

    This is a unified implementation used across all Gear test files to eliminate
    code duplication while maintaining exact functionality.

    Args:
        images: [T, 3, H, W] - RGB images (torch tensor)
        pred_depths: [T, 1, H, W] - Predicted metric depth (torch tensor)
        gt_depths: [T, 1, H, W] - GT metric depth (torch tensor)
        valid_mask: [T, 1, H, W] - Valid mask (torch tensor)
        sequence_id: int - Sequence index for file naming
        save_dir: Path - Output directory for videos
        config: Config object with eval settings (out_mp4, save_res, video_fps, gif_duration)
        depth_colorizer: Optional callable - Custom depth colorization function.
                        If None, uses default depth_to_colored_frame.
                        Signature: depth_colorizer(depth_np, valid_mask_np) -> rgb_array

    Returns:
        dict: Grid information from save_gifs_as_grid or save_grid_to_mp4
              (includes paths to saved files)

    Example:
        >>> grid = save_video(
        ...     images, pred_depths, gt_depths, valid_mask,
        ...     sequence_id=0,
        ...     save_dir=Path("results"),
        ...     config=config
        ... )
        >>> print(f"Saved to: {grid['output_path']}")
    """
    # Use default colorizer if not provided
    if depth_colorizer is None:
        depth_colorizer = depth_to_colored_frame

    T = images.shape[0]

    # Convert to numpy arrays for video creation
    video_frames = torch_batch_to_np_arr(images)  # [T, H, W, 3]

    # Batch convert to numpy (faster than per-frame conversion)
    pred_np = pred_depths.squeeze(1).cpu().numpy()  # [T, H, W]
    gt_np = gt_depths.squeeze(1).cpu().numpy()      # [T, H, W]
    valid_np = valid_mask.squeeze(1).cpu().numpy().astype(bool)  # [T, H, W]

    # Convert depth to colorized numpy arrays (per-frame normalization)
    pred_frames = []
    gt_frames = []

    for t in range(T):
        # Process pred depth (auto valid mask)
        pred_colored = depth_colorizer(pred_np[t])
        pred_frames.append(pred_colored)

        # Process GT depth (with provided valid mask)
        gt_colored = depth_colorizer(gt_np[t], valid_np[t])
        gt_frames.append(gt_colored)

    # Generate video paths
    base_name = f"sequence_{sequence_id:04d}"
    gif_path = save_dir / f"{base_name}.gif"
    mp4_path = save_dir / f"{base_name}.mp4"

    # Save based on config
    if config.eval.get('out_mp4', False):
        # Save as MP4 (with separate pred-only video)
        logger.info(f"Saving MP4 videos for sequence {sequence_id}...")
        grid = save_grid_to_mp4(
            video_frames,
            gt_frames,
            pred_frames,
            output_path=str(mp4_path),
            fixed_height=config.eval.get('save_res', 256),
            fps=config.eval.get('video_fps', 10)
        )
        logger.info(f"Saved: {mp4_path}")
        logger.info(f"Saved: {grid['pred_video_path']}")
    else:
        # Save as GIF (default)
        logger.info(f"Saving GIF for sequence {sequence_id}...")
        grid = save_gifs_as_grid(
            video_frames,
            gt_frames,
            pred_frames,
            output_path=str(gif_path),
            fixed_height=config.eval.get('save_res', 256),
            duration=config.eval.get('gif_duration', 110)
        )
        logger.info(f"Saved: {gif_path}")

    return grid
