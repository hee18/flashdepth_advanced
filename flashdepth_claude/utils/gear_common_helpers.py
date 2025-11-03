"""
Common helper functions for Gear testing and visualization.

This module provides shared utilities used across test_gear2.py, test_gear3.py,
and test_gear3_upgrade.py to reduce code duplication.
"""

import numpy as np
import matplotlib.pyplot as plt


def depth_to_colored_frame(depth, valid_mask=None):
    """
    Convert single depth frame to colored RGB using plasma_r colormap.

    This is an optimized helper that applies colormap once per frame with
    percentile-based normalization for better contrast.

    plasma_r: near=bright yellow, far=dark purple

    Args:
        depth: Numpy array of shape (H, W) containing depth values
        valid_mask: Optional boolean mask of shape (H, W) indicating valid pixels.
                   If None, uses (depth > 0) & (depth < 70) as default.

    Returns:
        Numpy array of shape (H, W, 3) with RGB values in range [0, 255]
        Invalid pixels (>70m) are set to black (0, 0, 0).

    Example:
        >>> depth_np = pred_depths.squeeze(1).cpu().numpy()  # [T, H, W]
        >>> for t in range(T):
        ...     colored = depth_to_colored_frame(depth_np[t])
    """
    # Default valid mask: positive depth and less than 70m
    MAX_DEPTH = 70.0
    if valid_mask is None:
        valid_mask = (depth > 0) & (depth < MAX_DEPTH)

    # Normalize using percentile for better contrast
    # Use NaN for invalid pixels so they map to black via set_bad
    if valid_mask.sum() > 0:
        display = np.where(valid_mask, depth, np.nan)  # Invalid = NaN
        vmin = np.nanpercentile(display, 2)
        vmax = np.nanpercentile(display, 98)
        normalized = np.clip((display - vmin) / (vmax - vmin + 1e-8), 0, 1)
    else:
        # No valid pixels, return all black
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    # Apply plasma_r colormap with set_bad for NaN pixels
    # plasma_r: near (low depth) = bright yellow, far (high depth) = dark purple
    cmap = plt.cm.plasma_r.copy()
    cmap.set_bad(color='black')  # NaN pixels = black
    colored = (cmap(normalized)[:, :, :3] * 255).astype(np.uint8)

    return colored
