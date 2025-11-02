"""
Improved visualization utilities for sparse depth data (e.g., Waymo LiDAR)

This module provides enhanced visualization methods that handle sparse depth
more effectively than standard dense visualization approaches.
"""

import numpy as np
import cv2
from scipy import ndimage
from typing import Tuple, Optional


def inpaint_sparse_depth(
    depth: np.ndarray,
    valid_mask: np.ndarray,
    method: str = 'nearest'
) -> np.ndarray:
    """
    Inpaint sparse depth map to create a dense visualization.

    IMPORTANT: Only inpaints within valid row range (rows with at least one valid pixel).
    Rows above/below valid range are left empty (for Waymo LiDAR visualization).

    Args:
        depth: [H, W] depth array (sparse)
        valid_mask: [H, W] boolean mask indicating valid depth values
        method: 'nearest' or 'telea' (cv2 inpainting methods)

    Returns:
        Dense depth array [H, W] with inpainted values (only within valid row range)
    """
    if method == 'nearest':
        # Use scipy's nearest neighbor interpolation (faster)
        # Create indices of valid points
        valid_indices = np.argwhere(valid_mask)

        if len(valid_indices) == 0:
            return depth

        # Find valid row range (rows with at least one valid pixel)
        valid_rows = np.any(valid_mask, axis=1)
        valid_row_indices = np.where(valid_rows)[0]

        if len(valid_row_indices) == 0:
            return depth

        min_valid_row = valid_row_indices.min()
        max_valid_row = valid_row_indices.max()

        # Create a coordinate grid
        h, w = depth.shape
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        # For each invalid pixel, find nearest valid pixel
        # BUT only within valid row range
        invalid_mask = ~valid_mask
        invalid_indices = np.argwhere(invalid_mask)

        if len(invalid_indices) == 0:
            return depth

        # Use distance transform for efficiency
        from scipy.ndimage import distance_transform_edt
        distances, indices = distance_transform_edt(
            invalid_mask, return_indices=True
        )

        # Create dense depth by filling invalid pixels with nearest valid value
        dense_depth = depth.copy()

        # Only inpaint within valid row range
        row_mask = (yy >= min_valid_row) & (yy <= max_valid_row)
        inpaint_mask = invalid_mask & row_mask

        dense_depth[inpaint_mask] = depth[indices[0][inpaint_mask], indices[1][inpaint_mask]]

        return dense_depth

    elif method == 'telea':
        # Use OpenCV's Telea inpainting (slower but smoother)
        # Prepare for CV2 (needs uint8 mask and float32 depth)
        depth_normalized = cv2.normalize(
            depth.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX
        )
        inpaint_mask = (~valid_mask).astype(np.uint8) * 255

        # Inpaint
        inpainted = cv2.inpaint(
            depth_normalized.astype(np.float32),
            inpaint_mask,
            inpaintRadius=5,
            flags=cv2.INPAINT_TELEA
        )

        # Denormalize back to original depth range
        valid_depth = depth[valid_mask]
        if len(valid_depth) > 0:
            depth_min, depth_max = valid_depth.min(), valid_depth.max()
            inpainted = inpainted / 255.0 * (depth_max - depth_min) + depth_min

        return inpainted

    else:
        raise ValueError(f"Unknown inpaint method: {method}")


def create_enhanced_sparse_depth_vis(
    depth: np.ndarray,
    valid_mask: np.ndarray,
    colormap: str = 'plasma',
    inpaint: bool = True,
    show_valid_overlay: bool = True,
    percentile_range: Tuple[float, float] = (2, 98)
) -> Tuple[np.ndarray, dict]:
    """
    Create enhanced visualization for sparse depth data.

    Args:
        depth: [H, W] depth array (sparse)
        valid_mask: [H, W] boolean mask indicating valid depth values
        colormap: matplotlib colormap name
        inpaint: whether to inpaint sparse regions
        show_valid_overlay: whether to overlay valid pixel markers
        percentile_range: (min, max) percentiles for normalization

    Returns:
        vis_rgb: [H, W, 3] RGB visualization (0-255 uint8)
        info: dict with statistics
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    h, w = depth.shape

    # Get valid depth statistics
    valid_depth = depth[valid_mask]
    if len(valid_depth) == 0:
        # No valid depth - return empty visualization
        return np.zeros((h, w, 3), dtype=np.uint8), {
            'valid_ratio': 0.0,
            'depth_min': 0.0,
            'depth_max': 0.0,
            'depth_mean': 0.0
        }

    valid_ratio = valid_mask.sum() / (h * w)
    depth_min = valid_depth.min()
    depth_max = valid_depth.max()
    depth_mean = valid_depth.mean()

    # Prepare depth for visualization
    if inpaint and valid_ratio < 0.8:
        # Inpaint only if sparse (< 80% valid)
        vis_depth = inpaint_sparse_depth(depth, valid_mask, method='nearest')
    else:
        vis_depth = depth.copy()

    # Normalize using percentiles of VALID depths
    vmin, vmax = np.nanpercentile(valid_depth, percentile_range)
    vis_depth_normalized = np.clip(
        (vis_depth - vmin) / (vmax - vmin + 1e-8), 0, 1
    )

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    vis_rgb = cmap(vis_depth_normalized)[:, :, :3]  # [H, W, 3] float [0, 1]

    # Overlay valid pixel markers if sparse
    if show_valid_overlay and valid_ratio < 0.5:
        # Add small bright dots at valid pixel locations
        # Downsample valid mask for cleaner visualization
        stride = max(1, int(np.sqrt(1 / valid_ratio) / 2))
        overlay = np.zeros_like(vis_rgb)
        overlay[::stride, ::stride, :] = 1.0  # White dots
        overlay_mask = valid_mask[::stride, ::stride]

        # Blend overlay
        for i in range(0, h, stride):
            for j in range(0, w, stride):
                if i < h and j < w and valid_mask[i, j]:
                    # Add small cross marker
                    y_start, y_end = max(0, i-1), min(h, i+2)
                    x_start, x_end = max(0, j-1), min(w, j+2)
                    vis_rgb[y_start:y_end, x_start:x_end, :] = 0.9 * vis_rgb[y_start:y_end, x_start:x_end, :] + 0.1

    # Convert to uint8
    vis_rgb = (vis_rgb * 255).astype(np.uint8)

    info = {
        'valid_ratio': valid_ratio,
        'depth_min': depth_min,
        'depth_max': depth_max,
        'depth_mean': depth_mean,
        'vmin': vmin,
        'vmax': vmax
    }

    return vis_rgb, info


def create_dual_sparse_depth_vis(
    depth: np.ndarray,
    valid_mask: np.ndarray,
    colormap: str = 'plasma',
    percentile_range: Tuple[float, float] = (2, 98)
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Create dual visualization: sparse (original) + dense (inpainted).

    Args:
        depth: [H, W] depth array (sparse)
        valid_mask: [H, W] boolean mask
        colormap: matplotlib colormap name
        percentile_range: normalization percentiles

    Returns:
        sparse_vis: [H, W, 3] sparse visualization (valid pixels only)
        dense_vis: [H, W, 3] inpainted dense visualization
        info: statistics dict
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    h, w = depth.shape
    valid_depth = depth[valid_mask]

    if len(valid_depth) == 0:
        empty = np.zeros((h, w, 3), dtype=np.uint8)
        return empty, empty, {'valid_ratio': 0.0}

    # Get normalization range from valid depths
    vmin, vmax = np.nanpercentile(valid_depth, percentile_range)

    # Sparse visualization (show only valid pixels, rest is gray)
    sparse_depth_vis = np.full((h, w), np.nan)
    sparse_depth_vis[valid_mask] = depth[valid_mask]
    sparse_normalized = np.clip(
        (sparse_depth_vis - vmin) / (vmax - vmin + 1e-8), 0, 1
    )

    cmap = cm.get_cmap(colormap)
    sparse_vis = np.ones((h, w, 3)) * 0.3  # Gray background
    valid_colored = cmap(sparse_normalized[valid_mask])[:, :3]
    sparse_vis[valid_mask, :] = valid_colored
    sparse_vis = (sparse_vis * 255).astype(np.uint8)

    # Dense visualization (inpainted)
    dense_depth = inpaint_sparse_depth(depth, valid_mask, method='nearest')
    dense_normalized = np.clip(
        (dense_depth - vmin) / (vmax - vmin + 1e-8), 0, 1
    )
    dense_vis = cmap(dense_normalized)[:, :, :3]
    dense_vis = (dense_vis * 255).astype(np.uint8)

    info = {
        'valid_ratio': valid_mask.sum() / (h * w),
        'depth_min': valid_depth.min(),
        'depth_max': valid_depth.max(),
        'depth_mean': valid_depth.mean(),
        'vmin': vmin,
        'vmax': vmax
    }

    return sparse_vis, dense_vis, info


def create_dual_sparse_depth_vis_with_white_bg(
    depth: np.ndarray,
    valid_mask: np.ndarray,
    colormap: str = 'plasma',
    percentile_range: Tuple[float, float] = (2, 98),
    density_threshold: int = 5,
    density_window: int = 5
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Create dual visualization for waymo_seg: sparse + dense (inpainted with white background).

    Key differences from waymo visualization:
    1. Rows outside valid row range are WHITE (not colored)
    2. Dense regions (e.g., street lights) outside valid range are still inpainted

    Args:
        depth: [H, W] depth array (sparse)
        valid_mask: [H, W] boolean mask
        colormap: matplotlib colormap name
        percentile_range: normalization percentiles
        density_threshold: minimum valid pixels in window to enable inpainting
        density_window: window size for density calculation (e.g., 5x5)

    Returns:
        sparse_vis: [H, W, 3] sparse visualization (valid pixels only, rest gray)
        dense_vis: [H, W, 3] inpainted visualization (white outside valid range)
        info: statistics dict
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from scipy.ndimage import convolve

    h, w = depth.shape
    valid_depth = depth[valid_mask]

    if len(valid_depth) == 0:
        empty = np.zeros((h, w, 3), dtype=np.uint8)
        return empty, empty, {'valid_ratio': 0.0}

    # Get normalization range from valid depths
    vmin, vmax = np.nanpercentile(valid_depth, percentile_range)

    # Sparse visualization (same as original - gray background)
    sparse_depth_vis = np.full((h, w), np.nan)
    sparse_depth_vis[valid_mask] = depth[valid_mask]
    sparse_normalized = np.clip(
        (sparse_depth_vis - vmin) / (vmax - vmin + 1e-8), 0, 1
    )

    cmap = cm.get_cmap(colormap)
    sparse_vis = np.ones((h, w, 3)) * 0.3  # Gray background
    valid_colored = cmap(sparse_normalized[valid_mask])[:, :3]
    sparse_vis[valid_mask, :] = valid_colored
    sparse_vis = (sparse_vis * 255).astype(np.uint8)

    # Dense visualization with smart inpainting
    # Step 1: Find valid row range
    valid_rows = np.any(valid_mask, axis=1)
    valid_row_indices = np.where(valid_rows)[0]

    if len(valid_row_indices) == 0:
        white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
        return sparse_vis, white_bg, {'valid_ratio': 0.0}

    min_valid_row = valid_row_indices.min()
    max_valid_row = valid_row_indices.max()

    # Step 2: Calculate local density (number of valid pixels in neighborhood)
    kernel = np.ones((density_window, density_window))
    local_density = convolve(valid_mask.astype(np.float32), kernel, mode='constant', cval=0)

    # Step 3: Create extended inpaint mask
    # - Inside valid row range: inpaint everything
    # - Outside valid row range: only inpaint if local density is high (e.g., street lights)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    inside_valid_range = (yy >= min_valid_row) & (yy <= max_valid_row)
    high_density_region = local_density >= density_threshold

    # Combine: inpaint if inside valid range OR high density
    extended_inpaint_mask = inside_valid_range | high_density_region

    # Step 4: Inpaint using nearest neighbor (only where extended_inpaint_mask is True)
    dense_depth = depth.copy()
    invalid_mask = ~valid_mask

    if np.any(invalid_mask & extended_inpaint_mask):
        from scipy.ndimage import distance_transform_edt
        distances, indices = distance_transform_edt(invalid_mask, return_indices=True)

        # Only fill pixels in extended_inpaint_mask
        inpaint_pixels = invalid_mask & extended_inpaint_mask
        dense_depth[inpaint_pixels] = depth[indices[0][inpaint_pixels], indices[1][inpaint_pixels]]

    # Step 5: Create visualization with white background
    # Initialize with white background
    dense_vis = np.ones((h, w, 3), dtype=np.float32)  # White (1.0, 1.0, 1.0)

    # Apply colormap only where we have inpainted or valid data
    colored_mask = valid_mask | extended_inpaint_mask
    if np.any(colored_mask):
        dense_normalized = np.clip(
            (dense_depth - vmin) / (vmax - vmin + 1e-8), 0, 1
        )
        colored_regions = cmap(dense_normalized[colored_mask])[:, :3]
        dense_vis[colored_mask, :] = colored_regions

    dense_vis = (dense_vis * 255).astype(np.uint8)

    info = {
        'valid_ratio': valid_mask.sum() / (h * w),
        'depth_min': valid_depth.min(),
        'depth_max': valid_depth.max(),
        'depth_mean': valid_depth.mean(),
        'vmin': vmin,
        'vmax': vmax,
        'min_valid_row': min_valid_row,
        'max_valid_row': max_valid_row
    }

    return sparse_vis, dense_vis, info


def create_sparse_depth_vis_no_inpaint(
    depth: np.ndarray,
    valid_mask: np.ndarray,
    colormap: str = 'plasma',
    percentile_range: Tuple[float, float] = (2, 98)
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Sparse depth visualization WITHOUT inpainting (just show valid pixels).
    Valid row range 밖은 흰색, 안쪽은 valid 픽셀만 표시 (invalid는 회색).

    Args:
        depth: [H, W] depth array (sparse)
        valid_mask: [H, W] boolean mask
        colormap: matplotlib colormap name
        percentile_range: normalization percentiles

    Returns:
        sparse_vis: [H, W, 3] sparse visualization (valid pixels only)
        dense_vis: [H, W, 3] same as sparse_vis but white bg outside valid range
        info: statistics dict
    """
    import matplotlib.cm as cm

    h, w = depth.shape
    valid_depth = depth[valid_mask]

    if len(valid_depth) == 0:
        empty = np.zeros((h, w, 3), dtype=np.uint8)
        return empty, empty, {'valid_ratio': 0.0}

    # Get normalization range from valid depths
    vmin, vmax = np.nanpercentile(valid_depth, percentile_range)

    # Sparse visualization (show only valid pixels, rest is gray)
    sparse_depth_vis = np.full((h, w), np.nan)
    sparse_depth_vis[valid_mask] = depth[valid_mask]
    sparse_normalized = np.clip(
        (sparse_depth_vis - vmin) / (vmax - vmin + 1e-8), 0, 1
    )

    cmap = cm.get_cmap(colormap)
    sparse_vis = np.ones((h, w, 3)) * 0.3  # Gray background
    valid_colored = cmap(sparse_normalized[valid_mask])[:, :3]
    sparse_vis[valid_mask, :] = valid_colored
    sparse_vis = (sparse_vis * 255).astype(np.uint8)

    # Dense visualization: white background with only valid pixels colored
    dense_vis = np.ones((h, w, 3), dtype=np.float32)  # White background

    # Color valid pixels only
    dense_vis[valid_mask, :] = valid_colored

    dense_vis = (dense_vis * 255).astype(np.uint8)

    # For info, calculate valid row range
    valid_rows = np.any(valid_mask, axis=1)
    valid_row_indices = np.where(valid_rows)[0]

    if len(valid_row_indices) == 0:
        min_valid_row = 0
        max_valid_row = 0
    else:
        min_valid_row = valid_row_indices.min()
        max_valid_row = valid_row_indices.max()

    info = {
        'valid_ratio': valid_mask.sum() / (h * w),
        'depth_min': valid_depth.min(),
        'depth_max': valid_depth.max(),
        'depth_mean': valid_depth.mean(),
        'vmin': vmin,
        'vmax': vmax,
        'min_valid_row': min_valid_row,
        'max_valid_row': max_valid_row
    }

    return sparse_vis, dense_vis, info


def create_dual_sparse_depth_vis_white_bg(
    depth: np.ndarray,
    valid_mask: np.ndarray,
    colormap: str = 'plasma',
    percentile_range: Tuple[float, float] = (2, 98)
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Create dual visualization with WHITE background outside valid row range.
    Same as create_dual_sparse_depth_vis but uses white instead of colormap outside valid rows.

    Args:
        depth: [H, W] depth array (sparse)
        valid_mask: [H, W] boolean mask
        colormap: matplotlib colormap name
        percentile_range: normalization percentiles

    Returns:
        sparse_vis: [H, W, 3] sparse visualization (valid pixels only)
        dense_vis: [H, W, 3] inpainted dense visualization (white outside valid range)
        info: statistics dict
    """
    import matplotlib.cm as cm

    h, w = depth.shape
    valid_depth = depth[valid_mask]

    if len(valid_depth) == 0:
        empty = np.zeros((h, w, 3), dtype=np.uint8)
        return empty, empty, {'valid_ratio': 0.0}

    # Get normalization range from valid depths
    vmin, vmax = np.nanpercentile(valid_depth, percentile_range)

    # Sparse visualization (show only valid pixels, rest is gray)
    sparse_depth_vis = np.full((h, w), np.nan)
    sparse_depth_vis[valid_mask] = depth[valid_mask]
    sparse_normalized = np.clip(
        (sparse_depth_vis - vmin) / (vmax - vmin + 1e-8), 0, 1
    )

    cmap = cm.get_cmap(colormap)
    sparse_vis = np.ones((h, w, 3)) * 0.3  # Gray background
    valid_colored = cmap(sparse_normalized[valid_mask])[:, :3]
    sparse_vis[valid_mask, :] = valid_colored
    sparse_vis = (sparse_vis * 255).astype(np.uint8)

    # Dense visualization: Use OpenCV's advanced inpainting for better smoothness
    # Step 1: Inpaint using OpenCV (better than nearest neighbor for lidar data)
    valid_depth_for_norm = depth[valid_mask]
    if len(valid_depth_for_norm) == 0:
        white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
        return sparse_vis, white_bg, {'valid_ratio': 0.0}

    # Normalize depth to 0-255 range for OpenCV
    depth_min, depth_max = valid_depth_for_norm.min(), valid_depth_for_norm.max()
    depth_normalized = np.zeros_like(depth, dtype=np.float32)
    depth_normalized[valid_mask] = (depth[valid_mask] - depth_min) / (depth_max - depth_min + 1e-8) * 255

    # Create inpaint mask (255 for invalid pixels)
    inpaint_mask_cv = (~valid_mask).astype(np.uint8) * 255

    # Use OpenCV inpainting (Navier-Stokes based, more sophisticated)
    inpainted_normalized = cv2.inpaint(
        depth_normalized.astype(np.float32),
        inpaint_mask_cv,
        inpaintRadius=3,
        flags=cv2.INPAINT_NS  # Navier-Stokes based method
    )

    # Denormalize back to original depth range
    dense_depth = inpainted_normalized / 255.0 * (depth_max - depth_min) + depth_min

    # Step 2: Find valid row range
    valid_rows = np.any(valid_mask, axis=1)
    valid_row_indices = np.where(valid_rows)[0]

    if len(valid_row_indices) == 0:
        white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
        return sparse_vis, white_bg, {'valid_ratio': 0.0}

    min_valid_row = valid_row_indices.min()
    max_valid_row = valid_row_indices.max()

    # Step 3: Normalize and apply colormap
    dense_normalized = np.clip(
        (dense_depth - vmin) / (vmax - vmin + 1e-8), 0, 1
    )
    colored_depth = cmap(dense_normalized)[:, :, :3]

    # Step 4: Start with white background, apply colormap only to valid row range
    dense_vis = np.ones((h, w, 3), dtype=np.float32)  # White

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    valid_row_mask = (yy >= min_valid_row) & (yy <= max_valid_row)

    dense_vis[valid_row_mask] = colored_depth[valid_row_mask]
    dense_vis = (dense_vis * 255).astype(np.uint8)

    info = {
        'valid_ratio': valid_mask.sum() / (h * w),
        'depth_min': valid_depth.min(),
        'depth_max': valid_depth.max(),
        'depth_mean': valid_depth.mean(),
        'vmin': vmin,
        'vmax': vmax,
        'min_valid_row': min_valid_row,
        'max_valid_row': max_valid_row
    }

    return sparse_vis, dense_vis, info
