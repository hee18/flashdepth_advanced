"""
Mask operations for segmentation-based depth extraction.
"""

import cv2
import numpy as np


def create_mask_from_polygon(polygon_points, image_shape):
    """
    Create a binary mask from polygon points.

    Args:
        polygon_points: [N, 2] float array of (x, y) polygon vertices
        image_shape: (H, W) tuple

    Returns:
        mask: [H, W] uint8 binary mask (0 or 1)
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    pts = polygon_points.astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def erode_mask(mask, kernel_size=5, iterations=1):
    """
    Erode a binary mask to shrink it slightly (removes noisy edges).

    Args:
        mask: [H, W] uint8 binary mask
        kernel_size: erosion kernel size
        iterations: number of erosion iterations

    Returns:
        eroded: [H, W] uint8 binary mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.erode(mask, kernel, iterations=iterations)


def get_mask_center(mask):
    """
    Get the centroid of a binary mask using image moments.

    Args:
        mask: [H, W] uint8 binary mask

    Returns:
        (center_x, center_y) or None if mask is empty
    """
    M = cv2.moments(mask, binaryImage=True)
    if M['m00'] == 0:
        return None
    center_x = int(M['m10'] / M['m00'])
    center_y = int(M['m01'] / M['m00'])
    return center_x, center_y


def extract_depth_from_mask(depth_map, mask):
    """
    Extract mean depth value within the masked region.

    Args:
        depth_map: [H, W] depth values
        mask: [H, W] uint8 binary mask (0 or 1)

    Returns:
        mean_depth: float, or None if no valid pixels
    """
    valid = (mask == 1) & (depth_map > 0)
    if not np.any(valid):
        return None
    return float(np.mean(depth_map[valid]))
