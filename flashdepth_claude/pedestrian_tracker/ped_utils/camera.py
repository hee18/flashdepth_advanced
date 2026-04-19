"""
Camera utilities for pixel-to-metric coordinate conversion.
"""

import numpy as np


def calculate_lateral_position(depth, center_x, fx, cx):
    """
    Convert pixel x-coordinate to metric lateral position.

    Using pinhole camera model: X = (u - cx) * Z / fx

    Args:
        depth: metric depth in meters
        center_x: pixel x-coordinate of object center
        fx: focal length in pixels (x-axis)
        cx: principal point x-coordinate in pixels

    Returns:
        lateral_pos: lateral position in meters (positive = right)
    """
    return (center_x - cx) * depth / fx


def calculate_3d_position(depth, center_x, center_y, fx, fy, cx, cy):
    """
    Convert pixel coordinates + depth to 3D position (X, Y, Z).

    Args:
        depth: metric depth in meters
        center_x, center_y: pixel coordinates
        fx, fy: focal lengths
        cx, cy: principal point

    Returns:
        (X, Y, Z) tuple in meters
    """
    X = (center_x - cx) * depth / fx
    Y = (center_y - cy) * depth / fy
    Z = depth
    return X, Y, Z
