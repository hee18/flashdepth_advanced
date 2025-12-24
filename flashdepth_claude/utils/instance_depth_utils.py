"""
Instance Depth Utilities

마스크 처리, depth 계산, 위치 계산을 위한 유틸리티 함수들.
YOLOv11 인스턴스 세그멘테이션과 depth 추정 모델 결합에 사용.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def get_eroded_mask_and_center(mask: np.ndarray, kernel_size: int = 15) -> Tuple[np.ndarray, int]:
    """
    마스크를 침식(erosion)하여 중심부만 남기고 centroid를 계산합니다.

    침식 처리는 마스크 경계의 노이즈를 제거하고 더 정확한 depth 추출을 가능하게 합니다.

    Args:
        mask: Binary mask (H, W) with 0/1 values
        kernel_size: Erosion kernel size (default: 15)

    Returns:
        eroded_mask: Eroded binary mask
        center_x: X coordinate of centroid
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)

    # 침식으로 마스크가 비어버리면 원본 사용
    if eroded_mask.sum() == 0:
        eroded_mask = mask.astype(np.uint8)

    # Centroid 계산
    M = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    if M['m00'] == 0:
        # Fallback: 마스크 픽셀들의 평균 위치
        ys, xs = np.where(mask == 1)
        center_x = int(np.mean(xs)) if len(xs) > 0 else 0
    else:
        center_x = int(M['m10'] / M['m00'])

    return eroded_mask, center_x


def get_circle_mask_and_center(mask: np.ndarray, radius: int = 10) -> Tuple[np.ndarray, int]:
    """
    세그먼트된 마스크의 무게중심을 중심으로 원형 마스크를 생성합니다.

    Args:
        mask: Binary mask (H, W) with 0/1 values
        radius: Circle radius in pixels (default: 10)

    Returns:
        circle_mask: Circular binary mask centered at centroid
        center_x: X coordinate of centroid
    """
    h, w = mask.shape

    # Centroid 계산
    M = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    if M['m00'] == 0:
        # Fallback: 마스크 픽셀들의 평균 위치
        ys, xs = np.where(mask == 1)
        if len(xs) > 0:
            center_x = int(np.mean(xs))
            center_y = int(np.mean(ys))
        else:
            return mask.astype(np.uint8), 0
    else:
        center_x = int(M['m10'] / M['m00'])
        center_y = int(M['m01'] / M['m00'])

    # 원형 마스크 생성
    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, (center_x, center_y), radius, 1, -1)

    # 원형 마스크와 원본 마스크의 교집합 (원이 마스크 밖으로 나가는 경우 방지)
    circle_mask = circle_mask & mask.astype(np.uint8)

    # 교집합이 비어있으면 원본 마스크 사용
    if circle_mask.sum() == 0:
        circle_mask = mask.astype(np.uint8)

    return circle_mask, center_x


def get_center_mask(mask: np.ndarray, kernel_size: int = 15, radius: int = 10) -> Tuple[np.ndarray, int]:
    """
    center-mask를 생성: 먼저 침식(erosion) 적용 후 중심원 추출.

    Args:
        mask: Binary mask (H, W) with 0/1 values
        kernel_size: Erosion kernel size (default: 15)
        radius: Circle radius in pixels (default: 10)

    Returns:
        center_mask: Center binary mask (erosion + circle)
        center_x: X coordinate of centroid
    """
    # Step 1: Apply erosion first
    eroded_mask, _ = get_eroded_mask_and_center(mask, kernel_size=kernel_size)

    # Step 2: Extract center circle from eroded mask
    center_mask, center_x = get_circle_mask_and_center(eroded_mask, radius=radius)

    return center_mask, center_x


def get_mask_center(mask: np.ndarray) -> Tuple[int, int]:
    """
    마스크의 centroid (center_x, center_y)를 계산합니다.

    Args:
        mask: Binary mask (H, W)

    Returns:
        center_x, center_y: Centroid coordinates
    """
    M = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    if M['m00'] == 0:
        ys, xs = np.where(mask == 1)
        if len(xs) > 0:
            return int(np.mean(xs)), int(np.mean(ys))
        return 0, 0

    center_x = int(M['m10'] / M['m00'])
    center_y = int(M['m01'] / M['m00'])
    return center_x, center_y


def calculate_mask_depth(mask: np.ndarray, depth_map: np.ndarray,
                          method: str = 'mean') -> float:
    """
    마스크 영역 내의 depth 값을 계산합니다.

    Args:
        mask: Binary mask (H, W)
        depth_map: Depth map (H, W) in meters
        method: Aggregation method ('mean', 'median', 'min')

    Returns:
        Depth value in meters (or 1000.0 if invalid)
    """
    ys, xs = np.where(mask == 1)
    if len(xs) == 0:
        return 1000.0

    depths = depth_map[ys, xs]

    # 유효한 depth만 필터링 (0보다 크고 1000보다 작은 값)
    valid_mask = (depths > 0) & (depths < 1000)
    valid_depths = depths[valid_mask]

    if valid_depths.size == 0:
        return 1000.0

    if method == 'mean':
        return float(np.mean(valid_depths))
    elif method == 'median':
        return float(np.median(valid_depths))
    elif method == 'min':
        return float(np.min(valid_depths))
    else:
        return float(np.mean(valid_depths))


def calculate_lateral_position(depth: float, center_x: int,
                                fx: float, cx: float) -> float:
    """
    Depth와 픽셀 좌표를 사용하여 lateral (X) position을 계산합니다.

    Pinhole camera model: x_metric = (x_px - cx) * depth / fx

    Args:
        depth: Depth in meters
        center_x: Pixel X coordinate of object center
        fx: Focal length X (pixels)
        cx: Principal point X (pixels)

    Returns:
        Lateral position in meters (positive = right, negative = left)
    """
    if depth <= 0 or depth >= 1000:
        return 0.0
    return (center_x - cx) * depth / fx


def calculate_3d_position(depth: float, center_x: int, center_y: int,
                          fx: float, fy: float, cx: float, cy: float) -> Tuple[float, float, float]:
    """
    Depth와 픽셀 좌표를 사용하여 3D position (X, Y, Z)을 계산합니다.

    Args:
        depth: Depth in meters (Z coordinate)
        center_x, center_y: Pixel coordinates of object center
        fx, fy: Focal lengths (pixels)
        cx, cy: Principal point (pixels)

    Returns:
        (X, Y, Z) position in meters
        - X: lateral (positive = right)
        - Y: vertical (positive = down)
        - Z: forward (depth)
    """
    if depth <= 0 or depth >= 1000:
        return 0.0, 0.0, depth

    x = (center_x - cx) * depth / fx
    y = (center_y - cy) * depth / fy
    z = depth

    return x, y, z


def create_mask_from_polygons(polygons: list, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Polygon points로부터 binary mask를 생성합니다.

    Args:
        polygons: List of polygon arrays (each Nx2 float array)
        image_shape: (H, W) tuple

    Returns:
        Binary mask (H, W) with 0/1 values
    """
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    for polygon in polygons:
        if len(polygon) > 0:
            polygon_int = polygon.astype(np.int32)
            cv2.fillPoly(mask, [polygon_int], 1)
    return mask


def create_mask_from_yolo_result(yolo_mask_xy: np.ndarray,
                                  frame_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    YOLO segmentation 결과의 mask.xy를 binary mask로 변환합니다.

    Args:
        yolo_mask_xy: YOLO masks.xy output (Nx2 float array)
        frame_shape: (H, W, C) tuple

    Returns:
        Binary mask (H, W) with 0/1 values
    """
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if len(yolo_mask_xy) > 0:
        polygon = yolo_mask_xy.astype(np.int32)
        cv2.fillPoly(mask, [polygon], 1)

    return mask


def apply_depth_to_mask(depth_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    마스크 영역에만 depth를 적용하고 나머지는 0으로 설정합니다.

    Args:
        depth_map: Full depth map (H, W)
        mask: Binary mask (H, W)

    Returns:
        Masked depth map (H, W)
    """
    masked_depth = np.zeros_like(depth_map)
    masked_depth[mask == 1] = depth_map[mask == 1]
    return masked_depth


def resize_depth_to_frame(depth_map: np.ndarray,
                          target_shape: Tuple[int, int],
                          interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """
    Depth map을 target frame 크기로 resize합니다.

    Args:
        depth_map: Original depth map (h, w)
        target_shape: Target (H, W)
        interpolation: OpenCV interpolation method

    Returns:
        Resized depth map (H, W)
    """
    if depth_map.shape[:2] == target_shape:
        return depth_map

    return cv2.resize(depth_map, (target_shape[1], target_shape[0]),
                      interpolation=interpolation)


def compute_instance_statistics(trajectory: list) -> dict:
    """
    Instance trajectory에 대한 통계를 계산합니다.

    Args:
        trajectory: List of trajectory points with 'depth_m' key

    Returns:
        Dictionary with min, max, avg, std depth values
    """
    if not trajectory:
        return {
            'min_depth': 0.0,
            'max_depth': 0.0,
            'avg_depth': 0.0,
            'depth_std': 0.0,
            'total_frames': 0
        }

    depths = [p['depth_m'] for p in trajectory if p['depth_m'] < 1000]

    if not depths:
        return {
            'min_depth': 0.0,
            'max_depth': 0.0,
            'avg_depth': 0.0,
            'depth_std': 0.0,
            'total_frames': len(trajectory)
        }

    return {
        'min_depth': float(np.min(depths)),
        'max_depth': float(np.max(depths)),
        'avg_depth': float(np.mean(depths)),
        'depth_std': float(np.std(depths)),
        'total_frames': len(trajectory)
    }


# NuScenes default camera intrinsics
NUSCENES_INTRINSICS = {
    'fx': 1266.4,
    'fy': 1266.4,
    'cx': 816.3,   # ~half of 1600
    'cy': 450.0    # ~half of 900
}


def get_default_intrinsics(width: int = 1600, height: int = 900) -> dict:
    """
    기본 카메라 intrinsics를 반환합니다.

    Args:
        width, height: Image dimensions

    Returns:
        Dictionary with fx, fy, cx, cy
    """
    # NuScenes 기본값 또는 이미지 중심 기반 추정
    return {
        'fx': 1266.4,
        'fy': 1266.4,
        'cx': width / 2,
        'cy': height / 2
    }
