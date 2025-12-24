"""
LiDAR Projection Utilities

nuScenes LiDAR 데이터를 카메라 프레임에 투영하고,
인스턴스 마스크 영역의 depth를 추출하는 유틸리티 함수들.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation


def load_lidar_pcd_bin(path: str) -> np.ndarray:
    """
    NuScenes LiDAR .pcd.bin 파일을 로드합니다.

    Args:
        path: .pcd.bin 파일 경로

    Returns:
        points: (N, 5) array - [x, y, z, intensity, ring_index]
    """
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    return points


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """
    Quaternion을 회전 행렬로 변환합니다.

    Args:
        quaternion: [w, x, y, z] 형식의 quaternion

    Returns:
        R: (3, 3) rotation matrix
    """
    # scipy는 [x, y, z, w] 형식을 사용
    q = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    R = Rotation.from_quat(q).as_matrix()
    return R


def project_lidar_to_camera(
    lidar_points: np.ndarray,
    lidar_translation: np.ndarray,
    lidar_rotation: np.ndarray,
    cam_translation: np.ndarray,
    cam_rotation: np.ndarray,
    cam_intrinsics: np.ndarray,
    img_size: Tuple[int, int],
    ego_pose_lidar: Optional[dict] = None,
    ego_pose_cam: Optional[dict] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    LiDAR 포인트를 카메라 이미지 좌표로 투영합니다.

    nuScenes 투영 파이프라인:
    1. LiDAR sensor → Ego vehicle (lidar calibration)
    2. Ego vehicle → Global (lidar ego_pose)
    3. Global → Ego vehicle (camera ego_pose inverse)
    4. Ego vehicle → Camera sensor (camera calibration inverse)
    5. Camera sensor → Image plane (camera intrinsics)

    Args:
        lidar_points: (N, 3 or 5) LiDAR 포인트 [x, y, z, ...]
        lidar_translation: LiDAR sensor → ego translation (3,)
        lidar_rotation: LiDAR sensor → ego rotation quaternion [w, x, y, z]
        cam_translation: Camera sensor → ego translation (3,)
        cam_rotation: Camera sensor → ego rotation quaternion [w, x, y, z]
        cam_intrinsics: (3, 3) camera intrinsic matrix
        img_size: (width, height)
        ego_pose_lidar: Optional dict with 'translation' and 'rotation' for LiDAR ego pose
        ego_pose_cam: Optional dict with 'translation' and 'rotation' for camera ego pose

    Returns:
        uv_coords: (M, 2) valid projected pixel coordinates [u, v]
        depths: (M,) corresponding depths in meters
        point_indices: (M,) indices of valid points in original array
    """
    # 3D points만 추출 (intensity, ring_index 제외)
    points = lidar_points[:, :3].copy()

    # 1. LiDAR sensor → Ego vehicle
    R_lidar = quaternion_to_rotation_matrix(lidar_rotation)
    points = (R_lidar @ points.T).T + lidar_translation

    # 2. Ego vehicle → Global (if ego_pose provided)
    if ego_pose_lidar is not None:
        R_ego_lidar = quaternion_to_rotation_matrix(np.array(ego_pose_lidar['rotation']))
        points = (R_ego_lidar @ points.T).T + np.array(ego_pose_lidar['translation'])

    # 3. Global → Camera Ego vehicle (if ego_pose provided)
    if ego_pose_cam is not None:
        R_ego_cam = quaternion_to_rotation_matrix(np.array(ego_pose_cam['rotation']))
        t_ego_cam = np.array(ego_pose_cam['translation'])
        # Inverse transform
        points = (R_ego_cam.T @ (points - t_ego_cam).T).T

    # 4. Ego vehicle → Camera sensor (inverse of camera calibration)
    R_cam = quaternion_to_rotation_matrix(cam_rotation)
    # Inverse transform
    points = (R_cam.T @ (points - cam_translation).T).T

    # 5. Filter points behind camera (Z <= 0)
    valid_mask = points[:, 2] > 0
    valid_indices = np.where(valid_mask)[0]
    points = points[valid_mask]

    if len(points) == 0:
        return np.array([]).reshape(0, 2), np.array([]), np.array([])

    # 6. Project to image plane
    depths = points[:, 2].copy()

    # Homogeneous projection
    points_homo = points / points[:, 2:3]  # Normalize by Z
    uv_homo = (cam_intrinsics @ points_homo.T).T  # (N, 3)
    uv_coords = uv_homo[:, :2]  # (N, 2)

    # 7. Filter points outside image bounds
    width, height = img_size
    in_bounds = (
        (uv_coords[:, 0] >= 0) & (uv_coords[:, 0] < width) &
        (uv_coords[:, 1] >= 0) & (uv_coords[:, 1] < height)
    )

    return uv_coords[in_bounds], depths[in_bounds], valid_indices[in_bounds]


def project_lidar_simple(
    lidar_points: np.ndarray,
    lidar_to_cam_transform: np.ndarray,
    cam_intrinsics: np.ndarray,
    img_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    단순화된 LiDAR→Camera 투영 (미리 계산된 transform 사용).

    Args:
        lidar_points: (N, 3 or 5) LiDAR points
        lidar_to_cam_transform: (4, 4) transformation matrix from LiDAR to camera
        cam_intrinsics: (3, 3) camera intrinsic matrix
        img_size: (width, height)

    Returns:
        uv_coords: (M, 2) pixel coordinates
        depths: (M,) depths in meters
        point_indices: (M,) indices of valid points
    """
    points = lidar_points[:, :3].copy()

    # Apply transformation
    points_homo = np.hstack([points, np.ones((len(points), 1))])  # (N, 4)
    points_cam = (lidar_to_cam_transform @ points_homo.T).T[:, :3]  # (N, 3)

    # Filter points behind camera
    valid_mask = points_cam[:, 2] > 0
    valid_indices = np.where(valid_mask)[0]
    points_cam = points_cam[valid_mask]

    if len(points_cam) == 0:
        return np.array([]).reshape(0, 2), np.array([]), np.array([])

    depths = points_cam[:, 2].copy()

    # Project to image
    points_norm = points_cam / points_cam[:, 2:3]
    uv = (cam_intrinsics @ points_norm.T).T[:, :2]

    # Filter out of bounds
    width, height = img_size
    in_bounds = (
        (uv[:, 0] >= 0) & (uv[:, 0] < width) &
        (uv[:, 1] >= 0) & (uv[:, 1] < height)
    )

    return uv[in_bounds], depths[in_bounds], valid_indices[in_bounds]


def extract_depth_from_lidar_in_mask(
    mask: np.ndarray,
    uv_coords: np.ndarray,
    depths: np.ndarray,
    k_nearest: int = 5
) -> Tuple[float, int]:
    """
    마스크 영역 내의 LiDAR depth를 추출합니다.

    Args:
        mask: (H, W) binary mask
        uv_coords: (N, 2) projected LiDAR pixel coordinates
        depths: (N,) corresponding depths
        k_nearest: fallback K-nearest points if mask is empty

    Returns:
        depth: mean depth in meters (or -1 if invalid)
        num_points: number of LiDAR points used
    """
    if len(uv_coords) == 0:
        return -1.0, 0

    # 마스크 내 LiDAR 포인트 찾기
    u_int = uv_coords[:, 0].astype(int)
    v_int = uv_coords[:, 1].astype(int)

    # 범위 체크
    h, w = mask.shape
    valid = (u_int >= 0) & (u_int < w) & (v_int >= 0) & (v_int < h)
    u_int = u_int[valid]
    v_int = v_int[valid]
    valid_depths = depths[valid]

    if len(u_int) == 0:
        return -1.0, 0

    # 마스크 내 포인트 선택
    in_mask = mask[v_int, u_int] == 1
    mask_depths = valid_depths[in_mask]

    if len(mask_depths) > 0:
        return float(np.mean(mask_depths)), len(mask_depths)

    # Fallback: 마스크 내 포인트가 없으면 -1 반환 (K-nearest는 center 기준으로 별도 처리)
    return -1.0, 0


def extract_depth_from_lidar_with_knearest(
    center_mask: np.ndarray,
    full_mask: np.ndarray,
    uv_coords: np.ndarray,
    depths: np.ndarray,
    center_x: int,
    center_y: int,
    k_nearest: int = 5
) -> Tuple[float, float, int]:
    """
    Center mask에서 LiDAR depth를 추출하고, 없으면 K-nearest fallback 사용.

    Args:
        center_mask: (H, W) center region mask (erosion + circle)
        full_mask: (H, W) full instance mask
        uv_coords: (N, 2) projected LiDAR pixel coordinates
        depths: (N,) corresponding depths
        center_x, center_y: center point coordinates
        k_nearest: K for K-nearest fallback

    Returns:
        depth_m: depth in meters
        lateral_m: lateral position in meters (from LiDAR X)
        num_points: number of LiDAR points used
    """
    if len(uv_coords) == 0:
        return -1.0, 0.0, 0

    h, w = center_mask.shape
    u_int = uv_coords[:, 0].astype(int)
    v_int = uv_coords[:, 1].astype(int)

    # 범위 체크
    valid = (u_int >= 0) & (u_int < w) & (v_int >= 0) & (v_int < h)
    u_valid = u_int[valid]
    v_valid = v_int[valid]
    depths_valid = depths[valid]
    uv_valid = uv_coords[valid]

    if len(u_valid) == 0:
        return -1.0, 0.0, 0

    # 1. 먼저 center mask에서 시도
    in_center = center_mask[v_valid, u_valid] == 1
    if np.sum(in_center) > 0:
        center_depths = depths_valid[in_center]
        center_uvs = uv_valid[in_center]
        depth_m = float(np.mean(center_depths))
        # Lateral은 center_x 기준으로 계산 (여기서는 간단히 평균 u 사용)
        mean_u = np.mean(center_uvs[:, 0])
        return depth_m, mean_u, int(np.sum(in_center))

    # 2. Center mask에 포인트가 없으면 full mask에서 K-nearest
    in_full = full_mask[v_valid, u_valid] == 1
    if np.sum(in_full) == 0:
        return -1.0, 0.0, 0

    full_uvs = uv_valid[in_full]
    full_depths = depths_valid[in_full]

    # Center로부터의 거리 계산
    distances = np.sqrt((full_uvs[:, 0] - center_x)**2 + (full_uvs[:, 1] - center_y)**2)

    # K-nearest 선택
    k = min(k_nearest, len(distances))
    nearest_indices = np.argsort(distances)[:k]

    nearest_depths = full_depths[nearest_indices]
    nearest_uvs = full_uvs[nearest_indices]

    depth_m = float(np.mean(nearest_depths))
    mean_u = float(np.mean(nearest_uvs[:, 0]))

    return depth_m, mean_u, k


def calculate_lateral_from_lidar(
    depth_m: float,
    mean_u: float,
    fx: float,
    cx: float
) -> float:
    """
    LiDAR depth와 평균 u 좌표로 lateral position 계산.

    Args:
        depth_m: depth in meters
        mean_u: mean u coordinate of LiDAR points
        fx: focal length x
        cx: principal point x

    Returns:
        lateral_m: lateral position in meters
    """
    if depth_m <= 0:
        return 0.0
    return (mean_u - cx) * depth_m / fx


def compute_lidar_to_camera_transform(
    lidar_translation: np.ndarray,
    lidar_rotation: np.ndarray,
    cam_translation: np.ndarray,
    cam_rotation: np.ndarray,
    ego_pose_lidar: Optional[dict] = None,
    ego_pose_cam: Optional[dict] = None
) -> np.ndarray:
    """
    LiDAR에서 Camera로의 전체 변환 행렬을 계산합니다.

    Args:
        lidar_translation: LiDAR calibration translation
        lidar_rotation: LiDAR calibration rotation quaternion [w, x, y, z]
        cam_translation: Camera calibration translation
        cam_rotation: Camera calibration rotation quaternion [w, x, y, z]
        ego_pose_lidar: Optional ego pose for LiDAR timestamp
        ego_pose_cam: Optional ego pose for camera timestamp

    Returns:
        T: (4, 4) transformation matrix from LiDAR to camera coordinates
    """
    # Build transformation matrices
    R_lidar = quaternion_to_rotation_matrix(lidar_rotation)
    T_lidar = np.eye(4)
    T_lidar[:3, :3] = R_lidar
    T_lidar[:3, 3] = lidar_translation

    R_cam = quaternion_to_rotation_matrix(cam_rotation)
    T_cam = np.eye(4)
    T_cam[:3, :3] = R_cam
    T_cam[:3, 3] = cam_translation
    T_cam_inv = np.linalg.inv(T_cam)

    if ego_pose_lidar is not None and ego_pose_cam is not None:
        R_ego_lidar = quaternion_to_rotation_matrix(np.array(ego_pose_lidar['rotation']))
        T_ego_lidar = np.eye(4)
        T_ego_lidar[:3, :3] = R_ego_lidar
        T_ego_lidar[:3, 3] = np.array(ego_pose_lidar['translation'])

        R_ego_cam = quaternion_to_rotation_matrix(np.array(ego_pose_cam['rotation']))
        T_ego_cam = np.eye(4)
        T_ego_cam[:3, :3] = R_ego_cam
        T_ego_cam[:3, 3] = np.array(ego_pose_cam['translation'])
        T_ego_cam_inv = np.linalg.inv(T_ego_cam)

        # Full pipeline: LiDAR → ego → global → cam_ego → cam
        T = T_cam_inv @ T_ego_cam_inv @ T_ego_lidar @ T_lidar
    else:
        # Same ego pose (synchronized)
        T = T_cam_inv @ T_lidar

    return T


def create_sparse_depth_map(
    uv_coords: np.ndarray,
    depths: np.ndarray,
    img_size: Tuple[int, int]
) -> np.ndarray:
    """
    Sparse LiDAR depth map을 생성합니다.

    Args:
        uv_coords: (N, 2) pixel coordinates
        depths: (N,) depths
        img_size: (width, height)

    Returns:
        depth_map: (H, W) sparse depth map (0 where no LiDAR)
    """
    width, height = img_size
    depth_map = np.zeros((height, width), dtype=np.float32)

    if len(uv_coords) == 0:
        return depth_map

    u_int = uv_coords[:, 0].astype(int)
    v_int = uv_coords[:, 1].astype(int)

    valid = (u_int >= 0) & (u_int < width) & (v_int >= 0) & (v_int < height)
    depth_map[v_int[valid], u_int[valid]] = depths[valid]

    return depth_map
