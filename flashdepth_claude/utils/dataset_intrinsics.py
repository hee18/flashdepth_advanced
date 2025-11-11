"""
Central registry for dataset camera intrinsics.

This module provides a centralized location for managing camera intrinsic information
across all datasets used in the FlashDepth project.

Author: Claude Code (Anthropic)
Date: 2025-01-05
"""

import logging

logger = logging.getLogger(__name__)


# Dataset intrinsics registry
# Format: 'type' can be 'fixed', 'per_frame', 'per_sequence', or 'computed'
DATASET_INTRINSICS = {
    'tartanair': {
        'type': 'fixed',
        'fx': 320.0,
        'fy': 320.0,
        'cx': 320.0,
        'cy': 240.0,  # V1: 640×480, V2 uses cy=320 for 640×640
        'resolution': (640, 480),
        'fov_deg': 90.0,
        'source': 'Official TartanAir documentation (https://tartanair.org)',
        'notes': 'V1 uses 640×480, V2 uses 640×640. Both have fx=fy=320.'
    },

    'spring': {
        'type': 'per_frame',
        'file_pattern': 'cam_data/intrinsics.txt',
        'format': 'txt',
        'fields': ['fx', 'fy', 'cx', 'cy'],
        'resolution': (1920, 1080),
        'baseline_m': 0.065,
        'source': 'Provided with Spring dataset',
        'notes': 'Each line contains: fx fy cx cy (4 values per frame)'
    },

    'waymo': {
        'type': 'per_sequence',
        'file_pattern': 'camera_calibration/*.parquet',
        'format': 'parquet',
        'field_fx': 'f_u',
        'field_fy': 'f_v',
        'field_cx': 'c_u',
        'field_cy': 'c_v',
        'resolution': (1920, 1280),
        'typical_fx': 2059.0,
        'source': 'Waymo Open Dataset calibration',
        'notes': 'Also includes distortion coefficients (k1, k2, k3, p1, p2)'
    },

    'waymo_seg': {
        'type': 'per_sequence',
        'file_pattern': 'camera_calibration/*.parquet',
        'format': 'parquet',
        'field_fx': 'f_u',
        'field_fy': 'f_v',
        'field_cx': 'c_u',
        'field_cy': 'c_v',
        'resolution': (1920, 1280),
        'typical_fx': 2059.0,
        'source': 'Waymo Open Dataset calibration',
        'notes': 'Same as waymo, with semantic segmentation annotations'
    },

    'dynamicreplica': {
        'type': 'per_frame',
        'file': 'frame_annotations_train.jgz',
        'format': 'json_gzip',
        'intrinsics_format': 'ndc_isotropic',
        'field_focal_length': 'viewpoint.focal_length',
        'conversion': 'fx_pixel = fx_ndc × width / 2',
        'typical_fx_ndc': 1.9444444444444444,
        'typical_fx_pixel': 1244.44,  # For 1280×720 images
        'resolution': (720, 1280),
        'source': 'DynamicReplica frame_annotations_train.jgz',
        'notes': 'NDC (Normalized Device Coordinates) format. All frames have same focal length. fx_ndc ≈ 1.944, converts to ~1244 pixels for 1280×720 images.'
    },

    'mvs_synth': {
        'type': 'per_frame',
        'file_pattern': 'poses/*.json',
        'format': 'json',
        'field_fx': 'f_x',
        'field_fy': 'f_y',
        'field_cx': 'c_x',
        'field_cy': 'c_y',
        'resolution': (1920, 1080),
        'typical_fx': 1156.0,
        'source': 'MVS-Synth dataset pose files',
        'notes': 'Each frame has its own JSON file with intrinsics and extrinsics'
    },

    'pointodyssey': {
        'type': 'per_frame',
        'file_pattern': 'info_extracted/intrinsics.npy',
        'format': 'npy',
        'matrix_shape': '[N, 3, 3]',
        'source': 'PointOdyssey dataset pre-extracted intrinsics',
        'notes': 'NumPy array of 3×3 intrinsic matrices, one per frame'
    },

    'sintel': {
        'type': 'per_frame',
        'file_pattern': 'cam_data/training/camdata_left/*.cam',
        'format': 'binary',
        'header': 'PIEH',
        'source': 'Sintel dataset camera files',
        'notes': 'Binary format, similar to optical flow format. Needs parsing.'
    },

    'urbansyn': {
        'type': 'computed',
        'file': 'camera_metadata.json',
        'format': 'json',
        'fields': {
            'focal_length_mm': 'focalLength_mm',
            'sensor_width_mm': 'sensorWidth_mm',
            'sensor_height_mm': 'sensorHeight_mm',
            'fov_deg': 'fov_deg'
        },
        'formula': 'fx = focal_length_mm * width / sensor_width_mm',
        'source': 'UrbanSyn camera_metadata.json',
        'notes': 'Compute fx from physical camera parameters'
    },

    'eth3d': {
        'type': 'per_image',
        'file_pattern': 'dslr_calibration_undistorted/cameras.txt',
        'image_file': 'dslr_calibration_undistorted/images.txt',
        'format': 'txt',
        'camera_model': 'PINHOLE',
        'colmap_format': True,
        'typical_fx': 3409.0,  # ~3400-3411 pixels for 6048×4032 images
        'resolution': (6048, 4032),  # High-res DSLR images
        'source': 'ETH3D dataset documentation (https://www.eth3d.net/documentation)',
        'notes': 'COLMAP format. cameras.txt: CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy. images.txt maps image names to camera IDs.'
    },
}


def get_intrinsics_info(dataset_name):
    """
    Get intrinsics metadata for a dataset.

    Args:
        dataset_name (str): Name of the dataset (lowercase)

    Returns:
        dict: Dictionary with intrinsics information, or None if not found
    """
    dataset_key = dataset_name.lower().replace('-', '_').replace(' ', '_')
    info = DATASET_INTRINSICS.get(dataset_key)

    if info is None:
        logger.warning(f"No intrinsics info found for dataset: {dataset_name}")

    return info


def validate_focal_length(fx, width, dataset_name='unknown'):
    """
    Validate focal length and return fallback if invalid.

    Args:
        fx (float): Focal length in pixels
        width (int): Image width in pixels
        dataset_name (str): Dataset name for logging

    Returns:
        float: Validated focal length (or fallback if invalid)
    """
    # Check for None or invalid values
    if fx is None or fx <= 0:
        fallback_fx = width * 0.7
        logger.warning(
            f"[{dataset_name}] Invalid focal length fx={fx}, using fallback={fallback_fx:.1f} "
            f"(width={width}, ratio=0.7, FOV~60-70°)"
        )
        return fallback_fx

    # Check for reasonable range (FOV between 20° and 120°)
    # FOV = 2 * arctan(width / (2 * fx))
    # FOV 120° → fx = width * 0.29
    # FOV  20° → fx = width * 2.75
    min_fx = width * 0.25  # Ultra-wide (FOV ~120°)
    max_fx = width * 3.0   # Telephoto (FOV ~20°)

    if fx < min_fx:
        fallback_fx = width * 0.7
        logger.warning(
            f"[{dataset_name}] Focal length fx={fx:.1f} too small (FOV > 120°), "
            f"using fallback={fallback_fx:.1f}"
        )
        return fallback_fx

    if fx > max_fx:
        fallback_fx = width * 0.7
        logger.warning(
            f"[{dataset_name}] Focal length fx={fx:.1f} too large (FOV < 20°), "
            f"using fallback={fallback_fx:.1f}"
        )
        return fallback_fx

    return fx


def compute_fov_from_fx(fx, width):
    """
    Compute horizontal field of view from focal length.

    Args:
        fx (float): Focal length in pixels
        width (int): Image width in pixels

    Returns:
        float: Field of view in degrees
    """
    import math
    fov_rad = 2 * math.atan(width / (2 * fx))
    fov_deg = math.degrees(fov_rad)
    return fov_deg


def compute_fx_from_fov(fov_deg, width):
    """
    Compute focal length from field of view.

    Args:
        fov_deg (float): Field of view in degrees
        width (int): Image width in pixels

    Returns:
        float: Focal length in pixels
    """
    import math
    fov_rad = math.radians(fov_deg)
    fx = width / (2 * math.tan(fov_rad / 2))
    return fx


# Default fallback strategy
DEFAULT_FOV_DEG = 65.0
DEFAULT_WIDTH_RATIO = 0.7  # fx = width * 0.7 → FOV ~ 65°


# ============================================================================
# Canonical Space Definition
# ============================================================================
# FlashDepth uses a canonical space to normalize depth predictions across
# different cameras and datasets. This ensures consistent metric depth learning.
#
# Canonical Transform:
#   depth_canonical = depth_actual × (CANONICAL_FOCAL_LENGTH / fx_actual)
#
# For inverse depth (1/m), the ratio is REVERSED:
#   inverse_canonical = inverse_actual × (fx_actual / CANONICAL_FOCAL_LENGTH)
#
# ============================================================================

CANONICAL_FOCAL_LENGTH = 500.0  # pixels at canonical resolution
CANONICAL_RESOLUTION = (518, 518)  # height, width
ACTUAL_MAX_DEPTH = 70.0  # meters in actual space (for valid mask)


def get_fallback_fx(width):
    """
    Get fallback focal length for unknown datasets.

    Args:
        width (int): Image width in pixels

    Returns:
        float: Fallback focal length (width * 0.7)
    """
    return width * DEFAULT_WIDTH_RATIO


# Print registry summary on import (for debugging)
if __name__ == '__main__':
    print("=" * 80)
    print("Dataset Camera Intrinsics Registry")
    print("=" * 80)

    for dataset_name, info in DATASET_INTRINSICS.items():
        print(f"\n{dataset_name.upper()}:")
        print(f"  Type: {info['type']}")

        if info['type'] == 'fixed':
            print(f"  fx={info['fx']}, fy={info['fy']}")
            print(f"  cx={info['cx']}, cy={info['cy']}")
            print(f"  Resolution: {info['resolution']}")
            if 'fov_deg' in info:
                print(f"  FOV: {info['fov_deg']}°")

        elif info['type'] in ['per_frame', 'per_sequence', 'per_image']:
            print(f"  File: {info.get('file_pattern', info.get('file', 'N/A'))}")
            print(f"  Format: {info['format']}")

        elif info['type'] == 'computed':
            print(f"  Formula: {info.get('formula', 'See fields')}")

        print(f"  Source: {info['source']}")
        if 'notes' in info:
            print(f"  Notes: {info['notes']}")

    print("\n" + "=" * 80)
    print(f"Total datasets: {len(DATASET_INTRINSICS)}")
    print("=" * 80)
