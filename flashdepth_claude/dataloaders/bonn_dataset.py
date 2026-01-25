"""
Bonn RGB-D Dynamic Dataset loader for test_gear5.

Dataset info:
- 26 sequences (rgbd_bonn_balloon, rgbd_bonn_crowd, etc.)
- Resolution: 640×480
- Depth: uint16 PNG in millimeters
- Camera intrinsics: fx=542.82, fy=542.58, cx=315.59, cy=237.76 (fixed)
- File structure: rgb/, depth/, rgb.txt, depth.txt per sequence
"""

import os
import cv2
import torch
import numpy as np
import logging
from .base_dataset_pairs import BaseDatasetPairs


class BonnDepth(BaseDatasetPairs):
    """Bonn RGB-D Dynamic Dataset for test_gear5 (inverse depth processing)."""

    # Fixed camera intrinsics for Bonn dataset
    FX = 542.822841
    FY = 542.576870
    CX = 315.593520
    CY = 237.756098
    ORIGINAL_WIDTH = 640
    ORIGINAL_HEIGHT = 480

    def __init__(self, root_dir, split, load_cache=None):
        self.root_dir = os.path.join(root_dir, 'bonn')
        super().__init__(dataset_name='bonn', root_dir=self.root_dir, split=split, load_cache=load_cache)
        # Set default parameters
        self.reshape_list['resolution'] = (640, 480)
        self.reshape_list['stride'] = 1

        # Cache for timestamp matching
        self._timestamp_cache = {}

    def get_cache_path(self, cache_dir):
        return os.path.join(cache_dir, 'bonn_pairs.pkl')

    def get_all_scenes(self, scenes_path):
        """Get all Bonn sequence directories."""
        scene_names = []
        # Exclude static scenes (no camera motion, not suitable for temporal evaluation)
        excluded_scenes = ['rgbd_bonn_static']
        for s in os.listdir(scenes_path):
            scene_path = os.path.join(scenes_path, s)
            if os.path.isdir(scene_path) and s.startswith('rgbd_bonn_') and s not in excluded_scenes:
                # Check if rgb and depth directories exist
                if os.path.isdir(os.path.join(scene_path, 'rgb')) and \
                   os.path.isdir(os.path.join(scene_path, 'depth')):
                    scene_names.append(s)
        return sorted(scene_names)

    def get_filter_scenes(self, split):
        """No scenes to filter - use all 26 sequences."""
        return []

    def get_rgb_depth_paths(self, scenes_path, scene_name):
        """Return paths to rgb and depth directories."""
        scene_path = os.path.join(scenes_path, scene_name)
        return (os.path.join(scene_path, 'rgb'),
                os.path.join(scene_path, 'depth'))

    def _parse_timestamp_file(self, txt_path):
        """Parse rgb.txt or depth.txt file and return list of (timestamp, filename) tuples."""
        entries = []
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    timestamp = float(parts[0])
                    filename = parts[1]
                    entries.append((timestamp, filename))
        return sorted(entries, key=lambda x: x[0])

    def _match_rgb_depth_timestamps(self, scene_path):
        """Match RGB and depth frames based on closest timestamps."""
        rgb_txt = os.path.join(scene_path, 'rgb.txt')
        depth_txt = os.path.join(scene_path, 'depth.txt')

        if not os.path.exists(rgb_txt) or not os.path.exists(depth_txt):
            return []

        rgb_entries = self._parse_timestamp_file(rgb_txt)
        depth_entries = self._parse_timestamp_file(depth_txt)

        if not rgb_entries or not depth_entries:
            return []

        # For each RGB frame, find closest depth frame
        matched_pairs = []
        depth_idx = 0

        for rgb_ts, rgb_file in rgb_entries:
            # Find closest depth timestamp
            best_depth_idx = depth_idx
            best_diff = abs(depth_entries[depth_idx][0] - rgb_ts)

            # Search forward
            while depth_idx < len(depth_entries) - 1:
                diff = abs(depth_entries[depth_idx + 1][0] - rgb_ts)
                if diff < best_diff:
                    best_diff = diff
                    best_depth_idx = depth_idx + 1
                    depth_idx += 1
                else:
                    break

            # Only match if time difference is reasonable (< 50ms)
            if best_diff < 0.05:
                matched_pairs.append({
                    'rgb_file': os.path.basename(rgb_file),
                    'depth_file': os.path.basename(depth_entries[best_depth_idx][1]),
                    'timestamp': rgb_ts
                })

        return matched_pairs

    def get_sorted_image_files(self, rgb_path):
        """Get sorted list of RGB image files based on timestamp matching."""
        scene_path = os.path.dirname(rgb_path)

        # Use cached matching if available
        if scene_path not in self._timestamp_cache:
            self._timestamp_cache[scene_path] = self._match_rgb_depth_timestamps(scene_path)

        matched_pairs = self._timestamp_cache[scene_path]
        return [p['rgb_file'] for p in matched_pairs]

    def get_depth_name(self, img_name):
        """Get corresponding depth filename for an RGB image."""
        # Find in timestamp cache
        for scene_path, pairs in self._timestamp_cache.items():
            for p in pairs:
                if p['rgb_file'] == img_name:
                    return p['depth_file']

        # Fallback: same filename (shouldn't happen with proper matching)
        return img_name

    def depth_read(self, path, is_inverse=False, return_torch=False, **kwargs):
        """
        Read depth from uint16 PNG file.

        Bonn RGB-D follows TUM RGB-D format:
        - Depth stored as uint16 PNG with factor 5000
        - pixel_value / 5000.0 = depth in meters
        - pixel_value 0 = invalid/missing

        Args:
            path: Path to depth PNG file
            is_inverse: If True, return inverse depth (1/depth)
            return_torch: If True, return as torch tensor

        Returns:
            Depth map (metric or inverse)
        """
        # Read uint16 PNG
        depth_uint16 = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if depth_uint16 is None:
            logging.warning(f"Failed to load depth: {path}")
            depth = np.zeros((self.ORIGINAL_HEIGHT, self.ORIGINAL_WIDTH), dtype=np.float32)
            if return_torch:
                depth = torch.from_numpy(depth).float()
            return depth

        # Convert to float32 using TUM RGB-D factor (5000, not 1000!)
        # Reference: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
        depth = depth_uint16.astype(np.float32) / 5000.0

        # Invalid mask: depth == 0 means no measurement
        invalid_mask = depth == 0

        if is_inverse:
            # Convert to inverse depth
            depth[invalid_mask] = -1
            inverse_depth = 1.0 / (depth + 1e-8)
            inverse_depth[invalid_mask] = -1

            if kwargs.get('print_minmax', False):
                valid_inv = inverse_depth[~invalid_mask]
                if len(valid_inv) > 0:
                    logging.info(f"minmax inverse depth for {path}: {valid_inv.min():.3f}, {valid_inv.max():.3f}")

            if return_torch:
                inverse_depth = torch.from_numpy(inverse_depth).float()

            return inverse_depth
        else:
            # Return metric depth
            depth[invalid_mask] = -1

            if kwargs.get('print_minmax', False):
                valid_depth = depth[~invalid_mask]
                if len(valid_depth) > 0:
                    logging.info(f"minmax metric depth for {path}: {valid_depth.min():.3f}, {valid_depth.max():.3f}")

            if return_torch:
                depth = torch.from_numpy(depth).float()

            return depth

    def get_focal_length(self, pair, image_shape):
        """
        Get focal length for Bonn dataset.

        Bonn uses fixed camera intrinsics:
        - fx = 542.822841, fy = 542.576870 at 640×480

        Args:
            pair (dict): Data pair (not used, Bonn has fixed intrinsics)
            image_shape (tuple): (H, W) image shape AFTER resizing

        Returns:
            float: Focal length in pixels for current image shape
        """
        # Scale focal length to current image width
        current_width = image_shape[1]
        fx_scaled = self.FX * (current_width / self.ORIGINAL_WIDTH)

        return fx_scaled
