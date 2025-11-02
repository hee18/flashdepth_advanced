"""
Waymo Open Dataset with segmentation support for object-wise evaluation.

Dataset structure (preprocessed):
    data_root/
        waymo_seg/val/
            segment-{context_name}/
                FRONT/
                    rgb/original/*.jpg
                    depth/*.npy (sparse format: N×3 [x, y, depth_meters])
                    segmentation/*.png (uint8, semantic class 0-18)

Segmentation classes (Waymo v2.0):
    0: undefined, 1: vehicle, 2: pedestrian, 3: sign, 4: cyclist,
    5: traffic_light, 6: pole, 7: construction_cone, 8: bicycle, 9: motorcycle,
    10: building, 11: vegetation, 12: tree_trunk, 13: curb, 14: road,
    15: lane_marker, 16: other_ground, 17: walkable, 18: sidewalk
"""

import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class WaymoSegmentationDataset(Dataset):
    """
    Waymo Open Dataset with depth and semantic segmentation for object-wise evaluation.
    Loads preprocessed files (not parquet).
    """

    # Waymo semantic class mapping (v2.0)
    SEMANTIC_CLASSES = {
        0: 'undefined',
        1: 'vehicle',
        2: 'pedestrian',
        3: 'sign',
        4: 'cyclist',
        5: 'traffic_light',
        6: 'pole',
        7: 'construction_cone',
        8: 'bicycle',
        9: 'motorcycle',
        10: 'building',
        11: 'vegetation',
        12: 'tree_trunk',
        13: 'curb',
        14: 'road',
        15: 'lane_marker',
        16: 'other_ground',
        17: 'walkable',
        18: 'sidewalk'
    }

    def __init__(
        self,
        data_root: str,
        split: str = 'val',
        video_length: int = 5,
        resolution: int = 518,
        max_depth: float = 80.0,
        camera_name: str = 'FRONT'
    ):
        """
        Initialize Waymo dataset.

        Args:
            data_root: Root directory (expects waymo_seg/)
            split: Dataset split ('val')
            video_length: Number of consecutive frames per sequence
            resolution: Target resolution (square)
            max_depth: Maximum depth value (meters)
            camera_name: Camera to use ('FRONT', 'FRONT_LEFT', etc.)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.video_length = video_length

        # Handle resolution: convert 'base' to 518, or ensure int
        if isinstance(resolution, str):
            self.resolution = 518 if resolution == 'base' else int(resolution)
        else:
            self.resolution = int(resolution)

        self.max_depth = max_depth
        self.camera_name = camera_name

        # Paths for preprocessed dataset
        self.waymo_root = self.data_root / split

        if not self.waymo_root.exists():
            raise ValueError(f"Waymo root not found: {self.waymo_root}")

        # Load sequences
        self.sequences = self._load_sequences()
        logger.info(f"Loaded {len(self.sequences)} sequences from Waymo {split} split")

    def _load_sequences(self):
        """
        Load all valid sequences from preprocessed directory.

        Returns:
            List of tuples (sequence_dir, frame_count, frame_indices)
        """
        sequences = []

        # Get all sequence directories
        seq_dirs = sorted([d for d in self.waymo_root.iterdir()
                          if d.is_dir() and d.name.startswith('segment-')])

        logger.info(f"Found {len(seq_dirs)} sequences in {self.waymo_root}")

        for seq_dir in seq_dirs:
            camera_dir = seq_dir / self.camera_name

            if not camera_dir.exists():
                logger.warning(f"Camera {self.camera_name} not found in {seq_dir.name}, skipping")
                continue

            rgb_dir = camera_dir / 'rgb' / 'original'
            seg_dir = camera_dir / 'segmentation'
            depth_dir = camera_dir / 'depth'

            # Check required directories
            if not rgb_dir.exists() or not seg_dir.exists() or not depth_dir.exists():
                logger.warning(f"Missing required directories in {seq_dir.name}, skipping")
                continue

            # Count frames
            rgb_files = sorted(rgb_dir.glob('*.jpg'))
            seg_files = sorted(seg_dir.glob('*.png'))
            depth_files = sorted(depth_dir.glob('*.npy'))

            num_frames = len(rgb_files)

            if num_frames < self.video_length:
                logger.warning(f"Sequence {seq_dir.name} has only {num_frames} frames, skipping")
                continue

            # Verify all have same number of files
            if len(seg_files) != num_frames or len(depth_files) != num_frames:
                logger.warning(f"Mismatch in file counts for {seq_dir.name}: "
                             f"RGB={num_frames}, Seg={len(seg_files)}, Depth={len(depth_files)}, skipping")
                continue

            # Create sliding window sequences
            for start_idx in range(0, num_frames - self.video_length + 1, self.video_length // 2):
                frame_indices = list(range(start_idx, start_idx + self.video_length))
                sequences.append((seq_dir, num_frames, frame_indices))

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Get a sequence with images, depth, and segmentation.

        Returns:
            Dictionary with:
                - images: (T, 3, H, W) tensor
                - depth: (T, H, W) tensor (sparse depth converted to dense)
                - segmentation: (H, W) tensor (last frame only)
                - valid_mask: (H, W) tensor
                - sequence_name: str
        """
        seq_dir, num_frames, frame_indices = self.sequences[idx]
        sequence_name = seq_dir.name

        try:
            camera_dir = seq_dir / self.camera_name
            rgb_dir = camera_dir / 'rgb' / 'original'
            seg_dir = camera_dir / 'segmentation'
            depth_dir = camera_dir / 'depth'

            images = []
            depths = []

            for frame_idx in frame_indices:
                # Load RGB image
                rgb_path = rgb_dir / f'{frame_idx:04d}.jpg'
                image = Image.open(rgb_path).convert('RGB')
                image = image.resize((self.resolution, self.resolution), Image.BILINEAR)
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image).permute(2, 0, 1)  # (3, H, W)
                images.append(image)

                # Load sparse depth and convert to dense
                depth_path = depth_dir / f'{frame_idx:04d}.npy'
                sparse_depth = np.load(depth_path)  # (N, 3) [x, y, depth_meters]

                # Convert sparse to dense
                depth_map = np.full((self.resolution, self.resolution), -1.0, dtype=np.float32)

                if len(sparse_depth) > 0:
                    # Original resolution (1920×1280)
                    orig_h, orig_w = 1280, 1920

                    # Scale coordinates to target resolution
                    scale_x = self.resolution / orig_w
                    scale_y = self.resolution / orig_h

                    x_coords = (sparse_depth[:, 0] * scale_x).astype(np.int32)
                    y_coords = (sparse_depth[:, 1] * scale_y).astype(np.int32)
                    depth_values = sparse_depth[:, 2]

                    # Filter coordinates within bounds
                    valid_mask = (
                        (x_coords >= 0) & (x_coords < self.resolution) &
                        (y_coords >= 0) & (y_coords < self.resolution) &
                        (depth_values > 0) & (depth_values < self.max_depth)
                    )

                    x_coords = x_coords[valid_mask]
                    y_coords = y_coords[valid_mask]
                    depth_values = depth_values[valid_mask]

                    # Assign depth values (last write wins for duplicate coordinates)
                    if len(x_coords) > 0:
                        depth_map[y_coords, x_coords] = depth_values

                depth_map = torch.from_numpy(depth_map).float()
                depths.append(depth_map)

            # Load segmentation (last frame only)
            last_frame_idx = frame_indices[-1]
            seg_path = seg_dir / f'{last_frame_idx:04d}.png'
            seg_mask = Image.open(seg_path)
            seg_mask = seg_mask.resize((self.resolution, self.resolution), Image.NEAREST)
            seg_mask = np.array(seg_mask).astype(np.uint8)
            seg_mask = torch.from_numpy(seg_mask)

            # Create valid mask (where depth is valid and segmentation is not undefined/ignore)
            last_depth = depths[-1]
            valid_mask = (last_depth > 0) & (seg_mask > 0) & (seg_mask <= 18)

            # Stack images and depths
            images = torch.stack(images, dim=0)  # (T, 3, H, W)
            depths = torch.stack(depths, dim=0)  # (T, H, W)

            return {
                'images': images,
                'depth': depths,
                'segmentation': seg_mask,
                'valid_mask': valid_mask,
                'sequence_name': sequence_name
            }

        except Exception as e:
            logger.error(f"Error loading sequence {sequence_name}, frame indices {frame_indices}: {e}")
            # Return dummy data
            return {
                'images': torch.zeros(self.video_length, 3, self.resolution, self.resolution),
                'depth': torch.zeros(self.video_length, self.resolution, self.resolution),
                'segmentation': torch.zeros(self.resolution, self.resolution, dtype=torch.uint8),
                'valid_mask': torch.zeros(self.resolution, self.resolution, dtype=torch.bool),
                'sequence_name': sequence_name
            }


def collate_fn(batch):
    """Custom collate function to handle None values."""
    # Filter out None values
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    # Stack tensors
    return {
        'images': torch.stack([item['images'] for item in batch]),
        'depth': torch.stack([item['depth'] for item in batch]),
        'segmentation': torch.stack([item['segmentation'] for item in batch]),
        'valid_mask': torch.stack([item['valid_mask'] for item in batch]),
        'sequence_name': [item['sequence_name'] for item in batch]
    }
