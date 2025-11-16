"""
Virtual KITTI 2 Dataset with segmentation support for object-wise evaluation.

Dataset structure:
    data_root/
        vkitti/
            Scene01/
                clone/
                    frames/
                        rgb/Camera_0/rgb_00000.jpg
                        depth/Camera_0/depth_00000.png (uint16, centimeters)
                        classSegmentation/Camera_0/classgt_00000.png (uint8, class 0-12)
                overcast/...
                rain/...
            Scene02/...

Segmentation classes (VKITTI2 - 13 classes):
    0: Terrain, 1: Tree, 2: Vegetation, 3: Building, 4: Road, 5: GuardRail,
    6: TrafficSign, 7: TrafficLight, 8: Pole, 9: Misc, 10: Truck, 11: Car, 12: Van
    (Note: Sky is not included as a separate class in VKITTI2)
"""

import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import logging
import cv2

logger = logging.getLogger(__name__)


class VKITTISegmentationDataset(Dataset):
    """
    Virtual KITTI 2 Dataset with depth and semantic segmentation for object-wise evaluation.
    """

    # VKITTI2 semantic class mapping (13 classes)
    SEMANTIC_CLASSES = {
        0: 'Terrain',
        1: 'Tree',
        2: 'Vegetation',
        3: 'Building',
        4: 'Road',
        5: 'GuardRail',
        6: 'TrafficSign',
        7: 'TrafficLight',
        8: 'Pole',
        9: 'Misc',
        10: 'Truck',
        11: 'Car',
        12: 'Van'
    }

    def __init__(
        self,
        data_root: str,
        split: str = 'test',
        video_length: int = 50,
        resolution: int = 518,
        max_depth: float = 80.0,
        only_clone: bool = True
    ):
        """
        Initialize VKITTI2 dataset.

        Args:
            data_root: Root directory (expects vkitti/ subdirectory)
            split: Dataset split ('test' or 'train')
            video_length: Number of consecutive frames per sequence
            resolution: Target resolution ('base', '2k', or int for square)
            max_depth: Maximum depth value (meters)
            only_clone: If True, only use 'clone' condition; else use all conditions
        """
        self.data_root = Path(data_root)
        self.split = split
        self.video_length = video_length
        self.only_clone = only_clone

        logger.info(f"VKITTISegmentationDataset initialized with video_length={video_length}, only_clone={only_clone}")

        # Handle resolution like CombinedDataset (preserves aspect ratio)
        # Original VKITTI2 is 1242×375 (3.312 ratio)
        if isinstance(resolution, str):
            if resolution == 'base':
                self.resolution = (1712, 518)  # (width, height) - 3.306 ratio
            elif resolution == '2k':
                self.resolution = (4224, 1276)  # (width, height) - 3.309 ratio
            else:
                # Try to parse as int
                self.resolution = int(resolution)
        else:
            self.resolution = int(resolution)

        # Store width and height
        if isinstance(self.resolution, tuple):
            self.width, self.height = self.resolution
        else:
            # Square resolution (backward compatibility)
            self.width = self.height = self.resolution

        logger.info(f"VKITTISegmentationDataset resolution: width={self.width}, height={self.height}")

        self.max_depth = max_depth

        # VKITTI2 root
        self.vkitti_root = self.data_root / 'vkitti'

        if not self.vkitti_root.exists():
            raise ValueError(f"VKITTI root not found: {self.vkitti_root}")

        # Load sequences
        self.sequences = self._load_sequences()
        logger.info(f"Loaded {len(self.sequences)} sequences from VKITTI2 {split} split")

    def _load_sequences(self):
        """
        Load all valid sequences from VKITTI2 directory.

        Returns:
            List of tuples (scene_path, condition, frame_indices)
        """
        sequences = []

        # Get all scene directories (Scene01, Scene02, ...)
        scene_dirs = sorted([d for d in self.vkitti_root.iterdir()
                            if d.is_dir() and d.name.startswith('Scene')])

        logger.info(f"Found {len(scene_dirs)} scenes in VKITTI2")

        # Condition types in VKITTI2
        if self.only_clone:
            conditions_to_use = ['clone']
        else:
            conditions_to_use = ['clone', 'overcast', 'sunset', 'morning', 'rain', 'fog',
                                '15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right']

        for scene_dir in scene_dirs:
            scene_name = scene_dir.name

            # Get available conditions for this scene
            available_conditions = [c for c in conditions_to_use
                                   if (scene_dir / c / 'frames').exists()]

            for condition in available_conditions:
                condition_path = scene_dir / condition / 'frames'
                rgb_dir = condition_path / 'rgb' / 'Camera_0'
                depth_dir = condition_path / 'depth' / 'Camera_0'
                seg_dir = condition_path / 'classSegmentation' / 'Camera_0'

                # Check required directories
                if not rgb_dir.exists() or not depth_dir.exists() or not seg_dir.exists():
                    logger.warning(f"Missing required directories for {scene_name}/{condition}, skipping")
                    continue

                # Get sorted RGB files
                rgb_files = sorted([f for f in rgb_dir.iterdir() if f.name.startswith('rgb_') and f.suffix == '.jpg'])

                # VKITTI2 has dense segmentation (all frames have segmentation)
                # Extract frame indices
                frame_indices = []
                for rgb_file in rgb_files:
                    frame_num = int(rgb_file.stem.split('_')[1])
                    depth_file = depth_dir / f'depth_{frame_num:05d}.png'
                    seg_file = seg_dir / f'classgt_{frame_num:05d}.png'

                    if depth_file.exists() and seg_file.exists():
                        frame_indices.append(frame_num)

                if len(frame_indices) < 5:
                    logger.warning(f"Sequence {scene_name}/{condition} has only {len(frame_indices)} valid frames (< 5), skipping")
                    continue

                # Create sliding window sequences
                if len(frame_indices) <= self.video_length:
                    # Single sequence with all frames
                    sequences.append((scene_dir, condition, frame_indices))
                else:
                    # Multiple sequences with sliding window
                    for start_idx in range(0, len(frame_indices) - self.video_length + 1, self.video_length // 2):
                        seq_frame_indices = frame_indices[start_idx:start_idx + self.video_length]
                        sequences.append((scene_dir, condition, seq_frame_indices))

        return sequences

    def __len__(self):
        return len(self.sequences)

    def get_focal_length(self, scene_dir, condition):
        """
        Get focal length for VKITTI2 dataset.

        VKITTI2 uses fixed camera intrinsics:
        - fx = 725.0
        - fy = 725.0
        - cx = 620.5
        - cy = 187.0
        - Original resolution: 1242×375

        Args:
            scene_dir (Path): Scene directory path
            condition (str): Condition name (clone, rain, etc.)

        Returns:
            float: Focal length in pixels for current image resolution
        """
        # VKITTI2 fixed intrinsics for original 1242×375 resolution
        fx_original = 725.0
        original_width = 1242

        # Scale focal length to current image width
        fx_scaled = fx_original * (self.width / original_width)

        return fx_scaled

    def __getitem__(self, idx):
        """
        Get a sequence with images, depth, and segmentation.

        Returns:
            Dictionary with:
                - images: (T, 3, H, W) tensor
                - depth: (T, H, W) tensor (inverse depth 1/m, consistent with other seg datasets)
                - segmentations: (T, H, W) tensor (per-frame segmentation)
                - focal_lengths: (T,) tensor
                - sequence_name: str
        """
        scene_dir, condition, frame_indices = self.sequences[idx]
        scene_name = scene_dir.name
        sequence_name = f"{scene_name}_{condition}"

        try:
            condition_path = scene_dir / condition / 'frames'
            rgb_dir = condition_path / 'rgb' / 'Camera_0'
            depth_dir = condition_path / 'depth' / 'Camera_0'
            seg_dir = condition_path / 'classSegmentation' / 'Camera_0'

            images = []
            depths = []
            segmentations = []

            # Load all frames
            for frame_num in frame_indices:
                # Load RGB
                rgb_path = rgb_dir / f'rgb_{frame_num:05d}.jpg'
                rgb = Image.open(rgb_path).convert('RGB')

                # Load depth (uint16 PNG in centimeters)
                depth_path = depth_dir / f'depth_{frame_num:05d}.png'
                depth_cm = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
                if depth_cm is None:
                    raise ValueError(f"Failed to load depth from {depth_path}")

                # Convert centimeters to meters
                depth_meters = depth_cm.astype(np.float32) / 100.0

                # Load segmentation
                seg_path = seg_dir / f'classgt_{frame_num:05d}.png'
                seg = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
                if seg is None:
                    raise ValueError(f"Failed to load segmentation from {seg_path}")

                # Resize to target resolution
                rgb_resized = rgb.resize((self.width, self.height), Image.BILINEAR)
                depth_resized = cv2.resize(depth_meters, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                seg_resized = cv2.resize(seg, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

                # Convert metric depth to inverse depth (1/m) to match WaymoSegmentationDataset/UrbanSynSegmentationDataset
                # This ensures test_comparison.py can handle all segmentation datasets uniformly
                inverse_depth = np.zeros_like(depth_resized)
                valid_mask = depth_resized > 0
                inverse_depth[valid_mask] = 1.0 / depth_resized[valid_mask]
                inverse_depth[~valid_mask] = 0  # Invalid pixels

                # Convert to tensors
                rgb_tensor = TF.to_tensor(rgb_resized)  # [3, H, W], 0-1 range
                depth_tensor = torch.from_numpy(inverse_depth).float()  # [H, W] - inverse depth
                seg_tensor = torch.from_numpy(seg_resized).long()  # [H, W]

                images.append(rgb_tensor)
                depths.append(depth_tensor)
                segmentations.append(seg_tensor)

            # Stack into sequences
            images = torch.stack(images)  # [T, 3, H, W]
            depths = torch.stack(depths)  # [T, H, W] - inverse depth
            segmentations = torch.stack(segmentations)  # [T, H, W]

            # Get focal length
            focal_length = self.get_focal_length(scene_dir, condition)
            focal_lengths = torch.full((len(frame_indices),), focal_length, dtype=torch.float32)

            return {
                'images': images,
                'depth': depths,
                'segmentations': segmentations,
                'focal_lengths': focal_lengths,
                'sequence_name': sequence_name
            }

        except Exception as e:
            logger.error(f"Error loading sequence {sequence_name}: {e}")
            raise


def collate_fn(batch):
    """
    Custom collate function for VKITTI segmentation dataset.
    Handles variable-length sequences and filters None values.
    """
    # Filter out None values
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    # Stack all items
    return {
        'images': torch.stack([item['images'] for item in batch]),  # [B, T, 3, H, W]
        'depth': torch.stack([item['depth'] for item in batch]),  # [B, T, H, W]
        'segmentations': torch.stack([item['segmentations'] for item in batch]),  # [B, T, H, W]
        'focal_lengths': torch.stack([item['focal_lengths'] for item in batch]),  # [B, T]
        'sequence_name': [item['sequence_name'] for item in batch]  # List of strings
    }
