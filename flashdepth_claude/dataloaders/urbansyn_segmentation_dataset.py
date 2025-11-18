"""
UrbanSyn Dataset with segmentation support for object-wise evaluation.

Dataset structure:
    data_root/
        urbansyn/
            rgb/rgb_XXXX.png
            depth/depth_XXXX.exr
            ss_trainid/ss_color_XXXX.png (semantic segmentation, Cityscapes 19-class trainId format)
            camera_metadata.json

Segmentation classes (Cityscapes trainId format):
    0: road, 1: sidewalk, 2: building, 3: wall, 4: fence,
    5: pole, 6: traffic_light, 7: traffic_sign, 8: vegetation,
    9: terrain, 10: sky, 11: person, 12: rider, 13: car,
    14: truck, 15: bus, 16: train, 17: motorcycle, 18: bicycle,
    255: ignore
"""

import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import logging
import json
import cv2
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

logger = logging.getLogger(__name__)


class UrbanSynSegmentationDataset(Dataset):
    """
    UrbanSyn Dataset with depth and semantic segmentation for object-wise evaluation.
    """

    # Cityscapes semantic class mapping (trainId format)
    SEMANTIC_CLASSES = {
        0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence',
        5: 'pole', 6: 'traffic_light', 7: 'traffic_sign', 8: 'vegetation',
        9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car',
        14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle',
        255: 'ignore'
    }

    def __init__(
        self,
        data_root: str,
        split: str = 'test',
        video_length: int = 5,
        resolution: int = 518,
        max_frames: int = 1000
    ):
        """
        Initialize UrbanSyn dataset with segmentation.

        Args:
            data_root: Root directory (expects urbansyn/)
            split: Dataset split ('test', 'train', 'val')
            video_length: Number of consecutive frames per sequence
            resolution: Target resolution ('base', '2k', int for square, or None for original 2048×1024)
            max_frames: Maximum number of frames to use (UrbanSyn has 1000 total)
        """
        self.data_root = Path(data_root) / 'urbansyn'
        self.split = split
        self.video_length = video_length
        self.max_frames = max_frames

        logger.info(f"UrbanSynSegmentationDataset initialized with video_length={video_length}")

        # Handle resolution like CombinedDataset (preserves aspect ratio)
        # Original UrbanSyn is 2048×1024 (2.0 ratio)
        if resolution is None:
            # Use original resolution for test_comparison.py
            self.resolution = (2048, 1024)  # Original resolution
        elif isinstance(resolution, str):
            if resolution == 'base':
                self.resolution = (1036, 518)  # (width, height) - 2.0 ratio
            elif resolution == '2k':
                self.resolution = (2044, 1022)  # (width, height) - 2.0 ratio
            else:
                # Try to parse as int
                self.resolution = int(resolution)
        else:
            self.resolution = resolution

        # Parse resolution
        if isinstance(self.resolution, tuple):
            self.width, self.height = self.resolution
        else:
            # Square resolution
            self.width = self.height = self.resolution

        logger.info(f"UrbanSyn resolution: {self.width}×{self.height}")

        # Load camera metadata
        self.camera_fx = self._load_camera_metadata()

        # Get list of available frames
        rgb_dir = self.data_root / 'rgb'
        seg_dir = self.data_root / 'ss_trainid'

        if not rgb_dir.exists():
            raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
        if not seg_dir.exists():
            raise FileNotFoundError(f"Segmentation directory not found: {seg_dir}")

        # Get all RGB files
        all_rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
        # Extract frame numbers (rgb_XXXX.png -> XXXX)
        self.available_frames = []
        for f in all_rgb_files[:self.max_frames]:
            frame_num = int(f.split('_')[1].split('.')[0])
            # Check if corresponding segmentation exists
            seg_file = f'ss_color_{frame_num:04d}.png'
            if (seg_dir / seg_file).exists():
                self.available_frames.append(frame_num)

        logger.info(f"Found {len(self.available_frames)} frames with segmentation")

        # Create sequences (video_length consecutive frames)
        self.sequences = []
        for i in range(0, len(self.available_frames) - self.video_length + 1):
            # Check if frames are consecutive
            frames = self.available_frames[i:i + self.video_length]
            if frames[-1] - frames[0] == self.video_length - 1:
                self.sequences.append(frames)

        logger.info(f"Created {len(self.sequences)} sequences of length {self.video_length}")

    def _load_camera_metadata(self):
        """
        Load camera metadata from root directory.
        UrbanSyn has a single camera_metadata.json file at the root.
        """
        metadata_path = self.data_root / 'camera_metadata.json'
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Extract camera parameters
            camera_params = metadata['parameters'][0]['Camera']
            focal_length_mm = camera_params['focalLength_mm']
            sensor_width_mm = camera_params['sensorWidth_mm']

            # Compute fx in pixels for 2048×1024 images
            width = 2048
            fx = focal_length_mm * width / sensor_width_mm

            logger.info(f"UrbanSyn camera fx: {fx:.2f} pixels (focal={focal_length_mm}mm, sensor={sensor_width_mm}mm)")
            return fx
        except Exception as e:
            logger.warning(f"Error reading camera metadata from {metadata_path}: {e}, using fallback fx=1731.0")
            return 1731.0  # Typical value for 2048×1024 with ~60° FOV

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Get a sequence with images, depth, and segmentation.

        Returns:
            Dictionary with:
                - image: (T, 3, H, W) tensor
                - depth: (T, H, W) tensor (inverse depth)
                - segmentations: (T, H, W) tensor (per-frame segmentation)
                - focal_lengths: (T,) tensor (focal length for each frame)
                - sequence_name: str
        """
        frame_indices = self.sequences[idx]

        rgb_dir = self.data_root / 'rgb'
        depth_dir = self.data_root / 'depth'
        seg_dir = self.data_root / 'ss_trainid'

        images = []
        depths = []
        segmentations = []
        focal_lengths = []

        for frame_idx in frame_indices:
            # Load RGB image
            rgb_path = rgb_dir / f'rgb_{frame_idx:04d}.png'
            image = Image.open(rgb_path).convert('RGB')
            image = image.resize((self.width, self.height), Image.BILINEAR)
            image = np.array(image).astype(np.float32) / 255.0  # [0, 1]

            # Apply ImageNet normalization (same as CombinedDataset)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            image = (image - mean) / std  # ImageNet normalize

            image = torch.from_numpy(image).permute(2, 0, 1).float()  # (3, H, W)

            # Load depth (EXR format, in meters)
            depth_path = depth_dir / f'depth_{frame_idx:04d}.exr'
            depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH).astype(np.float32)
            depth *= 1e5  # According to documentation, *1e5 gives meters

            # Load segmentation mask for sky detection
            seg_path = seg_dir / f'ss_color_{frame_idx:04d}.png'
            seg_mask = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE).astype(np.uint8)

            # Handle sky (class ID 10)
            sky_mask = seg_mask == 10
            depth[sky_mask] = -1  # Mark sky as invalid

            # Convert to inverse depth (1/m)
            inverse_depth = np.zeros_like(depth)
            valid_mask = depth > 0
            inverse_depth[valid_mask] = 1.0 / depth[valid_mask]
            inverse_depth[~valid_mask] = 0  # Invalid pixels

            # Resize depth and segmentation
            inverse_depth = cv2.resize(inverse_depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            seg_mask = cv2.resize(seg_mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

            depths.append(torch.from_numpy(inverse_depth).float())
            segmentations.append(torch.from_numpy(seg_mask).long())
            images.append(image)

            # Focal length (scaled to current resolution)
            fx_scaled = self.camera_fx * (self.width / 2048)
            focal_lengths.append(fx_scaled)

        # Stack into tensors
        images = torch.stack(images, dim=0)  # (T, 3, H, W)
        depths = torch.stack(depths, dim=0)  # (T, H, W)
        segmentations = torch.stack(segmentations, dim=0)  # (T, H, W)
        focal_lengths = torch.tensor(focal_lengths, dtype=torch.float32)  # (T,)

        return {
            'image': images,
            'depth': depths,
            'segmentations': segmentations,
            'focal_lengths': focal_lengths,
            'sequence_name': f'urbansyn_{frame_indices[0]:04d}',
            'dataset_name': 'urbansyn'  # For intrinsics lookup
        }


def urbansyn_collate_fn(batch):
    """
    Custom collate function for UrbanSyn segmentation dataset.
    Adds batch dimension and ensures consistent shapes.
    """
    if len(batch) == 0:
        return None

    # Batch size should be 1 for testing
    assert len(batch) == 1, "Batch size must be 1 for UrbanSyn segmentation dataset"

    sample = batch[0]

    # Add batch dimension
    # Use 'images' and 'depths' (plural) to match ComparisonDataset format
    return {
        'images': sample['image'].unsqueeze(0),  # (1, T, 3, H, W) - Changed to 'images' (plural)
        'depths': sample['depth'].unsqueeze(0),  # (1, T, H, W) - Changed to 'depths' (plural)
        'dataset_name': [sample['dataset_name']],  # List for consistency with other datasets
        'segmentations': sample['segmentations'].unsqueeze(0),  # (1, T, H, W)
        'focal_lengths_actual': sample['focal_lengths'].unsqueeze(0),  # (1, T) - Match CombinedDataset naming
        'sequence_name': [sample['sequence_name']]  # List for consistency
    }
