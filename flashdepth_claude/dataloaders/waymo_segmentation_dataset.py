"""
Waymo Open Dataset v2.0 loader with segmentation support for object-wise evaluation.

Dataset structure:
    data_root/
        waymo/
            2.0.1/val/
                camera_image/*.parquet           # RGB images (JPEG encoded)
                camera_segmentation/*.parquet    # Panoptic segmentation (PNG encoded)
                lidar_camera_projection/*.parquet  # LiDAR range images

Segmentation format:
    - Panoptic label: 32-bit integer PNG
    - semantic_class = panoptic_label // panoptic_label_divisor (usually 1000)
    - instance_id = panoptic_label % panoptic_label_divisor

Camera naming (key.camera_name):
    1: FRONT, 2: FRONT_LEFT, 3: FRONT_RIGHT, 4: SIDE_LEFT, 5: SIDE_RIGHT
"""

import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import logging
import io

try:
    import pyarrow.parquet as pq
    import pandas as pd
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    logging.warning("PyArrow not available. Install with: pip install pyarrow")

logger = logging.getLogger(__name__)


class WaymoSegmentationDataset(Dataset):
    """
    Waymo Open Dataset v2.0 with depth and semantic segmentation for object-wise evaluation.
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
        camera_name: int = 1,  # 1 = FRONT camera
        use_depth: bool = False  # Depth extraction from LiDAR is complex
    ):
        """
        Initialize Waymo dataset.

        Args:
            data_root: Root directory (expects waymo_seg/ or similar)
            split: Dataset split ('train', 'val')
            video_length: Number of consecutive frames per sequence
            resolution: Target resolution (square)
            max_depth: Maximum depth value (meters)
            camera_name: Camera to use (1=FRONT, 2=FRONT_LEFT, etc.)
            use_depth: Whether to load depth (complex, set False for seg-only)
        """
        if not PYARROW_AVAILABLE:
            raise ImportError("PyArrow is required. Install with: pip install pyarrow")

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
        self.use_depth = use_depth

        # Paths
        self.waymo_root = self.data_root / 'waymo' / '2.0.1' / split
        self.image_dir = self.waymo_root / 'camera_image'
        self.seg_dir = self.waymo_root / 'camera_segmentation'
        self.depth_dir = self.waymo_root / 'lidar_camera_projection'

        # Validate paths
        if not self.waymo_root.exists():
            raise ValueError(f"Waymo root not found: {self.waymo_root}")
        if not self.image_dir.exists():
            raise ValueError(f"Image directory not found: {self.image_dir}")
        if not self.seg_dir.exists():
            logger.warning(f"Segmentation directory not found: {self.seg_dir}")

        # Load sequences
        self.sequences = self._load_sequences()
        logger.info(f"Loaded {len(self.sequences)} sequences from Waymo {split} split")

    def _load_sequences(self):
        """
        Load all valid sequences with images and segmentation.

        Returns:
            List of tuples (parquet_file_path, frame_indices)
        """
        sequences = []

        # Get all parquet files
        image_files = sorted(self.image_dir.glob('*.parquet'))

        logger.info(f"Found {len(image_files)} parquet files in {self.image_dir}")

        for image_file in image_files:
            # Check if corresponding segmentation file exists
            seg_file = self.seg_dir / image_file.name

            # Load image parquet to get frame count
            try:
                img_table = pq.read_table(image_file)
                img_df = img_table.to_pandas()

                # Filter by camera
                camera_df = img_df[img_df['key.camera_name'] == self.camera_name]
                num_frames = len(camera_df)

                if num_frames < self.video_length:
                    logger.warning(f"Sequence {image_file.name} has only {num_frames} frames for camera {self.camera_name}, skipping")
                    continue

                # Check if segmentation exists and has data
                has_seg = False
                if seg_file.exists():
                    seg_table = pq.read_table(seg_file)
                    if seg_table.num_rows > 0:
                        seg_df = seg_table.to_pandas()
                        seg_camera_df = seg_df[seg_df['key.camera_name'] == self.camera_name]
                        has_seg = len(seg_camera_df) > 0

                if not has_seg:
                    logger.warning(f"Sequence {image_file.name} has no segmentation data for camera {self.camera_name}, skipping")
                    continue

                # Create sliding window sequences
                for start_idx in range(0, num_frames - self.video_length + 1, self.video_length // 2):
                    frame_indices = list(range(start_idx, start_idx + self.video_length))
                    sequences.append((image_file, seg_file, frame_indices))

            except Exception as e:
                logger.error(f"Error loading {image_file.name}: {e}")
                continue

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Get a sequence with images, depth, and segmentation.

        Returns:
            Dictionary with:
                - images: (T, 3, H, W) tensor
                - depth: (T, H, W) tensor (zeros if use_depth=False)
                - segmentation: (H, W) tensor (last frame only)
                - valid_mask: (H, W) tensor
                - sequence_name: str
        """
        image_file, seg_file, frame_indices = self.sequences[idx]

        try:
            # Load image and segmentation dataframes
            img_table = pq.read_table(image_file)
            img_df = img_table.to_pandas()
            img_camera_df = img_df[img_df['key.camera_name'] == self.camera_name].reset_index(drop=True)

            seg_table = pq.read_table(seg_file)
            seg_df = seg_table.to_pandas()
            seg_camera_df = seg_df[seg_df['key.camera_name'] == self.camera_name].reset_index(drop=True)

            images = []
            depths = []
            seg_mask = None

            for i, frame_idx in enumerate(frame_indices):
                # Load RGB image
                img_row = img_camera_df.iloc[frame_idx]
                img_bytes = img_row['[CameraImageComponent].image']
                image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                image = image.resize((self.resolution, self.resolution), Image.BILINEAR)
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image).permute(2, 0, 1)  # (3, H, W)
                images.append(image)

                # Depth: placeholder (LiDAR projection is complex)
                if self.use_depth:
                    # TODO: Implement LiDAR range image to depth conversion
                    depth = np.zeros((self.resolution, self.resolution), dtype=np.float32)
                else:
                    depth = np.zeros((self.resolution, self.resolution), dtype=np.float32)

                depths.append(torch.from_numpy(depth))

                # Load segmentation (only for last frame)
                if i == len(frame_indices) - 1:
                    if frame_idx < len(seg_camera_df):
                        seg_row = seg_camera_df.iloc[frame_idx]
                        seg_bytes = seg_row['[CameraSegmentationLabelComponent].panoptic_label']
                        divisor = seg_row['[CameraSegmentationLabelComponent].panoptic_label_divisor']

                        # Decode panoptic segmentation
                        seg_img = Image.open(io.BytesIO(seg_bytes))
                        seg_panoptic = np.array(seg_img).astype(np.int64)

                        # Extract semantic class: semantic_class = panoptic_label // divisor
                        seg_semantic = seg_panoptic // divisor
                        seg_semantic = self._resize_segmentation(seg_semantic, (self.resolution, self.resolution))
                        seg_mask = seg_semantic
                    else:
                        logger.warning(f"Segmentation frame {frame_idx} not found, using zeros")
                        seg_mask = np.zeros((self.resolution, self.resolution), dtype=np.int64)

            # Stack into tensors
            images = torch.stack(images, dim=0)  # (T, 3, H, W)
            depths = torch.stack(depths, dim=0)  # (T, H, W)
            seg_mask = torch.from_numpy(seg_mask) if seg_mask is not None else torch.zeros((self.resolution, self.resolution), dtype=torch.int64)

            # Create valid mask
            if self.use_depth:
                valid_mask = (depths[-1] > 0) & (depths[-1] < self.max_depth)
            else:
                # Without depth, mark all pixels as valid except undefined class
                valid_mask = (seg_mask != 0)  # 0 = undefined class

            return {
                'image': images,
                'depth': depths,
                'segmentation': seg_mask,
                'valid_mask': valid_mask,
                'sequence_name': image_file.stem
            }

        except Exception as e:
            logger.error(f"Error loading sequence {idx}: {e}")
            return None

    def _resize_segmentation(self, seg: np.ndarray, target_size: tuple) -> np.ndarray:
        """Resize segmentation mask using nearest neighbor interpolation."""
        seg_tensor = torch.from_numpy(seg).unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
        seg_resized = torch.nn.functional.interpolate(
            seg_tensor,
            size=target_size,
            mode='nearest'
        )
        return seg_resized.squeeze().numpy().astype(np.int64)


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences."""
    # Filter out None samples
    batch = [sample for sample in batch if sample is not None]

    if len(batch) == 0:
        return None

    # Stack batch
    images = torch.stack([sample['image'] for sample in batch], dim=0)  # (B, T, 3, H, W)
    depths = torch.stack([sample['depth'] for sample in batch], dim=0)  # (B, T, H, W)
    segmentation = torch.stack([sample['segmentation'] for sample in batch], dim=0)  # (B, H, W)
    valid_mask = torch.stack([sample['valid_mask'] for sample in batch], dim=0)  # (B, H, W)

    return {
        'image': images,
        'depth': depths,
        'segmentation': segmentation,
        'valid_mask': valid_mask,
        'sequence_names': [sample['sequence_name'] for sample in batch]
    }


if __name__ == "__main__":
    # Test dataset loading
    logging.basicConfig(level=logging.INFO)

    dataset = WaymoSegmentationDataset(
        data_root='/home/cvlab/hsy/Datasets/waymo_seg',
        split='val',
        video_length=5,
        resolution=518,
        camera_name=1,  # FRONT camera
        use_depth=False  # Depth extraction is complex, use False for seg-only
    )

    logger.info(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        if sample is not None:
            logger.info(f"Images shape: {sample['images'].shape}")
            logger.info(f"Depth shape: {sample['depth'].shape}")
            logger.info(f"Segmentation shape: {sample['segmentation'].shape}")
            logger.info(f"Valid mask shape: {sample['valid_mask'].shape}")
            logger.info(f"Unique semantic classes: {torch.unique(sample['segmentation'])}")
            logger.info(f"Sequence: {sample['sequence_name']}")
    else:
        logger.warning("No sequences found! Check dataset paths.")
