"""
KITTI dataset loader with segmentation support for object-wise evaluation.

KITTI dataset structure:
    data_root/
        raw/
            2011_09_26/
                2011_09_26_drive_0001_sync/
                    image_02/data/  # Left camera images
                    image_03/data/  # Right camera images
        depth/
            2011_09_26_drive_0001_sync/
                proj_depth/groundtruth/image_02/  # LiDAR depth
        segmentation/  # Optional: instance/semantic segmentation
            2011_09_26_drive_0001_sync/
                image_02/  # Segmentation masks

For object-wise evaluation, you need segmentation masks. Options:
1. Use KITTI instance segmentation (cars, pedestrians, cyclists)
2. Use semantic KITTI labels (road, building, vegetation, etc.)
3. Generate masks using Segment Anything Model (SAM)
"""

import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class KITTISegmentationDataset(Dataset):
    """
    KITTI dataset with depth and segmentation for object-wise evaluation.
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'val',
        video_length: int = 5,
        resolution: int = 518,
        max_depth: float = 80.0,
        use_instance_seg: bool = True
    ):
        """
        Initialize KITTI dataset.

        Args:
            data_root: Root directory of KITTI dataset
            split: Dataset split ('train', 'val', 'test')
            video_length: Number of consecutive frames per sequence
            resolution: Target resolution (square)
            max_depth: Maximum depth value (meters)
            use_instance_seg: Use instance segmentation (True) or semantic (False)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.video_length = video_length
        self.resolution = resolution
        self.max_depth = max_depth
        self.use_instance_seg = use_instance_seg

        # Paths
        self.raw_dir = self.data_root / 'raw'
        self.depth_dir = self.data_root / 'depth'
        self.seg_dir = self.data_root / 'segmentation'

        # Class mapping for instance segmentation
        self.instance_classes = {
            0: 'background',
            1: 'car',
            2: 'pedestrian',
            3: 'cyclist'
        }

        # Load sequences
        self.sequences = self._load_sequences()
        logger.info(f"Loaded {len(self.sequences)} sequences from KITTI {split} split")

    def _load_sequences(self):
        """
        Load all valid sequences with depth and segmentation.

        Returns:
            List of tuples (sequence_dir, frame_indices)
        """
        sequences = []

        # Scan for sequences
        if not self.raw_dir.exists():
            logger.warning(f"Raw directory not found: {self.raw_dir}")
            return sequences

        for date_dir in sorted(self.raw_dir.glob('*')):
            if not date_dir.is_dir():
                continue

            for seq_dir in sorted(date_dir.glob('*_sync')):
                image_dir = seq_dir / 'image_02' / 'data'
                if not image_dir.exists():
                    continue

                # Get all image files
                image_files = sorted(image_dir.glob('*.png'))
                if len(image_files) < self.video_length:
                    continue

                # Check if depth and segmentation exist
                seq_name = seq_dir.name
                depth_seq_dir = self.depth_dir / seq_name / 'proj_depth' / 'groundtruth' / 'image_02'
                seg_seq_dir = self.seg_dir / seq_name / 'image_02'

                if not depth_seq_dir.exists():
                    logger.warning(f"Depth not found for {seq_name}, skipping")
                    continue

                if not seg_seq_dir.exists():
                    logger.warning(f"Segmentation not found for {seq_name}, skipping")
                    continue

                # Create sliding window sequences
                for start_idx in range(0, len(image_files) - self.video_length + 1, self.video_length // 2):
                    frame_indices = list(range(start_idx, start_idx + self.video_length))
                    sequences.append((seq_dir, frame_indices))

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Get a sequence with images, depth, and segmentation.

        Returns:
            Dictionary with:
                - images: (T, 3, H, W) tensor
                - depth: (T, H, W) tensor
                - segmentation: (H, W) tensor (last frame only)
                - valid_mask: (H, W) tensor
        """
        seq_dir, frame_indices = self.sequences[idx]

        images = []
        depths = []
        seg_mask = None

        for i, frame_idx in enumerate(frame_indices):
            # Load RGB image
            image_path = seq_dir / 'image_02' / 'data' / f'{frame_idx:010d}.png'
            image = Image.open(image_path).convert('RGB')
            image = image.resize((self.resolution, self.resolution), Image.BILINEAR)
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)  # (3, H, W)
            images.append(image)

            # Load depth
            seq_name = seq_dir.name
            depth_path = self.depth_dir / seq_name / 'proj_depth' / 'groundtruth' / 'image_02' / f'{frame_idx:010d}.png'

            if depth_path.exists():
                depth = self._load_depth(depth_path)
                depth = self._resize_depth(depth, (self.resolution, self.resolution))
            else:
                # If depth not available for this frame, use zeros
                depth = np.zeros((self.resolution, self.resolution), dtype=np.float32)

            depths.append(torch.from_numpy(depth))

            # Load segmentation (only for last frame)
            if i == len(frame_indices) - 1:
                seg_path = self.seg_dir / seq_name / 'image_02' / f'{frame_idx:010d}.png'
                if seg_path.exists():
                    seg_mask = self._load_segmentation(seg_path)
                    seg_mask = self._resize_segmentation(seg_mask, (self.resolution, self.resolution))
                else:
                    # If segmentation not available, create dummy mask
                    seg_mask = np.zeros((self.resolution, self.resolution), dtype=np.int64)

        # Stack into tensors
        images = torch.stack(images, dim=0)  # (T, 3, H, W)
        depths = torch.stack(depths, dim=0)  # (T, H, W)
        seg_mask = torch.from_numpy(seg_mask)  # (H, W)

        # Create valid mask
        valid_mask = (depths[-1] > 0) & (depths[-1] < self.max_depth)

        return {
            'images': images,
            'depth': depths,
            'segmentation': seg_mask,
            'valid_mask': valid_mask,
            'sequence_name': seq_dir.name
        }

    def _load_depth(self, depth_path: Path) -> np.ndarray:
        """
        Load KITTI depth map.

        KITTI depth is stored as uint16 PNG with values in mm.
        """
        depth = Image.open(depth_path)
        depth = np.array(depth).astype(np.float32)
        depth = depth / 256.0  # Convert from mm to meters
        return depth

    def _load_segmentation(self, seg_path: Path) -> np.ndarray:
        """
        Load segmentation mask.

        Format depends on whether using instance or semantic segmentation.
        """
        seg = Image.open(seg_path)
        seg = np.array(seg).astype(np.int64)

        if self.use_instance_seg:
            # Instance segmentation: map instance IDs to class IDs
            # This is dataset-specific mapping - adjust as needed
            # Example: IDs 0-999 = background, 1000-1999 = car, etc.
            seg_classes = np.zeros_like(seg)
            seg_classes[seg == 0] = 0  # background
            seg_classes[(seg >= 1000) & (seg < 2000)] = 1  # car
            seg_classes[(seg >= 2000) & (seg < 3000)] = 2  # pedestrian
            seg_classes[(seg >= 3000) & (seg < 4000)] = 3  # cyclist
            return seg_classes
        else:
            # Semantic segmentation: use class IDs directly
            return seg

    def _resize_depth(self, depth: np.ndarray, target_size: tuple) -> np.ndarray:
        """Resize depth map using nearest neighbor interpolation."""
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        depth_resized = torch.nn.functional.interpolate(
            depth_tensor,
            size=target_size,
            mode='nearest'
        )
        return depth_resized.squeeze().numpy()

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
    images = torch.stack([sample['images'] for sample in batch], dim=0)  # (B, T, 3, H, W)
    depths = torch.stack([sample['depth'] for sample in batch], dim=0)  # (B, T, H, W)
    segmentation = torch.stack([sample['segmentation'] for sample in batch], dim=0)  # (B, H, W)
    valid_mask = torch.stack([sample['valid_mask'] for sample in batch], dim=0)  # (B, H, W)

    return {
        'images': images,
        'depth': depths,
        'segmentation': segmentation,
        'valid_mask': valid_mask,
        'sequence_names': [sample['sequence_name'] for sample in batch]
    }


if __name__ == "__main__":
    # Test dataset loading
    logging.basicConfig(level=logging.INFO)

    dataset = KITTISegmentationDataset(
        data_root='/home/cvlab/hsy/Datasets/KITTI',
        split='val',
        video_length=5,
        resolution=518
    )

    logger.info(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        logger.info(f"Images shape: {sample['images'].shape}")
        logger.info(f"Depth shape: {sample['depth'].shape}")
        logger.info(f"Segmentation shape: {sample['segmentation'].shape}")
        logger.info(f"Valid mask shape: {sample['valid_mask'].shape}")
        logger.info(f"Unique segmentation classes: {torch.unique(sample['segmentation'])}")
    else:
        logger.warning("No sequences found! Check dataset paths.")
