"""
MPI-Sintel dataset loader with segmentation support for object-wise evaluation.

Dataset structure:
    data_root/
        sintel/
            images/training/clean/
                [sequence_name]/frame_XXXX.png
            depth/training/depth/
                [sequence_name]/frame_XXXX.dpt
            training/segmentation/
                [sequence_name]/frame_XXXX.png  # 24-bit instance labels
            training/segmentation_invalid/
                [sequence_name]/frame_XXXX.png  # Invalid mask

Segmentation format: 24-bit PNG where label_id = (R * 256 + G) * 256 + B
Instance labels are consistent within a sequence but not across sequences.
"""

import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

# Depth file tag for validation
TAG_FLOAT = 202021.25


class SintelSegmentationDataset(Dataset):
    """
    MPI-Sintel dataset with depth and instance segmentation for object-wise evaluation.
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'val',
        video_length: int = 5,
        resolution: int = 518,
        max_depth: float = 1000.0,
        pass_type: str = 'clean'
    ):
        """
        Initialize Sintel dataset.

        Args:
            data_root: Root directory (expects sintel_seg/ or similar)
            split: Dataset split ('train', 'val')
            video_length: Number of consecutive frames per sequence
            resolution: Target resolution (square)
            max_depth: Maximum depth value (meters)
            pass_type: Rendering pass ('clean' or 'final')
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
        self.pass_type = pass_type

        # Paths
        self.sintel_root = self.data_root / 'sintel'
        self.image_dir = self.sintel_root / 'images' / 'training' / pass_type
        self.depth_dir = self.sintel_root / 'depth' / 'training' / 'depth'
        self.seg_dir = self.sintel_root / 'training' / 'segmentation'
        self.seg_invalid_dir = self.sintel_root / 'training' / 'segmentation_invalid'

        # Validate paths
        if not self.sintel_root.exists():
            raise ValueError(f"Sintel root not found: {self.sintel_root}")
        if not self.image_dir.exists():
            raise ValueError(f"Image directory not found: {self.image_dir}")
        if not self.depth_dir.exists():
            logger.warning(f"Depth directory not found: {self.depth_dir}")
        if not self.seg_dir.exists():
            raise ValueError(f"Segmentation directory not found: {self.seg_dir}")

        # Load sequences
        self.sequences = self._load_sequences()
        logger.info(f"Loaded {len(self.sequences)} sequences from Sintel {split} split")

    def _load_sequences(self):
        """
        Load all valid sequences with depth and segmentation.

        Returns:
            List of tuples (sequence_name, frame_indices)
        """
        sequences = []

        # Get all sequence directories
        all_sequences = sorted([d.name for d in self.image_dir.iterdir() if d.is_dir()])

        # Split sequences (val = last 15, train = first 8, following common practice)
        if self.split == 'val':
            selected_sequences = all_sequences[8:]  # Last 15 sequences
        elif self.split == 'train':
            selected_sequences = all_sequences[:8]  # First 8 sequences
        else:
            selected_sequences = all_sequences

        logger.info(f"Split '{self.split}': {len(selected_sequences)} sequences")
        logger.info(f"Selected sequences: {selected_sequences}")

        for seq_name in selected_sequences:
            seq_image_dir = self.image_dir / seq_name
            seq_depth_dir = self.depth_dir / seq_name
            seq_seg_dir = self.seg_dir / seq_name

            # Check if all directories exist
            if not seq_depth_dir.exists():
                logger.warning(f"Depth not found for {seq_name}, skipping")
                continue
            if not seq_seg_dir.exists():
                logger.warning(f"Segmentation not found for {seq_name}, skipping")
                continue

            # Get frame files
            frame_files = sorted([f.name for f in seq_image_dir.glob('frame_*.png')])
            if len(frame_files) < self.video_length:
                logger.warning(f"Sequence {seq_name} has only {len(frame_files)} frames, skipping")
                continue

            # Create sliding window sequences
            num_frames = len(frame_files)
            for start_idx in range(0, num_frames - self.video_length + 1, self.video_length // 2):
                frame_indices = list(range(start_idx, start_idx + self.video_length))
                sequences.append((seq_name, frame_indices))

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
                - sequence_name: str
        """
        seq_name, frame_indices = self.sequences[idx]

        images = []
        depths = []
        seg_mask = None
        seg_invalid_mask = None

        for i, frame_idx in enumerate(frame_indices):
            frame_name = f'frame_{frame_idx + 1:04d}'  # Sintel uses 1-based indexing

            # Load RGB image
            image_path = self.image_dir / seq_name / f'{frame_name}.png'
            if not image_path.exists():
                logger.error(f"Image not found: {image_path}")
                return None

            image = Image.open(image_path).convert('RGB')
            image = image.resize((self.resolution, self.resolution), Image.BILINEAR)
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)  # (3, H, W)
            images.append(image)

            # Load depth
            depth_path = self.depth_dir / seq_name / f'{frame_name}.dpt'
            if depth_path.exists():
                depth = self._load_depth(depth_path)
                depth = self._resize_depth(depth, (self.resolution, self.resolution))
            else:
                logger.warning(f"Depth not found: {depth_path}")
                depth = np.zeros((self.resolution, self.resolution), dtype=np.float32)

            depths.append(torch.from_numpy(depth))

            # Load segmentation (only for last frame)
            if i == len(frame_indices) - 1:
                seg_path = self.seg_dir / seq_name / f'{frame_name}.png'
                if seg_path.exists():
                    seg_mask = self._load_segmentation(seg_path)
                    seg_mask = self._resize_segmentation(seg_mask, (self.resolution, self.resolution))
                else:
                    logger.warning(f"Segmentation not found: {seg_path}")
                    seg_mask = np.zeros((self.resolution, self.resolution), dtype=np.int64)

                # Load invalid mask if available
                seg_invalid_path = self.seg_invalid_dir / seq_name / f'{frame_name}.png'
                if seg_invalid_path.exists():
                    seg_invalid_mask = Image.open(seg_invalid_path)
                    seg_invalid_mask = seg_invalid_mask.resize((self.resolution, self.resolution), Image.NEAREST)
                    seg_invalid_mask = np.array(seg_invalid_mask) > 0
                else:
                    seg_invalid_mask = np.zeros((self.resolution, self.resolution), dtype=bool)

        # Stack into tensors
        images = torch.stack(images, dim=0)  # (T, 3, H, W)
        depths = torch.stack(depths, dim=0)  # (T, H, W)
        seg_mask = torch.from_numpy(seg_mask)  # (H, W)

        # Create valid mask
        # Sintel depth is stored as inverse depth, with -1 for invalid pixels
        valid_mask = (depths[-1] > 0) & (depths[-1] < self.max_depth)
        if seg_invalid_mask is not None:
            valid_mask = valid_mask & ~torch.from_numpy(seg_invalid_mask)

        return {
            'image': images,
            'depth': depths,
            'segmentation': seg_mask,
            'valid_mask': valid_mask,
            'sequence_name': seq_name
        }

    def _load_depth(self, depth_path: Path) -> np.ndarray:
        """
        Load Sintel depth map from .dpt file.

        Sintel stores depth in a custom binary format.
        Returns inverse depth (1/depth) with -1 for invalid pixels.
        """
        with open(depth_path, 'rb') as f:
            # Check tag
            check = np.fromfile(f, dtype=np.float32, count=1)[0]
            if check != TAG_FLOAT:
                logger.error(f"Wrong tag in depth file {depth_path}: {check} (expected {TAG_FLOAT})")
                return None

            # Read dimensions
            width = np.fromfile(f, dtype=np.int32, count=1)[0]
            height = np.fromfile(f, dtype=np.int32, count=1)[0]

            # Read depth data
            depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))

        # Handle invalid values
        invalid_mask = np.logical_or.reduce((
            np.isinf(depth),
            np.isnan(depth),
            depth == 0,
            depth < 1e-5
        ))

        sky_mask = depth > 1e4

        # Convert to inverse depth (as done in original sintel_dataset.py)
        depth[invalid_mask] = -1
        inverse_depth = np.zeros_like(depth)
        valid_depth_mask = (depth > 0) & ~sky_mask
        inverse_depth[valid_depth_mask] = 1.0 / depth[valid_depth_mask]
        inverse_depth[sky_mask] = 0
        inverse_depth[invalid_mask] = -1

        # Convert back to depth for consistency with other datasets
        depth_metric = np.zeros_like(inverse_depth)
        valid_inv_mask = inverse_depth > 1e-5
        depth_metric[valid_inv_mask] = 1.0 / inverse_depth[valid_inv_mask]
        depth_metric[~valid_inv_mask] = 0

        return depth_metric.astype(np.float32)

    def _load_segmentation(self, seg_path: Path) -> np.ndarray:
        """
        Load Sintel segmentation mask.

        Sintel segmentation is stored as 24-bit PNG where:
        label_id = (R * 256 + G) * 256 + B
        """
        seg_img = Image.open(seg_path)
        seg_rgb = np.array(seg_img)

        if seg_rgb.ndim == 2:
            # Grayscale image - use directly
            return seg_rgb.astype(np.int64)

        # Convert 24-bit RGB to label ID
        r, g, b = seg_rgb[:, :, 0], seg_rgb[:, :, 1], seg_rgb[:, :, 2]
        seg_mask = ((r.astype(np.int64) * 256 + g.astype(np.int64)) * 256 + b.astype(np.int64))

        return seg_mask

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

    dataset = SintelSegmentationDataset(
        data_root='/home/cvlab/hsy/Datasets/sintel_seg',
        split='val',
        video_length=5,
        resolution=518
    )

    logger.info(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        if sample is not None:
            logger.info(f"Images shape: {sample['images'].shape}")
            logger.info(f"Depth shape: {sample['depth'].shape}")
            logger.info(f"Segmentation shape: {sample['segmentation'].shape}")
            logger.info(f"Valid mask shape: {sample['valid_mask'].shape}")
            logger.info(f"Depth range: {sample['depth'][-1][sample['valid_mask']].min():.2f} - {sample['depth'][-1][sample['valid_mask']].max():.2f}")
            logger.info(f"Unique segmentation IDs (first 10): {torch.unique(sample['segmentation'])[:10]}")
            logger.info(f"Sequence: {sample['sequence_name']}")
    else:
        logger.warning("No sequences found! Check dataset paths.")
