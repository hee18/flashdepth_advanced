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
        camera_name: str = 'FRONT',
        objwise_mode: bool = False
    ):
        """
        Initialize Waymo dataset.

        Args:
            data_root: Root directory (expects waymo_seg/)
            split: Dataset split ('val')
            video_length: Number of consecutive frames per sequence
            resolution: Target resolution ('base', '2k', or int for square)
            max_depth: Maximum depth value (meters)
            camera_name: Camera to use ('FRONT', 'FRONT_LEFT', etc.)
            objwise_mode: If True, only use frames 0-19 with segmentation annotation
        """
        self.data_root = Path(data_root)
        self.split = split
        self.video_length = video_length
        self.objwise_mode = objwise_mode

        # DEBUG: Log video_length and objwise_mode
        logger.info(f"WaymoSegmentationDataset initialized with video_length={video_length}, objwise_mode={objwise_mode}")

        # Handle resolution like CombinedDataset (preserves aspect ratio)
        # Original Waymo is 1920×1280 (1.5 ratio)
        if isinstance(resolution, str):
            if resolution == 'base':
                self.resolution = (784, 518)  # (width, height) - 1.514 ratio
            elif resolution == '2k':
                self.resolution = (1918, 1274)  # (width, height) - 1.505 ratio
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

        logger.info(f"WaymoSegmentationDataset resolution: width={self.width}, height={self.height}")

        self.max_depth = max_depth
        self.camera_name = camera_name

        # Paths for preprocessed dataset
        self.waymo_root = self.data_root / split

        if not self.waymo_root.exists():
            raise ValueError(f"Waymo root not found: {self.waymo_root}")

        # Load sequences
        self.sequences = self._load_sequences()
        logger.info(f"Loaded {len(self.sequences)} sequences from Waymo {split} split")

    def _load_sequences_unfiltered(self):
        """
        Load all sequences without validation filtering (for whole-test mode).

        Returns:
            List of tuples (sequence_dir, frame_count, frame_indices)
        """
        sequences = []

        # Get all sequence directories without filtering
        seq_dirs = sorted([d for d in self.waymo_root.iterdir()
                          if d.is_dir() and d.name.startswith('segment-')])

        logger.info(f"Loading all {len(seq_dirs)} sequences (unfiltered)")

        for seq_dir in seq_dirs:
            camera_dir = seq_dir / self.camera_name

            if not camera_dir.exists():
                logger.warning(f"Camera {self.camera_name} not found in {seq_dir.name}, skipping")
                continue

            rgb_dir = camera_dir / 'rgb' / 'original'
            seg_dir = camera_dir / 'segmentation'
            depth_dir = camera_dir / 'depth'

            if not rgb_dir.exists() or not depth_dir.exists() or not seg_dir.exists():
                logger.warning(f"Missing data directories in {seq_dir.name}, skipping")
                continue

            # Count frames
            rgb_files = sorted([f for f in rgb_dir.iterdir() if f.suffix == '.jpg'])
            seg_files = sorted([f for f in seg_dir.iterdir() if f.suffix == '.png'])
            depth_files = sorted([f for f in depth_dir.iterdir() if f.suffix == '.npy'])

            num_frames = len(rgb_files)

            # Check file count consistency (sparse segmentation: RGB=Depth, Seg<=RGB)
            if num_frames != len(depth_files):
                logger.warning(
                    f"RGB/Depth mismatch for {seq_dir.name}: "
                    f"RGB={num_frames}, Depth={len(depth_files)}, skipping"
                )
                continue

            # For sparse segmentation datasets, seg count can be less than rgb count
            if len(seg_files) == 0:
                logger.warning(
                    f"No segmentation files for {seq_dir.name}, skipping"
                )
                continue

            # Skip sequences with too few frames
            if num_frames < 5:
                logger.warning(f"Sequence {seq_dir.name} has only {num_frames} frames (< 5), skipping")
                continue

            # Use available frames (min of num_frames and video_length)
            actual_video_length = min(num_frames, self.video_length)

            # If num_frames < video_length, create a single sequence with all available frames
            if num_frames <= self.video_length:
                frame_indices = list(range(0, num_frames))
                sequences.append((seq_dir, num_frames, frame_indices))
            else:
                # Normal case: create sliding windows
                for start_idx in range(0, num_frames - self.video_length + 1, self.video_length // 2):
                    frame_indices = list(range(start_idx, start_idx + self.video_length))
                    sequences.append((seq_dir, num_frames, frame_indices))

        return sequences

    def _load_sequences(self):
        """
        Load all valid sequences from preprocessed directory.

        Returns:
            List of tuples (sequence_dir, frame_count, frame_indices)
        """
        sequences = []

        # Get all sequence directories
        all_seq_dirs = sorted([d for d in self.waymo_root.iterdir()
                              if d.is_dir() and d.name.startswith('segment-')])

        # Apply same filtering as WaymoDepth for validation split
        # WaymoDepth uses sorted(all_scenes)[:8] for val split (use first 8 scenes only)
        if self.split == 'val':
            seq_dirs = all_seq_dirs[:8]  # Use first 8 scenes for validation
            logger.info(f"Found {len(all_seq_dirs)} total sequences, using {len(seq_dirs)} for validation (first 8 scenes)")
        else:
            seq_dirs = all_seq_dirs
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

            # Check file count consistency (sparse segmentation: RGB=Depth, Seg<=RGB)
            if num_frames != len(depth_files):
                logger.warning(f"RGB/Depth mismatch for {seq_dir.name}: "
                             f"RGB={num_frames}, Depth={len(depth_files)}, skipping")
                continue

            # For sparse segmentation datasets, seg count can be less than rgb count
            if len(seg_files) == 0:
                logger.warning(f"No segmentation files for {seq_dir.name}, skipping")
                continue

            # For objwise mode: use ALL frames that have segmentation (ignore video_length)
            if self.objwise_mode:
                # Extract actual frame indices from segmentation filenames
                seg_frame_indices = sorted([int(f.stem) for f in seg_files])

                if len(seg_frame_indices) >= 5:  # Minimum sequence length
                    sequences.append((seq_dir, num_frames, seg_frame_indices))
                    logger.info(f"Objwise mode: {seq_dir.name} using {len(seg_frame_indices)} frames with segmentation")
                else:
                    logger.warning(f"Sequence {seq_dir.name} has only {len(seg_frame_indices)} frames with segmentation (< 5), skipping")
            else:
                # Normal mode: create sliding window sequences
                # Use available frames (min of num_frames and video_length)
                # This allows testing on sequences with fewer frames than requested
                actual_video_length = min(num_frames, self.video_length)

                if actual_video_length < 5:
                    logger.warning(f"Sequence {seq_dir.name} has only {num_frames} frames (< 5), skipping")
                    continue

                # Create sliding window sequences
                # If num_frames < video_length, create a single sequence with all available frames
                if num_frames <= self.video_length:
                    frame_indices = list(range(0, num_frames))
                    sequences.append((seq_dir, num_frames, frame_indices))
                else:
                    # Normal case: create sliding windows
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
                - segmentations: (T, H, W) tensor (per-frame segmentation)
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
            segmentations = []
            valid_frame_indices = []  # Track which frames have valid segmentation

            for frame_idx in frame_indices:
                # Load segmentation FIRST to check if we should process this frame
                seg_path = seg_dir / f'{frame_idx:04d}.png'
                
                if not seg_path.exists():
                    # Segmentation file doesn't exist - skip this frame
                    continue
                
                seg_mask = Image.open(seg_path)
                seg_mask_np = np.array(seg_mask).astype(np.uint8)
                
                # Check if segmentation has any valid annotation (> 0 pixels)
                if (seg_mask_np > 0).sum() == 0:
                    # No annotation in this frame - skip
                    continue
                
                # This frame has valid segmentation - NOW load image and depth
                # Load RGB image and resize to (width, height)
                rgb_path = rgb_dir / f'{frame_idx:04d}.jpg'
                image = Image.open(rgb_path).convert('RGB')
                image = image.resize((self.width, self.height), Image.BILINEAR)
                image = np.array(image).astype(np.float32) / 255.0  # [0, 1]

                # Apply ImageNet normalization (same as CombinedDataset)
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                image = (image - mean) / std  # ImageNet normalize

                image = torch.from_numpy(image).permute(2, 0, 1).float()  # (3, H, W)

                # Load sparse depth and convert to dense
                depth_path = depth_dir / f'{frame_idx:04d}.npy'
                sparse_depth = np.load(depth_path)  # (N, 3) [x, y, depth_meters]

                # Convert sparse to dense (height, width)
                depth_map = np.full((self.height, self.width), -1.0, dtype=np.float32)

                if len(sparse_depth) > 0:
                    # Original resolution (1920×1280)
                    orig_h, orig_w = 1280, 1920

                    # Scale coordinates to target resolution
                    scale_x = self.width / orig_w
                    scale_y = self.height / orig_h

                    x_coords = (sparse_depth[:, 0] * scale_x).astype(np.int32)
                    y_coords = (sparse_depth[:, 1] * scale_y).astype(np.int32)
                    depth_values = sparse_depth[:, 2]

                    # Filter coordinates within bounds
                    valid_mask = (
                        (x_coords >= 0) & (x_coords < self.width) &
                        (y_coords >= 0) & (y_coords < self.height) &
                        (depth_values > 0) & (depth_values < self.max_depth)
                    )

                    x_coords = x_coords[valid_mask]
                    y_coords = y_coords[valid_mask]
                    depth_values = depth_values[valid_mask]

                    # Assign depth values (last write wins for duplicate coordinates)
                    if len(x_coords) > 0:
                        depth_map[y_coords, x_coords] = depth_values

                # Convert metric depth (m) to inverse depth (1/m) to match other dataloaders
                # Keep -1.0 for invalid pixels
                valid_depth_mask = depth_map > 0
                depth_map_inverse = depth_map.copy()
                depth_map_inverse[valid_depth_mask] = 1.0 / depth_map[valid_depth_mask]

                depth_map = torch.from_numpy(depth_map_inverse).float()
                
                # Resize segmentation
                seg_mask = seg_mask.resize((self.width, self.height), Image.NEAREST)
                seg_mask_np = np.array(seg_mask).astype(np.uint8)
                seg_mask = torch.from_numpy(seg_mask_np)

                # CRITICAL: Append in same order - image, depth, seg from SAME frame_idx
                images.append(image)
                depths.append(depth_map)
                segmentations.append(seg_mask)
                valid_frame_indices.append(frame_idx)

            # Check if we have any valid frames
            if len(images) == 0:
                logger.error(f"No frames with valid segmentation found in {sequence_name}")
                raise ValueError(f"No valid frames in sequence {sequence_name}")

            # Stack tensors
            images = torch.stack(images, dim=0)  # (T, 3, H, W)
            depths = torch.stack(depths, dim=0)  # (T, H, W)
            segmentations = torch.stack(segmentations, dim=0)  # (T, H, W)

            logger.info(f"[DATASET] {sequence_name}: Loaded {len(images)} frames with segmentation (frames {valid_frame_indices})")

            return {
                'images': images,
                'depth': depths,
                'segmentations': segmentations,  # Changed to plural - per-frame
                'sequence_name': sequence_name,
                'frame_indices': valid_frame_indices  # Actual frame numbers
            }

        except Exception as e:
            logger.error(f"Error loading sequence {sequence_name}, frame indices {frame_indices}: {e}")
            import traceback
            traceback.print_exc()
            # Return None to be filtered by collate_fn
            return None


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
        'segmentations': torch.stack([item['segmentations'] for item in batch]),  # Per-frame segmentations
        'sequence_name': [item['sequence_name'] for item in batch],
        'frame_indices': [item['frame_indices'] for item in batch]  # Actual frame numbers
    }
