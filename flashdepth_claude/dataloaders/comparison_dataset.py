"""
Comparison Dataset for Depth Estimation Model Benchmarking

This dataset provides ORIGINAL resolution images and ground truth depth
to allow each model to process inputs at their optimal resolution.

Key differences from CombinedDataset (training):
- No forced resizing to fixed resolution
- GT depth at original resolution
- Simple image normalization only
- Each model handles resizing internally

Supported datasets:
- ETH3D
- KITTI
- Sintel
- ScanNet
- TartanAir
- Bonn
- NYU Depth V2
"""

import os
import cv2
import torch
import numpy as np
import logging
from torch.utils.data import Dataset
from PIL import Image

logger = logging.getLogger(__name__)


class ComparisonDataset(Dataset):
    """
    Dataset for comparing depth estimation models at original resolution.

    Each model receives original resolution images and can process them
    according to their own requirements.
    """

    def __init__(self, dataset_name, data_root, split='test', video_length=50, chunk_size=None, objwise_enabled=False, only_clone=False, unrealstereo4k_seq_list=None, unrealstereo4k_seq=None):
        """
        Args:
            dataset_name: Name of dataset ('eth3d', 'kitti', 'sintel', etc.)
            data_root: Root directory containing datasets
            split: 'test' or 'val'
            video_length: Maximum sequence length
            chunk_size: If set, load frames in chunks to reduce memory usage
                       Useful for high-resolution datasets (e.g., 50 for 4K images)
            objwise_enabled: If True, load segmentation masks for object-wise evaluation
            only_clone: If True and dataset is VKITTI, only use 'clone' condition (5 sequences instead of 50)
            unrealstereo4k_seq_list: If set, only use these sequence numbers (list of 0-8) for UnrealStereo4K
            unrealstereo4k_seq: (deprecated, use unrealstereo4k_seq_list) Single sequence number for backward compatibility
        """
        # Handle dataset name aliases
        dataset_name_lower = dataset_name.lower()
        if dataset_name_lower in ['unreal4k', 'unreal']:
            dataset_name_lower = 'unrealstereo4k'

        self.dataset_name = dataset_name_lower
        self.data_root = data_root

        # Handle backward compatibility: convert single seq to list
        if unrealstereo4k_seq_list is not None:
            self.unrealstereo4k_seq_list = unrealstereo4k_seq_list
        elif unrealstereo4k_seq is not None:
            self.unrealstereo4k_seq_list = [unrealstereo4k_seq]
        else:
            self.unrealstereo4k_seq_list = None

        self.split = split
        self.video_length = video_length
        self.objwise_enabled = objwise_enabled
        self.only_clone = only_clone

        # Auto-set chunk_size for efficient memory usage
        # DISABLED: chunk_size limits total frames loaded, which breaks video models
        # Instead, we rely on:
        # 1. video_length to limit sequence length (user-specified via --vid-len)
        # 2. num_workers=0 for high-res datasets to avoid OOM from parallel loading
        # 3. GPU memory management (empty_cache after each sequence)
        if chunk_size is None:
            self.chunk_size = None  # Disabled by default
        else:
            self.chunk_size = chunk_size  # Allow user override if needed

        # Load dataset-specific configuration
        self.dataset_path = os.path.join(data_root, self.dataset_name)

        # Cache for intrinsics to avoid repeated file reads
        self._intrinsics_cache = {}

        # Build sequence list
        self.sequences = self._build_sequences()

        logger.info(f"[ComparisonDataset] {dataset_name} {split}: {len(self.sequences)} sequences")

    def _build_sequences(self):
        """Build list of sequences for the dataset"""
        if self.dataset_name == 'eth3d':
            return self._build_eth3d_sequences()
        elif self.dataset_name == 'kitti':
            return self._build_kitti_sequences()
        elif self.dataset_name == 'sintel':
            return self._build_sintel_sequences()
        elif self.dataset_name == 'scannet':
            return self._build_scannet_sequences()
        elif self.dataset_name == 'tartanair':
            return self._build_tartanair_sequences()
        elif self.dataset_name == 'bonn':
            return self._build_bonn_sequences()
        elif self.dataset_name == 'nyu':
            return self._build_nyu_sequences()
        elif self.dataset_name == 'vkitti':
            return self._build_vkitti_sequences()
        elif self.dataset_name == 'waymo_seg':
            return self._build_waymo_seg_sequences()
        elif self.dataset_name == 'unrealstereo4k':
            return self._build_unrealstereo4k_sequences()
        elif self.dataset_name == 'urbansyn':
            return self._build_urbansyn_sequences()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def _build_eth3d_sequences(self):
        """Build ETH3D sequences

        Sequence selection (matching FlashDepth's eth3d_dataset.py):
        - 'val' split: Use first 8 scenes only (filter out scenes[8:])
        - 'test' split: Use all scenes (no filtering)
        """
        scenes_path = self.dataset_path
        sequences = []

        # Get all scene directories
        all_scenes = sorted([s for s in os.listdir(scenes_path)
                           if os.path.isdir(os.path.join(scenes_path, s))])

        # Exclude multi_view_training_dslr_undistorted if present
        all_scenes = [s for s in all_scenes if s != 'multi_view_training_dslr_undistorted']

        # Filter scenes based on split (matching FlashDepth)
        if self.split == 'val':
            # FlashDepth filters out scenes[8:], so use first 8 scenes only
            scenes = all_scenes[:8]
            logger.info(f"ETH3D val split: Using first 8 scenes (for validation)")
        else:
            # 'test' split or other: Use all scenes (no filtering)
            scenes = all_scenes
            logger.info(f"ETH3D {self.split} split: Using all {len(scenes)} scenes")

        for scene in scenes:
            scene_path = os.path.join(scenes_path, scene)
            rgb_path = os.path.join(scene_path, 'images', 'dslr_images')
            depth_path = os.path.join(scene_path, 'ground_truth_depth', 'dslr_images')
            cameras_file = os.path.join(scene_path, 'dslr_calibration_undistorted', 'cameras.txt')

            if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                continue

            # Get sorted image files
            img_files = sorted([f for f in os.listdir(rgb_path) if f.endswith('.JPG')],
                             key=lambda x: int(x.split('DSC_')[1].split('.JPG')[0]))

            # Build sequence
            sequence = []
            for img_file in img_files[:self.video_length]:
                sequence.append({
                    'image': os.path.join(rgb_path, img_file),
                    'depth': os.path.join(depth_path, img_file),
                    'cameras_file': cameras_file,
                    'scene_name': scene,
                    'img_name': img_file
                })

            if len(sequence) > 0:
                sequences.append(sequence)

        return sequences

    def _build_kitti_sequences(self):
        """Build KITTI sequences (placeholder)"""
        # TODO: Implement KITTI sequence building
        logger.warning("KITTI dataset not yet implemented")
        return []

    def _build_sintel_sequences(self):
        """
        Build Sintel sequences

        Sequence selection (matching FlashDepth's sintel_dataset.py):
        - 'val' split: Exclude first 8 scenes (use remaining scenes for validation)
        - Other splits: Use all scenes
        """
        scenes_path = os.path.join(self.dataset_path, 'images', 'training', 'clean')
        sequences = []

        if not os.path.exists(scenes_path):
            logger.warning(f"Sintel path not found: {scenes_path}")
            return []

        # Get all scene directories
        all_scenes = sorted([s for s in os.listdir(scenes_path)
                           if os.path.isdir(os.path.join(scenes_path, s))])

        # Filter scenes based on split
        if self.split == 'val':
            scenes = all_scenes[8:]  # Exclude first 8 scenes
            logger.info(f"Sintel val split: Using last {len(scenes)} scenes (excluding first 8)")
        else:
            scenes = all_scenes
            logger.info(f"Sintel: Using all {len(scenes)} scenes")

        for scene in scenes:
            scene_path = os.path.join(scenes_path, scene)
            depth_path = scene_path.replace('images/training/clean', 'depth/training/depth')

            if not os.path.exists(depth_path):
                continue

            # Get sorted image files
            img_files = sorted([f for f in os.listdir(scene_path) if f.endswith('.png')],
                             key=lambda x: int(x.split('_')[1].split('.')[0]))

            # Build sequence
            sequence = []
            for img_file in img_files[:self.video_length]:
                depth_file = img_file.replace('.png', '.dpt')
                sequence.append({
                    'image': os.path.join(scene_path, img_file),
                    'depth': os.path.join(depth_path, depth_file),
                    'scene_name': scene,
                    'img_name': img_file
                })

            if len(sequence) > 0:
                sequences.append(sequence)

        return sequences

    def _build_scannet_sequences(self):
        """Build ScanNet sequences (placeholder)"""
        logger.warning("ScanNet dataset not yet implemented")
        return []

    def _build_tartanair_sequences(self):
        """Build TartanAir sequences (placeholder)"""
        logger.warning("TartanAir dataset not yet implemented")
        return []

    def _build_bonn_sequences(self):
        """Build Bonn sequences (placeholder)"""
        logger.warning("Bonn dataset not yet implemented")
        return []

    def _build_nyu_sequences(self):
        """
        Build NYU Depth V2 sequences

        Expects preprocessed structure:
        nyuv2_preprocessed/val/
            seq_000/
                0000/
                    rgb.png
                    depth.png
                0001/...
            seq_001/...
        """
        sequences = []
        nyu_root = os.path.join(self.data_root, 'nyuv2_preprocessed', self.split)

        if not os.path.exists(nyu_root):
            logger.warning(f"NYU preprocessed root not found: {nyu_root}")
            logger.warning("Please run: python scripts/preprocess_nyu.py")
            return []

        # Get all sequence directories
        seq_dirs = sorted([d for d in os.listdir(nyu_root)
                          if os.path.isdir(os.path.join(nyu_root, d)) and d.startswith('seq_')])

        logger.info(f"NYU: Found {len(seq_dirs)} sequences")

        for seq_name in seq_dirs:
            seq_path = os.path.join(nyu_root, seq_name)

            # Get all frame directories
            frame_dirs = sorted([d for d in os.listdir(seq_path)
                                if os.path.isdir(os.path.join(seq_path, d))],
                               key=lambda x: int(x))

            # Build sequence
            sequence = []
            for frame_dir in frame_dirs[:self.video_length]:
                frame_path = os.path.join(seq_path, frame_dir)
                rgb_path = os.path.join(frame_path, 'rgb.png')
                depth_path = os.path.join(frame_path, 'depth.png')

                if os.path.exists(rgb_path) and os.path.exists(depth_path):
                    sequence.append({
                        'image': rgb_path,
                        'depth': depth_path,
                        'scene_name': seq_name,
                        'img_name': frame_dir
                    })

            if len(sequence) > 0:
                sequences.append(sequence)

        return sequences

    def _build_waymo_seg_sequences(self):
        """
        Build Waymo segmentation sequences

        Note: waymo_seg uses validation split with specific sequences
        """
        sequences = []
        waymo_root = os.path.join(self.data_root, 'waymo_seg', self.split)

        if not os.path.exists(waymo_root):
            logger.warning(f"Waymo root not found: {waymo_root}")
            return []

        # Get all sequence directories
        all_seq_dirs = sorted([d for d in os.listdir(waymo_root)
                              if os.path.isdir(os.path.join(waymo_root, d)) and d.startswith('segment-')])

        # For validation split: use exactly these 8 sequences
        if self.split == 'val':
            val_sequence_names = [
                'segment-10017090168044687777_6380_000_6400_000',
                'segment-10023947602400723454_1120_000_1140_000',
                'segment-1005081002024129653_5313_150_5333_150',
                'segment-10061305430875486848_1080_000_1100_000',
                'segment-10072140764565668044_4060_000_4080_000',
                'segment-10072231702153043603_5725_000_5745_000',
                'segment-10075870402459732738_1060_000_1080_000',
                'segment-10094743350625019937_3420_000_3440_000',
            ]
            seq_name_to_dir = {d: d for d in all_seq_dirs}
            seq_dirs = [seq_name_to_dir[name] for name in val_sequence_names if name in seq_name_to_dir]
            logger.info(f"Waymo val split: Using {len(seq_dirs)} specified sequences")
        else:
            seq_dirs = all_seq_dirs
            logger.info(f"Waymo: Using all {len(seq_dirs)} sequences")

        for seq_name in seq_dirs:
            seq_dir = os.path.join(waymo_root, seq_name)
            camera_dir = os.path.join(seq_dir, 'FRONT')
            rgb_dir = os.path.join(camera_dir, 'rgb', 'original')
            depth_dir = os.path.join(camera_dir, 'depth')

            if not os.path.exists(rgb_dir) or not os.path.exists(depth_dir):
                continue

            # Get sorted files
            rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.jpg')])

            # Build sequence
            sequence = []
            for rgb_file in rgb_files[:self.video_length]:
                frame_idx = int(rgb_file.split('.')[0])
                depth_file = f'{frame_idx:04d}.npy'
                sequence.append({
                    'image': os.path.join(rgb_dir, rgb_file),
                    'depth': os.path.join(depth_dir, depth_file),
                    'scene_name': seq_name,
                    'img_name': rgb_file
                })

            if len(sequence) > 0:
                sequences.append(sequence)

        return sequences

    def _build_vkitti_sequences(self):
        """
        Build Virtual KITTI 2 sequences

        Structure:
        vkitti/
            Scene01/
                clone/
                    frames/
                        rgb/Camera_0/rgb_00000.jpg
                        depth/Camera_0/depth_00000.png
                        classSegmentation/Camera_0/classgt_00000.png
                overcast/...
                rain/...
            Scene02/...
        """
        sequences = []
        vkitti_root = os.path.join(self.data_root, 'vkitti')

        if not os.path.exists(vkitti_root):
            logger.warning(f"VKITTI root not found: {vkitti_root}")
            return []

        # Get all scene directories (Scene01, Scene02, ...)
        scene_dirs = sorted([d for d in os.listdir(vkitti_root)
                           if os.path.isdir(os.path.join(vkitti_root, d)) and d.startswith('Scene')])

        logger.info(f"VKITTI: Found {len(scene_dirs)} scenes")

        # Condition types in VKITTI2
        all_conditions = ['clone', 'overcast', 'sunset', 'morning', 'rain', 'fog',
                         '15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right']

        for scene_name in scene_dirs:
            scene_path = os.path.join(vkitti_root, scene_name)

            # Get available conditions for this scene
            available_conditions = [c for c in all_conditions
                                   if os.path.isdir(os.path.join(scene_path, c))]

            # Filter by only_clone flag
            if self.only_clone:
                available_conditions = ['clone'] if 'clone' in available_conditions else []

            for condition in available_conditions:
                condition_path = os.path.join(scene_path, condition, 'frames')
                rgb_dir = os.path.join(condition_path, 'rgb', 'Camera_0')
                depth_dir = os.path.join(condition_path, 'depth', 'Camera_0')
                seg_dir = os.path.join(condition_path, 'classSegmentation', 'Camera_0') if self.objwise_enabled else None

                if not os.path.exists(rgb_dir) or not os.path.exists(depth_dir):
                    continue

                # Check if segmentation is available when required
                if self.objwise_enabled:
                    if os.path.exists(seg_dir):
                        logger.info(f"[VKITTI DEBUG] Segmentation directory found: {seg_dir}")
                    else:
                        logger.warning(f"[VKITTI DEBUG] Object-wise enabled but no segmentation found: {seg_dir}")
                        continue

                # Get sorted RGB files
                rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.startswith('rgb_') and f.endswith('.jpg')])

                # Build sequence
                sequence = []
                seg_count = 0
                for rgb_file in rgb_files[:self.video_length]:
                    # Extract frame number
                    frame_num = int(rgb_file.split('_')[1].split('.')[0])
                    depth_file = f'depth_{frame_num:05d}.png'
                    seg_file = f'classgt_{frame_num:05d}.png' if self.objwise_enabled else None

                    depth_path = os.path.join(depth_dir, depth_file)
                    seg_path = os.path.join(seg_dir, seg_file) if self.objwise_enabled else None

                    # Check if required files exist
                    if not os.path.exists(depth_path):
                        continue
                    if self.objwise_enabled and not os.path.exists(seg_path):
                        if seg_count == 0:  # Only log first missing file
                            logger.warning(f"[VKITTI DEBUG] Segmentation file not found: {seg_path}")
                        continue

                    if self.objwise_enabled and os.path.exists(seg_path):
                        seg_count += 1

                    frame_info = {
                        'image': os.path.join(rgb_dir, rgb_file),
                        'depth': depth_path,
                        'scene_name': f'{scene_name}_{condition}',
                        'img_name': rgb_file,
                        'condition': condition
                    }

                    if self.objwise_enabled:
                        frame_info['segmentation'] = seg_path

                    sequence.append(frame_info)

                if len(sequence) > 0:
                    if self.objwise_enabled:
                        logger.info(f"[VKITTI DEBUG] Built sequence {scene_name}_{condition}: {len(sequence)} frames, {seg_count} with segmentation")
                    sequences.append(sequence)

        if self.only_clone:
            logger.info(f"VKITTI: Created {len(sequences)} sequences (clone condition only)")
        else:
            logger.info(f"VKITTI: Created {len(sequences)} sequences (all conditions)")
        return sequences

    def _build_unrealstereo4k_sequences(self):
        """
        Build UnrealStereo4K sequences

        If unrealstereo4k_seq_list is specified, only load those sequence numbers (0-8).
        """
        sequences = []
        unreal_root = os.path.join(self.data_root, 'unrealstereo4k')

        if not os.path.exists(unreal_root):
            logger.warning(f"UnrealStereo4K root not found: {unreal_root}")
            return []

        # Get all scene directories
        all_scenes = sorted([s for s in os.listdir(unreal_root)
                           if os.path.isdir(os.path.join(unreal_root, s))])

        # Filter by sequence numbers if specified
        if self.unrealstereo4k_seq_list is not None:
            selected_scenes = []
            for seq_idx in self.unrealstereo4k_seq_list:
                if seq_idx < 0 or seq_idx >= len(all_scenes):
                    logger.error(f"UnrealStereo4K seq {seq_idx} out of range (0-{len(all_scenes)-1})")
                    continue
                selected_scenes.append(all_scenes[seq_idx])

            if len(selected_scenes) == 0:
                logger.error(f"No valid sequences in seq_list: {self.unrealstereo4k_seq_list}")
                return []

            all_scenes = selected_scenes
            logger.info(f"UnrealStereo4K: Using sequences {self.unrealstereo4k_seq_list}: {all_scenes}")
        else:
            logger.info(f"UnrealStereo4K: Using all {len(all_scenes)} scenes")

        for scene in all_scenes:
            scene_path = os.path.join(unreal_root, scene)
            rgb_path = os.path.join(scene_path, 'Image0')
            depth_path = os.path.join(scene_path, 'Disp0')

            if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                continue

            # Get sorted image files
            img_files = sorted([f for f in os.listdir(rgb_path) if f.endswith('.png')],
                             key=lambda x: int(os.path.basename(x).split('.png')[0]))

            # Build sequence
            sequence = []
            for img_file in img_files[:self.video_length]:
                depth_file = img_file.replace('.png', '.npy')
                sequence.append({
                    'image': os.path.join(rgb_path, img_file),
                    'depth': os.path.join(depth_path, depth_file),
                    'scene_name': scene,
                    'img_name': img_file
                })

            if len(sequence) > 0:
                sequences.append(sequence)

        return sequences

    def _build_urbansyn_sequences(self):
        """
        Build UrbanSyn sequences
        """
        sequences = []
        urbansyn_root = os.path.join(self.data_root, 'urbansyn')

        if not os.path.exists(urbansyn_root):
            logger.warning(f"UrbanSyn root not found: {urbansyn_root}")
            return []

        rgb_dir = os.path.join(urbansyn_root, 'rgb')
        depth_dir = os.path.join(urbansyn_root, 'depth')

        if not os.path.exists(rgb_dir) or not os.path.exists(depth_dir):
            logger.warning(f"UrbanSyn rgb or depth directory not found")
            return []

        # Get all RGB files (up to 1000 frames)
        all_rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])[:1000]

        # Extract frame numbers and check if depth exists
        available_frames = []
        for f in all_rgb_files:
            frame_num = int(f.split('_')[1].split('.')[0])
            depth_file = f'depth_{frame_num:04d}.exr'
            if os.path.exists(os.path.join(depth_dir, depth_file)):
                available_frames.append(frame_num)

        logger.info(f"UrbanSyn: Found {len(available_frames)} frames")

        # Create sequences (consecutive frames)
        for i in range(0, len(available_frames) - self.video_length + 1):
            frames = available_frames[i:i + self.video_length]
            # Check if frames are consecutive
            if frames[-1] - frames[0] == self.video_length - 1:
                sequence = []
                for frame_num in frames:
                    sequence.append({
                        'image': os.path.join(rgb_dir, f'rgb_{frame_num:04d}.png'),
                        'depth': os.path.join(depth_dir, f'depth_{frame_num:04d}.exr'),
                        'scene_name': f'urbansyn_{frame_num:04d}',
                        'img_name': f'rgb_{frame_num:04d}.png'
                    })
                sequences.append(sequence)

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Returns a sequence of frames at ORIGINAL resolution

        Note: For datasets with variable-sized images within a sequence (e.g., ETH3D),
        all images are resized to match the first image's size.

        Returns:
            dict with keys:
                'images': Tensor [T, 3, H, W] - RGB images (0-1 normalized)
                'depths': Tensor [T, H, W] - GT depth in meters
                'intrinsics': Tensor [T, 4] - Camera intrinsics [fx, fy, cx, cy]
                'scene_name': str - Scene identifier
        """
        sequence = self.sequences[idx]

        # Apply chunk_size to limit memory usage
        if self.chunk_size and self.chunk_size < len(sequence):
            max_frames = self.chunk_size
            logger.info(f"Applying chunk_size: loading {max_frames}/{len(sequence)} frames from sequence {idx}")
        else:
            max_frames = len(sequence)

        # Warning for large sequences
        if max_frames > 200:
            logger.warning(f"Loading large sequence with {max_frames} frames at high resolution. "
                          f"This may take several minutes and require significant memory. "
                          f"Consider using --vid-len 100 to reduce sequence length.")

        images = []
        depths = []
        intrinsics_list = []
        segmentations = [] if self.objwise_enabled else None

        # First pass: load frames (respecting chunk_size)
        loaded_frames = []
        for frame_idx, frame in enumerate(sequence[:max_frames]):
            # Progress indicator for large sequences
            if max_frames > 200 and frame_idx % 100 == 0:
                logger.info(f"Loading sequence {idx}: {frame_idx}/{max_frames} frames...")
            try:
                # Load image at ORIGINAL resolution
                image = self._load_image(frame['image'])  # [3, H, W], 0-1 range

                # Load depth at ORIGINAL resolution
                depth = self._load_depth(frame['depth'], frame)  # [H, W], meters

                # Load intrinsics
                intrinsics = self._load_intrinsics(frame)  # [4] - fx, fy, cx, cy

                frame_data = {
                    'image': image,
                    'depth': depth,
                    'intrinsics': intrinsics
                }

                # Load segmentation if object-wise mode
                if self.objwise_enabled and 'segmentation' in frame:
                    seg = self._load_segmentation(frame['segmentation'])  # [H, W], class IDs
                    frame_data['segmentation'] = seg
                    if frame_idx == 0 and self.dataset_name == 'vkitti':  # Debug log for first frame only
                        logger.info(f"[VKITTI DEBUG] Loaded segmentation for frame 0: shape={seg.shape}, unique_classes={len(torch.unique(seg))}")

                loaded_frames.append(frame_data)

            except Exception as e:
                logger.warning(f"Error loading frame: {e}")
                continue

        if len(loaded_frames) == 0:
            logger.warning(f"No valid frames in sequence {idx}")
            return None

        # Get target size from first image
        target_H, target_W = loaded_frames[0]['image'].shape[1:]

        # Second pass: resize if needed and collect
        import torch.nn.functional as F

        for i, frame_data in enumerate(loaded_frames):
            image = frame_data['image']
            depth = frame_data['depth']
            intrinsics = frame_data['intrinsics']
            seg = frame_data.get('segmentation', None)

            img_H, img_W = image.shape[1:]
            depth_H, depth_W = depth.shape[0:]

            # Track original intrinsics for scaling
            orig_fx, orig_fy, orig_cx, orig_cy = intrinsics

            # Resize image if size doesn't match target
            if img_H != target_H or img_W != target_W:
                logger.debug(f"Resizing image {i} from {img_H}x{img_W} to {target_H}x{target_W}")

                image = F.interpolate(
                    image.unsqueeze(0),  # [1, 3, H, W]
                    size=(target_H, target_W),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)  # [3, H, W]

                # Scale intrinsics based on image resize
                orig_fx = orig_fx * (target_W / img_W)
                orig_fy = orig_fy * (target_H / img_H)
                orig_cx = orig_cx * (target_W / img_W)
                orig_cy = orig_cy * (target_H / img_H)

            # Resize depth if size doesn't match target
            # Note: ETH3D depth is always 4032x6048, but images vary in size
            if depth_H != target_H or depth_W != target_W:
                logger.debug(f"Resizing depth {i} from {depth_H}x{depth_W} to {target_H}x{target_W}")

                depth = F.interpolate(
                    depth.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
                    size=(target_H, target_W),
                    mode='nearest'
                ).squeeze(0).squeeze(0)  # [H, W]

            # Resize segmentation if needed (use nearest for class labels)
            if seg is not None:
                seg_H, seg_W = seg.shape[0:]
                if seg_H != target_H or seg_W != target_W:
                    logger.debug(f"Resizing segmentation {i} from {seg_H}x{seg_W} to {target_H}x{target_W}")
                    seg = F.interpolate(
                        seg.unsqueeze(0).unsqueeze(0).float(),  # [1, 1, H, W]
                        size=(target_H, target_W),
                        mode='nearest'
                    ).squeeze(0).squeeze(0).long()  # [H, W]

            # Update intrinsics
            intrinsics = torch.tensor([orig_fx, orig_fy, orig_cx, orig_cy])

            images.append(image)
            depths.append(depth)
            intrinsics_list.append(intrinsics)

            if seg is not None:
                segmentations.append(seg)

        # Extract dataset and scene information
        scene_name = sequence[0]['scene_name']
        dataset_scene = f"{self.dataset_name}/{scene_name}"

        # Extract focal lengths from intrinsics [T, 4] -> [T]
        intrinsics_tensor = torch.stack(intrinsics_list)  # [T, 4]
        focal_lengths_actual = intrinsics_tensor[:, 0]  # [T] - fx values

        batch = {
            'images': torch.stack(images),  # [T, 3, H, W]
            'depths': torch.stack(depths),  # [T, H, W]
            'intrinsics': intrinsics_tensor,  # [T, 4]
            'focal_lengths': focal_lengths_actual,  # [T] - actual fx for compatibility
            'focal_lengths_actual': focal_lengths_actual,  # [T] - actual fx
            'scene_name': scene_name,
            'dataset_name': dataset_scene  # e.g., 'eth3d/pipes'
        }

        # Add segmentations if object-wise mode
        if self.objwise_enabled and len(segmentations) > 0:
            batch['segmentations'] = torch.stack(segmentations)  # [T, H, W]
            if self.dataset_name == 'vkitti':
                logger.info(f"[VKITTI DEBUG] Added segmentations to batch: shape={batch['segmentations'].shape}")
        elif self.objwise_enabled and len(segmentations) == 0:
            if self.dataset_name == 'vkitti':
                logger.warning(f"[VKITTI DEBUG] Object-wise enabled but no segmentations loaded for sequence {idx}")

        return batch

    def _load_image(self, path):
        """Load RGB image at original resolution"""
        # Load with PIL
        img = Image.open(path).convert('RGB')

        # Convert to numpy
        img_np = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3], 0-1 range

        # Convert to torch (CHW format)
        img_torch = torch.from_numpy(img_np).permute(2, 0, 1).float()  # [3, H, W]

        return img_torch

    def _load_depth(self, path, frame_info):
        """Load depth map at original resolution"""
        if self.dataset_name == 'eth3d':
            return self._load_eth3d_depth(path)
        elif self.dataset_name == 'kitti':
            return self._load_kitti_depth(path)
        elif self.dataset_name == 'sintel':
            return self._load_sintel_depth(path)
        elif self.dataset_name == 'nyu':
            return self._load_nyu_depth(path)
        elif self.dataset_name == 'vkitti':
            return self._load_vkitti_depth(path)
        elif self.dataset_name == 'waymo_seg':
            return self._load_waymo_depth(path, frame_info)
        elif self.dataset_name == 'unrealstereo4k':
            return self._load_unrealstereo4k_depth(path)
        elif self.dataset_name == 'urbansyn':
            return self._load_urbansyn_depth(path)
        else:
            raise NotImplementedError(f"Depth loading not implemented for {self.dataset_name}")

    def _load_eth3d_depth(self, path):
        """
        Load ETH3D depth (binary float32 file)

        ETH3D stores NORMAL depth (m) in binary float32 format at 6048x4032.
        We return it as-is for metric depth evaluation.

        Returns:
            torch.Tensor: Depth in meters [H, W]
        """
        # ETH3D depth is stored as binary float32 at 6048x4032
        w, h = 6048, 4032

        depth = np.fromfile(path, dtype=np.float32)
        if depth.size != h * w:
            raise ValueError(f"ETH3D depth file size mismatch: {depth.size} != {h*w}")

        depth = depth.reshape((h, w))

        # Handle invalid values (infinity represents invalid depth)
        # Keep infinity as-is for proper masking in visualization
        # infinity will be filtered by valid_mask (depth > 0 & depth < MAX_DEPTH)

        # ETH3D files already store normal depth in meters
        # No conversion needed - just return as-is
        return torch.from_numpy(depth).float()  # [H, W] in meters

    def _load_kitti_depth(self, path):
        """Load KITTI depth (placeholder)"""
        raise NotImplementedError("KITTI depth loading not implemented")

    def _load_nyu_depth(self, path):
        """
        Load NYU Depth V2 depth (uint16 PNG)

        NYU preprocessed depth is stored as uint16 PNG in millimeters.
        Convert to meters for metric depth evaluation.

        Returns:
            torch.Tensor: Depth in meters [H, W]
        """
        # Load uint16 depth in millimeters
        depth_mm = cv2.imread(path, cv2.IMREAD_ANYDEPTH)

        if depth_mm is None:
            raise ValueError(f"Failed to load NYU depth from {path}")

        # Convert millimeters to meters
        depth_meters = depth_mm.astype(np.float32) / 1000.0

        # Handle invalid values
        invalid_mask = (depth_meters <= 0) | np.isinf(depth_meters) | np.isnan(depth_meters)
        depth_meters[invalid_mask] = 0

        return torch.from_numpy(depth_meters).float()  # [H, W] in meters

    def _load_vkitti_depth(self, path):
        """
        Load VKITTI2 depth (uint16 PNG)

        VKITTI2 stores depth as uint16 PNG in centimeters.
        Convert to meters for metric depth evaluation.

        Returns:
            torch.Tensor: Depth in meters [H, W]
        """
        # Load uint16 depth in centimeters
        depth_cm = cv2.imread(path, cv2.IMREAD_ANYDEPTH)

        if depth_cm is None:
            raise ValueError(f"Failed to load VKITTI depth from {path}")

        # Convert centimeters to meters
        depth_meters = depth_cm.astype(np.float32) / 100.0

        # Handle invalid values
        invalid_mask = (depth_meters <= 0) | np.isinf(depth_meters) | np.isnan(depth_meters)
        depth_meters[invalid_mask] = 0

        return torch.from_numpy(depth_meters).float()  # [H, W] in meters

    def _load_segmentation(self, path):
        """
        Load segmentation mask (class IDs)

        For VKITTI: classgt_XXXXX.png contains RGB color-coded segmentation
        For other datasets: may contain direct class IDs

        Returns:
            torch.Tensor: Segmentation mask [H, W] with class IDs
        """
        # Load segmentation mask
        seg = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if seg is None:
            raise ValueError(f"Failed to load segmentation from {path}")

        # VKITTI uses RGB color-coded segmentation
        if self.dataset_name == 'vkitti' and len(seg.shape) == 3:
            # RGB to class ID mapping for VKITTI
            # Based on https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/
            seg_rgb = seg[:, :, ::-1]  # BGR to RGB

            # Create class ID map
            class_map = np.zeros((seg.shape[0], seg.shape[1]), dtype=np.int64)

            # VKITTI2 RGB -> Class ID mapping
            # Source: VKITTI2 colors.txt (official homepage)
            # https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/
            rgb_to_class = {
                (210, 0, 200): 0,       # Terrain
                (90, 200, 255): 1,      # Sky
                (0, 199, 0): 2,         # Tree
                (90, 240, 0): 3,        # Vegetation
                (140, 140, 140): 4,     # Building
                (100, 60, 100): 5,      # Road
                (250, 100, 255): 6,     # GuardRail
                (255, 255, 0): 7,       # TrafficSign
                (200, 200, 0): 8,       # TrafficLight
                (255, 130, 0): 9,       # Pole
                (80, 80, 80): 10,       # Misc
                (160, 60, 60): 11,      # Truck (OBJECT)
                (255, 127, 80): 12,     # Car (OBJECT)
                (0, 139, 139): 13,      # Van (OBJECT)
                (0, 0, 0): 14,          # Undefined
            }

            # Convert RGB to class ID
            for rgb, class_id in rgb_to_class.items():
                mask = (seg_rgb[:, :, 0] == rgb[0]) & \
                       (seg_rgb[:, :, 1] == rgb[1]) & \
                       (seg_rgb[:, :, 2] == rgb[2])
                class_map[mask] = class_id

            return torch.from_numpy(class_map).long()  # [H, W]
        else:
            # Direct class ID (single channel)
            return torch.from_numpy(seg).long()  # [H, W]

    def _load_sintel_depth(self, path):
        """
        Load Sintel depth (.dpt binary file)

        Sintel stores NORMAL depth (m) in binary float32 format with header.
        Format: TAG_FLOAT, width, height, depth_values
        Returns normal depth in meters for comparison.
        """
        TAG_FLOAT = 202021.25

        with open(path, 'rb') as f:
            # Read header
            check = np.fromfile(f, dtype=np.float32, count=1)[0]
            if abs(check - TAG_FLOAT) > 0.01:
                raise ValueError(f"Wrong tag in depth file: {check} (expected {TAG_FLOAT})")

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

        # Mark sky (very large depth)
        sky_mask = depth > 1e4

        # Set invalid to 0 (will be masked in evaluation)
        depth[invalid_mask | sky_mask] = 0

        return torch.from_numpy(depth).float()  # [H, W] in meters

    def _load_waymo_depth(self, path, frame_info):
        """
        Load Waymo depth (sparse .npy file)

        Waymo stores sparse depth in format [N, 3]: [x, y, depth_meters]
        Convert to dense depth map and return normal depth in meters.
        """
        # Load sparse depth points
        sparse_depth = np.load(path)  # [N, 3]: [x, y, depth_meters]

        # Get original image size (Waymo is 1920×1280)
        orig_h, orig_w = 1280, 1920

        # Create dense depth map
        depth_map = np.zeros((orig_h, orig_w), dtype=np.float32)

        if len(sparse_depth) > 0:
            x_coords = sparse_depth[:, 0].astype(np.int32)
            y_coords = sparse_depth[:, 1].astype(np.int32)
            depth_values = sparse_depth[:, 2]

            # Filter coordinates within bounds
            valid_mask = (
                (x_coords >= 0) & (x_coords < orig_w) &
                (y_coords >= 0) & (y_coords < orig_h) &
                (depth_values > 0)
            )

            x_coords = x_coords[valid_mask]
            y_coords = y_coords[valid_mask]
            depth_values = depth_values[valid_mask]

            # Assign depth values (last write wins for duplicate coordinates)
            if len(x_coords) > 0:
                depth_map[y_coords, x_coords] = depth_values

        return torch.from_numpy(depth_map).float()  # [H, W] in meters

    def _load_unrealstereo4k_depth(self, path):
        """
        Load UnrealStereo4K depth (.npy file)

        UnrealStereo4K stores METRIC DEPTH (m) in .npy format.
        Despite variable names suggesting "disparity" in some places, the actual
        data is already in meters and should NOT be inverted.

        Confirmed by data analysis: median values 20-250m are reasonable metric depths,
        while 1/x conversion would give unrealistic sub-centimeter values.
        """
        # Load metric depth (already in meters)
        depth_meters = np.load(path)

        # Handle invalid values
        invalid_mask = np.logical_or.reduce((
            np.isinf(depth_meters),
            np.isnan(depth_meters),
            depth_meters <= 0
        ))

        # Set invalid to 0
        depth_meters[invalid_mask] = 0

        return torch.from_numpy(depth_meters).float()  # [H, W] in meters

    def _load_urbansyn_depth(self, path):
        """
        Load UrbanSyn depth (.exr file)

        UrbanSyn stores depth in EXR format, needs *1e5 to get meters.
        Returns normal depth in meters for comparison.
        """
        # OpenEXR support
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

        # Load EXR depth
        depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(np.float32)

        if depth is None:
            raise ValueError(f"Failed to load EXR depth from {path}")

        # Convert to meters (*1e5 according to UrbanSyn documentation)
        depth = depth * 1e5

        # Handle invalid values (depth should be positive)
        invalid_mask = (depth <= 0) | np.isinf(depth) | np.isnan(depth)
        depth[invalid_mask] = 0

        return torch.from_numpy(depth).float()  # [H, W] in meters

    def _load_intrinsics(self, frame_info):
        """Load camera intrinsics"""
        if self.dataset_name == 'eth3d':
            return self._load_eth3d_intrinsics(frame_info)
        elif self.dataset_name == 'nyu':
            return self._load_nyu_intrinsics(frame_info)
        elif self.dataset_name == 'vkitti':
            return self._load_vkitti_intrinsics(frame_info)
        elif self.dataset_name == 'unrealstereo4k':
            return self._load_unrealstereo4k_intrinsics(frame_info)
        elif self.dataset_name == 'urbansyn':
            return self._load_urbansyn_intrinsics(frame_info)
        elif self.dataset_name == 'sintel':
            return self._load_sintel_intrinsics(frame_info)
        elif self.dataset_name == 'waymo_seg':
            return self._load_waymo_intrinsics(frame_info)
        else:
            # Default: estimate from image size
            # This is a fallback - actual intrinsics should be loaded
            logger.warning(f"Using estimated intrinsics for {self.dataset_name}")
            return torch.tensor([1000.0, 1000.0, 320.0, 240.0])  # Dummy values

    def _load_eth3d_intrinsics(self, frame_info):
        """
        Load ETH3D camera intrinsics from cameras.txt

        Format: CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy
        """
        cameras_file = frame_info['cameras_file']
        img_name = frame_info['img_name']

        if not os.path.exists(cameras_file):
            logger.warning(f"Cameras file not found: {cameras_file}")
            # ETH3D default intrinsics for 6048x4032
            return torch.tensor([4251.0, 4251.0, 3024.0, 2016.0])

        try:
            with open(cameras_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue

                    parts = line.split()
                    if len(parts) >= 8:
                        # CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy
                        fx = float(parts[4])
                        fy = float(parts[5])
                        cx = float(parts[6])
                        cy = float(parts[7])

                        return torch.tensor([fx, fy, cx, cy])

            # If no matching camera found, use default
            logger.warning(f"No intrinsics found in {cameras_file}, using default")
            return torch.tensor([4251.0, 4251.0, 3024.0, 2016.0])

        except Exception as e:
            logger.warning(f"Error loading intrinsics: {e}")
            return torch.tensor([4251.0, 4251.0, 3024.0, 2016.0])

    def _load_nyu_intrinsics(self, frame_info):
        """
        Load NYU Depth V2 camera intrinsics (hardcoded fixed values)

        NYU Depth V2 uses fixed camera intrinsics for all frames:
        - fx = 518.86
        - fy = 519.47
        - cx = 325.58
        - cy = 253.74
        - Resolution: 640×480

        Returns:
            torch.Tensor: [fx, fy, cx, cy]
        """
        return torch.tensor([518.86, 519.47, 325.58, 253.74])

    def _load_vkitti_intrinsics(self, frame_info):
        """
        Load VKITTI2 camera intrinsics (hardcoded fixed values)

        VKITTI2 uses fixed camera intrinsics for all scenes and conditions:
        - fx = 725.0
        - fy = 725.0
        - cx = 620.5  (image_width - 1) / 2
        - cy = 187.0  (image_height - 1) / 2
        - Resolution: 1242×375

        Returns:
            torch.Tensor: [fx, fy, cx, cy]
        """
        return torch.tensor([725.0, 725.0, 620.5, 187.0])

    def _load_unrealstereo4k_intrinsics(self, frame_info):
        """
        Load UnrealStereo4K camera intrinsics from Extrinsics file (with caching)

        Format (line 1): fx skew cx 0 fy cy 0 0 1
        This represents the K matrix in row-major order.

        UnrealStereo4K resolution: 3840×2160 (4K)
        Note: All frames in the same scene share the same intrinsics, so we cache by scene_name.
        """
        scene_name = frame_info['scene_name']

        # Check cache first
        cache_key = f"unreal_{scene_name}"
        if cache_key in self._intrinsics_cache:
            return self._intrinsics_cache[cache_key]

        # Load from first frame's extrinsics (all frames in scene have same intrinsics)
        extrinsics_dir = os.path.join(self.data_root, 'unrealstereo4k', scene_name, 'Extrinsics0')
        extrinsics_file = os.path.join(extrinsics_dir, '00000.txt')

        if not os.path.exists(extrinsics_file):
            logger.warning(f"Extrinsics file not found: {extrinsics_file}")
            # Default intrinsics for UnrealStereo4K (3840×2160)
            intrinsics = torch.tensor([1920.0, 1920.0, 1920.0, 1080.0])
            self._intrinsics_cache[cache_key] = intrinsics
            return intrinsics

        try:
            with open(extrinsics_file, 'r') as f:
                lines = f.readlines()
                if len(lines) < 1:
                    raise ValueError("Extrinsics file is empty")

                # Parse first line: fx skew cx 0 fy cy 0 0 1
                k_values = list(map(float, lines[0].split()))
                if len(k_values) >= 9:
                    fx = k_values[0]
                    cx = k_values[2]
                    fy = k_values[4]
                    cy = k_values[5]

                    intrinsics = torch.tensor([fx, fy, cx, cy])
                    self._intrinsics_cache[cache_key] = intrinsics
                    return intrinsics
                else:
                    raise ValueError(f"Invalid K matrix format: {len(k_values)} values")

        except Exception as e:
            logger.warning(f"Error loading UnrealStereo4K intrinsics: {e}")
            intrinsics = torch.tensor([1920.0, 1920.0, 1920.0, 1080.0])
            self._intrinsics_cache[cache_key] = intrinsics
            return intrinsics

    def _load_urbansyn_intrinsics(self, frame_info):
        """
        Load UrbanSyn camera intrinsics from camera_metadata.json (with caching)

        UrbanSyn provides physical camera parameters:
        - focalLength_mm: focal length in millimeters
        - sensorWidth_mm: sensor width in millimeters
        - sensorHeight_mm: sensor height in millimeters

        We convert to pixel units using:
        fx = focal_length_mm / sensor_width_mm * image_width
        fy = focal_length_mm / sensor_height_mm * image_height

        UrbanSyn resolution: 2048×1024
        Note: All frames share the same intrinsics.
        """
        # Check cache first (all UrbanSyn frames have same intrinsics)
        cache_key = "urbansyn"
        if cache_key in self._intrinsics_cache:
            return self._intrinsics_cache[cache_key]

        metadata_file = os.path.join(self.data_root, 'urbansyn', 'camera_metadata.json')

        if not os.path.exists(metadata_file):
            logger.warning(f"Camera metadata not found: {metadata_file}")
            # Default intrinsics for UrbanSyn (2048×1024)
            intrinsics = torch.tensor([1731.0, 1731.0, 1024.0, 512.0])
            self._intrinsics_cache[cache_key] = intrinsics
            return intrinsics

        try:
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Extract camera parameters
            camera_params = metadata['parameters'][0]['Camera']
            focal_length_mm = camera_params['focalLength_mm']
            sensor_width_mm = camera_params['sensorWidth_mm']
            sensor_height_mm = camera_params['sensorHeight_mm']

            # UrbanSyn resolution
            image_width = 2048
            image_height = 1024

            # Convert to pixel units
            fx = focal_length_mm / sensor_width_mm * image_width
            fy = focal_length_mm / sensor_height_mm * image_height
            cx = image_width / 2.0
            cy = image_height / 2.0

            intrinsics = torch.tensor([fx, fy, cx, cy])
            self._intrinsics_cache[cache_key] = intrinsics
            return intrinsics

        except Exception as e:
            logger.warning(f"Error loading UrbanSyn intrinsics: {e}")
            intrinsics = torch.tensor([1731.0, 1731.0, 1024.0, 512.0])
            self._intrinsics_cache[cache_key] = intrinsics
            return intrinsics

    def _load_sintel_intrinsics(self, frame_info):
        """
        Load Sintel camera intrinsics from .cam file

        Sintel provides per-frame intrinsics in cam_data/training/camdata_left/*.cam files.
        Binary format:
          - TAG_FLOAT (float32): validation value (202021.25)
          - Intrinsic matrix M: 9 float64 values (3×3) for original 1024×436 resolution
          - Extrinsic matrix N: 12 float64 values (3×4) [not used]

        Sintel resolution: 1024×436
        """
        TAG_FLOAT = 202021.25

        img_path = frame_info['image']
        cam_path = img_path.replace('images/training/clean', 'cam_data/training/camdata_left').replace('.png', '.cam')

        # Cache by frame path
        cache_key = f"sintel_{cam_path}"
        if cache_key in self._intrinsics_cache:
            return self._intrinsics_cache[cache_key]

        if not os.path.exists(cam_path):
            logger.warning(f"Sintel camera file not found: {cam_path}")
            # Fallback: typical Sintel intrinsics (1024×436)
            intrinsics = torch.tensor([920.0, 920.0, 512.0, 218.0])
            self._intrinsics_cache[cache_key] = intrinsics
            return intrinsics

        try:
            with open(cam_path, 'rb') as f:
                # Read TAG_FLOAT validation value (float32)
                tag_val = np.fromfile(f, dtype=np.float32, count=1)[0]
                if abs(tag_val - TAG_FLOAT) > 0.01:
                    raise ValueError(f"Unexpected tag: {tag_val} (expected {TAG_FLOAT})")

                # Read intrinsic matrix M (9 float64 values, reshape to 3×3)
                M = np.fromfile(f, dtype=np.float64, count=9).reshape(3, 3)
                fx = float(M[0, 0])
                fy = float(M[1, 1])
                cx = float(M[0, 2])
                cy = float(M[1, 2])

                intrinsics = torch.tensor([fx, fy, cx, cy])
                self._intrinsics_cache[cache_key] = intrinsics
                return intrinsics

        except Exception as e:
            logger.warning(f"Error reading Sintel camera from {cam_path}: {e}")
            # Fallback
            intrinsics = torch.tensor([920.0, 920.0, 512.0, 218.0])
            self._intrinsics_cache[cache_key] = intrinsics
            return intrinsics

    def _load_waymo_intrinsics(self, frame_info):
        """
        Load Waymo camera intrinsics from intrinsics.npy file

        Waymo stores intrinsics per sequence (not per frame) in:
        waymo_seg/{split}/{sequence_name}/FRONT/intrinsics.npy

        Format: [fx, fy, cx, cy] - numpy array of shape (4,)
        Original Waymo resolution: 1920×1280
        """
        scene_name = frame_info['scene_name']

        # Cache by scene (all frames in scene share intrinsics)
        cache_key = f"waymo_{scene_name}"
        if cache_key in self._intrinsics_cache:
            return self._intrinsics_cache[cache_key]

        # Build path to intrinsics file
        intrinsics_file = os.path.join(
            self.data_root, 'waymo_seg', self.split, scene_name, 'FRONT', 'intrinsics.npy'
        )

        if not os.path.exists(intrinsics_file):
            logger.warning(f"Waymo intrinsics file not found: {intrinsics_file}")
            # Fallback: typical Waymo intrinsics (1920×1280)
            intrinsics = torch.tensor([2060.0, 2060.0, 960.0, 640.0])
            self._intrinsics_cache[cache_key] = intrinsics
            return intrinsics

        try:
            # Load numpy array [fx, fy, cx, cy]
            K = np.load(intrinsics_file)
            if K.shape != (4,):
                raise ValueError(f"Expected shape (4,), got {K.shape}")

            fx, fy, cx, cy = K
            intrinsics = torch.tensor([float(fx), float(fy), float(cx), float(cy)])
            self._intrinsics_cache[cache_key] = intrinsics
            return intrinsics

        except Exception as e:
            logger.warning(f"Error loading Waymo intrinsics from {intrinsics_file}: {e}")
            # Fallback
            intrinsics = torch.tensor([2060.0, 2060.0, 960.0, 640.0])
            self._intrinsics_cache[cache_key] = intrinsics
            return intrinsics


def comparison_collate_fn(batch):
    """
    Collate function for ComparisonDataset

    Filters out None values and returns proper batch format
    """
    # Filter out None values
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    # For comparison testing, we process one sequence at a time
    # So batch size should be 1
    if len(batch) > 1:
        logger.warning(f"ComparisonDataset expects batch_size=1, got {len(batch)}")
        batch = [batch[0]]

    item = batch[0]

    # Add batch dimension
    result = {
        'images': item['images'].unsqueeze(0),  # [1, T, 3, H, W]
        'depths': item['depths'].unsqueeze(0),  # [1, T, H, W]
        'intrinsics': item['intrinsics'].unsqueeze(0),  # [1, T, 4]
        'focal_lengths': item['focal_lengths'].unsqueeze(0),  # [1, T] - actual fx
        'focal_lengths_actual': item['focal_lengths_actual'].unsqueeze(0),  # [1, T] - actual fx
        'scene_name': item['scene_name'],
        'dataset_name': item['dataset_name']  # e.g., 'eth3d/pipes'
    }

    # Add segmentations if available (for object-wise evaluation)
    if 'segmentations' in item:
        result['segmentations'] = item['segmentations'].unsqueeze(0)  # [1, T, H, W]

    return result
