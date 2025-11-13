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

    def __init__(self, dataset_name, data_root, split='test', video_length=50):
        """
        Args:
            dataset_name: Name of dataset ('eth3d', 'kitti', 'sintel', etc.)
            data_root: Root directory containing datasets
            split: 'test' or 'val'
            video_length: Maximum sequence length
        """
        self.dataset_name = dataset_name.lower()
        self.data_root = data_root
        self.split = split
        self.video_length = video_length

        # Load dataset-specific configuration
        self.dataset_path = os.path.join(data_root, self.dataset_name)

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
        - 'val' split: Exclude first 8 scenes (use remaining scenes for validation)
        - 'test' split: Use last 5 scenes only
        - Other: Use all scenes
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
            # Exclude first 8 scenes (use last N scenes for validation)
            # This matches FlashDepth's validation split
            scenes = all_scenes[8:]
            logger.info(f"ETH3D val split: Using last {len(scenes)} scenes (excluding first 8)")
        elif self.split == 'test':
            # Use last 5 scenes for testing
            scenes = all_scenes[-5:]
            logger.info(f"ETH3D test split: Using last 5 scenes")
        else:
            scenes = all_scenes
            logger.info(f"ETH3D: Using all {len(scenes)} scenes")

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
        """Build NYU Depth V2 sequences (placeholder)"""
        logger.warning("NYU dataset not yet implemented")
        return []

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

    def _build_unrealstereo4k_sequences(self):
        """
        Build UnrealStereo4K sequences
        """
        sequences = []
        unreal_root = os.path.join(self.data_root, 'unrealstereo4k')

        if not os.path.exists(unreal_root):
            logger.warning(f"UnrealStereo4K root not found: {unreal_root}")
            return []

        # Get all scene directories
        all_scenes = sorted([s for s in os.listdir(unreal_root)
                           if os.path.isdir(os.path.join(unreal_root, s))])

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

        images = []
        depths = []
        intrinsics_list = []

        # First pass: load all frames
        loaded_frames = []
        for frame in sequence:
            try:
                # Load image at ORIGINAL resolution
                image = self._load_image(frame['image'])  # [3, H, W], 0-1 range

                # Load depth at ORIGINAL resolution
                depth = self._load_depth(frame['depth'], frame)  # [H, W], meters

                # Load intrinsics
                intrinsics = self._load_intrinsics(frame)  # [4] - fx, fy, cx, cy

                loaded_frames.append({
                    'image': image,
                    'depth': depth,
                    'intrinsics': intrinsics
                })

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

            # Update intrinsics
            intrinsics = torch.tensor([orig_fx, orig_fy, orig_cx, orig_cy])

            images.append(image)
            depths.append(depth)
            intrinsics_list.append(intrinsics)

        # Extract dataset and scene information
        scene_name = sequence[0]['scene_name']
        dataset_scene = f"{self.dataset_name}/{scene_name}"

        return {
            'images': torch.stack(images),  # [T, 3, H, W]
            'depths': torch.stack(depths),  # [T, H, W]
            'intrinsics': torch.stack(intrinsics_list),  # [T, 4]
            'scene_name': scene_name,
            'dataset_name': dataset_scene  # e.g., 'eth3d/pipes'
        }

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
        invalid_mask = depth == np.inf
        depth[invalid_mask] = 0

        # ETH3D files already store normal depth in meters
        # No conversion needed - just return as-is
        return torch.from_numpy(depth).float()  # [H, W] in meters

    def _load_kitti_depth(self, path):
        """Load KITTI depth (placeholder)"""
        raise NotImplementedError("KITTI depth loading not implemented")

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

        UnrealStereo4K stores disparity/inverse depth (1/m) in .npy format.
        Convert to normal depth (m) for comparison.
        """
        # Load inverse depth or disparity
        inverse_depth = np.load(path)

        # Handle invalid values
        invalid_mask = np.logical_or.reduce((
            np.isinf(inverse_depth),
            np.isnan(inverse_depth),
            inverse_depth <= 0
        ))

        # Convert inverse depth to normal depth
        depth = np.zeros_like(inverse_depth)
        valid_mask = ~invalid_mask
        depth[valid_mask] = 1.0 / inverse_depth[valid_mask]  # (1/m) → (m)

        # Set invalid to 0
        depth[invalid_mask] = 0

        return torch.from_numpy(depth).float()  # [H, W] in meters

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
    return {
        'images': item['images'].unsqueeze(0),  # [1, T, 3, H, W]
        'depths': item['depths'].unsqueeze(0),  # [1, T, H, W]
        'intrinsics': item['intrinsics'].unsqueeze(0),  # [1, T, 4]
        'scene_name': item['scene_name'],
        'dataset_name': item['dataset_name']  # e.g., 'eth3d/pipes'
    }
