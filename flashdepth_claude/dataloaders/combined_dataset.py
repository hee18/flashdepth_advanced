import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import numpy as np
import cv2
import logging
import math
import os
from os.path import join

from .depthanything_preprocess import _load_and_process_image, _load_and_process_depth
from .base_dataset_pairs import BaseDatasetPairs

# Import new NuScenesDepth dataset
from .nuscenes_dataset import NuScenesDepth

from utils.dataset_intrinsics import (
    get_intrinsics_info,
    get_fallback_fx,
    CANONICAL_FOCAL_LENGTH,
    ACTUAL_MAX_DEPTH
)


class CombinedDataset(Dataset):
    def __init__(self, root_dir, enable_dataset_flags, resolution=None, split='train',
                 video_length=8, seed=42, tmp_res=None, color_aug=False, strict_focal_length=True,
                 unreal4k_seq=None, limit_scenes=None, seq_list=None, skip_gt_canonicalization=False):
        '''
        enable_dataset_flags: list of datasets to use; e.g. ['spring', 'mvs-synth', 'urbansyn', 'eth3d', 'waymo', 'waymo_seg']

        # must have a couple of 2k, preferably dynamic datasets for testing
        current options: eth3d, waymo, waymo_seg, spring
        eth resolution: 6048x4032 (ratio 1.5)
        waymo/waymo_seg: 1920x1280 (ratio 1.5)
        spring: 1920x1080 (ratio 1.77)

        # training datasets
        # use a unique resolution for each dataset to preserve aspect ratio, and potentially do cropping where possible (e.g. pointodyssey)
        current: mvs-synth (1920x1080)->1960x1120, urbansyn (2048x1024)->2072x1064, 
            pointodyssey (960x540)-> 504x280 (bc not downsampling) / 1008x560 (enc-dec), dynamic replica (1280x720)->1288x728
        
        
        # might not be able to do vkitti and sintel because their height is too low (300/400); 
        # would only work if I can pass them through without the unet; but current experiments show that it doesn't work
        to add: vkitti, tartanair, sintel (sintel might have slightly weird depth values)


        There aren't many other 2k videos for training (spring is the only one I'm familiar with),
        so I'll mix in hd as well
        # 2k: mvs-synth, urbansyn, unreal4k; (maybe waymo, do an ablation; maybe hoi4d)
        # lower res + dynamic: dynamic replica, pointodyssey, sintel, vkitti, tartanair, bedlam
        for the lower res datasets, I can either just do the same 4x downsample through unet;
        or have a condition in the model to not pass them through the unet


        raw resolutions: pointodyssey is 960x540, spring is 1920x1080, sintel is 1024x436;
        dynamic replica: 1280x720; vkitti: 1242, 375; 
        tartanair: 640x480; hypersim: 1024x768; IRS: 960x540
        3dkenburs: 512x512; bedlam: 1280x720, synscapes: 1440x720
        mapillary: 640x360; nyu depth: 640x480; bonn 640x480

        res > full hd
        spring, eth3d, unrealsstereo4k (3840x2160), waymo, ARKitScenes, mvs-synth, phonedepth, urbansyn, hoi4d

        Args:
            skip_gt_canonicalization: If True, skip GT canonicalization for test split.
                                      GT will be returned in actual space (1/m).
                                      Useful for testing where only pred needs de-canonicalization.

        '''
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Store unreal4k_seq parameter
        self.unreal4k_seq = unreal4k_seq
        
        # Store skip_gt_canonicalization flag
        self.skip_gt_canonicalization = skip_gt_canonicalization

        cache_dir = './dataloaders/pairs_cache' if split != 'test' else None

        self.pairslist = {}
        self.depth_read_list = {}
        self.reshape_list = {}
        self.focal_length_getter_list = {}  # Store focal length getters for each dataset
        self.tmp_res = tmp_res
        self.strict_focal_length = strict_focal_length  # Validate focal lengths strictly


        for dataset_name in enable_dataset_flags:
            # NuScenes is added here
            if dataset_name == 'nuscenes':
                dataset = NuScenesDepth(root_dir, split, load_cache=cache_dir, limit_scenes=limit_scenes)
            # waymo_seg only has 'val' split, no 'test' split
            else:
                actual_split = 'val' if dataset_name == 'waymo_seg' and split == 'test' else split

                logging.info(f"[DEBUG combined_dataset] Loading dataset: {dataset_name}, split={actual_split}")

                # Pass unreal4k_seq to unreal4k dataset
                if dataset_name.lower() == 'unreal4k' and unreal4k_seq is not None:
                    dataset = BaseDatasetPairs.create(dataset_name, root_dir, actual_split, load_cache=cache_dir, unreal4k_seq=unreal4k_seq)
                else:
                    dataset = BaseDatasetPairs.create(dataset_name, root_dir, actual_split, load_cache=cache_dir)
            
            self.pairslist[dataset_name] = dataset.pairs
            self.depth_read_list[dataset_name] = dataset.depth_read
            self.reshape_list[dataset_name] = dataset.reshape_list
            logging.info(f"[DEBUG combined_dataset] Dataset {dataset_name} loaded: {len(dataset.pairs)} pairs/sequences")

            # Apply sequence filtering if seq_list is provided
            if seq_list is not None and split != 'train':
                original_len = len(self.pairslist[dataset_name])
                # Convert seq_list to list of ints (handle Hydra string formats)
                if isinstance(seq_list, str):
                    # Hydra may pass as string like "[5]" or "5" or "0,3,7"
                    import ast
                    try:
                        # Try parsing as Python literal (e.g., "[5]")
                        seq_list = ast.literal_eval(seq_list)
                    except (ValueError, SyntaxError):
                        # If that fails, try comma-separated (e.g., "0,3,7")
                        seq_list = [s.strip() for s in seq_list.split(',')]

                # Convert all elements to int
                seq_list_int = [int(i) for i in seq_list]
                # Filter sequences by indices in seq_list
                filtered_pairs = [self.pairslist[dataset_name][i] for i in seq_list_int if i < original_len]
                self.pairslist[dataset_name] = filtered_pairs
                logging.info(f"[CombinedDataset] Filtered {dataset_name}: {original_len} → {len(filtered_pairs)} sequences (seq_list={seq_list_int})")

            # Store focal length getter method if dataset has it, otherwise None
            if hasattr(dataset, 'get_focal_length'):
                self.focal_length_getter_list[dataset_name] = dataset.get_focal_length
            else:
                self.focal_length_getter_list[dataset_name] = None

        if resolution == 'base':
            if split == 'train':
                for dataset in self.reshape_list:
                    self.reshape_list[dataset]['resolution'] = (518, 518)
                    self.reshape_list[dataset]['crop_type'] = 'center'
                    if dataset in ['spring', 'mvs-synth']:
                        self.reshape_list[dataset]['resize_factor'] = 0.5
                    if dataset in ['pointodyssey']:
                        self.reshape_list[dataset]['resize_factor'] = 1.0
                    if dataset in ['dynamicreplica']:
                        self.reshape_list[dataset]['resize_factor'] = 0.75
         
            else:
                for dataset in self.reshape_list:
                    self.reshape_list[dataset]['crop_type'] = None
                    if dataset in ['eth3d', 'waymo', 'waymo_seg']:
                        self.reshape_list[dataset]['resolution'] = (784,518)
                    elif dataset in ['sintel']:
                        self.reshape_list[dataset]['resolution'] = (1022,434)
                    elif dataset in ['urbansyn']:
                        self.reshape_list[dataset]['resolution'] = (1036,518)
                    elif dataset in ['unreal4k']:
                        self.reshape_list[dataset]['resolution'] = (924,518)
                    elif dataset in ['vkitti']:
                        self.reshape_list[dataset]['resolution'] = (1246, 378) # 3.296 ratio, near original, 14x divisible
                    elif dataset in ['nuscenes']:
                        self.reshape_list[dataset]['resolution'] = (924, 518)
                    elif dataset in ['bonn']:
                        self.reshape_list[dataset]['resolution'] = (630, 476)  # 4:3 ratio, 14x divisible, original 640x480


        elif resolution == '2k':
            if split == 'train':
                for dataset in self.reshape_list:
                    self.reshape_list[dataset]['resolution'] = (1918, 1078)
                    self.reshape_list[dataset]['crop_type'] = 'random'
                    self.reshape_list[dataset]['stride'] = 2
            else:
                for dataset in self.reshape_list:
                    self.reshape_list[dataset]['crop_type'] = None
                    if dataset in ['eth3d', 'waymo', 'waymo_seg']:
                        self.reshape_list[dataset]['resolution'] = (1918,1274)
                    if dataset in ['sintel']:
                        self.reshape_list[dataset]['resolution'] = (1022,434)
                    if dataset in ['urbansyn']:
                        self.reshape_list[dataset]['resolution'] = (2044,1022)
                    if dataset in ['unreal4k']:
                        self.reshape_list[dataset]['resolution'] = (2044,1148)
                    if dataset in ['vkitti']:
                        self.reshape_list[dataset]['resolution'] = (1246, 378)  # 3.296 ratio, near original, 14x divisible
                    elif dataset in ['nuscenes']:  # Add NuScenes resolution
                        self.reshape_list[dataset]['resolution'] = (1596, 896)
                    elif dataset in ['bonn']:
                        self.reshape_list[dataset]['resolution'] = (630, 476)  # 4:3 ratio, 14x divisible, original 640x480

        else:
            raise ValueError(f"Resolution should be 'base' or '2k' for training")
        
        self.pairs = []


        for dataset_name in enable_dataset_flags:
            indices = list(range(len(self.pairslist[dataset_name])))
            self.pairs.extend([(dataset_name, i) for i in indices])
            logging.info(f"length of {dataset_name} for {split}: {len(self.pairslist[dataset_name])}")

        if split != 'train':    
            logging.info(f"enabled datasets for {split}: {enable_dataset_flags}")
            logging.info(f"length of combined dataset: {len(self.pairs)}")


        self.video_length = video_length
        self.split = split

    def _get_focal_length(self, dataset_idx, pair, image_shape):
        """
        Get focal length for a given pair from a dataset.

        Args:
            dataset_idx (str): Dataset name
            pair (dict): Data pair with 'image' and 'depth' paths
            image_shape (tuple): (H, W) image shape after preprocessing

        Returns:
            float: Focal length in pixels

        Raises:
            RuntimeError: If strict_focal_length=True and focal length cannot be retrieved
        """
        error_info = None  # Track error for strict mode

        # Try dataset-specific getter first
        if self.focal_length_getter_list[dataset_idx] is not None:
            try:
                # NuScenesDepth's get_item_data returns focal length directly,
                # but CombinedDataset expects BaseDatasetPairs.create to provide
                # a dataset object that has a get_focal_length method.
                # For NuScenes, the get_focal_length will be called on the NuScenesDepth object
                fx = self.focal_length_getter_list[dataset_idx](pair, image_shape)
                return fx
            except Exception as e:
                error_info = f"Dataset-specific getter failed: {e}"
                logging.warning(f"[{dataset_idx}] {error_info}")

        # Fallback: use central registry
        intrinsics_info = get_intrinsics_info(dataset_idx)

        if intrinsics_info is None:
            # No info available
            if self.strict_focal_length:
                raise RuntimeError(
                    f"\n{'='*80}\n"
                    f"FOCAL LENGTH ERROR\n"
                    f"{'='*80}\n"
                    f"Dataset: {dataset_idx}\n"
                    f"Image path: {pair.get('image', 'N/A')}\n"
                    f"Depth path: {pair.get('depth', 'N/A')}\n"
                    f"Image shape: {image_shape}\n"
                    f"Error: No intrinsics info available in registry\n"
                    f"Previous error: {error_info or 'N/A'}\n"
                    f"\nPlease check:\n"
                    f"1. Dataset intrinsics are correctly defined in utils/dataset_intrinsics.py\n"
                    f"2. Dataset implements get_focal_length() method\n"
                    f"3. Intrinsic files exist and are accessible\n"
                    f"{'='*80}\n"
                )
            fx = get_fallback_fx(image_shape[1])
            logging.debug(f"[{dataset_idx}] No intrinsics info, using fallback fx={fx:.1f}")
            return fx

        # Handle different intrinsic types
        intrinsic_type = intrinsics_info['type']

        if intrinsic_type == 'fixed':
            fx = intrinsics_info['fx']
            return fx

        elif intrinsic_type == 'computed':
            # Compute from formula (e.g., DynamicReplica: fx = width / 2)
            if 'formula' in intrinsics_info:
                width = image_shape[1]
                if dataset_idx in ['dynamicreplica', 'replica']:
                    fx = width / 2.0
                else:
                    # Generic fallback
                    fx = get_fallback_fx(width)
                return fx

        # For per_frame, per_sequence, per_image types:
        # Dataset should implement get_focal_length()
        # If not implemented and strict mode, raise error
        if self.strict_focal_length:
            raise RuntimeError(
                f"\n{'='*80}\n"
                f"FOCAL LENGTH ERROR\n"
                f"{'='*80}\n"
                f"Dataset: {dataset_idx}\n"
                f"Image path: {pair.get('image', 'N/A')}\n"
                f"Depth path: {pair.get('depth', 'N/A')}\n"
                f"Image shape: {image_shape}\n"
                f"Intrinsic type: {intrinsic_type}\n"
                f"Error: Dataset requires get_focal_length() implementation but it failed\n"
                f"Previous error: {error_info or 'N/A'}\n"
                f"\nPlease check:\n"
                f"1. Dataset correctly implements get_focal_length() method\n"
                f"2. Intrinsic files exist at expected paths\n"
                f"3. File format matches expected structure\n"
                f"{'='*80}\n"
            )

        fx = get_fallback_fx(image_shape[1])
        logging.warning(
            f"[{dataset_idx}] Type '{intrinsic_type}' requires dataset implementation, "
            f"using fallback fx={fx:.1f}"
        )
        return fx

    def _apply_canonical_transform(self, inverse_depth_actual, fx_actual,
                                   original_h, original_w,
                                   target_resolution, resize_factor=1.0):
        """
        Apply Metric3D-style canonical transformation to inverse depth.

        This follows Metric3D's principle: when resizing images, focal length should
        be scaled proportionally. If actual resize differs from theoretical resize
        (fx_ratio), GT depth must be corrected.

        Canonical space: fx=500 at target_resolution (e.g., 518×518 or 784×518)

        Args:
            inverse_depth_actual (np.ndarray or torch.Tensor): Inverse depth in actual space (1/m)
            fx_actual (float): Actual focal length in pixels at original resolution
            original_h (int): Original image height
            original_w (int): Original image width
            target_resolution (tuple): Target (height, width) after resize+crop
            resize_factor (float): Dataset-specific pre-resize factor (default: 1.0)

        Returns:
            tuple: (inverse_depth_canonical, fx_canonical, fx_actual, actual_valid_mask, fx_ratio, resize_ratio)
                - inverse_depth_canonical: Corrected inverse depth for canonical space (1/m)
                - fx_canonical: Canonical focal length (500.0)
                - fx_actual: Original actual focal length (for reference)
                - actual_valid_mask: Valid mask in actual space (<70m)
                - fx_ratio: Focal length ratio (CANONICAL_FOCAL_LENGTH / fx_actual)
                - resize_ratio: Total resize ratio (resize_factor × small_resize_ratio)
        """
        # Convert to numpy for computation
        is_torch = isinstance(inverse_depth_actual, torch.Tensor)
        if is_torch:
            inverse_np = inverse_depth_actual.cpu().numpy()
        else:
            inverse_np = inverse_depth_actual

        # Convert to normal depth (m) to compute actual space valid mask
        # Avoid division by zero (suppress warning)
        with np.errstate(divide='ignore', invalid='ignore'):
            depth_actual = np.where(inverse_np > 1e-8, 1.0 / inverse_np, 0.0)

        # Compute actual space valid mask: depth > 0 AND depth < 70m
        actual_valid_mask = (depth_actual > 0) & (depth_actual < ACTUAL_MAX_DEPTH)

        # Metric3D-style canonicalization
        # Step 1: Apply dataset-specific pre-resize
        pre_h = int(original_h * resize_factor)
        pre_w = int(original_w * resize_factor)

        # Step 2: Compute small_resize_ratio (shorter side matches target)
        # NOTE: target_resolution is stored as (W, H), not (H, W)!
        target_w, target_h = target_resolution  # Unpack as (W, H)
        small_resize_ratio = max(target_w / pre_w, target_h / pre_h)  # W→W, H→H

        # Step 3: Focal length ratio
        fx_ratio = CANONICAL_FOCAL_LENGTH / fx_actual  # 500 / fx_actual

        # Step 4: Total resize ratio (original → final)
        total_resize_ratio = resize_factor * small_resize_ratio

        # Step 5: Depth correction for inverse depth space
        # Theory: depth_metric_corrected = depth_actual × (actual_resize / theoretical_resize)
        #         theoretical_resize = fx_ratio (to match focal length change)
        #         actual_resize = total_resize_ratio
        # For inverse depth: inverse_corrected = inverse_actual × (total_resize_ratio / fx_ratio)
        depth_correction_ratio = total_resize_ratio / fx_ratio
        inverse_canonical_np = inverse_np * depth_correction_ratio

        # Convert back to torch if input was torch
        if is_torch:
            inverse_canonical = torch.from_numpy(inverse_canonical_np).float()
            actual_valid_mask = torch.from_numpy(actual_valid_mask).bool()
        else:
            inverse_canonical = inverse_canonical_np

        # Return fx_ratio and resize_ratio for visualization
        return inverse_canonical, CANONICAL_FOCAL_LENGTH, fx_actual, actual_valid_mask, fx_ratio, total_resize_ratio

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        if self.split == 'val':
            dataset_idx, scene_idx = self.pairs[idx]
            scene = self.pairslist[dataset_idx][scene_idx]

            # Apply video_length limit for validation to ensure consistent batch sizes
            if len(scene) > self.video_length:
                scene = scene[:self.video_length]

            images = []
            depths_canonical = []
            focal_lengths_canonical = []
            focal_lengths_actual = []
            actual_valid_masks = []
            fx_ratios = []  # NEW
            resize_ratios = []  # NEW

            # Get dataset-specific settings
            target_resolution = self.reshape_list[dataset_idx]['resolution']  # (H, W)
            resize_factor = self.reshape_list[dataset_idx].get('resize_factor', 1.0)

            for pair in scene:
                # Debug: check if pair is string or dict
                if isinstance(pair, str):
                    continue  # Skip invalid pairs
                elif not isinstance(pair, dict):
                    continue  # Skip non-dict pairs

                try:
                    image, _current_crop = _load_and_process_image(pair['image'], **self.reshape_list[dataset_idx])
                    depth_inverse_actual = self.depth_read_list[dataset_idx](pair['depth'], is_inverse=True) # Load inverse depth (1/m)
                    # Keep GT at ORIGINAL resolution (like original FlashDepth) 
                    # Prediction will be interpolated to GT resolution during validation

                    # Get focal length for this frame (at original resolution)
                    original_h, original_w = depth_inverse_actual.shape
                    fx_actual = self._get_focal_length(dataset_idx, pair, (original_h, original_w))

                    # Apply Metric3D-style canonical transformation (now returns 6 values)
                    depth_inverse_canonical, fx_canonical, fx_actual_returned, actual_valid_mask, fx_ratio, resize_ratio = self._apply_canonical_transform(
                        depth_inverse_actual, fx_actual, original_h, original_w, target_resolution, resize_factor
                    )

                    images.append(image)
                    # Convert to torch.Tensor if numpy array (for validation consistency)
                    if isinstance(depth_inverse_canonical, np.ndarray):
                        depth_inverse_canonical = torch.from_numpy(depth_inverse_canonical).float()
                    if isinstance(actual_valid_mask, np.ndarray):
                        actual_valid_mask = torch.from_numpy(actual_valid_mask).bool()
                    depths_canonical.append(depth_inverse_canonical) # Canonical inverse depth
                    focal_lengths_canonical.append(fx_canonical) # 500.0
                    focal_lengths_actual.append(fx_actual_returned) # Original fx for visualization
                    actual_valid_masks.append(actual_valid_mask) # Actual space mask (<70m)
                    fx_ratios.append(fx_ratio)  # NEW
                    resize_ratios.append(resize_ratio)  # NEW
                except Exception as e:
                    print(f"Error loading validation pair: {e}")
                    continue

            # Skip if no valid pairs found
            if len(images) == 0:
                print(f"Warning: No valid pairs found for validation idx {idx}, skipping")
                return None

            return_name = dataset_idx
            focal_lengths_canonical_tensor = torch.tensor(focal_lengths_canonical, dtype=torch.float32)
            focal_lengths_actual_tensor = torch.tensor(focal_lengths_actual, dtype=torch.float32)
            fx_ratio_tensor = torch.tensor(fx_ratios, dtype=torch.float32)  # NEW
            resize_ratio_tensor = torch.tensor(resize_ratios, dtype=torch.float32)  # NEW
            return (torch.stack(images).float(),
                    torch.stack(depths_canonical).float(),
                    focal_lengths_canonical_tensor,
                    focal_lengths_actual_tensor,
                    torch.stack(actual_valid_masks).bool(),
                    fx_ratio_tensor,  # NEW
                    resize_ratio_tensor,  # NEW
                    return_name)


        elif self.split == 'test':
            dataset_idx, scene_idx = self.pairs[idx]
            scene = self.pairslist[dataset_idx][scene_idx]

            # Apply video_length limit for test split to prevent memory issues
            if len(scene) > self.video_length:
                # Take the first video_length frames
                scene = scene[:self.video_length]

            images = []
            depths = []  # Renamed from depths_canonical (may be actual or canonical)
            focal_lengths_canonical = []
            focal_lengths_actual = []
            actual_valid_masks = []
            fx_ratios = []  # NEW
            resize_ratios = []  # NEW
            image_paths = []  # For FG-wise evaluation

            # Get dataset-specific settings
            target_resolution = self.reshape_list[dataset_idx]['resolution']  # (H, W)
            resize_factor = self.reshape_list[dataset_idx].get('resize_factor', 1.0)

            for pair in scene:
                image, _current_crop = _load_and_process_image(pair['image'], **self.reshape_list[dataset_idx])
                depth_inverse_actual = self.depth_read_list[dataset_idx](pair['depth'], is_inverse=True) # Load inverse depth (1/m)

                # Get focal length for this frame (at original resolution)
                original_h, original_w = depth_inverse_actual.shape
                fx_actual = self._get_focal_length(dataset_idx, pair, (original_h, original_w))

                if self.skip_gt_canonicalization:
                    # Skip GT canonicalization: return GT in actual space (1/m)
                    # Only pred needs de-canonicalization in test_gear5.py

                    # Convert inverse depth to normal depth for valid mask computation
                    is_torch = isinstance(depth_inverse_actual, torch.Tensor)
                    if is_torch:
                        depth_np = depth_inverse_actual.cpu().numpy()
                    else:
                        depth_np = depth_inverse_actual

                    with np.errstate(divide='ignore', invalid='ignore'):
                        depth_actual = np.where(depth_np > 1e-8, 1.0 / depth_np, 0.0)

                    # Compute actual space valid mask: depth > 0 AND depth < 70m
                    actual_valid_mask = (depth_actual > 0) & (depth_actual < ACTUAL_MAX_DEPTH)

                    # Convert to tensors
                    if isinstance(depth_inverse_actual, np.ndarray):
                        depth_inverse_tensor = torch.from_numpy(depth_inverse_actual).float()
                    else:
                        depth_inverse_tensor = depth_inverse_actual.float()
                    actual_valid_mask = torch.from_numpy(actual_valid_mask).bool()

                    # Still need fx_actual for pred de-canonicalization
                    # fx_ratio and resize_ratio are used for de-canonicalization
                    # Compute these for pred de-canonicalization
                    pre_h = int(original_h * resize_factor)
                    pre_w = int(original_w * resize_factor)
                    target_w, target_h = target_resolution
                    small_resize_ratio = max(target_w / pre_w, target_h / pre_h)
                    fx_ratio = CANONICAL_FOCAL_LENGTH / fx_actual
                    total_resize_ratio = resize_factor * small_resize_ratio

                    images.append(image)
                    image_paths.append(pair['image'])
                    depths.append(depth_inverse_tensor)  # Actual space inverse depth (1/m)
                    focal_lengths_canonical.append(CANONICAL_FOCAL_LENGTH)  # Still 500.0 for consistency
                    focal_lengths_actual.append(fx_actual)
                    actual_valid_masks.append(actual_valid_mask)
                    fx_ratios.append(fx_ratio)
                    resize_ratios.append(total_resize_ratio)
                else:
                    # Apply Metric3D-style canonical transformation (now returns 6 values)
                    depth_inverse_canonical, fx_canonical, fx_actual_returned, actual_valid_mask, fx_ratio, resize_ratio = self._apply_canonical_transform(
                        depth_inverse_actual, fx_actual, original_h, original_w, target_resolution, resize_factor
                    )

                    images.append(image)
                    image_paths.append(pair['image'])  # Store image path for FG-wise eval
                    # Convert to torch.Tensor if numpy array (for validation/test consistency)
                    if isinstance(depth_inverse_canonical, np.ndarray):
                        depth_inverse_canonical = torch.from_numpy(depth_inverse_canonical).float()
                    if isinstance(actual_valid_mask, np.ndarray):
                        actual_valid_mask = torch.from_numpy(actual_valid_mask).bool()
                    depths.append(depth_inverse_canonical)  # Canonical inverse depth
                    focal_lengths_canonical.append(fx_canonical)  # 500.0
                    focal_lengths_actual.append(fx_actual_returned)  # Original fx for visualization
                    actual_valid_masks.append(actual_valid_mask)  # Actual space mask (<70m)
                    fx_ratios.append(fx_ratio)
                    resize_ratios.append(resize_ratio)

            return_name = os.path.join(dataset_idx, pair['scene_name'])
            focal_lengths_canonical_tensor = torch.tensor(focal_lengths_canonical, dtype=torch.float32)
            focal_lengths_actual_tensor = torch.tensor(focal_lengths_actual, dtype=torch.float32)
            fx_ratio_tensor = torch.tensor(fx_ratios, dtype=torch.float32)  # NEW
            resize_ratio_tensor = torch.tensor(resize_ratios, dtype=torch.float32)  # NEW
            return (torch.stack(images).float(),
                    torch.stack(depths).float(),  # May be actual or canonical depending on skip_gt_canonicalization
                    focal_lengths_canonical_tensor,
                    focal_lengths_actual_tensor,
                    torch.stack(actual_valid_masks).bool(),
                    fx_ratio_tensor,  # NEW
                    resize_ratio_tensor,  # NEW
                    return_name,
                    image_paths)  # Return image paths for FG-wise eval


        # dataset_idx: i-th dataset; e.g. pointodyssey is 0, spring is 1...etc
        # pair_idx: the i-th (img, depth) pair in the dataset, for instance, pair_idx \in [0, 5000] in Spring
        dataset_idx, pair_idx = self.pairs[idx]  
        dataset_list = self.pairslist[dataset_idx]
        pair = dataset_list[pair_idx]


        scene_index = pair['scene_index']
        scene_length = pair['scene_length']
        stride = self.reshape_list[dataset_idx]['stride']


        # Check if we can go both forward and backward
        can_go_forward = scene_index + (self.video_length - 1) * stride <= scene_length - 1
        can_go_backward = scene_index >= (self.video_length - 1) * stride
        
        if can_go_forward and can_go_backward:
            # Randomly choose direction
            if torch.rand(1).item() > 0.5:
                sequence_indices = list(range(scene_index, scene_index + self.video_length * stride, stride))
            else:
                start_pos = scene_index - (self.video_length - 1) * stride
                sequence_indices = list(range(start_pos, scene_index + 1, stride))
        elif can_go_forward:
            # Only enough frames ahead
            sequence_indices = list(range(scene_index, scene_index + self.video_length * stride, stride))
        elif can_go_backward:
            # Must go backward
            start_pos = scene_index - (self.video_length - 1) * stride
            sequence_indices = list(range(start_pos, scene_index + 1, stride))
        else:
            # Can't go either way - use remaining frames forward then wrap around backward
            remaining_forward = scene_length - scene_index
            remaining_forward_frames = math.ceil(remaining_forward / stride)
            remaining_needed = max(self.video_length - remaining_forward_frames, 0)

            # Get forward frames
            sequence_indices = list(range(scene_index, scene_length, stride))

            # Add backward frames if needed
            if remaining_needed > 0:
                start = scene_index - remaining_needed * stride
                backward_indices = list(range(start, scene_index, stride))
                sequence_indices.extend(backward_indices)

            # Final safeguard to enforce video_length
            if len(sequence_indices) > self.video_length:
                sequence_indices = sequence_indices[:self.video_length]
            elif len(sequence_indices) < self.video_length:
                # repeat the last frame 
                sequence_indices.append(sequence_indices[-1])
        
        # Get the base offset for this scene in the flat list
        scene_start_idx = pair_idx - scene_index  # This gives us the index where this scene starts
        
        
        
        # Load all frames in sequence
        images = []
        depths_canonical = []
        focal_lengths_canonical = []
        focal_lengths_actual = []
        actual_valid_masks = []
        fx_ratios = []  # NEW: Focal length ratios
        resize_ratios = []  # NEW: Total resize ratios

        # Get dataset-specific settings
        target_resolution = self.reshape_list[dataset_idx]['resolution']  # (H, W)
        resize_factor = self.reshape_list[dataset_idx].get('resize_factor', 1.0)

        # Transform scene-relative indices to dataset-relative indices
        sequence_indices = [scene_start_idx + s for s in sequence_indices]
        for seq_i, seq_idx in enumerate(sequence_indices):
            try:
                # pair = self.pairslist[dataset_idx][seq_idx]
                pair = dataset_list[seq_idx]
            except Exception as e:
                print("dataset, pair idx: ", dataset_idx, pair_idx)
                print(f"seq indices: {sequence_indices}")
                print("pairslist len: ", len(self.pairslist[dataset_idx]))
                raise e

            image, _current_crop = _load_and_process_image(pair['image'], **self.reshape_list[dataset_idx])
            print_depth_minmax = False #seq_i == 0
            depth_inverse_actual = self.depth_read_list[dataset_idx](pair['depth'], is_inverse=True, print_minmax=print_depth_minmax) # Load inverse depth (1/m)

            # Get focal length for this frame (at original resolution before any processing)
            original_h, original_w = depth_inverse_actual.shape
            fx_actual = self._get_focal_length(dataset_idx, pair, (original_h, original_w))

            # Apply Metric3D-style canonical transformation (now returns 6 values)
            # This corrects GT depth based on actual vs theoretical resize ratios
            depth_inverse_canonical, fx_canonical, fx_actual_returned, actual_valid_mask, fx_ratio, resize_ratio = self._apply_canonical_transform(
                depth_inverse_actual, fx_actual, original_h, original_w, target_resolution, resize_factor
            )

            # Convert back to numpy for resizing
            if isinstance(depth_inverse_canonical, torch.Tensor):
                depth_inverse_canonical_np = depth_inverse_canonical.cpu().numpy()
                actual_valid_mask_np = actual_valid_mask.cpu().numpy().astype(np.uint8)
            else:
                depth_inverse_canonical_np = depth_inverse_canonical
                actual_valid_mask_np = actual_valid_mask.astype(np.uint8)

            # Resize depth using same logic as before
            depth_resized = _load_and_process_depth(
                depth_inverse_canonical_np, image.shape, _current_crop, **self.reshape_list[dataset_idx]
            )

            # Resize mask using same logic (nearest neighbor for binary mask)
            mask_resized = cv2.resize(actual_valid_mask_np,
                                     (image.shape[2], image.shape[1]),
                                     interpolation=cv2.INTER_NEAREST)
            mask_resized = torch.from_numpy(mask_resized).bool()

            images.append(image)
            depths_canonical.append(depth_resized)
            focal_lengths_canonical.append(fx_canonical)
            focal_lengths_actual.append(fx_actual_returned)
            actual_valid_masks.append(mask_resized)
            fx_ratios.append(fx_ratio)  # NEW
            resize_ratios.append(resize_ratio)  # NEW

        try:
            images = torch.stack(images, dim=0)  # [T, C, H, W]
            depths_canonical_stacked = torch.stack(depths_canonical, dim=0)  # [T, H, W]
            focal_lengths_canonical_tensor = torch.tensor(focal_lengths_canonical, dtype=torch.float32)  # [T]
            focal_lengths_actual_tensor = torch.tensor(focal_lengths_actual, dtype=torch.float32)  # [T]
            actual_valid_masks_stacked = torch.stack(actual_valid_masks, dim=0)  # [T, H, W]
            fx_ratio_tensor = torch.tensor(fx_ratios, dtype=torch.float32)  # [T] - NEW
            resize_ratio_tensor = torch.tensor(resize_ratios, dtype=torch.float32)  # [T] - NEW
        except Exception as e:
            print(f"Error stacking tensors in dataset {dataset_idx}: {e}")
            print(f"Images length: {len(images)}")
            print(f"Depths canonical length: {len(depths_canonical)}")
            print(f"Masks length: {len(actual_valid_masks)}")
            print(f"Image shapes: {[img.shape if hasattr(img, 'shape') else type(img) for img in images]}")
            print(f"Depth shapes: {[d.shape if hasattr(d, 'shape') else type(d) for d in depths_canonical]}")
            print(f"Mask shapes: {[m.shape if hasattr(m, 'shape') else type(m) for m in actual_valid_masks]}")
            raise e

        return (images.float(),
                depths_canonical_stacked,
                focal_lengths_canonical_tensor,
                focal_lengths_actual_tensor,
                actual_valid_masks_stacked,
                fx_ratio_tensor,  # NEW
                resize_ratio_tensor,  # NEW
                dataset_idx)