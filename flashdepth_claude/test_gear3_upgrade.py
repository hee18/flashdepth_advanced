#!/usr/bin/env python3
"""
Test script for Gear3 Upgrade: Feature-level Metric Depth Learning with FG/BG Masks

Key differences from Gear3:
    - Produces FG/BG masks in addition to importance map
    - Visualizes FG/BG masks separately
    - Uses Gear3UpgradeMetricHead for prediction
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import logging
import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from PIL import Image
from einops import rearrange
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from flashdepth.model import FlashDepth
from flashdepth.gear3_upgrade_modules import Gear3UpgradeMetricHead
from dataloaders.combined_dataset import CombinedDataset
from dataloaders.waymo_segmentation_dataset import WaymoSegmentationDataset, collate_fn as waymo_collate_fn
from utils.metric_depth_metrics import MetricDepthMetrics, format_metrics
from utils.object_wise_evaluation import ObjectWiseMetrics
from utils.object_wise_visualization import create_object_wise_grid
from utils.helpers import save_gifs_as_grid, save_grid_to_mp4, depth_to_np_arr, torch_batch_to_np_arr
from utils.gear_common_helpers import depth_to_colored_frame
from utils.gear_video_utils import save_video as save_video_util

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Gear3UpgradeTester:
    """
    Test harness for Gear3 Upgrade model.

    Evaluates on:
        - Inverse depth metrics (TAE, AbsRel, δ1/δ2/δ3)
        - Metric depth visualization (no relative depth)
        - Importance map visualization
        - FG/BG mask visualization
    """
    def __init__(self, config):
        self.config = config
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Setup save directory - use results_dir if provided, otherwise use eval.outfolder
        save_dir_str = config.get('results_dir', config.eval.outfolder)
        self.save_dir = Path(save_dir_str)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Save directory: {self.save_dir}")

        # Object-wise evaluation configuration
        self.object_wise_enabled = config.get('object_wise', {}).get('enabled', False)
        self.object_wise_dataset = config.get('object_wise', {}).get('dataset', 'waymo')

        # Frame interval for visualization (only applies to sequence.png, not video)
        self.frame_interval = None

        if self.object_wise_enabled:
            logger.info(f"Object-wise evaluation ENABLED for dataset: {self.object_wise_dataset}")
            self.object_wise_metrics = ObjectWiseMetrics(dataset_type=self.object_wise_dataset)
        else:
            self.object_wise_metrics = None

        # Initialize model
        self.model = self._setup_model()

        # Setup test loader
        self.test_loader = self._setup_test_loader()

        # Setup metrics
        self.metrics = MetricDepthMetrics()

    def _setup_model(self):
        """Load trained Gear3 Upgrade model"""
        # Create base FlashDepth model
        model_config = dict(self.config.model)
        model_config['batch_size'] = 1
        model_config['use_metric_head'] = False

        model = FlashDepth(**model_config)

        # Add Gear3 Upgrade metric head
        embed_dim = 1024 if model.encoder == 'vitl' else 384
        dpt_dim = 256 if model.encoder == 'vitl' else 64
        num_heads = 16 if model.encoder == 'vitl' else 6
        separation_method = self.config.get('separation_method', 'cls_seg')

        model.gear3_upgrade_head = Gear3UpgradeMetricHead(
            embed_dim=embed_dim,
            dpt_dim=dpt_dim,
            num_heads=num_heads,
            separation_method=separation_method
        )

        # Enable attention weights storage
        # - 'multi_layer': Enable for multiple blocks (encoder-specific)
        #   - ViT-L (24 blocks): [3, 10, 16, 22]
        #   - ViT-S (12 blocks): [2, 5, 8, 11]
        # - Other methods: Enable only last block
        if separation_method == 'multi_layer':
            # Multi-layer: enable attention storage for specified blocks
            if model.encoder == 'vitl':
                target_blocks = [3, 10, 16, 22]  # ViT-L
            else:
                target_blocks = [2, 5, 8, 11]  # ViT-S

            for i, block in enumerate(model.pretrained.blocks):
                if i in target_blocks:
                    block.attn.store_attn_weights = True
                    logger.info(f"Enabled attention weights storage for block {i}")
                else:
                    block.attn.store_attn_weights = False
        else:
            # Other methods: enable only last block
            for i, block in enumerate(model.pretrained.blocks):
                if i == len(model.pretrained.blocks) - 1:
                    block.attn.store_attn_weights = True
                    logger.info(f"Enabled attention weights storage for block {i} (last block)")
                else:
                    block.attn.store_attn_weights = False

        # Load checkpoint
        checkpoint_path = self.config.get('load')
        if checkpoint_path and checkpoint_path != 'true':
            if os.path.exists(checkpoint_path):
                logger.info(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location='cpu')

                # Extract state dict
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                # Convert gear3_head keys to gear3_upgrade_head for backward compatibility
                converted_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('gear3_head.'):
                        new_key = key.replace('gear3_head.', 'gear3_upgrade_head.', 1)
                        converted_state_dict[new_key] = value
                        logger.debug(f"Converted key: {key} -> {new_key}")
                    else:
                        converted_state_dict[key] = value

                # Load state dict
                model.load_state_dict(converted_state_dict, strict=True)
                logger.info(f"Loaded checkpoint successfully")

                # Log training info if available
                if 'global_step' in checkpoint:
                    logger.info(f"Checkpoint step: {checkpoint['global_step']}")
                if 'phase' in checkpoint:
                    logger.info(f"Training phase: {checkpoint['phase']}")
            else:
                logger.warning(f"Checkpoint {checkpoint_path} not found")
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        else:
            logger.warning("No checkpoint specified!")

        model = model.to(self.device)
        model.eval()

        return model

    def _setup_test_loader(self):
        """Setup test data loader"""
        # Check if single sequence mode
        single_seq_path = self.config.get('single_sequence', None)

        # Check if whole-test mode (default: False)
        whole_test = self.config.get('whole_test', False)

        # Object-wise evaluation: use segmentation datasets
        if self.object_wise_enabled:
            video_length = int(self.config.get('vid_len', 50))  # Ensure integer (Hydra may pass string)
            resolution = self.config.get('resolution', self.config.eval.test_dataset_resolution)  # Allow resolution override
            data_root = self.config.dataset.data_root

            if self.object_wise_dataset == 'waymo':
                # WaymoSegmentationDataset expects data_root to be waymo_seg directory
                waymo_data_root = str(Path(data_root) / 'waymo_seg')

                # For waymo_seg in objwise mode: use 20 frames (0-19 with annotation)
                # Override video_length and frame_interval if not explicitly set
                if 'vid_len' not in self.config:
                    video_length = 20
                    logger.info(f"Auto-setting video_length=20 for waymo_seg objwise mode (frames with annotation)")

                # Set frame_interval to 2 for waymo_seg objwise (show every 2nd frame in sequence.png)
                if self.frame_interval is None:
                    self.frame_interval = 2
                    logger.info(f"Auto-setting frame_interval=2 for waymo_seg objwise mode")

                # whole_test controls which sequences to use:
                # - False (default): Use 'val' split which applies same filtering as WaymoDepth (first 8 scenes only)
                # - True: Use all sequences without filtering
                # objwise_mode=True: Only load frames 0-19 with segmentation annotation
                test_dataset = WaymoSegmentationDataset(
                    data_root=waymo_data_root,
                    split='val',
                    video_length=video_length,
                    resolution=resolution,
                    camera_name='FRONT',
                    objwise_mode=True  # Only use frames with segmentation annotation (0-19)
                )

                if whole_test:
                    # Reload all sequences without filtering
                    test_dataset.sequences = test_dataset._load_sequences_unfiltered()
                    logger.info(f"Using all sequences (whole_test=True): {len(test_dataset.sequences)} sequences")
                else:
                    # Already loaded with validation filtering (first 8 scenes only)
                    logger.info(f"Using validation subset (whole_test=False): {len(test_dataset.sequences)} sequences (first 8 scenes, same as training val)")

                collate_fn = waymo_collate_fn
            else:
                raise ValueError(f"Unknown object-wise dataset: {self.object_wise_dataset}")

            logger.info(f"Object-wise dataset: {self.object_wise_dataset} (size: {len(test_dataset)})")

        elif single_seq_path:
            # Single sequence mode: create custom dataset
            logger.info(f"Single sequence mode: {single_seq_path}")
            test_dataset = self._create_single_sequence_dataset(single_seq_path)
            collate_fn = self._collate_fn
        else:
            # Normal mode: use CombinedDataset
            # Use validation datasets if whole-test is False (default)
            if whole_test:
                test_datasets = self.config.eval.test_datasets
                logger.info("Using ALL test datasets (whole_test=True)")
            else:
                # Use validation datasets with waymo_seg (first 8 sequences)
                test_datasets = self.config.dataset.get('val_datasets', ['sintel', 'waymo_seg'])
                # Replace 'waymo' with 'waymo_seg' if present
                test_datasets = ['waymo_seg' if d == 'waymo' else d for d in test_datasets]
                logger.info(f"Using VALIDATION datasets only (whole_test=False): {test_datasets}")

            video_length = int(self.config.get('vid_len', 50))  # Ensure integer (Hydra may pass string)  # Get from config override
            resolution = self.config.get('resolution', self.config.eval.test_dataset_resolution)  # Allow resolution override

            logger.info(f"Test datasets: {test_datasets}")
            logger.info(f"Video length: {video_length}")
            logger.info(f"Resolution: {resolution}")

            test_dataset = CombinedDataset(
                root_dir=self.config.dataset.data_root,
                enable_dataset_flags=test_datasets,
                resolution=resolution,
                split='val',  # Use 'val' split which returns dict format
                video_length=video_length
            )
            collate_fn = self._collate_fn

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,  # Single process for testing
            pin_memory=True,
            collate_fn=collate_fn
        )

        logger.info(f"Test dataset size: {len(test_dataset)}")

        return test_loader

    def _create_single_sequence_dataset(self, sequence_path):
        """Create a dataset from a single sequence directory"""
        from torch.utils.data import Dataset
        import glob
        from PIL import Image

        class SingleSequenceDataset(Dataset):
            def __init__(self, seq_path, resolution=518):
                self.seq_path = Path(seq_path)
                
                # Infer dataset name from path for resolution mapping
                seq_path_str = str(seq_path).lower()
                dataset_name = None
                for ds in ['waymo_seg', 'waymo', 'eth3d', 'sintel', 'urbansyn', 'unreal4k', 'tartanair', 
                          'pointodyssey', 'dynamicreplica', 'spring', 'mvs-synth']:
                    if ds in seq_path_str:
                        dataset_name = ds
                        break
                
                # Resolution mapping based on combined_dataset.py logic
                # Test split uses non-square resolutions for different datasets
                if isinstance(resolution, str):
                    if resolution == 'base':
                        # Base resolution for test/val split (combined_dataset.py lines 86-99)
                        if dataset_name in ['eth3d', 'waymo', 'waymo_seg']:
                            self.resolution = (784, 518)  # (width, height)
                        elif dataset_name in ['sintel']:
                            self.resolution = (1022, 434)
                        elif dataset_name in ['urbansyn']:
                            self.resolution = (1036, 518)
                        elif dataset_name in ['unreal4k']:
                            self.resolution = (924, 518)
                        elif dataset_name in ['tartanair']:
                            self.resolution = (518, 518)
                        else:
                            # Default fallback for unknown datasets
                            logger.warning(f"Unknown dataset '{dataset_name}' in path, using default 518x518")
                            self.resolution = (518, 518)
                    
                    elif resolution == '2k':
                        # 2K resolution for test/val split (combined_dataset.py lines 108-118)
                        if dataset_name in ['eth3d', 'waymo', 'waymo_seg']:
                            self.resolution = (1918, 1274)  # (width, height)
                        elif dataset_name in ['sintel']:
                            self.resolution = (1022, 434)
                        elif dataset_name in ['urbansyn']:
                            self.resolution = (2044, 1022)
                        elif dataset_name in ['unreal4k']:
                            self.resolution = (2044, 1148)
                        else:
                            # Default fallback for unknown datasets
                            logger.warning(f"Unknown dataset '{dataset_name}' in path, using default 1918x1078")
                            self.resolution = (1918, 1078)
                    
                    else:
                        # Custom resolution string, try to parse as integer
                        try:
                            res_int = int(resolution)
                            self.resolution = (res_int, res_int)
                        except ValueError:
                            logger.error(f"Invalid resolution string: {resolution}, using 518x518")
                            self.resolution = (518, 518)
                else:
                    # Integer resolution, assume square
                    self.resolution = (int(resolution), int(resolution))

                # Find images and depths
                images_dir = self.seq_path / "images"
                depths_dir = self.seq_path / "depths"

                if not images_dir.exists():
                    raise FileNotFoundError(f"Images directory not found: {images_dir}")
                if not depths_dir.exists():
                    raise FileNotFoundError(f"Depths directory not found: {depths_dir}")

                # Get sorted image files
                self.image_files = sorted(glob.glob(str(images_dir / "*.png")))

                # Determine depth file pattern
                if len(self.image_files) > 0:
                    img_name = Path(self.image_files[0]).stem
                    # Try .geometric.png pattern (dynamicreplica)
                    depth_pattern = str(depths_dir / f"{img_name}_*.geometric.png")
                    depth_files = glob.glob(depth_pattern)

                    if len(depth_files) == 0:
                        # Try direct pattern matching
                        depth_files = sorted(glob.glob(str(depths_dir / "*.geometric.png")))

                    self.depth_files = sorted(depth_files)
                else:
                    self.depth_files = []

                assert len(self.image_files) > 0, f"No images found in {images_dir}"
                assert len(self.depth_files) > 0, f"No depth files found in {depths_dir}"
                assert len(self.image_files) == len(self.depth_files), \
                    f"Mismatch: {len(self.image_files)} images vs {len(self.depth_files)} depths"

                logger.info(f"Loaded single sequence: {len(self.image_files)} frames")
                logger.info(f"Dataset: {dataset_name}, Resolution: {self.resolution[0]}x{self.resolution[1]}")

            def __len__(self):
                return 1  # Single sequence

            def __getitem__(self, idx):
                # Load all frames
                images = []
                depths = []

                for img_path, depth_path in zip(self.image_files, self.depth_files):
                    # Load image
                    img = Image.open(img_path).convert('RGB')
                    # self.resolution is now (width, height) tuple from combined_dataset.py logic
                    # Use integer constants for PIL compatibility (2=BILINEAR, 0=NEAREST)
                    img = img.resize(self.resolution, 2)  # 2 = BILINEAR (positional arg)
                    img_array = np.array(img).astype(np.float32) / 255.0
                    # Normalize (ImageNet stats) - for model input only
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    img_normalized = (img_array - mean) / std  # This is for model input, not visualization
                    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float()  # [3, H, W], ensure float32
                    images.append(img_tensor)

                    # Load depth (.geometric.png for dynamicreplica)
                    depth_img = Image.open(depth_path)
                    depth_array = np.array(depth_img).astype(np.float32) / 1000.0  # mm to m
                    # Resize depth using same resolution as image
                    depth_pil = Image.fromarray(depth_array)
                    depth_resized = depth_pil.resize(self.resolution, 0)  # 0 = NEAREST (positional arg)
                    depth_array = np.array(depth_resized)
                    # Convert to inverse depth (1/m) like CombinedDataset
                    inverse_depth = np.zeros_like(depth_array)
                    valid_mask = depth_array > 0
                    inverse_depth[valid_mask] = 1.0 / depth_array[valid_mask]
                    depth_tensor = torch.from_numpy(inverse_depth)  # [H, W]
                    depths.append(depth_tensor)

                # Stack
                images = torch.stack(images, dim=0)  # [T, 3, H, W]
                depths = torch.stack(depths, dim=0)  # [T, H, W]

                return images, depths, "single_sequence"

        return SingleSequenceDataset(sequence_path, resolution=self.config.eval.test_dataset_resolution)

    def _collate_fn(self, batch):
        """Custom collate function to filter out None values and convert tuple to dict"""
        # Filter out None values
        batch = [item for item in batch if item is not None]

        # If all items are None, skip this batch
        if len(batch) == 0:
            return None

        # CombinedDataset returns (images, depths, focal_lengths, dataset_name) tuple for val/test splits
        # Convert to dict format for easier access
        if len(batch) > 0 and isinstance(batch[0], tuple):
            images, depths, focal_lengths, names = zip(*batch)
            return {
                'image': torch.stack(images, dim=0),
                'depth': torch.stack(depths, dim=0),
                'focal_lengths': torch.stack(focal_lengths, dim=0),
                'dataset_name': names
            }

        # Segmentation datasets return dict with 'images' key, rename to 'image' for compatibility
        if len(batch) > 0 and isinstance(batch[0], dict) and 'images' in batch[0]:
            # Already batched by segmentation collate_fn
            result = batch[0]  # Batch size is 1
            if 'images' in result:
                result['image'] = result.pop('images')  # Rename 'images' -> 'image'
            return result

        # Use default collate for dict items (training split)
        return torch.utils.data.dataloader.default_collate(batch)

    @torch.no_grad()
    def test(self):
        """Main testing loop"""
        logger.info("Starting testing...")

        all_metrics = []
        all_object_wise_metrics = []  # Track object-wise metrics separately
        sequence_id = 0

        for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Testing")):
            try:
                metrics = self.test_sequence(batch, sequence_id)
                # Add sequence_id for tracking
                metrics['sequence_id'] = sequence_id
                all_metrics.append(metrics)

                # Extract and store object-wise metrics
                if self.object_wise_enabled and 'object_wise' in metrics:
                    all_object_wise_metrics.append(metrics['object_wise'])

                sequence_id += 1

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Aggregate metrics
        if all_metrics:
            avg_metrics = self._aggregate_metrics(all_metrics)
            logger.info("\n" + "="*80)
            logger.info("FINAL RESULTS")
            logger.info("="*80)
            logger.info(format_metrics(avg_metrics))

            # Save overall results
            results_path = self.save_dir / "test_results.json"
            with open(results_path, 'w') as f:
                json.dump(avg_metrics, f, indent=2)
            logger.info(f"Results saved to {results_path}")

            # Save per-sequence results
            per_sequence_path = self.save_dir / "per_sequence_results.json"
            with open(per_sequence_path, 'w') as f:
                json.dump(all_metrics, f, indent=2)
            logger.info(f"Per-sequence results saved to {per_sequence_path}")

            # Find and save best sequence (lowest abs_rel)
            best_seq = min(all_metrics, key=lambda x: x['abs_rel'])
            best_seq_path = self.save_dir / "best_sequence.json"
            with open(best_seq_path, 'w') as f:
                json.dump(best_seq, f, indent=2)
            logger.info(f"\nBest sequence (lowest AbsRel): Sequence {best_seq['sequence_id']}")
            logger.info(f"  AbsRel: {best_seq['abs_rel']:.4f}")
            logger.info(f"  MAE: {best_seq['mae']:.4f}")
            logger.info(f"  RMSE: {best_seq['rmse']:.4f}")
            logger.info(f"  δ1: {best_seq['a1']:.4f}")
            logger.info(f"Best sequence saved to {best_seq_path}")

            # Aggregate and save object-wise metrics
            if self.object_wise_enabled and all_object_wise_metrics:
                logger.info("\n" + "="*80)
                logger.info("OBJECT-WISE EVALUATION RESULTS")
                logger.info("="*80)

                # Aggregate metrics across all sequences
                aggregated_class_metrics = self.object_wise_metrics.aggregate_metrics(all_object_wise_metrics)

                # Print summary
                self.object_wise_metrics.print_summary(aggregated_class_metrics)

                # Save to JSON
                object_wise_path = self.save_dir / "object_wise_results.json"
                self.object_wise_metrics.save_results(
                    aggregated_class_metrics,
                    object_wise_path
                )

        else:
            logger.warning("No metrics computed!")

    @torch.no_grad()
    def test_sequence(self, batch, sequence_id):
        """Test on a single sequence"""
        # Debug: Check batch type and structure
        if not isinstance(batch, dict):
            logger.error(f"Batch is not a dict! Type: {type(batch)}, Content: {batch if not isinstance(batch, torch.Tensor) else 'Tensor'}")
            raise TypeError(f"Expected dict, got {type(batch)}")

        # Handle both 'image' (CombinedDataset) and 'images' (WaymoSegmentationDataset) keys
        if 'images' in batch:
            images = batch['images'].to(self.device)  # [1, T, 3, H, W] or [T, 3, H, W]
            # Add batch dimension if missing (WaymoSegmentationDataset returns [T, 3, H, W])
            if images.ndim == 4:
                images = images.unsqueeze(0)  # [1, T, 3, H, W]
        else:
            images = batch['image'].to(self.device)  # [1, T, 3, H, W]
        gt_depth = batch['depth'].to(self.device)  # [1, T, H, W] or [T, H, W] - val split has no channel dim
        focal_lengths = batch['focal_lengths'].to(self.device)  # [1, T]

        # Add batch dimension if missing (WaymoSegmentationDataset returns [T, H, W])
        if gt_depth.ndim == 3:
            gt_depth = gt_depth.unsqueeze(0)  # [1, T, H, W]

        # Add channel dimension if needed
        if gt_depth.ndim == 4:
            gt_depth = gt_depth.unsqueeze(2)  # [1, T, 1, H, W]

        B, T = images.shape[:2]
        assert B == 1, "Batch size must be 1 for testing"

        # Dataloader gives inverse depth (1/m), scale to 100/m for training
        gt_depth_inverse_100 = gt_depth * 100.0  # [1, T, 1, H, W] in 100/m

        # Apply canonical transformation to GT (for fair comparison with model trained in canonical space)
        CANONICAL_FX = self.config.get('canonical_focal_length', 1000.0)
        use_canonical = self.config.get('use_canonical_space', False)
        if use_canonical:
            fx_actual = focal_lengths.view(1, T, 1, 1, 1)
            gt_depth_inverse_100 = gt_depth_inverse_100 * (CANONICAL_FX / fx_actual)

        # Storage for predictions
        pred_depths = []
        importance_maps = []
        fg_mask_list = []
        bg_mask_list = []

        # Best frame tracking
        best_frame_idx = 0
        best_frame_abs_rel = float('inf')

        # Warmup run for FPS measurement
        logger.info(f"Warmup run for FPS measurement...")

        # Initialize Mamba sequence for warmup
        if hasattr(self.model, 'mamba'):
            self.model.mamba.start_new_sequence()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            img_warmup = images[0, 0].unsqueeze(0)
            encoder_features_warmup = self.model.pretrained.get_intermediate_layers(
                img_warmup, self.model.intermediate_layer_idx[self.model.encoder]
            )

            # Collect attention weights (multi_layer or last block only)
            if self.config.get('separation_method', 'cls_seg') == 'multi_layer':
                # Multi-layer: collect from specified blocks
                if self.model.encoder == 'vitl':
                    target_blocks = [3, 10, 16, 22]
                else:
                    target_blocks = [2, 5, 8, 11]
                attention_weights_multi_layer_warmup = [
                    self.model.pretrained.blocks[i].attn.attn_weights for i in target_blocks
                ]
                attention_weights_warmup = None  # Not used in multi_layer mode
            else:
                # Other methods: use last block only
                last_block = self.model.pretrained.blocks[-1]
                attention_weights_warmup = last_block.attn.attn_weights
                attention_weights_multi_layer_warmup = None

            patch_tokens_warmup = encoder_features_warmup[-1]  # [B, N+1, embed_dim] (includes CLS)
            cls_token_warmup = patch_tokens_warmup[:, 0]  # [B, embed_dim]
            h_warmup, w_warmup = img_warmup.shape[2:]
            patch_h_warmup = h_warmup // self.model.patch_size
            patch_w_warmup = w_warmup // self.model.patch_size

            # Get DPT features first (without Mamba)
            dpt_features_warmup = self.model.depth_head.get_forward_features(
                encoder_features_warmup, patch_h_warmup, patch_w_warmup
            )
            path_1_warmup = dpt_features_warmup[-1]

            # Apply Gear3 Upgrade modulation
            path_1_modulated_warmup, _, _, _, _, _ = self.model.gear3_upgrade_head(
                patch_tokens_warmup, attention_weights_warmup, [path_1_warmup], patch_h_warmup, patch_w_warmup,
                cls_token=cls_token_warmup, attention_weights_multi_layer=attention_weights_multi_layer_warmup
            )

            # Apply Mamba temporal processing
            path_1_temporal_warmup = self.model.dpt_features_to_mamba(
                input_shape=(1, 3, h_warmup, w_warmup),
                dpt_features=path_1_modulated_warmup,
                in_dpt_layer=0
            )

            out_warmup = self.model.depth_head.scratch.output_conv1(path_1_temporal_warmup)
            out_warmup = F.interpolate(out_warmup, (h_warmup, w_warmup), mode="bilinear", align_corners=True)
            _ = self.model.depth_head.scratch.output_conv2(out_warmup)
        del encoder_features_warmup, attention_weights_warmup, patch_tokens_warmup, dpt_features_warmup
        del path_1_warmup, path_1_modulated_warmup, path_1_temporal_warmup, out_warmup
        torch.cuda.empty_cache()

        # FPS measurement (like original FlashDepth)
        warmup_frames = min(5, T)  # Warmup frames to skip initial overhead
        start_time = None  # Will start timing after warmup

        # Initialize Mamba sequence for actual test (critical for temporal processing!)
        if hasattr(self.model, 'mamba'):
            self.model.mamba.start_new_sequence()

        # Process each frame
        for t in range(T):
            # Start timing after warmup frames (like original FlashDepth)
            if t == warmup_frames:
                torch.cuda.synchronize()
                import time
                start_time = time.time()

            img_t = images[0, t]  # [3, H, W]
            gt_t_inverse = gt_depth_inverse_100[0, t]  # [1, H, W] in 100/m

            # Use BFloat16 for forward pass (same as train_gear3.py)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Extract features from DINOv2
                encoder_features = self.model.pretrained.get_intermediate_layers(
                    img_t.unsqueeze(0), self.model.intermediate_layer_idx[self.model.encoder]
                )

                # Collect attention weights (multi_layer or last block only)
                if self.config.get('separation_method', 'cls_seg') == 'multi_layer':
                    # Multi-layer: collect from specified blocks
                    if self.model.encoder == 'vitl':
                        target_blocks = [3, 10, 16, 22]
                    else:
                        target_blocks = [2, 5, 8, 11]
                    attention_weights_multi_layer = [
                        self.model.pretrained.blocks[i].attn.attn_weights for i in target_blocks
                    ]
                    attention_weights = None  # Not used in multi_layer mode
                else:
                    # Other methods: use last block only
                    last_block = self.model.pretrained.blocks[-1]
                    attention_weights = last_block.attn.attn_weights
                    attention_weights_multi_layer = None

                # Get patch tokens from last encoder layer (includes CLS token)
                patch_tokens = encoder_features[-1]  # [B, N+1, embed_dim]
                cls_token = patch_tokens[:, 0]  # [B, embed_dim]

                # Get DPT features first (without Mamba)
                h, w = img_t.shape[1:]
                patch_h, patch_w = h // self.model.patch_size, w // self.model.patch_size
                dpt_features = self.model.depth_head.get_forward_features(
                    encoder_features, patch_h, patch_w
                )
                path_1 = dpt_features[-1]

                # Apply Gear3 Upgrade modulation (produces FG/BG masks)
                # Pass cls_token for cls_seg mode support
                path_1_modulated, importance_map, fg_features, bg_features, fg_mask, bg_mask = self.model.gear3_upgrade_head(
                    patch_tokens, attention_weights, [path_1], patch_h, patch_w,
                    cls_token=cls_token, attention_weights_multi_layer=attention_weights_multi_layer
                )

                # Apply Mamba temporal processing
                path_1_temporal = self.model.dpt_features_to_mamba(
                    input_shape=(1, 3, h, w),
                    dpt_features=path_1_modulated,
                    in_dpt_layer=0
                )

                # Get depth prediction (output is inverse depth in 100/m scale)
                out = self.model.depth_head.scratch.output_conv1(path_1_temporal)
                out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
                out = self.model.depth_head.scratch.output_conv2(out)  # [1, 1, H, W]

                # Prediction is already positive (Softplus activation in output_conv2)
                pred_depth_inverse_100 = out  # [1, 1, H, W] in 100/m_canonical

                # De-canonicalization: convert from canonical space to actual metric space
                if use_canonical:
                    # pred_inverse_actual = pred_inverse_canonical * (fx_actual / CANONICAL_FX)
                    fx_t = focal_lengths[0, t]  # Focal length for this frame
                    pred_depth_inverse_100 = pred_depth_inverse_100 * (fx_t / CANONICAL_FX)

                # Interpolate prediction to GT resolution (like train_gear3.py validation)
                gt_t_shape = gt_t_inverse.shape[-2:]  # GT original resolution
                if pred_depth_inverse_100.shape[-2:] != gt_t_shape:
                    pred_depth_inverse_100 = F.interpolate(
                        pred_depth_inverse_100, size=gt_t_shape, mode="bilinear", align_corners=True
                    )

                # Convert to metric depth: 100/m -> m
                pred_depth_metric = 100.0 / (pred_depth_inverse_100[0] + 1e-8)  # [1, H, W]

                # Upsample importance_map to image resolution for smooth visualization
                # (like train_gear3_upgrade.py does before passing to visualizer)
                h_full, w_full = img_t.shape[1:]  # Image resolution
                importance_map_resized = F.interpolate(
                    importance_map, size=(h_full, w_full), mode='bilinear', align_corners=True
                )  # [1, 1, H, W] at image resolution

            # End timing for FPS measurement (after last frame, like original FlashDepth)
            if t == T - 1 and start_time is not None:
                torch.cuda.synchronize()
                end_time = time.time()

            # List append for visualization (outside FPS measurement)
            pred_depths.append(pred_depth_metric)
            # Save upsampled importance_map (already smooth, no need to interpolate again in visualization)
            importance_maps.append(importance_map_resized[0])  # [1, H, W] at image resolution
            fg_mask_list.append(fg_mask[0])
            bg_mask_list.append(bg_mask[0])

        # Calculate FPS (like original FlashDepth: exclude warmup frames)
        if start_time is not None:
            inference_time = end_time - start_time
            fps = (T - warmup_frames) / inference_time if inference_time > 0 else 0
            logger.info(f"Inference time: {inference_time:.4f}s for {T - warmup_frames} frames (warmup {warmup_frames} excluded)")
            logger.info(f"FPS: {fps:.2f} frames/second")
        else:
            # Too few frames for FPS measurement
            fps = 0
            logger.warning(f"Too few frames ({T}) for FPS measurement (need > {warmup_frames})")

        # Stack predictions
        pred_depths = torch.stack(pred_depths, dim=0)  # [T, 1, H, W] in meters
        importance_maps = torch.stack(importance_maps, dim=0)  # [T, 1, patch_h, patch_w]
        fg_mask_all = torch.stack(fg_mask_list, dim=0)  # [T, 1, patch_h, patch_w]
        bg_mask_all = torch.stack(bg_mask_list, dim=0)  # [T, 1, patch_h, patch_w]

        # Convert GT to metric depth for evaluation: 100/m -> m
        gt_depth_metric = 100.0 / (gt_depth_inverse_100[0] + 1e-8)  # [T, 1, H, W] in meters

        # Compute metrics (both pred and GT are now in meters)
        # Move to CPU and compute per-frame metrics (like test_metric_head.py)
        pred_depths_cpu = pred_depths.cpu()
        gt_depth_metric_cpu = gt_depth_metric.cpu()

        frame_metrics = []
        for t in range(pred_depths.shape[0]):
            # Get individual frames (already on CPU)
            pred_frame = pred_depths_cpu[t, 0]  # [H, W]
            gt_frame = gt_depth_metric_cpu[t, 0]  # [H, W]

            # Create valid mask for this frame (like train_gear3 validation)
            # Use same MAX_DEPTH as Gear3Visualizer (70m)
            MAX_DEPTH = 70.0
            gt_valid_mask = (gt_frame > 0) & (gt_frame < MAX_DEPTH)  # GT valid pixels
            pred_valid_mask = (pred_frame > 0) & (pred_frame < MAX_DEPTH)  # Filter extreme values
            valid_mask = gt_valid_mask & pred_valid_mask  # [H, W] bool tensor

            # Debug logging for first frame of first sequence
            if t == 0 and sequence_id == 0:
                logger.info(f"DEBUG Metrics - Frame {t}")
                logger.info(f"  GT depth range: [{gt_frame.min():.2f}, {gt_frame.max():.2f}] meters")
                logger.info(f"  Pred depth range: [{pred_frame.min():.2f}, {pred_frame.max():.2f}] meters")
                logger.info(f"  Valid pixels: {valid_mask.sum()} / {valid_mask.numel()} ({100*valid_mask.sum()/valid_mask.numel():.1f}%)")
                if valid_mask.sum() > 0:
                    gt_valid_values = gt_frame[valid_mask]
                    pred_valid_values = pred_frame[valid_mask]
                    logger.info(f"  GT valid range: [{gt_valid_values.min():.2f}, {gt_valid_values.max():.2f}]")
                    logger.info(f"  Pred valid range: [{pred_valid_values.min():.2f}, {pred_valid_values.max():.2f}]")
                    mae = torch.abs(pred_valid_values - gt_valid_values).mean()
                    logger.info(f"  MAE: {mae:.4f} meters")

            if valid_mask.sum() > 0:
                frame_metric = self.metrics.compute_metric_depth_metrics(
                    pred_frame,  # [H, W]
                    gt_frame,   # [H, W]
                    valid_mask  # [H, W]
                )
                frame_metrics.append(frame_metric)

                # Track best frame (lowest abs_rel)
                if frame_metric['abs_rel'] < best_frame_abs_rel:
                    best_frame_abs_rel = frame_metric['abs_rel']
                    best_frame_idx = t

        # Average metrics across frames
        if len(frame_metrics) == 0:
            logger.warning(f"No valid frames for sequence {sequence_id}")
            return {k: 0.0 for k in ["mae", "rmse", "abs_rel", "a1", "tae", "fps"]}

        metrics = {}
        for key in frame_metrics[0].keys():
            values = [m[key] for m in frame_metrics]
            metrics[key] = np.mean(values)

        # Compute TAE (Temporal Alignment Error) - sequence-level metric
        # TAE measures frame-to-frame consistency
        if len(pred_depths) > 1:
            tae_errors = []
            for t in range(len(pred_depths) - 1):
                pred_t = pred_depths_cpu[t, 0]  # [H, W]
                pred_t_next = pred_depths_cpu[t + 1, 0]  # [H, W]
                gt_t = gt_depth_metric_cpu[t, 0]  # [H, W]
                gt_t_next = gt_depth_metric_cpu[t + 1, 0]  # [H, W]

                # Valid mask for both frames
                MAX_DEPTH = 70.0
                valid_t = (gt_t > 0) & (gt_t < MAX_DEPTH) & (pred_t > 0) & (pred_t < MAX_DEPTH)
                valid_t_next = (gt_t_next > 0) & (gt_t_next < MAX_DEPTH) & (pred_t_next > 0) & (pred_t_next < MAX_DEPTH)
                valid_both = valid_t & valid_t_next  # [H, W]

                if valid_both.sum() > 0:
                    # Compute depth change (temporal derivative)
                    pred_change = pred_t_next - pred_t  # [H, W]
                    gt_change = gt_t_next - gt_t  # [H, W]

                    # TAE: mean absolute error in temporal change
                    tae = torch.abs(pred_change[valid_both] - gt_change[valid_both]).mean()
                    tae_errors.append(tae.item())

            metrics['tae'] = np.mean(tae_errors) if len(tae_errors) > 0 else 0.0
        else:
            metrics['tae'] = 0.0

        # Add FPS to metrics
        metrics['fps'] = fps

        # Object-wise evaluation: compute per-class metrics for all frames
        # Initialize variables
        seg_masks_np = None  # Will store per-frame segmentations
        per_frame_class_metrics = []  # Per-frame metrics

        if self.object_wise_enabled and 'segmentations' in batch:
            try:
                # Get per-frame segmentations
                seg_masks = batch['segmentations'][0]  # [T, H, W] - batch size is 1
                T_seg = seg_masks.shape[0]

                logger.info(f"Processing {T_seg} frames with segmentation")

                # Convert to numpy
                seg_masks_np = seg_masks.cpu().numpy() if isinstance(seg_masks, torch.Tensor) else seg_masks

                # Compute metrics for each frame
                for t in range(T_seg):
                    pred_frame = pred_depths_cpu[t, 0].numpy()  # [H, W]
                    gt_frame = gt_depth_metric_cpu[t, 0].numpy()  # [H, W]
                    seg_mask_frame = seg_masks_np[t]  # [H, W]

                    # Resize segmentation to match pred/GT if needed
                    if seg_mask_frame.shape != pred_frame.shape:
                        import cv2
                        seg_mask_frame = cv2.resize(
                            seg_mask_frame.astype(np.int32),
                            (pred_frame.shape[1], pred_frame.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )
                        # Update in array
                        seg_masks_np[t] = seg_mask_frame

                    # Compute per-class metrics
                    frame_class_metrics = self.object_wise_metrics.compute_metrics_per_class(
                        pred_depth=pred_frame,
                        gt_depth=gt_frame,
                        seg_mask=seg_mask_frame,
                        min_pixels=100
                    )
                    per_frame_class_metrics.append(frame_class_metrics)

                # Aggregate across all frames
                class_metrics = self.object_wise_metrics.aggregate_metrics(per_frame_class_metrics)
                metrics['object_wise'] = class_metrics
                logger.info(f"Computed object-wise metrics for {len(class_metrics)} classes across {T_seg} frames")

            except Exception as e:
                logger.error(f"Error computing object-wise metrics: {e}")
                import traceback
                traceback.print_exc()
                metrics['object_wise'] = {}
                seg_masks_np = None
                per_frame_class_metrics = []

        # Recreate valid_mask on GPU for visualization
        valid_mask = (gt_depth_metric > 0)  # [T, 1, H, W] on GPU

        # Visualize
        if self.config.eval.get('save_grid', True):
            self._visualize_sequence(
                images[0], pred_depths, gt_depth_metric, importance_maps,
                valid_mask, sequence_id, metrics, fps
            )

        # Save video (GIF or MP4)
        # Note: frame_interval is NOT applied to video - use all frames
        if self.config.eval.get('out_video', True):
            # Use original model resolution for images (following FlashDepth approach)
            # save_gifs_as_grid/save_grid_to_mp4 will handle downsampling to save_res
            save_video_util(
                images[0], pred_depths, gt_depth_metric, valid_mask, sequence_id,
                save_dir=self.save_dir,
                config=self.config
            )

        # Save best frame visualizations
        if len(frame_metrics) > 0:
            logger.info(f"Best frame for sequence {sequence_id}: Frame {best_frame_idx} (AbsRel={best_frame_abs_rel:.4f})")

            # Extract layer_weights for visualization (multi_layer separation only)
            layer_weights = None
            separation_method = self.config.get('separation_method', 'cls_seg')
            logger.info(f"Separation method: {separation_method}")

            if separation_method == 'multi_layer':
                try:
                    logger.info(f"Attempting to extract layer_weights...")
                    logger.info(f"Model has gear3_upgrade_head: {hasattr(self.model, 'gear3_upgrade_head')}")
                    if hasattr(self.model, 'gear3_upgrade_head'):
                        # Use same attribute name as training: multi_layer_fusion.fusion_weights
                        logger.info(f"gear3_upgrade_head has multi_layer_fusion: {hasattr(self.model.gear3_upgrade_head, 'multi_layer_fusion')}")
                        if hasattr(self.model.gear3_upgrade_head, 'multi_layer_fusion'):
                            fusion_weights = self.model.gear3_upgrade_head.multi_layer_fusion.fusion_weights
                            layer_weights = torch.softmax(fusion_weights, dim=0).detach().cpu().numpy()
                            logger.info(f"Successfully extracted layer_weights: {layer_weights}")
                        else:
                            logger.warning(f"gear3_upgrade_head does not have multi_layer_fusion attribute")
                    else:
                        logger.warning(f"Model does not have gear3_upgrade_head attribute")
                except Exception as e:
                    logger.error(f"Failed to extract layer_weights: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                logger.info(f"Skipping layer_weights extraction (separation_method={separation_method}, not 'multi_layer')")

            # Get segmentation and actual frame number for best frame
            if self.object_wise_enabled and seg_masks_np is not None:
                if best_frame_idx < len(seg_masks_np):
                    # Best frame has segmentation - use it
                    seg_mask_for_viz = seg_masks_np[best_frame_idx]  # [H, W]
                    class_metrics_for_viz = per_frame_class_metrics[best_frame_idx] if best_frame_idx < len(per_frame_class_metrics) else None

                    # Get actual frame number from frame_indices
                    actual_frame_number = batch['frame_indices'][0][best_frame_idx] if 'frame_indices' in batch else best_frame_idx
                    logger.info(f"Best frame batch_idx={best_frame_idx}, actual_frame={actual_frame_number} - showing object mask")
                else:
                    # Best frame doesn't have segmentation
                    seg_mask_for_viz = None
                    class_metrics_for_viz = None
                    actual_frame_number = best_frame_idx
                    logger.info(f"Best frame {best_frame_idx} has no segmentation - showing valid mask instead")
            else:
                seg_mask_for_viz = None
                class_metrics_for_viz = None
                actual_frame_number = best_frame_idx

            self._save_best_frame_visualizations(
                images[0, best_frame_idx],  # [3, H, W]
                pred_depths[best_frame_idx, 0],  # [H, W]
                gt_depth_metric[best_frame_idx, 0],  # [H, W]
                importance_maps[best_frame_idx, 0],  # [patch_h, patch_w]
                fg_mask_all[best_frame_idx],  # [1, patch_h, patch_w]
                bg_mask_all[best_frame_idx],  # [1, patch_h, patch_w]
                sequence_id,
                actual_frame_number,  # Use actual frame number, not batch index
                best_frame_abs_rel,
                fps,  # Add FPS
                seg_mask_for_viz,  # Add segmentation mask (only if matches best_frame)
                class_metrics_for_viz,  # Add class metrics
                layer_weights  # Add layer weights
            )

        return metrics

    def _visualize_sequence(self, images, pred_depths, gt_depths, importance_maps,
                           valid_mask, sequence_id, metrics, fps=None):
        """
        Create visualization grid for a sequence.

        Rows: Image, Metric Depth (Prediction), Metric Depth (GT), Importance Map
        """
        T = images.shape[0]
        frames_to_show = min(10, T)  # Show up to 10 frames

        # Use frame_interval if set (for waymo_seg objwise), otherwise auto-calculate
        if self.frame_interval is not None:
            interval = self.frame_interval
            logger.info(f"Using frame_interval={interval} for sequence.png visualization")
        else:
            interval = max(1, T // frames_to_show)

        frame_indices = list(range(0, T, interval))[:frames_to_show]

        # Create figure with actual number of frames (not frames_to_show)
        actual_frames = len(frame_indices)
        fig, axes = plt.subplots(4, actual_frames, figsize=(actual_frames * 3, 12))
        if actual_frames == 1:
            axes = axes.reshape(-1, 1)

        for col, t in enumerate(frame_indices):
            # Row 0: Image (denormalize ImageNet normalization)
            img = images[t].permute(1, 2, 0).cpu().numpy()
            # Min-Max normalization (FlashDepth original method)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)
            axes[0, col].imshow(img)
            axes[0, col].set_title(f'Frame {t}')
            axes[0, col].axis('off')

            # Row 1: Predicted metric depth (invalid pixels = black)
            MAX_DEPTH = 70.0
            pred = pred_depths[t, 0].cpu().numpy()
            pred_valid = (pred > 0) & (pred < MAX_DEPTH)  # Only <70m
            pred_display = np.where(pred_valid, pred, np.nan)  # Invalid = NaN
            if pred_valid.sum() > 0:
                pred_vmin = np.nanpercentile(pred_display, 2)
                pred_vmax = np.nanpercentile(pred_display, 98)
            else:
                pred_vmin, pred_vmax = 0, 1
            cmap_pred = plt.cm.plasma_r.copy()
            cmap_pred.set_bad(color='black')  # NaN pixels = black
            axes[1, col].imshow(pred_display, cmap=cmap_pred, vmin=pred_vmin, vmax=pred_vmax)  # plasma_r: near=bright, far=dark
            axes[1, col].set_title(f'Pred (m)')
            axes[1, col].axis('off')

            # Row 2: GT metric depth (invalid pixels = black)
            gt = gt_depths[t, 0].cpu().numpy()
            gt_valid = (gt > 0) & (gt < MAX_DEPTH)  # Only <70m
            gt_display = np.where(gt_valid, gt, np.nan)  # Invalid = NaN
            if gt_valid.sum() > 0:
                gt_vmin = np.nanpercentile(gt_display, 2)
                gt_vmax = np.nanpercentile(gt_display, 98)
            else:
                gt_vmin, gt_vmax = 0, 1
            cmap_gt = plt.cm.plasma_r.copy()
            cmap_gt.set_bad(color='black')  # NaN pixels = black
            axes[2, col].imshow(gt_display, cmap=cmap_gt, vmin=gt_vmin, vmax=gt_vmax)  # plasma_r: near=bright, far=dark
            axes[2, col].set_title(f'GT (m)')
            axes[2, col].axis('off')

            # Row 3: Importance map (already upsampled to image resolution in test_sequence)
            importance_resized = importance_maps[t]  # [1, H, W] already at image resolution
            importance_display = importance_resized.squeeze().cpu().numpy()  # [H, W]
            axes[3, col].imshow(importance_display, cmap='jet', vmin=0, vmax=1)
            axes[3, col].set_title(f'Importance')
            axes[3, col].axis('off')

        # Add overall title with metrics
        title_str = (
            f"Sequence {sequence_id} | "
            f"TAE: {metrics.get('tae', 0):.4f} | "
            f"AbsRel: {metrics.get('abs_rel', 0):.4f} | "
            f"δ1: {metrics.get('a1', 0):.4f}"
        )
        if fps is not None:
            title_str += f" | FPS: {fps:.1f}"

        fig.suptitle(title_str, fontsize=14)

        plt.tight_layout()
        save_path = self.save_dir / f"sequence_{sequence_id:04d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Saved visualization: {save_path}")

    def _save_best_frame_visualizations(self, image, pred_depth, gt_depth, importance_map,
                                        fg_mask, bg_mask, sequence_id, frame_idx, abs_rel, fps=None,
                                        seg_mask=None, class_metrics=None, layer_weights=None):
        """
        Save best frame visualization matching train_gear3_upgrade layout

        Creates a comprehensive 4x3 grid visualization with format:
            best_frame_seq{N}_{frame_idx}_absrel_{abs_rel:.4f}.png

        Layout:
            Row 1: Input Image | GT Depth | Pred Depth
            Row 2: Importance Map | FG Mask | BG Mask
            Row 3: Valid Mask | Error Map | Metrics
            Row 4: Depth Distribution (2 cols) | Importance Distribution

        Args:
            image: [3, H, W] - RGB image
            pred_depth: [H, W] - Predicted metric depth
            gt_depth: [H, W] - Ground truth metric depth
            importance_map: [patch_h, patch_w] - Importance map (0-1 normalized)
            fg_mask: [1, patch_h, patch_w] - Foreground mask from Gear3 Upgrade head
            bg_mask: [1, patch_h, patch_w] - Background mask from Gear3 Upgrade head
            sequence_id: int - Sequence index
            frame_idx: int - Frame index within sequence
            abs_rel: float - AbsRel metric for this frame
            fps: float - Optional FPS measurement
        """
        # Convert tensors to numpy and move to CPU
        if isinstance(image, torch.Tensor):
            if image.shape[0] == 3:  # [3, H, W]
                image = image.permute(1, 2, 0)  # [H, W, 3]
            image = image.float().cpu().numpy()

        if isinstance(pred_depth, torch.Tensor):
            pred_depth = pred_depth.float().cpu().numpy()
        if isinstance(gt_depth, torch.Tensor):
            gt_depth = gt_depth.float().cpu().numpy()
        if isinstance(importance_map, torch.Tensor):
            importance_map = importance_map.float().cpu().numpy()

        # Min-Max normalization (FlashDepth original method)
        image_np = (image * 1.0 - image.min()) / (image.max() - image.min() + 1e-8)
        image_np = np.clip(image_np, 0, 1)

        # Get image size
        img_h, img_w = image_np.shape[:2]

        # Create separate valid masks for GT and Pred
        MAX_DEPTH = 70.0  # Same as training (100/70 = 1.43 inverse depth threshold)
        gt_valid_mask = (gt_depth > 0) & (gt_depth < MAX_DEPTH)  # GT valid pixels
        pred_valid_mask = (pred_depth > 0) & (pred_depth < MAX_DEPTH)  # Pred valid pixels (<70m for visualization)
        error_valid_mask = gt_valid_mask & pred_valid_mask  # Both valid for error computation

        # Calculate error (only where both GT and Pred are valid)
        abs_error = np.abs(pred_depth - gt_depth)
        abs_error_masked = np.where(error_valid_mask, abs_error, np.nan)

        # Calculate importance statistics
        imp_mean = importance_map.mean()
        imp_std = importance_map.std()

        # Extract FG/BG masks from importance map (binary thresholding)
        # Use mean threshold to separate foreground from background (SAME AS test_gear3.py)
        fg_mask_binary = (importance_map >= imp_mean).astype(np.float32)
        bg_mask_binary = (importance_map < imp_mean).astype(np.float32)

        # Upsample binary masks with bilinear for smoother visualization (SAME AS test_gear3.py)
        # Bilinear interpolation of binary masks creates smooth boundaries
        fg_mask_upsampled = F.interpolate(
            torch.from_numpy(fg_mask_binary).unsqueeze(0).unsqueeze(0),
            size=(img_h, img_w),
            mode='bilinear',
            align_corners=True
        ).squeeze().numpy()

        bg_mask_upsampled = F.interpolate(
            torch.from_numpy(bg_mask_binary).unsqueeze(0).unsqueeze(0),
            size=(img_h, img_w),
            mode='bilinear',
            align_corners=True
        ).squeeze().numpy()

        # Create figure with 4x3 grid layout matching train_gear3_upgrade
        fig = plt.figure(figsize=(15, 16))
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

        # ==================== Row 1: Input, GT, Pred ====================

        # 1. Input Image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image_np)
        ax1.set_title('Input Image', fontsize=14, fontweight='bold')
        ax1.axis('off')

        # 2. Ground Truth Depth (invalid pixels = black)
        ax2 = fig.add_subplot(gs[0, 1])
        gt_display = np.where(gt_valid_mask, gt_depth, np.nan)  # Invalid = NaN
        if gt_valid_mask.sum() > 0:
            vmin = np.nanpercentile(gt_display, 2)
            vmax = np.nanpercentile(gt_display, 98)
        else:
            vmin, vmax = 0, 1
        cmap_gt = plt.cm.plasma_r.copy()
        cmap_gt.set_bad(color='black')  # NaN pixels = black
        im2 = ax2.imshow(gt_display, cmap=cmap_gt, vmin=vmin, vmax=vmax)  # plasma_r: near=bright, far=dark
        ax2.set_title('Ground Truth Depth (m)', fontsize=14, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # 3. Predicted Metric Depth (invalid pixels = black)
        ax3 = fig.add_subplot(gs[0, 2])
        pred_display = np.where(pred_valid_mask, pred_depth, np.nan)  # Invalid = NaN
        cmap_pred = plt.cm.plasma_r.copy()
        cmap_pred.set_bad(color='black')  # NaN pixels = black
        im3 = ax3.imshow(pred_display, cmap=cmap_pred, vmin=vmin, vmax=vmax)  # plasma_r: near=bright, far=dark
        ax3.set_title('Predicted Metric Depth (m)', fontsize=14, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # ==================== Row 2: Importance, FG, BG ====================

        # 4. Importance Map (upsampled with bilinear for smooth visualization)
        ax4 = fig.add_subplot(gs[1, 0])
        importance_upsampled = F.interpolate(
            torch.from_numpy(importance_map).unsqueeze(0).unsqueeze(0),
            size=(img_h, img_w),
            mode='bilinear',
            align_corners=True
        ).squeeze().numpy()
        im4 = ax4.imshow(importance_upsampled, cmap='jet', vmin=0, vmax=1)
        ax4.set_title(f'Importance Map\nmean={imp_mean:.3f}, std={imp_std:.3f}',
                     fontsize=14, fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        # 5. FG Mask (Red overlay)
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.imshow(image_np)
        # Create FG overlay (Red channel only)
        fg_overlay = np.zeros((*fg_mask_upsampled.shape, 3))
        fg_overlay[..., 0] = fg_mask_upsampled  # Red channel
        ax5.imshow(fg_overlay, alpha=0.5)
        fg_ratio = fg_mask_binary.mean() * 100  # Use binary mask for ratio
        ax5.set_title(f'FG Mask (Red)\n{fg_ratio:.1f}%', fontsize=14, fontweight='bold')
        ax5.axis('off')

        # 6. BG Mask (Blue overlay)
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.imshow(image_np)
        # Create BG overlay (Blue channel only)
        bg_overlay = np.zeros((*bg_mask_upsampled.shape, 3))
        bg_overlay[..., 2] = bg_mask_upsampled  # Blue channel
        ax6.imshow(bg_overlay, alpha=0.5)
        bg_ratio = bg_mask_binary.mean() * 100  # Use binary mask for ratio
        ax6.set_title(f'BG Mask (Blue)\n{bg_ratio:.1f}%', fontsize=14, fontweight='bold')
        ax6.axis('off')

        # ==================== Row 3: Valid/Object Mask, Error, Metrics ====================

        # 7. Valid Mask or Object Mask (depending on object_wise mode)
        ax7 = fig.add_subplot(gs[2, 0])
        if seg_mask is None:
            # Non-object-wise mode: show GT Valid Mask (valid=white, invalid=black)
            logger.info(f"[VISUALIZATION] seg_mask is None, showing GT Valid Mask")
            gt_valid_ratio = gt_valid_mask.sum() / gt_valid_mask.size
            ax7.imshow(gt_valid_mask.astype(np.uint8), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            ax7.set_title(f'GT Valid Mask ({gt_valid_ratio*100:.1f}%)\ninvalid: black',
                         fontsize=12, fontweight='bold')
            ax7.axis('off')
        else:
            # Object-wise mode: show Object Mask (only dynamic objects)
            logger.info(f"[VISUALIZATION] seg_mask shape: {seg_mask.shape}")
            logger.info(f"[VISUALIZATION] seg_mask dtype: {seg_mask.dtype}")
            logger.info(f"[VISUALIZATION] seg_mask unique values: {np.unique(seg_mask)}")
            logger.info(f"[VISUALIZATION] seg_mask min: {seg_mask.min()}, max: {seg_mask.max()}")
            logger.info(f"[VISUALIZATION] seg_mask > 0 count: {(seg_mask > 0).sum()}")

            # Waymo object class IDs (dynamic objects + important static objects)
            # Based on WAYMO_OBJECT_CLASSES in utils/object_wise_evaluation.py
            waymo_object_class_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            # 1: vehicle, 2: pedestrian, 3: sign, 4: cyclist, 5: traffic_light,
            # 6: pole, 7: construction_cone, 8: bicycle, 9: motorcycle

            # Create object mask: include dynamic objects and important static objects
            object_mask = np.zeros_like(seg_mask, dtype=np.uint8)
            for class_id in waymo_object_class_ids:
                object_mask |= (seg_mask == class_id).astype(np.uint8)

            ax7.imshow(object_mask, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            object_ratio = object_mask.sum() / object_mask.size
            # Count only object classes present in this frame
            num_object_classes = len([cid for cid in waymo_object_class_ids if (seg_mask == cid).any()])

            logger.info(f"[VISUALIZATION] object_mask sum: {object_mask.sum()}, ratio: {object_ratio*100:.1f}%, num_object_classes: {num_object_classes}")

            ax7.set_title(f'Object Mask\n{object_ratio*100:.1f}% ({object_mask.sum():,} pixels)\n{num_object_classes} object classes',
                         fontsize=14, fontweight='bold')
            ax7.axis('off')

        # 8. Absolute Error Map
        ax8 = fig.add_subplot(gs[2, 1])
        if error_valid_mask.sum() > 0:
            error_vmax = np.nanpercentile(abs_error_masked, 95)
        else:
            error_vmax = 1
        im8 = ax8.imshow(abs_error_masked, cmap='hot', vmin=0, vmax=error_vmax)
        ax8.set_title(f'Absolute Error (m)\nMean: {np.nanmean(abs_error_masked):.3f}',
                     fontsize=14, fontweight='bold')
        ax8.axis('off')
        plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)

        # 9. Depth Metrics
        ax9 = fig.add_subplot(gs[2, 2])
        y_pos = 0.95

        # Sequence info
        ax9.text(0.05, y_pos, f'Seq {sequence_id+1} Frame {frame_idx}', fontsize=11,
                transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'),
                fontweight='bold')
        y_pos -= 0.12

        # FG:BG ratio
        ax9.text(0.05, y_pos, f'FG:BG = {fg_ratio:.1f}:{bg_ratio:.1f}', fontsize=10,
                transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightcyan'))
        y_pos -= 0.10

        # FPS if available
        if fps is not None:
            ax9.text(0.05, y_pos, f'FPS: {fps:.1f}', fontsize=10,
                    transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
            y_pos -= 0.10

        # Layer weights if available (multi_layer separation only)
        if layer_weights is not None:
            layer_str = ':'.join([f'{w:.3f}' for w in layer_weights])
            ax9.text(0.05, y_pos, f'Layer weights: {layer_str}', fontsize=9,
                    transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightyellow'))
            y_pos -= 0.08

        # Depth metrics (computed on pixels where both GT and Pred are valid)
        if error_valid_mask.sum() > 0:
            valid_gt = torch.from_numpy(gt_depth[error_valid_mask])
            valid_pred = torch.from_numpy(pred_depth[error_valid_mask])

            rmse = torch.sqrt(torch.mean((valid_pred - valid_gt) ** 2))
            mae = torch.mean(torch.abs(valid_pred - valid_gt))

            threshold = 1.25
            max_ratio = torch.max(valid_pred / valid_gt, valid_gt / valid_pred)
            delta_1 = (max_ratio < threshold).float().mean()
            delta_2 = (max_ratio < threshold ** 2).float().mean()
            delta_3 = (max_ratio < threshold ** 3).float().mean()

            ax9.text(0.05, y_pos, f'AbsRel: {abs_rel:.4f}', fontsize=10,
                    transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightcoral'))
            y_pos -= 0.08
            ax9.text(0.05, y_pos, f'Delta_1: {delta_1:.3f}', fontsize=10,
                    transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
            y_pos -= 0.08
            ax9.text(0.05, y_pos, f'Delta_2: {delta_2:.3f}', fontsize=10,
                    transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
            y_pos -= 0.08
            ax9.text(0.05, y_pos, f'Delta_3: {delta_3:.3f}', fontsize=10,
                    transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
            y_pos -= 0.08
            ax9.text(0.05, y_pos, f'RMSE: {rmse:.3f}m', fontsize=9,
                    transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
            y_pos -= 0.08
            ax9.text(0.05, y_pos, f'MAE: {mae:.3f}m', fontsize=9,
                    transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightblue'))
            y_pos -= 0.08

        ax9.set_title('Depth Metrics', fontsize=14, fontweight='bold')
        ax9.axis('off')

        # ==================== Row 4: Depth Distribution, Importance Distribution ====================

        # 10. Depth Distribution Histogram
        ax10 = fig.add_subplot(gs[3, :2])
        if error_valid_mask.sum() > 0:
            gt_valid = gt_depth[error_valid_mask]
            pred_valid = pred_depth[error_valid_mask]

            bins = np.linspace(min(gt_valid.min(), pred_valid.min()),
                              max(gt_valid.max(), pred_valid.max()), 50)

            ax10.hist(gt_valid, bins=bins, alpha=0.6, label='Ground Truth',
                    color='blue', density=True)
            ax10.hist(pred_valid, bins=bins, alpha=0.6, label='Predicted',
                    color='red', density=True)
            ax10.set_xlabel('Depth (meters)', fontsize=12)
            ax10.set_ylabel('Density', fontsize=12)
            ax10.set_title('Depth Distribution', fontsize=14, fontweight='bold')
            ax10.legend(fontsize=12)
            ax10.grid(True, alpha=0.3)

        # 11. Importance Distribution
        ax11 = fig.add_subplot(gs[3, 2])
        importance_flat = importance_map.flatten()

        # Handle case where all values are identical (std=0)
        if imp_std < 1e-6:
            # Just show a vertical line at the constant value
            ax11.axvline(imp_mean, color='purple', linestyle='-', linewidth=3,
                        label=f'Constant: {imp_mean:.3f}')
            ax11.set_xlim(max(0, imp_mean - 0.1), min(1, imp_mean + 0.1))
            ax11.text(0.5, 0.5, f'All pixels = {imp_mean:.3f}\n(std = {imp_std:.6f})',
                     ha='center', va='center', transform=ax11.transAxes,
                     fontsize=14, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        else:
            # Normal histogram
            ax11.hist(importance_flat, bins=50, alpha=0.7, color='purple', density=True)
            ax11.axvline(imp_mean, color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {imp_mean:.3f}')

        ax11.set_xlabel('Importance Value', fontsize=12)
        ax11.set_ylabel('Density', fontsize=12)
        ax11.set_title('Importance Distribution', fontsize=14, fontweight='bold')
        ax11.legend(fontsize=10)
        ax11.grid(True, alpha=0.3)

        # Overall title
        plt.suptitle(f'Gear3 Upgrade: Sequence {sequence_id} Best Frame {frame_idx}',
                    fontsize=16, fontweight='bold')

        # Save with same naming convention
        save_path = self.save_dir / f"best_frame_seq{sequence_id}_{frame_idx}_absrel_{abs_rel:.4f}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        logger.info(f"Saved Gear3 Upgrade best frame visualization: {save_path}")

    def _aggregate_metrics(self, all_metrics):
        """Aggregate metrics across sequences"""
        metric_keys = all_metrics[0].keys()
        aggregated = {}

        for key in metric_keys:
            # Skip nested dictionaries (like object_wise metrics)
            if key == 'object_wise':
                continue

            values = [m[key] for m in all_metrics if key in m]
            if values:
                aggregated[key] = np.mean(values)

        return aggregated


@hydra.main(version_base=None, config_path="configs/gear3_upgrade", config_name="config")
def main(config: DictConfig):
    """Main entry point"""
    import os

    # Override config for testing
    config.inference = True

    tester = Gear3UpgradeTester(config)
    tester.test()


if __name__ == "__main__":
    main()
