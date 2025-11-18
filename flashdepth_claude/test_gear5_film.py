#!/usr/bin/env python3
"""
Test script for Gear5 FiLM: Temporal FiLM-style Feature Modulation

Key features:
    - Uses 2-layer CLS tokens [11, 23] for ViT-L or [5, 11] for ViT-S
    - FiLM-style modulation of DPT features before Mamba
    - Channel-wise gamma and beta parameters
    - Single forward pass (no 2-step structure)
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
from flashdepth.gear5_film_modules import Gear5FilmHead
from dataloaders.combined_dataset import CombinedDataset
from dataloaders.waymo_segmentation_dataset import WaymoSegmentationDataset, collate_fn as waymo_collate_fn
from dataloaders.urbansyn_dataset import UrbanSynDepth
from dataloaders.urbansyn_segmentation_dataset import UrbanSynSegmentationDataset, urbansyn_collate_fn
from dataloaders.vkitti_segmentation_dataset import VKITTISegmentationDataset, collate_fn as vkitti_collate_fn
from utils.metric_depth_metrics import MetricDepthMetrics, format_metrics
from utils.object_wise_evaluation import ObjectWiseMetrics
# from utils.object_wise_visualization import create_object_wise_grid  # Not used - see note in _save_best_frame_visualizations
from utils.helpers import save_gifs_as_grid, save_grid_to_mp4, depth_to_np_arr, torch_batch_to_np_arr
from utils.gear_common_helpers import depth_to_colored_frame
from utils.gear_video_utils import save_video as save_video_util
from utils.gear5_film_visualization import Gear5FilmVisualizer



def get_canonical_focal_length(config):
    """
    Get canonical focal length from config.

    Args:
        config: Configuration dict

    Returns:
        float: Canonical focal length (default 500.0 for 518×518 resolution)
    """
    return config.get('canonical_focal_length', 500.0)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Gear5FilmTester:
    """
    Test harness for Gear5 FiLM model.

    Evaluates on:
        - Metric depth metrics (MAE, RMSE, AbsRel, δ1/δ2/δ3)
        - Temporal Alignment Error (TAE)
        - Gamma/beta parameter analysis
    """
    def __init__(self, config):
        self.config = config
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Setup save directory - use results_dir if provided, otherwise use eval.outfolder
        save_dir_str = config.get('results_dir', config.eval.outfolder)
        self.save_dir = Path(save_dir_str)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Save directory: {self.save_dir}")

        # Detect phase from config
        # Phase: Determined by config directory name (gear5_film = Phase1, gear5_film/hybrid = Phase2)
        config_dir = config.get('config_dir', '')
        if 'hybrid' in str(config_dir).lower():
            self.phase = 2
        else:
            self.phase = 1

        logger.info(f"Testing Phase {self.phase}")

        # Object-wise evaluation configuration
        self.object_wise_enabled = config.get('object_wise', {}).get('enabled', False)
        self.object_wise_dataset = config.get('object_wise', {}).get('dataset', 'waymo')

        # Visualization control (master flag, default=True)
        self.enable_visualization = config.get('visualization', True)
        logger.info(f"Visualization: {'ENABLED' if self.enable_visualization else 'DISABLED (only JSON results)'}")

        # Frame interval for visualization (only applies to sequence.png, not video)
        # Can be overridden via command line: frame_interval=X
        self.frame_interval = self.config.get('frame_interval', None)

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

        # Setup visualizer
        self.visualizer = Gear5FilmVisualizer(save_dir=self.save_dir)

    def _setup_model(self):
        """Load trained Gear5 FiLM model with phase/step-specific configuration"""
        # Determine ViT size based on phase
        # Phase 1: Uses config's vit_size (typically 'vitl')
        # Phase 2: Uses 'vits' (hybrid with ViT-S+ViT-L)
        model_config = dict(self.config.model)
        if self.phase == 2:
            # Phase 2 (hybrid): Always use ViT-S as student model
            model_config['vit_size'] = 'vits'
            logger.info("Phase 2 (Hybrid): Using ViT-S for student model")
        else:
            # Phase 1: Use config's vit_size
            logger.info(f"Phase 1: Using ViT size from config: {model_config.get('vit_size', 'vitl')}")

        model_config['batch_size'] = 1
        model_config['use_metric_head'] = False

        model = FlashDepth(**model_config)

        # Add Gear5 FiLM head (FiLM-style modulation)
        embed_dim = 1024 if model.encoder == 'vitl' else 384
        dpt_dim = 256 if model.encoder == 'vitl' else 64

        model.gear5_film_head = Gear5FilmHead(
            embed_dim=embed_dim,
            dpt_dim=dpt_dim
        )

        # Enable attention weights storage for 2 layers
        # ViT-L: [11, 23] (middle 2 DPT layers)
        # ViT-S: [5, 11] (middle 2 DPT layers)
        target_blocks = {
            'vitl': [11, 23],
            'vits': [5, 11]
        }[model.encoder]

        for i, block in enumerate(model.pretrained.blocks):
            if i in target_blocks:
                block.attn.store_attn_weights = True
                logger.info(f"Enabled attention weights storage for block {i}")
            else:
                block.attn.store_attn_weights = False

        logger.info(f"2-layer CLS token extraction: blocks {target_blocks}")

        # Store target blocks and compute encoder_features indices
        intermediate_idx = model.intermediate_layer_idx[model.encoder]
        encoder_indices = [intermediate_idx.index(block) for block in target_blocks]
        self.encoder_indices = encoder_indices
        self.target_blocks = target_blocks
        logger.info(f"Encoder features indices: {encoder_indices} (for CLS token extraction)")

        # Load checkpoint
        checkpoint_path = self.config.get('load')
        if checkpoint_path and checkpoint_path != 'true':
            if os.path.exists(checkpoint_path):
                logger.info(f"Loading checkpoint from {checkpoint_path}")
                logger.info(f"Testing configuration: Phase {self.phase}")

                checkpoint = torch.load(checkpoint_path, map_location='cpu')

                # Extract state dict
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                # Remove module. prefix if present
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

                # Load state dict (strict=False to allow for missing/extra keys in hybrid models)
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

                if missing_keys:
                    logger.warning(f"Missing keys: {missing_keys[:10]}...")  # Show first 10
                if unexpected_keys:
                    logger.warning(f"Unexpected keys: {unexpected_keys[:10]}...")  # Show first 10

                logger.info(f"Loaded checkpoint successfully for Phase {self.phase}")

                # Log training info if available
                if 'global_step' in checkpoint:
                    logger.info(f"Checkpoint step: {checkpoint['global_step']}")
                if 'phase' in checkpoint:
                    logger.info(f"Checkpoint phase: {checkpoint['phase']}")
            else:
                logger.warning(f"Checkpoint {checkpoint_path} not found")
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        else:
            logger.warning("No checkpoint specified!")

        model = model.to(self.device)
        model.eval()

        return model

    def _get_actual_focal_length(self, dataset_name, image_shape):
        """
        Get actual focal length for a dataset based on intrinsics registry.

        Args:
            dataset_name (str): Dataset name
            image_shape (tuple): Image shape (B, T, C, H, W)

        Returns:
            float: Actual focal length in pixels
        """
        from utils.dataset_intrinsics import get_intrinsics_info, get_fallback_fx

        # Clean dataset name
        if isinstance(dataset_name, str):
            dataset_name = dataset_name.lower().replace('-', '_')

        # Get intrinsics from registry
        intrinsics_info = get_intrinsics_info(dataset_name)

        if intrinsics_info is None:
            # Fallback: Use width * 0.7
            width = image_shape[-1]
            fx = get_fallback_fx(width)
            logger.warning(f"No intrinsics for {dataset_name}, using fallback fx={fx:.1f}")
            return fx

        # Handle fixed focal length
        if intrinsics_info['type'] == 'fixed':
            return intrinsics_info['fx']

        # Handle computed focal length (e.g., dynamicreplica)
        if intrinsics_info['type'] == 'computed':
            if dataset_name in ['dynamicreplica', 'replica']:
                width = image_shape[-1]
                return width / 2.0
            else:
                width = image_shape[-1]
                return get_fallback_fx(width)

        # For per_frame/per_sequence types, use typical_fx if available
        if 'typical_fx' in intrinsics_info:
            return intrinsics_info['typical_fx']

        # Final fallback
        width = image_shape[-1]
        fx = get_fallback_fx(width)
        logger.warning(f"Could not determine fx for {dataset_name}, using fallback fx={fx:.1f}")
        return fx

    def _setup_test_loader(self):
        """Setup test data loader"""
        # Check if single sequence mode
        single_seq_path = self.config.get('single_sequence', None)

        # Check if whole-test mode (default: False)
        whole_seq_test = self.config.get('whole_seq_test', False)

        # Object-wise evaluation: use segmentation datasets
        if self.object_wise_enabled:
            video_length = int(self.config.get('vid_len', 50))  # Ensure integer
            resolution = self.config.get('resolution', self.config.eval.test_dataset_resolution)
            data_root = self.config.dataset.data_root

            if self.object_wise_dataset == 'waymo':
                # WaymoSegmentationDataset expects data_root to be waymo_seg directory
                waymo_data_root = str(Path(data_root) / 'waymo_seg')

                # For waymo_seg in objwise mode: use 20 frames (0-19 with annotation)
                if 'vid_len' not in self.config:
                    video_length = 20
                    logger.info(f"Auto-setting video_length=20 for waymo_seg objwise mode (frames with annotation)")

                # Set frame_interval to 2 for waymo_seg objwise
                if self.frame_interval is None:
                    self.frame_interval = 2
                    logger.info(f"Auto-setting frame_interval=2 for waymo_seg objwise visualization")

                # whole_seq_test controls which sequences to use:
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

                if whole_seq_test:
                    # Reload all sequences without filtering
                    test_dataset.sequences = test_dataset._load_sequences_unfiltered()
                    logger.info(f"Using all sequences (whole_seq_test=True): {len(test_dataset.sequences)} sequences")
                else:
                    # Already loaded with validation filtering (first 8 scenes only)
                    logger.info(f"Using validation subset (whole_seq_test=False): {len(test_dataset.sequences)} sequences (first 8 scenes, same as training val)")

                test_loader = DataLoader(
                    test_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=self.config.training.workers,
                    collate_fn=waymo_collate_fn
                )

            elif self.object_wise_dataset == 'urbansyn':
                # UrbanSynSegmentationDataset
                if 'vid_len' not in self.config:
                    video_length = 50
                    logger.info(f"Auto-setting video_length=50 for urbansyn objwise mode")

                test_dataset = UrbanSynSegmentationDataset(
                    data_root=data_root,
                    split='test',
                    video_length=video_length,
                    resolution=resolution,
                    max_frames=1000
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=self.config.training.workers,
                    collate_fn=urbansyn_collate_fn
                )
            elif self.object_wise_dataset == 'vkitti':
                only_clone = self.config.get('only_clone', True)
                test_dataset = VKITTISegmentationDataset(
                    data_root=data_root,
                    split='test',
                    video_length=video_length,
                    resolution=resolution,
                    only_clone=only_clone,
                    use_sliding_window=False  # One sequence per scene
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=self.config.training.workers,
                    collate_fn=vkitti_collate_fn
                )
                logger.info(f"Object-wise dataset: vkitti_seg (only_clone={only_clone})")
            else:
                raise ValueError(f"Unknown object_wise dataset: {self.object_wise_dataset}")

            logger.info(f"Object-wise test dataset: {len(test_dataset)} sequences")
            return test_loader

        # Single sequence mode
        if single_seq_path:
            # Not implemented for Gear5 FiLM (kept for consistency)
            raise NotImplementedError("Single sequence mode not implemented for Gear5 FiLM testing")

        # Normal mode: use CombinedDataset
        # Priority: 1) eval.test_datasets (CLI override), 2) whole_seq_test flag, 3) val_datasets

        # Check if eval.test_datasets is explicitly provided (CLI override)
        has_test_datasets_override = hasattr(self.config.eval, 'test_datasets') and self.config.eval.test_datasets is not None
        test_datasets = None

        if has_test_datasets_override:
            # Convert to list if string (Hydra may pass single dataset as string)
            test_datasets_raw = self.config.eval.test_datasets
            if isinstance(test_datasets_raw, str):
                test_datasets = [test_datasets_raw]
            else:
                test_datasets = list(test_datasets_raw)
            # Filter out empty strings
            test_datasets = [d for d in test_datasets if d]

            if len(test_datasets) > 0:
                # CLI override: use provided test_datasets regardless of whole_seq_test
                logger.info(f"Using CLI-specified datasets: {test_datasets}")
            else:
                test_datasets = None  # Empty list, treat as no override

        if test_datasets is None:
            if whole_seq_test:
                # Use all test datasets from config
                test_datasets = self.config.eval.test_datasets
                if isinstance(test_datasets, str):
                    test_datasets = [test_datasets]
                logger.info(f"Using ALL test datasets (whole_seq_test=True): {test_datasets}")
            else:
                # Use validation datasets (default behavior)
                # Default to all available datasets if val_datasets not specified
                default_val_datasets = ['sintel', 'waymo_seg', 'eth3d', 'urbansyn', 'unreal4k', 'tartanair']
                test_datasets = self.config.dataset.get('val_datasets', default_val_datasets)
                # Replace 'waymo' with 'waymo_seg' if present
                test_datasets = ['waymo_seg' if d == 'waymo' else d for d in test_datasets]
                logger.info(f"Using VALIDATION datasets (whole_seq_test=False): {test_datasets}")

        video_length = int(self.config.get('vid_len', 50))  # Ensure integer (Hydra may pass string)
        resolution = self.config.get('resolution', self.config.eval.test_dataset_resolution)  # Allow resolution override

        logger.info(f"Test datasets: {test_datasets}")
        logger.info(f"Video length: {video_length}")
        logger.info(f"Resolution: {resolution}")

        test_dataset = CombinedDataset(
            root_dir=self.config.dataset.data_root,
            enable_dataset_flags=test_datasets,
            resolution=resolution,
            split='test',
            video_length=video_length
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.training.workers,
            collate_fn=self._collate_fn
        )

        logger.info(f"Test dataset: {len(test_dataset)} sequences")
        return test_loader

    def _collate_fn(self, batch):
        """Custom collate function to filter out None values and convert tuple to dict"""
        # Filter out None values
        batch = [item for item in batch if item is not None]

        # If all items are None, skip this batch
        if len(batch) == 0:
            return None

        # CombinedDataset returns tuple for val/test splits
        # Convert to dict format for easier access
        if len(batch) > 0 and isinstance(batch[0], tuple):
            if len(batch[0]) == 8:  # New Metric3D format with fx_ratio and resize_ratio
                images, depths, focal_lengths_canonical, focal_lengths_actual, actual_valid_masks, fx_ratios, resize_ratios, names = zip(*batch)
                return {
                    'image': torch.stack(images, dim=0),
                    'depth': torch.stack(depths, dim=0),
                    'focal_lengths': torch.stack(focal_lengths_canonical, dim=0),  # Canonical (500.0)
                    'focal_lengths_actual': torch.stack(focal_lengths_actual, dim=0),  # Original focal lengths
                    'actual_valid_mask': torch.stack(actual_valid_masks, dim=0),
                    'fx_ratio': torch.stack(fx_ratios, dim=0),  # NEW: 500 / fx_actual
                    'resize_ratio': torch.stack(resize_ratios, dim=0),  # NEW: total resize ratio
                    'dataset_name': names
                }
            elif len(batch[0]) == 5:  # Old format with actual_valid_mask (backwards compatibility)
                images, depths, focal_lengths, actual_valid_masks, names = zip(*batch)
                return {
                    'image': torch.stack(images, dim=0),
                    'depth': torch.stack(depths, dim=0),
                    'focal_lengths': torch.stack(focal_lengths, dim=0),
                    'actual_valid_mask': torch.stack(actual_valid_masks, dim=0),
                    'dataset_name': names
                }
            else:  # Older format without actual_valid_mask (for backwards compatibility)
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
            # Add 'focal_lengths' key for compatibility (SegmentationDataset uses metric depth, no canonical transform)
            if 'focal_lengths_actual' in result and 'focal_lengths' not in result:
                result['focal_lengths'] = result['focal_lengths_actual']
            return result

        # Default behavior for other datasets
        return torch.utils.data.dataloader.default_collate(batch)

    def run_test(self):
        """Run testing on all sequences"""
        all_metrics = []
        all_object_wise_metrics = []

        for seq_id, batch in enumerate(tqdm(self.test_loader, desc="Testing")):
            if batch is None:
                continue

            metrics = self.test_sequence(batch, seq_id)
            all_metrics.append(metrics)

            # Collect object-wise metrics
            if self.object_wise_enabled and 'object_wise' in metrics:
                all_object_wise_metrics.append(metrics['object_wise'])

        # Compute average metrics
        if len(all_metrics) > 0:
            avg_metrics_raw = {}
            for key in all_metrics[0].keys():
                if key == 'object_wise':
                    continue
                values = [m[key] for m in all_metrics]
                avg_metrics_raw[key] = np.mean(values)

            # Reorder metrics: abs_rel, a1, a2, a3, fps, tae, f1, mae, rmse
            metric_order = ['abs_rel', 'a1', 'a2', 'a3', 'fps', 'tae', 'boundary_f1', 'mae', 'rmse']
            avg_metrics = {}
            for key in metric_order:
                if key in avg_metrics_raw:
                    avg_metrics[key] = avg_metrics_raw[key]
            # Add any remaining metrics not in the order list
            for key, value in avg_metrics_raw.items():
                if key not in avg_metrics:
                    avg_metrics[key] = value

            # Reorder per-sequence results
            reordered_metrics = []
            for result in all_metrics:
                reordered = {}
                # First add sequence_id if it exists
                if 'sequence_id' in result:
                    reordered['sequence_id'] = result['sequence_id']
                # Then add metrics in the desired order
                for key in metric_order:
                    if key in result:
                        reordered[key] = result[key]
                # Add any remaining keys
                for key, value in result.items():
                    if key not in reordered:
                        reordered[key] = value
                reordered_metrics.append(reordered)

            logger.info("=" * 80)
            logger.info("AVERAGE METRICS")
            logger.info("=" * 80)
            logger.info(format_metrics(avg_metrics))
            logger.info("=" * 80)

            # Save overall results
            results_path = self.save_dir / "test_results.json"
            with open(results_path, 'w') as f:
                json.dump(avg_metrics, f, indent=2)
            logger.info(f"Results saved to {results_path}")

            # Save per-sequence results
            per_sequence_path = self.save_dir / "per_sequence_results.json"
            with open(per_sequence_path, 'w') as f:
                json.dump(reordered_metrics, f, indent=2)
            logger.info(f"Per-sequence results saved to {per_sequence_path}")

            # Object-wise evaluation summary
            logger.info(f"DEBUG: object_wise_enabled={self.object_wise_enabled}, all_object_wise_metrics count={len(all_object_wise_metrics)}")
            if self.object_wise_enabled and len(all_object_wise_metrics) == 0:
                logger.warning(f"DEBUG: No object-wise metrics collected! Check if 'segmentations' key exists in batches.")

            if self.object_wise_enabled and len(all_object_wise_metrics) > 0:
                logger.info(f"DEBUG: Aggregating {len(all_object_wise_metrics)} object-wise metrics across sequences...")
                # Aggregate across all sequences
                aggregated_class_metrics = self.object_wise_metrics.aggregate_metrics(all_object_wise_metrics)

                # Print summary
                self.object_wise_metrics.print_summary(aggregated_class_metrics)

                # Save to JSON
                object_wise_path = self.save_dir / "object_wise_results.json"
                self.object_wise_metrics.save_results(
                    aggregated_class_metrics,
                    object_wise_path
                )
                logger.info(f"DEBUG: Saved object_wise_results.json to {object_wise_path}")

        else:
            logger.warning("No metrics computed!")

    @torch.no_grad()
    def test_sequence(self, batch, sequence_id):
        """Test on a single sequence"""
        # Handle both 'image' (CombinedDataset) and 'images' (WaymoSegmentationDataset) keys
        if 'images' in batch:
            images = batch['images'].to(self.device)  # [1, T, 3, H, W] or [T, 3, H, W]
            # Add batch dimension if missing
            if images.ndim == 4:
                images = images.unsqueeze(0)  # [1, T, 3, H, W]
        else:
            images = batch['image'].to(self.device)  # [1, T, 3, H, W]

        # Extract dataset name first for logging
        dataset_name = batch.get('dataset_name', 'unknown')
        if isinstance(dataset_name, (list, tuple)):
            dataset_name = dataset_name[0]
        dataset_name = dataset_name.lower() if isinstance(dataset_name, str) else 'unknown'

        # Handle both 'depths' (WaymoSegmentationDataset objwise) and 'depth' (CombinedDataset)
        if 'depths' in batch:
            gt_depth = batch['depths'].to(self.device)  # [1, T, H, W] - objwise mode
        else:
            gt_depth = batch['depth'].to(self.device)  # [1, T, H, W] or [T, H, W]

        # Handle both focal_lengths and focal_lengths_actual (CombinedDataset uses focal_lengths_actual)
        if 'focal_lengths_actual' in batch:
            focal_lengths = batch['focal_lengths_actual'].to(self.device)  # [1, T] or [T]
        else:
            focal_lengths = batch['focal_lengths'].to(self.device)  # [1, T], all 500.0 (canonical)

        # Get actual space valid mask if available
        if 'actual_valid_mask' in batch:
            actual_valid_mask = batch['actual_valid_mask'].to(self.device)  # [1, T, H, W]
        else:
            actual_valid_mask = None

        # Add batch dimension if missing
        if gt_depth.ndim == 3:
            gt_depth = gt_depth.unsqueeze(0)  # [1, T, H, W]

        # Add channel dimension if needed
        if gt_depth.ndim == 4:
            gt_depth = gt_depth.unsqueeze(2)  # [1, T, 1, H, W]

        B, T = images.shape[:2]
        assert B == 1, "Batch size must be 1 for testing"

        # Dataloader gives inverse depth (1/m) already in canonical space (fx=500), scale to 100/m
        gt_depth_inverse_100 = gt_depth * 100.0  # [1, T, 1, H, W] in canonical 100/m

        # NOTE: GT depth is already in canonical space (fx=500 at 518×518)
        # Canonical transformation is now handled in the dataloader
        CANONICAL_FX = get_canonical_focal_length(self.config)  # 500.0

        # Get fx_actual for de-canonical visualization
        # FIXED: Use actual per-frame focal lengths from batch instead of dataset-wide typical_fx
        # This is critical for datasets like Sintel where fx varies significantly per frame
        # (e.g., Sintel seq4 has fx=1120, but typical_fx fallback gives 715.4 - 36% error!)

        if 'focal_lengths_actual' in batch:
            # Best: Use per-frame actual focal lengths from dataloader
            fx_actual_tensor = batch['focal_lengths_actual'].to(self.device)  # [1, T]
            fx_actual_first = fx_actual_tensor[0, 0].item()  # First frame for logging
        elif 'fx_ratio' in batch:
            # Alternative: Compute from fx_ratio (fx_actual = CANONICAL_FX / fx_ratio)
            fx_ratio_tensor = batch['fx_ratio'].to(self.device)  # [1, T]
            fx_actual_tensor = CANONICAL_FX / fx_ratio_tensor  # [1, T]
            fx_actual_first = fx_actual_tensor[0, 0].item()
        else:
            # Fallback: Use dataset-wide typical_fx (less accurate for per-frame datasets)
            fx_actual_first = self._get_actual_focal_length(dataset_name, images.shape)
            fx_actual_tensor = torch.full((1, T), fx_actual_first, device=self.device)  # [1, T]
            logger.warning(f"Using fallback typical_fx={fx_actual_first:.1f} (may be inaccurate for per-frame datasets!)")

        # Compute de-canonical ratios (per-frame for accurate de-canonicalization)
        de_canonical_ratio_inverse = CANONICAL_FX / fx_actual_tensor  # [1, T] - For inverse depth space: canonical → actual
        de_canonical_ratio_metric = fx_actual_tensor / CANONICAL_FX   # [1, T] - For metric depth space: canonical → actual

        # Log for first frame
        logger.info(f"fx_actual (frame 0): {fx_actual_first:.1f} pixels")

        # Extract Metric3D canonicalization ratios from batch (if available)
        if 'fx_ratio' in batch and 'resize_ratio' in batch:
            fx_ratio = batch['fx_ratio'].to(self.device)  # [1, T] - focal length ratio (500 / fx_actual)
            resize_ratio = batch['resize_ratio'].to(self.device)  # [1, T] - total resize ratio
        else:
            # Fallback for datasets without Metric3D canonicalization
            fx_ratio = None
            resize_ratio = None

        # Use actual space valid mask from dataloader if available
        if actual_valid_mask is not None:
            canonical_gt_valid = actual_valid_mask.unsqueeze(2)  # [1, T, 1, H, W]
        else:
            MIN_INVERSE_CANONICAL = 100.0 / 70.0
            canonical_gt_valid = (gt_depth_inverse_100 > MIN_INVERSE_CANONICAL)  # [1, T, 1, H, W]

        # Storage for predictions (keep on GPU during FPS measurement to avoid .cpu() overhead)
        pred_depths_gpu = []
        gammas_gpu = []
        betas_gpu = []
        canonical_pred_valid_gpu = []  # Store canonical pred masks (on GPU)

        # Best frame tracking
        best_frame_idx = 0
        best_frame_abs_rel = float('inf')

        # Warmup run for FPS measurement
        logger.info(f"Warmup run for FPS measurement...")

        # Initialize Mamba sequence for warmup
        if hasattr(self.model, 'mamba'):
            self.model.mamba.start_new_sequence()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            img_warmup = images[0, 0].unsqueeze(0)  # [1, 3, H, W]
            h_warmup, w_warmup = img_warmup.shape[2:]
            patch_h_warmup = h_warmup // self.model.patch_size
            patch_w_warmup = w_warmup // self.model.patch_size

            # Extract features
            encoder_features_warmup = self.model.pretrained.get_intermediate_layers(
                img_warmup, self.model.intermediate_layer_idx[self.model.encoder]
            )

            # Extract 2-layer CLS tokens
            cls_tokens_list_warmup = [
                encoder_features_warmup[i][:, 0]
                for i in self.encoder_indices
            ]
            cls_tokens_multi_layer_warmup = [
                rearrange(cls_tokens, '(b t) d -> b t d', b=1, t=1)
                for cls_tokens in cls_tokens_list_warmup
            ]

            # Get DPT features
            dpt_features_warmup = self.model.depth_head.get_forward_features(
                encoder_features_warmup, patch_h_warmup, patch_w_warmup
            )

            # Extract attention weights for importance map
            attention_weights_list_warmup = [
                self.model.pretrained.blocks[block_idx].attn.attn_weights
                for block_idx in self.target_blocks
            ]

            # Apply FiLM modulation
            film_outputs_warmup = self.model.gear5_film_head(
                cls_tokens_multi_layer_warmup,
                attention_weights_list_warmup,
                dpt_features_warmup,
                patch_h_warmup, patch_w_warmup
            )
            path_1_modulated_warmup = film_outputs_warmup['path_1_modulated']
            gamma_warmup = film_outputs_warmup['gamma']
            beta_warmup = film_outputs_warmup['beta']

            # Apply Mamba
            path_1_temporal_warmup = self.model.dpt_features_to_mamba(
                input_shape=(1, 1, None, h_warmup, w_warmup),
                dpt_features=path_1_modulated_warmup,
                in_dpt_layer=0
            )

            # Final depth
            out_warmup = self.model.depth_head.scratch.output_conv1(path_1_temporal_warmup)
            out_warmup = F.interpolate(out_warmup, (h_warmup, w_warmup), mode="bilinear", align_corners=True)
            pred_depth_inverse_warmup = self.model.depth_head.scratch.output_conv2(out_warmup)

        del encoder_features_warmup, cls_tokens_list_warmup, cls_tokens_multi_layer_warmup
        del dpt_features_warmup, path_1_modulated_warmup, path_1_temporal_warmup
        del pred_depth_inverse_warmup, gamma_warmup, beta_warmup
        torch.cuda.empty_cache()

        # FPS measurement
        warmup_frames = min(10, T)
        start_time = None

        # Initialize Mamba sequence for actual test
        if hasattr(self.model, 'mamba'):
            self.model.mamba.start_new_sequence()

        # Process each frame
        for t in range(T):
            # Start timing after warmup frames
            if t == warmup_frames:
                torch.cuda.synchronize()
                import time
                start_time = time.time()

            img_t = images[0, t]  # [3, H, W]
            gt_t_inverse = gt_depth_inverse_100[0, t]  # [1, H, W]

            # Use BFloat16 for forward pass
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                h, w = img_t.shape[1:]
                patch_h, patch_w = h // self.model.patch_size, w // self.model.patch_size

                # Extract features from DINOv2
                encoder_features = self.model.pretrained.get_intermediate_layers(
                    img_t.unsqueeze(0), self.model.intermediate_layer_idx[self.model.encoder]
                )

                # Extract 2-layer CLS tokens
                cls_tokens_list = [
                    encoder_features[i][:, 0]
                    for i in self.encoder_indices
                ]
                # Reshape to [B, T, embed_dim] for each layer
                cls_tokens_multi_layer = [
                    rearrange(cls_tokens, '(b t) d -> b t d', b=1, t=1)
                    for cls_tokens in cls_tokens_list
                ]

                # Get DPT features (frozen)
                dpt_features = self.model.depth_head.get_forward_features(
                    encoder_features, patch_h, patch_w
                )

                # Extract attention weights for importance map
                attention_weights_list = [
                    self.model.pretrained.blocks[block_idx].attn.attn_weights
                    for block_idx in self.target_blocks
                ]

                # Apply FiLM modulation (trainable)
                film_outputs = self.model.gear5_film_head(
                    cls_tokens_multi_layer,  # List of [B, T, embed_dim]
                    attention_weights_list,  # List of 2 attention weights
                    dpt_features,  # List of 4 DPT features [B*T, dpt_dim, h, w]
                    patch_h, patch_w
                )
                path_1_modulated = film_outputs['path_1_modulated']
                gamma = film_outputs['gamma']
                beta = film_outputs['beta']
                importance_map = film_outputs['importance_map']

                # Apply Mamba to modulated features (trainable)
                path_1_temporal = self.model.dpt_features_to_mamba(
                    input_shape=(1, 1, None, h, w),
                    dpt_features=path_1_modulated,  # [B*T, dpt_dim, h, w]
                    in_dpt_layer=0
                )

                # Final depth prediction
                out = self.model.depth_head.scratch.output_conv1(path_1_temporal)  # Frozen
                out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
                pred_depth_inverse_100 = self.model.depth_head.scratch.output_conv2(out)  # Trainable [1, 1, H, W]

                # Save canonical pred mask (before de-canonicalization!)
                MIN_INVERSE_CANONICAL = 100.0 / 70.0
                canonical_pred_valid_t = (pred_depth_inverse_100 > MIN_INVERSE_CANONICAL)  # [1, 1, H, W]

                # De-canonicalization: convert from canonical space to actual space (inverse depth)
                pred_depth_inverse_100 = pred_depth_inverse_100 * de_canonical_ratio_inverse[0, t]  # [1, 1, H, W]

                # Interpolate prediction to GT resolution
                gt_t_shape = gt_t_inverse.shape[-2:]
                if pred_depth_inverse_100.shape[-2:] != gt_t_shape:
                    pred_depth_inverse_100 = F.interpolate(
                        pred_depth_inverse_100, size=gt_t_shape, mode="bilinear", align_corners=True
                    )

                # Convert to metric depth
                pred_depth_metric = 100.0 / (pred_depth_inverse_100[0] + 1e-8)  # [1, H, W]

            # End timing for FPS measurement (after last frame, like original FlashDepth)
            if t == T - 1 and start_time is not None:
                torch.cuda.synchronize()
                end_time = time.time()

            # List append: Keep on GPU during FPS measurement, move to CPU after
            # This prevents .cpu() overhead from affecting FPS while avoiding OOM on long sequences
            if start_time is None or t >= T - 1:
                # FPS measurement ended or not started - move to CPU immediately
                pred_depths_gpu.append(pred_depth_metric.cpu())
                gammas_gpu.append(gamma[0, 0].cpu())
                betas_gpu.append(beta[0, 0].cpu())
                canonical_pred_valid_gpu.append(canonical_pred_valid_t.cpu())
            else:
                # FPS measurement in progress - keep on GPU
                pred_depths_gpu.append(pred_depth_metric)
                gammas_gpu.append(gamma[0, 0])
                betas_gpu.append(beta[0, 0])
                canonical_pred_valid_gpu.append(canonical_pred_valid_t)

            # Release intermediate tensors to prevent GPU memory accumulation
            # Critical for long sequences (e.g., urbansyn 1000 frames)
            del encoder_features, cls_tokens_multi_layer, dpt_features
            del attention_weights_list, film_outputs, path_1_modulated, gamma, beta, importance_map
            del path_1_temporal, out, pred_depth_inverse_100, pred_depth_metric

        # Calculate FPS
        if start_time is not None:
            inference_time = end_time - start_time
            fps = (T - warmup_frames) / inference_time if inference_time > 0 else 0
            logger.info(f"Inference time: {inference_time:.4f}s for {T - warmup_frames} frames (warmup {warmup_frames} excluded)")
            logger.info(f"FPS: {fps:.2f} frames/second")
        else:
            fps = 0
            logger.warning(f"Too few frames ({T}) for FPS measurement (need > {warmup_frames})")

        # Stack predictions (mix of CPU and GPU tensors - move remaining GPU tensors to CPU)
        pred_depths = torch.stack([p.cpu() if p.is_cuda else p for p in pred_depths_gpu], dim=0)  # [T, 1, H, W] (CPU)
        gammas = torch.stack([g.cpu() if g.is_cuda else g for g in gammas_gpu], dim=0)  # [T, dpt_dim] (CPU)
        betas = torch.stack([b.cpu() if b.is_cuda else b for b in betas_gpu], dim=0)  # [T, dpt_dim] (CPU)
        canonical_pred_valid_all = [cpv.cpu() if cpv.is_cuda else cpv for cpv in canonical_pred_valid_gpu]

        # Clear GPU memory
        del pred_depths_gpu, gammas_gpu, betas_gpu, canonical_pred_valid_gpu
        torch.cuda.empty_cache()

        # Convert GT to metric depth for visualization
        # Move to CPU first to avoid OOM for long sequences (urbansyn 1000 frames)
        gt_depth_inverse_100_cpu = gt_depth_inverse_100[0].cpu()  # [T, 1, H, W] to CPU
        gt_depth_canonical = 100.0 / (gt_depth_inverse_100_cpu + 1e-8)  # [T, 1, H, W] (CPU)
        de_canonical_ratio = fx_actual_tensor[0].cpu() / CANONICAL_FX  # [T] (CPU)
        gt_depth_metric = gt_depth_canonical * de_canonical_ratio.view(T, 1, 1, 1)  # [T, 1, H, W] (CPU)

        # Compute metrics (already on CPU)
        pred_depths_cpu = pred_depths
        gt_depth_metric_cpu = gt_depth_metric

        frame_metrics = []
        for t in range(pred_depths.shape[0]):
            pred_frame = pred_depths_cpu[t, 0]  # [H, W]
            gt_frame = gt_depth_metric_cpu[t, 0]  # [H, W]

            # Create valid mask
            MAX_DEPTH = 70.0
            gt_valid_mask = (gt_frame > 0) & (gt_frame < MAX_DEPTH)
            pred_valid_mask = (pred_frame > 0) & (pred_frame < MAX_DEPTH)
            valid_mask = gt_valid_mask & pred_valid_mask

            # Debug logging for first frame
            if t == 0 and sequence_id == 0:
                logger.info(f"DEBUG Metrics - Frame {t}")
                logger.info(f"  GT depth range: [{gt_frame.min():.2f}, {gt_frame.max():.2f}] meters")
                logger.info(f"  Pred depth range: [{pred_frame.min():.2f}, {pred_frame.max():.2f}] meters")
                logger.info(f"  Valid pixels: {valid_mask.sum()} / {valid_mask.numel()} ({100*valid_mask.sum()/valid_mask.numel():.1f}%)")

            if valid_mask.sum() > 0:
                frame_metric = self.metrics.compute_metric_depth_metrics(
                    pred_frame,
                    gt_frame,
                    valid_mask
                )
                frame_metrics.append(frame_metric)

                # Track best frame
                if frame_metric['abs_rel'] < best_frame_abs_rel:
                    best_frame_abs_rel = frame_metric['abs_rel']
                    best_frame_idx = t

        # Average metrics
        if len(frame_metrics) == 0:
            logger.warning(f"No valid frames for sequence {sequence_id}")
            return {k: 0.0 for k in ["mae", "rmse", "abs_rel", "a1", "tae", "fps"]}

        metrics = {}
        for key in frame_metrics[0].keys():
            values = [m[key] for m in frame_metrics]
            metrics[key] = np.mean(values)

        # Compute TAE (Temporal Alignment Error)
        if len(pred_depths) > 1:
            tae_errors = []
            for t in range(len(pred_depths) - 1):
                pred_t = pred_depths_cpu[t, 0]
                pred_t_next = pred_depths_cpu[t + 1, 0]
                gt_t = gt_depth_metric_cpu[t, 0]
                gt_t_next = gt_depth_metric_cpu[t + 1, 0]

                MAX_DEPTH = 70.0
                valid_t = (gt_t > 0) & (gt_t < MAX_DEPTH) & (pred_t > 0) & (pred_t < MAX_DEPTH)
                valid_t_next = (gt_t_next > 0) & (gt_t_next < MAX_DEPTH) & (pred_t_next > 0) & (pred_t_next < MAX_DEPTH)
                valid_both = valid_t & valid_t_next

                if valid_both.sum() > 0:
                    pred_change = pred_t_next - pred_t
                    gt_change = gt_t_next - gt_t
                    tae = torch.abs(pred_change[valid_both] - gt_change[valid_both]).mean()
                    tae_errors.append(tae.item())

            metrics['tae'] = np.mean(tae_errors) if len(tae_errors) > 0 else 0.0
        else:
            metrics['tae'] = 0.0

        # Add FPS to metrics
        metrics['fps'] = fps

        # Object-wise evaluation
        seg_masks_np = None
        per_frame_class_metrics = []

        if self.object_wise_enabled:
            logger.info(f"DEBUG: object_wise_enabled=True, checking for segmentations in batch...")
            logger.info(f"DEBUG: Batch keys: {batch.keys()}")
            if 'segmentations' not in batch:
                logger.warning(f"DEBUG: 'segmentations' key NOT FOUND in batch! Cannot compute object-wise metrics.")

        if self.object_wise_enabled and 'segmentations' in batch:
            try:
                seg_masks = batch['segmentations'][0]  # [T, H, W]
                T_seg = seg_masks.shape[0]

                logger.info(f"Processing {T_seg} frames with segmentation")

                seg_masks_np = seg_masks.cpu().numpy() if isinstance(seg_masks, torch.Tensor) else seg_masks

                # Debug: Log unique class IDs in first frame
                if sequence_id == 0:
                    unique_classes = np.unique(seg_masks_np[0])
                    logger.info(f"DEBUG: Unique class IDs in first frame: {unique_classes}")
                    if self.object_wise_metrics:
                        class_names = [self.object_wise_metrics.classes.get(c, f'unknown_{c}') for c in unique_classes]
                        logger.info(f"DEBUG: Class names: {class_names}")

                for t in range(T_seg):
                    pred_frame = pred_depths_cpu[t, 0].numpy()
                    gt_frame = gt_depth_metric_cpu[t, 0].numpy()
                    seg_mask_frame = seg_masks_np[t]

                    if seg_mask_frame.shape != pred_frame.shape:
                        seg_mask_frame = cv2.resize(
                            seg_mask_frame.astype(np.int32),
                            (pred_frame.shape[1], pred_frame.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )
                        seg_masks_np[t] = seg_mask_frame

                    frame_class_metrics = self.object_wise_metrics.compute_metrics_per_class(
                        pred_depth=pred_frame,
                        gt_depth=gt_frame,
                        seg_mask=seg_mask_frame,
                        min_pixels=100
                    )
                    per_frame_class_metrics.append(frame_class_metrics)

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
        valid_mask = (gt_depth_metric > 0)  # [T, 1, H, W]

        # Visualize
        if self.enable_visualization and self.config.eval.get('save_grid', True):
            self._visualize_sequence(
                images[0], pred_depths, gt_depth_metric,
                valid_mask, sequence_id, metrics, fps, focal_lengths[0],
                gammas, betas
            )

        # Save video
        # Skip video for long sequences (urbansyn, unreal4k) to save time and disk space
        skip_video_datasets = ['urbansyn', 'unreal4k']
        should_save_video = dataset_name not in skip_video_datasets
        if self.enable_visualization and self.config.eval.get('out_video', True) and should_save_video:
            save_video_util(
                images[0], pred_depths, gt_depth_metric, valid_mask, sequence_id,
                save_dir=self.save_dir,
                config=self.config
            )
        elif not should_save_video:
            logger.info(f"Skipping video save for {dataset_name} (long sequence dataset)")

        # Save best frame visualizations
        if self.enable_visualization and len(frame_metrics) > 0:
            logger.info(f"Best frame for sequence {sequence_id}: Frame {best_frame_idx} (AbsRel={best_frame_abs_rel:.4f})")

            # Gear5 FiLM doesn't use layer fusion weights
            layer_weights = None
            logger.info(f"Gear5 FiLM uses FiLM-style modulation (no layer fusion weights)")

            # Get segmentation for best frame
            if self.object_wise_enabled and seg_masks_np is not None:
                if best_frame_idx < len(seg_masks_np):
                    seg_mask_for_viz = seg_masks_np[best_frame_idx]
                    class_metrics_for_viz = per_frame_class_metrics[best_frame_idx] if best_frame_idx < len(per_frame_class_metrics) else None
                    actual_frame_number = batch['frame_indices'][0][best_frame_idx] if 'frame_indices' in batch else best_frame_idx
                    logger.info(f"Best frame batch_idx={best_frame_idx}, actual_frame={actual_frame_number}")
                else:
                    seg_mask_for_viz = None
                    class_metrics_for_viz = None
                    actual_frame_number = best_frame_idx
            else:
                seg_mask_for_viz = None
                class_metrics_for_viz = None
                actual_frame_number = best_frame_idx

            # Create model_outputs dict for visualization
            model_outputs = {
                'pred_depth': pred_depths[best_frame_idx, 0],  # [H, W]
                'gamma': gammas[best_frame_idx],  # [dpt_dim]
                'beta': betas[best_frame_idx],    # [dpt_dim]
                'fx_ratio': fx_ratio[0, best_frame_idx].item() if fx_ratio is not None else None,
                'resize_ratio': resize_ratio[0, best_frame_idx].item() if resize_ratio is not None else None,
            }

            self._save_best_frame_visualizations(
                images[0, best_frame_idx],  # [3, H, W]
                gt_depth_metric[best_frame_idx, 0],  # [H, W]
                model_outputs,
                sequence_id,
                actual_frame_number,
                best_frame_abs_rel,
                fps,
                seg_mask_for_viz,
                class_metrics_for_viz,
                layer_weights,
                frame_metrics[best_frame_idx] if best_frame_idx < len(frame_metrics) else None,
                dataset_name=dataset_name
            )

        return metrics

    def _visualize_sequence(self, images, pred_depths, gt_depths,
                           valid_mask, sequence_id, metrics, fps=None, focal_lengths=None,
                           gammas=None, betas=None):
        """
        Create visualization grid for a sequence (test_gear5 style).

        Rows: Image, Metric Depth (Prediction), Metric Depth (GT), Gamma Distribution
        """
        T = images.shape[0]
        frames_to_show = min(10, T)

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
            # Convert BFloat16 to Float32 before numpy conversion
            img = images[t].permute(1, 2, 0).cpu().float().numpy()
            # Min-Max normalization (FlashDepth original method)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)
            axes[0, col].imshow(img)
            axes[0, col].set_title(f'Frame {t}')
            axes[0, col].axis('off')

            # Row 1: Predicted metric depth (invalid pixels = black)
            MAX_DEPTH = 70.0
            # Convert BFloat16 to Float32 before numpy conversion
            pred = pred_depths[t, 0].cpu().float().numpy()
            gt = gt_depths[t, 0].cpu().float().numpy()

            # Check if dataset is sparse (< 50% valid GT pixels)
            gt_exists = (gt > 0)
            gt_density = gt_exists.sum() / gt_exists.size
            is_sparse = gt_density < 0.5

            # GT valid mask (canonical 70m)
            gt_valid = (gt > 0) & (gt < MAX_DEPTH)

            if is_sparse:
                # Sparse dataset (waymo_seg): Show pred dense, GT sparse
                # 1. Find valid scan height range from GT
                valid_pixels_per_row = gt_exists.sum(axis=1)  # [H]
                min_valid_pixels_threshold = 10  # At least 10 GT pixels per row
                valid_rows = valid_pixels_per_row >= min_valid_pixels_threshold
                valid_row_indices = np.where(valid_rows)[0]

                if len(valid_row_indices) > 0:
                    min_valid_row = valid_row_indices.min()
                    max_valid_row = valid_row_indices.max()
                    height_mask = np.zeros_like(gt, dtype=bool)
                    height_mask[min_valid_row:max_valid_row+1, :] = True
                else:
                    height_mask = np.ones_like(gt, dtype=bool)

                # 2. Pred valid mask (canonical 70m) - DENSE
                pred_valid_depth = (pred > 0) & (pred < MAX_DEPTH)

                # 3. Show pred DENSE (all valid pixels within height range)
                pred_show_mask = height_mask & pred_valid_depth  # Dense prediction
            else:
                # Dense dataset (sintel): Just use GT valid mask
                pred_show_mask = gt_valid

            # Row 2: GT metric depth (invalid pixels = black)
            # IMPORTANT: Compute GT's vmin/vmax FIRST, then use for both Pred and GT
            gt_display = np.where(gt_valid, gt, np.nan)  # Invalid = NaN

            # Compute GT's percentile range (will be used for both Pred and GT)
            if gt_valid.sum() > 0:
                gt_vmin = np.nanpercentile(gt_display, 2)
                gt_vmax = np.nanpercentile(gt_display, 98)
            else:
                gt_vmin, gt_vmax = 0, 1

            # Row 1: Predicted metric depth (use GT's vmin/vmax for consistent normalization)
            pred_display = np.where(pred_show_mask, pred, np.nan)  # Invalid = NaN (will be black)
            cmap_pred = plt.cm.plasma_r.copy()
            cmap_pred.set_bad(color='black')  # NaN pixels = black
            axes[1, col].imshow(pred_display, cmap=cmap_pred, vmin=gt_vmin, vmax=gt_vmax)  # Use GT's range!
            axes[1, col].set_title(f'Pred (m)')
            axes[1, col].axis('off')

            # Display GT with its own range
            cmap_gt = plt.cm.plasma_r.copy()
            cmap_gt.set_bad(color='black')  # NaN pixels = black
            axes[2, col].imshow(gt_display, cmap=cmap_gt, vmin=gt_vmin, vmax=gt_vmax)  # plasma_r: near=bright, far=dark
            axes[2, col].set_title(f'GT (m)')
            axes[2, col].axis('off')

            # Row 3: Gamma distribution (histogram) - Gear5 FiLM specific
            if gammas is not None:
                # Convert BFloat16 to Float32 before numpy conversion
                gamma_t = gammas[t].cpu().float().numpy()  # [dpt_dim]
                axes[3, col].hist(gamma_t, bins=30, color='blue', alpha=0.7)
                axes[3, col].set_title(f'Gamma dist\nmean={gamma_t.mean():.3f}')
                axes[3, col].set_xlabel('Gamma')
                axes[3, col].set_ylabel('Count')
            else:
                axes[3, col].axis('off')

        # Add overall title with metrics (test_gear5 style)
        title_str = (
            f"Sequence {sequence_id} | "
            f"TAE: {metrics.get('tae', 0):.4f} | "
            f"AbsRel: {metrics.get('abs_rel', 0):.4f} | "
            f"δ1: {metrics.get('a1', 0):.4f} | "
            f"F1: {metrics.get('boundary_f1', 0):.3f}"
        )
        if fps is not None:
            title_str += f" | FPS: {fps:.1f}"

        # Add focal length and max depth info (test_gear5 style)
        if focal_lengths is not None:
            # Use first frame's focal length (resized)
            fx_value = focal_lengths[0].item()
            # Calculate max GT depth from valid region
            gt_max = 0.0
            if valid_mask.sum() > 0:
                # gt_depths: [T, 1, H, W], valid_mask: [T, 1, H, W]
                # Remove channel dimension from both and apply mask
                valid_mask_no_channel = valid_mask[:, 0] if valid_mask.ndim == 4 else valid_mask
                gt_depths_no_channel = gt_depths[:, 0] if gt_depths.ndim == 4 else gt_depths
                gt_valid_depths = gt_depths_no_channel[valid_mask_no_channel]  # Extract valid depths
                if len(gt_valid_depths) > 0:
                    gt_max = gt_valid_depths.max().item()

            # Show valid GT range (canonical 70m in actual space)
            use_canonical = self.config.get('use_canonical_space', False)
            if use_canonical:
                CANONICAL_FX = get_canonical_focal_length(self.config)
                de_canonical_ratio = fx_value / CANONICAL_FX
                MAX_DEPTH_ACTUAL = 70.0 * de_canonical_ratio  # canonical 70m → actual space
                title_str += f"\nresized_fx: {fx_value:.1f} | canonical: {use_canonical} (fx={CANONICAL_FX:.1f}) | max_gt: {gt_max:.1f}m | valid_range: <{MAX_DEPTH_ACTUAL:.1f}m"
            else:
                title_str += f"\nresized_fx: {fx_value:.1f} | canonical: False | max_gt: {gt_max:.1f}m | valid_range: <70.0m"

        # Add FiLM gamma/beta mean info (Gear5 FiLM specific)
        if gammas is not None and betas is not None:
            gamma_mean = gammas.mean().item()
            beta_mean = betas.mean().item()
            title_str += f"\ngamma_mean: {gamma_mean:.3f} | beta_mean: {beta_mean:.3f}"

        fig.suptitle(title_str, fontsize=14)

        plt.tight_layout()
        save_path = self.save_dir / f"sequence_{sequence_id:04d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Saved visualization: {save_path}")

    def _save_best_frame_visualizations(self, image, gt_depth, model_outputs,
                                        sequence_id, frame_idx, abs_rel, fps=None,
                                        seg_mask=None, class_metrics=None, layer_weights=None, frame_metrics=None,
                                        dataset_name='unknown'):
        """
        Save best frame visualization for Gear5 FiLM model

        Creates a comprehensive grid visualization with format:
            best_frame_seq{N}_{frame_idx}_absrel_{abs_rel:.4f}.png

        Layout:
            Row 1: Input Image | GT Depth | Pred Depth
            Row 2: Gamma Distribution | Beta Distribution | Gamma vs Beta
            Row 3: Valid Mask | Error Map | Depth Distribution
            Row 4: Object-wise metrics (if available)

        Args:
            image: [3, H, W] - RGB image
            gt_depth: [H, W] - Ground truth metric depth
            model_outputs: dict with keys:
                - 'pred_depth': [H, W] - Predicted metric depth
                - 'gamma': [dpt_dim] - FiLM gamma parameters
                - 'beta': [dpt_dim] - FiLM beta parameters
            sequence_id: int - Sequence index
            frame_idx: int - Frame index within sequence
            abs_rel: float - AbsRel metric for this frame
            fps: float - Optional FPS measurement
            seg_mask: [H, W] - Optional segmentation mask
            class_metrics: dict - Optional object-wise metrics
            frame_metrics: dict - Optional pre-computed metrics
        """
        pred_depth = model_outputs['pred_depth']
        gamma = model_outputs['gamma']
        beta = model_outputs['beta']

        # Convert tensors to numpy and move to CPU
        if isinstance(image, torch.Tensor):
            if image.shape[0] == 3:  # [3, H, W]
                image = image.permute(1, 2, 0)  # [H, W, 3]
            image = image.float().cpu().numpy()

        if isinstance(pred_depth, torch.Tensor):
            pred_depth = pred_depth.float().cpu().numpy()
        if isinstance(gt_depth, torch.Tensor):
            gt_depth = gt_depth.float().cpu().numpy()
        if isinstance(gamma, torch.Tensor):
            gamma = gamma.float().cpu().numpy()
        if isinstance(beta, torch.Tensor):
            beta = beta.float().cpu().numpy()

        # Min-Max normalization (FlashDepth original method)
        image_np = (image * 1.0 - image.min()) / (image.max() - image.min() + 1e-8)
        image_np = np.clip(image_np, 0, 1)

        # Get image size
        img_h, img_w = image_np.shape[:2]

        # Create separate valid masks for GT and Pred
        MAX_DEPTH = 70.0  # Same as training (100/70 = 1.43 inverse depth threshold)

        # Check if dataset is sparse (< 50% valid GT pixels)
        gt_exists = (gt_depth > 0)
        gt_density = gt_exists.sum() / gt_exists.size
        is_sparse = gt_density < 0.5

        # GT valid mask (canonical 70m)
        gt_valid_mask = (gt_depth > 0) & (gt_depth < MAX_DEPTH)  # GT valid pixels

        if is_sparse:
            # Sparse dataset (waymo_seg): Apply height mask
            # 1. Find valid scan height range from GT (to exclude sky/non-scanned regions)
            valid_pixels_per_row = gt_exists.sum(axis=1)  # [H]
            min_valid_pixels_threshold = 10  # At least 10 GT pixels per row
            valid_rows = valid_pixels_per_row >= min_valid_pixels_threshold
            valid_row_indices = np.where(valid_rows)[0]

            if len(valid_row_indices) > 0:
                min_valid_row = valid_row_indices.min()
                max_valid_row = valid_row_indices.max()
                height_mask = np.zeros_like(gt_depth, dtype=bool)
                height_mask[min_valid_row:max_valid_row+1, :] = True
            else:
                height_mask = np.ones_like(gt_depth, dtype=bool)

            # 2. Pred valid mask (canonical 70m) - DENSE
            pred_valid_depth = (pred_depth > 0) & (pred_depth < MAX_DEPTH)

            # 3. Show pred DENSE (all valid pixels within height range)
            pred_show_mask = height_mask & pred_valid_depth  # Dense prediction

            # 4. Error mask (both GT and Pred valid)
            error_valid_mask = gt_valid_mask & pred_valid_depth
        else:
            # Dense dataset (sintel): Just use GT valid mask
            pred_show_mask = gt_valid_mask
            pred_valid_depth = (pred_depth > 0) & (pred_depth < MAX_DEPTH)
            error_valid_mask = gt_valid_mask & pred_valid_depth

        # Calculate error (only where both GT and Pred are valid)
        abs_error = np.abs(pred_depth - gt_depth)
        abs_error_masked = np.where(error_valid_mask, abs_error, np.nan)

        # Calculate gamma/beta statistics
        gamma_mean = gamma.mean()
        gamma_std = gamma.std()
        beta_mean = beta.mean()
        beta_std = beta.std()

        # Decide on number of rows (3 or 4) based on object-wise availability
        num_rows = 4 if (self.object_wise_enabled and seg_mask is not None) else 3

        # Create figure with 4x3 or 3x3 grid layout
        fig = plt.figure(figsize=(15, 16 if num_rows == 4 else 12))
        gs = gridspec.GridSpec(num_rows, 3, figure=fig, hspace=0.3, wspace=0.3)

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
        pred_display = np.where(pred_show_mask, pred_depth, np.nan)  # Invalid = NaN (will be black)
        cmap_pred = plt.cm.plasma_r.copy()
        cmap_pred.set_bad(color='black')  # NaN pixels = black
        im3 = ax3.imshow(pred_display, cmap=cmap_pred, vmin=vmin, vmax=vmax)  # plasma_r: near=bright, far=dark
        ax3.set_title('Predicted Metric Depth (m)', fontsize=14, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # ==================== Row 2: Gamma, Beta, Gamma vs Beta ====================

        # 4. Gamma Distribution (histogram)
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.hist(gamma, bins=40, color='blue', alpha=0.7, edgecolor='black')
        ax4.set_title(f'Gamma Distribution\nmean={gamma_mean:.3f}, std={gamma_std:.3f}',
                     fontsize=14, fontweight='bold')
        ax4.set_xlabel('Gamma', fontsize=12)
        ax4.set_ylabel('Count', fontsize=12)
        ax4.grid(True, alpha=0.3)

        # 5. Beta Distribution (histogram)
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.hist(beta, bins=40, color='green', alpha=0.7, edgecolor='black')
        ax5.set_title(f'Beta Distribution\nmean={beta_mean:.3f}, std={beta_std:.3f}',
                     fontsize=14, fontweight='bold')
        ax5.set_xlabel('Beta', fontsize=12)
        ax5.set_ylabel('Count', fontsize=12)
        ax5.grid(True, alpha=0.3)

        # 6. Gamma vs Beta (2D scatter plot)
        ax6 = fig.add_subplot(gs[1, 2])
        scatter = ax6.scatter(gamma, beta, c=np.arange(len(gamma)), cmap='viridis', alpha=0.6, s=20)
        ax6.set_title('Gamma vs Beta\n(colored by channel index)', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Gamma', fontsize=12)
        ax6.set_ylabel('Beta', fontsize=12)
        ax6.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax6, fraction=0.046, pad=0.04)

        # ==================== Row 3: Valid Mask, Error Map, Depth Distribution ====================

        # 7. Valid Mask (combined GT and Pred)
        ax7 = fig.add_subplot(gs[2, 0])
        valid_combined = np.zeros((*gt_valid_mask.shape, 3))
        valid_combined[gt_valid_mask, 1] = 0.5  # GT valid = Green
        valid_combined[pred_valid_depth, 0] = 0.5  # Pred valid = Red
        valid_combined[error_valid_mask, :] = [1, 1, 0]  # Both valid = Yellow
        ax7.imshow(valid_combined)
        ax7.set_title('Valid Mask\n(GT=Green, Pred=Red, Both=Yellow)', fontsize=14, fontweight='bold')
        ax7.axis('off')

        # 8. Error Map (absolute error)
        ax8 = fig.add_subplot(gs[2, 1])
        cmap_error = plt.cm.hot_r.copy()
        cmap_error.set_bad(color='black')  # NaN pixels = black
        if error_valid_mask.sum() > 0:
            error_vmax = np.nanpercentile(abs_error_masked, 95)  # Cap at 95th percentile
        else:
            error_vmax = 1.0
        im8 = ax8.imshow(abs_error_masked, cmap=cmap_error, vmin=0, vmax=error_vmax)
        ax8.set_title('Absolute Error (m)\n(low=white, high=red)', fontsize=14, fontweight='bold')
        ax8.axis('off')
        plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)

        # 9. Depth Distribution (GT vs Pred)
        ax9 = fig.add_subplot(gs[2, 2])
        if gt_valid_mask.sum() > 0:
            ax9.hist(gt_depth[gt_valid_mask], bins=50, alpha=0.5, color='blue', label='GT', edgecolor='black')
        if pred_show_mask.sum() > 0:
            ax9.hist(pred_depth[pred_show_mask], bins=50, alpha=0.5, color='red', label='Pred', edgecolor='black')
        ax9.set_title('Depth Distribution', fontsize=14, fontweight='bold')
        ax9.set_xlabel('Depth (m)', fontsize=12)
        ax9.set_ylabel('Count', fontsize=12)
        ax9.legend(fontsize=12)
        ax9.grid(True, alpha=0.3)

        # ==================== Row 4: Object-wise (if available) ====================

        # NOTE: Inline object-wise visualization is disabled because create_object_wise_grid()
        # saves to file instead of returning an image. Object-wise metrics are still saved
        # separately to JSON files via object_wise_results.json
        #
        # if num_rows == 4 and self.object_wise_enabled and seg_mask is not None and class_metrics is not None:
        #     # Create object-wise grid (reuse from test_gear5.py)
        #     objwise_grid = create_object_wise_grid(
        #         image_np, pred_depth, gt_depth, seg_mask,
        #         class_metrics, self.object_wise_metrics,
        #         show_top_n=3
        #     )
        #
        #     # 10. Object-wise visualization (spans all 3 columns)
        #     ax10 = fig.add_subplot(gs[3, :])
        #     ax10.imshow(objwise_grid)
        #     ax10.set_title('Object-wise Top-3 Classes (by pixel count)', fontsize=14, fontweight='bold')
        #     ax10.axis('off')

        # ==================== Overall Title ====================

        # Compute frame metrics if not provided
        if frame_metrics is None and error_valid_mask.sum() > 0:
            frame_metrics = self.metrics.compute_metric_depth_metrics(
                torch.from_numpy(pred_depth),
                torch.from_numpy(gt_depth),
                torch.from_numpy(error_valid_mask)
            )

        # Build title string
        title_parts = [f"Seq {sequence_id} | Frame {frame_idx} | AbsRel={abs_rel:.4f}"]
        if frame_metrics:
            title_parts.append(f"δ1={frame_metrics.get('a1', 0):.3f}")
            title_parts.append(f"MAE={frame_metrics.get('mae', 0):.3f}m")
            title_parts.append(f"RMSE={frame_metrics.get('rmse', 0):.3f}m")
        if fps is not None:
            title_parts.append(f"FPS={fps:.1f}")

        title_str = " | ".join(title_parts)
        title_str += f"\nGamma: μ={gamma_mean:.3f} σ={gamma_std:.3f} | Beta: μ={beta_mean:.3f} σ={beta_std:.3f}"

        fig.suptitle(title_str, fontsize=16, fontweight='bold')

        # Save
        save_path = self.save_dir / f'best_frame_seq{sequence_id}_{frame_idx}_absrel_{abs_rel:.4f}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

        logger.info(f"Saved best frame: {save_path.name}")


@hydra.main(version_base=None, config_path="configs/gear5_film", config_name="config")
def main(config: DictConfig):
    """Main entry point"""
    import os

    # Override config for testing
    config.inference = True

    logger.info("Starting Gear5 FiLM testing...")

    # Enable object-wise evaluation if --objwise flag was passed
    # (flag is removed from sys.argv in __main__ block before Hydra processes it)
    if getattr(main, '_objwise_mode', False):
        OmegaConf.update(config, 'object_wise.enabled', True, merge=False)
        logger.info("Object-wise evaluation ENABLED via --objwise flag")

    logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    tester = Gear5FilmTester(config)
    tester.run_test()

    logger.info("Testing completed!")


if __name__ == "__main__":
    import sys

    # Handle --objwise flag BEFORE Hydra processes arguments
    # This prevents "unrecognized arguments" error
    objwise_mode = False
    if '--objwise' in sys.argv:
        objwise_mode = True
        sys.argv = [arg for arg in sys.argv if arg != '--objwise']

    # Store objwise_mode as function attribute so main() can access it
    main._objwise_mode = objwise_mode

    main()
