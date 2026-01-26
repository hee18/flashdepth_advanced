#!/usr/bin/env python3
"""
Test script for Gear5: Unified Single-Stage Temporal Scale Prediction

Key features:
    - Uses 2-layer CLS tokens [11, 23] for ViT-L or [5, 11] for ViT-S
    - GRU-based temporal scale and shift prediction
    - Importance map for attention-based weighting
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
from omegaconf import DictConfig, OmegaConf, ListConfig
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from flashdepth.model import FlashDepth
from flashdepth.gear5_modules import Gear5MetricHead, BankaiMetricHead
from dataloaders.combined_dataset import CombinedDataset
from dataloaders.waymo_segmentation_dataset import WaymoSegmentationDataset, collate_fn as waymo_collate_fn
from dataloaders.urbansyn_dataset import UrbanSynDepth
from dataloaders.urbansyn_segmentation_dataset import UrbanSynSegmentationDataset, urbansyn_collate_fn
from dataloaders.vkitti_segmentation_dataset import VKITTISegmentationDataset, collate_fn as vkitti_collate_fn
from utils.metric_depth_metrics import MetricDepthMetrics, format_metrics
from utils.object_wise_evaluation import ObjectWiseMetrics
from utils.object_wise_visualization import create_object_wise_grid
from utils.fgwise_evaluation import (
    FGWiseMetrics, aggregate_fgwise_metrics,
    save_fgwise_visualization, draw_fg_contours, create_depth_with_fg_overlay
)
from utils.helpers import save_gifs_as_grid, save_grid_to_mp4, depth_to_np_arr, torch_batch_to_np_arr
from utils.gear_common_helpers import depth_to_colored_frame
from utils.gear_video_utils import save_video as save_video_util
from utils.reprojection_tae import ReprojectionTAECalculator



def get_canonical_focal_length(config):
    """
    Get canonical focal length (fixed at 500.0 for all resolutions).

    Args:
        config: Configuration dict

    Returns:
        float: Canonical focal length (always 500.0)
    """
    return 500.0

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Gear5Tester:
    """
    Test harness for Gear5 unified model.

    Evaluates on:
        - Metric depth metrics (MAE, RMSE, AbsRel, δ1/δ2/δ3)
        - Temporal Alignment Error (TAE)
        - Importance map visualization
        - Scale/shift parameter analysis
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
        # Phase: Determined by config directory name (gear5 = Phase1, gear5/hybrid = Phase2)
        config_dir = config.get('config_dir', '')
        if 'hybrid' in str(config_dir).lower():
            self.phase = 2
        else:
            self.phase = 1

        # Bankai mode detection
        self.use_bankai = config.get('use_bankai', False)
        if self.use_bankai:
            self.bankai_phase = config.get('bankai_phase', 2)  # Default to phase 2 for testing
            logger.info(f"=== BANKAI MODE ===")
            logger.info(f"  Bankai Phase: {self.bankai_phase}")

        logger.info(f"Testing Phase {self.phase}")

        # Object-wise evaluation configuration
        self.object_wise_enabled = config.get('object_wise', {}).get('enabled', False)
        self.object_wise_dataset = config.get('object_wise', {}).get('dataset', 'waymo')

        # Visualization control (master flag, default=True)
        self.enable_visualization = config.get('visualization', True)
        logger.info(f"Visualization: {'ENABLED' if self.enable_visualization else 'DISABLED (only JSON results)'}")

        # Frame interval for visualization (only applies to scene.png, not video)
        # Can be overridden via command line: frame_interval=X
        self.frame_interval = self.config.get('frame_interval', None)

        # Figure export options
        self.export_best_figure = self.config.get('best_figure', False)
        self.export_frame = self.config.get('frame', None)  # Specific frame index (int or None)
        if self.export_best_figure:
            logger.info(f"Best-figure export ENABLED (will save best_frame ±4 intervals as individual images, frame_interval={self.frame_interval or 1})")
        if self.export_frame is not None:
            logger.info(f"Frame-specific export ENABLED (will save frame {self.export_frame} ±4 intervals as individual images, frame_interval={self.frame_interval or 1})")

        if self.object_wise_enabled:
            logger.info(f"Object-wise evaluation ENABLED for dataset: {self.object_wise_dataset}")
            self.object_wise_metrics = ObjectWiseMetrics(dataset_type=self.object_wise_dataset)
        else:
            self.object_wise_metrics = None

        # FGwise evaluation configuration
        self.fgwise_enabled = config.get('fg_wise', {}).get('enabled', False)
        if self.fgwise_enabled:
            data_root = config.dataset.get('data_root', '/home/cvlab/hsy/Datasets')
            logger.info(f"FG-wise evaluation ENABLED (data_root: {data_root})")
            # FGWiseMetrics will be created per-dataset in test method
            self.fgwise_data_root = data_root
        else:
            self.fgwise_data_root = None

        # Initialize model
        self.model = self._setup_model()

        # Setup test loader
        self.test_loader = self._setup_test_loader()

        # Setup metrics
        self.metrics = MetricDepthMetrics()

        # Setup reprojection TAE calculator
        data_root = config.dataset.get('data_root', '/home/cvlab/hsy/Datasets')
        self.reproj_tae_calculator = ReprojectionTAECalculator(data_root)
        logger.info(f"Reprojection TAE calculator initialized (supported: {self.reproj_tae_calculator.SUPPORTED_DATASETS})")

    def _setup_model(self):
        """Load trained Gear5 model with phase/step-specific configuration"""
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

        # Add Gear5 metric head (unified single-stage)
        model_embed_dim = 1024 if model.encoder == 'vitl' else 384

        # Get use_mamba_temporal from config (matches train_gear5.py)
        use_mamba_temporal = self.config.model.get('use_mamba_temporal', False)
        if use_mamba_temporal:
            logger.info("TemporalScalePredictor: Using Mamba2 for temporal modeling")
        else:
            logger.info("TemporalScalePredictor: Using GRU for temporal modeling")

        # Determine GSP embed_dim based on tsp_mode (same logic as train_gear5.py)
        tsp_mode = self.config.model.get('tsp_mode', 'auto')
        if tsp_mode == 'l':
            gsp_embed_dim = 1024
            logger.info(f"TSP mode 'l': Using TSP-L with 1024-dim CLS tokens (forced)")
        elif tsp_mode == 's':
            gsp_embed_dim = 384
            logger.info(f"TSP mode 's': Using TSP-S with 384-dim CLS tokens (forced)")
        else:  # auto
            gsp_embed_dim = model_embed_dim
            logger.info(f"TSP mode 'auto': Using TSP with {gsp_embed_dim}-dim CLS tokens")

        # Create metric head based on mode
        dpt_dim = 256 if model.encoder == 'vitl' else 64

        if self.use_bankai:
            # Bankai mode: Unified Mamba for temporal depth + metric prediction
            bankai_downsample = self.config.get('bankai_downsample', 0.1 if model.encoder == 'vitl' else 0.05)
            bankai_num_layers = self.config.get('bankai_num_mamba_layers', 4)
            use_hybrid_cls_fusion = (self.phase == 2 and model.hybrid_configs is not None)

            # Get d_state from config (must match FlashDepth for weight loading)
            bankai_d_state = self.config.model.get('mamba_d_state', 256)
            bankai_d_conv = self.config.model.get('mamba_d_conv', 4)

            model.gear5_metric_head = BankaiMetricHead(
                dpt_dim=dpt_dim,
                cls_embed_dim=gsp_embed_dim,
                num_mamba_layers=bankai_num_layers,
                downsample_factor=bankai_downsample,
                use_hybrid_cls_fusion=use_hybrid_cls_fusion,
                teacher_cls_dim=1024 if use_hybrid_cls_fusion else gsp_embed_dim,
                d_state=bankai_d_state,
                d_conv=bankai_d_conv
            )
            logger.info(f"=== BANKAI MetricHead Created ===")
            logger.info(f"  DPT dim: {dpt_dim}")
            logger.info(f"  CLS embed dim: {gsp_embed_dim}")
            logger.info(f"  Mamba layers: {bankai_num_layers}")
            logger.info(f"  Downsample factor: {bankai_downsample}")
            logger.info(f"  d_state: {bankai_d_state}, d_conv: {bankai_d_conv}")

            # Copy FlashDepth Mamba weights to UnifiedMamba BEFORE deleting original
            if hasattr(model, 'mamba'):
                self._copy_mamba_weights_to_unified(model)

            # Remove Original Mamba to save memory (Bankai's UnifiedMamba replaces it)
            if hasattr(model, 'mamba'):
                original_mamba_params = sum(p.numel() for p in model.mamba.parameters())
                del model.mamba
                model.use_mamba = False  # Prevent forward from using mamba
                logger.info(f"  Original Mamba REMOVED: {original_mamba_params:,} params freed")
        else:
            # Standard Gear5 mode
            model.gear5_metric_head = Gear5MetricHead(
                embed_dim=gsp_embed_dim,
                feature_dim=256,
                hidden_dim=128,
                use_mamba=use_mamba_temporal  # Support Mamba2 option
            )

        # Enable attention weights storage
        # CLS layer selection: user can specify which intermediate layers to use (1-4)
        # Default: [2, 4] (2nd and 4th intermediate layers)
        #
        # Mapping (1-indexed user input to 0-indexed intermediate_layer_idx):
        #   ViT-L: intermediate_layer_idx = [4, 11, 17, 23]
        #          Layer 1→block 4, Layer 2→block 11, Layer 3→block 17, Layer 4→block 23
        #   ViT-S: intermediate_layer_idx = [2, 5, 8, 11]
        #          Layer 1→block 2, Layer 2→block 5, Layer 3→block 8, Layer 4→block 11

        # Get cls_layers from config (default: [2, 4])
        cls_layers = self.config.get('cls_layers', [2, 4])

        # Convert OmegaConf ListConfig to plain Python list if needed
        if isinstance(cls_layers, ListConfig):
            cls_layers = OmegaConf.to_container(cls_layers)

        # Handle string input like '[2,4]' from command line
        if isinstance(cls_layers, str):
            # Remove brackets and split by comma
            cls_layers = cls_layers.strip('[]').split(',')
            cls_layers = [int(x.strip()) for x in cls_layers if x.strip()]

        # Ensure it's a flat list of integers
        if isinstance(cls_layers, (list, tuple)):
            cls_layers = [int(x) for x in cls_layers]
        else:
            cls_layers = [int(cls_layers)]  # Single value case

        # Validate cls_layers (must be 1-4)
        for layer in cls_layers:
            if layer < 1 or layer > 4:
                raise ValueError(f"cls_layers must be between 1 and 4, got {layer}")

        # Get intermediate_layer_idx for the encoder
        intermediate_idx = model.intermediate_layer_idx[model.encoder]

        # Convert user's 1-indexed layer numbers to actual block indices
        # cls_layers=[4] → encoder_indices=[3] → target_blocks=[23] for ViT-L
        # cls_layers=[2,4] → encoder_indices=[1,3] → target_blocks=[11,23] for ViT-L
        encoder_indices = [layer - 1 for layer in cls_layers]  # Convert to 0-indexed
        target_blocks = [intermediate_idx[idx] for idx in encoder_indices]

        logger.info(f"CLS layer selection: user specified layers {cls_layers}")
        logger.info(f"  → encoder_indices: {encoder_indices}")
        logger.info(f"  → target_blocks: {target_blocks} (actual ViT block indices)")

        for i, block in enumerate(model.pretrained.blocks):
            if i in target_blocks:
                block.attn.store_attn_weights = True
                logger.info(f"Enabled attention weights storage for block {i}")
            else:
                block.attn.store_attn_weights = False

        logger.info(f"{len(target_blocks)}-layer attention storage: blocks {target_blocks}")

        # Store target blocks and encoder_indices for CLS token extraction
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

    def _copy_mamba_weights_to_unified(self, model):
        """
        Copy FlashDepth Mamba weights to UnifiedMamba in BankaiMetricHead.

        Weight mapping:
            mamba.blocks.0.X.* → gear5_metric_head.unified_mamba.blocks.0.X.*
            mamba.final_layer.* → gear5_metric_head.unified_mamba.final_layer.*

        New components (random init, not copied):
            gear5_metric_head.unified_mamba.cls_proj.*
            gear5_metric_head.unified_mamba.scale_head.*
            gear5_metric_head.unified_mamba.shift_head.*
        """
        if not hasattr(model, 'mamba') or not hasattr(model, 'gear5_metric_head'):
            logger.warning("Cannot copy Mamba weights: model.mamba or gear5_metric_head not found")
            return

        unified_mamba = model.gear5_metric_head.unified_mamba
        original_mamba = model.mamba

        copied_count = 0
        skipped_count = 0

        # Copy blocks weights
        # FlashDepth: mamba.blocks[0][layer_idx] → UnifiedMamba: blocks[0][layer_idx]
        if hasattr(original_mamba, 'blocks') and len(original_mamba.blocks) > 0:
            src_blocks = original_mamba.blocks[0]  # First (and usually only) block group
            dst_blocks = unified_mamba.blocks[0]

            if len(src_blocks) == len(dst_blocks):
                for layer_idx in range(len(src_blocks)):
                    src_block = src_blocks[layer_idx]
                    dst_block = dst_blocks[layer_idx]

                    for (src_name, src_param), (dst_name, dst_param) in zip(
                        src_block.named_parameters(), dst_block.named_parameters()
                    ):
                        if src_param.shape == dst_param.shape:
                            dst_param.data.copy_(src_param.data)
                            copied_count += 1
                        else:
                            logger.warning(f"Shape mismatch: blocks.0.{layer_idx}.{src_name} "
                                         f"{src_param.shape} vs {dst_param.shape}")
                            skipped_count += 1
            else:
                logger.warning(f"Block count mismatch: {len(src_blocks)} vs {len(dst_blocks)}")

        # Copy final_layer weights
        if hasattr(original_mamba, 'final_layer') and hasattr(unified_mamba, 'final_layer'):
            for (src_name, src_param), (dst_name, dst_param) in zip(
                original_mamba.final_layer.named_parameters(),
                unified_mamba.final_layer.named_parameters()
            ):
                if src_param.shape == dst_param.shape:
                    dst_param.data.copy_(src_param.data)
                    copied_count += 1
                else:
                    logger.warning(f"Shape mismatch: final_layer.{src_name} "
                                 f"{src_param.shape} vs {dst_param.shape}")
                    skipped_count += 1

        logger.info(f"=== FlashDepth Mamba → UnifiedMamba Weight Copy ===")
        logger.info(f"  Copied: {copied_count} parameter tensors")
        logger.info(f"  Skipped (shape mismatch): {skipped_count}")
        logger.info(f"  New components (random init): cls_proj, scale_head, shift_head")

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

    def _extract_scene_frame_for_fgwise(self, image_path: str, dataset_name: str):
        """
        Extract scene and frame identifiers from image path for FG-wise evaluation.

        Each dataset has different path structure, so we need dataset-specific parsing.

        Args:
            image_path: Full path to the image file
            dataset_name: Name of the dataset (e.g., 'sintel', 'waymo_seg', 'bonn')
                          Can include scene path like 'waymo_seg/segment-xxx'

        Returns:
            Tuple (scene, frame) or (None, None) if parsing fails
        """
        import os
        try:
            parts = image_path.replace('\\', '/').split('/')
            # Extract base dataset name (e.g., 'waymo_seg/segment-xxx' -> 'waymo_seg')
            base_dataset_name = dataset_name.lower().split('/')[0]

            if base_dataset_name == 'eth3d':
                # /data/datasets/eth3d/{scene}/{image_name}.jpg
                # FG pattern: eth3d/{scene}/fg_masks/{frame}.png
                scene_idx = parts.index('eth3d') + 1 if 'eth3d' in parts else -1
                if scene_idx > 0 and scene_idx < len(parts):
                    scene = parts[scene_idx]
                    frame = os.path.splitext(parts[-1])[0]  # Remove extension
                    return scene, frame

            elif base_dataset_name == 'sintel':
                # /data/datasets/sintel/training/clean/{scene}/{frame}.png
                # FG pattern: sintel/fg_masks/training/clean/{scene}/{frame}.png
                if 'clean' in parts:
                    clean_idx = parts.index('clean')
                    if clean_idx + 2 < len(parts):
                        scene = parts[clean_idx + 1]
                        frame = os.path.splitext(parts[-1])[0]
                        return scene, frame

            elif base_dataset_name == 'waymo_seg':
                # /data/datasets/waymo_seg/val/{segment}/FRONT/{frame}.jpg
                # FG pattern: waymo_seg/val/{segment}/FRONT/fg_masks/{frame}.png
                if 'FRONT' in parts:
                    front_idx = parts.index('FRONT')
                    if front_idx >= 2:
                        segment = parts[front_idx - 1]
                        frame = os.path.splitext(parts[-1])[0]
                        return segment, frame

            elif base_dataset_name == 'vkitti':
                # /data/datasets/vkitti/{scene}/clone/frames/rgb/Camera_0/rgb_{frame}.jpg
                # FG pattern: vkitti/{scene}/clone/frames/fg_masks/Camera_0/fg_{frame}.png
                if 'vkitti' in parts and 'rgb' in parts[-1]:
                    vkitti_idx = parts.index('vkitti')
                    if vkitti_idx + 1 < len(parts):
                        scene = parts[vkitti_idx + 1]
                        # Extract frame number from rgb_{frame}.jpg → {frame}
                        filename = os.path.splitext(parts[-1])[0]
                        if filename.startswith('rgb_'):
                            frame = filename[4:]  # Remove 'rgb_' prefix
                            return scene, frame

            elif base_dataset_name == 'unreal4k':
                # /data/datasets/unreal4k/UnrealStereo4K_{scene}/{frame}.png
                # FG pattern: unreal4k/UnrealStereo4K_{scene}/fg_masks/{frame}.png
                for part in parts:
                    if part.startswith('UnrealStereo4K_'):
                        scene = part.replace('UnrealStereo4K_', '')
                        frame = os.path.splitext(parts[-1])[0]
                        return scene, frame

            elif base_dataset_name == 'urbansyn':
                # /data/datasets/urbansyn/{scene}/rgb/{frame}.png
                # FG pattern: urbansyn/{scene}/fg_masks/{frame}.png
                if 'urbansyn' in parts and 'rgb' in parts:
                    urbansyn_idx = parts.index('urbansyn')
                    rgb_idx = parts.index('rgb')
                    if urbansyn_idx + 1 == rgb_idx - 1:
                        scene = parts[urbansyn_idx + 1]
                        frame = os.path.splitext(parts[-1])[0]
                        return scene, frame

            elif base_dataset_name == 'tartanair':
                # /data/datasets/tartanair/{scene}/image_left/{frame}_left.png
                # FG pattern: tartanair/{scene}/fg_masks/{frame}.png
                if 'tartanair' in parts and 'image_left' in parts:
                    tartanair_idx = parts.index('tartanair')
                    if tartanair_idx + 1 < len(parts):
                        scene = parts[tartanair_idx + 1]
                        filename = os.path.splitext(parts[-1])[0]
                        # Remove _left suffix if present
                        if filename.endswith('_left'):
                            frame = filename[:-5]
                        else:
                            frame = filename
                        return scene, frame

            elif base_dataset_name == 'bonn':
                # /data/datasets/bonn/rgbd_bonn_{name}/rgb/{frame}.png
                # FG pattern: bonn/{scene}/fg_masks/{frame}.png where scene=rgbd_bonn_{name}
                for part in parts:
                    if part.startswith('rgbd_bonn_'):
                        scene = part  # Full name including rgbd_bonn_ prefix
                        frame = os.path.splitext(parts[-1])[0]
                        return scene, frame

            # Generic fallback: try to extract from parent directory structure
            if len(parts) >= 2:
                scene = parts[-2]
                frame = os.path.splitext(parts[-1])[0]
                return scene, frame

        except Exception as e:
            logger.warning(f"Failed to extract scene/frame from {image_path}: {e}")

        return None, None

    def _setup_test_loader(self):
        """Setup test data loader"""
        # Check if single sequence mode
        single_seq_path = self.config.get('single_sequence', None)

        # Check if whole-test mode (default: False)
        whole_seq_test = self.config.get('whole_seq_test', False)

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

                # Set frame_interval to 2 for waymo_seg objwise (show every 2nd frame in scene.png)
                if self.frame_interval is None:
                    self.frame_interval = 2
                    logger.info(f"Auto-setting frame_interval=2 for waymo_seg objwise mode")

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

                collate_fn = waymo_collate_fn
            elif self.object_wise_dataset == 'urbansyn':
                test_dataset = UrbanSynSegmentationDataset(
                    data_root=data_root,
                    split='test',
                    video_length=video_length,
                    resolution=resolution,
                    max_frames=1000
                )
                collate_fn = urbansyn_collate_fn
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
                collate_fn = vkitti_collate_fn
                logger.info(f"Object-wise dataset: vkitti_seg (only_clone={only_clone})")
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
            # Priority: 1) eval.test_datasets (CLI override), 2) whole_seq_test flag

            # Check if eval.test_datasets is explicitly provided (CLI override)
            has_test_datasets_override = hasattr(self.config.eval, 'test_datasets') and len(self.config.eval.test_datasets) > 0

            if has_test_datasets_override:
                # CLI override: use provided test_datasets regardless of whole_seq_test
                test_datasets = self.config.eval.test_datasets
                logger.info(f"Using CLI-specified datasets: {test_datasets}")
            elif whole_seq_test:
                # Use all test datasets from config
                test_datasets = self.config.eval.test_datasets
                logger.info("Using ALL test datasets (whole_seq_test=True)")
            else:
                # Use validation datasets with waymo_seg (first 8 sequences)
                # Default to all available datasets if val_datasets not specified
                default_val_datasets = ['sintel', 'waymo_seg', 'eth3d', 'urbansyn', 'unreal4k', 'tartanair']
                test_datasets = self.config.dataset.get('val_datasets', default_val_datasets)
                # Replace 'waymo' with 'waymo_seg' if present
                test_datasets = ['waymo_seg' if d == 'waymo' else d for d in test_datasets]
                logger.info(f"Using VALIDATION datasets (whole_seq_test=False): {test_datasets}")

            video_length = int(self.config.get('vid_len', 50))  # Ensure integer (Hydra may pass string)  # Get from config override
            resolution = self.config.get('resolution', self.config.eval.test_dataset_resolution)  # Allow resolution override

            logger.info(f"Test datasets: {test_datasets}")
            logger.info(f"Video length: {video_length}")
            logger.info(f"Resolution: {resolution}")

            # Always use CombinedDataset for test_gear5
            # Get limit_scenes from config if available (for NuScenes dataset limiting)
            limit_scenes = self.config.dataset.get('limit_scenes', None)
            if limit_scenes is not None:
                logger.info(f"Limiting scenes to: {limit_scenes}")

            # Get seq_list from config if available (for sequence filtering)
            seq_list = self.config.get('seq_list', None)
            if seq_list is not None:
                logger.info(f"Filtering to sequences: {seq_list}")

            test_dataset = CombinedDataset(
                root_dir=self.config.dataset.data_root,
                enable_dataset_flags=test_datasets,
                resolution=resolution,
                split='test',  # Use 'test' split (full test dataset)
                video_length=video_length,
                limit_scenes=limit_scenes,
                seq_list=seq_list,
                skip_gt_canonicalization=True  # GT returned in actual space; only pred needs de-canon
            )
            collate_fn = self._collate_fn

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.training.get('workers', 0),  # Use config workers (default 0 for testing)
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
                        elif dataset_name in ['vkitti']:
                            self.resolution = (1246, 378)  # 3.296 ratio, near original, 14x divisible
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
                        elif dataset_name in ['vkitti']:
                            self.resolution = (1246, 378)  # 3.296 ratio, near original, 14x divisible
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

        # CombinedDataset returns tuple for val/test splits
        # Convert to dict format for easier access
        if len(batch) > 0 and isinstance(batch[0], tuple):
            if len(batch[0]) == 9:  # Newest format with image_paths for FG-wise eval
                images, depths, focal_lengths_canonical, focal_lengths_actual, actual_valid_masks, fx_ratios, resize_ratios, names, image_paths = zip(*batch)
                return {
                    'image': torch.stack(images, dim=0),
                    'depth': torch.stack(depths, dim=0),
                    'focal_lengths': torch.stack(focal_lengths_canonical, dim=0),  # Canonical (500.0)
                    'focal_lengths_actual': torch.stack(focal_lengths_actual, dim=0),  # Original focal lengths
                    'actual_valid_mask': torch.stack(actual_valid_masks, dim=0),
                    'fx_ratio': torch.stack(fx_ratios, dim=0),  # 500 / fx_actual
                    'resize_ratio': torch.stack(resize_ratios, dim=0),  # total resize ratio
                    'dataset_name': names,
                    'image_paths': image_paths  # For FG-wise evaluation
                }
            elif len(batch[0]) == 8:  # Metric3D format without image_paths
                images, depths, focal_lengths_canonical, focal_lengths_actual, actual_valid_masks, fx_ratios, resize_ratios, names = zip(*batch)
                return {
                    'image': torch.stack(images, dim=0),
                    'depth': torch.stack(depths, dim=0),
                    'focal_lengths': torch.stack(focal_lengths_canonical, dim=0),  # Canonical (500.0)
                    'focal_lengths_actual': torch.stack(focal_lengths_actual, dim=0),  # Original focal lengths
                    'actual_valid_mask': torch.stack(actual_valid_masks, dim=0),
                    'fx_ratio': torch.stack(fx_ratios, dim=0),  # 500 / fx_actual
                    'resize_ratio': torch.stack(resize_ratios, dim=0),  # total resize ratio
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

        # Use default collate for dict items (training split)
        return torch.utils.data.dataloader.default_collate(batch)

    @torch.no_grad()
    def test(self):
        """Main testing loop"""
        logger.info("Starting testing...")

        all_metrics = []
        all_object_wise_metrics = []  # Track object-wise metrics separately
        all_fgwise_metrics = []  # Track FG-wise metrics separately
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

                # Extract and store FG-wise metrics
                if self.fgwise_enabled and 'fg_wise' in metrics:
                    all_fgwise_metrics.append(metrics['fg_wise'])

                # Clear GPU cache to prevent memory accumulation between sequences
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info(f"Cleared GPU cache after processing sequence {sequence_id}")

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

            # Reorder per-sequence results
            metric_order = ['abs_rel', 'a1', 'a2', 'a3', 'fps', 'tae', 'tae_reproj', 'tae_reproj_gt', 'mae', 'rmse',
                            'tsp_scale_mean', 'tsp_shift_mean', 'tsp_scale_std', 'tsp_shift_std',
                            'tsp_scale_max', 'tsp_scale_min', 'tsp_shift_max', 'tsp_shift_min']
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
                # Add any remaining keys (exclude _per_frame data and nested dicts)
                for key, value in result.items():
                    if key not in reordered and not key.startswith('_per_frame') and not isinstance(value, dict):
                        reordered[key] = value
                reordered_metrics.append(reordered)

            # Save per-sequence results
            per_sequence_path = self.save_dir / "per_sequence_results.json"
            with open(per_sequence_path, 'w') as f:
                json.dump(reordered_metrics, f, indent=2)
            logger.info(f"Per-sequence results saved to {per_sequence_path}")

            # Find and save best sequence (lowest abs_rel)
            best_seq_raw = min(all_metrics, key=lambda x: x['abs_rel'])
            # Reorder best_seq metrics
            best_seq = {}
            if 'sequence_id' in best_seq_raw:
                best_seq['sequence_id'] = best_seq_raw['sequence_id']
            for key in metric_order:
                if key in best_seq_raw:
                    best_seq[key] = best_seq_raw[key]
            for key, value in best_seq_raw.items():
                if key not in best_seq:
                    best_seq[key] = value

            best_seq_path = self.save_dir / "best_sequence.json"
            with open(best_seq_path, 'w') as f:
                json.dump(best_seq, f, indent=2)
            logger.info(f"\nBest sequence (lowest AbsRel): Sequence {best_seq['sequence_id']}")
            logger.info(f"  AbsRel: {best_seq['abs_rel']:.4f}")
            logger.info(f"  MAE: {best_seq['mae']:.4f}")
            logger.info(f"  RMSE: {best_seq['rmse']:.4f}")
            logger.info(f"  δ1: {best_seq['a1']:.4f}")
            logger.info(f"Best sequence saved to {best_seq_path}")

            # Find and save worst sequence (highest abs_rel)
            worst_seq_raw = max(all_metrics, key=lambda x: x['abs_rel'])
            worst_seq = {}
            if 'sequence_id' in worst_seq_raw:
                worst_seq['sequence_id'] = worst_seq_raw['sequence_id']
            for key in metric_order:
                if key in worst_seq_raw:
                    worst_seq[key] = worst_seq_raw[key]
            for key, value in worst_seq_raw.items():
                if key not in worst_seq:
                    worst_seq[key] = value

            worst_seq_path = self.save_dir / "worst_sequence.json"
            with open(worst_seq_path, 'w') as f:
                json.dump(worst_seq, f, indent=2)
            logger.info(f"\nWorst sequence (highest AbsRel): Sequence {worst_seq['sequence_id']}")
            logger.info(f"  AbsRel: {worst_seq['abs_rel']:.4f}")
            logger.info(f"  MAE: {worst_seq['mae']:.4f}")
            logger.info(f"  RMSE: {worst_seq['rmse']:.4f}")
            logger.info(f"  δ1: {worst_seq['a1']:.4f}")
            logger.info(f"Worst sequence saved to {worst_seq_path}")

            # Save scale/shift comparison JSON (TSP predictions vs per-frame optimal oracle)
            scale_shift_comparison = []
            for result in all_metrics:
                seq_id = result.get('sequence_id', -1)

                # Get per-frame values
                pred_scales = result.get('_per_frame_scales', [])
                pred_shifts = result.get('_per_frame_shifts', [])
                opt_scales = result.get('_per_frame_optimal_scales', [])
                opt_shifts = result.get('_per_frame_optimal_shifts', [])

                # Compute per-frame errors (TSP prediction vs per-frame optimal)
                per_frame_scale_errors = []
                per_frame_shift_errors = []
                if len(pred_scales) == len(opt_scales) and len(pred_scales) > 0:
                    per_frame_scale_errors = [abs(p - o) for p, o in zip(pred_scales, opt_scales)]
                    per_frame_shift_errors = [abs(p - o) for p, o in zip(pred_shifts, opt_shifts)]

                comparison_entry = {
                    'sequence_id': seq_id,
                    'abs_rel': result.get('abs_rel', 0.0),
                    'tsp_scale_mean': result.get('tsp_scale_mean', 1.0),
                    'tsp_shift_mean': result.get('tsp_shift_mean', 0.0),
                    'optimal_scale_mean': result.get('optimal_scale', 1.0),  # mean of per-frame optimal
                    'optimal_shift_mean': result.get('optimal_shift', 0.0),
                    # Per-frame error statistics
                    'scale_error_mean': float(np.mean(per_frame_scale_errors)) if per_frame_scale_errors else 0.0,
                    'scale_error_max': float(np.max(per_frame_scale_errors)) if per_frame_scale_errors else 0.0,
                    'shift_error_mean': float(np.mean(per_frame_shift_errors)) if per_frame_shift_errors else 0.0,
                    'shift_error_max': float(np.max(per_frame_shift_errors)) if per_frame_shift_errors else 0.0,
                }

                # Add per-frame details
                if pred_scales:
                    comparison_entry['per_frame_tsp_scales'] = pred_scales
                    comparison_entry['per_frame_optimal_scales'] = opt_scales
                    comparison_entry['per_frame_scale_errors'] = per_frame_scale_errors
                    # Scale drift analysis
                    if len(pred_scales) > 1:
                        scale_diffs = [abs(pred_scales[i+1] - pred_scales[i]) for i in range(len(pred_scales)-1)]
                        comparison_entry['scale_drift'] = {
                            'mean_frame_change': float(np.mean(scale_diffs)),
                            'max_frame_change': float(np.max(scale_diffs)),
                            'total_drift': float(abs(pred_scales[-1] - pred_scales[0]))
                        }

                if pred_shifts:
                    comparison_entry['per_frame_tsp_shifts'] = pred_shifts
                    comparison_entry['per_frame_optimal_shifts'] = opt_shifts
                    comparison_entry['per_frame_shift_errors'] = per_frame_shift_errors
                    # Shift drift analysis
                    if len(pred_shifts) > 1:
                        shift_diffs = [abs(pred_shifts[i+1] - pred_shifts[i]) for i in range(len(pred_shifts)-1)]
                        comparison_entry['shift_drift'] = {
                            'mean_frame_change': float(np.mean(shift_diffs)),
                            'max_frame_change': float(np.max(shift_diffs)),
                            'total_drift': float(abs(pred_shifts[-1] - pred_shifts[0]))
                        }

                scale_shift_comparison.append(comparison_entry)

            scale_shift_path = self.save_dir / "scale_shift_comparison.json"
            with open(scale_shift_path, 'w') as f:
                json.dump(scale_shift_comparison, f, indent=2)
            logger.info(f"Scale/shift comparison saved to {scale_shift_path}")

            # Save depth range analysis JSON
            depth_range_analysis = []
            for result in all_metrics:
                if '_depth_range_analysis' in result:
                    entry = {
                        'sequence_id': result.get('sequence_id', -1),
                        'abs_rel': result.get('abs_rel', 0.0),
                        'depth_ranges': result['_depth_range_analysis']
                    }
                    depth_range_analysis.append(entry)

            if depth_range_analysis:
                # Aggregate across sequences
                aggregated_ranges = {}
                for range_name in ['0-10m', '10-30m', '30-70m']:
                    range_abs_rels = [e['depth_ranges'].get(range_name, {}).get('abs_rel', 0) for e in depth_range_analysis]
                    range_a1s = [e['depth_ranges'].get(range_name, {}).get('a1', 0) for e in depth_range_analysis]
                    range_pixels = [e['depth_ranges'].get(range_name, {}).get('pixel_count', 0) for e in depth_range_analysis]
                    aggregated_ranges[range_name] = {
                        'abs_rel': float(np.mean(range_abs_rels)) if range_abs_rels else 0.0,
                        'a1': float(np.mean(range_a1s)) if range_a1s else 0.0,
                        'total_pixels': int(np.sum(range_pixels))
                    }

                depth_range_result = {
                    'aggregated': aggregated_ranges,
                    'per_sequence': depth_range_analysis
                }
                depth_range_path = self.save_dir / "depth_range_analysis.json"
                with open(depth_range_path, 'w') as f:
                    json.dump(depth_range_result, f, indent=2)
                logger.info(f"Depth range analysis saved to {depth_range_path}")

            # Save temporal analysis JSON (using reprojection TAE)
            temporal_analysis = []
            for result in all_metrics:
                entry = {
                    'sequence_id': result.get('sequence_id', -1),
                    'tae_reproj': result.get('tae_reproj', 0.0),
                    'tae_reproj_gt': result.get('tae_reproj_gt', 0.0),
                    'tae_simple': result.get('tae', 0.0),  # Keep simple TAE for reference
                    'per_frame_tae': result.get('_per_frame_tae', []),
                    'tae_spike_frames': result.get('_tae_spike_frames', []),
                    'tae_spike_count': len(result.get('_tae_spike_frames', []))
                }
                temporal_analysis.append(entry)

            if temporal_analysis:
                temporal_path = self.save_dir / "temporal_analysis.json"
                with open(temporal_path, 'w') as f:
                    json.dump(temporal_analysis, f, indent=2)
                logger.info(f"Temporal analysis saved to {temporal_path}")

            # Aggregate and save object-wise metrics
            logger.info(f"DEBUG: object_wise_enabled={self.object_wise_enabled}, all_object_wise_metrics count={len(all_object_wise_metrics)}")
            if self.object_wise_enabled and len(all_object_wise_metrics) == 0:
                logger.warning(f"DEBUG: No object-wise metrics collected! Check if 'segmentations' key exists in batches.")

            if self.object_wise_enabled and all_object_wise_metrics:
                logger.info(f"DEBUG: Aggregating {len(all_object_wise_metrics)} object-wise metrics across sequences...")
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
                logger.info(f"DEBUG: Saved object_wise_results.json to {object_wise_path}")

            # Aggregate and save FG-wise metrics
            if self.fgwise_enabled and all_fgwise_metrics:
                logger.info("\n" + "="*80)
                logger.info("FG-WISE EVALUATION RESULTS")
                logger.info("="*80)

                # Aggregate FG-wise metrics across all sequences
                aggregated_fgwise = aggregate_fgwise_metrics(all_fgwise_metrics)

                # Print summary
                fg_abs_rel = aggregated_fgwise.get('fg_abs_rel', float('nan'))
                bg_abs_rel = aggregated_fgwise.get('bg_abs_rel', float('nan'))
                fg_a1 = aggregated_fgwise.get('fg_a1', float('nan'))
                bg_a1 = aggregated_fgwise.get('bg_a1', float('nan'))
                fg_pixels = aggregated_fgwise.get('fg_num_pixels', 0)
                bg_pixels = aggregated_fgwise.get('bg_num_pixels', 0)

                logger.info(f"Foreground (FG) metrics:")
                logger.info(f"  AbsRel: {fg_abs_rel:.4f}")
                logger.info(f"  δ1: {fg_a1:.4f}")
                logger.info(f"  Pixels: {fg_pixels:,}")
                logger.info(f"Background (BG) metrics:")
                logger.info(f"  AbsRel: {bg_abs_rel:.4f}")
                logger.info(f"  δ1: {bg_a1:.4f}")
                logger.info(f"  Pixels: {bg_pixels:,}")

                if 'fg_bg_absrel_ratio' in aggregated_fgwise:
                    logger.info(f"FG/BG AbsRel ratio: {aggregated_fgwise['fg_bg_absrel_ratio']:.4f}")

                # Save to JSON
                fgwise_path = self.save_dir / "fgwise_results.json"
                with open(fgwise_path, 'w') as f:
                    json.dump(aggregated_fgwise, f, indent=2)
                logger.info(f"Saved FG-wise results to {fgwise_path}")

                # Also add to avg_metrics for unified output
                avg_metrics['fg_wise'] = aggregated_fgwise

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

        # Extract dataset name first for logging
        dataset_name = batch.get('dataset_name', 'unknown')
        if isinstance(dataset_name, (list, tuple)):
            dataset_name = dataset_name[0]
        dataset_name = dataset_name.lower() if isinstance(dataset_name, str) else 'unknown'

        # Handle both 'depths' (WaymoSegmentationDataset objwise) and 'depth' (CombinedDataset)
        if 'depths' in batch:
            gt_depth = batch['depths']  # [1, T, H, W] - objwise mode
        else:
            gt_depth = batch['depth']  # [1, T, H, W] or [T, H, W] - val split

        # Handle both focal_lengths and focal_lengths_actual (CombinedDataset uses focal_lengths_actual)
        if 'focal_lengths_actual' in batch:
            focal_lengths = batch['focal_lengths_actual'].to(self.device)  # [1, T] or [T]
        else:
            focal_lengths = batch['focal_lengths'].to(self.device)  # [1, T], all 500.0 (canonical)

        # Get actual space valid mask if available (from updated CombinedDataset)
        if 'actual_valid_mask' in batch:
            actual_valid_mask = batch['actual_valid_mask'].to(self.device)  # [1, T, H, W]
        else:
            # Fallback for datasets without actual_valid_mask (e.g., WaymoSegmentationDataset)
            actual_valid_mask = None

        # Get depth file paths for completed depth loading (visualization)
        depth_paths = batch.get('depth_paths', None)  # List[str] or None

        # Add batch dimension if missing (WaymoSegmentationDataset returns [T, H, W])
        if gt_depth.ndim == 3:
            gt_depth = gt_depth.unsqueeze(0)  # [1, T, H, W]

        # Add channel dimension if needed
        if gt_depth.ndim == 4:
            gt_depth = gt_depth.unsqueeze(2)  # [1, T, 1, H, W]

        B, T = images.shape[:2]
        assert B == 1, "Batch size must be 1 for testing"

        # Dataloader gives inverse depth (1/m) in actual space (skip_gt_canonicalization=True), scale to 100/m
        gt_depth_inverse_100 = gt_depth * 100.0  # [1, T, 1, H, W] in actual 100/m

        # NOTE: GT depth is now in actual space (skip_gt_canonicalization=True)
        # Only prediction needs de-canonicalization (pred is still in canonical space)
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
            dataset_name = batch.get('dataset_name', 'unknown')
            if isinstance(dataset_name, (list, tuple)):
                dataset_name = dataset_name[0]
            fx_actual_first = self._get_actual_focal_length(dataset_name, images.shape)
            fx_actual_tensor = torch.full((1, T), fx_actual_first, device=self.device)  # [1, T]
            logger.warning(f"Using fallback typical_fx={fx_actual_first:.1f} (may be inaccurate for per-frame datasets!)")

        # Extract Metric3D canonicalization ratios from batch (if available)
        # Must be done BEFORE computing de_canonical_ratio
        if 'fx_ratio' in batch and 'resize_ratio' in batch:
            fx_ratio = batch['fx_ratio'].to(self.device)  # [1, T] - focal length ratio (500 / fx_actual)
            resize_ratio = batch['resize_ratio'].to(self.device)  # [1, T] - total resize ratio
        else:
            # Fallback for datasets without Metric3D canonicalization
            fx_ratio = None
            resize_ratio = None

        # Compute de-canonical ratios (per-frame for accurate de-canonicalization)
        # De-canonicalization is the inverse of canonicalization:
        #   Canonicalization: inverse_canonical = inverse_actual × (resize_ratio / fx_ratio)
        #   De-canonicalization: inverse_actual = inverse_canonical × (fx_ratio / resize_ratio)
        if fx_ratio is not None and resize_ratio is not None:
            # Correct: use fx_ratio / resize_ratio
            # fx_ratio = 500 / fx_actual (e.g., 0.147 for ETH3D)
            # resize_ratio = total resize (e.g., 0.130 for ETH3D base resolution)
            # de_canonical_ratio = 0.147 / 0.130 ≈ 1.13
            de_canonical_ratio_inverse = fx_ratio / resize_ratio  # [1, T] - canonical → actual
            logger.info(f"Using Metric3D de-canonicalization: fx_ratio={fx_ratio[0,0].item():.4f}, resize_ratio={resize_ratio[0,0].item():.4f}, ratio={de_canonical_ratio_inverse[0,0].item():.4f}")
        else:
            # Fallback: assume resize_ratio ≈ fx_ratio (no correction needed)
            de_canonical_ratio_inverse = CANONICAL_FX / fx_actual_tensor  # [1, T]
            logger.warning(f"Fallback de-canonicalization (no resize_ratio): 500 / {fx_actual_first:.1f} = {de_canonical_ratio_inverse[0,0].item():.4f}")

        de_canonical_ratio_metric = 1.0 / de_canonical_ratio_inverse  # [1, T] - For metric depth space

        # Log for first frame
        logger.info(f"fx_actual (frame 0): {fx_actual_first:.1f} pixels")

        # Use actual space valid mask from dataloader if available
        # Otherwise fallback to computing from canonical depth
        if actual_valid_mask is not None:
            # Use actual space mask (<70m in actual space, computed before canonical transform)
            canonical_gt_valid = actual_valid_mask.unsqueeze(2)  # [1, T, 1, H, W]
        else:
            # Fallback: compute from canonical depth (70m threshold in canonical space)
            MIN_INVERSE_CANONICAL = 100.0 / 70.0
            canonical_gt_valid = (gt_depth_inverse_100 > MIN_INVERSE_CANONICAL)  # [1, T, 1, H, W]

        # Storage for predictions (keep on GPU during FPS measurement to avoid .cpu() overhead)
        pred_depths_gpu = []
        importance_maps_gpu = []
        scales_gpu = []
        shifts_gpu = []
        canonical_pred_valid_gpu = []  # Store canonical pred masks (on GPU)

        # Best/Worst frame tracking
        best_frame_idx = 0
        best_frame_abs_rel = float('inf')
        worst_frame_idx = 0
        worst_frame_abs_rel = 0.0

        # Warmup run for FPS measurement
        logger.info(f"Warmup run for FPS measurement...")

        # Initialize Mamba sequence for warmup
        if hasattr(self.model, 'mamba') and not self.use_bankai:
            # Standard Gear5: Use FlashDepth's original Mamba
            self.model.mamba.start_new_sequence()
        # Note: Bankai's UnifiedMamba handles sequence initialization internally

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            img_warmup = images[0, 0].unsqueeze(0)  # [1, 3, H, W]
            h_warmup, w_warmup = img_warmup.shape[2:]
            patch_h_warmup = h_warmup // self.model.patch_size
            patch_w_warmup = w_warmup // self.model.patch_size

            # Extract features from encoder
            encoder_features_warmup = self.model.pretrained.get_intermediate_layers(
                img_warmup, self.model.intermediate_layer_idx[self.model.encoder]
            )

            # Extract 2-layer CLS tokens
            cls_tokens_list_warmup = [
                encoder_features_warmup[i][:, 0]  # CLS token: [1, embed_dim]
                for i in self.encoder_indices
            ]
            # Average and reshape for GRU: [1, 1, 1024]
            cls_tokens_averaged_warmup = torch.stack(cls_tokens_list_warmup, dim=1).mean(dim=1)  # [1, 1024]
            cls_tokens_warmup = cls_tokens_averaged_warmup.view(1, 1, -1)  # [1, 1, 1024]

            # Get attention weights from 2 layers
            attention_weights_list_warmup = [
                self.model.pretrained.blocks[block_idx].attn.attn_weights
                for block_idx in self.target_blocks
            ]

            # Get DPT features (frozen)
            dpt_features_warmup = self.model.depth_head.get_forward_features(
                encoder_features_warmup, patch_h_warmup, patch_w_warmup
            )
            path_1_warmup = dpt_features_warmup[-1]

            if self.use_bankai:
                # ========== BANKAI MODE WARMUP ==========
                gear5_outputs_warmup = self.model.gear5_metric_head(
                    path_1=path_1_warmup,
                    cls_tokens=cls_tokens_warmup,
                    attention_weights_list=attention_weights_list_warmup,
                    input_shape=(1, 1, 3, h_warmup, w_warmup)
                )
                path_1_temporal_warmup = gear5_outputs_warmup['spatial_out']

                # Get relative depth
                out_warmup = self.model.depth_head.scratch.output_conv1(path_1_temporal_warmup)
                out_warmup = F.interpolate(out_warmup, (h_warmup, w_warmup), mode="bilinear", align_corners=True)
                relative_depth_warmup = self.model.depth_head.scratch.output_conv2(out_warmup)
            else:
                # ========== STANDARD GEAR5 WARMUP ==========
                # Apply Mamba temporal processing (frozen)
                path_1_temporal_warmup = self.model.dpt_features_to_mamba(
                    input_shape=(1, 1, None, h_warmup, w_warmup),
                    dpt_features=path_1_warmup,
                    in_dpt_layer=0
                )

                # Get relative depth (frozen)
                out_warmup = self.model.depth_head.scratch.output_conv1(path_1_temporal_warmup)
                out_warmup = F.interpolate(out_warmup, (h_warmup, w_warmup), mode="bilinear", align_corners=True)
                relative_depth_warmup = self.model.depth_head.scratch.output_conv2(out_warmup)

                # Get scale/shift from Gear5MetricHead
                gear5_outputs_warmup = self.model.gear5_metric_head(
                    cls_tokens=cls_tokens_warmup,
                    attention_weights_list=attention_weights_list_warmup,
                    patch_h=patch_h_warmup,
                    patch_w=patch_w_warmup
                )

        del encoder_features_warmup, cls_tokens_list_warmup, attention_weights_list_warmup
        del dpt_features_warmup, path_1_warmup, path_1_temporal_warmup
        del relative_depth_warmup, gear5_outputs_warmup
        torch.cuda.empty_cache()

        # FPS measurement (like original FlashDepth)
        # ETH3D: shorter sequences (30 frames) → use 5 warmup frames
        # Other datasets: longer sequences → use 10 warmup frames
        if dataset_name == 'eth3d':
            warmup_frames = min(5, T)
        else:
            warmup_frames = min(10, T)
        start_time = None  # Will start timing after warmup

        # Initialize Mamba sequence for actual test (critical for temporal processing!)
        if hasattr(self.model, 'mamba') and not self.use_bankai:
            # Standard Gear5: Use FlashDepth's original Mamba
            self.model.mamba.start_new_sequence()
        # Note: Bankai's UnifiedMamba handles sequence initialization per call

        # Process each frame
        for t in range(T):
            # Start timing after warmup frames (like original FlashDepth)
            if t == warmup_frames:
                torch.cuda.synchronize()
                import time
                start_time = time.time()

            img_t = images[0, t]  # [3, H, W]
            gt_t_inverse = gt_depth_inverse_100[0, t]  # [1, H, W] in 100/m

            # Use BFloat16 for forward pass (same as train_gear5.py)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                h, w = img_t.shape[1:]
                patch_h, patch_w = h // self.model.patch_size, w // self.model.patch_size

                # Extract features from DINOv2
                encoder_features = self.model.pretrained.get_intermediate_layers(
                    img_t.unsqueeze(0), self.model.intermediate_layer_idx[self.model.encoder]
                )

                # Extract 2-layer CLS tokens
                cls_tokens_list = [
                    encoder_features[i][:, 0]  # CLS token: [1, embed_dim]
                    for i in self.encoder_indices
                ]
                # Average and reshape for GRU: [1, 1, 1024]
                cls_tokens_averaged = torch.stack(cls_tokens_list, dim=1).mean(dim=1)  # [1, 1024]
                cls_tokens = cls_tokens_averaged.view(1, 1, -1)  # [1, 1, 1024]

                # Get attention weights from 2 layers
                attention_weights_list = [
                    self.model.pretrained.blocks[block_idx].attn.attn_weights
                    for block_idx in self.target_blocks
                ]

                # Get DPT features (frozen)
                dpt_features = self.model.depth_head.get_forward_features(
                    encoder_features, patch_h, patch_w
                )
                path_1 = dpt_features[-1]

                if self.use_bankai:
                    # ========== BANKAI MODE ==========
                    # UnifiedMamba processes path_1 + CLS for temporal depth + metric

                    # Call BankaiMetricHead with path_1 + CLS tokens
                    gear5_outputs = self.model.gear5_metric_head(
                        path_1=path_1,  # [1, dpt_dim, h, w]
                        cls_tokens=cls_tokens,  # [1, 1, embed_dim]
                        attention_weights_list=attention_weights_list,
                        input_shape=(1, 1, 3, h, w)  # B=1, T=1
                    )

                    scale = gear5_outputs['scale']  # [1, 1]
                    shift = gear5_outputs['shift']  # [1, 1]
                    importance_map = gear5_outputs['importance_map']  # [1, 1, patch_h, patch_w]
                    path_1_temporal = gear5_outputs['spatial_out']  # [1, dpt_dim, h, w] - enhanced by UnifiedMamba

                    # Get relative depth through output_conv
                    out = self.model.depth_head.scratch.output_conv1(path_1_temporal)
                    out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
                    relative_depth = self.model.depth_head.scratch.output_conv2(out)  # [1, 1, H, W]
                else:
                    # ========== STANDARD GEAR5 MODE ==========
                    # Apply Mamba temporal processing (frozen)
                    path_1_temporal = self.model.dpt_features_to_mamba(
                        input_shape=(1, 1, None, h, w),
                        dpt_features=path_1,
                        in_dpt_layer=0
                    )

                    # Get relative depth (frozen)
                    out = self.model.depth_head.scratch.output_conv1(path_1_temporal)
                    out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
                    relative_depth = self.model.depth_head.scratch.output_conv2(out)  # [1, 1, H, W]

                    # Get scale/shift/importance_map from Gear5MetricHead
                    gear5_outputs = self.model.gear5_metric_head(
                        cls_tokens=cls_tokens,
                        attention_weights_list=attention_weights_list,
                        patch_h=patch_h,
                        patch_w=patch_w
                    )

                    scale = gear5_outputs['scale']  # [1, 1]
                    shift = gear5_outputs['shift']  # [1, 1]
                    importance_map = gear5_outputs['importance_map']  # [1, 1, patch_h, patch_w]

                # Apply scale/shift to relative depth
                scale_expanded = scale.view(1, 1, 1, 1)  # [1, 1, 1, 1]
                shift_expanded = shift.view(1, 1, 1, 1)  # [1, 1, 1, 1]
                pred_depth_inverse_100 = scale_expanded * relative_depth + shift_expanded  # [1, 1, H, W]

                # Save canonical pred mask (before de-canonicalization!)
                MIN_INVERSE_CANONICAL = 100.0 / 70.0
                canonical_pred_valid_t = (pred_depth_inverse_100 > MIN_INVERSE_CANONICAL)  # [1, 1, H, W]

                # De-canonicalization: convert from canonical space to actual space (inverse depth)
                # pred_inverse_actual = pred_inverse_canonical * (CANONICAL_FX / fx_actual)
                # Use per-frame fx_actual for correct de-canonicalization
                pred_depth_inverse_100 = pred_depth_inverse_100 * de_canonical_ratio_inverse[0, t]  # [1, 1, H, W] in actual space

                # Interpolate prediction to GT resolution (like train_gear5.py validation)
                gt_t_shape = gt_t_inverse.shape[-2:]  # GT original resolution
                if pred_depth_inverse_100.shape[-2:] != gt_t_shape:
                    pred_depth_inverse_100 = F.interpolate(
                        pred_depth_inverse_100, size=gt_t_shape, mode="bilinear", align_corners=True
                    )

                # Convert to metric depth (already in actual space after de-canonicalization)
                pred_depth_metric = 100.0 / (pred_depth_inverse_100[0] + 1e-8)  # [1, H, W] in actual meters

                # Upsample importance_map to image resolution for smooth visualization
                h_full, w_full = img_t.shape[1:]  # Image resolution
                importance_map_resized = F.interpolate(
                    importance_map, size=(h_full, w_full), mode='bilinear', align_corners=True
                )  # [1, 1, H, W] at image resolution

            # End timing for FPS measurement (after last frame, like original FlashDepth)
            if t == T - 1 and start_time is not None:
                torch.cuda.synchronize()
                end_time = time.time()

            # List append: Keep on GPU during FPS measurement, move to CPU after
            # This prevents .cpu() overhead from affecting FPS while avoiding OOM on long sequences
            if start_time is None or t >= T - 1:
                # FPS measurement ended or not started - move to CPU immediately
                pred_depths_gpu.append(pred_depth_metric.cpu())
                importance_maps_gpu.append(importance_map_resized[0].cpu())
                scales_gpu.append(scale[0].cpu())
                shifts_gpu.append(shift[0].cpu())
                canonical_pred_valid_gpu.append(canonical_pred_valid_t.cpu())
            else:
                # FPS measurement in progress - keep on GPU
                pred_depths_gpu.append(pred_depth_metric)
                importance_maps_gpu.append(importance_map_resized[0])
                scales_gpu.append(scale[0])
                shifts_gpu.append(shift[0])
                canonical_pred_valid_gpu.append(canonical_pred_valid_t)

            # Release intermediate tensors to prevent GPU memory accumulation
            # Critical for long sequences (e.g., urbansyn 1000 frames)
            del encoder_features, cls_tokens_list, cls_tokens_averaged, cls_tokens
            del attention_weights_list, dpt_features, path_1, path_1_temporal
            del relative_depth, gear5_outputs, scale, shift
            del importance_map, importance_map_resized, pred_depth_inverse_100, pred_depth_metric

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

        # Stack predictions (mix of CPU and GPU tensors - move remaining GPU tensors to CPU)
        pred_depths = torch.stack([p.cpu() if p.is_cuda else p for p in pred_depths_gpu], dim=0)  # [T, 1, H, W] in meters
        importance_maps = torch.stack([im.cpu() if im.is_cuda else im for im in importance_maps_gpu], dim=0)  # [T, 1, H, W]
        scales = torch.stack([s.cpu() if s.is_cuda else s for s in scales_gpu], dim=0)  # [T, 1]
        shifts = torch.stack([s.cpu() if s.is_cuda else s for s in shifts_gpu], dim=0)  # [T, 1]
        canonical_pred_valid_all = [cpv.cpu() if cpv.is_cuda else cpv for cpv in canonical_pred_valid_gpu]  # Mixed CPU/GPU

        # Clear memory
        del pred_depths_gpu, importance_maps_gpu, scales_gpu, shifts_gpu, canonical_pred_valid_gpu
        torch.cuda.empty_cache()

        # Convert GT to metric depth for visualization
        # GT is already in actual space (skip_gt_canonicalization=True in dataloader)
        # Move to CPU first to avoid OOM for long sequences (urbansyn 1000 frames)
        gt_depth_inverse_100_cpu = gt_depth_inverse_100[0]  # [T, 1, H, W] actual 100/m
        gt_depth_metric = 100.0 / (gt_depth_inverse_100_cpu + 1e-8)  # [T, 1, H, W] actual meters directly

        # Compute metrics (both pred and GT are now in meters on CPU)
        # Already on CPU, no need to call .cpu() again
        pred_depths_cpu = pred_depths
        gt_depth_metric_cpu = gt_depth_metric

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

                    # Additional debug: AbsRel analysis
                    abs_diff = torch.abs(pred_valid_values - gt_valid_values)
                    abs_rel_per_pixel = abs_diff / gt_valid_values
                    logger.info(f"  AbsRel stats: mean={abs_rel_per_pixel.mean():.4f}, median={abs_rel_per_pixel.median():.4f}, max={abs_rel_per_pixel.max():.4f}")

                    # Find pixels with extreme AbsRel (>10)
                    extreme_mask = abs_rel_per_pixel > 10.0
                    if extreme_mask.sum() > 0:
                        logger.info(f"  Extreme AbsRel pixels (>10): {extreme_mask.sum()} / {len(abs_rel_per_pixel)} ({100*extreme_mask.sum()/len(abs_rel_per_pixel):.1f}%)")
                        extreme_gt = gt_valid_values[extreme_mask]
                        extreme_pred = pred_valid_values[extreme_mask]
                        logger.info(f"    Their GT range: [{extreme_gt.min():.4f}, {extreme_gt.max():.4f}]")
                        logger.info(f"    Their Pred range: [{extreme_pred.min():.4f}, {extreme_pred.max():.4f}]")

                    # GT distribution analysis
                    gt_bins = [0, 1, 5, 10, 20, 50, 70]
                    for i in range(len(gt_bins)-1):
                        bin_mask = (gt_valid_values >= gt_bins[i]) & (gt_valid_values < gt_bins[i+1])
                        if bin_mask.sum() > 0:
                            logger.info(f"  GT bin [{gt_bins[i]}, {gt_bins[i+1]}m): {bin_mask.sum()} pixels ({100*bin_mask.sum()/len(gt_valid_values):.1f}%)")

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
                # Track worst frame (highest abs_rel)
                if frame_metric['abs_rel'] > worst_frame_abs_rel:
                    worst_frame_abs_rel = frame_metric['abs_rel']
                    worst_frame_idx = t

        # Average metrics across frames
        if len(frame_metrics) == 0:
            logger.warning(f"No valid frames for sequence {sequence_id}")
            return {k: 0.0 for k in ["mae", "rmse", "abs_rel", "a1", "tae", "fps"]}

        metrics = {}
        for key in frame_metrics[0].keys():
            values = [m[key] for m in frame_metrics]
            metrics[key] = np.mean(values)

            # Add per-frame statistics for abs_rel and a1 (min/max only, no std)
            if key in ['abs_rel', 'a1']:
                metrics[f'{key}_min'] = float(np.min(values))
                metrics[f'{key}_max'] = float(np.max(values))

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
            # Store per-frame TAE for temporal analysis
            metrics['_per_frame_tae'] = tae_errors
            # Identify TAE spike frames (TAE > 2x mean)
            if len(tae_errors) > 0:
                tae_mean = np.mean(tae_errors)
                tae_spikes = [i for i, tae in enumerate(tae_errors) if tae > 2 * tae_mean]
                metrics['_tae_spike_frames'] = tae_spikes
            else:
                metrics['_tae_spike_frames'] = []
        else:
            metrics['tae'] = 0.0
            metrics['_per_frame_tae'] = []
            metrics['_tae_spike_frames'] = []

        # Compute depth range analysis
        depth_ranges = [(0, 10), (10, 30), (30, 70)]
        depth_range_metrics = {}
        for depth_min, depth_max in depth_ranges:
            range_name = f"{depth_min}-{depth_max}m"
            range_abs_rels = []
            range_a1s = []
            range_pixel_counts = []

            for t in range(pred_depths_cpu.shape[0]):
                pred_frame = pred_depths_cpu[t, 0]
                gt_frame = gt_depth_metric_cpu[t, 0]
                # Range mask
                range_mask = (gt_frame >= depth_min) & (gt_frame < depth_max) & (gt_frame > 0)
                if range_mask.sum() > 0:
                    pred_valid = pred_frame[range_mask]
                    gt_valid = gt_frame[range_mask]
                    # AbsRel
                    abs_rel = torch.abs(pred_valid - gt_valid) / gt_valid
                    range_abs_rels.append(abs_rel.mean().item())
                    # δ1
                    thresh = torch.maximum(gt_valid / (pred_valid + 1e-8), pred_valid / (gt_valid + 1e-8))
                    a1 = (thresh < 1.25).float().mean().item()
                    range_a1s.append(a1)
                    range_pixel_counts.append(range_mask.sum().item())

            if range_abs_rels:
                depth_range_metrics[range_name] = {
                    'abs_rel': float(np.mean(range_abs_rels)),
                    'a1': float(np.mean(range_a1s)),
                    'pixel_count': int(np.sum(range_pixel_counts))
                }
            else:
                depth_range_metrics[range_name] = {'abs_rel': 0.0, 'a1': 0.0, 'pixel_count': 0}

        metrics['_depth_range_analysis'] = depth_range_metrics

        # Compute Reprojection-based TAE (for datasets with camera poses)
        # Supported: sintel, eth3d, bonn, vkitti
        if len(pred_depths) > 1 and 'image_paths' in batch and self.reproj_tae_calculator.is_supported(dataset_name):
            try:
                image_paths_for_tae = batch['image_paths'][0]  # batch size is 1
                reproj_tae_result = self.reproj_tae_calculator.compute_tae(
                    pred_depths[:, 0],  # [T, H, W]
                    gt_depth_metric[:, 0],  # [T, H, W]
                    dataset_name,
                    image_paths_for_tae
                )
                metrics['tae_reproj'] = reproj_tae_result.get('tae_reproj', 0.0)
                metrics['tae_reproj_gt'] = reproj_tae_result.get('tae_reproj_gt', 0.0)
                if reproj_tae_result.get('tae_reproj_supported', False):
                    logger.info(f"Reprojection TAE: {metrics['tae_reproj']:.4f} (GT ref: {metrics['tae_reproj_gt']:.4f})")
            except Exception as e:
                logger.warning(f"Failed to compute reprojection TAE: {e}")
                metrics['tae_reproj'] = 0.0
                metrics['tae_reproj_gt'] = 0.0
        else:
            metrics['tae_reproj'] = 0.0
            metrics['tae_reproj_gt'] = 0.0

        # Add FPS to metrics
        metrics['fps'] = fps

        # Add TSP scale/shift statistics to metrics
        metrics['tsp_scale_mean'] = float(scales.mean().item())
        metrics['tsp_shift_mean'] = float(shifts.mean().item())
        metrics['tsp_scale_std'] = float(scales.std().item())
        metrics['tsp_shift_std'] = float(shifts.std().item())
        metrics['tsp_scale_max'] = float(scales.max().item())
        metrics['tsp_scale_min'] = float(scales.min().item())
        metrics['tsp_shift_max'] = float(shifts.max().item())
        metrics['tsp_shift_min'] = float(shifts.min().item())

        # Store per-frame scale/shift for JSON export
        metrics['_per_frame_scales'] = [float(s.item()) for s in scales[:, 0]]
        metrics['_per_frame_shifts'] = [float(s.item()) for s in shifts[:, 0]]

        # Compute per-frame optimal scale/shift that minimizes AbsRel (least squares in depth space)
        # This serves as the "oracle" reference for comparing with TSP predictions
        MAX_DEPTH = 70.0
        per_frame_optimal_scales = []
        per_frame_optimal_shifts = []

        for t in range(pred_depths_cpu.shape[0]):
            pred_frame = pred_depths_cpu[t, 0]
            gt_frame = gt_depth_metric_cpu[t, 0]
            valid_mask = (gt_frame > 0) & (gt_frame < MAX_DEPTH) & (pred_frame > 0) & (pred_frame < MAX_DEPTH)

            if valid_mask.sum() > 100:  # Need sufficient pixels for stable estimation
                pred_valid = pred_frame[valid_mask]
                gt_valid = gt_frame[valid_mask]
                # Least squares: gt = scale * pred + shift
                A = torch.stack([pred_valid, torch.ones_like(pred_valid)], dim=1)
                try:
                    solution = torch.linalg.lstsq(A, gt_valid, rcond=None).solution
                    opt_scale = float(solution[0].item())
                    opt_shift = float(solution[1].item())
                except:
                    # Fallback: median scaling
                    opt_scale = float(torch.median(gt_valid / (pred_valid + 1e-8)).item())
                    opt_shift = 0.0
            else:
                opt_scale = 1.0
                opt_shift = 0.0

            per_frame_optimal_scales.append(opt_scale)
            per_frame_optimal_shifts.append(opt_shift)

        # Store per-frame optimal values
        metrics['_per_frame_optimal_scales'] = per_frame_optimal_scales
        metrics['_per_frame_optimal_shifts'] = per_frame_optimal_shifts
        # Sequence-level summary (mean of per-frame optmals)
        metrics['optimal_scale'] = float(np.mean(per_frame_optimal_scales))
        metrics['optimal_shift'] = float(np.mean(per_frame_optimal_shifts))

        # Object-wise evaluation: compute per-class metrics for all frames
        # Initialize variables
        seg_masks_np = None  # Will store per-frame segmentations
        per_frame_class_metrics = []  # Per-frame metrics

        if self.object_wise_enabled:
            logger.info(f"DEBUG: object_wise_enabled=True, checking for segmentations in batch...")
            logger.info(f"DEBUG: Batch keys: {batch.keys()}")
            if 'segmentations' not in batch:
                logger.warning(f"DEBUG: 'segmentations' key NOT FOUND in batch! Cannot compute object-wise metrics.")

        if self.object_wise_enabled and 'segmentations' in batch:
            try:
                # Get per-frame segmentations
                seg_masks = batch['segmentations'][0]  # [T, H, W] - batch size is 1
                T_seg = seg_masks.shape[0]

                logger.info(f"Processing {T_seg} frames with segmentation")

                # Convert to numpy
                seg_masks_np = seg_masks.cpu().numpy() if isinstance(seg_masks, torch.Tensor) else seg_masks

                # Debug: Log unique class IDs in first frame
                if sequence_id == 0:
                    unique_classes = np.unique(seg_masks_np[0])
                    logger.info(f"DEBUG: Unique class IDs in first frame: {unique_classes}")
                    if self.object_wise_metrics:
                        class_names = [self.object_wise_metrics.classes.get(c, f'unknown_{c}') for c in unique_classes]
                        logger.info(f"DEBUG: Class names: {class_names}")

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

                # Store object-wise metrics for final aggregation
                metrics['object_wise'] = class_metrics
                logger.info(f"Computed and stored object-wise metrics for {len(class_metrics)} classes across {T_seg} frames for dataset '{dataset_name}'")

            except Exception as e:
                logger.error(f"Error computing object-wise metrics: {e}")
                import traceback
                traceback.print_exc()
                metrics['object_wise'] = {}
                seg_masks_np = None
                per_frame_class_metrics = []

        # FG-wise evaluation: compute metrics separately for FG/BG regions
        if self.fgwise_enabled and 'image_paths' in batch:
            try:
                fgwise_metrics_calc = FGWiseMetrics(self.fgwise_data_root, dataset_name)

                fgwise_metrics_list = []
                image_paths = batch['image_paths'][0]  # batch size is 1
                T_fg = min(len(image_paths), pred_depths.shape[0])

                for t in range(T_fg):
                    pred_frame = pred_depths_cpu[t, 0].numpy()  # [H, W]
                    gt_frame = gt_depth_metric_cpu[t, 0].numpy()  # [H, W]

                    # Extract scene and frame from image path
                    image_path = image_paths[t]
                    scene, frame = self._extract_scene_frame_for_fgwise(image_path, dataset_name)

                    if scene and frame:
                        # Create valid mask with 70m limit (consistent with regular metrics)
                        MAX_DEPTH_FG = 70.0
                        gt_valid_mask = (gt_frame > 0) & (gt_frame < MAX_DEPTH_FG)
                        pred_valid_mask = (pred_frame > 0) & (pred_frame < MAX_DEPTH_FG)
                        valid_mask_fgwise = gt_valid_mask & pred_valid_mask

                        frame_fgwise = fgwise_metrics_calc.compute_frame_metrics(
                            pred_depth=pred_frame,
                            gt_depth=gt_frame,
                            scene=scene,
                            frame=frame,
                            valid_mask=valid_mask_fgwise,
                            min_pixels=100
                        )
                        if frame_fgwise:
                            fgwise_metrics_list.append(frame_fgwise)

                # Aggregate FG-wise metrics across frames
                if fgwise_metrics_list:
                    aggregated_fgwise = aggregate_fgwise_metrics(fgwise_metrics_list)
                    metrics['fg_wise'] = aggregated_fgwise
                    fg_abs_rel = aggregated_fgwise.get('fg_abs_rel', float('nan'))
                    bg_abs_rel = aggregated_fgwise.get('bg_abs_rel', float('nan'))
                    logger.info(f"FG-wise metrics: FG AbsRel={fg_abs_rel:.4f}, BG AbsRel={bg_abs_rel:.4f}")
                else:
                    logger.warning(f"No FG masks found for dataset '{dataset_name}'")
                    metrics['fg_wise'] = {}

                # Clear cache
                fgwise_metrics_calc.clear_cache()

            except Exception as e:
                logger.error(f"Error computing FG-wise metrics: {e}")
                import traceback
                traceback.print_exc()
                metrics['fg_wise'] = {}

        # Recreate valid_mask for visualization (on CPU)
        valid_mask = (gt_depth_metric > 0)  # [T, 1, H, W] on CPU

        # Visualize
        if self.enable_visualization and self.config.eval.get('save_grid', True):
            self._visualize_sequence(
                images[0], pred_depths, gt_depth_metric, importance_maps,
                valid_mask, sequence_id, metrics, fps, focal_lengths[0]
            )

            # Save error heatmaps (only when visualization enabled)
            self._save_error_heatmaps(
                pred_depths_cpu, gt_depth_metric_cpu, sequence_id
            )

        # Save video (GIF or MP4)
        # Note: frame_interval is NOT applied to video - use all frames
        # Skip video for long sequences (urbansyn, unreal4k) to save time and disk space
        skip_video_datasets = ['urbansyn', 'unreal4k']
        should_save_video = not any(skip_name in dataset_name.lower() for skip_name in skip_video_datasets)
        logger.info(f"Video save decision: dataset_name='{dataset_name}', skip_list={skip_video_datasets}, should_save={should_save_video}")
        if self.enable_visualization and self.config.eval.get('out_video', True) and should_save_video:
            # Use original model resolution for images (following FlashDepth approach)
            # save_gifs_as_grid/save_grid_to_mp4 will handle downsampling to save_res
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

            # Gear5 doesn't use multi-layer fusion weights (single temporal backend: GRU or Mamba2)
            layer_weights = None
            use_mamba = self.config.model.get('use_mamba_temporal', False)
            temporal_backend = "Mamba2" if use_mamba else "GRU"
            logger.info(f"Gear5 uses {temporal_backend}-based temporal modeling (no layer fusion weights)")

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

            # Create model_outputs dict for visualization
            model_outputs = {
                'pred_depth': pred_depths[best_frame_idx, 0],  # [H, W]
                'importance_map': importance_maps[best_frame_idx, 0],  # [H, W]
                'scale': scales[best_frame_idx, 0],  # scalar
                'shift': shifts[best_frame_idx, 0],  # scalar
                'fx_ratio': fx_ratio[0, best_frame_idx].item() if fx_ratio is not None else None,  # scalar - NEW
                'resize_ratio': resize_ratio[0, best_frame_idx].item() if resize_ratio is not None else None,  # scalar - NEW
            }

            self._save_best_frame_visualizations(
                images[0, best_frame_idx],  # [3, H, W]
                gt_depth_metric[best_frame_idx, 0],  # [H, W]
                model_outputs,
                sequence_id,
                actual_frame_number,  # Use actual frame number, not batch index
                best_frame_abs_rel,
                fps,  # Add FPS
                seg_mask_for_viz,  # Add segmentation mask (only if matches best_frame)
                class_metrics_for_viz,  # Add class metrics
                layer_weights,  # Add layer weights
                frame_metrics[best_frame_idx] if best_frame_idx < len(frame_metrics) else None,  # Add frame metrics (includes boundary_f1)
                dataset_name  # Add dataset name for object class mapping
            )

            # Save worst frame visualization
            logger.info(f"Worst frame for sequence {sequence_id}: Frame {worst_frame_idx} (AbsRel={worst_frame_abs_rel:.4f})")

            # Get segmentation for worst frame
            if self.object_wise_enabled and seg_masks_np is not None and worst_frame_idx < len(seg_masks_np):
                worst_seg_mask = seg_masks_np[worst_frame_idx]
                worst_class_metrics = per_frame_class_metrics[worst_frame_idx] if worst_frame_idx < len(per_frame_class_metrics) else None
                worst_actual_frame = batch['frame_indices'][0][worst_frame_idx] if 'frame_indices' in batch else worst_frame_idx
            else:
                worst_seg_mask = None
                worst_class_metrics = None
                worst_actual_frame = worst_frame_idx

            worst_model_outputs = {
                'pred_depth': pred_depths[worst_frame_idx, 0],
                'importance_map': importance_maps[worst_frame_idx, 0],
                'scale': scales[worst_frame_idx, 0],
                'shift': shifts[worst_frame_idx, 0],
                'fx_ratio': fx_ratio[0, worst_frame_idx].item() if fx_ratio is not None else None,
                'resize_ratio': resize_ratio[0, worst_frame_idx].item() if resize_ratio is not None else None,
            }

            self._save_worst_frame_visualizations(
                images[0, worst_frame_idx],
                gt_depth_metric[worst_frame_idx, 0],
                worst_model_outputs,
                sequence_id,
                worst_actual_frame,
                worst_frame_abs_rel,
                fps,
                worst_seg_mask,
                worst_class_metrics,
                layer_weights,
                frame_metrics[worst_frame_idx] if worst_frame_idx < len(frame_metrics) else None,
                dataset_name
            )

        # Export individual frames if --best-figure or --frame option is enabled
        # NOTE: This is independent of --visualization flag
        export_frame_idx = None
        if self.export_best_figure and len(frame_metrics) > 0:
            export_frame_idx = best_frame_idx
            interval_info = f"interval={self.frame_interval}" if self.frame_interval else "interval=1"
            logger.info(f"Exporting best frame {best_frame_idx} ±4 intervals ({interval_info}) (--best-figure)")
        elif self.export_frame is not None:
            # User specified exact frame index
            if self.export_frame < len(pred_depths):
                export_frame_idx = self.export_frame
                interval_info = f"interval={self.frame_interval}" if self.frame_interval else "interval=1"
                logger.info(f"Exporting user-specified frame {self.export_frame} ±4 intervals ({interval_info}) (--frame)")
            else:
                logger.warning(f"Requested frame {self.export_frame} exceeds sequence length {len(pred_depths)}, skipping export")

        if export_frame_idx is not None:
            self._export_figure_frames(
                images=images[0],  # [T, 3, H, W]
                pred_depths=pred_depths,  # [T, 1, H, W]
                gt_depths=gt_depth_metric,  # [T, 1, H, W]
                best_frame_idx=export_frame_idx,
                sequence_id=sequence_id,
                dataset_name=dataset_name,
                depth_paths=depth_paths  # For completed depth visualization
            )

        return metrics

    def _visualize_sequence(self, images, pred_depths, gt_depths, importance_maps,
                           valid_mask, sequence_id, metrics, fps=None, focal_lengths=None):
        """
        Save individual frame PNGs for a sequence.

        Each frame is saved as: frames/seq{sequence_id:04d}/frame_{t:04d}.png
        Layout per frame: 1x3 (Image, GT Depth, Pred Depth) - same as GIF format
        """
        T = images.shape[0]
        MAX_DEPTH = 70.0

        # Create sequence directory
        seq_dir = self.save_dir / "frames" / f"seq{sequence_id:04d}"
        seq_dir.mkdir(parents=True, exist_ok=True)

        # Use frame_interval if set, otherwise save all frames
        if self.frame_interval is not None:
            interval = self.frame_interval
            frame_indices = list(range(0, T, interval))
        else:
            frame_indices = list(range(T))

        logger.info(f"Saving {len(frame_indices)} individual frame PNGs for sequence {sequence_id}")

        for t in frame_indices:
            # Create 1x3 figure for this frame (Image, GT, Pred)
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Image (denormalize)
            img = images[t].permute(1, 2, 0).cpu().float().numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)
            axes[0].imshow(img)
            axes[0].set_title(f'Image (Frame {t})')
            axes[0].axis('off')

            # Get pred and GT
            pred = pred_depths[t, 0].cpu().float().numpy()
            gt = gt_depths[t, 0].cpu().float().numpy()

            # Compute GT valid mask and display range
            gt_valid = (gt > 0) & (gt < MAX_DEPTH)
            gt_display = np.where(gt_valid, gt, np.nan)
            if gt_valid.sum() > 0:
                gt_vmin = np.nanpercentile(gt_display, 2)
                gt_vmax = np.nanpercentile(gt_display, 98)
            else:
                gt_vmin, gt_vmax = 0, 1

            # Check if sparse dataset
            gt_exists = (gt > 0)
            gt_density = gt_exists.sum() / gt_exists.size
            is_sparse = gt_density < 0.5

            if is_sparse:
                valid_pixels_per_row = gt_exists.sum(axis=1)
                valid_rows = valid_pixels_per_row >= 10
                valid_row_indices = np.where(valid_rows)[0]
                if len(valid_row_indices) > 0:
                    height_mask = np.zeros_like(gt, dtype=bool)
                    height_mask[valid_row_indices.min():valid_row_indices.max()+1, :] = True
                else:
                    height_mask = np.ones_like(gt, dtype=bool)
                pred_valid_depth = (pred > 0) & (pred < MAX_DEPTH)
                pred_show_mask = height_mask & pred_valid_depth
            else:
                pred_show_mask = gt_valid

            # GT depth (middle)
            cmap_gt = plt.cm.plasma_r.copy()
            cmap_gt.set_bad(color='black')
            axes[1].imshow(gt_display, cmap=cmap_gt, vmin=gt_vmin, vmax=gt_vmax)
            axes[1].set_title('GT Depth (m)')
            axes[1].axis('off')

            # Predicted depth (right)
            pred_display = np.where(pred_show_mask, pred, np.nan)
            cmap_pred = plt.cm.plasma_r.copy()
            cmap_pred.set_bad(color='black')
            axes[2].imshow(pred_display, cmap=cmap_pred, vmin=gt_vmin, vmax=gt_vmax)
            axes[2].set_title('Pred Depth (m)')
            axes[2].axis('off')

            plt.tight_layout()
            save_path = seq_dir / f"frame_{t:04d}.png"
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

        logger.info(f"Saved {len(frame_indices)} frame PNGs to {seq_dir}")

    def _save_error_heatmaps(self, pred_depths, gt_depths, sequence_id):
        """
        Save error heatmaps for each frame showing |pred - gt| / gt.
        Only called when visualization is enabled.
        """
        T = pred_depths.shape[0]
        MAX_DEPTH = 70.0

        # Create heatmaps directory
        heatmap_dir = self.save_dir / "error_heatmaps" / f"seq{sequence_id:04d}"
        heatmap_dir.mkdir(parents=True, exist_ok=True)

        # Use frame_interval if set
        if self.frame_interval is not None:
            frame_indices = list(range(0, T, self.frame_interval))
        else:
            frame_indices = list(range(T))

        for t in frame_indices:
            pred = pred_depths[t, 0].numpy()
            gt = gt_depths[t, 0].numpy()

            # Valid mask
            valid = (gt > 0) & (gt < MAX_DEPTH) & (pred > 0) & (pred < MAX_DEPTH)

            if valid.sum() > 0:
                # Compute relative error
                error = np.abs(pred - gt) / (gt + 1e-8)
                error_display = np.where(valid, error, np.nan)

                fig, ax = plt.subplots(figsize=(8, 6))
                cmap = plt.cm.hot.copy()
                cmap.set_bad(color='black')
                im = ax.imshow(error_display, cmap=cmap, vmin=0, vmax=1)
                plt.colorbar(im, ax=ax, label='AbsRel Error')
                ax.set_title(f'Error Heatmap - Seq {sequence_id} Frame {t}')
                ax.axis('off')

                save_path = heatmap_dir / f"error_{t:04d}.png"
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                plt.close(fig)

        logger.info(f"Saved error heatmaps to {heatmap_dir}")

    def _save_worst_frame_visualizations(self, image, gt_depth, model_outputs,
                                         sequence_id, frame_idx, abs_rel, fps=None,
                                         seg_mask=None, class_metrics=None, layer_weights=None, frame_metrics=None, dataset_name='unknown'):
        """Save worst frame visualization (wrapper for _save_frame_visualizations)"""
        self._save_frame_visualizations(
            image, gt_depth, model_outputs, sequence_id, frame_idx, abs_rel,
            fps, seg_mask, class_metrics, layer_weights, frame_metrics, dataset_name,
            frame_type='worst'
        )

    def _save_best_frame_visualizations(self, image, gt_depth, model_outputs,
                                        sequence_id, frame_idx, abs_rel, fps=None,
                                        seg_mask=None, class_metrics=None, layer_weights=None, frame_metrics=None, dataset_name='unknown'):
        """Save best frame visualization (wrapper for _save_frame_visualizations)"""
        self._save_frame_visualizations(
            image, gt_depth, model_outputs, sequence_id, frame_idx, abs_rel,
            fps, seg_mask, class_metrics, layer_weights, frame_metrics, dataset_name,
            frame_type='best'
        )

    def _save_frame_visualizations(self, image, gt_depth, model_outputs,
                                   sequence_id, frame_idx, abs_rel, fps=None,
                                   seg_mask=None, class_metrics=None, layer_weights=None, frame_metrics=None, dataset_name='unknown',
                                   frame_type='best'):
        """
        Save frame visualization for Gear5 unified model

        Creates a comprehensive grid visualization with format:
            {frame_type}_frame_seq{N}_{frame_idx}_absrel_{abs_rel:.4f}.png

        Layout:
            Row 1: Input Image | GT Depth | Pred Depth
            Row 2: Importance Map | Scale/Shift Info | Metrics
            Row 3: Valid Mask | Error Map | Depth Distribution

        Args:
            image: [3, H, W] - RGB image
            gt_depth: [H, W] - Ground truth metric depth
            model_outputs: dict with keys:
                - 'pred_depth': [H, W] - Predicted metric depth
                - 'importance_map': [H, W] - Importance map (0-1 normalized)
                - 'scale': scalar - Scale factor
                - 'shift': scalar - Shift value
            sequence_id: int - Sequence index
            frame_idx: int - Frame index within sequence
            abs_rel: float - AbsRel metric for this frame
            fps: float - Optional FPS measurement
            frame_metrics: dict - Optional pre-computed metrics
        """
        pred_depth = model_outputs['pred_depth']
        importance_map = model_outputs['importance_map']
        scale = model_outputs['scale']
        shift = model_outputs['shift']
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

        # Check if dataset is sparse (< 50% valid GT pixels)
        gt_exists = (gt_depth > 0)
        gt_density = gt_exists.sum() / gt_exists.size
        is_sparse = gt_density < 0.5

        # GT valid mask (canonical 70m)
        gt_valid_mask = (gt_depth > 0) & (gt_depth < MAX_DEPTH)  # GT valid pixels

        if is_sparse:
            # Sparse dataset (waymo_seg): Apply height mask + fill sparse gaps
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

        # Create figure with 4x3 grid layout matching train_gear4
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
        pred_display = np.where(pred_show_mask, pred_depth, np.nan)  # Invalid = NaN (will be black)
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

            # Dataset-specific object class IDs
            # Based on WAYMO_OBJECT_CLASSES, URBANSYN_OBJECT_CLASSES, VKITTI2_OBJECT_CLASSES
            # in utils/object_wise_evaluation.py

            # Determine object class IDs by dataset name
            if 'waymo' in dataset_name:
                # Waymo: class IDs 1-9
                object_class_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
                # 1: vehicle, 2: pedestrian, 3: sign, 4: cyclist, 5: traffic_light,
                # 6: pole, 7: construction_cone, 8: bicycle, 9: motorcycle
                dataset_type = 'Waymo'
            elif 'vkitti' in dataset_name:
                # VKITTI2: class IDs 11-13 (truck, car, van)
                object_class_ids = [11, 12, 13]
                # 11: truck, 12: car, 13: van
                dataset_type = 'VKITTI2'
            elif 'urbansyn' in dataset_name:
                # UrbanSyn: Cityscapes format, class IDs 11-18
                object_class_ids = [11, 12, 13, 14, 15, 16, 17, 18]
                # 11: person, 12: rider, 13: car, 14: truck, 15: bus,
                # 16: train, 17: motorcycle, 18: bicycle
                dataset_type = 'UrbanSyn'
            else:
                # Unknown dataset: use all classes from segmentation
                logger.warning(f"Unknown dataset '{dataset_name}' for object class mapping, using all non-zero classes")
                object_class_ids = list(np.unique(seg_mask[seg_mask > 0]))
                dataset_type = 'Unknown'

            # Create object mask: include dynamic objects
            object_mask = np.zeros_like(seg_mask, dtype=np.uint8)
            for class_id in object_class_ids:
                object_mask |= (seg_mask == class_id).astype(np.uint8)

            ax7.imshow(object_mask, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            object_ratio = object_mask.sum() / object_mask.size
            # Count only object classes present in this frame
            num_object_classes = len([cid for cid in object_class_ids if (seg_mask == cid).any()])

            logger.info(f"[VISUALIZATION] object_mask sum: {object_mask.sum()}, ratio: {object_ratio*100:.1f}%, num_object_classes: {num_object_classes}")

            ax7.set_title(f'Object Mask ({dataset_type})\n{object_ratio*100:.1f}% ({object_mask.sum():,} pixels)\n{num_object_classes} object classes',
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

        # Scale/Shift info (Gear5 specific, wheat box)
        scale_val = scale.item() if isinstance(scale, torch.Tensor) else scale
        shift_val = shift.item() if isinstance(shift, torch.Tensor) else shift
        ax9.text(0.05, y_pos, f'scale: {scale_val:.3f}, shift: {shift_val:.3f}',
                fontsize=10, transform=ax9.transAxes,
                bbox=dict(boxstyle="round", facecolor='wheat'))
        y_pos -= 0.10

        # fx_ratio, resize_ratio, and FG_ratio (like train, wheat box)
        fx_ratio_val = model_outputs.get('fx_ratio')
        resize_ratio_val = model_outputs.get('resize_ratio')
        fg_ratio_computed = fg_ratio  # Already computed above from fg_mask_binary.mean() * 100

        if fx_ratio_val is not None and resize_ratio_val is not None:
            ax9.text(0.05, y_pos, f'fx_ratio: {fx_ratio_val:.3f} | resize_ratio: {resize_ratio_val:.3f} | FG_ratio: {fg_ratio_computed:.1f}%',
                    fontsize=10, transform=ax9.transAxes,
                    bbox=dict(boxstyle="round", facecolor='wheat'))
        else:
            ax9.text(0.05, y_pos, f'FG_ratio: {fg_ratio_computed:.1f}%',
                    fontsize=10, transform=ax9.transAxes,
                    bbox=dict(boxstyle="round", facecolor='wheat'))
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
        frame_type_label = frame_type.capitalize()
        plt.suptitle(f'Gear5: Sequence {sequence_id} {frame_type_label} Frame {frame_idx}',
                    fontsize=16, fontweight='bold')

        # Save with same naming convention
        save_path = self.save_dir / f"{frame_type}_frame_seq{sequence_id}_{frame_idx}_absrel_{abs_rel:.4f}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        logger.info(f"Saved Gear5 {frame_type} frame visualization: {save_path}")

    def _export_figure_frames(self, images, pred_depths, gt_depths, best_frame_idx, sequence_id, dataset_name, depth_paths=None):
        """
        Export individual frames around best_frame (±4 intervals, total 9 frames).
        Saves: original image, GT depth (colormap), pred depth (colormap)

        For ETH3D and Waymo datasets, uses completed depth maps for visualization instead of sparse GT.

        Args:
            images: [T, 3, H, W] tensor
            pred_depths: [T, 1, H, W] tensor in meters
            gt_depths: [T, 1, H, W] tensor in meters
            best_frame_idx: int, index of best frame
            sequence_id: int, sequence identifier
            dataset_name: str, name of dataset (e.g., 'eth3d/pipes', 'waymo_seg/segment-xxx')
            depth_paths: List[str] or None, paths to depth files for completed depth loading
        """
        import cv2
        import matplotlib.pyplot as plt

        T = images.shape[0]

        # Determine frame range with frame_interval support
        # Goal: Get 9 frames (center ± 4 intervals) or as many as possible
        # If one end is constrained, extend in the other direction
        # Example: frame=193, interval=10, T=200 → frames 113,123,133,143,153,163,173,183,193
        frame_interval = self.frame_interval if self.frame_interval is not None else 1
        target_num_frames = 9  # We want 9 frames total (center ± 4)
        half_count = (target_num_frames - 1) // 2  # 4 frames on each side

        # Generate candidate frame indices centered around best_frame_idx
        frame_indices = []
        for i in range(-half_count, half_count + 1):
            idx = best_frame_idx + i * frame_interval
            if 0 <= idx < T:
                frame_indices.append(idx)

        # If we don't have enough frames, extend in the available direction
        while len(frame_indices) < target_num_frames:
            # Try extending backward first (if back is constrained, extend forward)
            first = frame_indices[0]
            new_back = first - frame_interval
            last = frame_indices[-1]
            new_forward = last + frame_interval

            extended = False
            if new_back >= 0:
                frame_indices.insert(0, new_back)
                extended = True
            elif new_forward < T:
                frame_indices.append(new_forward)
                extended = True

            if not extended:
                break  # Can't extend anymore

        # Ensure best_frame_idx is included even if not perfectly aligned
        if best_frame_idx not in frame_indices and best_frame_idx < T:
            frame_indices.append(best_frame_idx)
            frame_indices.sort()

        logger.info(f"Exporting figure frames for sequence {sequence_id}: frames {frame_indices} (center={best_frame_idx}, interval={frame_interval})")

        # Create figures directory
        figures_dir = self.save_dir / "figures" / f"seq{sequence_id:04d}"
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Check if completed depth is available for this dataset
        # Extract base dataset name (e.g., 'eth3d' from 'eth3d/pipes')
        base_dataset = dataset_name.split('/')[0] if '/' in dataset_name else dataset_name
        use_completed_depth = base_dataset in ['eth3d', 'waymo_seg'] and depth_paths is not None

        if use_completed_depth:
            try:
                from utils.completed_depth import load_completed_depth, depth_to_colormap
                logger.info(f"Using completed depth for {base_dataset} visualization")
            except ImportError:
                logger.warning("Could not import completed_depth module, using sparse GT")
                use_completed_depth = False

        for t in frame_indices:
            # 1. Save original image
            img = images[t].cpu().numpy()  # [3, H, W]

            # Check if ImageNet normalized (value range check)
            if img.min() < -2.0 or img.max() > 2.0:
                # ImageNet normalized - unnormalize first
                mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                img = img * std + mean  # Unnormalize to [0, 1]

            img = np.transpose(img, (1, 2, 0))  # [H, W, 3]
            img = np.clip(img, 0, 1)  # Ensure [0, 1] range
            img = (img * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for cv2
            img_path = figures_dir / f"frame_{t:04d}_image.png"
            cv2.imwrite(str(img_path), img)

            # 2. Save GT depth (colormap)
            # For ETH3D/Waymo, try to use completed depth for visualization
            gt_depth_sparse = gt_depths[t, 0].cpu().numpy()  # [H, W] in meters (sparse)
            target_size = (gt_depth_sparse.shape[0], gt_depth_sparse.shape[1])
            pred_depth = pred_depths[t, 0].cpu().numpy()  # [H, W] in meters

            # Compute gt_valid mask and determine sparse/dense FIRST
            MAX_DEPTH = 70.0
            gt_valid = (gt_depth_sparse > 0) & (gt_depth_sparse < MAX_DEPTH)
            gt_density = gt_valid.sum() / gt_valid.size
            is_sparse = gt_density < 0.5

            completed_depth_loaded = False
            if use_completed_depth and t < len(depth_paths):
                completed_depth = load_completed_depth(
                    depth_paths[t], base_dataset, target_size=target_size
                )
                if completed_depth is not None:
                    completed_depth_np = completed_depth.numpy()
                    # Use completed depth for visualization
                    gt_depth_vis = depth_to_colormap(completed_depth_np)
                    gt_depth_vis = cv2.cvtColor(gt_depth_vis, cv2.COLOR_RGB2BGR)  # RGB to BGR
                    completed_depth_loaded = True

                    # Get vmin/vmax from completed depth for pred normalization
                    # For Waymo, exclude -1 regions (no LiDAR coverage)
                    if base_dataset == 'waymo_seg':
                        valid_mask = (completed_depth_np > 0) & np.isfinite(completed_depth_np)
                    else:
                        valid_mask = np.isfinite(completed_depth_np) & (completed_depth_np > 0)

                    if valid_mask.any():
                        gt_vmin = np.nanpercentile(completed_depth_np[valid_mask], 2)
                        gt_vmax = np.nanpercentile(completed_depth_np[valid_mask], 98)
                    else:
                        gt_vmin, gt_vmax = None, None

            if not completed_depth_loaded:
                # For both sparse and dense datasets:
                # - Use gt_valid mask for GT visualization (exclude invalid and far depth)
                # - Compute vmin/vmax from gt_valid pixels
                gt_depth_vis = self._depth_to_colormap(gt_depth_sparse, external_mask=gt_valid)

                # Get vmin/vmax from gt_valid pixels (not just depth > 0)
                if gt_valid.any():
                    gt_vmin = np.nanpercentile(gt_depth_sparse[gt_valid], 2)
                    gt_vmax = np.nanpercentile(gt_depth_sparse[gt_valid], 98)
                else:
                    gt_vmin, gt_vmax = None, None

            gt_path = figures_dir / f"frame_{t:04d}_gt_depth.png"
            cv2.imwrite(str(gt_path), gt_depth_vis)

            # 3. Save pred depth (colormap) - use GT range for comparison
            # Same logic as main visualization (scene.png)

            if is_sparse:
                # Sparse dataset: use height mask (LiDAR scan range)
                valid_pixels_per_row = gt_valid.sum(axis=1)
                min_valid_pixels_threshold = 10
                valid_rows = valid_pixels_per_row >= min_valid_pixels_threshold
                valid_row_indices = np.where(valid_rows)[0]

                if len(valid_row_indices) > 0:
                    min_valid_row = valid_row_indices.min()
                    max_valid_row = valid_row_indices.max()
                    height_mask = np.zeros_like(pred_depth, dtype=bool)
                    height_mask[min_valid_row:max_valid_row+1, :] = True
                else:
                    height_mask = np.ones_like(pred_depth, dtype=bool)

                pred_valid_depth = (pred_depth > 0) & (pred_depth < MAX_DEPTH)
                pred_show_mask = height_mask & pred_valid_depth  # Dense prediction within height range
            else:
                # Dense dataset: use GT valid mask (same as main visualization)
                pred_show_mask = gt_valid

            pred_depth_vis = self._depth_to_colormap(pred_depth, vmin=gt_vmin, vmax=gt_vmax, external_mask=pred_show_mask)
            pred_path = figures_dir / f"frame_{t:04d}_pred_depth.png"
            cv2.imwrite(str(pred_path), pred_depth_vis)

        completed_info = " (using completed depth)" if use_completed_depth else ""
        logger.info(f"Exported {len(frame_indices)} frames × 3 types = {len(frame_indices) * 3} images to {figures_dir}{completed_info}")

    def _depth_to_colormap(self, depth, vmin=None, vmax=None, percentile_range=(2, 98), external_mask=None):
        """
        Convert depth map to colormap visualization (matching gear5_visualization.py style).

        Args:
            depth: [H, W] numpy array in meters
            vmin: minimum depth for colormap (default: use 2nd percentile)
            vmax: maximum depth for colormap (default: use 98th percentile)
            percentile_range: tuple of (low, high) percentiles for auto-scaling
            external_mask: [H, W] boolean mask to restrict valid region (e.g., height_mask for sparse datasets)

        Returns:
            [H, W, 3] BGR image (uint8)
        """
        import matplotlib
        import cv2

        # Handle invalid values
        # If external_mask is provided, use it to restrict valid region
        if external_mask is not None:
            valid_mask = np.isfinite(depth) & (depth > 0) & external_mask
        else:
            valid_mask = np.isfinite(depth) & (depth > 0)
        if not valid_mask.any():
            return np.zeros((*depth.shape, 3), dtype=np.uint8)

        valid_depth = depth[valid_mask]

        # Use percentile normalization if vmin/vmax not provided (matching gear5_visualization.py)
        if vmin is None:
            vmin = np.nanpercentile(valid_depth, percentile_range[0])
        if vmax is None:
            vmax = np.nanpercentile(valid_depth, percentile_range[1])

        # Create depth with NaN for invalid pixels
        depth_vis = np.where(valid_mask, depth, np.nan)

        # Normalize to [0, 1]
        depth_normalized = np.clip((depth_vis - vmin) / (vmax - vmin + 1e-8), 0, 1)

        # Apply colormap (plasma_r to match gear5_visualization.py)
        # Use new matplotlib API to avoid deprecation warning
        cmap = matplotlib.colormaps.get_cmap('plasma_r').copy()
        cmap.set_bad(color='black')  # NaN pixels = black
        depth_colored_rgba = cmap(depth_normalized)
        depth_colored = (depth_colored_rgba[:, :, :3] * 255).astype(np.uint8)

        # Convert RGB to BGR for cv2
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)

        return depth_colored

    def _aggregate_metrics(self, all_metrics):
        """Aggregate metrics across sequences"""
        metric_keys = all_metrics[0].keys()
        aggregated_raw = {}

        for key in metric_keys:
            # Skip nested dictionaries (like object_wise, fg_wise metrics) and non-aggregable keys
            if key in ('object_wise', 'fg_wise', 'sequence_id'):
                continue

            values = [m[key] for m in all_metrics if key in m]
            if values:
                # Check if values are numeric (not dicts or other non-numeric types)
                if all(isinstance(v, (int, float, np.number)) for v in values):
                    aggregated_raw[key] = np.mean(values)

        # Reorder metrics: abs_rel, a1, a2, a3, fps, tae, tae_reproj, mae, rmse, then TSP stats
        metric_order = ['abs_rel', 'a1', 'a2', 'a3', 'fps', 'tae', 'tae_reproj', 'tae_reproj_gt', 'mae', 'rmse',
                        'tsp_scale_mean', 'tsp_shift_mean', 'tsp_scale_std', 'tsp_shift_std']
        aggregated = {}
        for key in metric_order:
            if key in aggregated_raw:
                aggregated[key] = aggregated_raw[key]
        # Add any remaining metrics not in the order list
        for key, value in aggregated_raw.items():
            if key not in aggregated:
                aggregated[key] = value

        return aggregated


@hydra.main(version_base=None, config_path="configs/gear5", config_name="config")
def main(config: DictConfig):
    """Main entry point"""
    import os

    # Override config for testing
    config.inference = True

    # Enable object-wise evaluation if --objwise flag was passed
    # (flag is removed from sys.argv in __main__ block before Hydra processes it)
    if getattr(main, '_objwise_mode', False):
        OmegaConf.update(config, 'object_wise.enabled', True, merge=False)

    tester = Gear5Tester(config)
    tester.test()


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
