#!/usr/bin/env python3
"""
Test script for VIDEO depth estimation methods

Evaluation framework for video-based depth estimation methods that process
entire sequences at once (not frame-by-frame).

Supported methods:
- Video-Depth-Anything (vda) - Processes [B, T, C, H, W]
- DepthCrafter (depthcrafter) - Processes [T, C, H, W] or [T, H, W, C]

Note: For single-frame image depth models (Metric3D, UniDepth, DepthPro, etc.),
use test_comparison.py instead.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import logging
import json
from pathlib import Path
from tqdm import tqdm
import argparse

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dataloaders.combined_dataset import CombinedDataset
from dataloaders.comparison_dataset import ComparisonDataset, comparison_collate_fn
from dataloaders.waymo_segmentation_dataset import WaymoSegmentationDataset, collate_fn as waymo_collate_fn
from dataloaders.urbansyn_segmentation_dataset import UrbanSynSegmentationDataset, urbansyn_collate_fn
from dataloaders.vkitti_segmentation_dataset import VKITTISegmentationDataset, collate_fn as vkitti_collate_fn
from utils.metric_depth_metrics import MetricDepthMetrics, RelativeDepthMetrics
from utils.object_wise_evaluation import ObjectWiseMetrics
from utils.fgwise_evaluation import (
    FGWiseMetrics, aggregate_fgwise_metrics,
    save_fgwise_visualization, draw_fg_contours, create_depth_with_fg_overlay
)
from utils.comparison_visualization import visualize_sequence_simplified, visualize_best_frame_simplified

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoComparisonTester:
    """
    Tester for VIDEO depth estimation methods (sequence-based processing)
    """

    def __init__(self, method_name, config, adapter):
        """
        Args:
            method_name: str - Method identifier (e.g., 'vda', 'metric3d_v1')
            config: dict - Configuration dictionary
            adapter: MethodAdapter - Adapter instance for the specific method
        """
        self.method_name = method_name
        self.config = config
        self.adapter = adapter
        self.device = torch.device(f"cuda:{config.get('gpu', 0)}" if torch.cuda.is_available() else "cpu")

        # Depth mode and visualization settings
        self.depth_mode = config.get('depth_mode', 'metric')
        self.frame_interval = config.get('frame_interval', None)
        self.enable_visualization = config.get('visualization', True)

        logger.info(f"Depth evaluation mode: {self.depth_mode}")
        if self.frame_interval is not None:
            logger.info(f"Frame interval for visualization: {self.frame_interval}")

        # Figure export options
        self.export_best_figure = config.get('best_figure', False)
        self.export_frame = config.get('frame', None)  # Specific frame index (int or None)
        if self.export_best_figure:
            logger.info("Best-figure export ENABLED (will save best_frame ±4 as individual images)")
        if self.export_frame is not None:
            logger.info(f"Frame-specific export ENABLED (will save frame {self.export_frame} ±4 as individual images)")

        # Validate depth mode matches adapter output type
        self._validate_depth_mode()

        # Setup save directory
        dataset_name = config.get('dataset', 'waymo')
        self.save_dir = Path(config.get('results_dir', f'refer_test/test_results/{method_name}/{dataset_name}'))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Testing method: {method_name}")
        logger.info(f"Results will be saved to: {self.save_dir}")

        # Object-wise evaluation setup
        self.object_wise_enabled = config.get('object_wise', {}).get('enabled', False)
        self.object_wise_dataset = config.get('object_wise', {}).get('dataset', 'waymo')
        self.dataset_name = dataset_name  # Store for later use

        # DEBUG: Log object-wise configuration
        logger.info(f"[OBJWISE DEBUG] Config object_wise dict: {config.get('object_wise', {})}")
        logger.info(f"[OBJWISE DEBUG] object_wise_enabled: {self.object_wise_enabled}")
        logger.info(f"[OBJWISE DEBUG] dataset_name: {dataset_name}")

        if self.object_wise_enabled:
            self.object_wise_metrics = ObjectWiseMetrics(
                dataset_type=self.object_wise_dataset,
                depth_mode=self.depth_mode
            )
            logger.info(f"Object-wise evaluation enabled for {self.object_wise_dataset}")

            # Auto-set frame_interval for waymo_seg objwise mode (like test_gear*)
            if self.dataset_name == 'waymo_seg' and self.frame_interval is None:
                self.frame_interval = 2
                logger.info(f"Auto-setting frame_interval=2 for waymo_seg objwise mode")

        # FG-wise evaluation setup
        self.fgwise_enabled = config.get('fg_wise', {}).get('enabled', False)
        if self.fgwise_enabled:
            data_root = config.dataset.get('data_root', '/data/datasets')
            logger.info(f"FG-wise evaluation ENABLED (data_root: {data_root})")
            self.fgwise_data_root = data_root
        else:
            self.fgwise_data_root = None

        # Metrics calculators
        self.metrics = MetricDepthMetrics()
        self.relative_metrics = RelativeDepthMetrics()

        # Load model
        self.model = self._setup_model()

        # Setup test loader
        self.test_loader = self._setup_test_loader()

        # Storage for results
        self.all_results = []

    def _setup_model(self):
        """Load model using adapter"""
        logger.info(f"Loading model for {self.method_name}...")

        checkpoint_path = self.config.get('checkpoint_path', None)
        model = self.adapter.load_model(checkpoint_path)
        model = model.to(self.device)
        model.eval()

        # Set adapter device for inference
        self.adapter.device = self.device

        logger.info(f"Model loaded successfully")
        return model

    def _validate_depth_mode(self):
        """
        Validate that depth_mode matches adapter's output type
        
        Some adapters always output relative depth (0-1 normalized), while others
        output metric depth (meters). This validation prevents incorrect evaluation.
        """
        method = self.config.get('method', '')
        metric_flag = self.config.get('metric', False)
        
        # Define which methods output relative depth vs metric depth
        relative_depth_methods = {
            'depthcrafter': 'DepthCrafter always outputs relative depth (0-1 normalized)',
        }
        
        # VideoDepthAnything depends on the metric flag
        if method == 'vda' and not metric_flag:
            relative_depth_methods['vda'] = 'VideoDepthAnything without --metric flag outputs relative depth'
        
        # Check if method outputs relative depth but depth_mode is metric
        if self.depth_mode == 'metric' and method in relative_depth_methods:
            error_msg = (
                f"\n{'='*80}\n"
                f"ERROR: Depth mode mismatch!\n"
                f"{'='*80}\n"
                f"Method '{method}' outputs RELATIVE DEPTH, but depth_mode is set to 'metric'.\n"
                f"\n"
                f"Reason: {relative_depth_methods[method]}\n"
                f"\n"
                f"This will produce INCORRECT metrics!\n"
                f"\n"
                f"Solutions:\n"
                f"1. Add --depth-mode relative to your command line\n"
                f"2. For VideoDepthAnything, use --metric flag if you want metric depth\n"
                f"\n"
                f"Example correct usage:\n"
            )

            if method == 'depthcrafter':
                error_msg += f"  python test_comparison.py --method depthcrafter --depth-mode relative --dataset {{dataset}}\n"
            elif method == 'vda':
                error_msg += f"  python test_comparison.py --method vda --depth-mode relative --dataset {{dataset}}\n"
                error_msg += f"  # OR for metric VideoDepthAnything:\n"
                error_msg += f"  python test_comparison.py --method vda --metric --depth-mode metric --dataset {{dataset}}\n"

            error_msg += f"{'='*80}\n"

            logger.error(error_msg)
            raise ValueError(f"Depth mode mismatch: {method} outputs relative depth but depth_mode='metric'")
        
        # Log successful validation
        if method in relative_depth_methods:
            logger.info(f"✓ Validated: {method} outputs relative depth, depth_mode='{self.depth_mode}'")
        else:
            logger.info(f"✓ Validated: {method} outputs metric depth, depth_mode='{self.depth_mode}'")

    def _setup_test_loader(self):
        """Setup test dataloader"""
        dataset_name = self.config.get('dataset', 'waymo')
        data_root = self.config.get('data_root', '/home/cvlab/hsy/Datasets')
        video_length = self.config.get('video_length', 50)
        batch_size = 1  # Always 1 for testing

        logger.info(f"Setting up test loader for dataset: {dataset_name}")

        # Object-wise datasets
        if dataset_name.endswith('_seg'):
            base_dataset_name = dataset_name.replace('_seg', '')
            if base_dataset_name == 'waymo':
                # WaymoSegmentationDataset expects data_root to be waymo_seg directory
                waymo_data_root = str(Path(data_root) / 'waymo_seg')

                # Use objwise_mode only if --objwise flag is set
                # objwise_mode=True: Use ALL frames with segmentation (ignores video_length)
                # objwise_mode=False: Use video_length frames (standard sliding window)
                objwise_mode = self.object_wise_enabled

                dataset = WaymoSegmentationDataset(
                    data_root=waymo_data_root,
                    split='val',
                    video_length=video_length,
                    objwise_mode=objwise_mode
                )
                collate_fn = waymo_collate_fn
                if objwise_mode:
                    logger.info(f"Object-wise dataset: waymo_seg (using ALL frames with segmentation, ignoring video_length)")
                else:
                    logger.info(f"Standard dataset: waymo_seg (using video_length={video_length} frames)")
            elif base_dataset_name == 'urbansyn':
                dataset = UrbanSynSegmentationDataset(
                    data_root=data_root,
                    split='test',
                    video_length=video_length,
                    max_frames=1000
                )
                collate_fn = urbansyn_collate_fn
                logger.info(f"Object-wise dataset: urbansyn_seg")
            elif base_dataset_name == 'vkitti':
                only_clone = self.config.get('only_clone', True)
                dataset = VKITTISegmentationDataset(
                    data_root=data_root,
                    split='test',
                    video_length=video_length,
                    only_clone=only_clone
                )
                collate_fn = vkitti_collate_fn
                logger.info(f"Object-wise dataset: vkitti_seg (only_clone={only_clone})")
            else:
                raise ValueError(f"Unknown segmentation dataset: {dataset_name}")
        else:
            # Standard datasets - use ComparisonDataset for fair comparison
            # ComparisonDataset provides ORIGINAL resolution images
            # Match FlashDepth original inference behavior (split='test' for comprehensive testing)
            # waymo: uses 'val' split (hardcoded directory path in dataset implementation)
            # others: use 'test' split (all scenes)
            if dataset_name == 'waymo':
                split = 'val'
            else:
                split = 'test'

            # Get only_clone flag for VKITTI
            only_clone = self.config.get('only_clone', False) if dataset_name == 'vkitti' else False

            # Get seq_list from config (supports all datasets now)
            seq_list = self.config.get('seq_list', None)

            dataset = ComparisonDataset(
                dataset_name=dataset_name,
                data_root=data_root,
                split=split,
                video_length=video_length,
                objwise_enabled=self.object_wise_enabled,
                only_clone=only_clone,
                seq_list=seq_list,  # Supports all datasets now
                limit_scenes=self.config.get('limit_scenes') # Pass limit_scenes
            )
            collate_fn = comparison_collate_fn

        # For high resolution datasets, use num_workers=0 to avoid OOM and slow worker processes
        num_workers = self.config.get('workers', 4)
        # Handle aliases: unreal, unrealstereo4k → unreal4k
        if dataset_name in ['eth3d', 'unreal4k', 'unreal', 'unrealstereo4k']:
            num_workers = 0
            logger.info(f"Using num_workers=0 for {dataset_name} (high resolution dataset)")

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

        logger.info(f"Test loader created with {len(dataset)} sequences")
        return loader

    def test(self):
        """Run testing on all sequences"""
        logger.info(f"Starting evaluation on {len(self.test_loader)} sequences...")

        for sequence_id, batch in enumerate(tqdm(self.test_loader, desc="Testing")):
            try:
                metrics = self.test_sequence(batch, sequence_id)
                metrics['sequence_id'] = sequence_id
                self.all_results.append(metrics)

                logger.info(f"Sequence {sequence_id}: "
                           f"AbsRel={metrics['abs_rel']:.4f}, "
                           f"δ1={metrics['a1']:.4f}, "
                           f"TAE={metrics['tae']:.4f}, "
                           f"F1={metrics.get('boundary_f1', 0):.3f}")

            except Exception as e:
                logger.error(f"Error processing sequence {sequence_id}: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Clear GPU cache after each sequence (like test_comparison.py)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Aggregate and save results
        self._aggregate_and_save_results()

    @torch.no_grad()
    def test_sequence(self, batch, sequence_id):
        """Test on a single sequence"""
        # Log GPU info at start
        if sequence_id == 0 and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(self.device)
            gpu_mem_total = torch.cuda.get_device_properties(self.device).total_memory / 1e9
            logger.info(f"GPU: {gpu_name} (Total Memory: {gpu_mem_total:.1f}GB)")

        # Get inputs
        if 'images' in batch:
            # ComparisonDataset or SegmentationDataset format (original resolution)
            images = batch['images'].to(self.device)  # [1, T, 3, H, W]
            H, W = images.shape[-2:]
            logger.info(f"Sequence {sequence_id}: Processing {images.shape[1]} frames at {H}x{W} resolution")

            # Handle different depth key names
            # ComparisonDataset: 'depths' (plural)
            # SegmentationDataset: 'depth' (singular)
            if 'depths' in batch:
                gt_depths = batch['depths'].to(self.device)  # [1, T, H, W] - ComparisonDataset (meters)
            else:
                gt_depths = batch['depth'].to(self.device)  # [1, T, H, W] - SegmentationDataset (inverse depth)

            intrinsics = batch.get('intrinsics', None)
            if intrinsics is not None:
                intrinsics = intrinsics.to(self.device)  # [1, T, 4] - fx, fy, cx, cy

            # Get depth file paths for completed depth loading (visualization)
            depth_paths = batch.get('depth_paths', None)  # List[str] or None
        else:
            # Legacy CombinedDataset format (resized)
            images = batch['image'].to(self.device)  # [1, T, 3, H, W]
            gt_depths = batch['depth'].to(self.device)  # [1, T, H, W] - inverse depth!
            intrinsics = None
            depth_paths = None  # Not available in legacy format

        # Ensure proper dimensions
        if images.ndim == 4:
            images = images.unsqueeze(0)  # [T, 3, H, W] -> [1, T, 3, H, W]
        if gt_depths.ndim == 3:
            gt_depths = gt_depths.unsqueeze(0)  # [T, H, W] -> [1, T, H, W]

        B, T = images.shape[:2]
        assert B == 1, "Batch size must be 1 for testing"

        # Process GT depth
        # Dataset name determines depth format:
        # - *_seg datasets (waymo_seg, urbansyn_seg, vkitti_seg) → SegmentationDataset → inverse depth (1/m)
        # - other datasets → ComparisonDataset → metric depth (m)

        dataset_name = self.config.get('dataset', 'waymo')
        is_segmentation_dataset = dataset_name.endswith('_seg')

        if is_segmentation_dataset:
            # SegmentationDataset - convert from inverse depth to metric
            gt_depth_processed = 1.0 / (gt_depths + 1e-8)  # [1, T, H, W] inverse → meters
            gt_depth_processed = gt_depth_processed.unsqueeze(2)  # [1, T, 1, H, W]
        else:
            # ComparisonDataset - already in meters
            if gt_depths.ndim == 4:
                gt_depth_processed = gt_depths.unsqueeze(2)  # [1, T, H, W] -> [1, T, 1, H, W]
            else:
                gt_depth_processed = gt_depths # Already [1, T, 1, H, W]

        # Get focal lengths (for models that need them)
        if intrinsics is not None:
            focal_lengths = intrinsics[:, :, 0]  # [1, T] - fx values
        else:
            focal_lengths = batch.get('focal_lengths', None)
            if focal_lengths is not None:
                focal_lengths = focal_lengths.to(self.device)

        # Check if images are ImageNet normalized (from segmentation datasets)
        # ComparisonDataset returns [0, 1] range
        # Segmentation datasets return ImageNet normalized
        sample_img = images[0, 0]  # Check first frame
        is_imagenet_normalized = (sample_img.min() < -2.0 or sample_img.max() > 2.0)

        # Unnormalize if needed (video models expect [0, 1] range)
        if is_imagenet_normalized:
            # Unnormalize ImageNet: x_original = x * std + mean
            mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 1, 3, 1, 1)
            images_unnorm = images * std + mean  # [1, T, 3, H, W] back to [0, 1]
        else:
            images_unnorm = images

        # Process entire sequence at once (VIDEO MODEL)
        logger.info(f"Processing {T} frames as a single video sequence...")

        # FPS measurement - warmup with first inference, then measure
        import time

        # Warmup run
        logger.info("Warmup inference...")
        torch.cuda.synchronize()
        with torch.no_grad():
            if self.config.get('amp', False):
                amp_dtype = torch.bfloat16 if self.config.get('amp_dtype', 'bf16') == 'bf16' else torch.float16
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    _ = self.adapter.inference(
                        images_unnorm,  # [1, T, 3, H, W]
                        intrinsics=intrinsics  # [1, T, 4] or None
                    )
            else:
                _ = self.adapter.inference(
                    images_unnorm,  # [1, T, 3, H, W]
                    intrinsics=intrinsics  # [1, T, 4] or None
                )

        # Timed run
        logger.info("Timed inference...")
        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            if self.config.get('amp', False):
                amp_dtype = torch.bfloat16 if self.config.get('amp_dtype', 'bf16') == 'bf16' else torch.float16
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    pred_depths = self.adapter.inference(
                        images_unnorm,  # [1, T, 3, H, W]
                        intrinsics=intrinsics  # [1, T, 4] or None
                    )  # Returns [1, T, H, W] in meters
            else:
                pred_depths = self.adapter.inference(
                    images_unnorm,  # [1, T, 3, H, W]
                    intrinsics=intrinsics  # [1, T, 4] or None
                )  # Returns [1, T, H, W] in meters

        torch.cuda.synchronize()
        end_time = time.time()

        # Calculate FPS
        inference_time = end_time - start_time
        fps = T / inference_time if inference_time > 0 else 0
        logger.info(f"Inference time: {inference_time:.4f}s for {T} frames")
        logger.info(f"FPS: {fps:.2f}")

        # Log GPU memory usage
        if torch.cuda.is_available():
            gpu_mem_allocated = torch.cuda.memory_allocated(self.device) / 1e9
            gpu_mem_reserved = torch.cuda.memory_reserved(self.device) / 1e9
            logger.info(f"GPU Memory: Allocated={gpu_mem_allocated:.2f}GB, Reserved={gpu_mem_reserved:.2f}GB")

        # Ensure correct shape [T, 1, H, W] for compatibility with visualization
        if pred_depths.ndim == 4 and pred_depths.shape[0] == 1:
            pred_depths = pred_depths[0]  # [1, T, H, W] -> [T, H, W]
        if pred_depths.ndim == 3:
            pred_depths = pred_depths.unsqueeze(1)  # [T, H, W] -> [T, 1, H, W]

        # Best frame tracking
        best_frame_idx = 0
        best_frame_abs_rel = float('inf')
        frame_metrics = []

        # Upsample predictions to match GT original resolution for visualization
        # This handles cases where model outputs resized depth (e.g., 518x518)
        # but we want to visualize at original resolution (e.g., 1920x1280)
        gt_depth_processed_cpu = gt_depth_processed[0].cpu()  # [T, 1, H, W]
        pred_H, pred_W = pred_depths.shape[2:]
        gt_H, gt_W = gt_depth_processed_cpu.shape[2:]

        if (gt_H != pred_H) or (gt_W != pred_W):
            logger.info(f"Upsampling predictions from {pred_H}x{pred_W} to {gt_H}x{gt_W} to match GT original resolution")
            pred_depths = torch.nn.functional.interpolate(
                pred_depths,  # [T, 1, H, W]
                size=(gt_H, gt_W),
                mode='bilinear',  # Use bilinear for smooth depth upsampling
                align_corners=False
            )

        # Compute metrics
        pred_depths_cpu = pred_depths.cpu()

        # Define MAX_DEPTH for valid mask creation (used in both regular and TAE computation)
        MAX_DEPTH = 70.0

        # Track best frame (criteria differs by depth mode)
        if self.depth_mode == 'relative':
            best_frame_f1 = 0.0  # For relative: maximize F1

        # Compute regular metrics if:
        # 1. Object-wise mode is disabled, OR
        # 2. Object-wise mode is enabled but no segmentations available (fallback)
        has_segmentations = 'segmentations' in batch
        compute_regular_metrics = not self.object_wise_enabled or (self.object_wise_enabled and not has_segmentations)

        if compute_regular_metrics:
            # Regular metrics computation (full image)
            for t in range(pred_depths.shape[0]):
                pred_frame = pred_depths_cpu[t, 0]  # [H, W]
                gt_frame = gt_depth_processed_cpu[t, 0]  # [H, W]

                # Create valid mask
                gt_valid_mask = (gt_frame > 0) & (gt_frame < MAX_DEPTH)
                pred_valid_mask = (pred_frame > 0) & (pred_frame < MAX_DEPTH)
                valid_mask = gt_valid_mask & pred_valid_mask

                if valid_mask.sum() > 0:
                    # Compute metrics based on depth mode
                    if self.depth_mode == 'metric':
                        # Skip boundary F1 for ETH3D (too slow at 6048x4032 resolution)
                        skip_f1 = (self.config.get('dataset', '') == 'eth3d')
                        frame_metric = self.metrics.compute_metric_depth_metrics(
                            pred_frame, gt_frame, valid_mask, skip_boundary_f1=skip_f1
                        )
                        # Track best frame (lowest AbsRel for metric)
                        if frame_metric['abs_rel'] < best_frame_abs_rel:
                            best_frame_abs_rel = frame_metric['abs_rel']
                            best_frame_idx = t
                    else:  # relative
                        # Skip boundary F1 for ETH3D (too slow at 6048x4032 resolution)
                        skip_f1 = (self.config.get('dataset', '') == 'eth3d')
                        frame_metric = self.relative_metrics.compute_relative_depth_metrics(
                            pred_frame, gt_frame, valid_mask, skip_boundary_f1=skip_f1
                        )
                        # Track best frame (highest F1 for relative)
                        if frame_metric['boundary_f1'] > best_frame_f1:
                            best_frame_f1 = frame_metric['boundary_f1']
                            best_frame_idx = t

                    frame_metrics.append(frame_metric)

            # Average metrics
            if len(frame_metrics) == 0:
                logger.warning(f"No valid frames for sequence {sequence_id}")
                return {k: 0.0 for k in ["mae", "rmse", "abs_rel", "a1", "tae", "fps", "boundary_f1"]}

            metrics = {}
            for key in frame_metrics[0].keys():
                values = [m[key] for m in frame_metrics]
                metrics[key] = np.mean(values)

            # Compute TAE (Temporal Alignment Error) for regular metrics mode
            if len(pred_depths) > 1:
                tae_errors = []
                for t in range(len(pred_depths) - 1):
                    pred_t = pred_depths_cpu[t, 0]
                    pred_t_next = pred_depths_cpu[t + 1, 0]
                    gt_t = gt_depth_processed_cpu[t, 0]
                    gt_t_next = gt_depth_processed_cpu[t + 1, 0]

                    valid_t = (gt_t > 0) & (gt_t < MAX_DEPTH) & (pred_t > 0) & (pred_t < MAX_DEPTH)
                    valid_t_next = (gt_t_next > 0) & (gt_t_next < MAX_DEPTH) & (pred_t_next > 0) & (pred_t_next < MAX_DEPTH)

                    if valid_t.sum() > 0 and valid_t_next.sum() > 0:
                        if self.depth_mode == 'metric':
                            # Metric depth: direct frame-to-frame comparison
                            valid_both = valid_t & valid_t_next
                            if valid_both.sum() > 0:
                                pred_change = pred_t_next - pred_t
                                gt_change = gt_t_next - gt_t
                                tae = torch.abs(pred_change[valid_both] - gt_change[valid_both]).mean()
                                tae_errors.append(tae.item())
                        else:  # relative
                            # Relative depth: scale-invariant TAE
                            tae_si = self.relative_metrics.compute_tae_scale_invariant(
                                pred_t, pred_t_next, gt_t, gt_t_next, valid_t, valid_t_next
                            )
                            if tae_si < float('inf'):
                                tae_errors.append(tae_si)

                metrics['tae'] = np.mean(tae_errors) if len(tae_errors) > 0 else 0.0
            else:
                metrics['tae'] = 0.0

            metrics['fps'] = fps

        # Object-wise evaluation (ONLY when enabled - replaces regular metrics)
        if self.object_wise_enabled and 'segmentations' in batch:
            try:
                seg_masks = batch['segmentations'][0]  # [T, H, W]
                seg_masks_np = seg_masks.cpu().numpy() if isinstance(seg_masks, torch.Tensor) else seg_masks

                # Extract dataset_name (handle both string and list)
                dataset_name_raw = batch.get('dataset_name', 'unknown')
                if isinstance(dataset_name_raw, list):
                    dataset_name = dataset_name_raw[0] if len(dataset_name_raw) > 0 else 'unknown'
                else:
                    dataset_name = dataset_name_raw

                if isinstance(dataset_name, str) and 'vkitti' in dataset_name.lower():
                    logger.info(f"[VKITTI DEBUG] Starting object-wise evaluation: seg_masks shape={seg_masks_np.shape}")

                per_frame_class_metrics = []
                per_frame_aggregated = []  # For best frame selection

                for t in range(len(seg_masks_np)):
                    pred_frame = pred_depths_cpu[t, 0].numpy()
                    gt_frame = gt_depth_processed_cpu[t, 0].numpy()
                    seg_mask_frame = seg_masks_np[t]

                    # Resize segmentation if needed
                    if seg_mask_frame.shape != pred_frame.shape:
                        import cv2
                        seg_mask_frame = cv2.resize(
                            seg_mask_frame.astype(np.int32),
                            (pred_frame.shape[1], pred_frame.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )

                    frame_class_metrics = self.object_wise_metrics.compute_metrics_per_class(
                        pred_depth=pred_frame,
                        gt_depth=gt_frame,
                        seg_mask=seg_mask_frame,
                        min_pixels=100,
                        max_depth=MAX_DEPTH
                    )
                    per_frame_class_metrics.append(frame_class_metrics)

                    if t == 0 and isinstance(dataset_name, str) and 'vkitti' in dataset_name.lower():
                        logger.info(f"[VKITTI DEBUG] Frame 0 metrics: {len(frame_class_metrics)} classes found")

                    # Aggregate this frame's class metrics for best frame selection
                    if frame_class_metrics:
                        # Compute mean across all classes for this frame
                        all_abs_rels = [m['abs_rel'] for m in frame_class_metrics.values() if 'abs_rel' in m]
                        if all_abs_rels:
                            frame_abs_rel = np.mean(all_abs_rels)
                            per_frame_aggregated.append({'abs_rel': frame_abs_rel, 'frame_idx': t})

                            # Track best frame (lowest aggregated AbsRel)
                            if frame_abs_rel < best_frame_abs_rel:
                                best_frame_abs_rel = frame_abs_rel
                                best_frame_idx = t

                # Aggregate per-class metrics across all frames
                class_metrics = self.object_wise_metrics.aggregate_metrics(per_frame_class_metrics)

                # Convert object-wise aggregated metrics to regular format (for display/logging)
                # Compute mean of all class metrics
                if class_metrics:
                    all_class_values = {}
                    for class_name, class_metric in class_metrics.items():
                        for key, value in class_metric.items():
                            if key not in all_class_values:
                                all_class_values[key] = []
                            all_class_values[key].append(value)

                    # Average across classes
                    metrics = {key: np.mean(values) for key, values in all_class_values.items()}
                    metrics['object_wise'] = class_metrics  # Keep detailed breakdown
                    metrics['fps'] = fps
                    metrics['tae'] = 0.0  # TAE not computed in object-wise mode

                    logger.info(f"Object-wise: {len(class_metrics)} classes, Avg AbsRel={metrics.get('abs_rel', 0):.4f}")
                else:
                    logger.warning(f"No object-wise metrics computed for sequence {sequence_id}")
                    metrics = {k: 0.0 for k in ["mae", "rmse", "abs_rel", "a1", "tae", "fps"]}
                    metrics['fps'] = fps

            except Exception as e:
                logger.error(f"Error computing object-wise metrics: {e}")
                import traceback
                traceback.print_exc()
                metrics = {k: 0.0 for k in ["mae", "rmse", "abs_rel", "a1", "tae", "fps"]}
                metrics['fps'] = fps
                metrics['object_wise'] = {}
        elif self.object_wise_enabled:
            # Object-wise enabled but no segmentations available
            # Regular metrics already computed as fallback
            logger.info(f"Object-wise mode enabled but no segmentations available - using regular metrics")

        # DEBUG: Log evaluation path selection
        logger.info(f"[OBJWISE DEBUG] Sequence {sequence_id}: object_wise_enabled={self.object_wise_enabled}, has_segmentations={has_segmentations}")
        logger.info(f"[OBJWISE DEBUG] Sequence {sequence_id}: compute_regular_metrics={compute_regular_metrics}")
        if has_segmentations and 'dataset_name' in batch:
            dataset_name_raw = batch.get('dataset_name', 'unknown')
            logger.info(f"[OBJWISE DEBUG] Sequence {sequence_id}: dataset_name={dataset_name_raw}")

        # FG-wise evaluation: compute metrics separately for FG/BG regions
        if self.fgwise_enabled and 'image_paths' in batch:
            try:
                dataset_name = batch.get('dataset_name', [''])[0] if isinstance(batch.get('dataset_name'), list) else batch.get('dataset_name', '')
                fgwise_metrics_calc = FGWiseMetrics(self.fgwise_data_root, dataset_name)

                fgwise_metrics_list = []
                image_paths = batch['image_paths'][0]  # batch size is 1
                T_fg = min(len(image_paths), pred_depths.shape[0])

                for t in range(T_fg):
                    pred_frame = pred_depths[t, 0].cpu().numpy() if isinstance(pred_depths, torch.Tensor) else pred_depths[t, 0]
                    gt_frame = gt_depth_processed[0, t, 0].cpu().numpy() if isinstance(gt_depth_processed, torch.Tensor) else gt_depth_processed[0, t, 0]

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

        # Visualize sequence (simplified version without importance maps)
        # Skip for ETH3D due to very high resolution (6205x4135) making this too slow
        if self.enable_visualization:
            dataset_name = batch.get('dataset_name', [''])[0] if isinstance(batch.get('dataset_name'), list) else batch.get('dataset_name', '')
            if isinstance(dataset_name, str):
                dataset_name_lower = dataset_name.lower()
            else:
                dataset_name_lower = str(dataset_name).lower()

            # Skip visualization for high-resolution datasets
            skip_viz = any(x in dataset_name_lower for x in ['eth3d', 'unreal4k', 'unreal', 'unrealstereo'])

            if not skip_viz:
                # Prepare segmentation masks for visualization if object-wise mode
                seg_masks_for_viz = None
                if self.object_wise_enabled and 'segmentations' in batch:
                    seg_masks_for_viz = batch['segmentations'][0]  # [T, H, W]

                visualize_sequence_simplified(
                    images[0], pred_depths, gt_depth_processed[0],
                    valid_mask=(gt_depth_processed[0] > 0),
                    sequence_id=sequence_id,
                    metrics=metrics,
                    fps=fps,
                    save_dir=self.save_dir,
                    focal_lengths=focal_lengths[0] if focal_lengths is not None else None,
                    frame_interval=self.frame_interval,  # Pass frame interval for visualization
                    seg_masks=seg_masks_for_viz,
                    objwise_enabled=self.object_wise_enabled,
                    object_classes=self.object_wise_metrics.object_classes if self.object_wise_enabled else None,
                    depth_paths=depth_paths,  # For completed depth visualization (ETH3D/Waymo)
                    dataset_name=dataset_name_lower  # e.g., 'eth3d', 'waymo_seg'
                )
            else:
                logger.info(f"Skipping sequence visualization for {dataset_name} (high resolution dataset)")

        # Save best frame visualization
        # For object-wise mode: check if metrics exist, for regular: check frame_metrics
        # Also handle fallback case where object-wise is enabled but no segmentations (uses regular metrics)
        has_metrics = len(frame_metrics) > 0 or (self.object_wise_enabled and 'object_wise' in metrics)
        if self.enable_visualization and has_metrics:
            # Log best frame with appropriate metric
            if self.depth_mode == 'metric':
                logger.info(f"Best frame for sequence {sequence_id}: Frame {best_frame_idx} (AbsRel={best_frame_abs_rel:.4f})")
                best_metric_value = best_frame_abs_rel
            else:  # relative
                logger.info(f"Best frame for sequence {sequence_id}: Frame {best_frame_idx} (F1={best_frame_f1:.4f})")
                best_metric_value = best_frame_f1

            # Extract dataset name from batch (handle both string and list)
            dataset_name_raw = batch.get('dataset_name', 'unknown')
            if isinstance(dataset_name_raw, list):
                dataset_name = dataset_name_raw[0] if len(dataset_name_raw) > 0 else 'unknown'
            else:
                dataset_name = dataset_name_raw

            # Extract focal length for this frame
            if focal_lengths is not None and focal_lengths.shape[1] > best_frame_idx:
                frame_focal_length = focal_lengths[0, best_frame_idx].item()
            else:
                frame_focal_length = None

            # Extract segmentation mask for best frame if available
            frame_seg_mask = None
            if self.object_wise_enabled and 'segmentations' in batch:
                seg_masks = batch['segmentations'][0]  # [T, H, W]
                seg_masks_np = seg_masks.cpu().numpy() if isinstance(seg_masks, torch.Tensor) else seg_masks
                if best_frame_idx < len(seg_masks_np):
                    frame_seg_mask = seg_masks_np[best_frame_idx]  # [H, W]
                    # Resize if needed to match prediction resolution
                    if frame_seg_mask.shape != pred_depths[best_frame_idx, 0].shape:
                        import cv2
                        frame_seg_mask = cv2.resize(
                            frame_seg_mask.astype(np.int32),
                            (pred_depths[best_frame_idx, 0].shape[1], pred_depths[best_frame_idx, 0].shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )

            # Get frame-specific metrics for visualization
            if self.object_wise_enabled:
                # Use aggregated object-wise metrics (averaged across classes)
                frame_viz_metrics = {k: v for k, v in metrics.items() if k != 'object_wise'}
            else:
                # Use regular frame-specific metrics
                frame_viz_metrics = frame_metrics[best_frame_idx] if best_frame_idx < len(frame_metrics) else {}

            visualize_best_frame_simplified(
                image=images[0, best_frame_idx],  # [3, H, W]
                gt_depth=gt_depth_processed[0, best_frame_idx, 0],  # [H, W]
                pred_depth=pred_depths[best_frame_idx, 0],  # [H, W]
                metrics=frame_viz_metrics,
                save_dir=self.save_dir,
                sequence_id=sequence_id,
                frame_idx=best_frame_idx,
                dataset_name=dataset_name,
                focal_length=frame_focal_length,
                seg_mask=frame_seg_mask,
                objwise_enabled=self.object_wise_metrics.object_classes if self.object_wise_enabled else None,
                class_names_dict=self.object_wise_metrics.classes if self.object_wise_enabled else None,
                gt_depth_path=depth_paths[best_frame_idx] if depth_paths and best_frame_idx < len(depth_paths) else None
            )

        # Export individual frames if --best-figure or --frame option is enabled
        # NOTE: This is independent of --visualization flag
        has_metrics = len(frame_metrics) > 0 or (self.object_wise_enabled and 'object_wise' in metrics)
        export_frame_idx = None
        if self.export_best_figure and has_metrics:
            export_frame_idx = best_frame_idx
            logger.info(f"Exporting best frame {best_frame_idx} ±4 (--best-figure)")
        elif self.export_frame is not None:
            # User specified exact frame index
            if self.export_frame < len(pred_depths):
                export_frame_idx = self.export_frame
                logger.info(f"Exporting user-specified frame {self.export_frame} ±4 (--frame)")
            else:
                logger.warning(f"Requested frame {self.export_frame} exceeds sequence length {len(pred_depths)}, skipping export")

        if export_frame_idx is not None:
            self._export_figure_frames(
                images=images[0],  # [T, 3, H, W]
                pred_depths=pred_depths,  # [T, 1, H, W]
                gt_depths=gt_depth_processed[0],  # [T, 1, H, W]
                best_frame_idx=export_frame_idx,
                sequence_id=sequence_id,
                dataset_name=dataset_name,
                depth_paths=depth_paths  # For completed depth visualization
            )

        return metrics

    def _aggregate_and_save_results(self):
        """Aggregate metrics and save results"""
        if len(self.all_results) == 0:
            logger.warning("No results to aggregate")
            return

        # Compute average metrics
        avg_metrics_raw = {}
        for key in ['mae', 'rmse', 'abs_rel', 'sq_rel', 'rmse_log', 'a1', 'a2', 'a3', 'tae', 'boundary_f1', 'fps']:
            values = [r[key] for r in self.all_results if key in r]
            if len(values) > 0:
                avg_metrics_raw[key] = float(np.mean(values))

        # Reorder metrics according to desired order: abs_rel, a1, a2, a3, fps, tae, f1, mae, rmse
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
        reordered_results = []
        for result in self.all_results:
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
            reordered_results.append(reordered)

        # Get processing resolution from adapter
        processing_resolution = self.adapter.processing_resolution
        if isinstance(processing_resolution, tuple):
            proc_res_str = f"{processing_resolution[0]}×{processing_resolution[1]}"
        else:
            proc_res_str = str(processing_resolution)

        # Save test results
        test_results = {
            'method': self.method_name,
            'dataset': self.config.get('dataset', 'waymo'),
            'num_sequences': len(self.all_results),
            'processing_resolution': proc_res_str,
            'metrics': avg_metrics
        }

        results_path = self.save_dir / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2)

        logger.info(f"\nTest Results Summary:")
        logger.info(f"  Method: {self.method_name}")
        logger.info(f"  Dataset: {test_results['dataset']}")
        logger.info(f"  Sequences: {test_results['num_sequences']}")
        logger.info(f"  Processing Resolution: {proc_res_str}")
        logger.info(f"  MAE: {avg_metrics.get('mae', 0):.4f}")
        logger.info(f"  RMSE: {avg_metrics.get('rmse', 0):.4f}")
        logger.info(f"  AbsRel: {avg_metrics.get('abs_rel', 0):.4f}")
        logger.info(f"  δ1: {avg_metrics.get('a1', 0):.4f}")
        logger.info(f"  TAE: {avg_metrics.get('tae', 0):.4f}")
        logger.info(f"  F1: {avg_metrics.get('boundary_f1', 0):.3f}")
        logger.info(f"  FPS: {avg_metrics.get('fps', 0):.2f}")
        logger.info(f"Results saved to {results_path}")

        # Save per-sequence results with reordered metrics
        per_seq_path = self.save_dir / 'per_sequence_results.json'
        with open(per_seq_path, 'w') as f:
            json.dump(reordered_results, f, indent=2)

        logger.info(f"Per-sequence results saved to {per_seq_path}")

        # Aggregate and save object-wise metrics
        if self.object_wise_enabled:
            # Collect all object-wise metrics from sequences
            all_object_wise_metrics = [r['object_wise'] for r in self.all_results if 'object_wise' in r and r['object_wise']]

            if all_object_wise_metrics:
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
                logger.info(f"Object-wise results saved to {object_wise_path}")

        # Aggregate and save FG-wise metrics
        if self.fgwise_enabled:
            all_fgwise_metrics = [r['fg_wise'] for r in self.all_results if 'fg_wise' in r and r['fg_wise']]

            if all_fgwise_metrics:
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
                logger.info(f"FG-wise results saved to {fgwise_path}")

    def _export_figure_frames(self, images, pred_depths, gt_depths, best_frame_idx, sequence_id, dataset_name, depth_paths=None):
        """
        Export individual frames around best_frame (±4 frames, total 9 frames).
        Saves: original image, GT depth (colormap), pred depth (colormap)

        For ETH3D and Waymo datasets, uses completed depth maps for GT visualization
        while keeping sparse depth for metrics calculation.

        Args:
            images: [T, 3, H, W] tensor
            pred_depths: [T, 1, H, W] tensor in meters
            gt_depths: [T, 1, H, W] tensor in meters
            best_frame_idx: int, index of best frame
            sequence_id: int, sequence identifier
            dataset_name: str, name of dataset
            depth_paths: List[str] or None, paths to original depth files for completed depth loading
        """
        import cv2
        import matplotlib.pyplot as plt

        T = images.shape[0]

        # Determine frame range with frame_interval support
        # If frame_interval is set, expand range and use interval for sampling
        # Example: frame=9, interval=2 → frames 1, 3, 5, 7, 9, 11, 13, 15, 17
        frame_interval = self.frame_interval if self.frame_interval is not None else 1
        frame_offset = 4 * frame_interval  # ±4 intervals
        start_idx = max(0, best_frame_idx - frame_offset)
        end_idx = min(T, best_frame_idx + frame_offset + 1)  # +1 for inclusive end

        # Generate frame indices with interval
        frame_indices = list(range(start_idx, end_idx, frame_interval))
        # Ensure best_frame_idx is included even if not perfectly aligned
        if best_frame_idx not in frame_indices:
            frame_indices.append(best_frame_idx)
            frame_indices.sort()

        logger.info(f"Exporting figure frames for sequence {sequence_id}: frames {frame_indices} (center={best_frame_idx}, interval={frame_interval})")

        # Create figures directory
        figures_dir = self.save_dir / "figures" / f"seq{sequence_id:04d}"
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Check if this is a sparse dataset that has completed depth
        use_completed_depth = dataset_name in ['eth3d', 'waymo_seg'] and depth_paths is not None
        if use_completed_depth:
            try:
                from utils.completed_depth import load_completed_depth, depth_to_colormap
                logger.info(f"Using completed depth for {dataset_name} visualization")
            except ImportError:
                logger.warning("Could not import completed_depth utilities, falling back to sparse GT")
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
            gt_depth = gt_depths[t, 0].cpu().numpy()  # [H, W] in meters (sparse GT for metrics)
            H, W = gt_depth.shape

            # Try to use completed depth for visualization
            gt_depth_for_vis = None
            if use_completed_depth and t < len(depth_paths):
                try:
                    completed_depth = load_completed_depth(
                        depth_paths[t], dataset_name, target_size=(H, W)
                    )
                    if completed_depth is not None:
                        gt_depth_for_vis = completed_depth.numpy()
                except Exception as e:
                    logger.warning(f"Failed to load completed depth for frame {t}: {e}")

            # Use completed depth if available, otherwise fall back to sparse GT
            if gt_depth_for_vis is not None:
                # Use depth_to_colormap from completed_depth module (handles -1 values)
                gt_depth_vis = depth_to_colormap(gt_depth_for_vis)
                gt_depth_vis = cv2.cvtColor(gt_depth_vis, cv2.COLOR_RGB2BGR)
            else:
                # Fallback to sparse GT with default colormap
                gt_depth_vis = self._depth_to_colormap(gt_depth)
            
            gt_path = figures_dir / f"frame_{t:04d}_gt_depth.png"
            cv2.imwrite(str(gt_path), gt_depth_vis)
            
            # Get vmin/vmax from GT for pred normalization
            gt_valid_mask = np.isfinite(gt_depth) & (gt_depth > 0)
            if gt_valid_mask.any():
                gt_vmin = np.nanpercentile(gt_depth[gt_valid_mask], 2)
                gt_vmax = np.nanpercentile(gt_depth[gt_valid_mask], 98)
            else:
                gt_vmin, gt_vmax = None, None

            # 3. Save pred depth (colormap) - use GT range and GT valid mask for comparison
            pred_depth = pred_depths[t, 0].cpu().numpy()  # [H, W] in meters
            pred_depth_vis = self._depth_to_colormap(pred_depth, vmin=gt_vmin, vmax=gt_vmax, external_mask=gt_valid_mask)
            pred_path = figures_dir / f"frame_{t:04d}_pred_depth.png"
            cv2.imwrite(str(pred_path), pred_depth_vis)

        logger.info(f"Exported {len(frame_indices)} frames × 3 types = {len(frame_indices) * 3} images to {figures_dir}")

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
                scene_idx = parts.index('eth3d') + 1 if 'eth3d' in parts else -1
                if scene_idx > 0 and scene_idx < len(parts):
                    scene = parts[scene_idx]
                    frame = os.path.splitext(parts[-1])[0]
                    return scene, frame

            elif base_dataset_name == 'sintel':
                if 'clean' in parts:
                    clean_idx = parts.index('clean')
                    if clean_idx + 2 < len(parts):
                        scene = parts[clean_idx + 1]
                        frame = os.path.splitext(parts[-1])[0]
                        return scene, frame

            elif base_dataset_name == 'waymo_seg':
                if 'FRONT' in parts:
                    front_idx = parts.index('FRONT')
                    if front_idx >= 2:
                        segment = parts[front_idx - 1]
                        frame = os.path.splitext(parts[-1])[0]
                        return segment, frame

            elif base_dataset_name == 'vkitti':
                if 'vkitti' in parts and 'rgb' in parts[-1]:
                    vkitti_idx = parts.index('vkitti')
                    if vkitti_idx + 1 < len(parts):
                        scene = parts[vkitti_idx + 1]
                        filename = os.path.splitext(parts[-1])[0]
                        if filename.startswith('rgb_'):
                            frame = filename[4:]
                            return scene, frame

            elif base_dataset_name == 'unreal4k':
                for part in parts:
                    if part.startswith('UnrealStereo4K_'):
                        scene = part.replace('UnrealStereo4K_', '')
                        frame = os.path.splitext(parts[-1])[0]
                        return scene, frame

            elif base_dataset_name == 'urbansyn':
                if 'urbansyn' in parts and 'rgb' in parts:
                    urbansyn_idx = parts.index('urbansyn')
                    rgb_idx = parts.index('rgb')
                    if urbansyn_idx + 1 == rgb_idx - 1:
                        scene = parts[urbansyn_idx + 1]
                        frame = os.path.splitext(parts[-1])[0]
                        return scene, frame

            elif base_dataset_name == 'tartanair':
                if 'tartanair' in parts and 'image_left' in parts:
                    tartanair_idx = parts.index('tartanair')
                    if tartanair_idx + 1 < len(parts):
                        scene = parts[tartanair_idx + 1]
                        filename = os.path.splitext(parts[-1])[0]
                        if filename.endswith('_left'):
                            frame = filename[:-5]
                        else:
                            frame = filename
                        return scene, frame

            elif base_dataset_name == 'bonn':
                for part in parts:
                    if part.startswith('rgbd_bonn_'):
                        scene = part
                        frame = os.path.splitext(parts[-1])[0]
                        return scene, frame

            # Generic fallback
            if len(parts) >= 2:
                scene = parts[-2]
                frame = os.path.splitext(parts[-1])[0]
                return scene, frame

        except Exception as e:
            logger.warning(f"Failed to extract scene/frame from {image_path}: {e}")

        return None, None

    def _depth_to_colormap(self, depth, vmin=None, vmax=None, percentile_range=(2, 98), external_mask=None):
        """
        Convert depth map to colormap visualization (matching gear5_visualization.py style).

        Args:
            depth: [H, W] numpy array in meters
            vmin: minimum depth for colormap (default: use 2nd percentile)
            vmax: maximum depth for colormap (default: use 98th percentile)
            percentile_range: tuple of (low, high) percentiles for auto-scaling
            external_mask: [H, W] boolean mask to restrict valid region (e.g., GT valid mask for pred depth)

        Returns:
            [H, W, 3] BGR image (uint8)
        """
        import matplotlib
        import cv2

        # Handle invalid values (matching test_gear5 exactly)
        # If external_mask is provided (e.g., GT valid mask), use it to restrict valid region
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


def main():
    parser = argparse.ArgumentParser(description='Test VIDEO depth estimation methods')
    parser.add_argument('--method', type=str, required=True,
                       help='Method name: vda, depthcrafter (video models only)')
    parser.add_argument('--version', type=str, default=None,
                       help='Method version (for metric3d, unidepth): v1, v2')
    parser.add_argument('--dataset', type=str, default='waymo',
                       help='Dataset name')
    parser.add_argument('--data-root', type=str, default='/home/cvlab/hsy/Datasets',
                       help='Data root directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Model checkpoint path')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Results directory')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--video-length', type=int, default=50,
                       help='Video sequence length')
    parser.add_argument('--objwise', action='store_true',
                       help='Enable object-wise evaluation')
    parser.add_argument('--only-clone', type=str, default='true', choices=['true', 'false'],
                       help='For VKITTI: use only clone condition (default: true)')

    # New options for depth mode and model-specific settings
    parser.add_argument('--depth-mode', type=str, default='metric', choices=['metric', 'relative'],
                       help='Depth evaluation mode: metric (absolute depth) or relative (scale-invariant)')
    parser.add_argument('--indoor', action='store_true',
                       help='Use indoor checkpoint (for depthanythingv2 only)')
    parser.add_argument('--metric', action='store_true',
                       help='Use metric mode (for videodepthanything only)')
    parser.add_argument('--frame-interval', type=int, default=None,
                       help='Frame interval for sequence.png visualization')
    parser.add_argument('--visualization', type=str, default='true', choices=['true', 'false'],
                       help='Enable/disable visualizations (sequence.png, best_frame.png, etc.). Default: true')
    parser.add_argument('--seq', type=str, default=None,
                       help='Sequence number(s) to test (e.g., --seq 0, --seq 2,5, --seq 0,3,7)')
    parser.add_argument('--figure', action='store_true',
                       help='Export best_frame ±4 frames (9 total) as individual images/depth maps (requires --seq)')
    parser.add_argument('--frame', type=int, default=None,
                       help='Export specific frame ±4 frames (9 total) as individual images (e.g., --seq 6 --frame 459)')
    parser.add_argument('--amp', action='store_true',
                       help='Enable Automatic Mixed Precision (AMP) for inference')
    parser.add_argument('--amp-dtype', type=str, default='bf16', choices=['bf16', 'fp16'],
                       help='Data type for AMP (bfloat16 or float16)')
    parser.add_argument('--limit-scenes', type=int, default=None,
                       help='Limit the number of NuScenes scenes to process (for debugging)')

    args = parser.parse_args()

    # Validate method is a video model
    VIDEO_MODELS = ['vda', 'depthcrafter']
    if args.method not in VIDEO_MODELS:
        logger.error(f"❌ Error: '{args.method}' is not a video model")
        logger.error(f"   This script is for VIDEO models only: {', '.join(VIDEO_MODELS)}")
        logger.error(f"   For image models (metric3d, unidepth, depthpro, etc.), use test_comparison.py instead")
        sys.exit(1)

    # Parse --seq (supports single or multiple sequences, for all datasets)
    seq_list = None
    if args.seq is not None:
        # Parse comma-separated sequence numbers
        try:
            seq_list = [int(s.strip()) for s in args.seq.split(',')]
            logger.info(f"{args.dataset}: Testing sequences {seq_list}")
        except ValueError:
            logger.error(f"❌ Invalid --seq format: {args.seq}")
            logger.error(f"   Examples: --seq 0, --seq 2,5, --seq 0,3,7")
            sys.exit(1)

    # Build method name with version
    method_name = args.method
    if args.version:
        method_name = f"{args.method}_{args.version}"

    # Create config
    config = {
        'method': args.method,
        'version': args.version,
        'dataset': args.dataset,
        'data_root': args.data_root,
        'checkpoint_path': args.checkpoint,
        'results_dir': args.results_dir or f'refer_test/test_results/{method_name}/{args.dataset}',
        'gpu': args.gpu,
        'workers': args.workers,
        'video_length': args.video_length,
        'object_wise': {
            'enabled': args.objwise,
            'dataset': args.dataset.replace('_seg', '')
        },
        # New depth mode and model-specific settings
        'depth_mode': args.depth_mode,
        'indoor': args.indoor,
        'metric': args.metric,
        'frame_interval': args.frame_interval,
        'only_clone': (args.only_clone == 'true'),  # Convert string to bool
        'visualization': (args.visualization == 'true'),  # Convert string to bool
        'seq_list': seq_list,  # Pass seq_list for filtering
        'figure': args.figure,  # Export individual frames if enabled
        'frame': args.frame,  # Export specific frame ±4 frames
        'amp': args.amp,
        'amp_dtype': args.amp_dtype,
        'limit_scenes': args.limit_scenes
    }

    # Import and create adapter (VIDEO MODELS ONLY)
    try:
        if args.method == 'vda':
            from adapters.video_depth_anything_adapter import VideoDepthAnythingAdapter
            adapter = VideoDepthAnythingAdapter(metric=args.metric)
        elif args.method == 'depthcrafter':
            from adapters.depthcrafter_adapter import DepthCrafterAdapter
            adapter = DepthCrafterAdapter()
        else:
            # Should never reach here due to earlier validation
            raise ValueError(f"Unknown video method: {args.method}")
    except ImportError as e:
        logger.error(f"Failed to import adapter for {args.method}: {e}")
        logger.error("Make sure the adapter is implemented in adapters/ directory")
        sys.exit(1)

    # Create tester and run
    tester = VideoComparisonTester(method_name, config, adapter)
    tester.test()

    logger.info("Testing completed successfully!")


if __name__ == '__main__':
    main()
