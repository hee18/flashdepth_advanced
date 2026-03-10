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
from utils.metric_depth_metrics import MetricDepthMetrics, RelativeDepthMetrics
from utils.fgwise_evaluation import (
    FGWiseMetrics, aggregate_fgwise_metrics,
    save_fgwise_visualization, draw_fg_contours, create_depth_with_fg_overlay
)
from utils.comparison_visualization import visualize_sequence_simplified, visualize_best_frame_simplified
from utils.reprojection_tae import ReprojectionTAECalculator
from utils.temporal_consistency import FlowTemporalConsistency

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

        self.dataset_name = dataset_name  # Store for later use

        # FG-wise evaluation setup
        self.fgwise_enabled = config.get('fg_wise', {}).get('enabled', False)
        if self.fgwise_enabled:
            data_root = config.dataset.get('data_root', '/home/cvlab/hsy/Datasets')
            logger.info(f"FG-wise evaluation ENABLED (data_root: {data_root})")
            self.fgwise_data_root = data_root
        else:
            self.fgwise_data_root = None

        # Metrics calculators
        self.metrics = MetricDepthMetrics()
        self.relative_metrics = RelativeDepthMetrics()

        # Setup reprojection TAE calculator
        data_root = config.get('data_root', '/home/cvlab/hsy/Datasets')
        self.reproj_tae_calculator = ReprojectionTAECalculator(data_root)
        logger.info(f"Reprojection TAE calculator initialized (supported: {self.reproj_tae_calculator.SUPPORTED_DATASETS})")

        # Flow-based temporal consistency (lazy-loaded)
        self.flow_tc = None
        self.tc_threshold = config.get('tc_threshold', 1.1)
        self.max_depth = config.get('max_depth', 80.0)
        self.test_mode = config.get('test_mode', None)

        logger.info(f"Max depth threshold: {self.max_depth}m")

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
            'flashdepth': 'FlashDepth outputs relative depth (requires scale/shift alignment)',
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
                error_msg += f"  python test_video_comparison.py --method depthcrafter --depth-mode relative --dataset {{dataset}}\n"
            elif method == 'vda':
                error_msg += f"  python test_video_comparison.py --method vda --depth-mode relative --dataset {{dataset}}\n"
                error_msg += f"  # OR for metric VideoDepthAnything:\n"
                error_msg += f"  python test_video_comparison.py --method vda --metric --depth-mode metric --dataset {{dataset}}\n"
            elif method == 'flashdepth':
                error_msg += f"  python test_video_comparison.py --method flashdepth --depth-mode relative --dataset {{dataset}}\n"

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

        # Remove _seg suffix if present (for unified naming)
        base_dataset_name = dataset_name.replace('_seg', '') if dataset_name.endswith('_seg') else dataset_name

        # Standard evaluation - use ComparisonDataset (metric depth)
        # ComparisonDataset provides ORIGINAL resolution images
        # waymo_seg: uses 'val' split
        # others: use 'test' split
        if base_dataset_name == 'waymo':
            split = 'val'
        else:
            split = 'test'

        # Get only_clone flag for VKITTI
        only_clone = self.config.get('only_clone', False) if base_dataset_name == 'vkitti' else False

        # Get seq_list from config (supports all datasets now)
        seq_list = self.config.get('seq_list', None)

        # Use base_dataset_name (with _seg removed) for ComparisonDataset
        # ComparisonDataset will look for waymo_seg, vkitti, urbansyn directories
        dataset = ComparisonDataset(
            dataset_name=base_dataset_name if base_dataset_name != 'waymo' else 'waymo_seg',
            data_root=data_root,
            split=split,
            video_length=video_length,
            objwise_enabled=False,
            only_clone=only_clone,
            seq_list=seq_list,
            limit_scenes=self.config.get('limit_scenes')
        )
        collate_fn = comparison_collate_fn
        logger.info(f"Standard dataset: {base_dataset_name} (ComparisonDataset, metric depth)")

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

                if self.test_mode == 'tc':
                    logger.info(f"Sequence {sequence_id}: "
                               f"rTC={metrics.get('rtc', 0):.4f}, "
                               f"PSR={metrics.get('psr', 0):.4f}, "
                               f"TAE={metrics.get('tae', 0):.4f}")
                else:
                    logger.info(f"Sequence {sequence_id}: "
                               f"AbsRel={metrics.get('abs_rel', 0):.4f}, "
                               f"δ1={metrics.get('a1', 0):.4f}, "
                               f"TAE={metrics.get('tae', 0):.4f}")

            except Exception as e:
                logger.error(f"Error processing sequence {sequence_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
            finally:
                # Clear GPU cache between sequences to prevent memory accumulation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # TC-only mode: save temporal_consistency.json + tc_summary.json and return
        if self.test_mode == 'tc':
            self._save_temporal_consistency(self.all_results)
            self._save_tc_summary(self.all_results)
            logger.info("TC-only mode: saved temporal_consistency.json, tc_summary.json")
            return

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

        # Get inputs - ComparisonDataset format (metric depth in meters, original resolution)
        images = batch['images'].to(self.device)  # [1, T, 3, H, W]
        H, W = images.shape[-2:]
        logger.info(f"Sequence {sequence_id}: Processing {images.shape[1]} frames at {H}x{W} resolution")

        gt_depths = batch['depths'].to(self.device)  # [1, T, H, W] - metric depth (meters)

        intrinsics = batch.get('intrinsics', None)
        if intrinsics is not None:
            intrinsics = intrinsics.to(self.device)  # [1, T, 4] - fx, fy, cx, cy

        # Get depth file paths for completed depth loading (visualization)
        depth_paths = batch.get('depth_paths', None)  # List[str] or None

        # Ensure proper dimensions
        if images.ndim == 4:
            images = images.unsqueeze(0)  # [T, 3, H, W] -> [1, T, 3, H, W]
        if gt_depths.ndim == 3:
            gt_depths = gt_depths.unsqueeze(0)  # [T, H, W] -> [1, T, H, W]

        B, T = images.shape[:2]
        assert B == 1, "Batch size must be 1 for testing"

        # Process GT depth - ComparisonDataset provides metric depth (meters)
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

        # ComparisonDataset returns [0, 1] range - no unnormalization needed

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
                        images,  # [1, T, 3, H, W]
                        intrinsics=intrinsics  # [1, T, 4] or None
                    )
            else:
                _ = self.adapter.inference(
                    images,  # [1, T, 3, H, W]
                    intrinsics=intrinsics  # [1, T, 4] or None
                )

        # Free warmup result to prevent OOM on high-res datasets
        del _
        torch.cuda.empty_cache()

        # Timed run
        logger.info("Timed inference...")
        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            if self.config.get('amp', False):
                amp_dtype = torch.bfloat16 if self.config.get('amp_dtype', 'bf16') == 'bf16' else torch.float16
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    pred_depths = self.adapter.inference(
                        images,  # [1, T, 3, H, W]
                        intrinsics=intrinsics  # [1, T, 4] or None
                    )  # Returns [1, T, H, W] in meters
            else:
                pred_depths = self.adapter.inference(
                    images,  # [1, T, 3, H, W]
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

        # Free GPU images, keep CPU copy in batch for visualization/rTC
        del images
        torch.cuda.empty_cache()
        images = batch['images']  # [1, T, 3, H, W] CPU

        # Ensure correct shape [T, 1, H, W] for compatibility with visualization
        if pred_depths.ndim == 4 and pred_depths.shape[0] == 1:
            pred_depths = pred_depths[0]  # [1, T, H, W] -> [T, H, W]
        if pred_depths.ndim == 3:
            pred_depths = pred_depths.unsqueeze(1)  # [T, H, W] -> [T, 1, H, W]

        # Move pred to CPU immediately to free GPU memory
        pred_depths_cpu = pred_depths.cpu()
        del pred_depths
        torch.cuda.empty_cache()
        pred_depths = pred_depths_cpu  # Reassign to CPU version for later use

        # Best frame tracking
        best_frame_idx = 0
        best_frame_abs_rel = float('inf')
        frame_metrics = []

        # === Resolution handling (upsample pred to GT resolution for metrics) ===
        MAX_DEPTH = self.max_depth
        gt_depth_metric_cpu = gt_depth_processed[0].cpu()  # [T, 1, gt_H, gt_W] CPU (no GPU downsample)
        pred_H, pred_W = pred_depths_cpu.shape[2:]
        gt_H, gt_W = gt_depth_metric_cpu.shape[2:]
        need_upsample = (gt_H, gt_W) != (pred_H, pred_W)

        if need_upsample:
            logger.info(f"Resolution mismatch: pred ({pred_H},{pred_W}) vs GT ({gt_H},{gt_W}). "
                        f"Will upsample pred to GT resolution for metrics.")

        # GT at pred resolution (for TC, visualization, PSR)
        if need_upsample:
            ds_lower = self.dataset_name.lower() if isinstance(self.dataset_name, str) else 'unknown'
            is_sparse_ds = any(s in ds_lower for s in ['eth3d', 'waymo_seg'])
            interp_mode = 'nearest' if is_sparse_ds else 'bilinear'
            interp_kwargs = {} if is_sparse_ds else {'align_corners': False}
            gt_at_pred_res_cpu = F.interpolate(
                gt_depth_metric_cpu, size=(pred_H, pred_W), mode=interp_mode, **interp_kwargs
            )
            # Ensure images match pred resolution for TC flow estimation
            if images.shape[-2:] != (pred_H, pred_W):
                images = F.interpolate(
                    images[0], size=(pred_H, pred_W),
                    mode='bilinear', align_corners=False
                ).unsqueeze(0)
                batch['images'] = images.cpu()
        else:
            gt_at_pred_res_cpu = gt_depth_metric_cpu  # Same tensor, no copy needed

        # Free GPU tensors
        del gt_depths, gt_depth_processed
        torch.cuda.empty_cache()

        # === Scale/shift alignment for relative depth methods (per-sequence) ===
        # Compute alignment but keep raw predictions for rTC/PSR (scale-invariant metrics).
        # Only TAE needs aligned predictions (metric-scale reprojection).
        seq_scale, seq_shift = 1.0, 0.0
        pred_depths_aligned = pred_depths_cpu  # Default: no alignment (metric methods)

        if self.depth_mode == 'relative':
            gt_all = gt_depth_metric_cpu[:, 0]  # [T, gt_H, gt_W]
            # Upsample pred to GT resolution for alignment
            if need_upsample:
                pred_for_align = torch.stack([
                    F.interpolate(pred_depths_cpu[t:t+1], size=(gt_H, gt_W),
                                  mode='bilinear', align_corners=True)[0]
                    for t in range(pred_depths_cpu.shape[0])
                ], dim=0)[:, 0]  # [T, gt_H, gt_W]
            else:
                pred_for_align = pred_depths_cpu[:, 0]  # [T, gt_H, gt_W]

            valid_align = (gt_all > 0) & (gt_all < MAX_DEPTH) & \
                          (pred_for_align > 0) & torch.isfinite(pred_for_align)
            pred_valid = pred_for_align[valid_align]
            gt_valid = gt_all[valid_align]

            if len(pred_valid) > 100:
                # Least-squares in disparity space: gt_disp = s * pred_disp + t
                pred_disp = 1.0 / pred_valid
                gt_disp = 1.0 / gt_valid
                A = torch.stack([pred_disp, torch.ones_like(pred_disp)], dim=1)
                try:
                    solution = torch.linalg.lstsq(A, gt_disp, rcond=None).solution
                    disp_scale, disp_shift = solution[0].item(), solution[1].item()
                except:
                    disp_scale = (torch.median(gt_disp / (pred_disp + 1e-8))).item()
                    disp_shift = 0.0

                # Create aligned copy (for TAE only), keep raw pred_depths_cpu intact
                pred_disp_all = 1.0 / pred_depths_cpu.clamp(min=1e-8)
                pred_disp_aligned = disp_scale * pred_disp_all + disp_shift
                pred_depths_aligned = 1.0 / pred_disp_aligned.clamp(min=1e-8)

                aligned_valid = pred_depths_aligned[pred_depths_aligned > 0]
                logger.info(f"Sequence {sequence_id} alignment (disparity): s={disp_scale:.4f}, t={disp_shift:.4f}")
                logger.info(f"  Aligned depth range: min={aligned_valid.min():.2f}, max={aligned_valid.max():.2f}, mean={aligned_valid.mean():.2f}")
                seq_scale, seq_shift = disp_scale, disp_shift
            else:
                logger.warning(f"Sequence {sequence_id}: too few valid pixels ({len(pred_valid)}), skipping alignment")
            del pred_for_align

        # === TC mode: compute rTC, PSR (raw preds), TAE (aligned preds) ===
        if self.test_mode == 'tc':
            metrics = {
                'fps': float(fps), 'dataset': str(batch.get('dataset_name', 'unknown')),
                'num_frames': pred_depths_cpu.shape[0],
                'tae': 0.0, 'tae_reproj': 0.0, 'tae_reproj_gt': 0.0,
            }
            if self.depth_mode == 'relative':
                metrics['scale'] = seq_scale
                metrics['shift'] = seq_shift

            # --- rTC: use RAW predictions (scale-invariant) ---
            T_tc = pred_depths_cpu.shape[0]
            if T_tc > 1:
                if self.flow_tc is None:
                    self.flow_tc = FlowTemporalConsistency(
                        device=self.device, thr=self.tc_threshold, max_depth=MAX_DEPTH
                    )
                images_for_tc = batch['images'][0] if 'images' in batch else None
                if images_for_tc is not None:
                    tc_result = self.flow_tc.compute_rtc(
                        images_for_tc, pred_depths_cpu, gt_depths=gt_at_pred_res_cpu
                    )
                    metrics['rtc'] = tc_result['rtc']
                    metrics['rtc_gt'] = tc_result['rtc_gt']
                    metrics['_per_frame_rtc'] = tc_result['per_frame_rtc']
                    metrics['_per_frame_rtc_gt'] = tc_result['per_frame_rtc_gt']
                    metrics['_rtc_ratio_stats'] = tc_result['ratio_stats']
                    metrics['_rtc_per_frame_ratio_stats'] = tc_result['per_frame_ratio_stats']
                    metrics['_rtc_best_frame_idx'] = tc_result['best_frame_idx']
                    metrics['_rtc_worst_frame_idx'] = tc_result['worst_frame_idx']
                    logger.info(f"Flow TC: rTC={metrics['rtc']:.4f}, rTC_gt={metrics['rtc_gt']:.4f}")

                    # TC visualizations
                    try:
                        rtc_best = tc_result['best_frame_idx']
                        rtc_worst = tc_result['worst_frame_idx']
                        per_frame_rtc = tc_result['per_frame_rtc']
                        per_frame_rtc_gt = tc_result['per_frame_rtc_gt']

                        self.flow_tc.save_visualization(
                            pred_depths_cpu, gt_at_pred_res_cpu, rtc_worst, sequence_id,
                            self.save_dir, per_frame_rtc[rtc_worst], label='worst',
                            dataset_name=self.dataset_name
                        )
                        self.flow_tc.save_visualization(
                            pred_depths_cpu, gt_at_pred_res_cpu, rtc_best, sequence_id,
                            self.save_dir, per_frame_rtc[rtc_best], label='best',
                            dataset_name=self.dataset_name
                        )
                        self.flow_tc.save_ratio_heatmap(
                            images_for_tc, pred_depths_cpu, rtc_worst, sequence_id,
                            self.save_dir, per_frame_rtc[rtc_worst], label='worst',
                            dataset_name=self.dataset_name
                        )
                        self.flow_tc.save_ratio_heatmap(
                            images_for_tc, pred_depths_cpu, rtc_best, sequence_id,
                            self.save_dir, per_frame_rtc[rtc_best], label='best',
                            dataset_name=self.dataset_name
                        )
                        self.flow_tc.save_rtc_plot(
                            per_frame_rtc, per_frame_rtc_gt, rtc_best, rtc_worst,
                            sequence_id, self.save_dir,
                            dataset_name=self.dataset_name
                        )
                        logger.info(f"TC visualizations saved for sequence {sequence_id}")
                    except Exception as e:
                        logger.warning(f"Failed to save TC visualizations: {e}")
                else:
                    metrics['rtc'] = 0.0
                    metrics['rtc_gt'] = 0.0
            else:
                metrics['rtc'] = 0.0
                metrics['rtc_gt'] = 0.0

            # --- TAE: use ALIGNED predictions (needs metric scale) ---
            T_tc_frames = pred_depths_cpu.shape[0]
            dataset_name_for_tae = batch.get('dataset_name', 'unknown')
            if isinstance(dataset_name_for_tae, (list, tuple)):
                dataset_name_for_tae = dataset_name_for_tae[0]
            dataset_name_for_tae = dataset_name_for_tae.lower() if isinstance(dataset_name_for_tae, str) else 'unknown'
            if T_tc_frames > 1 and 'image_paths' in batch and self.reproj_tae_calculator.is_supported(dataset_name_for_tae):
                try:
                    image_paths_for_tae = batch['image_paths'][0]
                    # TAE at GT resolution using aligned predictions
                    if need_upsample:
                        pred_at_gt_res = torch.stack([
                            F.interpolate(pred_depths_aligned[t:t+1], size=(gt_H, gt_W),
                                          mode='bilinear', align_corners=True)[0]
                            for t in range(T_tc_frames)
                        ], dim=0)
                    else:
                        pred_at_gt_res = pred_depths_aligned
                    reproj_tae_result = self.reproj_tae_calculator.compute_tae(
                        pred_at_gt_res[:, 0],
                        gt_depth_metric_cpu[:, 0],
                        dataset_name_for_tae,
                        image_paths_for_tae,
                        max_depth=MAX_DEPTH
                    )
                    if need_upsample:
                        del pred_at_gt_res
                    metrics['tae_reproj'] = reproj_tae_result.get('tae_reproj', 0.0)
                    metrics['tae_reproj_gt'] = reproj_tae_result.get('tae_reproj_gt', 0.0)
                    metrics['tae'] = reproj_tae_result.get('tae', 0.0)
                    logger.info(f"Reprojection TAE: {metrics['tae_reproj']:.4f} (GT ref: {metrics['tae_reproj_gt']:.4f}), TAE diff: {metrics['tae']:.4f}")
                except Exception as e:
                    logger.warning(f"Failed to compute reprojection TAE: {e}")

            # --- PSR: use RAW predictions (measures scale stability) ---
            per_frame_scale_ratios_tc = []
            for t in range(T_tc_frames):
                pred_frame = pred_depths_cpu[t, 0]
                gt_frame = gt_at_pred_res_cpu[t, 0]
                valid_mask = (gt_frame > 0) & (gt_frame < MAX_DEPTH) & (pred_frame > 0) & (pred_frame < MAX_DEPTH)
                if valid_mask.sum() > 0:
                    r_t = float(pred_frame[valid_mask].mean() / gt_frame[valid_mask].mean().clamp(min=1e-8))
                    per_frame_scale_ratios_tc.append(r_t)
                else:
                    per_frame_scale_ratios_tc.append(per_frame_scale_ratios_tc[-1] if per_frame_scale_ratios_tc else 1.0)
            if T_tc_frames > 1 and len(per_frame_scale_ratios_tc) > 1:
                psr_values = [abs(per_frame_scale_ratios_tc[i] - per_frame_scale_ratios_tc[i-1]) for i in range(1, len(per_frame_scale_ratios_tc))]
                metrics['psr'] = float(np.mean(psr_values))
                metrics['psr_max'] = float(np.max(psr_values))
                metrics['_per_frame_psr'] = [float(v) for v in psr_values]
                metrics['_per_frame_scale_ratio'] = [float(v) for v in per_frame_scale_ratios_tc]
            else:
                metrics['psr'] = 0.0
                metrics['psr_max'] = 0.0
                metrics['_per_frame_psr'] = []
                metrics['_per_frame_scale_ratio'] = []

            return metrics

        # Regular metrics computation (at GT resolution)
        per_frame_scale_ratios = []
        for t in range(pred_depths.shape[0]):
            # Main metrics at GT resolution (upsample pred per-frame)
            gt_frame = gt_depth_metric_cpu[t, 0]  # [gt_H, gt_W]
            if need_upsample:
                pred_frame = F.interpolate(
                    pred_depths_cpu[t:t+1], size=(gt_H, gt_W),
                    mode='bilinear', align_corners=True
                )[0, 0]  # [gt_H, gt_W]
            else:
                pred_frame = pred_depths_cpu[t, 0]  # [H, W]

            # Create valid mask
            gt_valid_mask = (gt_frame > 0) & (gt_frame < MAX_DEPTH)
            pred_valid_mask = (pred_frame > 0) & (pred_frame < MAX_DEPTH)
            valid_mask = gt_valid_mask & pred_valid_mask

            # PSR: compute per-frame scale ratio (mean_pred / mean_gt) on valid pixels
            if valid_mask.sum() > 0:
                r_t = float(pred_frame[valid_mask].mean() / gt_frame[valid_mask].mean().clamp(min=1e-8))
                per_frame_scale_ratios.append(r_t)
            else:
                per_frame_scale_ratios.append(per_frame_scale_ratios[-1] if per_frame_scale_ratios else 1.0)

            if valid_mask.sum() > 0:
                # Compute metrics based on depth mode
                if self.depth_mode == 'metric':
                    frame_metric = self.metrics.compute_metric_depth_metrics(
                        pred_frame, gt_frame, valid_mask
                    )
                    # Track best frame (lowest AbsRel for metric)
                    if frame_metric['abs_rel'] < best_frame_abs_rel:
                        best_frame_abs_rel = frame_metric['abs_rel']
                        best_frame_idx = t
                else:  # relative
                    frame_metric = self.relative_metrics.compute_relative_depth_metrics(
                        pred_frame, gt_frame, valid_mask
                    )
                    # Track best frame (lowest AbsRel for relative)
                    if frame_metric['abs_rel_si'] < best_frame_abs_rel:
                        best_frame_abs_rel = frame_metric['abs_rel_si']
                        best_frame_idx = t

                frame_metrics.append(frame_metric)

        # Average metrics
        if len(frame_metrics) == 0:
            logger.warning(f"No valid frames for sequence {sequence_id}")
            return {k: 0.0 for k in ["mae", "rmse", "abs_rel", "a1", "tae", "fps"]}

        metrics = {}
        for key in frame_metrics[0].keys():
            values = [m[key] for m in frame_metrics]
            metrics[key] = np.mean(values)

        # Compute Reprojection-based TAE (for datasets with camera poses)
        # tae = tae_reproj - tae_reproj_gt (pure prediction error, removing occlusion effects)
        dataset_name_for_tae = batch.get('dataset_name', 'unknown')
        if isinstance(dataset_name_for_tae, (list, tuple)):
            dataset_name_for_tae = dataset_name_for_tae[0]
        dataset_name_for_tae = dataset_name_for_tae.lower() if isinstance(dataset_name_for_tae, str) else 'unknown'

        if self.test_mode == 'ea':
            # EA mode: skip TAE
            metrics['tae_reproj'] = 0.0
            metrics['tae_reproj_gt'] = 0.0
            metrics['tae'] = 0.0
        elif len(pred_depths) > 1 and 'image_paths' in batch and self.reproj_tae_calculator.is_supported(dataset_name_for_tae):
            try:
                image_paths_for_tae = batch['image_paths'][0]  # batch size is 1
                # TAE at GT resolution — use aligned predictions for relative depth
                pred_for_tae = pred_depths_aligned if self.depth_mode == 'relative' else pred_depths_cpu
                if need_upsample:
                    pred_at_gt_res = torch.stack([
                        F.interpolate(pred_for_tae[t:t+1], size=(gt_H, gt_W),
                                      mode='bilinear', align_corners=True)[0]
                        for t in range(pred_for_tae.shape[0])
                    ], dim=0)
                else:
                    pred_at_gt_res = pred_for_tae
                reproj_tae_result = self.reproj_tae_calculator.compute_tae(
                    pred_at_gt_res[:, 0],  # [T, gt_H, gt_W]
                    gt_depth_metric_cpu[:, 0],  # [T, gt_H, gt_W]
                    dataset_name_for_tae,
                    image_paths_for_tae,
                    max_depth=MAX_DEPTH
                )
                if need_upsample:
                    del pred_at_gt_res
                metrics['tae_reproj'] = reproj_tae_result.get('tae_reproj', 0.0)
                metrics['tae_reproj_gt'] = reproj_tae_result.get('tae_reproj_gt', 0.0)
                metrics['tae'] = reproj_tae_result.get('tae', 0.0)  # tae_reproj - tae_reproj_gt
                if reproj_tae_result.get('tae_reproj_supported', False):
                    logger.info(f"Reprojection TAE: {metrics['tae_reproj']:.4f} (GT ref: {metrics['tae_reproj_gt']:.4f}), TAE diff: {metrics['tae']:.4f}")
            except Exception as e:
                logger.warning(f"Failed to compute reprojection TAE: {e}")
                metrics['tae_reproj'] = 0.0
                metrics['tae_reproj_gt'] = 0.0
                metrics['tae'] = 0.0
        else:
            metrics['tae_reproj'] = 0.0
            metrics['tae_reproj_gt'] = 0.0
            metrics['tae'] = 0.0

        # === Flow-based Temporal Consistency (rTC) ===
        if self.test_mode == 'ea':
            # EA mode: skip rTC (avoids loading SEA-RAFT, saves ~200MB GPU)
            metrics['rtc'] = 0.0
            metrics['rtc_gt'] = 0.0
        elif len(pred_depths) > 1:
            if self.flow_tc is None:
                self.flow_tc = FlowTemporalConsistency(
                    device=self.device, thr=self.tc_threshold, max_depth=MAX_DEPTH
                )
            images_for_tc = batch['images'][0] if 'images' in batch else None
            if images_for_tc is not None:
                tc_result = self.flow_tc.compute_rtc(
                    images_for_tc, pred_depths_cpu, gt_depths=gt_at_pred_res_cpu
                )
                metrics['rtc'] = tc_result['rtc']
                metrics['rtc_gt'] = tc_result['rtc_gt']
                metrics['_per_frame_rtc'] = tc_result['per_frame_rtc']
                metrics['_per_frame_rtc_gt'] = tc_result['per_frame_rtc_gt']
                metrics['_rtc_ratio_stats'] = tc_result['ratio_stats']
                metrics['_rtc_per_frame_ratio_stats'] = tc_result['per_frame_ratio_stats']
                metrics['_rtc_best_frame_idx'] = tc_result['best_frame_idx']
                metrics['_rtc_worst_frame_idx'] = tc_result['worst_frame_idx']
                logger.info(f"Flow TC: rTC={metrics['rtc']:.4f}, rTC_gt={metrics['rtc_gt']:.4f}")
            else:
                metrics['rtc'] = 0.0
                metrics['rtc_gt'] = 0.0
        else:
            metrics['rtc'] = 0.0
            metrics['rtc_gt'] = 0.0

        # === Prediction Stability Ratio (PSR) ===
        if self.test_mode == 'ea':
            # EA mode: skip PSR
            metrics['psr'] = 0.0
            metrics['psr_max'] = 0.0
            metrics['_per_frame_psr'] = []
            metrics['_per_frame_scale_ratio'] = []
        else:
            T = pred_depths.shape[0]
            if T > 1 and len(per_frame_scale_ratios) > 1:
                psr_values = []
                for i in range(1, len(per_frame_scale_ratios)):
                    psr_values.append(abs(per_frame_scale_ratios[i] - per_frame_scale_ratios[i-1]))
                metrics['psr'] = float(np.mean(psr_values))
                metrics['psr_max'] = float(np.max(psr_values))
                metrics['_per_frame_psr'] = [float(v) for v in psr_values]
                metrics['_per_frame_scale_ratio'] = [float(v) for v in per_frame_scale_ratios]
            else:
                metrics['psr'] = 0.0
                metrics['psr_max'] = 0.0
                metrics['_per_frame_psr'] = []
                metrics['_per_frame_scale_ratio'] = []

        metrics['fps'] = fps

        # FG-wise evaluation: compute metrics separately for FG/BG regions
        if self.fgwise_enabled and 'image_paths' in batch:
            try:
                dataset_name = batch.get('dataset_name', [''])[0] if isinstance(batch.get('dataset_name'), list) else batch.get('dataset_name', '')
                fgwise_metrics_calc = FGWiseMetrics(self.fgwise_data_root, dataset_name)

                fgwise_metrics_list = []
                image_paths = batch['image_paths'][0]  # batch size is 1
                T_fg = min(len(image_paths), pred_depths.shape[0])

                for t in range(T_fg):
                    # FG-wise metrics at GT resolution
                    gt_frame = gt_depth_metric_cpu[t, 0]
                    if need_upsample:
                        pred_frame = F.interpolate(
                            pred_depths_cpu[t:t+1], size=(gt_H, gt_W),
                            mode='bilinear', align_corners=True
                        )[0, 0]
                    else:
                        pred_frame = pred_depths_cpu[t, 0]
                    pred_frame = pred_frame.numpy()
                    gt_frame = gt_frame.numpy()

                    # Extract scene and frame from image path
                    image_path = image_paths[t]
                    scene, frame = self._extract_scene_frame_for_fgwise(image_path, dataset_name)

                    if scene and frame:
                        # Create valid mask with max_depth limit (consistent with regular metrics)
                        gt_valid_mask = (gt_frame > 0) & (gt_frame < MAX_DEPTH)
                        pred_valid_mask = (pred_frame > 0) & (pred_frame < MAX_DEPTH)
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
                visualize_sequence_simplified(
                    images[0], pred_depths, gt_at_pred_res_cpu,
                    valid_mask=(gt_at_pred_res_cpu > 0),
                    sequence_id=sequence_id,
                    metrics=metrics,
                    fps=fps,
                    save_dir=self.save_dir,
                    focal_lengths=focal_lengths[0] if focal_lengths is not None else None,
                    frame_interval=self.frame_interval,  # Pass frame interval for visualization
                    depth_paths=depth_paths,  # For completed depth visualization (ETH3D/Waymo)
                    dataset_name=dataset_name_lower,  # e.g., 'eth3d', 'waymo_seg'
                    max_depth=MAX_DEPTH
                )
            else:
                logger.info(f"Skipping sequence visualization for {dataset_name} (high resolution dataset)")

        # Save best frame visualization
        has_metrics = len(frame_metrics) > 0
        if self.enable_visualization and has_metrics:
            # Log best frame with appropriate metric
            if self.depth_mode == 'metric':
                logger.info(f"Best frame for sequence {sequence_id}: Frame {best_frame_idx} (AbsRel={best_frame_abs_rel:.4f})")
                best_metric_value = best_frame_abs_rel
            else:  # relative
                logger.info(f"Best frame for sequence {sequence_id}: Frame {best_frame_idx} (AbsRel={best_frame_abs_rel:.4f})")
                best_metric_value = best_frame_abs_rel

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

            # Get frame-specific metrics for visualization
            frame_viz_metrics = frame_metrics[best_frame_idx] if best_frame_idx < len(frame_metrics) else {}

            visualize_best_frame_simplified(
                image=images[0, best_frame_idx],  # [3, H, W]
                gt_depth=gt_at_pred_res_cpu[best_frame_idx, 0],  # [H, W] at pred resolution
                pred_depth=pred_depths[best_frame_idx, 0],  # [H, W]
                metrics=frame_viz_metrics,
                save_dir=self.save_dir,
                sequence_id=sequence_id,
                frame_idx=best_frame_idx,
                dataset_name=dataset_name,
                focal_length=frame_focal_length,
                gt_depth_path=depth_paths[best_frame_idx] if depth_paths and best_frame_idx < len(depth_paths) else None,
                max_depth=MAX_DEPTH
            )

        # Export individual frames if --best-figure or --frame option is enabled
        # NOTE: This is independent of --visualization flag
        has_metrics = len(frame_metrics) > 0
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
                gt_depths=gt_at_pred_res_cpu,  # [T, 1, H, W] at pred resolution
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
        for key in ['mae', 'rmse', 'abs_rel', 'sq_rel', 'rmse_log', 'a1', 'a2', 'a3', 'tae', 'tae_reproj', 'tae_reproj_gt', 'rtc', 'rtc_gt', 'psr', 'psr_max', 'fps']:
            values = [r[key] for r in self.all_results if key in r]
            if len(values) > 0:
                avg_metrics_raw[key] = float(np.mean(values))

        # Reorder metrics according to desired order
        metric_order = ['abs_rel', 'a1', 'a2', 'a3', 'fps', 'tae', 'tae_reproj', 'tae_reproj_gt', 'rtc', 'rtc_gt', 'psr', 'psr_max', 'mae', 'rmse']
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
        logger.info(f"  FPS: {avg_metrics.get('fps', 0):.2f}")
        logger.info(f"Results saved to {results_path}")

        # Save per-sequence results with reordered metrics
        per_seq_path = self.save_dir / 'per_sequence_results.json'
        with open(per_seq_path, 'w') as f:
            json.dump(reordered_results, f, indent=2)

        logger.info(f"Per-sequence results saved to {per_seq_path}")

        # Find and save best/worst sequences (by AbsRel)
        if len(self.all_results) > 0:
            best_seq = min(self.all_results, key=lambda x: x.get('abs_rel', float('inf')))
            worst_seq = max(self.all_results, key=lambda x: x.get('abs_rel', 0))

            best_seq_path = self.save_dir / 'best_sequence.json'
            with open(best_seq_path, 'w') as f:
                json.dump(best_seq, f, indent=2)
            logger.info(f"\nBest sequence (lowest AbsRel): {best_seq.get('sequence_id', 'N/A')}")
            logger.info(f"  AbsRel: {best_seq.get('abs_rel', 0):.4f}, δ1: {best_seq.get('a1', 0):.4f}")

            worst_seq_path = self.save_dir / 'worst_sequence.json'
            with open(worst_seq_path, 'w') as f:
                json.dump(worst_seq, f, indent=2)
            logger.info(f"Worst sequence (highest AbsRel): {worst_seq.get('sequence_id', 'N/A')}")
            logger.info(f"  AbsRel: {worst_seq.get('abs_rel', 0):.4f}, δ1: {worst_seq.get('a1', 0):.4f}")

        # Save temporal_consistency.json (flow-based rTC)
        self._save_temporal_consistency(self.all_results)

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

    def _save_temporal_consistency(self, all_results):
        """Save temporal_consistency.json (flow-based rTC metrics)."""
        has_rtc = any(m.get('rtc', 0) > 0 for m in all_results)
        if not has_rtc:
            return

        per_sequence = []
        for result in all_results:
            entry = {
                'sequence_id': result.get('sequence_id', -1),
                'rtc': result.get('rtc', 0.0),
                'rtc_gt': result.get('rtc_gt', 0.0),
                'per_frame_rtc': result.get('_per_frame_rtc', []),
                'per_frame_rtc_gt': result.get('_per_frame_rtc_gt', []),
                'per_frame_ratio_stats': result.get('_rtc_per_frame_ratio_stats', []),
                'ratio_stats': result.get('_rtc_ratio_stats', {}),
                'best_frame_idx': result.get('_rtc_best_frame_idx', 0),
                'worst_frame_idx': result.get('_rtc_worst_frame_idx', 0)
            }
            per_sequence.append(entry)

        rtc_values = [m.get('rtc', 0.0) for m in all_results if m.get('rtc', 0) > 0]
        rtc_gt_values = [m.get('rtc_gt', 0.0) for m in all_results if m.get('rtc_gt', 0) > 0]

        all_ratio_stats = [m.get('_rtc_ratio_stats', {}) for m in all_results if m.get('_rtc_ratio_stats')]
        agg_ratio_stats = {}
        if all_ratio_stats:
            for key in ['avg', 'min', 'max', 'p90', 'p95']:
                values = [rs.get(key, 0.0) for rs in all_ratio_stats if key in rs]
                if values:
                    agg_ratio_stats[key] = float(np.mean(values))

        tc_output = {
            'config': {'threshold': self.tc_threshold, 'flow_model': 'sea_raft'},
            'aggregated': {
                'rtc': float(np.mean(rtc_values)) if rtc_values else 0.0,
                'rtc_gt': float(np.mean(rtc_gt_values)) if rtc_gt_values else 0.0,
                'ratio_stats': agg_ratio_stats
            },
            'per_sequence': per_sequence
        }

        tc_path = self.save_dir / "temporal_consistency.json"
        with open(tc_path, 'w') as f:
            json.dump(tc_output, f, indent=2, default=str)
        logger.info(f"Temporal consistency saved to {tc_path}")

    def _save_tc_summary(self, all_results):
        """Save tc_summary.json with rTC + TAE + PSR (dataset aggregate + per-sequence)."""
        per_sequence = []
        for r in all_results:
            entry = {
                'sequence_id': r.get('sequence_id', -1),
                'rtc': r.get('rtc', 0.0),
                'rtc_gt': r.get('rtc_gt', 0.0),
                'tae': r.get('tae', 0.0),
                'tae_reproj': r.get('tae_reproj', 0.0),
                'tae_reproj_gt': r.get('tae_reproj_gt', 0.0),
                'psr': r.get('psr', 0.0),
                'psr_max': r.get('psr_max', 0.0),
                'fps': r.get('fps', 0.0),
            }
            # Include scale/shift for relative depth alignment
            if 'scale' in r:
                entry['scale'] = r['scale']
                entry['shift'] = r['shift']
            per_sequence.append(entry)

        agg = {}
        for key in ['rtc', 'rtc_gt', 'tae', 'tae_reproj', 'tae_reproj_gt', 'psr', 'psr_max', 'fps']:
            vals = [s[key] for s in per_sequence if s[key] != 0.0]
            agg[key] = float(np.mean(vals)) if vals else 0.0

        summary = {'aggregated': agg, 'per_sequence': per_sequence}

        path = self.save_dir / "tc_summary.json"
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"TC summary saved to {path}")

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
            pred_depth = pred_depths[t, 0].cpu().numpy()  # [H, W] in meters

            # Compute gt_valid mask and determine sparse/dense FIRST
            MAX_DEPTH = self.max_depth
            gt_valid = (gt_depth > 0) & (gt_depth < MAX_DEPTH)
            gt_density = gt_valid.sum() / gt_valid.size
            is_sparse = gt_density < 0.5

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

                # Get vmin/vmax from completed depth for pred normalization
                if dataset_name == 'waymo_seg':
                    valid_mask = (gt_depth_for_vis > 0) & np.isfinite(gt_depth_for_vis)
                else:
                    valid_mask = np.isfinite(gt_depth_for_vis) & (gt_depth_for_vis > 0)

                if valid_mask.any():
                    gt_vmin = np.nanpercentile(gt_depth_for_vis[valid_mask], 2)
                    gt_vmax = np.nanpercentile(gt_depth_for_vis[valid_mask], 98)
                else:
                    gt_vmin, gt_vmax = None, None
            else:
                # For both sparse and dense datasets:
                # - Use gt_valid mask for GT visualization (exclude invalid and far depth)
                # - Compute vmin/vmax from gt_valid pixels
                gt_depth_vis = self._depth_to_colormap(gt_depth, external_mask=gt_valid)

                # Get vmin/vmax from gt_valid pixels (not just depth > 0)
                if gt_valid.any():
                    gt_vmin = np.nanpercentile(gt_depth[gt_valid], 2)
                    gt_vmax = np.nanpercentile(gt_depth[gt_valid], 98)
                else:
                    gt_vmin, gt_vmax = None, None

            gt_path = figures_dir / f"frame_{t:04d}_gt_depth.png"
            cv2.imwrite(str(gt_path), gt_depth_vis)

            # 3. Save pred depth (colormap) - use GT range for comparison
            # Same logic as main visualization (sequence.png)

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
    parser.add_argument('--test-mode', type=str, default=None, choices=['tc', 'ea'],
                       help='Test mode: tc (temporal consistency only), ea (error & accuracy only, skip rTC)')
    parser.add_argument('--tc-threshold', type=float, default=1.1,
                       help='Threshold for rTC metric (default: 1.1)')
    parser.add_argument('--max-depth', type=float, default=80.0,
                       help='Maximum depth threshold in meters for valid mask filtering (default: 80.0)')

    args = parser.parse_args()

    # Validate method is a video model
    VIDEO_MODELS = ['vda', 'depthcrafter', 'flashdepth']
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
        # Depth mode and model-specific settings
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
        'limit_scenes': args.limit_scenes,
        'test_mode': args.test_mode,
        'tc_threshold': args.tc_threshold,
        'max_depth': args.max_depth
    }

    # Import and create adapter (VIDEO MODELS ONLY)
    try:
        if args.method == 'vda':
            from adapters.video_depth_anything_adapter import VideoDepthAnythingAdapter
            adapter = VideoDepthAnythingAdapter(metric=args.metric)
        elif args.method == 'depthcrafter':
            from adapters.depthcrafter_adapter import DepthCrafterAdapter
            adapter = DepthCrafterAdapter()
        elif args.method == 'flashdepth':
            from adapters.flashdepth_adapter import FlashDepthAdapter
            # Infer variant from checkpoint path
            variant = 'l'  # default
            if args.checkpoint:
                if 'flashdepth-s' in args.checkpoint:
                    variant = 's'
                elif 'flashdepth/' in args.checkpoint and 'flashdepth-' not in args.checkpoint:
                    variant = 'hybrid'
            adapter = FlashDepthAdapter(config_variant=variant, checkpoint_path=args.checkpoint)
            adapter.set_dataset(config.get('dataset', ''))
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
