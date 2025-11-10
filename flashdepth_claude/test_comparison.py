#!/usr/bin/env python3
"""
Test script for comparison depth estimation methods

Unified evaluation framework for comparing different depth estimation methods
using the same datasets and metrics as test_gear.

Supported methods:
- Video-Depth-Anything (vda)
- DepthCrafter (depthcrafter)
- Metric3D v1/v2 (metric3d)
- UniDepth v1/v2 (unidepth)
- ZoeDepth (zoedepth)
- DepthPro (depthpro)
- CUT3R (cut3r)
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
from dataloaders.waymo_segmentation_dataset import WaymoSegmentationDataset, collate_fn as waymo_collate_fn
from dataloaders.urbansyn_segmentation_dataset import UrbanSynSegmentationDataset, urbansyn_collate_fn
from utils.metric_depth_metrics import MetricDepthMetrics
from utils.object_wise_evaluation import ObjectWiseMetrics
from utils.comparison_visualization import visualize_sequence_simplified, visualize_best_frame_simplified

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComparisonTester:
    """
    Unified tester for comparison depth estimation methods
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

        # Setup save directory
        dataset_name = config.get('dataset', 'waymo')
        self.save_dir = Path(config.get('results_dir', f'refer_test/test_results/{method_name}/{dataset_name}'))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Testing method: {method_name}")
        logger.info(f"Results will be saved to: {self.save_dir}")

        # Object-wise evaluation setup
        self.object_wise_enabled = config.get('object_wise', {}).get('enabled', False)
        self.object_wise_dataset = config.get('object_wise', {}).get('dataset', 'waymo')

        if self.object_wise_enabled:
            self.object_wise_metrics = ObjectWiseMetrics(dataset_name=self.object_wise_dataset)
            logger.info(f"Object-wise evaluation enabled for {self.object_wise_dataset}")

        # Metrics calculator
        self.metrics = MetricDepthMetrics()

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

        logger.info(f"Model loaded successfully")
        return model

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
                dataset = WaymoSegmentationDataset(
                    data_root=data_root,
                    split='val',
                    video_length=video_length,
                    stride=1
                )
                collate_fn = waymo_collate_fn
            elif base_dataset_name == 'urbansyn':
                dataset = UrbanSynSegmentationDataset(
                    data_root=data_root,
                    split='val',
                    video_length=video_length,
                    stride=1
                )
                collate_fn = urbansyn_collate_fn
            else:
                raise ValueError(f"Unknown segmentation dataset: {dataset_name}")
        else:
            # Standard datasets
            dataset = CombinedDataset(
                data_root=data_root,
                datasets=[dataset_name],
                split='val',
                video_length=video_length,
                stride=1,
                use_canonical_space=False  # Comparison methods don't use canonical space
            )
            collate_fn = None

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.get('workers', 4),
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

        # Aggregate and save results
        self._aggregate_and_save_results()

    @torch.no_grad()
    def test_sequence(self, batch, sequence_id):
        """Test on a single sequence"""
        # Get inputs
        if 'images' in batch:
            images = batch['images'].to(self.device)  # [1, T, 3, H, W] or [T, 3, H, W]
            if images.ndim == 4:
                images = images.unsqueeze(0)
        else:
            images = batch['image'].to(self.device)  # [1, T, 3, H, W]

        gt_depth = batch['depth'].to(self.device)  # [1, T, H, W] or [T, H, W]
        if gt_depth.ndim == 3:
            gt_depth = gt_depth.unsqueeze(0)
        if gt_depth.ndim == 4:
            gt_depth = gt_depth.unsqueeze(2)  # [1, T, 1, H, W]

        focal_lengths = batch.get('focal_lengths', None)
        if focal_lengths is not None:
            focal_lengths = focal_lengths.to(self.device)

        B, T = images.shape[:2]
        assert B == 1, "Batch size must be 1 for testing"

        # GT is in meters (1/m format, so convert: depth_m = 1 / (gt_depth + eps))
        gt_depth_metric = 1.0 / (gt_depth + 1e-8)  # [1, T, 1, H, W] in meters

        # Storage for predictions
        pred_depths = []

        # FPS measurement
        warmup_frames = min(5, T)
        start_time = None

        # Best frame tracking
        best_frame_idx = 0
        best_frame_abs_rel = float('inf')
        frame_metrics = []

        # Process each frame
        for t in range(T):
            # Start timing after warmup
            if t == warmup_frames:
                torch.cuda.synchronize()
                import time
                start_time = time.time()

            img_t = images[0, t]  # [3, H, W]

            # Method-specific inference using adapter
            pred_depth_t = self.adapter.inference(
                img_t.unsqueeze(0),  # [1, 3, H, W]
                intrinsics=focal_lengths[0, t] if focal_lengths is not None else None
            )  # Returns [1, H, W] in meters

            # Store prediction
            pred_depths.append(pred_depth_t)

            # End timing
            if t == T - 1 and start_time is not None:
                torch.cuda.synchronize()
                end_time = time.time()

        # Calculate FPS
        if start_time is not None:
            inference_time = end_time - start_time
            fps = (T - warmup_frames) / inference_time if inference_time > 0 else 0
            logger.info(f"Inference time: {inference_time:.4f}s for {T - warmup_frames} frames")
            logger.info(f"FPS: {fps:.2f}")
        else:
            fps = 0

        # Stack predictions
        pred_depths = torch.stack(pred_depths, dim=0)  # [T, 1, H, W]

        # Compute metrics
        pred_depths_cpu = pred_depths.cpu()
        gt_depth_metric_cpu = gt_depth_metric[0].cpu()  # [T, 1, H, W]

        for t in range(pred_depths.shape[0]):
            pred_frame = pred_depths_cpu[t, 0]  # [H, W]
            gt_frame = gt_depth_metric_cpu[t, 0]  # [H, W]

            # Create valid mask
            MAX_DEPTH = 70.0
            gt_valid_mask = (gt_frame > 0) & (gt_frame < MAX_DEPTH)
            pred_valid_mask = (pred_frame > 0) & (pred_frame < MAX_DEPTH)
            valid_mask = gt_valid_mask & pred_valid_mask

            if valid_mask.sum() > 0:
                frame_metric = self.metrics.compute_metric_depth_metrics(
                    pred_frame, gt_frame, valid_mask
                )
                frame_metrics.append(frame_metric)

                # Track best frame
                if frame_metric['abs_rel'] < best_frame_abs_rel:
                    best_frame_abs_rel = frame_metric['abs_rel']
                    best_frame_idx = t

        # Average metrics
        if len(frame_metrics) == 0:
            logger.warning(f"No valid frames for sequence {sequence_id}")
            return {k: 0.0 for k in ["mae", "rmse", "abs_rel", "a1", "tae", "fps", "boundary_f1"]}

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

        metrics['fps'] = fps

        # Object-wise evaluation
        if self.object_wise_enabled and 'segmentations' in batch:
            try:
                seg_masks = batch['segmentations'][0]  # [T, H, W]
                seg_masks_np = seg_masks.cpu().numpy() if isinstance(seg_masks, torch.Tensor) else seg_masks

                per_frame_class_metrics = []
                for t in range(len(seg_masks_np)):
                    pred_frame = pred_depths_cpu[t, 0].numpy()
                    gt_frame = gt_depth_metric_cpu[t, 0].numpy()
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
                        min_pixels=100
                    )
                    per_frame_class_metrics.append(frame_class_metrics)

                class_metrics = self.object_wise_metrics.aggregate_metrics(per_frame_class_metrics)
                metrics['object_wise'] = class_metrics
                logger.info(f"Computed object-wise metrics for {len(class_metrics)} classes")

            except Exception as e:
                logger.error(f"Error computing object-wise metrics: {e}")
                metrics['object_wise'] = {}

        # Visualize sequence (simplified version without importance maps)
        visualize_sequence_simplified(
            images[0], pred_depths, gt_depth_metric[0],
            valid_mask=(gt_depth_metric[0] > 0),
            sequence_id=sequence_id,
            metrics=metrics,
            fps=fps,
            save_dir=self.save_dir,
            focal_lengths=focal_lengths[0] if focal_lengths is not None else None
        )

        # Save best frame visualization
        if len(frame_metrics) > 0:
            logger.info(f"Best frame for sequence {sequence_id}: Frame {best_frame_idx} (AbsRel={best_frame_abs_rel:.4f})")

            visualize_best_frame_simplified(
                images[0, best_frame_idx],  # [3, H, W]
                pred_depths[best_frame_idx, 0],  # [H, W]
                gt_depth_metric[0, best_frame_idx, 0],  # [H, W]
                sequence_id,
                best_frame_idx,
                best_frame_abs_rel,
                fps,
                self.save_dir,
                frame_metrics=frame_metrics[best_frame_idx] if best_frame_idx < len(frame_metrics) else None
            )

        return metrics

    def _aggregate_and_save_results(self):
        """Aggregate metrics and save results"""
        if len(self.all_results) == 0:
            logger.warning("No results to aggregate")
            return

        # Compute average metrics
        avg_metrics = {}
        for key in ['mae', 'rmse', 'abs_rel', 'sq_rel', 'rmse_log', 'a1', 'a2', 'a3', 'tae', 'boundary_f1', 'fps']:
            values = [r[key] for r in self.all_results if key in r]
            if len(values) > 0:
                avg_metrics[key] = float(np.mean(values))

        # Save test results
        test_results = {
            'method': self.method_name,
            'dataset': self.config.get('dataset', 'waymo'),
            'num_sequences': len(self.all_results),
            'metrics': avg_metrics
        }

        results_path = self.save_dir / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2)

        logger.info(f"\nTest Results Summary:")
        logger.info(f"  Method: {self.method_name}")
        logger.info(f"  Dataset: {test_results['dataset']}")
        logger.info(f"  Sequences: {test_results['num_sequences']}")
        logger.info(f"  MAE: {avg_metrics['mae']:.4f}")
        logger.info(f"  RMSE: {avg_metrics['rmse']:.4f}")
        logger.info(f"  AbsRel: {avg_metrics['abs_rel']:.4f}")
        logger.info(f"  δ1: {avg_metrics['a1']:.4f}")
        logger.info(f"  TAE: {avg_metrics['tae']:.4f}")
        logger.info(f"  F1: {avg_metrics.get('boundary_f1', 0):.3f}")
        logger.info(f"  FPS: {avg_metrics['fps']:.2f}")
        logger.info(f"Results saved to {results_path}")

        # Save per-sequence results
        per_seq_path = self.save_dir / 'per_sequence_results.json'
        with open(per_seq_path, 'w') as f:
            json.dump(self.all_results, f, indent=2)

        logger.info(f"Per-sequence results saved to {per_seq_path}")


def main():
    parser = argparse.ArgumentParser(description='Test comparison depth estimation methods')
    parser.add_argument('--method', type=str, required=True,
                       help='Method name: vda, depthcrafter, metric3d, unidepth, zoedepth, depthpro, cut3r')
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

    args = parser.parse_args()

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
        }
    }

    # Import and create adapter
    try:
        if args.method == 'vda':
            from adapters.video_depth_anything_adapter import VideoDepthAnythingAdapter
            adapter = VideoDepthAnythingAdapter()
        elif args.method == 'depthcrafter':
            from adapters.depthcrafter_adapter import DepthCrafterAdapter
            adapter = DepthCrafterAdapter()
        elif args.method == 'metric3d':
            from adapters.metric3d_adapter import Metric3DAdapter
            adapter = Metric3DAdapter(version=args.version or 'v2')
        elif args.method == 'unidepth':
            from adapters.unidepth_adapter import UniDepthAdapter
            adapter = UniDepthAdapter(version=args.version or 'v2')
        elif args.method == 'zoedepth':
            from adapters.zoedepth_adapter import ZoeDepthAdapter
            adapter = ZoeDepthAdapter()
        elif args.method == 'depthpro':
            from adapters.depthpro_adapter import DepthProAdapter
            adapter = DepthProAdapter()
        elif args.method == 'cut3r':
            from adapters.cut3r_adapter import CUT3RAdapter
            adapter = CUT3RAdapter()
        else:
            raise ValueError(f"Unknown method: {args.method}")
    except ImportError as e:
        logger.error(f"Failed to import adapter for {args.method}: {e}")
        logger.error("Make sure the adapter is implemented in adapters/ directory")
        sys.exit(1)

    # Create tester and run
    tester = ComparisonTester(method_name, config, adapter)
    tester.test()

    logger.info("Testing completed successfully!")


if __name__ == '__main__':
    main()
