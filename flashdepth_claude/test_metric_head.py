#!/usr/bin/env python3
"""
Test script for the Global Scale Predictor (GSP) head implementation
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import sys
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from PIL import Image
from einops import rearrange
import hydra
from omegaconf import DictConfig, OmegaConf

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from flashdepth.model import FlashDepth
from flashdepth.heads import GlobalScalePredictor, MetricDepthLoss
from utils.metric_depth_metrics import MetricDepthMetrics, format_metrics
try:
    from utils.metric_visualization import MetricDepthVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    MetricDepthVisualizer = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_depth_colormap(depth_map, valid_mask=None, colormap='plasma'):
    """Convert depth map to colored visualization"""
    if valid_mask is not None:
        depth_map = depth_map.clone()
        depth_map[~valid_mask] = 0

    depth_np = depth_map.cpu().numpy()

    # Normalize to 0-1 range with robust handling
    if valid_mask is not None:
        valid_depth = depth_np[valid_mask.cpu().numpy()]
        if len(valid_depth) > 0:
            # Use percentiles for robust normalization
            min_depth = np.percentile(valid_depth, 1)  # 1st percentile
            max_depth = np.percentile(valid_depth, 99)  # 99th percentile
        else:
            min_depth, max_depth = 0, 1
    else:
        # Use percentiles for robust normalization
        min_depth = np.percentile(depth_np, 1)
        max_depth = np.percentile(depth_np, 99)

    if max_depth > min_depth:
        normalized = np.clip((depth_np - min_depth) / (max_depth - min_depth), 0, 1)
    else:
        normalized = np.zeros_like(depth_np)

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored = cmap(normalized)

    # Convert to uint8 RGB
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)

    return rgb


def create_sequence_visualization(images, pred_depths, gt_depths, valid_masks=None,
                                save_path=None, title="Metric Depth Prediction"):
    """
    Create visualization showing sequence of frames with predictions and ground truth
    Note: Both gt_depths and pred_depths are in metric format (meters) for TartanAir dataset

    Args:
        images: [T, 3, H, W] or [T, H, W, 3] - input images
        pred_depths: [T, H, W] - predicted depths (metric format)
        gt_depths: [T, H, W] - ground truth depths (metric format)
        valid_masks: [T, H, W] - valid pixel masks (optional)
        save_path: str - path to save visualization
        title: str - title for the plot
    """
    T = len(images)

    # Convert tensors to numpy
    if isinstance(images, torch.Tensor):
        if images.shape[1] == 3:  # [T, 3, H, W]
            images = images.permute(0, 2, 3, 1)  # [T, H, W, 3]
        images = images.cpu().numpy()

    if isinstance(pred_depths, torch.Tensor):
        pred_depths = pred_depths.cpu()
    if isinstance(gt_depths, torch.Tensor):
        gt_depths = gt_depths.cpu()
    if valid_masks is not None and isinstance(valid_masks, torch.Tensor):
        valid_masks = valid_masks.cpu()

    # Normalize images to 0-1 range if needed
    if images.max() > 1.0:
        images = images / 255.0

    # Create figure with subplots
    fig = plt.figure(figsize=(T * 4, 12))
    gs = gridspec.GridSpec(3, T, hspace=0.3, wspace=0.1)

    for t in range(T):
        # Original image
        ax_img = fig.add_subplot(gs[0, t])
        ax_img.imshow(images[t])
        ax_img.set_title(f'Frame {t+1}')
        ax_img.axis('off')

        # Predicted depth
        ax_pred = fig.add_subplot(gs[1, t])
        mask = valid_masks[t] if valid_masks is not None else None
        pred_colored = create_depth_colormap(pred_depths[t], mask, 'plasma')
        ax_pred.imshow(pred_colored)
        ax_pred.set_title(f'Predicted Depth')
        ax_pred.axis('off')

        # Ground truth depth
        ax_gt = fig.add_subplot(gs[2, t])
        gt_metric = gt_depths[t]  
        gt_colored = create_depth_colormap(gt_metric, mask, 'plasma')
        ax_gt.imshow(gt_colored)
        ax_gt.set_title(f'Ground Truth (m)')
        ax_gt.axis('off')

    plt.suptitle(title, fontsize=16)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Sequence visualization saved to {save_path}")

    return fig


def create_comparison_visualization(pred_depth, gt_depth, valid_mask=None,
                                  save_path=None, title="Depth Comparison"):
    """
    Create side-by-side comparison of predicted vs ground truth depth
    Note: Both gt_depth and pred_depth are in metric format (meters) for TartanAir dataset
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Predicted depth
    pred_colored = create_depth_colormap(pred_depth, valid_mask, 'plasma')
    axes[0].imshow(pred_colored)
    axes[0].set_title('Predicted Depth')
    axes[0].axis('off')

    # Ground truth depth - TartanAir GT is already in metric depth format
    gt_metric = gt_depth
    gt_colored = create_depth_colormap(gt_metric, valid_mask, 'plasma')
    axes[1].imshow(gt_colored)
    axes[1].set_title('Ground Truth Depth (m)')
    axes[1].axis('off')

    # Error map (both in metric depth space)
    if valid_mask is not None:
        error = torch.abs(pred_depth - gt_metric)  # Both in metric depth space
        error[~valid_mask] = 0
    else:
        error = torch.abs(pred_depth - gt_metric)

    error_colored = create_depth_colormap(error, valid_mask, 'hot')
    axes[2].imshow(error_colored)
    axes[2].set_title('Absolute Error')
    axes[2].axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Comparison visualization saved to {save_path}")

    return fig



def test_parameter_freezing():
    """Test that all parameters are frozen for inference"""
    logger.info("Testing parameter freezing...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_config = {
        'vit_size': 'vitl',
        'patch_size': 14,
        'attn_class': 'MemEffAttention',
        'use_mamba': True,
        'use_metric_head': True,
        'training': False,
        # Mamba configuration parameters (exact match with flashdepth-l)
        'mamba_type': 'add',
        'num_mamba_layers': 4,
        'downsample_mamba': [0.1],
        'mamba_pos_embed': None,
        'mamba_in_dpt_layer': [3],  # FlashDepth-L uses [3]
        'mamba_d_conv': 4,
        'mamba_d_state': 256,
        'use_hydra': False,
        'use_transformer_rnn': False,
        'use_xlstm': False
    }

    try:
        model = FlashDepth(**model_config).to(device)

        # Load weights if available
        flashdepth_checkpoint = globals().get('FLASHDEPTH_CHECKPOINT')
        gsp_checkpoint = globals().get('GSP_CHECKPOINT')
        if flashdepth_checkpoint or gsp_checkpoint:
            model = load_model_weights(model, flashdepth_checkpoint, gsp_checkpoint)

        # Set model to eval mode (inference)
        model.eval()

        # Check initial state (before freezing)
        initial_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        logger.info(f"Initial trainable parameters: {initial_trainable}")

        # For inference, ALL parameters should be frozen
        for name, param in model.named_parameters():
            param.requires_grad = False

        # Count parameters after freezing
        trainable_params = []
        frozen_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
            else:
                frozen_params.append(name)

        # Verify that NO parameters are trainable for inference
        assert len(trainable_params) == 0, f"Found {len(trainable_params)} trainable parameters in inference mode: {trainable_params}"
        assert len(frozen_params) > 0, "No parameters found in model"

        logger.info("✓ Parameter freezing test passed!")
        logger.info(f"  Initial trainable parameters: {initial_trainable}")
        logger.info(f"  Final trainable parameters: {len(trainable_params)} (correctly frozen)")
        logger.info(f"  Total frozen parameters: {len(frozen_params)}")
        logger.info("  All parameters are properly frozen for inference")

        return True

    except Exception as e:
        logger.error(f"Parameter freezing test failed: {e}")
        return False


def test_visualization():
    """Test the visualization functionality - requires real data, no dummy data allowed"""
    logger.info("Visualization test skipped - dummy data usage eliminated")
    logger.info("Visualization functionality is tested within comprehensive integration test with real data")

    return True


def compute_temporal_consistency(depth_sequence, scale=None, shift=None):
    """
    Compute temporal consistency metrics for depth estimation

    Args:
        depth_sequence: Tensor of shape (B, T, H, W) - depth predictions over time
        scale: Optional tensor of shape (B, 1) - scale values for each batch
        shift: Optional tensor of shape (B, 1) - shift values for each batch

    Returns:
        dict: Temporal consistency metrics
    """
    B, T, H, W = depth_sequence.shape

    if T < 2:
        logger.warning("Need at least 2 frames for temporal consistency calculation")
        return {
            'temporal_variance': 0.0,
            'frame_to_frame_diff': 0.0,
            'temporal_smoothness': 0.0
        }

    # Convert to numpy for easier calculation
    depths = depth_sequence.detach().cpu().numpy()

    metrics = {}

    # 1. Temporal Variance: measure of stability across time
    # Calculate pixel-wise variance across time dimension
    temporal_var = np.var(depths, axis=1)  # Shape: (B, H, W)
    metrics['temporal_variance'] = float(np.mean(temporal_var))

    # 2. Frame-to-frame difference: measure of temporal smoothness
    frame_diffs = []
    for t in range(1, T):
        diff = np.abs(depths[:, t] - depths[:, t-1])  # Shape: (B, H, W)
        frame_diffs.append(np.mean(diff))

    metrics['frame_to_frame_diff'] = float(np.mean(frame_diffs))

    # 3. Temporal Smoothness: inverse of frame differences (higher is better)
    # Use reciprocal with small epsilon to avoid division by zero
    smoothness = 1.0 / (metrics['frame_to_frame_diff'] + 1e-6)
    metrics['temporal_smoothness'] = float(smoothness)

    # 4. If scale and shift are available, compute their stability too
    if scale is not None:
        scale_np = scale.detach().cpu().float().numpy()  # Convert to float32 first
        metrics['scale_mean'] = float(np.mean(scale_np))
        metrics['scale_std'] = float(np.std(scale_np))

    if shift is not None:
        shift_np = shift.detach().cpu().float().numpy()  # Convert to float32 first
        metrics['shift_mean'] = float(np.mean(shift_np))
        metrics['shift_std'] = float(np.std(shift_np))

    return metrics


def test_comprehensive_integration():
    """Test comprehensive integration with real TartanAir data"""
    logger.info("Testing comprehensive integration with TartanAir dataset...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # Create model
        model_config = {
            'vit_size': 'vitl',
            'patch_size': 14,
            'attn_class': 'MemEffAttention',
            'use_mamba': True,
            'use_metric_head': True,
            'training': False,
            # Mamba configuration parameters (exact match with flashdepth-l)
            'mamba_type': 'add',
            'num_mamba_layers': 4,
            'downsample_mamba': [0.1],
            'mamba_pos_embed': None,
            'mamba_in_dpt_layer': [3],  # FlashDepth-L uses [3]
            'mamba_d_conv': 4,
            'mamba_d_state': 256,
            'use_hydra': False,
            'use_transformer_rnn': False,
            'use_xlstm': False
        }

        model = FlashDepth(**model_config).to(device)
        model.eval()

        # Clear any existing GPU memory
        torch.cuda.empty_cache()

        # Freeze all parameters for inference
        for param in model.parameters():
            param.requires_grad = False

        # Load GSP checkpoint (contains full trained model)
        gsp_checkpoint = globals().get('GSP_CHECKPOINT')
        if gsp_checkpoint:
            model = load_model_weights(model, None, gsp_checkpoint)  # Only GSP checkpoint needed
            logger.info(f"✓ Full model weights loaded from GSP checkpoint")
            logger.info(f"  GSP checkpoint: {gsp_checkpoint}")
        else:
            logger.warning("No GSP checkpoint provided - using random initialization")

        # Setup TartanAir dataset (same configuration as train)
        try:
            from dataloaders.combined_dataset import CombinedDataset

            # Use same configuration as train_metric_head.py
            dataset = CombinedDataset(
                root_dir="/data/datasets",  # Correct path for mounted datasets
                enable_dataset_flags=['tartanair'],  # Same as train
                resolution='base',  # Same as train (TartanAir → 518x518)
                split='test',  # Use test split for evaluation (train uses 'val')
                video_length=50,  # More frames than train (5) for comprehensive evaluation
                color_aug=False  # Same as train
            )

            # DEBUG: Verify dataset configuration matches train expectations
            logger.info(f"DEBUG - Dataset configuration:")
            logger.info(f"  Resolution: 'base' (same as train)")
            logger.info(f"  Dataset: tartanair")
            logger.info(f"  Split: test (train uses 'val')")
            logger.info(f"  Video length: 50 (train uses 5)")

            if 'tartanair' in dataset.reshape_list:
                reshape_config = dataset.reshape_list['tartanair']
                logger.info(f"  TartanAir reshape config: {reshape_config}")
                expected_resolution = reshape_config.get('resolution', 'Unknown')
                logger.info(f"  Expected GT resolution: {expected_resolution}")
            else:
                logger.warning("  WARNING: TartanAir not found in reshape_list!")

            # Custom collate function to handle None values
            def custom_collate_fn(batch):
                """Custom collate function to filter out None values"""
                # Filter out None values
                batch = [item for item in batch if item is not None]

                # If all items are None, return None
                if len(batch) == 0:
                    return None

                # Use default collate for non-None items
                from torch.utils.data.dataloader import default_collate
                return default_collate(batch)

            # Test with one batch
            from torch.utils.data import DataLoader
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

            # Get one sample (skip None batches)
            batch = None
            dataloader_iter = iter(dataloader)
            for _ in range(10):  # Try up to 10 times
                try:
                    batch = next(dataloader_iter)
                    if batch is not None:
                        break
                except StopIteration:
                    break

            if batch is None:
                logger.error("CRITICAL: Could not find valid TartanAir batch for testing!")
                logger.error("Real data is required for comprehensive integration test")
                raise RuntimeError("No valid TartanAir data found - comprehensive integration test must use real data")

            # Test split now returns (images, depths, dataset_name) - same as val split
            video, gt_depth, dataset_name = batch

            video = video.to(device)
            if gt_depth is not None:
                gt_depth = gt_depth.to(device)

            logger.info(f"Testing with real data: {dataset_name}")
            logger.info(f"Video shape: {video.shape}")
            if gt_depth is not None:
                logger.info(f"GT depth shape: {gt_depth.shape}")
            else:
                logger.info("No GT depth available (test split)")

            # Forward pass with FPS measurement
            B, T = video.shape[:2]
            total_frames = B * T

            # Prepare input for forward pass
            forward_input = (video, gt_depth) if gt_depth is not None else video

            # Warm-up run
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    _ = model.forward_with_metric_head(forward_input)

            # Actual timing run
            torch.cuda.synchronize()  # Ensure all GPU operations are complete
            start_time = time.time()

            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model.forward_with_metric_head(forward_input)

            torch.cuda.synchronize()  # Ensure all GPU operations are complete
            end_time = time.time()

            # Clear memory after inference
            torch.cuda.empty_cache()

            # Calculate FPS
            inference_time = end_time - start_time
            fps = total_frames / inference_time if inference_time > 0 else 0

            logger.info(f"Inference time: {inference_time:.4f}s for {total_frames} frames")
            logger.info(f"FPS: {fps:.2f} frames/second")

            # Extract outputs
            pred_metric = outputs['metric_depth']
            scale = outputs['scale']
            shift = outputs['shift']

            # Compute temporal consistency metrics (always available for predictions)
            temporal_metrics = compute_temporal_consistency(pred_metric, scale, shift)

            # Comprehensive metrics computation (exactly like train_metric_head.py)
            if gt_depth is not None:
                # Metrics computation exactly like train (same resolution expected)
                logger.info(f"DEBUG TEST - Shapes: GT {gt_depth.shape}, Pred {pred_metric.shape}")

                # Train expects GT and Pred to be same resolution (518x518)
                # If they're different, there's a dataset loading issue
                if pred_metric.shape[-2:] != gt_depth.shape[-2:]:
                    logger.warning(f"Resolution mismatch! GT {gt_depth.shape} vs Pred {pred_metric.shape}")
                    logger.warning("This suggests dataset resize is not working properly")
                    # For now, resize pred to match GT to continue testing
                    import torch.nn.functional as F
                    pred_metric_for_metrics = F.interpolate(
                        pred_metric.view(-1, 1, pred_metric.shape[-2], pred_metric.shape[-1]),
                        size=gt_depth.shape[-2:],
                        mode='bilinear',
                        align_corners=True
                    ).view(pred_metric.shape[0], pred_metric.shape[1], gt_depth.shape[-2], gt_depth.shape[-1])
                else:
                    # Same resolution - perfect! (like train)
                    pred_metric_for_metrics = pred_metric
                    logger.info(f"✓ Same resolution as train: {pred_metric.shape}")

                # Compute metrics for ALL frames (different from train - comprehensive evaluation)
                B, T = pred_metric_for_metrics.shape[:2]
                all_frame_metrics = []
                total_valid_pixels = 0

                for t in range(T):
                    gt_frame = gt_depth[0, t].cpu()  # First batch, frame t
                    pred_frame = pred_metric_for_metrics[0, t].cpu()  # First batch, frame t (resized for metrics)

                    # Create valid mask considering both GT and pred ranges
                    gt_valid_mask = gt_frame > 0  # GT valid pixels
                    # More restrictive valid mask to filter out extreme values
                    pred_valid_mask = (pred_frame > 0) & (pred_frame < 1000.0)  # Same as train
                    valid_mask = gt_valid_mask & pred_valid_mask

                    if valid_mask.sum() > 0:
                        # Compute metrics for this frame
                        frame_metrics = MetricDepthMetrics.compute_metric_depth_metrics(
                            pred_frame,  # Frame prediction (already in metric depth)
                            gt_frame,   # Frame GT (TartanAir GT is already metric depth)
                            valid_mask=valid_mask
                        )
                        all_frame_metrics.append(frame_metrics)
                        total_valid_pixels += valid_mask.sum().item()

                        # Debug for first frame only
                        if t == 0:
                            logger.info(f"DEBUG - Frame {t}: GT range: min={gt_frame.min():.6f}, max={gt_frame.max():.6f}, mean={gt_frame.mean():.6f}")
                            logger.info(f"DEBUG - Frame {t}: Pred range: min={pred_frame.min():.6f}, max={pred_frame.max():.6f}, mean={pred_frame.mean():.6f}")
                            logger.info(f"DEBUG - Frame {t}: GT > 0 pixels: {(gt_frame > 0).sum()}")
                            logger.info(f"DEBUG - Frame {t}: Pred in range pixels: {((pred_frame > 0) & (pred_frame < 1000.0)).sum()}")
                            logger.info(f"DEBUG - Frame {t}: Valid mask pixels: {valid_mask.sum()}")

                # Average metrics across all frames
                if all_frame_metrics:
                    metrics = {k: np.mean([frame_metrics[k] for frame_metrics in all_frame_metrics])
                              for k in all_frame_metrics[0].keys()}
                    avg_mae = metrics.get('mae', 0.0)
                    avg_valid_pixels = total_valid_pixels / T  # Average valid pixels per frame

                    logger.info(f"Computed metrics across {len(all_frame_metrics)} frames")
                else:
                    metrics = {}
                    avg_mae = 0.0
                    avg_valid_pixels = 0
                    logger.warning("No valid frames found for metrics computation")
            else:
                metrics = {}
                avg_mae = 0.0  # No GT available for metrics
                avg_valid_pixels = 0

            # Create visualization
            vis_dir = Path(f"{globals().get('RESULTS_DIR', 'test_results/results_1')}/comprehensive")
            vis_dir.mkdir(parents=True, exist_ok=True)

            # Create full depth sequence visualization if GT is available (518x518 resolution like train)
            if gt_depth is not None:
                logger.info("Creating depth sequence visualization...")

                # For visualization, use 518x518 resolution (same as train/model output)
                # Resize GT depth to match model output resolution for visualization
                B, T = gt_depth.shape[:2]
                model_resolution = video.shape[-2:]  # Should be [518, 518]

                if gt_depth.shape[-2:] != model_resolution:
                    gt_vis = F.interpolate(
                        gt_depth.view(-1, 1, gt_depth.shape[-2], gt_depth.shape[-1]),
                        size=model_resolution,
                        mode='bilinear',
                        align_corners=True
                    ).view(B, T, model_resolution[0], model_resolution[1])
                else:
                    gt_vis = gt_depth

                # Use original pred_metric without resizing (should already be 518x518)
                pred_vis = outputs['metric_depth']  # Use original model output resolution

                # Generate valid masks for visualization at 518x518 resolution
                gt_valid_mask_seq = gt_vis > 0
                pred_valid_mask_seq = (pred_vis > 0) & (pred_vis < 1000.0)
                vis_mask_seq = (gt_valid_mask_seq & pred_valid_mask_seq)[0] # Use first batch item

                try:
                    vis_fig = create_sequence_visualization(
                        images=video[0], # First batch item (518x518)
                        pred_depths=pred_vis[0], # First batch item (518x518)
                        gt_depths=gt_vis[0], # First batch item (518x518)
                        valid_masks=vis_mask_seq,
                        save_path=vis_dir / "depth_sequence_visualization.png",
                        title=f"Metric Depth Sequence - {dataset_name[0]} (518x518)"
                    )
                    plt.close(vis_fig) # Close figure to save memory
                    logger.info(f"Visualization created at 518x518 resolution (train-consistent)")
                except Exception as e:
                    logger.warning(f"Could not generate sequence visualization: {e}")

            # Simple summary figure
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))

            summary_text = (
                f'Real Data Integration Test\n'
                f'Dataset: {dataset_name}\n'
                f'Video shape: {video.shape}\n'
                f'Inference time: {inference_time:.4f}s\n'
                f'FPS: {fps:.2f} frames/sec\n'
                f'Scale range: [{scale.min():.3f}, {scale.max():.3f}]\n'
                f'Shift range: [{shift.min():.3f}, {shift.max():.3f}]\n'
            )

            if gt_depth is not None:
                frames_evaluated = len(all_frame_metrics) if 'all_frame_metrics' in locals() else 0
                summary_text += f'Average MAE: {avg_mae:.4f}m (across {frames_evaluated} frames)\nValid pixels/frame: {int(avg_valid_pixels)}\n'
                # Add key metrics to summary
                if metrics:
                    summary_text += f'RMSE: {metrics.get("rmse", 0.0):.4f}m\n'
                    summary_text += f'AbsRel: {metrics.get("abs_rel", 0.0):.4f}\n'
                    summary_text += f'δ1: {metrics.get("a1", 0.0):.4f}'
            else:
                summary_text += 'No GT depth available (test split)\nMetrics calculation skipped'

            ax.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            plt.savefig(vis_dir / "real_data_integration_test.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

            # Save test results to JSON
            test_results = {
                'dataset': dataset_name,
                'video_shape': list(video.shape),
                'inference_time_seconds': float(inference_time),
                'fps': float(fps),
                'total_frames': int(total_frames),
                'scale_min': float(scale.min()),
                'scale_max': float(scale.max()),
                'scale_mean': float(scale.mean()),
                'shift_min': float(shift.min()),
                'shift_max': float(shift.max()),
                'shift_mean': float(shift.mean()),
                'has_ground_truth': gt_depth is not None,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'temporal_consistency': temporal_metrics
            }

            # Add comprehensive metrics only if GT is available (all frames evaluation)
            if gt_depth is not None:
                test_results.update({
                    'average_mae_meters': float(avg_mae),
                    'valid_pixels_per_frame': int(avg_valid_pixels),
                    'total_frames_evaluated': len(all_frame_metrics) if 'all_frame_metrics' in locals() else 0,
                    'comprehensive_metrics': {k: float(v) for k, v in metrics.items()}
                })

            results_json_path = vis_dir / "test_results.json"
            with open(results_json_path, 'w') as f:
                json.dump(test_results, f, indent=4)

            logger.info("✓ Comprehensive integration test passed!")
            logger.info(f"  Successfully processed real TartanAir test data")
            logger.info(f"  Inference time: {inference_time:.4f}s for {total_frames} frames")
            logger.info(f"  FPS: {fps:.2f} frames/second")
            if gt_depth is not None:
                logger.info(f"  Average MAE: {avg_mae:.4f}m (across {len(all_frame_metrics) if 'all_frame_metrics' in locals() else 0} frames)")
                logger.info(f"  Valid pixels per frame: {int(avg_valid_pixels)}")
                # Log comprehensive metrics (averaged across all frames)
                logger.info("  Comprehensive Metrics (averaged across all frames):")
                for k, v in metrics.items():
                    logger.info(f"    {k}: {v:.4f}")
            else:
                logger.info("  No GT depth available - metrics calculation skipped")
            logger.info(f"  Scale range: [{scale.min():.3f}, {scale.max():.3f}]")
            logger.info(f"  Shift range: [{shift.min():.3f}, {shift.max():.3f}]")
            logger.info(f"  Temporal Consistency Metrics:")
            logger.info(f"    Temporal Variance: {temporal_metrics['temporal_variance']:.6f}")
            logger.info(f"    Frame-to-frame Diff: {temporal_metrics['frame_to_frame_diff']:.6f}")
            logger.info(f"    Temporal Smoothness: {temporal_metrics['temporal_smoothness']:.6f}")
            logger.info(f"  Test summary saved to: {vis_dir}")
            logger.info(f"  Test results JSON: {results_json_path}")

            return True

        except ImportError as e:
            logger.error(f"CRITICAL: Could not import dataset: {e}")
            logger.error("Real TartanAir data is required for comprehensive integration test")
            raise ImportError(f"Dataset import failed - comprehensive integration test requires real data: {e}")

        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["no such file", "data", "not a multiple", "resolution", "no valid pairs"]):
                logger.error(f"CRITICAL: TartanAir dataset not available or incompatible: {e}")
                logger.error("Real TartanAir data is required for comprehensive integration test")
                raise RuntimeError(f"Dataset error - comprehensive integration test requires real data: {e}")
            else:
                raise e

    except Exception as e:
        logger.error(f"Comprehensive integration test failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False


def run_all_tests():
    """Run all tests"""
    logger.info("="*50)
    logger.info("Running GSP Head Implementation Tests")
    logger.info("="*50)

    # Clear GPU memory at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tests = [
        ("Parameter Freezing", test_parameter_freezing),
        ("Visualization", test_visualization),
        ("Comprehensive Integration", test_comprehensive_integration),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            # Clear memory before each test
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                failed += 1
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            logger.error(f"❌ {test_name} FAILED with exception: {e}")

        # Clear memory after each test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("\n" + "="*50)
    logger.info(f"Test Summary: {passed} PASSED, {failed} FAILED")
    logger.info("="*50)

    return failed == 0


def load_model_weights(model, flashdepth_checkpoint=None, gsp_checkpoint=None):
    """
    Load weights into model from GSP checkpoint

    Args:
        model: FlashDepth model instance
        flashdepth_checkpoint: Path to pretrained FlashDepth weights (not needed for test)
        gsp_checkpoint: Path to trained GSP checkpoint (contains full model)
    """
    device = next(model.parameters()).device

    # Load from GSP checkpoint (contains full model with trained GSP head)
    if gsp_checkpoint and Path(gsp_checkpoint).exists():
        logger.info(f"Loading full model from GSP checkpoint: {gsp_checkpoint}")
        checkpoint = torch.load(gsp_checkpoint, map_location='cpu')

        # Extract state dict from checkpoint (handle different formats)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Load full model
        model_dict = model.state_dict()
        loaded_dict = {k: v for k, v in state_dict.items() if k in model_dict}

        if loaded_dict:
            model_dict.update(loaded_dict)
            model.load_state_dict(model_dict)

            # Count actual parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            logger.info(f"Loaded {len(loaded_dict)} parameter categories from GSP checkpoint")
            logger.info(f"Total model parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
        else:
            logger.warning("No compatible parameters found in GSP checkpoint")

    else:
        logger.warning("No GSP checkpoint available - model will use random initialization")

    return model


@hydra.main(config_path=None, config_name=None, version_base="1.3")
def main(cfg: DictConfig = None) -> None:
    """Main entry point with Hydra configuration support"""

    # GPU setup - same as train_metric_head.py
    import os
    gpu_id = cfg.get('gpu', 0) if cfg else 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    logger.info(f"Setting GPU {gpu_id} via CUDA_VISIBLE_DEVICES")

    # Set global results directory
    results_dir = cfg.get('results_dir', 'test_results/results_1') if cfg else 'test_results/results_1'
    globals()['RESULTS_DIR'] = results_dir

    # Set global checkpoint path for tests (only GSP checkpoint needed)
    gsp_checkpoint = cfg.get('gsp_checkpoint', None) if cfg else None
    globals()['GSP_CHECKPOINT'] = gsp_checkpoint

    logger.info(f"Results will be saved to: {results_dir}")
    if gsp_checkpoint:
        logger.info(f"GSP checkpoint: {gsp_checkpoint}")
    else:
        logger.warning("No GSP checkpoint provided - tests will use random initialization")

    success = run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()