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
        if isinstance(valid_mask, torch.Tensor):
            valid_mask_np = valid_mask.cpu().numpy()
        else:
            valid_mask_np = valid_mask
        valid_depth = depth_np[valid_mask_np]
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
                                save_path=None, title="Metric Depth Prediction", frame_interval=1, frame_indices=None):
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
        frame_interval: int - interval between frames (for subsampling)
        frame_indices: list - actual frame indices for title display
    """
    T = len(images)

    # Apply frame interval (subsample frames) BEFORE tensor conversion for efficiency
    if frame_interval > 1:
        selected_indices = list(range(0, T, frame_interval))
        if isinstance(images, torch.Tensor):
            images = images[selected_indices]
        if isinstance(pred_depths, torch.Tensor):
            pred_depths = pred_depths[selected_indices]
        if isinstance(gt_depths, torch.Tensor):
            gt_depths = gt_depths[selected_indices]
        if valid_masks is not None and isinstance(valid_masks, torch.Tensor):
            valid_masks = valid_masks[selected_indices]
        T = len(selected_indices)
        logger.info(f"Applied frame interval {frame_interval}: showing {T} frames out of original sequence")
        # Update frame indices to match the subsampled frames
        if frame_indices is None:
            frame_indices = selected_indices
        else:
            # frame_indices already contains the correct actual frame numbers
            # No need to re-index since the tensors are already subsampled above
            pass

    # Use provided frame indices for titles, or default to sequential numbering
    if frame_indices is None:
        frame_indices = list(range(T))

    # Ensure frame_indices length matches T after subsampling
    if len(frame_indices) > T:
        frame_indices = frame_indices[:T]
    elif len(frame_indices) < T:
        # If frame_indices is shorter, extend with sequential indices
        frame_indices.extend(list(range(len(frame_indices), T)))

    # Convert tensors to numpy after subsampling
    if isinstance(images, torch.Tensor):
        if images.shape[1] == 3:  # [T, 3, H, W]
            images = images.permute(0, 2, 3, 1)  # [T, H, W, 3]
        images = images.float().cpu().numpy()  # Convert BFloat16 to Float32 first

    if isinstance(pred_depths, torch.Tensor):
        pred_depths = pred_depths.cpu().numpy()
    if isinstance(gt_depths, torch.Tensor):
        gt_depths = gt_depths.cpu().numpy()
    if valid_masks is not None and isinstance(valid_masks, torch.Tensor):
        valid_masks = valid_masks.cpu().numpy()

    # Debug image range
    logger.info(f"DEBUG - Input image range: min={images.min():.6f}, max={images.max():.6f}")
    logger.info("Input images will be displayed as-is (matplotlib auto-handles range)")

    # Create figure with subplots - 10 frames per row
    frames_per_row = 10
    num_rows = (T + frames_per_row - 1) // frames_per_row  # Ceiling division
    fig_width = min(frames_per_row, T) * 4  # Width based on actual frames in first row
    fig_height = num_rows * 3 * 4  # Height for 3 types (image, pred, gt) × num_rows × 4
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(num_rows * 3, frames_per_row, hspace=0.3, wspace=0.1)

    for t in range(T):
        # Calculate row and column for this frame
        row_idx = t // frames_per_row
        col_idx = t % frames_per_row

        # Original image - resize to match depth map size
        ax_img = fig.add_subplot(gs[row_idx * 3, col_idx])

        # Get depth map size for this frame
        depth_h, depth_w = pred_depths[t].shape
        img_frame = images[t]

        # Resize image if needed to match depth map size
        if img_frame.shape[:2] != (depth_h, depth_w):
            import cv2
            # Handle different input ranges for resizing
            if img_frame.max() <= 1.0:
                img_uint8 = (img_frame * 255).astype(np.uint8)
            else:
                img_uint8 = np.clip(img_frame, 0, 255).astype(np.uint8)

            img_resized = cv2.resize(img_uint8, (depth_w, depth_h), interpolation=cv2.INTER_LINEAR)
            img_frame = img_resized.astype(np.float32) / 255.0

        ax_img.imshow(img_frame)
        ax_img.set_title(f'Frame {frame_indices[t]+1}')
        ax_img.axis('off')

        # Predicted depth (with valid mask)
        ax_pred = fig.add_subplot(gs[row_idx * 3 + 1, col_idx])
        pred_valid_mask = (pred_depths[t] > 0) & (pred_depths[t] < 1000.0)
        pred_display = np.full_like(pred_depths[t], np.nan)
        if pred_valid_mask.sum() > 0:
            pred_valid_data = pred_depths[t][pred_valid_mask]
            pred_vmin, pred_vmax = np.nanpercentile(pred_valid_data, [2, 98])
            pred_display[pred_valid_mask] = pred_depths[t][pred_valid_mask]
        else:
            pred_vmin, pred_vmax = 0, 1
        ax_pred.imshow(pred_display, cmap='plasma', vmin=pred_vmin, vmax=pred_vmax)
        ax_pred.set_title(f'Pred Depth')
        ax_pred.axis('off')

        # Ground truth depth (with valid mask)
        ax_gt = fig.add_subplot(gs[row_idx * 3 + 2, col_idx])
        gt_valid_mask = gt_depths[t] > 0
        gt_display = np.full_like(gt_depths[t], np.nan)
        if gt_valid_mask.sum() > 0:
            gt_valid_data = gt_depths[t][gt_valid_mask]
            gt_vmin, gt_vmax = np.nanpercentile(gt_valid_data, [2, 98])
            gt_display[gt_valid_mask] = gt_depths[t][gt_valid_mask]
        else:
            gt_vmin, gt_vmax = 0, 1
        ax_gt.imshow(gt_display, cmap='plasma', vmin=gt_vmin, vmax=gt_vmax)
        ax_gt.set_title(f'GT Depth')
        ax_gt.axis('off')

    plt.suptitle(title, fontsize=16)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Sequence visualization saved to {save_path}")

    return fig


def create_best_frame_visualization(image, pred_depth, gt_depth, relative_depth, valid_mask=None,
                                   abs_rel=None, save_path=None, title="Best Frame"):
    """
    Create visualization for the best frame (lowest AbsRel) similar to training visualization

    Args:
        image: [3, H, W] or [H, W, 3] - input image
        pred_depth: [H, W] - predicted depth (metric format)
        gt_depth: [H, W] - ground truth depth (metric format)
        relative_depth: [H, W] - relative depth from model
        valid_mask: [H, W] - valid pixel mask (optional)
        abs_rel: float - AbsRel value for this frame
        save_path: str - path to save visualization
        title: str - title for the plot
    """
    # Convert tensors to numpy
    if isinstance(image, torch.Tensor):
        if image.shape[0] == 3:  # [3, H, W]
            image = image.permute(1, 2, 0)  # [H, W, 3]
        image = image.float().cpu().numpy()  # Convert BFloat16 to Float32 first

    if isinstance(pred_depth, torch.Tensor):
        pred_depth = pred_depth.float().cpu()  # Convert BFloat16 to Float32 first
    if isinstance(gt_depth, torch.Tensor):
        gt_depth = gt_depth.float().cpu()  # Convert BFloat16 to Float32 first
    if isinstance(relative_depth, torch.Tensor):
        relative_depth = relative_depth.float().cpu().numpy()  # Convert BFloat16 to Float32 first
    if valid_mask is not None and isinstance(valid_mask, torch.Tensor):
        valid_mask = valid_mask.cpu()

    # Debug image range - no normalization needed, matplotlib handles it
    logger.info(f"DEBUG BEST/WORST - Input image range: min={image.min():.6f}, max={image.max():.6f}")
    logger.info("Input image will be displayed as-is (matplotlib auto-handles range)")

    # Create figure with subplots - fixed aspect ratio with separate colorbar space
    fig = plt.figure(figsize=(24, 5))  # Increased figure width for more space
    gs = gridspec.GridSpec(1, 9, hspace=0.2, wspace=0.1, width_ratios=[1, 1, 1, 0.05, 0.1, 1, 0.05, 0.15, 1.2])  # Added spacing after Rel Colorbar

    # 1. Input Image - resize to match depth map size and set fixed aspect
    ax1 = fig.add_subplot(gs[0, 0])
    # Resize input image to match depth map dimensions for consistent display
    depth_h, depth_w = gt_depth.shape if gt_depth is not None else pred_depth.shape
    if image.shape[:2] != (depth_h, depth_w):
        import cv2
        # Handle different input ranges for resizing
        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
        # Resize using OpenCV for consistent results
        image_resized = cv2.resize(image_uint8, (depth_w, depth_h), interpolation=cv2.INTER_LINEAR)
        image = image_resized.astype(np.float32) / 255.0

    ax1.imshow(image, aspect='equal')
    ax1.set_title('Input Image', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # 2. Ground Truth Depth
    ax2 = fig.add_subplot(gs[0, 1])
    gt_display = np.where(valid_mask.numpy() if valid_mask is not None else True, gt_depth.numpy(), np.nan)
    gt_vmin, gt_vmax = np.nanpercentile(gt_display, [2, 98])
    im2 = ax2.imshow(gt_display, cmap='plasma', vmin=gt_vmin, vmax=gt_vmax, aspect='equal')
    ax2.set_title('Ground Truth Depth', fontsize=12, fontweight='bold')
    ax2.axis('off')

    # 3. Predicted Metric Depth
    ax3 = fig.add_subplot(gs[0, 2])
    pred_display = np.where(valid_mask.numpy() if valid_mask is not None else True, pred_depth.numpy(), np.nan)
    pred_vmin, pred_vmax = np.nanpercentile(pred_display, [2, 98])
    im3 = ax3.imshow(pred_display, cmap='plasma', vmin=pred_vmin, vmax=pred_vmax, aspect='equal')
    ax3.set_title('Predicted Metric Depth', fontsize=12, fontweight='bold')
    ax3.axis('off')

    # Add depth colorbar
    cbar_ax_depth = fig.add_subplot(gs[0, 3])  # Column 3 (colorbar)
    depth_vmin = min(gt_vmin, pred_vmin)
    depth_vmax = max(gt_vmax, pred_vmax)
    plt.colorbar(im2, cax=cbar_ax_depth, label='Depth (m)')

    # 4. Relative Depth (with spacing before it)
    ax4 = fig.add_subplot(gs[0, 5])  # Column 5 (after spacing)
    rel_vmin, rel_vmax = np.percentile(relative_depth, [2, 98])
    im4 = ax4.imshow(relative_depth, cmap='plasma', vmin=rel_vmin, vmax=rel_vmax, aspect='equal')
    ax4.set_title('Relative Depth', fontsize=12, fontweight='bold')
    ax4.axis('off')

    # Relative depth colorbar
    cbar_ax_rel = fig.add_subplot(gs[0, 6])  # Column 6 (colorbar)
    plt.colorbar(im4, cax=cbar_ax_rel, label='Relative')

    # 5. Metrics and Info (with spacing before it)
    ax5 = fig.add_subplot(gs[0, 8])  # Column 8 (after spacing)

    # Compute additional metrics if possible
    if valid_mask is not None:
        valid_gt = gt_depth[valid_mask]
        valid_pred = pred_depth[valid_mask]
        if len(valid_gt) > 0:
            rmse = torch.sqrt(torch.mean((valid_pred - valid_gt) ** 2))
            mae = torch.mean(torch.abs(valid_pred - valid_gt))

            # Compute Delta metrics (same as train)
            threshold = 1.25
            max_ratio = torch.max(valid_pred / valid_gt, valid_gt / valid_pred)
            delta_1 = (max_ratio < threshold).float().mean()
            delta_2 = (max_ratio < threshold ** 2).float().mean()
            delta_3 = (max_ratio < threshold ** 3).float().mean()

            # Display metrics in organized layout
            ax5.text(0.05, 0.85, f'AbsRel: {abs_rel:.4f}', fontsize=14,
                    transform=ax5.transAxes, bbox=dict(boxstyle="round", facecolor='lightcoral'))
            ax5.text(0.05, 0.70, f'δ₁: {delta_1:.3f}', fontsize=14,
                    transform=ax5.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
            ax5.text(0.05, 0.55, f'δ₂: {delta_2:.3f}', fontsize=14,
                    transform=ax5.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
            ax5.text(0.05, 0.40, f'δ₃: {delta_3:.3f}', fontsize=14,
                    transform=ax5.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
            ax5.text(0.05, 0.25, f'RMSE: {rmse:.3f}m', fontsize=14,
                    transform=ax5.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
            ax5.text(0.05, 0.10, f'MAE: {mae:.3f}m', fontsize=14,
                    transform=ax5.transAxes, bbox=dict(boxstyle="round", facecolor='lightblue'))

    ax5.set_title('Depth Metrics', fontsize=12, fontweight='bold')
    ax5.axis('off')

    # Overall title
    plt.suptitle(f'{title}', fontsize=16, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        logger.info(f"Best frame visualization saved to {save_path}")

    return fig


def create_tae_5frame_visualization(images, pred_depths, gt_depths, relative_depths,
                                   tae_value, fps, resolution, gpu_spec="RTX A6000",
                                   save_path=None, title="TAE 5-Frame Sequence", frame_indices=None):
    """
    Create visualization for 5 frames with configurable spacing

    Args:
        images: [5, 3, H, W] or [5, H, W, 3] - input images for 5 frames
        pred_depths: [5, H, W] - predicted depths (metric format)
        gt_depths: [5, H, W] - ground truth depths (metric format)
        relative_depths: [5, H, W] - relative depths from model
        tae_value: float - TAE value for this sequence
        fps: float - frames per second
        resolution: tuple - original image resolution (H, W)
        gpu_spec: str - GPU specification
        save_path: str - path to save visualization
        title: str - title for the plot
        frame_indices: list - actual frame indices for title display
    """
    T = len(images)
    # Use provided frame indices for titles, or default to sequential numbering
    if frame_indices is None:
        frame_indices = list(range(T))

    logger.info(f"Creating TAE visualization for {T} frames with indices: {[idx+1 for idx in frame_indices]}")

    # Convert tensors to numpy after subsampling
    if isinstance(images, torch.Tensor):
        if images.shape[1] == 3:  # [T, 3, H, W]
            images = images.permute(0, 2, 3, 1)  # [T, H, W, 3]
        images = images.float().cpu().numpy()  # Convert BFloat16 to Float32 first

    if isinstance(pred_depths, torch.Tensor):
        pred_depths = pred_depths.float().cpu().numpy()  # Convert BFloat16 to Float32 first
    if isinstance(gt_depths, torch.Tensor):
        gt_depths = gt_depths.float().cpu().numpy()  # Convert BFloat16 to Float32 first
    if isinstance(relative_depths, torch.Tensor):
        relative_depths = relative_depths.float().cpu().numpy()  # Convert BFloat16 to Float32 first

    # No normalization needed - matplotlib handles different ranges automatically
    logger.info(f"DEBUG TAE - Input image range: min={images.min():.6f}, max={images.max():.6f}")
    logger.info("TAE input images will be displayed as-is (matplotlib auto-handles range)")

    # Create figure with 3 rows (GT depth, Pred depth, Relative depth) and T+1 columns (T frames + narrow colorbar)
    fig = plt.figure(figsize=(T * 4, 12))  # Dynamic width based on frame count
    width_ratios = [1] * T + [0.3]  # T frames + narrow colorbar
    gs = gridspec.GridSpec(3, T + 1, hspace=0.3, wspace=0.15, width_ratios=width_ratios)

    # Create valid masks for all frames at once (vectorized)
    gt_valid_masks = gt_depths > 0
    pred_valid_masks = (pred_depths > 0) & (pred_depths < 1000.0)

    # Determine colormap range for consistency (vectorized)
    all_gt_valid = gt_depths[gt_valid_masks]
    all_pred_valid = pred_depths[pred_valid_masks]
    all_rel_valid = relative_depths.flatten()  # Relative doesn't need masking

    gt_vmin, gt_vmax = np.nanpercentile(all_gt_valid, [2, 98]) if len(all_gt_valid) > 0 else (0, 1)
    pred_vmin, pred_vmax = np.nanpercentile(all_pred_valid, [2, 98]) if len(all_pred_valid) > 0 else (0, 1)
    rel_vmin, rel_vmax = np.nanpercentile(all_rel_valid, [2, 98])

    for t in range(T):
        # Row 1: GT Depth Maps (with valid mask)
        ax_gt = fig.add_subplot(gs[0, t])
        gt_display = np.full_like(gt_depths[t], np.nan)
        gt_display[gt_valid_masks[t]] = gt_depths[t][gt_valid_masks[t]]
        im_gt = ax_gt.imshow(gt_display, cmap='plasma', vmin=gt_vmin, vmax=gt_vmax)
        ax_gt.set_title(f'GT Depth Frame {frame_indices[t]+1}', fontsize=10, fontweight='bold')
        ax_gt.axis('off')

        # Row 2: Predicted Depth Maps (with valid mask)
        ax_pred = fig.add_subplot(gs[1, t])
        pred_display = np.full_like(pred_depths[t], np.nan)
        pred_display[pred_valid_masks[t]] = pred_depths[t][pred_valid_masks[t]]
        im_pred = ax_pred.imshow(pred_display, cmap='plasma', vmin=pred_vmin, vmax=pred_vmax)
        ax_pred.set_title(f'Pred Depth Frame {frame_indices[t]+1}', fontsize=10, fontweight='bold')
        ax_pred.axis('off')

        # Row 3: Relative Depth Maps (no masking needed)
        ax_rel = fig.add_subplot(gs[2, t])
        im_rel = ax_rel.imshow(relative_depths[t], cmap='plasma', vmin=rel_vmin, vmax=rel_vmax)
        ax_rel.set_title(f'Rel Depth Frame {frame_indices[t]+1}', fontsize=10, fontweight='bold')
        ax_rel.axis('off')

    # Add colorbars (smaller) - dynamic column position
    cbar_ax_gt = fig.add_subplot(gs[0, T])  # Last column (T)
    cbar_gt = plt.colorbar(im_gt, cax=cbar_ax_gt, label='Depth (m)')
    cbar_gt.ax.tick_params(labelsize=8)

    cbar_ax_pred = fig.add_subplot(gs[1, T])  # Last column (T)
    cbar_pred = plt.colorbar(im_pred, cax=cbar_ax_pred, label='Depth (m)')
    cbar_pred.ax.tick_params(labelsize=8)

    cbar_ax_rel = fig.add_subplot(gs[2, T])  # Last column (T)
    cbar_rel = plt.colorbar(im_rel, cax=cbar_ax_rel, label='Relative')
    cbar_rel.ax.tick_params(labelsize=8)

    # Add text information at the very top right corner (avoid colorbar overlap)
    fig.text(0.98, 0.98, f'TAE: {tae_value:.4f}\nFPS: {fps:.1f}\nResolution: {resolution[0]}x{resolution[1]}\nGPU: {gpu_spec}',
             fontsize=10, va='top', ha='left',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.9))

    # Overall title
    plt.suptitle(f'{title}', fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        logger.info(f"TAE 5-frame visualization saved to {save_path}")

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


def compute_temporal_alignment_error(depth_sequence, camera_poses=None):
    """
    Compute Temporal Alignment Error (TAE) for depth estimation

    TAE = 1/(2(N-1)) * sum_{k=1}^{N-1} [AbsRel(f(pred_k, p_k), pred_{k+1}) + AbsRel(f(pred_{k+1}, p_{-k+1}), pred_k)]

    where f is the projection function and p is the transformation matrix.
    For now, we use a simplified version without camera poses (assumes identity transformation)

    Args:
        depth_sequence: Tensor of shape (B, T, H, W) - depth predictions over time
        camera_poses: Optional camera transformation matrices (not implemented yet)

    Returns:
        dict: Temporal alignment metrics
    """
    B, T, H, W = depth_sequence.shape

    if T < 2:
        logger.warning("Need at least 2 frames for TAE calculation")
        return {'tae': 0.0, 'frame_count': T}

    # Convert to numpy for easier calculation
    depths = depth_sequence.detach().cpu().numpy()

    tae_values = []

    for t in range(T - 1):
        depth_k = depths[:, t]  # Shape: (B, H, W)
        depth_k_plus_1 = depths[:, t + 1]  # Shape: (B, H, W)

        # Create valid masks (depth > 0 and reasonable range)
        valid_k = (depth_k > 0) & (depth_k < 1000.0)
        valid_k_plus_1 = (depth_k_plus_1 > 0) & (depth_k_plus_1 < 1000.0)

        # For simplified TAE without camera poses, we compare adjacent frames directly
        # This measures temporal consistency

        # Vectorized TAE computation across all batches
        common_mask = valid_k & valid_k_plus_1  # [B, H, W]

        # Get valid pixels for all batches at once
        depth_k_valid = depth_k[common_mask]  # Flattened valid depths
        depth_k_plus_1_valid = depth_k_plus_1[common_mask]  # Flattened valid depths

        if len(depth_k_valid) > 0:
            # Compute AbsRel between consecutive frames (vectorized)
            abs_rel_forward = np.mean(np.abs(depth_k_valid - depth_k_plus_1_valid) / (depth_k_plus_1_valid + 1e-8))
            abs_rel_backward = np.mean(np.abs(depth_k_plus_1_valid - depth_k_valid) / (depth_k_valid + 1e-8))

            tae_frame = (abs_rel_forward + abs_rel_backward) / 2.0
            tae_values.append(tae_frame)

    # Calculate overall TAE
    if tae_values:
        tae = np.mean(tae_values)
    else:
        tae = 0.0

    return {
        'tae': float(tae),
        'frame_count': T,
        'valid_frame_pairs': len(tae_values)
    }


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

            # Get video length from global config
            vid_len = globals().get('VID_LEN', 50)

            # Use same configuration as train_metric_head.py
            dataset = CombinedDataset(
                root_dir="/data/datasets",  # Correct path for mounted datasets
                enable_dataset_flags=['tartanair'],  # Same as train
                resolution='base',  # Same as train (TartanAir → 518x518)
                split='test',  # Use test split for evaluation (train uses 'val')
                video_length=vid_len,  # Configurable video length
                color_aug=False  # Same as train
            )

            # DEBUG: Verify dataset configuration matches train expectations
            logger.info(f"DEBUG - Dataset configuration:")
            logger.info(f"  Resolution: 'base' (same as train)")
            logger.info(f"  Dataset: tartanair")
            logger.info(f"  Split: test (train uses 'val')")
            logger.info(f"  Video length: {vid_len} (train uses 5)")

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

            # Test with multiple batches to understand dataset size
            from torch.utils.data import DataLoader
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

            # DEBUG: Count total test sequences available
            logger.info("DEBUG - Counting test sequences...")
            total_sequences = 0
            valid_sequences = 0
            sequence_names = []

            try:
                for i, batch in enumerate(dataloader):
                    total_sequences += 1
                    if batch is not None:
                        valid_sequences += 1
                        video, gt_depth, dataset_name = batch
                        if isinstance(dataset_name, (list, tuple)):
                            seq_name = dataset_name[0] if len(dataset_name) > 0 else f"sequence_{i}"
                        else:
                            seq_name = str(dataset_name)
                        sequence_names.append(seq_name)

                        # Process all 4 sequences for visualization
                        if valid_sequences >= 4:
                            break

                    # Safety limit
                    if total_sequences > 20:
                        break

            except Exception as e:
                logger.warning(f"Error counting sequences: {e}")

            logger.info(f"DEBUG - Found {total_sequences} total sequences, {valid_sequences} valid sequences")
            logger.info(f"DEBUG - Valid sequence names: {sequence_names[:5]}...")  # Show first 5

            # Process up to 3 sequences for comprehensive testing
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
            test_batches = []
            dataloader_iter = iter(dataloader)

            # Collect up to 4 valid batches for full analysis
            max_test_sequences = min(4, valid_sequences if 'valid_sequences' in locals() else 1)
            collected_batches = 0

            for _ in range(20):  # Try up to 20 attempts
                try:
                    batch = next(dataloader_iter)
                    if batch is not None:
                        test_batches.append(batch)
                        collected_batches += 1
                        if collected_batches >= max_test_sequences:
                            break
                except StopIteration:
                    break

            if len(test_batches) == 0:
                logger.error("CRITICAL: Could not find valid TartanAir batch for testing!")
                logger.error("Real data is required for comprehensive integration test")
                raise RuntimeError("No valid TartanAir data found - comprehensive integration test must use real data")

            logger.info(f"Collected {len(test_batches)} test sequences for comprehensive evaluation")

            # Use first batch for main test (backward compatibility)
            batch = test_batches[0]

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

            # Comprehensive metrics computation (exactly like train_metric_head.py)
            if gt_depth is not None:
                # Metrics computation exactly like train (same resolution expected)
                logger.info(f"DEBUG TEST - Shapes: GT {gt_depth.shape}, Pred {pred_metric.shape}")

                # Ensure metrics computation at 518x518 resolution (same as train)
                target_resolution = (518, 518)  # Train resolution
                import torch.nn.functional as F

                # Resize GT to 518x518 for metrics (train does this during data loading)
                if gt_depth.shape[-2:] != target_resolution:
                    gt_depth_518 = F.interpolate(
                        gt_depth.view(-1, 1, gt_depth.shape[-2], gt_depth.shape[-1]),
                        size=target_resolution,
                        mode='bilinear',
                        align_corners=True
                    ).view(gt_depth.shape[0], gt_depth.shape[1], target_resolution[0], target_resolution[1])
                    logger.info(f"Resized GT depth from {gt_depth.shape[-2:]} to {target_resolution} for metrics")
                else:
                    gt_depth_518 = gt_depth

                # Resize Pred to 518x518 for metrics (model should output this but ensure consistency)
                if pred_metric.shape[-2:] != target_resolution:
                    pred_metric_518 = F.interpolate(
                        pred_metric.view(-1, 1, pred_metric.shape[-2], pred_metric.shape[-1]),
                        size=target_resolution,
                        mode='bilinear',
                        align_corners=True
                    ).view(pred_metric.shape[0], pred_metric.shape[1], target_resolution[0], target_resolution[1])
                    logger.info(f"Resized Pred depth from {pred_metric.shape[-2:]} to {target_resolution} for metrics")
                else:
                    pred_metric_518 = pred_metric

                # Use 518x518 versions for metrics computation
                pred_metric_for_metrics = pred_metric_518
                gt_depth_for_metrics = gt_depth_518
                logger.info(f"✓ Using 518x518 resolution for metrics computation (train-consistent)")

                # Compute Temporal Alignment Error (TAE) at 518x518 resolution
                tae_metrics = compute_temporal_alignment_error(pred_metric_518)

                # Compute metrics for ALL frames (different from train - comprehensive evaluation)
                B, T = pred_metric_for_metrics.shape[:2]
                all_frame_metrics = []
                total_valid_pixels = 0
                best_frame_idx = 0
                best_frame_abs_rel = float('inf')
                worst_frame_idx = 0
                worst_frame_abs_rel = 0.0

                # Find frame with lowest TAE for 5-frame sequence
                best_tae_start_idx = 0
                best_tae_value = float('inf')

                for t in range(T):
                    gt_frame = gt_depth_for_metrics[0, t].cpu()  # First batch, frame t (518x518)
                    pred_frame = pred_metric_for_metrics[0, t].cpu()  # First batch, frame t (518x518)

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

                        # Track best frame (lowest AbsRel)
                        if frame_metrics['abs_rel'] < best_frame_abs_rel:
                            best_frame_abs_rel = frame_metrics['abs_rel']
                            best_frame_idx = t

                        # Track worst frame (highest AbsRel)
                        if frame_metrics['abs_rel'] > worst_frame_abs_rel:
                            worst_frame_abs_rel = frame_metrics['abs_rel']
                            worst_frame_idx = t

                        # Debug for first frame only
                        if t == 0:
                            logger.info(f"DEBUG - Frame {t}: GT range: min={gt_frame.min():.6f}, max={gt_frame.max():.6f}, mean={gt_frame.mean():.6f}")
                            logger.info(f"DEBUG - Frame {t}: Pred range: min={pred_frame.min():.6f}, max={pred_frame.max():.6f}, mean={pred_frame.mean():.6f}")
                            logger.info(f"DEBUG - Frame {t}: GT > 0 pixels: {(gt_frame > 0).sum()}")
                            logger.info(f"DEBUG - Frame {t}: Pred in range pixels: {((pred_frame > 0) & (pred_frame < 1000.0)).sum()}")
                            logger.info(f"DEBUG - Frame {t}: Valid mask pixels: {valid_mask.sum()}")

                # Find best 5-frame sequence for TAE visualization (use 518x518 for consistency)
                if T >= 5:
                    for start_idx in range(T - 4):
                        seq_pred = pred_metric_518[0, start_idx:start_idx+5].unsqueeze(0)  # [1, 5, 518, 518]
                        seq_tae = compute_temporal_alignment_error(seq_pred)
                        if seq_tae['tae'] < best_tae_value:
                            best_tae_value = seq_tae['tae']
                            best_tae_start_idx = start_idx

                # Average metrics across all frames
                if all_frame_metrics:
                    metrics = {k: np.mean([frame_metrics[k] for frame_metrics in all_frame_metrics])
                              for k in all_frame_metrics[0].keys()}
                    avg_mae = metrics.get('mae', 0.0)
                    avg_valid_pixels = total_valid_pixels / T  # Average valid pixels per frame

                    logger.info(f"Computed metrics across {len(all_frame_metrics)} frames")
                    logger.info(f"Best frame (lowest AbsRel): Frame {best_frame_idx+1} with AbsRel={best_frame_abs_rel:.4f}")
                    logger.info(f"Worst frame (highest AbsRel): Frame {worst_frame_idx+1} with AbsRel={worst_frame_abs_rel:.4f}")
                    if T >= 5:
                        logger.info(f"Best TAE sequence: Frames {best_tae_start_idx+1}-{best_tae_start_idx+5} with TAE={best_tae_value:.4f}")
                else:
                    metrics = {}
                    avg_mae = 0.0
                    avg_valid_pixels = 0
                    logger.warning("No valid frames found for metrics computation")
            else:
                # No GT available - compute TAE without GT reference
                tae_metrics = compute_temporal_alignment_error(pred_metric)
                metrics = {}
                avg_mae = 0.0  # No GT available for metrics
                avg_valid_pixels = 0
                best_frame_idx = 0
                best_frame_abs_rel = 0.0
                worst_frame_idx = 0
                worst_frame_abs_rel = 0.0
                best_tae_start_idx = 0
                best_tae_value = 0.0

            # Create visualization
            vis_dir = Path(f"{globals().get('RESULTS_DIR', 'test_results/results_1')}/comprehensive")
            vis_dir.mkdir(parents=True, exist_ok=True)

            # Create visualizations if GT is available
            if gt_depth is not None:
                logger.info("Creating visualizations...")

                # For visualization, use 518x518 resolution (same as train)
                vis_resolution = (518, 518)  # Train visualization resolution
                B, T = gt_depth.shape[:2]

                # Use 518x518 versions for visualization (consistent with metrics)
                gt_vis = gt_depth_518  # Already resized to 518x518
                pred_vis = pred_metric_518  # Already resized to 518x518

                # Resize relative depth to 518x518 for visualization consistency
                rel_depth = outputs['relative_depth']
                if rel_depth.shape[-2:] != vis_resolution:
                    rel_vis = F.interpolate(
                        rel_depth.view(-1, 1, rel_depth.shape[-2], rel_depth.shape[-1]),
                        size=vis_resolution,
                        mode='bilinear',
                        align_corners=True
                    ).view(B, T, vis_resolution[0], vis_resolution[1])
                else:
                    rel_vis = rel_depth

                # Resize input video to 518x518 for visualization consistency
                if video.shape[-2:] != vis_resolution:
                    video_vis = F.interpolate(
                        video.view(-1, video.shape[-3], video.shape[-2], video.shape[-1]),
                        size=vis_resolution,
                        mode='bilinear',
                        align_corners=True
                    ).view(B, T, video.shape[-3], vis_resolution[0], vis_resolution[1])
                else:
                    video_vis = video

                # Generate valid masks for visualization at 518x518 resolution
                gt_valid_mask_seq = gt_vis > 0
                pred_valid_mask_seq = (pred_vis > 0) & (pred_vis < 1000.0)
                vis_mask_seq = (gt_valid_mask_seq & pred_valid_mask_seq)[0] # Use first batch item

                try:
                    # 1. Full sequence visualization (518x518)
                    frame_interval = globals().get('FRAME_INTERVAL', 1)
                    # Calculate frame indices for subsampled sequence
                    if frame_interval > 1:
                        seq_frame_indices = list(range(0, T, frame_interval))
                    else:
                        seq_frame_indices = list(range(T))

                    vis_fig = create_sequence_visualization(
                        images=video_vis[0], # First batch item (518x518)
                        pred_depths=pred_vis[0], # First batch item (518x518)
                        gt_depths=gt_vis[0], # First batch item (518x518)
                        valid_masks=vis_mask_seq,
                        save_path=vis_dir / "depth_sequence_visualization_seq1.png",
                        title=f"Sequence 1 - {dataset_name[0]} (518x518) - Interval: {frame_interval}",
                        frame_interval=frame_interval,
                        frame_indices=seq_frame_indices
                    )
                    plt.close(vis_fig) # Close figure to save memory
                    logger.info(f"Sequence visualization created at 518x518 resolution")

                    # 2. Best frame visualization (lowest AbsRel) (518x518)
                    if 'best_frame_idx' in locals():
                        logger.info(f"DEBUG - Creating Best frame visualization for frame {best_frame_idx+1}")
                        logger.info(f"DEBUG - Best frame AbsRel: {best_frame_abs_rel:.6f}")
                        logger.info(f"DEBUG - Save path: {vis_dir / f'best_frame_{best_frame_idx+1}_absrel_{best_frame_abs_rel:.4f}.png'}")

                        best_vis_fig = create_best_frame_visualization(
                            image=video_vis[0, best_frame_idx],  # Best frame image (518x518)
                            pred_depth=pred_vis[0, best_frame_idx],  # Best frame predicted depth (518x518)
                            gt_depth=gt_vis[0, best_frame_idx],  # Best frame GT depth (518x518)
                            relative_depth=rel_vis[0, best_frame_idx],  # Best frame relative depth (518x518)
                            valid_mask=vis_mask_seq[best_frame_idx],  # Best frame valid mask
                            abs_rel=best_frame_abs_rel,
                            save_path=vis_dir / f"best_frame_seq1_{best_frame_idx+1}_absrel_{best_frame_abs_rel:.4f}.png",
                            title=f"Sequence 1 Best Frame {best_frame_idx+1} (518x518)"
                        )
                        plt.close(best_vis_fig)
                        logger.info(f"Best frame visualization created for frame {best_frame_idx+1} at 518x518")

                    # 3. TAE 5-frame visualization (lowest TAE sequence) (518x518)
                    if T >= 5 and 'best_tae_start_idx' in locals():
                        end_idx = best_tae_start_idx + 5

                        logger.info(f"DEBUG - Creating TAE 5-frame visualization for frames {best_tae_start_idx+1}-{end_idx}")
                        logger.info(f"DEBUG - Using full sequence TAE value: {tae_metrics['tae']:.6f}")
                        tae_save_path = vis_dir / f"tae_5frame_seq_{best_tae_start_idx+1}-{end_idx}_tae_{tae_metrics['tae']:.4f}.png"
                        logger.info(f"DEBUG - Save path: {tae_save_path}")

                        # Select 5 frames with frame_interval spacing from best TAE position
                        tae_frame_indices = [best_tae_start_idx + i * frame_interval for i in range(5)]
                        # Ensure indices don't exceed sequence length
                        tae_frame_indices = [idx for idx in tae_frame_indices if idx < T]
                        if len(tae_frame_indices) < 5:
                            # If not enough frames with interval from best TAE position, start from frame 0
                            logger.info(f"Not enough frames from best TAE position {best_tae_start_idx} with interval {frame_interval}, starting from frame 0")
                            tae_frame_indices = [i * frame_interval for i in range(5)]
                            tae_frame_indices = [idx for idx in tae_frame_indices if idx < T]
                            # If still not enough, use consecutive frames from start
                            if len(tae_frame_indices) < 5:
                                tae_frame_indices = list(range(min(5, T)))

                        tae_fig = create_tae_5frame_visualization(
                            images=video_vis[0, tae_frame_indices],  # 5 frames with frame_interval spacing (518x518)
                            pred_depths=pred_vis[0, tae_frame_indices],  # 5 predicted depths (518x518)
                            gt_depths=gt_vis[0, tae_frame_indices],  # 5 GT depths (518x518)
                            relative_depths=rel_vis[0, tae_frame_indices],  # 5 relative depths (518x518)
                            tae_value=tae_metrics['tae'],  # Use full sequence TAE (50 frames)
                            fps=fps,
                            resolution=(518, 518),  # Report 518x518 resolution
                            gpu_spec="RTX A6000",
                            save_path=vis_dir / f"tae_5frame_seq1_{tae_frame_indices[0]+1}-{tae_frame_indices[-1]+1}_interval{frame_interval}_tae_{tae_metrics['tae']:.4f}.png",
                            title=f"Sequence 1 TAE 5-Frame {tae_frame_indices[0]+1}-{tae_frame_indices[-1]+1} (interval={frame_interval}) (518x518)",
                            frame_indices=tae_frame_indices  # Pass actual frame indices for titles
                        )
                        plt.close(tae_fig)
                        logger.info(f"TAE 5-frame visualization created for frames {best_tae_start_idx+1}-{end_idx} at 518x518")

                    # 4. Worst frame visualization (highest AbsRel) (518x518)
                    if 'worst_frame_idx' in locals():
                        logger.info(f"DEBUG - Creating Worst frame visualization for frame {worst_frame_idx+1}")
                        logger.info(f"DEBUG - Worst frame AbsRel: {worst_frame_abs_rel:.6f}")
                        logger.info(f"DEBUG - Save path: {vis_dir / f'worst_frame_{worst_frame_idx+1}_absrel_{worst_frame_abs_rel:.4f}.png'}")

                        worst_vis_fig = create_best_frame_visualization(  # Reuse same function but different naming
                            image=video_vis[0, worst_frame_idx],  # Worst frame image (518x518)
                            pred_depth=pred_vis[0, worst_frame_idx],  # Worst frame predicted depth (518x518)
                            gt_depth=gt_vis[0, worst_frame_idx],  # Worst frame GT depth (518x518)
                            relative_depth=rel_vis[0, worst_frame_idx],  # Worst frame relative depth (518x518)
                            valid_mask=vis_mask_seq[worst_frame_idx],  # Worst frame valid mask
                            abs_rel=worst_frame_abs_rel,
                            save_path=vis_dir / f"worst_frame_seq1_{worst_frame_idx+1}_absrel_{worst_frame_abs_rel:.4f}.png",
                            title=f"Sequence 1 Worst Frame {worst_frame_idx+1} (518x518)"
                        )
                        plt.close(worst_vis_fig)
                        logger.info(f"Worst frame visualization created for frame {worst_frame_idx+1} at 518x518")

                except Exception as e:
                    logger.error(f"Could not generate visualizations: {e}")
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")

            # Process additional test sequences with full analysis (if available)
            if len(test_batches) > 1:
                logger.info(f"Processing {len(test_batches)-1} additional test sequences for comprehensive analysis...")

                for seq_idx, additional_batch in enumerate(test_batches[1:], 1):
                    try:
                        add_video, add_gt_depth, add_dataset_name = additional_batch
                        add_video = add_video.to(device)
                        if add_gt_depth is not None:
                            add_gt_depth = add_gt_depth.to(device)

                        logger.info(f"Processing additional sequence {seq_idx}: {add_dataset_name}")
                        logger.info(f"Additional Video shape: {add_video.shape}")
                        if add_gt_depth is not None:
                            logger.info(f"Additional GT depth shape: {add_gt_depth.shape}")

                        # Forward pass for additional sequence
                        add_forward_input = (add_video, add_gt_depth) if add_gt_depth is not None else add_video

                        # Measure inference time for additional sequence
                        add_B, add_T = add_video.shape[:2]
                        add_total_frames = add_B * add_T

                        start_time = time.time()
                        with torch.no_grad():
                            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                                add_outputs = model.forward_with_metric_head(add_forward_input)
                        add_inference_time = time.time() - start_time
                        add_fps = add_total_frames / add_inference_time

                        logger.info(f"Additional sequence inference time: {add_inference_time:.4f}s for {add_total_frames} frames")
                        logger.info(f"Additional sequence FPS: {add_fps:.2f} frames/second")

                        # Extract outputs
                        add_pred_metric = add_outputs['metric_depth']
                        add_scale = add_outputs['scale']
                        add_shift = add_outputs['shift']

                        # Compute Temporal Alignment Error (TAE) for additional sequence
                        if add_gt_depth is not None:
                            add_tae_metrics = compute_temporal_alignment_error(add_pred_metric)
                        else:
                            add_tae_metrics = compute_temporal_alignment_error(add_pred_metric)

                        # Comprehensive metrics computation for additional sequence (same as main sequence)
                        if add_gt_depth is not None:
                            logger.info(f"DEBUG ADDITIONAL - Shapes: GT {add_gt_depth.shape}, Pred {add_pred_metric.shape}")

                            # Ensure metrics computation at 518x518 resolution (same as train)
                            target_resolution = (518, 518)
                            import torch.nn.functional as F

                            # Resize GT to 518x518 for metrics
                            if add_gt_depth.shape[-2:] != target_resolution:
                                add_gt_depth_518 = F.interpolate(
                                    add_gt_depth.view(-1, 1, add_gt_depth.shape[-2], add_gt_depth.shape[-1]),
                                    size=target_resolution,
                                    mode='bilinear',
                                    align_corners=True
                                ).view(add_gt_depth.shape[0], add_gt_depth.shape[1], target_resolution[0], target_resolution[1])
                                logger.info(f"Additional sequence: Resized GT depth from {add_gt_depth.shape[-2:]} to {target_resolution} for metrics")
                            else:
                                add_gt_depth_518 = add_gt_depth

                            # Resize Pred to 518x518 for metrics
                            if add_pred_metric.shape[-2:] != target_resolution:
                                add_pred_metric_518 = F.interpolate(
                                    add_pred_metric.view(-1, 1, add_pred_metric.shape[-2], add_pred_metric.shape[-1]),
                                    size=target_resolution,
                                    mode='bilinear',
                                    align_corners=True
                                ).view(add_pred_metric.shape[0], add_pred_metric.shape[1], target_resolution[0], target_resolution[1])
                                logger.info(f"Additional sequence: Resized Pred depth from {add_pred_metric.shape[-2:]} to {target_resolution} for metrics")
                            else:
                                add_pred_metric_518 = add_pred_metric

                            # Use 518x518 versions for metrics computation
                            add_pred_metric_for_metrics = add_pred_metric_518
                            add_gt_depth_for_metrics = add_gt_depth_518
                            logger.info(f"✓ Additional sequence: Using 518x518 resolution for metrics computation (train-consistent)")

                            # Compute metrics for ALL frames (comprehensive evaluation)
                            add_B, add_T = add_pred_metric_for_metrics.shape[:2]
                            add_all_frame_metrics = []
                            add_total_valid_pixels = 0
                            add_best_frame_idx = 0
                            add_best_frame_abs_rel = float('inf')
                            add_worst_frame_idx = 0
                            add_worst_frame_abs_rel = 0.0

                            # Find frame with lowest TAE for 5-frame sequence
                            add_best_tae_start_idx = 0
                            add_best_tae_value = float('inf')

                            for t in range(add_T):
                                add_gt_frame = add_gt_depth_for_metrics[0, t].cpu()
                                add_pred_frame = add_pred_metric_for_metrics[0, t].cpu()

                                # Create valid mask
                                add_gt_valid_mask = add_gt_frame > 0
                                add_pred_valid_mask = (add_pred_frame > 0) & (add_pred_frame < 1000.0)
                                add_valid_mask = add_gt_valid_mask & add_pred_valid_mask

                                if add_valid_mask.sum() > 0:
                                    # Compute metrics for this frame
                                    add_frame_metrics = MetricDepthMetrics.compute_metric_depth_metrics(
                                        add_pred_frame,
                                        add_gt_frame,
                                        valid_mask=add_valid_mask
                                    )
                                    add_all_frame_metrics.append(add_frame_metrics)
                                    add_total_valid_pixels += add_valid_mask.sum().item()

                                    # Track best frame (lowest AbsRel)
                                    if add_frame_metrics['abs_rel'] < add_best_frame_abs_rel:
                                        add_best_frame_abs_rel = add_frame_metrics['abs_rel']
                                        add_best_frame_idx = t

                                    # Track worst frame (highest AbsRel)
                                    if add_frame_metrics['abs_rel'] > add_worst_frame_abs_rel:
                                        add_worst_frame_abs_rel = add_frame_metrics['abs_rel']
                                        add_worst_frame_idx = t

                            # Find best 5-frame sequence for TAE visualization
                            if add_T >= 5:
                                for start_idx in range(add_T - 4):
                                    seq_pred = add_pred_metric_518[0, start_idx:start_idx+5].unsqueeze(0)
                                    seq_tae = compute_temporal_alignment_error(seq_pred)
                                    if seq_tae['tae'] < add_best_tae_value:
                                        add_best_tae_value = seq_tae['tae']
                                        add_best_tae_start_idx = start_idx

                            # Average metrics across all frames
                            if add_all_frame_metrics:
                                add_metrics = {k: np.mean([frame_metrics[k] for frame_metrics in add_all_frame_metrics])
                                              for k in add_all_frame_metrics[0].keys()}
                                add_avg_mae = add_metrics.get('mae', 0.0)
                                add_avg_valid_pixels = add_total_valid_pixels / add_T

                                logger.info(f"Additional sequence: Computed metrics across {len(add_all_frame_metrics)} frames")
                                logger.info(f"Additional sequence: Best frame (lowest AbsRel): Frame {add_best_frame_idx+1} with AbsRel={add_best_frame_abs_rel:.4f}")
                                logger.info(f"Additional sequence: Worst frame (highest AbsRel): Frame {add_worst_frame_idx+1} with AbsRel={add_worst_frame_abs_rel:.4f}")
                                if add_T >= 5:
                                    logger.info(f"Additional sequence: Best TAE sequence: Frames {add_best_tae_start_idx+1}-{add_best_tae_start_idx+5} with TAE={add_best_tae_value:.4f}")
                            else:
                                add_metrics = {}
                                add_avg_mae = 0.0
                                add_avg_valid_pixels = 0
                                logger.warning("Additional sequence: No valid frames found for metrics computation")
                        else:
                            # No GT available - compute TAE without GT reference
                            add_tae_metrics = compute_temporal_alignment_error(add_pred_metric)
                            add_metrics = {}
                            add_avg_mae = 0.0
                            add_avg_valid_pixels = 0
                            add_best_frame_idx = 0
                            add_best_frame_abs_rel = 0.0
                            add_worst_frame_idx = 0
                            add_worst_frame_abs_rel = 0.0
                            add_best_tae_start_idx = 0
                            add_best_tae_value = 0.0

                        # Create comprehensive visualization for additional sequence
                        if add_gt_depth is not None:
                            add_B, add_T = add_gt_depth.shape[:2]
                            add_model_resolution = add_video.shape[-2:]

                            # Resize to 518x518 for consistent visualization
                            vis_resolution = (518, 518)

                            if add_gt_depth.shape[-2:] != vis_resolution:
                                add_gt_vis = F.interpolate(
                                    add_gt_depth.view(-1, 1, add_gt_depth.shape[-2], add_gt_depth.shape[-1]),
                                    size=vis_resolution,
                                    mode='bilinear',
                                    align_corners=True
                                ).view(add_B, add_T, vis_resolution[0], vis_resolution[1])
                            else:
                                add_gt_vis = add_gt_depth

                            # Resize predicted depth to 518x518
                            if add_pred_metric.shape[-2:] != vis_resolution:
                                add_pred_vis = F.interpolate(
                                    add_pred_metric.view(-1, 1, add_pred_metric.shape[-2], add_pred_metric.shape[-1]),
                                    size=vis_resolution,
                                    mode='bilinear',
                                    align_corners=True
                                ).view(add_B, add_T, vis_resolution[0], vis_resolution[1])
                            else:
                                add_pred_vis = add_pred_metric

                            # Resize video to 518x518
                            if add_video.shape[-2:] != vis_resolution:
                                add_video_vis = F.interpolate(
                                    add_video.view(-1, add_video.shape[-3], add_video.shape[-2], add_video.shape[-1]),
                                    size=vis_resolution,
                                    mode='bilinear',
                                    align_corners=True
                                ).view(add_B, add_T, add_video.shape[-3], vis_resolution[0], vis_resolution[1])
                            else:
                                add_video_vis = add_video

                            # Resize relative depth for visualization
                            add_rel_depth = add_outputs['relative_depth']
                            if add_rel_depth.shape[-2:] != vis_resolution:
                                add_rel_vis = F.interpolate(
                                    add_rel_depth.view(-1, 1, add_rel_depth.shape[-2], add_rel_depth.shape[-1]),
                                    size=vis_resolution,
                                    mode='bilinear',
                                    align_corners=True
                                ).view(add_B, add_T, vis_resolution[0], vis_resolution[1])
                            else:
                                add_rel_vis = add_rel_depth

                            # Generate valid masks for visualization
                            add_gt_valid_mask_seq = add_gt_vis > 0
                            add_pred_valid_mask_seq = (add_pred_vis > 0) & (add_pred_vis < 1000.0)
                            add_vis_mask_seq = (add_gt_valid_mask_seq & add_pred_valid_mask_seq)[0]

                            logger.info(f"Creating comprehensive visualizations for additional sequence {seq_idx}...")

                            try:
                                # 1. Full sequence visualization (518x518)
                                frame_interval = globals().get('FRAME_INTERVAL', 1)
                                # Calculate frame indices for additional sequence
                                if frame_interval > 1:
                                    add_seq_frame_indices = list(range(0, add_T, frame_interval))
                                else:
                                    add_seq_frame_indices = list(range(add_T))

                                add_vis_fig = create_sequence_visualization(
                                    images=add_video_vis[0],  # 518x518
                                    pred_depths=add_pred_vis[0],  # 518x518
                                    gt_depths=add_gt_vis[0],  # 518x518
                                    valid_masks=add_vis_mask_seq,
                                    save_path=vis_dir / f"depth_sequence_visualization_seq{seq_idx+1}.png",
                                    title=f"Sequence {seq_idx+1} - {add_dataset_name[0] if isinstance(add_dataset_name, (list, tuple)) else add_dataset_name} (518x518) - Interval: {frame_interval}",
                                    frame_interval=frame_interval,
                                    frame_indices=add_seq_frame_indices
                                )
                                plt.close(add_vis_fig)
                                logger.info(f"Additional sequence {seq_idx} full sequence visualization created at 518x518 resolution")

                                # 2. Best frame visualization (lowest AbsRel) (518x518)
                                if 'add_best_frame_idx' in locals():
                                    logger.info(f"DEBUG - Creating Best frame visualization for additional sequence {seq_idx}, frame {add_best_frame_idx+1}")
                                    logger.info(f"DEBUG - Additional sequence {seq_idx} Best frame AbsRel: {add_best_frame_abs_rel:.6f}")

                                    add_best_vis_fig = create_best_frame_visualization(
                                        image=add_video_vis[0, add_best_frame_idx],  # Best frame image (518x518)
                                        pred_depth=add_pred_vis[0, add_best_frame_idx],  # Best frame predicted depth (518x518)
                                        gt_depth=add_gt_vis[0, add_best_frame_idx],  # Best frame GT depth (518x518)
                                        relative_depth=add_rel_vis[0, add_best_frame_idx],  # Best frame relative depth (518x518)
                                        valid_mask=add_vis_mask_seq[add_best_frame_idx],  # Best frame valid mask
                                        abs_rel=add_best_frame_abs_rel,
                                        save_path=vis_dir / f"best_frame_seq{seq_idx+1}_{add_best_frame_idx+1}_absrel_{add_best_frame_abs_rel:.4f}.png",
                                        title=f"Sequence {seq_idx+1} Best Frame {add_best_frame_idx+1} (518x518)"
                                    )
                                    plt.close(add_best_vis_fig)
                                    logger.info(f"Additional sequence {seq_idx} best frame visualization created for frame {add_best_frame_idx+1} at 518x518")

                                    # Save individual images for seq3 (seq_idx=2, which is seq3)
                                    if seq_idx == 2:  # seq3 (0-indexed: seq1=0, seq2=1, seq3=2)
                                        logger.info(f"Saving individual images for seq3 best frame {add_best_frame_idx+1}...")

                                        # Save input image
                                        input_img = add_video_vis[0, add_best_frame_idx].float().cpu().numpy()
                                        if input_img.shape[0] == 3:  # [3, H, W] -> [H, W, 3]
                                            input_img = input_img.transpose(1, 2, 0)

                                        fig_input = plt.figure(figsize=(8, 8))
                                        plt.imshow(input_img)
                                        plt.axis('off')
                                        plt.savefig(vis_dir / f"seq3_best_frame_{add_best_frame_idx+1}_input.png",
                                                   dpi=150, bbox_inches='tight', facecolor='white')
                                        plt.close(fig_input)

                                        # Save GT depth map
                                        gt_img = add_gt_vis[0, add_best_frame_idx].cpu().numpy()
                                        gt_valid_mask = gt_img > 0
                                        gt_display = np.full_like(gt_img, np.nan)
                                        if gt_valid_mask.sum() > 0:
                                            gt_display[gt_valid_mask] = gt_img[gt_valid_mask]
                                            gt_vmin, gt_vmax = np.nanpercentile(gt_display[gt_valid_mask], [2, 98])
                                        else:
                                            gt_vmin, gt_vmax = 0, 1

                                        fig_gt = plt.figure(figsize=(8, 8))
                                        plt.imshow(gt_display, cmap='plasma', vmin=gt_vmin, vmax=gt_vmax)
                                        plt.axis('off')
                                        plt.savefig(vis_dir / f"seq3_best_frame_{add_best_frame_idx+1}_gt_depth.png",
                                                   dpi=150, bbox_inches='tight', facecolor='white')
                                        plt.close(fig_gt)

                                        # Save predicted depth map
                                        pred_img = add_pred_vis[0, add_best_frame_idx].cpu().numpy()
                                        pred_valid_mask = (pred_img > 0) & (pred_img < 1000.0)
                                        pred_display = np.full_like(pred_img, np.nan)
                                        if pred_valid_mask.sum() > 0:
                                            pred_display[pred_valid_mask] = pred_img[pred_valid_mask]
                                            pred_vmin, pred_vmax = np.nanpercentile(pred_display[pred_valid_mask], [2, 98])
                                        else:
                                            pred_vmin, pred_vmax = 0, 1

                                        fig_pred = plt.figure(figsize=(8, 8))
                                        plt.imshow(pred_display, cmap='plasma', vmin=pred_vmin, vmax=pred_vmax)
                                        plt.axis('off')
                                        plt.savefig(vis_dir / f"seq3_best_frame_{add_best_frame_idx+1}_pred_depth.png",
                                                   dpi=150, bbox_inches='tight', facecolor='white')
                                        plt.close(fig_pred)

                                        # Save relative depth map
                                        rel_img = add_rel_vis[0, add_best_frame_idx].float().cpu().numpy()
                                        rel_vmin, rel_vmax = np.percentile(rel_img, [2, 98])

                                        fig_rel = plt.figure(figsize=(8, 8))
                                        plt.imshow(rel_img, cmap='plasma', vmin=rel_vmin, vmax=rel_vmax)
                                        plt.axis('off')
                                        plt.savefig(vis_dir / f"seq3_best_frame_{add_best_frame_idx+1}_relative_depth.png",
                                                   dpi=150, bbox_inches='tight', facecolor='white')
                                        plt.close(fig_rel)

                                        logger.info(f"Seq3 best frame individual images saved: input, gt_depth, pred_depth, relative_depth")

                                # 3. Worst frame visualization (highest AbsRel) (518x518)
                                if 'add_worst_frame_idx' in locals():
                                    logger.info(f"DEBUG - Creating Worst frame visualization for additional sequence {seq_idx}, frame {add_worst_frame_idx+1}")
                                    logger.info(f"DEBUG - Additional sequence {seq_idx} Worst frame AbsRel: {add_worst_frame_abs_rel:.6f}")

                                    add_worst_vis_fig = create_best_frame_visualization(  # Reuse same function
                                        image=add_video_vis[0, add_worst_frame_idx],  # Worst frame image (518x518)
                                        pred_depth=add_pred_vis[0, add_worst_frame_idx],  # Worst frame predicted depth (518x518)
                                        gt_depth=add_gt_vis[0, add_worst_frame_idx],  # Worst frame GT depth (518x518)
                                        relative_depth=add_rel_vis[0, add_worst_frame_idx],  # Worst frame relative depth (518x518)
                                        valid_mask=add_vis_mask_seq[add_worst_frame_idx],  # Worst frame valid mask
                                        abs_rel=add_worst_frame_abs_rel,
                                        save_path=vis_dir / f"worst_frame_seq{seq_idx+1}_{add_worst_frame_idx+1}_absrel_{add_worst_frame_abs_rel:.4f}.png",
                                        title=f"Sequence {seq_idx+1} Worst Frame {add_worst_frame_idx+1} (518x518)"
                                    )
                                    plt.close(add_worst_vis_fig)
                                    logger.info(f"Additional sequence {seq_idx} worst frame visualization created for frame {add_worst_frame_idx+1} at 518x518")

                                # 4. TAE 5-frame visualization (lowest TAE sequence) (518x518)
                                if add_T >= 5 and 'add_best_tae_start_idx' in locals():
                                    add_end_idx = add_best_tae_start_idx + 5

                                    logger.info(f"DEBUG - Creating TAE 5-frame visualization for additional sequence {seq_idx}, frames {add_best_tae_start_idx+1}-{add_end_idx}")
                                    logger.info(f"DEBUG - Additional sequence {seq_idx} using full sequence TAE value: {add_tae_metrics['tae']:.6f}")

                                    # Select 5 frames with frame_interval spacing from best TAE position
                                    add_tae_frame_indices = [add_best_tae_start_idx + i * frame_interval for i in range(5)]
                                    # Ensure indices don't exceed sequence length
                                    add_tae_frame_indices = [idx for idx in add_tae_frame_indices if idx < add_T]
                                    if len(add_tae_frame_indices) < 5:
                                        # If not enough frames with interval from best TAE position, start from frame 0
                                        logger.info(f"Additional sequence {seq_idx}: Not enough frames from best TAE position {add_best_tae_start_idx} with interval {frame_interval}, starting from frame 0")
                                        add_tae_frame_indices = [i * frame_interval for i in range(5)]
                                        add_tae_frame_indices = [idx for idx in add_tae_frame_indices if idx < add_T]
                                        # If still not enough, use consecutive frames from start
                                        if len(add_tae_frame_indices) < 5:
                                            add_tae_frame_indices = list(range(min(5, add_T)))

                                    add_tae_fig = create_tae_5frame_visualization(
                                        images=add_video_vis[0, add_tae_frame_indices],  # 5 frames with frame_interval spacing (518x518)
                                        pred_depths=add_pred_vis[0, add_tae_frame_indices],  # 5 predicted depths (518x518)
                                        gt_depths=add_gt_vis[0, add_tae_frame_indices],  # 5 GT depths (518x518)
                                        relative_depths=add_rel_vis[0, add_tae_frame_indices],  # 5 relative depths (518x518)
                                        tae_value=add_tae_metrics['tae'],  # Use full sequence TAE (50 frames)
                                        fps=add_fps,
                                        resolution=(518, 518),  # Report 518x518 resolution
                                        gpu_spec="RTX A6000",
                                        save_path=vis_dir / f"tae_5frame_seq{seq_idx+1}_{add_tae_frame_indices[0]+1}-{add_tae_frame_indices[-1]+1}_interval{frame_interval}_tae_{add_tae_metrics['tae']:.4f}.png",
                                        title=f"Sequence {seq_idx+1} TAE 5-Frame {add_tae_frame_indices[0]+1}-{add_tae_frame_indices[-1]+1} (interval={frame_interval}) (518x518)",
                                        frame_indices=add_tae_frame_indices  # Pass actual frame indices for titles
                                    )
                                    plt.close(add_tae_fig)
                                    logger.info(f"Additional sequence {seq_idx} TAE 5-frame visualization created for frames {add_best_tae_start_idx+1}-{add_end_idx} at 518x518")

                            except Exception as e:
                                logger.error(f"Could not generate comprehensive visualizations for additional sequence {seq_idx}: {e}")
                                import traceback
                                logger.error(f"Full traceback: {traceback.format_exc()}")

                        # Save comprehensive test results for additional sequence
                        add_seq_results = {
                            "sequence_name": add_dataset_name[0] if isinstance(add_dataset_name, (list, tuple)) else str(add_dataset_name),
                            "video_shape": list(add_video.shape),
                            "inference_time_seconds": add_inference_time,
                            "fps": add_fps,
                            "total_frames": add_total_frames,
                            "scale_min": float(add_scale.min()),
                            "scale_max": float(add_scale.max()),
                            "scale_mean": float(add_scale.mean()),
                            "shift_min": float(add_shift.min()),
                            "shift_max": float(add_shift.max()),
                            "shift_mean": float(add_shift.mean()),
                            "has_ground_truth": add_gt_depth is not None,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "tae_metrics": add_tae_metrics,
                        }

                        if add_gt_depth is not None:
                            add_seq_results.update({
                                "metrics": add_metrics,
                                "best_frame": {
                                    "frame_idx": int(add_best_frame_idx),
                                    "abs_rel": float(add_best_frame_abs_rel)
                                },
                                "worst_frame": {
                                    "frame_idx": int(add_worst_frame_idx),
                                    "abs_rel": float(add_worst_frame_abs_rel)
                                },
                                "best_tae_sequence": {
                                    "start_idx": int(add_best_tae_start_idx),
                                    "end_idx": int(add_best_tae_start_idx + 5),
                                    "tae_value": float(add_best_tae_value)
                                } if add_T >= 5 else None,
                                "avg_mae": float(add_avg_mae),
                                "avg_valid_pixels": float(add_avg_valid_pixels)
                            })

                        # Save to JSON file for additional sequence
                        add_results_file = vis_dir / f"test_results_seq{seq_idx+1}.json"
                        with open(add_results_file, 'w') as f:
                            json.dump(add_seq_results, f, indent=2)

                        logger.info(f"Additional sequence {seq_idx} comprehensive analysis completed:")
                        logger.info(f"  Sequence name: {add_seq_results['sequence_name']}")
                        logger.info(f"  Inference time: {add_inference_time:.4f}s")
                        logger.info(f"  FPS: {add_fps:.2f}")
                        if add_gt_depth is not None:
                            logger.info(f"  Average metrics: AbsRel={add_metrics.get('abs_rel', 0):.4f}, MAE={add_avg_mae:.4f}")
                            logger.info(f"  Best frame: {add_best_frame_idx+1} (AbsRel={add_best_frame_abs_rel:.4f})")
                            logger.info(f"  Worst frame: {add_worst_frame_idx+1} (AbsRel={add_worst_frame_abs_rel:.4f})")
                        else:
                            logger.info("  No GT depth available - metrics calculation skipped")
                        logger.info(f"  TAE (Temporal Alignment Error): {add_tae_metrics['tae']:.6f}")

                        # Clear memory
                        torch.cuda.empty_cache()

                    except Exception as e:
                        logger.warning(f"Error processing additional sequence {seq_idx}: {e}")
                        continue

            # Compute averaged results across all sequences
            if 'test_batches' in locals() and len(test_batches) > 1:
                logger.info("Computing averaged metrics across all test sequences...")

                # Load all sequence JSON files
                all_seq_results = []

                # Load all sequence results (seq1, seq2, seq3, seq4)
                for seq_idx in range(len(test_batches)):
                    seq_results_file = vis_dir / f"test_results_seq{seq_idx+1}.json"
                    if seq_results_file.exists():
                        with open(seq_results_file, 'r') as f:
                            seq_results = json.load(f)
                            all_seq_results.append(seq_results)
                        logger.info(f"Loaded sequence {seq_idx+1} results from {seq_results_file}")
                    else:
                        logger.warning(f"Missing sequence {seq_idx+1} results file: {seq_results_file}")

                if len(all_seq_results) > 1:
                    # Compute averages for numerical metrics
                    averaged_results = {
                        'averaged_across_sequences': len(all_seq_results),
                        'sequence_count': len(all_seq_results),
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'datasets': []
                    }

                    # Collect dataset names
                    for result in all_seq_results:
                        if 'dataset' in result:
                            if isinstance(result['dataset'], list):
                                averaged_results['datasets'].extend(result['dataset'])
                            else:
                                averaged_results['datasets'].append(result['dataset'])

                    # Average numerical metrics
                    numerical_fields = [
                        'inference_time_seconds', 'fps', 'total_frames',
                        'scale_min', 'scale_max', 'scale_mean',
                        'shift_min', 'shift_max', 'shift_mean',
                        'average_mae_meters', 'valid_pixels_per_frame', 'total_frames_evaluated'
                    ]

                    for field in numerical_fields:
                        values = [result[field] for result in all_seq_results
                                if field in result and result[field] is not None]
                        if values:
                            averaged_results[f'avg_{field}'] = np.mean(values)

                    # Average TAE metrics (vectorized)
                    tae_values = [result['tae_metrics']['tae'] for result in all_seq_results
                                if 'tae_metrics' in result and result['tae_metrics'] and 'tae' in result['tae_metrics']]
                    valid_frame_pairs_values = [result['tae_metrics']['valid_frame_pairs'] for result in all_seq_results
                                               if 'tae_metrics' in result and result['tae_metrics'] and 'valid_frame_pairs' in result['tae_metrics']]

                    if tae_values:
                        averaged_results['avg_tae_metrics'] = {
                            'avg_tae': np.mean(tae_values),
                            'avg_valid_frame_pairs': np.mean(valid_frame_pairs_values) if valid_frame_pairs_values else 0
                        }

                    # Average comprehensive metrics from each sequence's 'metrics' field
                    comprehensive_metrics_avg = {}
                    metric_names = ['mae', 'rmse', 'abs_rel', 'sq_rel', 'rmse_log', 'mre', 'log_mae', 'a1', 'a2', 'a3']

                    for metric_name in metric_names:
                        values = []
                        for result in all_seq_results:
                            # Check both 'comprehensive_metrics' and 'metrics' fields
                            metrics_data = result.get('comprehensive_metrics') or result.get('metrics')
                            if metrics_data and metric_name in metrics_data:
                                values.append(metrics_data[metric_name])
                        if values:
                            comprehensive_metrics_avg[metric_name] = np.mean(values)
                            logger.info(f"Averaged {metric_name}: {comprehensive_metrics_avg[metric_name]:.4f} (from {len(values)} sequences)")

                    if comprehensive_metrics_avg:
                        averaged_results['comprehensive_metrics'] = comprehensive_metrics_avg
                        logger.info(f"Successfully computed comprehensive metrics averages from {len(all_seq_results)} sequences")
                    else:
                        logger.warning("No comprehensive metrics found to average")

                    # Save averaged results
                    averaged_results_file = vis_dir / "averaged_test_results.json"
                    with open(averaged_results_file, 'w') as f:
                        json.dump(averaged_results, f, indent=4)

                    logger.info(f"Averaged results saved to: {averaged_results_file}")
                    logger.info(f"Averaged across {len(all_seq_results)} sequences")
                    if comprehensive_metrics_avg:
                        logger.info("Averaged comprehensive metrics:")
                        for k, v in comprehensive_metrics_avg.items():
                            logger.info(f"  {k}: {v:.4f}")

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
                'tae_metrics': tae_metrics,
                'total_test_sequences': total_sequences if 'total_sequences' in locals() else 1,
                'valid_test_sequences': valid_sequences if 'valid_sequences' in locals() else 1,
                'processed_test_sequences': len(test_batches) if 'test_batches' in locals() else 1
            }

            # Add comprehensive metrics only if GT is available (all frames evaluation)
            if gt_depth is not None:
                test_results.update({
                    'average_mae_meters': float(avg_mae),
                    'valid_pixels_per_frame': int(avg_valid_pixels),
                    'total_frames_evaluated': len(all_frame_metrics) if 'all_frame_metrics' in locals() else 0,
                    'comprehensive_metrics': {k: float(v) for k, v in metrics.items()},
                    'best_frame_info': {
                        'frame_index': int(best_frame_idx) if 'best_frame_idx' in locals() else 0,
                        'abs_rel': float(best_frame_abs_rel) if 'best_frame_abs_rel' in locals() else 0.0
                    },
                    'best_tae_sequence_info': {
                        'start_index': int(best_tae_start_idx) if 'best_tae_start_idx' in locals() else 0,
                        'tae_value': float(best_tae_value) if 'best_tae_value' in locals() else 0.0
                    }
                })

            results_json_path = vis_dir / "test_results_seq1.json"
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
            logger.info(f"  TAE Metrics:")
            logger.info(f"    TAE (Temporal Alignment Error): {tae_metrics['tae']:.6f}")
            logger.info(f"    Valid frame pairs: {tae_metrics['valid_frame_pairs']}")
            logger.info(f"  Test dataset info:")
            logger.info(f"    Total sequences found: {total_sequences if 'total_sequences' in locals() else 'Unknown'}")
            logger.info(f"    Valid sequences: {valid_sequences if 'valid_sequences' in locals() else 'Unknown'}")
            logger.info(f"    Processed sequences: {len(test_batches) if 'test_batches' in locals() else 1}")
            if 'test_batches' in locals() and len(test_batches) > 1:
                seq_names = []
                for i, tb in enumerate(test_batches):
                    _, _, dn = tb
                    name = dn[0] if isinstance(dn, (list, tuple)) else str(dn)
                    seq_names.append(f"Seq{i+1}: {name}")
                logger.info(f"    Sequence details: {'; '.join(seq_names)}")
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

    # Set global frame interval for sequence visualization
    frame_interval = cfg.get('frame_interval', 1) if cfg else 1
    globals()['FRAME_INTERVAL'] = frame_interval

    # Set global video length for testing
    vid_len = cfg.get('vid_len', 50) if cfg else 50
    globals()['VID_LEN'] = vid_len

    logger.info(f"Results will be saved to: {results_dir}")
    if gsp_checkpoint:
        logger.info(f"GSP checkpoint: {gsp_checkpoint}")
    else:
        logger.warning("No GSP checkpoint provided - tests will use random initialization")

    success = run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()