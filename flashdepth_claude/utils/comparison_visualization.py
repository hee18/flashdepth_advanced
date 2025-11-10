#!/usr/bin/env python3
"""
Simplified visualization for comparison methods (without Gear-specific components)

This module provides visualization functions for depth estimation comparison methods.
It removes Gear-specific visualizations like:
- Importance maps
- FG/BG masks
- Importance distribution
- Layer weights
- FG:BG ratio metrics

Simplified layouts:
- sequence.png: 3-row grid (Image | Predicted Depth | GT Depth)
- best_frame.png: 2x3 grid (simplified, no Gear-specific elements)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn.functional as F
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def visualize_sequence_simplified(images, pred_depths, gt_depths, valid_mask,
                                   sequence_id, metrics, fps, save_dir,
                                   frame_interval=None, focal_lengths=None, config=None):
    """
    Create simplified sequence visualization (3-row grid without importance maps)

    Args:
        images: [T, 3, H, W] - RGB images
        pred_depths: [T, 1, H, W] - Predicted metric depth
        gt_depths: [T, 1, H, W] - Ground truth metric depth
        valid_mask: [T, 1, H, W] - Valid pixels mask
        sequence_id: int - Sequence index
        metrics: dict - Evaluation metrics
        fps: float - Frames per second
        save_dir: Path - Directory to save visualization
        frame_interval: int - Optional frame sampling interval
        focal_lengths: [T] - Optional focal lengths
        config: dict - Optional configuration
    """
    T = images.shape[0]
    frames_to_show = min(10, T)

    # Determine frame interval
    if frame_interval is not None:
        interval = frame_interval
        logger.info(f"Using frame_interval={interval} for sequence.png visualization")
    else:
        interval = max(1, T // frames_to_show)

    frame_indices = list(range(0, T, interval))[:frames_to_show]
    actual_frames = len(frame_indices)

    # Create 3-row figure (removed importance map row)
    fig, axes = plt.subplots(3, actual_frames, figsize=(actual_frames * 3, 9))
    if actual_frames == 1:
        axes = axes.reshape(-1, 1)

    for col, t in enumerate(frame_indices):
        # Row 0: Image
        img = images[t].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        axes[0, col].imshow(img)
        axes[0, col].set_title(f'Frame {t}')
        axes[0, col].axis('off')

        # Row 1: Predicted metric depth
        MAX_DEPTH = 70.0
        pred = pred_depths[t, 0].cpu().numpy()
        gt = gt_depths[t, 0].cpu().numpy()

        # GT valid mask
        gt_valid = (gt > 0) & (gt < MAX_DEPTH)

        # Check if sparse dataset
        gt_exists = (gt > 0)
        gt_density = gt_exists.sum() / gt_exists.size
        is_sparse = gt_density < 0.5

        if is_sparse:
            # Sparse: height mask + fill
            valid_pixels_per_row = gt_exists.sum(axis=1)
            min_valid_pixels_threshold = 10
            valid_rows = valid_pixels_per_row >= min_valid_pixels_threshold
            valid_row_indices = np.where(valid_rows)[0]

            if len(valid_row_indices) > 0:
                scan_top = valid_row_indices[0]
                scan_bottom = valid_row_indices[-1]
                height_mask = np.zeros_like(pred, dtype=bool)
                height_mask[scan_top:scan_bottom+1, :] = True
                pred_show_mask = height_mask & (pred > 0) & (pred < MAX_DEPTH)
            else:
                pred_show_mask = (pred > 0) & (pred < MAX_DEPTH)
        else:
            # Dense: simple valid mask
            pred_show_mask = (pred > 0) & (pred < MAX_DEPTH)

        # Compute GT's percentile range
        if gt_valid.sum() > 0:
            gt_display = np.where(gt_valid, gt, np.nan)
            gt_vmin = np.nanpercentile(gt_display, 2)
            gt_vmax = np.nanpercentile(gt_display, 98)
        else:
            gt_vmin, gt_vmax = 0, 1

        # Use GT's range for Pred
        pred_display = np.where(pred_show_mask, pred, np.nan)
        cmap_pred = plt.cm.plasma_r.copy()
        cmap_pred.set_bad(color='black')
        axes[1, col].imshow(pred_display, cmap=cmap_pred, vmin=gt_vmin, vmax=gt_vmax)
        axes[1, col].set_title(f'Pred Depth')
        axes[1, col].axis('off')

        # Row 2: GT metric depth
        gt_display = np.where(gt_valid, gt, np.nan)
        cmap_gt = plt.cm.plasma_r.copy()
        cmap_gt.set_bad(color='black')
        axes[2, col].imshow(gt_display, cmap=cmap_gt, vmin=gt_vmin, vmax=gt_vmax)
        axes[2, col].set_title(f'GT Depth')
        axes[2, col].axis('off')

    # Add overall title with metrics (removed importance-related metrics)
    title_str = (
        f"Sequence {sequence_id} | "
        f"TAE: {metrics.get('tae', 0):.4f} | "
        f"AbsRel: {metrics.get('abs_rel', 0):.4f} | "
        f"δ1: {metrics.get('a1', 0):.4f} | "
        f"F1: {metrics.get('boundary_f1', 0):.3f}"
    )
    if fps is not None:
        title_str += f" | FPS: {fps:.1f}"

    # Add focal length info if available
    if focal_lengths is not None:
        fx_value = focal_lengths[0].item()
        title_str += f"\nfx: {fx_value:.1f}"

    fig.suptitle(title_str, fontsize=14)

    plt.tight_layout()
    save_path = save_dir / f"sequence_{sequence_id:04d}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Saved simplified visualization: {save_path}")


def visualize_best_frame_simplified(image, pred_depth, gt_depth, sequence_id, frame_idx,
                                     abs_rel, fps, save_dir, frame_metrics=None,
                                     seg_mask=None, class_metrics=None):
    """
    Save simplified best frame visualization (2x3 grid, no Gear-specific elements)

    Layout:
        Row 1: Input Image | GT Depth | Pred Depth
        Row 2: [Empty/Logo] | Error Map | Metrics

    Args:
        image: [3, H, W] or [H, W, 3] - RGB image
        pred_depth: [H, W] - Predicted metric depth
        gt_depth: [H, W] - Ground truth metric depth
        sequence_id: int - Sequence index
        frame_idx: int - Frame index
        abs_rel: float - AbsRel metric
        fps: float - Optional FPS
        save_dir: Path - Save directory
        frame_metrics: dict - Optional pre-computed metrics (includes boundary_f1)
        seg_mask: np.ndarray - Optional segmentation mask
        class_metrics: dict - Optional per-class metrics
    """
    # Convert tensors to numpy
    if isinstance(image, torch.Tensor):
        if image.shape[0] == 3:  # [3, H, W]
            image = image.permute(1, 2, 0)
        image = image.cpu().numpy()

    if isinstance(pred_depth, torch.Tensor):
        pred_depth = pred_depth.cpu().numpy()

    if isinstance(gt_depth, torch.Tensor):
        gt_depth = gt_depth.cpu().numpy()

    # Normalize image
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    image = np.clip(image, 0, 1)

    # Create figure with 2x3 grid
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # ==================== Row 1: Images and Depths ====================

    # 1. Input Image
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(image)
    ax0.set_title('Input Image', fontsize=14, fontweight='bold')
    ax0.axis('off')

    # 2. GT Depth
    ax1 = fig.add_subplot(gs[0, 1])
    MAX_DEPTH = 70.0
    gt_valid = (gt_depth > 0) & (gt_depth < MAX_DEPTH)
    gt_display = np.where(gt_valid, gt_depth, np.nan)

    if gt_valid.sum() > 0:
        gt_vmin = np.nanpercentile(gt_display, 2)
        gt_vmax = np.nanpercentile(gt_display, 98)
    else:
        gt_vmin, gt_vmax = 0, 1

    cmap_gt = plt.cm.plasma_r.copy()
    cmap_gt.set_bad(color='black')
    im1 = ax1.imshow(gt_display, cmap=cmap_gt, vmin=gt_vmin, vmax=gt_vmax)
    ax1.set_title('Ground Truth Depth (m)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # 3. Predicted Depth
    ax2 = fig.add_subplot(gs[0, 2])
    pred_valid = (pred_depth > 0) & (pred_depth < MAX_DEPTH)
    pred_display = np.where(pred_valid, pred_depth, np.nan)

    cmap_pred = plt.cm.plasma_r.copy()
    cmap_pred.set_bad(color='black')
    im2 = ax2.imshow(pred_display, cmap=cmap_pred, vmin=gt_vmin, vmax=gt_vmax)
    ax2.set_title('Predicted Depth (m)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # ==================== Row 2: Empty/Logo, Error Map, Metrics ====================

    # 4. Empty or Project Logo
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.text(0.5, 0.5, 'FlashDepth\nComparison',
             ha='center', va='center', fontsize=16, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax3.axis('off')

    # 5. Absolute Error Map
    ax4 = fig.add_subplot(gs[1, 1])
    error_valid_mask = gt_valid & pred_valid
    abs_error = np.abs(pred_depth - gt_depth)
    abs_error_masked = np.where(error_valid_mask, abs_error, np.nan)

    if error_valid_mask.sum() > 0:
        error_vmax = np.nanpercentile(abs_error_masked, 95)
    else:
        error_vmax = 1

    im4 = ax4.imshow(abs_error_masked, cmap='hot', vmin=0, vmax=error_vmax)
    ax4.set_title(f'Absolute Error (m)\nMean: {np.nanmean(abs_error_masked):.3f}',
                 fontsize=14, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    # 6. Depth Metrics
    ax5 = fig.add_subplot(gs[1, 2])
    y_pos = 0.95

    # Sequence info
    ax5.text(0.05, y_pos, f'Seq {sequence_id+1} Frame {frame_idx}', fontsize=11,
            transform=ax5.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'),
            fontweight='bold')
    y_pos -= 0.15

    # FPS if available
    if fps is not None:
        ax5.text(0.05, y_pos, f'FPS: {fps:.1f}', fontsize=10,
                transform=ax5.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
        y_pos -= 0.12

    # Compute metrics if valid pixels exist
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

        ax5.text(0.05, y_pos, f'AbsRel: {abs_rel:.4f}', fontsize=10,
                transform=ax5.transAxes, bbox=dict(boxstyle="round", facecolor='lightcoral'))
        y_pos -= 0.10
        ax5.text(0.05, y_pos, f'δ1: {delta_1:.3f}', fontsize=10,
                transform=ax5.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
        y_pos -= 0.10
        ax5.text(0.05, y_pos, f'δ2: {delta_2:.3f}', fontsize=10,
                transform=ax5.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
        y_pos -= 0.10
        ax5.text(0.05, y_pos, f'δ3: {delta_3:.3f}', fontsize=10,
                transform=ax5.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
        y_pos -= 0.10
        ax5.text(0.05, y_pos, f'RMSE: {rmse:.3f}m', fontsize=9,
                transform=ax5.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        y_pos -= 0.10
        ax5.text(0.05, y_pos, f'MAE: {mae:.3f}m', fontsize=9,
                transform=ax5.transAxes, bbox=dict(boxstyle="round", facecolor='lightblue'))
        y_pos -= 0.10

        # Add boundary F1 score if available
        if frame_metrics is not None and 'boundary_f1' in frame_metrics:
            boundary_f1 = frame_metrics['boundary_f1']
            ax5.text(0.05, y_pos, f'F1: {boundary_f1:.3f}', fontsize=9,
                    transform=ax5.transAxes, bbox=dict(boxstyle="round", facecolor='lavender'))
            y_pos -= 0.10

    ax5.set_title('Depth Metrics', fontsize=14, fontweight='bold')
    ax5.axis('off')

    # Overall title
    plt.suptitle(f'Comparison Method: Sequence {sequence_id} Best Frame {frame_idx}',
                fontsize=16, fontweight='bold')

    plt.tight_layout()
    save_path = save_dir / f"best_frame_seq{sequence_id}_{frame_idx}_absrel_{abs_rel:.4f}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Saved simplified best frame visualization: {save_path}")
