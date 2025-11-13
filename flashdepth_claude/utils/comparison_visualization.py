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


def visualize_best_frame_simplified(image, gt_depth, pred_depth, metrics,
                                     save_dir, sequence_id, frame_idx,
                                     dataset_name=None, focal_length=None):
    """
    Save improved best frame visualization (3×3 grid with Valid Mask and Depth Distribution)

    Layout:
        Row 0: Input Image | GT Depth | Pred Depth
        Row 1: Valid Mask | Error Map | Depth Metrics (with Dataset info)
        Row 2: Depth Distribution (2 cols) | Empty

    Args:
        image: [3, H, W] or [H, W, 3] - RGB image
        gt_depth: [H, W] - Ground truth metric depth
        pred_depth: [H, W] - Predicted metric depth
        metrics: dict - Pre-computed metrics dictionary
        save_dir: Path - Save directory
        sequence_id: int - Sequence index
        frame_idx: int - Frame index
        dataset_name: str - Optional dataset name (e.g., 'eth3d/pipes')
        focal_length: float - Optional focal length
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

    # Create figure with 3×3 grid
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig,
                          height_ratios=[1, 1, 1],
                          hspace=0.35, wspace=0.3)

    MAX_DEPTH = 70.0

    # ==================== Row 0: Input, GT, Pred ====================

    # Row 0, Col 0: Input Image
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image)
    ax_img.set_title('Input Image', fontsize=12, fontweight='bold', pad=8)
    ax_img.axis('off')

    # Row 0, Col 1: GT Depth
    ax_gt = fig.add_subplot(gs[0, 1])
    gt_valid = (gt_depth > 0) & (gt_depth < MAX_DEPTH)
    gt_display = np.where(gt_valid, gt_depth, np.nan)

    if gt_valid.sum() > 0:
        gt_vmin = np.nanpercentile(gt_display, 2)
        gt_vmax = np.nanpercentile(gt_display, 98)
    else:
        gt_vmin, gt_vmax = 0, 1

    cmap_gt = plt.cm.plasma_r.copy()
    cmap_gt.set_bad(color='black')
    im_gt = ax_gt.imshow(gt_display, cmap=cmap_gt, vmin=gt_vmin, vmax=gt_vmax)
    ax_gt.set_title('Ground Truth Depth (m)', fontsize=12, fontweight='bold', pad=8)
    ax_gt.axis('off')
    plt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)

    # Row 0, Col 2: Predicted Depth
    ax_pred = fig.add_subplot(gs[0, 2])
    pred_valid = (pred_depth > 0) & (pred_depth < MAX_DEPTH)
    pred_display = np.where(pred_valid, pred_depth, np.nan)

    cmap_pred = plt.cm.plasma_r.copy()
    cmap_pred.set_bad(color='black')
    im_pred = ax_pred.imshow(pred_display, cmap=cmap_pred, vmin=gt_vmin, vmax=gt_vmax)
    ax_pred.set_title('Predicted Depth (m)', fontsize=12, fontweight='bold', pad=8)
    ax_pred.axis('off')
    plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)

    # ==================== Row 1: Valid Mask, Error Map, Metrics ====================

    # Row 1, Col 0: Valid Mask (GT valid only, 70m threshold)
    ax_valid = fig.add_subplot(gs[1, 0])
    valid_mask_vis = gt_valid.astype(np.uint8)
    ax_valid.imshow(valid_mask_vis, cmap='gray', vmin=0, vmax=1)
    ax_valid.set_title('Valid Mask (GT ≤70m)', fontsize=12, fontweight='bold', pad=8)
    ax_valid.axis('off')

    # Row 1, Col 1: Absolute Error Map
    ax_error = fig.add_subplot(gs[1, 1])
    error_valid_mask = gt_valid & pred_valid
    abs_error = np.abs(pred_depth - gt_depth)
    abs_error_masked = np.where(error_valid_mask, abs_error, np.nan)

    if error_valid_mask.sum() > 0:
        error_vmax = np.nanpercentile(abs_error_masked, 95)
    else:
        error_vmax = 1

    im_error = ax_error.imshow(abs_error_masked, cmap='hot', vmin=0, vmax=error_vmax)
    ax_error.set_title(f'Absolute Error (m)\nMean: {np.nanmean(abs_error_masked):.3f}',
                      fontsize=12, fontweight='bold', pad=8)
    ax_error.axis('off')
    plt.colorbar(im_error, ax=ax_error, fraction=0.046, pad=0.04)

    # Row 1, Col 2: Depth Metrics (with Dataset info)
    ax_metrics = fig.add_subplot(gs[1, 2])

    # Build metrics text
    text_lines = []

    # Dataset info
    if dataset_name:
        text_lines.append(f"Dataset: {dataset_name}")
    else:
        text_lines.append("Dataset: unknown")

    # Sequence and frame info
    text_lines.append(f"Seq: {sequence_id} | Frame: {frame_idx}")
    text_lines.append("")  # Blank line

    # Metrics from dictionary
    text_lines.append(f"MAE:    {metrics.get('mae', 0):.3f}")
    text_lines.append(f"RMSE:   {metrics.get('rmse', 0):.3f}")
    text_lines.append(f"AbsRel: {metrics.get('abs_rel', 0):.3f}")
    text_lines.append(f"δ1:     {metrics.get('a1', 0):.3f}")
    text_lines.append(f"δ2:     {metrics.get('a2', 0):.3f}")
    text_lines.append(f"δ3:     {metrics.get('a3', 0):.3f}")
    text_lines.append(f"F1:     {metrics.get('boundary_f1', 0):.3f}")

    # Add focal length if available
    if focal_length is not None:
        text_lines.append("")
        text_lines.append(f"fx:     {focal_length:.1f}")

    text = "\n".join(text_lines)
    ax_metrics.text(0.05, 0.95, text, fontsize=10,
                   verticalalignment='top', family='monospace',
                   transform=ax_metrics.transAxes)
    ax_metrics.set_title('Depth Metrics', fontsize=12, fontweight='bold', pad=8)
    ax_metrics.axis('off')

    # ==================== Row 2: Depth Distribution + Empty ====================

    # Row 2, Cols 0-1: Depth Distribution Histogram
    ax_dist = fig.add_subplot(gs[2, 0:2])

    # Compute valid masks
    valid_mask_both = gt_valid & pred_valid

    if valid_mask_both.sum() > 0:
        gt_valid_pixels = gt_depth[valid_mask_both]
        pred_valid_pixels = pred_depth[valid_mask_both]

        # Compute histogram bins
        all_depths = np.concatenate([gt_valid_pixels, pred_valid_pixels])
        bins = np.linspace(all_depths.min(), all_depths.max(), 50)

        # Plot histograms
        ax_dist.hist(gt_valid_pixels, bins=bins, alpha=0.6,
                    label='Ground Truth', color='blue', density=True)
        ax_dist.hist(pred_valid_pixels, bins=bins, alpha=0.6,
                    label='Predicted', color='red', density=True)

        ax_dist.set_xlabel('Depth (m)', fontsize=10)
        ax_dist.set_ylabel('Density', fontsize=10)
        ax_dist.set_title('Depth Distribution (Valid Pixels ≤70m)', fontsize=12, fontweight='bold')
        ax_dist.legend(fontsize=9, loc='upper right')
        ax_dist.grid(True, alpha=0.3)
    else:
        ax_dist.text(0.5, 0.5, 'No valid pixels for distribution',
                    ha='center', va='center', fontsize=11,
                    transform=ax_dist.transAxes)
        ax_dist.set_title('Depth Distribution', fontsize=12, fontweight='bold')
        ax_dist.axis('off')

    # Row 2, Col 2: Empty (reserved for future use)
    ax_empty = fig.add_subplot(gs[2, 2])
    ax_empty.axis('off')

    # Overall title
    abs_rel_value = metrics.get('abs_rel', 0)
    plt.suptitle(f'Best Frame Visualization: Sequence {sequence_id}, Frame {frame_idx}',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_path = save_dir / f"best_frame_seq{sequence_id}_{frame_idx}_absrel_{abs_rel_value:.4f}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Saved simplified best frame visualization: {save_path}")
