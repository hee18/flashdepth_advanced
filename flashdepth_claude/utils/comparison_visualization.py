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

# Import completed depth utilities for ETH3D/Waymo visualization
from utils.completed_depth import load_completed_depth, has_completed_depth

logger = logging.getLogger(__name__)


def visualize_sequence_simplified(images, pred_depths, gt_depths, valid_mask,
                                   sequence_id, metrics, fps, save_dir,
                                   frame_interval=None, focal_lengths=None, config=None,
                                   seg_masks=None, objwise_enabled=False, object_classes=None,
                                   depth_paths=None, dataset_name=None, max_depth=80.0):
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
        seg_masks: [T, H, W] - Optional segmentation masks for object-wise mode
        objwise_enabled: bool - Whether to overlay object masks
        object_classes: list - List of object class names (e.g., ['vehicle', 'pedestrian'])
        depth_paths: list[str] - Optional list of GT depth file paths for completed depth loading
        dataset_name: str - Dataset name (e.g., 'eth3d', 'waymo_seg') for completed depth
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
        # Row 0: Image with optional object mask overlay
        img = images[t].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)

        # Overlay object mask if object-wise mode (no changes to img display)
        # Object mask visualization is handled separately in object_wise_visualization.py

        axes[0, col].imshow(img)
        axes[0, col].set_title(f'Frame {t}')
        axes[0, col].axis('off')

        # Row 1: Predicted metric depth
        MAX_DEPTH = max_depth
        pred = pred_depths[t, 0].cpu().numpy()
        gt = gt_depths[t, 0].cpu().numpy()

        # GT valid mask
        gt_valid = (gt > 0) & (gt < MAX_DEPTH)

        # Check if sparse dataset
        gt_exists = (gt > 0)
        gt_density = gt_exists.sum() / gt_exists.size
        is_sparse = gt_density < 0.5

        if is_sparse:
            # Sparse: height mask + dense prediction within range
            valid_pixels_per_row = gt_valid.sum(axis=1)
            min_valid_pixels_threshold = 10
            valid_rows = valid_pixels_per_row >= min_valid_pixels_threshold
            valid_row_indices = np.where(valid_rows)[0]

            if len(valid_row_indices) > 0:
                min_valid_row = valid_row_indices.min()
                max_valid_row = valid_row_indices.max()
                height_mask = np.zeros_like(pred, dtype=bool)
                height_mask[min_valid_row:max_valid_row+1, :] = True
            else:
                height_mask = np.ones_like(pred, dtype=bool)

            pred_valid_depth = (pred > 0) & (pred < MAX_DEPTH)
            pred_show_mask = height_mask & pred_valid_depth  # Dense prediction within height range
        else:
            # Dense: use GT valid mask (same as test_gear5.py)
            pred_show_mask = gt_valid

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

        # Row 2: GT metric depth (use completed depth for ETH3D/Waymo visualization)
        gt_for_display = gt  # Default to sparse GT
        gt_display_title = 'GT Depth'

        # Try to load completed depth for ETH3D/Waymo
        if depth_paths is not None and dataset_name in ['eth3d', 'waymo_seg']:
            try:
                depth_path = depth_paths[t] if t < len(depth_paths) else None
                if depth_path:
                    H, W = gt.shape
                    completed = load_completed_depth(depth_path, dataset_name, target_size=(H, W))
                    if completed is not None:
                        gt_for_display = completed.numpy()
                        gt_display_title = 'GT Depth (completed)'
            except Exception as e:
                logger.debug(f"Could not load completed depth for frame {t}: {e}")

        # Create display with handling for Waymo's no-LiDAR regions (-1)
        if dataset_name == 'waymo_seg':
            # For Waymo: show no-LiDAR regions as black
            no_lidar_mask = gt_for_display < 0
            valid_for_display = (gt_for_display > 0) & (gt_for_display < MAX_DEPTH)
            gt_display = np.where(valid_for_display, gt_for_display, np.nan)
        else:
            gt_display = np.where((gt_for_display > 0) & (gt_for_display < MAX_DEPTH), gt_for_display, np.nan)

        cmap_gt = plt.cm.plasma_r.copy()
        cmap_gt.set_bad(color='black')
        axes[2, col].imshow(gt_display, cmap=cmap_gt, vmin=gt_vmin, vmax=gt_vmax)
        axes[2, col].set_title(gt_display_title)
        axes[2, col].axis('off')

    # Add overall title with metrics (removed importance-related metrics)
    title_str = (
        f"Sequence {sequence_id} | "
        f"TAE: {metrics.get('tae', 0):.4f} | "
        f"AbsRel: {metrics.get('abs_rel', 0):.4f} | "
        f"δ1: {metrics.get('a1', 0):.4f}"
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
                                     dataset_name=None, focal_length=None,
                                     seg_mask=None, objwise_enabled=False, object_classes=None, class_names_dict=None,
                                     gt_depth_path=None, max_depth=80.0):
    """
    Save improved best frame visualization (3×3 grid with Valid/Object Mask and Depth Distribution)

    Layout:
        Row 0: Input Image | GT Depth | Pred Depth
        Row 1: Valid/Object Mask | Error Map | Depth Metrics (with Dataset info)
        Row 2: Depth Distribution (2 cols) | Empty

    Args:
        image: [3, H, W] or [H, W, 3] - RGB image
        gt_depth: [H, W] - Ground truth metric depth (used for error/metrics calculation)
        pred_depth: [H, W] - Predicted metric depth
        metrics: dict - Pre-computed metrics dictionary
        save_dir: Path - Save directory
        sequence_id: int - Sequence index
        frame_idx: int - Frame index
        dataset_name: str - Optional dataset name (e.g., 'eth3d/pipes')
        focal_length: float - Optional focal length
        seg_mask: [H, W] - Optional segmentation mask for object-wise visualization
        objwise_enabled: bool - Whether to show object mask instead of valid mask
        gt_depth_path: str - Optional path to GT depth file for completed depth loading (ETH3D/Waymo)
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

    MAX_DEPTH = max_depth

    # ==================== Row 0: Input, GT, Pred ====================

    # Row 0, Col 0: Input Image
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image)
    ax_img.set_title('Input Image', fontsize=12, fontweight='bold', pad=8)
    ax_img.axis('off')

    # Row 0, Col 1: GT Depth (use completed depth for ETH3D/Waymo visualization)
    ax_gt = fig.add_subplot(gs[0, 1])
    gt_valid = (gt_depth > 0) & (gt_depth < MAX_DEPTH)  # Keep original for error calculation
    pred_valid = (pred_depth > 0) & (pred_depth < MAX_DEPTH)  # Valid prediction mask

    # Try to load completed depth for visualization
    gt_for_display = gt_depth  # Default to sparse GT
    gt_display_title = 'Ground Truth Depth (m)'

    # Extract base dataset name (e.g., 'eth3d' from 'eth3d/pipes')
    base_dataset = dataset_name.split('/')[0] if dataset_name else None

    if gt_depth_path is not None and base_dataset in ['eth3d', 'waymo_seg']:
        try:
            H, W = gt_depth.shape
            completed = load_completed_depth(gt_depth_path, base_dataset, target_size=(H, W))
            if completed is not None:
                gt_for_display = completed.numpy()
                gt_display_title = 'GT Depth (completed)'
        except Exception as e:
            logger.debug(f"Could not load completed depth: {e}")

    # Create display with handling for Waymo's no-LiDAR regions (-1)
    if base_dataset == 'waymo_seg':
        valid_for_display = (gt_for_display > 0) & (gt_for_display < MAX_DEPTH)
        gt_display = np.where(valid_for_display, gt_for_display, np.nan)
    else:
        gt_display = np.where((gt_for_display > 0) & (gt_for_display < MAX_DEPTH), gt_for_display, np.nan)

    if gt_valid.sum() > 0:
        # Use original sparse GT for vmin/vmax calculation
        gt_sparse_display = np.where(gt_valid, gt_depth, np.nan)
        gt_vmin = np.nanpercentile(gt_sparse_display, 2)
        gt_vmax = np.nanpercentile(gt_sparse_display, 98)
    else:
        gt_vmin, gt_vmax = 0, 1

    cmap_gt = plt.cm.plasma_r.copy()
    cmap_gt.set_bad(color='black')
    im_gt = ax_gt.imshow(gt_display, cmap=cmap_gt, vmin=gt_vmin, vmax=gt_vmax)
    ax_gt.set_title(gt_display_title, fontsize=12, fontweight='bold', pad=8)
    ax_gt.axis('off')
    plt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)

    # Row 0, Col 2: Predicted Depth
    # Use same sparse/dense logic as sequence visualization
    ax_pred = fig.add_subplot(gs[0, 2])

    gt_density = gt_valid.sum() / gt_valid.size
    is_sparse = gt_density < 0.5

    if is_sparse:
        # Sparse: height mask + dense prediction within range
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
        pred_show_mask = height_mask & pred_valid_depth
    else:
        # Dense: use GT valid mask
        pred_show_mask = gt_valid

    pred_display = np.where(pred_show_mask, pred_depth, np.nan)

    cmap_pred = plt.cm.plasma_r.copy()
    cmap_pred.set_bad(color='black')
    im_pred = ax_pred.imshow(pred_display, cmap=cmap_pred, vmin=gt_vmin, vmax=gt_vmax)
    ax_pred.set_title('Predicted Depth (m)', fontsize=12, fontweight='bold', pad=8)
    ax_pred.axis('off')
    plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)

    # ==================== Row 1: Valid/Object Mask, Error Map, Metrics ====================

    # Row 1, Col 0: Valid Mask or Object Mask (depending on objwise mode)
    ax_valid = fig.add_subplot(gs[1, 0])

    if objwise_enabled and seg_mask is not None:
        # Object-wise mode: show Object Mask (only dynamic objects from object_classes)
        object_mask = np.zeros_like(seg_mask, dtype=np.uint8)

        if object_classes is not None and class_names_dict is not None:
            # Get class IDs for object classes
            object_class_ids = []
            for class_id, class_name in class_names_dict.items():
                if class_name in object_classes:
                    object_class_ids.append(class_id)

            # Create object mask (white for objects, black for background/non-objects)
            for class_id in object_class_ids:
                object_mask |= (seg_mask == class_id).astype(np.uint8)
        else:
            # Fallback: use all non-zero classes
            object_mask = (seg_mask > 0).astype(np.uint8)

        object_ratio = object_mask.sum() / object_mask.size
        ax_valid.imshow(object_mask, cmap='gray', vmin=0, vmax=1)
        ax_valid.set_title(f'Object Mask\n{object_ratio*100:.1f}% ({object_mask.sum():,} pixels)',
                          fontsize=12, fontweight='bold', pad=8)
    else:
        # Regular mode: show GT Valid Mask (valid=white, invalid=black)
        valid_mask_vis = gt_valid.astype(np.uint8)
        gt_valid_ratio = gt_valid.sum() / gt_valid.size
        ax_valid.imshow(valid_mask_vis, cmap='gray', vmin=0, vmax=1)
        ax_valid.set_title(f'Valid Mask (GT ≤{MAX_DEPTH:.0f}m)\n{gt_valid_ratio*100:.1f}% valid',
                          fontsize=12, fontweight='bold', pad=8)

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
        ax_dist.set_title(f'Depth Distribution (Valid Pixels ≤{MAX_DEPTH:.0f}m)', fontsize=12, fontweight='bold')
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
