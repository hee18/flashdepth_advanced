"""
Onepiece Visualization Utilities.

Layout (3 rows x 3 columns) - NO importance map / FG / BG row:
    Row 1: Input Image, GT Depth, Pred Depth
    Row 2: Valid Mask, Error Map, Metrics & Training Info
    Row 3: Depth Distribution (colspan=2), empty (colspan=1)
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import logging
from .metric_depth_metrics import MetricDepthMetrics
from .sparse_depth_visualization import create_sparse_depth_vis_no_inpaint


class OnepieceVisualizer:
    """Visualization utilities for Onepiece training (no importance map)."""

    SPARSE_DATASETS = ['waymo', 'waymo_seg', 'nuscenes']

    def __init__(self, save_dir="./visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logging.getLogger(__name__)
        plt.style.use('default')

    def create_validation_summary(self, sample_batch, model_outputs, step,
                                   save_name=None, prefix="validation",
                                   fps=None, loss_dict=None,
                                   dataset_name=None, config=None):
        """
        Create validation summary for Onepiece.

        Layout (3 rows x 3 columns):
            Row 1: Input, GT Depth, Pred Depth
            Row 2: Valid Mask, Error Map, Metrics & Training Info
            Row 3: Depth Distribution (colspan=2), Scale/Shift info (colspan=1)
        """
        try:
            if not isinstance(sample_batch, tuple) or len(sample_batch) != 5:
                raise ValueError(f"Expected 5-element tuple, got {type(sample_batch)} len={len(sample_batch) if hasattr(sample_batch, '__len__') else 'N/A'}")

            images, gt_depth, dataset_idx, fx_ratio, resize_ratio = sample_batch
            pred_depth = model_outputs['pred_depth']

            # Convert to tensors if needed
            for name, val in [('images', images), ('gt_depth', gt_depth), ('pred_depth', pred_depth)]:
                if not hasattr(val, 'cpu'):
                    if isinstance(val, np.ndarray):
                        val = torch.from_numpy(val)

            # Extract first batch, first frame
            if images.ndim == 5:
                input_img = images[0, 0].float().cpu().numpy().transpose(1, 2, 0)
            else:
                input_img = images[0].float().cpu().numpy().transpose(1, 2, 0)

            input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min() + 1e-8)
            input_img = np.clip(input_img, 0, 1)

            if gt_depth.ndim == 4:
                gt_depth_frame = gt_depth[0, 0].float().cpu().numpy()
            else:
                gt_depth_frame = gt_depth[0].float().cpu().numpy()

            if pred_depth.ndim == 4:
                pred_depth_frame = pred_depth[0, 0].float().cpu().numpy()
            else:
                pred_depth_frame = pred_depth[0].float().cpu().numpy()

            while gt_depth_frame.ndim > 2:
                gt_depth_frame = gt_depth_frame[0]
            while pred_depth_frame.ndim > 2:
                pred_depth_frame = pred_depth_frame[0]

            # Create valid masks
            if 'canonical_gt_valid' in model_outputs:
                canonical_gt_valid = model_outputs['canonical_gt_valid'][0, 0].bool().cpu().numpy()
                canonical_pred_valid = model_outputs['canonical_pred_valid'][0, 0].bool().cpu().numpy()

                MAX_DEPTH_OUTLIER = 200.0
                pred_outlier_mask = (pred_depth_frame > 0) & (pred_depth_frame < MAX_DEPTH_OUTLIER)
                valid_mask_metrics = canonical_gt_valid & pred_outlier_mask
                valid_mask_gt_vis = canonical_gt_valid

                gt_exists = (gt_depth_frame > 0)
                gt_density = gt_exists.sum() / gt_exists.size
                is_sparse = gt_density < 0.5

                if is_sparse:
                    valid_pixels_per_row = gt_exists.sum(axis=1)
                    valid_rows = valid_pixels_per_row >= 10
                    valid_row_indices = np.where(valid_rows)[0]
                    if len(valid_row_indices) > 0:
                        height_mask = np.zeros_like(gt_depth_frame, dtype=bool)
                        height_mask[valid_row_indices.min():valid_row_indices.max()+1, :] = True
                    else:
                        height_mask = np.ones_like(gt_depth_frame, dtype=bool)
                    gt_missing = ~gt_exists
                    valid_mask_pred_vis = height_mask & (canonical_gt_valid | (gt_missing & canonical_pred_valid))
                else:
                    valid_mask_pred_vis = canonical_gt_valid

                valid_mask_vis = canonical_gt_valid
            else:
                MAX_DEPTH = 70.0
                MAX_DEPTH_OUTLIER = 200.0
                gt_valid = (gt_depth_frame > 0) & (gt_depth_frame < MAX_DEPTH)
                pred_valid = (pred_depth_frame > 0) & (pred_depth_frame < MAX_DEPTH)
                pred_outlier_mask = (pred_depth_frame > 0) & (pred_depth_frame < MAX_DEPTH_OUTLIER)
                valid_mask_metrics = gt_valid & pred_outlier_mask
                valid_mask_gt_vis = gt_valid

                gt_exists = (gt_depth_frame > 0)
                gt_density = gt_exists.sum() / gt_exists.size
                is_sparse = gt_density < 0.5

                if is_sparse:
                    valid_pixels_per_row = gt_exists.sum(axis=1)
                    valid_rows = valid_pixels_per_row >= 10
                    valid_row_indices = np.where(valid_rows)[0]
                    if len(valid_row_indices) > 0:
                        height_mask = np.zeros_like(gt_depth_frame, dtype=bool)
                        height_mask[valid_row_indices.min():valid_row_indices.max()+1, :] = True
                    else:
                        height_mask = np.ones_like(gt_depth_frame, dtype=bool)
                    gt_missing = ~gt_exists
                    valid_mask_pred_vis = height_mask & (gt_valid | (gt_missing & pred_valid))
                else:
                    valid_mask_pred_vis = gt_valid

                valid_mask_vis = gt_valid

            # Create figure: 3 rows x 3 columns (no importance row)
            fig = plt.figure(figsize=(15, 12))
            gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

            num_valid_metrics = valid_mask_metrics.sum()
            abs_error = np.abs(pred_depth_frame - gt_depth_frame)
            abs_error_masked = np.where(valid_mask_metrics, abs_error, np.nan)
            has_valid_pixels = num_valid_metrics > 0

            is_sparse_dataset = dataset_name in self.SPARSE_DATASETS if dataset_name else False

            # ========== Row 1: Input, GT, Pred ==========

            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(input_img)
            ax1.set_title('Input Image', fontsize=14, fontweight='bold')
            ax1.axis('off')

            ax2 = fig.add_subplot(gs[0, 1])
            if is_sparse_dataset:
                _, gt_dense_vis, gt_info = create_sparse_depth_vis_no_inpaint(
                    gt_depth_frame, valid_mask_gt_vis, colormap='plasma_r', percentile_range=(2, 98)
                )
                vmin, vmax = gt_info['vmin'], gt_info['vmax']
                cmap_gt = plt.cm.plasma_r.copy()
                cmap_gt.set_bad(color='black')
                gt_display_capped = np.where(valid_mask_gt_vis, gt_depth_frame, np.nan)
                im2 = ax2.imshow(gt_display_capped, cmap=cmap_gt, vmin=vmin, vmax=vmax)
                valid_ratio_gt = valid_mask_gt_vis.sum() / valid_mask_gt_vis.size
                ax2.set_title(f'GT Depth (Sparse)\n{valid_ratio_gt*100:.1f}% valid',
                             fontsize=14, fontweight='bold')
            else:
                gt_display = np.where(valid_mask_gt_vis, gt_depth_frame, np.nan)
                if valid_mask_gt_vis.sum() > 0:
                    vmin = np.nanpercentile(gt_display, 2)
                    vmax = np.nanpercentile(gt_display, 98)
                else:
                    vmin, vmax = 0, 1
                cmap = plt.cm.plasma_r.copy()
                cmap.set_bad(color='black')
                im2 = ax2.imshow(gt_display, cmap=cmap, vmin=vmin, vmax=vmax)
                ax2.set_title('Ground Truth Depth (m)', fontsize=14, fontweight='bold')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            ax3 = fig.add_subplot(gs[0, 2])
            pred_display = np.where(valid_mask_pred_vis, pred_depth_frame, np.nan)
            cmap_pred = plt.cm.plasma_r.copy()
            cmap_pred.set_bad(color='black')
            im3 = ax3.imshow(pred_display, cmap=cmap_pred, vmin=vmin, vmax=vmax)
            if is_sparse_dataset:
                mae_str = f'{np.nanmean(abs_error_masked):.3f}m' if has_valid_pixels else 'N/A'
                ax3.set_title(f'Pred Depth\nMAE: {mae_str}', fontsize=14, fontweight='bold')
            else:
                ax3.set_title('Predicted Metric Depth (m)', fontsize=14, fontweight='bold')
            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

            # ========== Row 2: Valid Mask, Error, Metrics ==========

            ax4 = fig.add_subplot(gs[1, 0])
            ax4.imshow(valid_mask_vis.astype(np.uint8), cmap='gray', vmin=0, vmax=1)
            valid_ratio_pct = (valid_mask_vis.sum() / valid_mask_vis.size) * 100
            ax4.set_title(f'Valid Mask ({valid_ratio_pct:.1f}%)\nGT valid only',
                         fontsize=14, fontweight='bold')
            ax4.axis('off')

            ax5 = fig.add_subplot(gs[1, 1])
            if has_valid_pixels:
                error_vmax = np.nanpercentile(abs_error_masked, 95)
                im5 = ax5.imshow(abs_error_masked, cmap='hot', vmin=0, vmax=error_vmax)
                ax5.set_title(f'Absolute Error (m)\nMean: {np.nanmean(abs_error_masked):.3f}',
                             fontsize=14, fontweight='bold')
                plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
            else:
                ax5.text(0.5, 0.5, 'No Valid Pixels',
                        ha='center', va='center', transform=ax5.transAxes,
                        fontsize=12, color='red', fontweight='bold')
                ax5.set_title('Absolute Error (m)\nN/A', fontsize=14, fontweight='bold')
            ax5.axis('off')

            ax6 = fig.add_subplot(gs[1, 2])
            y_pos = 0.95

            # Dataset + Step
            if isinstance(dataset_idx, str):
                dataset_str = dataset_idx
            elif isinstance(dataset_idx, (list, tuple)):
                dataset_str = str(dataset_idx[0])
            elif torch.is_tensor(dataset_idx):
                dataset_str = str(dataset_idx[0].item() if dataset_idx.dim() > 0 else dataset_idx.item())
            else:
                dataset_str = str(dataset_idx)
            ax6.text(0.05, y_pos, f'Dataset: {dataset_str} | Step: {step}', fontsize=11,
                    transform=ax6.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
            y_pos -= 0.10

            # Scale/Shift
            if 'scale' in model_outputs and 'shift' in model_outputs:
                scale = model_outputs['scale']
                shift = model_outputs['shift']
                if torch.is_tensor(scale):
                    scale_val = scale[0, 0].item() if scale.ndim >= 2 else scale[0].item()
                else:
                    scale_val = float(scale)
                if torch.is_tensor(shift):
                    shift_val = shift[0, 0].item() if shift.ndim >= 2 else shift[0].item()
                else:
                    shift_val = float(shift)
                ax6.text(0.05, y_pos, f'scale: {scale_val:.3f}, shift: {shift_val:.3f}',
                        fontsize=10, transform=ax6.transAxes,
                        bbox=dict(boxstyle="round", facecolor='wheat'))
                y_pos -= 0.10

            # fx_ratio, resize_ratio
            if fx_ratio is not None and resize_ratio is not None:
                fx_ratio_value = fx_ratio[0, 0].item() if fx_ratio.ndim >= 2 else fx_ratio[0].item()
                resize_ratio_value = resize_ratio[0, 0].item() if resize_ratio.ndim >= 2 else resize_ratio[0].item()
                ax6.text(0.05, y_pos, f'fx_ratio: {fx_ratio_value:.3f} | resize_ratio: {resize_ratio_value:.3f}',
                        fontsize=10, transform=ax6.transAxes,
                        bbox=dict(boxstyle="round", facecolor='wheat'))
                y_pos -= 0.10

            if fps is not None:
                ax6.text(0.05, y_pos, f'FPS: {fps:.1f}', fontsize=10,
                        transform=ax6.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
                y_pos -= 0.10

            # Depth metrics
            if valid_mask_metrics.sum() > 0:
                pred_tensor = torch.from_numpy(pred_depth_frame).float()
                gt_tensor = torch.from_numpy(gt_depth_frame).float()
                valid_tensor = torch.from_numpy(valid_mask_metrics).bool()
                metrics = MetricDepthMetrics.compute_metric_depth_metrics(
                    pred_tensor, gt_tensor, valid_tensor
                )
                ax6.text(0.05, y_pos, f'AbsRel: {metrics["abs_rel"]:.4f}', fontsize=10,
                        transform=ax6.transAxes, bbox=dict(boxstyle="round", facecolor='lightcoral'))
                y_pos -= 0.08
                ax6.text(0.05, y_pos, f'Delta_1: {metrics["a1"]:.4f}', fontsize=10,
                        transform=ax6.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
                y_pos -= 0.08
                ax6.text(0.05, y_pos, f'Delta_2: {metrics["a2"]:.4f}', fontsize=10,
                        transform=ax6.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
                y_pos -= 0.08
                ax6.text(0.05, y_pos, f'Delta_3: {metrics["a3"]:.4f}', fontsize=10,
                        transform=ax6.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
                y_pos -= 0.08
                ax6.text(0.05, y_pos, f'RMSE: {metrics["rmse"]:.3f}m', fontsize=9,
                        transform=ax6.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
                y_pos -= 0.08
                ax6.text(0.05, y_pos, f'MAE: {metrics.get("mae", 0):.3f}m', fontsize=9,
                        transform=ax6.transAxes, bbox=dict(boxstyle="round", facecolor='lightblue'))
                y_pos -= 0.08

            # Loss values
            if loss_dict is not None:
                if 'val_loss' in loss_dict:
                    ax6.text(0.05, y_pos, f'Val Loss: {loss_dict["val_loss"]:.4f}', fontsize=9,
                            transform=ax6.transAxes, bbox=dict(boxstyle="round", facecolor='lightcoral'))
                    y_pos -= 0.08
                if 'loss' in loss_dict and 'val_loss' not in loss_dict:
                    ax6.text(0.05, y_pos, f'Total Loss: {loss_dict["loss"]:.4f}', fontsize=9,
                            transform=ax6.transAxes, bbox=dict(boxstyle="round", facecolor='lightcoral'))
                    y_pos -= 0.08
                # Support both key names: 'depth_loss' (validation) and 'log_l1_loss' (training)
                log_l1_val = loss_dict.get('depth_loss', loss_dict.get('log_l1_loss', None))
                if log_l1_val is not None:
                    ax6.text(0.05, y_pos, f'Log L1: {log_l1_val:.4f}', fontsize=9,
                            transform=ax6.transAxes, bbox=dict(boxstyle="round", facecolor='lightsalmon'))
                    y_pos -= 0.08
                if 'tgm_loss' in loss_dict and loss_dict['tgm_loss'] > 0:
                    ax6.text(0.05, y_pos, f'TGM: {loss_dict["tgm_loss"]:.4f}', fontsize=9,
                            transform=ax6.transAxes, bbox=dict(boxstyle="round", facecolor='lightskyblue'))
                    y_pos -= 0.08
                if 'feat_cons_loss' in loss_dict and loss_dict.get('feat_cons_loss', 0) > 0:
                    ax6.text(0.05, y_pos, f'FeatCons: {loss_dict["feat_cons_loss"]:.4f}', fontsize=9,
                            transform=ax6.transAxes, bbox=dict(boxstyle="round", facecolor='lightyellow'))
                    y_pos -= 0.08

            ax6.set_title('Depth Metrics & Training Info', fontsize=14, fontweight='bold')
            ax6.axis('off')

            # ========== Row 3: Depth Distribution ==========

            ax7 = fig.add_subplot(gs[2, :2])
            if valid_mask_metrics.sum() > 0:
                gt_valid_vals = gt_depth_frame[valid_mask_metrics]
                pred_valid_vals = pred_depth_frame[valid_mask_metrics]
                bins = np.linspace(min(gt_valid_vals.min(), pred_valid_vals.min()),
                                  max(gt_valid_vals.max(), pred_valid_vals.max()), 50)
                ax7.hist(gt_valid_vals, bins=bins, alpha=0.6, label='Ground Truth',
                        color='blue', density=True)
                ax7.hist(pred_valid_vals, bins=bins, alpha=0.6, label='Predicted',
                        color='red', density=True)
                ax7.set_xlabel('Depth (meters)', fontsize=12)
                ax7.set_ylabel('Density', fontsize=12)
                ax7.set_title('Depth Distribution', fontsize=14, fontweight='bold')
                ax7.legend(fontsize=12)
                ax7.grid(True, alpha=0.3)

            # Scale/Shift summary in the remaining cell
            ax8 = fig.add_subplot(gs[2, 2])
            ax8.text(0.5, 0.5, 'Onepiece\n(No Importance Map)',
                    ha='center', va='center', transform=ax8.transAxes,
                    fontsize=12, fontweight='bold', color='gray')
            ax8.axis('off')

            # Save
            if save_name:
                save_path = self.save_dir / f"{save_name}.png"
            else:
                save_path = self.save_dir / f"{prefix}_step_{step:06d}.png"

            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved visualization to {save_path}")
            plt.close(fig)
            return fig

        except Exception as e:
            self.logger.error(f"Failed to create validation summary: {e}")
            import traceback
            traceback.print_exc()
            if 'fig' in locals():
                plt.close(fig)
            return None
