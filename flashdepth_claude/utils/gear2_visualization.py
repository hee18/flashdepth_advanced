import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
import cv2
from PIL import Image
import logging
from .metric_depth_metrics import MetricDepthMetrics
from .sparse_depth_visualization import create_sparse_depth_vis_no_inpaint


class Gear2Visualizer:
    """
    Visualization utilities for Gear2 training (Ablation: No FG/BG separation)
    """

    # Sparse depth datasets that need inpainting for visualization
    SPARSE_DATASETS = ['waymo', 'waymo_seg', 'nuscenes']

    def __init__(self, save_dir="./visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logging.getLogger(__name__)

        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")

    def create_validation_summary(self, sample_batch, model_outputs, step, save_name=None, prefix="validation", fps=None, loss_dict=None, dataset_name=None):
        """
        Create a comprehensive validation summary for Gear2

        Layout (4 rows × 3 columns) - UNIFIED WITH GEAR3 UPGRADE:
        Row 1: Input, GT Depth, Pred Depth
        Row 2: N/A (No Importance), N/A (No FG Mask), N/A (No BG Mask)
        Row 3: Valid Mask, Error Map, Metrics & Training Info
        Row 4: Depth Distribution (colspan=2), Empty (colspan=1)

        Args:
            sample_batch: Validation batch (images, gt_depth, dataset_idx)
            model_outputs: Dictionary with 'pred_depth' and 'importance_map' (importance_map will be None for Gear2)
            step: Training step number
            save_name: Optional custom save name
            prefix: Prefix for the visualization
            fps: Forward pass FPS (optional)
            loss_dict: Dictionary with loss values (optional)
                - 'depth_loss': Depth loss value
            dataset_name: Name of the dataset (for sparse depth detection)

        Returns:
            fig: Matplotlib figure object
        """
        try:
            images, gt_depth, dataset_idx = sample_batch
            pred_depth = model_outputs['pred_depth']
            importance_map = model_outputs.get('importance_map', None)  # Will be None for Gear2

            # Debug shapes
            if not hasattr(images, 'cpu'):
                images = torch.from_numpy(images) if isinstance(images, np.ndarray) else images
            if not hasattr(gt_depth, 'cpu'):
                gt_depth = torch.from_numpy(gt_depth) if isinstance(gt_depth, np.ndarray) else gt_depth
            if not hasattr(pred_depth, 'cpu'):
                pred_depth = torch.from_numpy(pred_depth) if isinstance(pred_depth, np.ndarray) else pred_depth

            # Use first batch and first frame for visualization
            # Handle both [B, T, ...] and [B, ...] formats
            if images.ndim == 5:  # [B, T, C, H, W]
                input_img = images[0, 0].cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
            else:  # [B, C, H, W]
                input_img = images[0].cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]

            if gt_depth.ndim == 4:  # [B, T, H, W] or [B, 1, H, W]
                gt_depth_frame = gt_depth[0, 0].cpu().numpy()  # [H, W]
            else:  # [B, H, W]
                gt_depth_frame = gt_depth[0].cpu().numpy()  # [H, W]

            if pred_depth.ndim == 4:  # [B, T, H, W] or [B, 1, H, W]
                pred_depth_frame = pred_depth[0, 0].cpu().numpy()  # [H, W]
            else:  # [B, H, W]
                pred_depth_frame = pred_depth[0].cpu().numpy()  # [H, W]

            # Normalize input image for display
            input_img = np.clip((input_img + 1) / 2, 0, 1)  # Assuming normalized input

            # Ensure all frames are 2D
            while gt_depth_frame.ndim > 2:
                gt_depth_frame = gt_depth_frame[0]
            while pred_depth_frame.ndim > 2:
                pred_depth_frame = pred_depth_frame[0]

            # Create two separate masks:
            # 1. Metrics mask: 70m threshold (same as training loss)
            # 2. Visualization mask: No upper limit (show all depths)
            MAX_DEPTH_METRICS = 70.0  # For metrics calculation (100/70 = 1.43 inverse depth threshold)

            # Metrics mask: 70m threshold for fair comparison with training
            gt_valid_metrics = (gt_depth_frame > 0) & (gt_depth_frame < MAX_DEPTH_METRICS)
            pred_valid_metrics = (pred_depth_frame > 0) & (pred_depth_frame < MAX_DEPTH_METRICS)
            valid_mask_metrics = gt_valid_metrics & pred_valid_metrics

            # Visualization mask: Filter out invalid values (<=0) and extreme outliers (>1000m)
            MAX_DEPTH_VIS = 1000.0  # Same as TartanAir's maximum valid depth
            gt_valid_vis = (gt_depth_frame > 0) & (gt_depth_frame < MAX_DEPTH_VIS)
            pred_valid_vis = (pred_depth_frame > 0) & (pred_depth_frame < MAX_DEPTH_VIS)
            valid_mask_vis = gt_valid_vis & pred_valid_vis

            if valid_mask_metrics.sum() > 0:
                gt_valid = gt_depth_frame[valid_mask_metrics]
                pred_valid = pred_depth_frame[valid_mask_metrics]

            # Create figure with subplots - NEW LAYOUT: 4 rows × 3 columns
            fig = plt.figure(figsize=(15, 16))
            gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

            # Calculate valid ratio and error BEFORE visualization
            # Use metrics mask for statistics (70m threshold)
            num_valid_metrics = valid_mask_metrics.sum()
            valid_ratio_metrics = num_valid_metrics / valid_mask_metrics.size
            valid_ratio_vis = valid_mask_vis.sum() / valid_mask_vis.size
            abs_error = np.abs(pred_depth_frame - gt_depth_frame)
            abs_error_masked = np.where(valid_mask_metrics, abs_error, np.nan)  # Metrics use 70m threshold

            # Check if we have valid pixels for metrics calculation
            has_valid_pixels = num_valid_metrics > 0

            # ==================== Row 1: Input, GT, Pred ====================

            # 1. Input Image
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(input_img)
            ax1.set_title('Input Image', fontsize=14, fontweight='bold')
            ax1.axis('off')

            # 2. Ground Truth Depth (with sparse depth handling)
            ax2 = fig.add_subplot(gs[0, 1])

            # Use inpainting ONLY for sparse datasets (waymo, nuscenes)
            is_sparse_dataset = dataset_name in self.SPARSE_DATASETS if dataset_name else False

            if is_sparse_dataset:
                # Sparse depth: show valid pixels only (no inpainting)
                # Use vis mask (no 70m limit) for visualization
                _, gt_dense_vis, gt_info = create_sparse_depth_vis_no_inpaint(
                    gt_depth_frame, valid_mask_vis, colormap='plasma', percentile_range=(2, 98)
                )
                im2 = ax2.imshow(gt_dense_vis)
                ax2.set_title(f'GT Depth (Sparse)\n{valid_ratio_vis*100:.1f}% valid\n(Metrics: {valid_ratio_metrics*100:.1f}%)',
                             fontsize=12, fontweight='bold')
                vmin, vmax = gt_info['vmin'], gt_info['vmax']
            else:
                # Dense depth: use standard visualization (no 70m limit)
                gt_display = np.where(valid_mask_vis, gt_depth_frame, np.nan)
                vmin, vmax = np.nanpercentile(gt_display, [2, 98])
                im2 = ax2.imshow(gt_display, cmap='plasma', vmin=vmin, vmax=vmax)
                ax2.set_title(f'Ground Truth Depth (m)\n(Metrics: {valid_ratio_metrics*100:.1f}%)',
                             fontsize=12, fontweight='bold')

            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            # 3. Predicted Metric Depth (with sparse depth handling)
            ax3 = fig.add_subplot(gs[0, 2])

            if is_sparse_dataset:
                # Pred depth is already dense (model predicts all pixels), just visualize directly
                im3 = ax3.imshow(pred_depth_frame, cmap='plasma', vmin=vmin, vmax=vmax)
                mae_str = f'{np.nanmean(abs_error_masked):.3f}m' if has_valid_pixels else 'N/A'
                ax3.set_title(f'Pred Depth\nMAE: {mae_str}',
                             fontsize=12, fontweight='bold')
            else:
                # Dense depth: use standard visualization (no 70m limit)
                pred_display = np.where(valid_mask_vis, pred_depth_frame, np.nan)
                im3 = ax3.imshow(pred_display, cmap='plasma', vmin=vmin, vmax=vmax)
                mae_str = f'{np.nanmean(abs_error_masked):.3f}m' if has_valid_pixels else 'N/A'
                ax3.set_title(f'Predicted Metric Depth (m)\nMAE: {mae_str}',
                             fontsize=12, fontweight='bold')

            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

            # ==================== Row 2: N/A (No Importance Map for Gear2) ====================

            # 4. Importance Map → N/A for Gear2
            ax4 = fig.add_subplot(gs[1, 0])
            ax4.text(0.5, 0.5, 'No Importance Map\n\nUniform Modulation',
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=16, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax4.set_title('Importance Map (N/A)', fontsize=14, fontweight='bold')
            ax4.axis('off')

            # 5. FG Mask → N/A for Gear2
            ax5 = fig.add_subplot(gs[1, 1])
            ax5.text(0.5, 0.5, 'No FG/BG Separation\n\n(Gear2: Uniform)',
                    ha='center', va='center', transform=ax5.transAxes,
                    fontsize=16, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax5.set_title('FG Mask (N/A)', fontsize=14, fontweight='bold')
            ax5.axis('off')

            # 6. BG Mask → N/A for Gear2
            ax6 = fig.add_subplot(gs[1, 2])
            ax6.text(0.5, 0.5, 'No FG/BG Separation\n\n(Gear2: Uniform)',
                    ha='center', va='center', transform=ax6.transAxes,
                    fontsize=16, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax6.set_title('BG Mask (N/A)', fontsize=14, fontweight='bold')
            ax6.axis('off')

            # ==================== Row 3: Valid Mask, Error, Metrics & Training Info ====================

            # 7. Valid Mask (for metrics calculation, 70m threshold)
            ax7 = fig.add_subplot(gs[2, 0])
            ax7.imshow(valid_mask_metrics.astype(np.uint8), cmap='gray_r', vmin=0, vmax=1)
            ax7.set_title(f'Valid Mask\n({valid_mask_metrics.sum():,} pixels)',
                         fontsize=12, fontweight='bold')
            ax7.axis('off')

            # 8. Absolute Error Map
            ax8 = fig.add_subplot(gs[2, 1])
            if has_valid_pixels:
                error_vmax = np.nanpercentile(abs_error_masked, 95)
                im8 = ax8.imshow(abs_error_masked, cmap='hot', vmin=0, vmax=error_vmax)
                ax8.set_title(f'Absolute Error (m)\nMean: {np.nanmean(abs_error_masked):.3f}',
                             fontsize=14, fontweight='bold')
                plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
            else:
                # No valid pixels - show placeholder
                ax8.text(0.5, 0.5, 'No Valid Pixels\n(All depths > 70m or invalid)',
                        ha='center', va='center', transform=ax8.transAxes,
                        fontsize=12, color='red', fontweight='bold')
                ax8.set_title('Absolute Error (m)\nN/A', fontsize=14, fontweight='bold')
            ax8.axis('off')

            # 9. Depth Metrics & Training Info (COMBINED)
            ax9 = fig.add_subplot(gs[2, 2])

            y_pos = 0.95
            # Step info
            ax9.text(0.05, y_pos, f'Step: {step}', fontsize=11,
                    transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'),
                    fontweight='bold')
            y_pos -= 0.12

            # Dataset info
            if isinstance(dataset_idx, str):
                dataset_str = dataset_idx
            elif isinstance(dataset_idx, (list, tuple)):
                dataset_str = str(dataset_idx[0])
            elif torch.is_tensor(dataset_idx):
                dataset_str = str(dataset_idx[0].item() if dataset_idx.dim() > 0 else dataset_idx.item())
            else:
                dataset_str = str(dataset_idx)
            ax9.text(0.05, y_pos, f'Dataset: {dataset_str}', fontsize=11, transform=ax9.transAxes)
            y_pos -= 0.10

            # FG:BG ratio → N/A for Gear2
            ax9.text(0.05, y_pos, f'FG:BG = N/A', fontsize=10,
                    transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightgray'))
            y_pos -= 0.10

            # FPS if available
            if fps is not None:
                ax9.text(0.05, y_pos, f'FPS: {fps:.1f}', fontsize=10,
                        transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
                y_pos -= 0.10

            # Depth metrics
            if valid_mask_metrics.sum() > 0:
                pred_tensor = torch.from_numpy(pred_depth_frame).float()
                gt_tensor = torch.from_numpy(gt_depth_frame).float()
                valid_tensor = torch.from_numpy(valid_mask_metrics).bool()

                metrics = MetricDepthMetrics.compute_metric_depth_metrics(
                    pred_tensor, gt_tensor, valid_tensor
                )

                ax9.text(0.05, y_pos, f'AbsRel: {metrics["abs_rel"]:.4f}', fontsize=10,
                        transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightcoral'))
                y_pos -= 0.08
                ax9.text(0.05, y_pos, f'Delta_1: {metrics["a1"]:.4f}', fontsize=10,
                        transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
                y_pos -= 0.08
                ax9.text(0.05, y_pos, f'Delta_2: {metrics["a2"]:.4f}', fontsize=10,
                        transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
                y_pos -= 0.08
                ax9.text(0.05, y_pos, f'Delta_3: {metrics["a3"]:.4f}', fontsize=10,
                        transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
                y_pos -= 0.08
                ax9.text(0.05, y_pos, f'RMSE: {metrics["rmse"]:.3f}m', fontsize=9,
                        transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
                y_pos -= 0.08
                ax9.text(0.05, y_pos, f'MAE: {metrics.get("mae", 0):.3f}m', fontsize=9,
                        transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightblue'))
                y_pos -= 0.08

            # Loss values if available
            if loss_dict is not None:
                # Validation loss (for validation visualization)
                if 'val_loss' in loss_dict:
                    ax9.text(0.05, y_pos, f'Val Loss: {loss_dict["val_loss"]:.4f}', fontsize=9,
                            transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightcoral'))
                    y_pos -= 0.08

                # Depth loss (for training visualization)
                if 'depth_loss' in loss_dict:
                    ax9.text(0.05, y_pos, f'Log L1: {loss_dict["depth_loss"]:.4f}', fontsize=9,
                            transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor='lightcoral'))
                    y_pos -= 0.08

            ax9.set_title('Depth Metrics & Training Info', fontsize=14, fontweight='bold')
            ax9.axis('off')

            # ==================== Row 4: Depth Distribution ====================

            # 10. Depth Distribution Histogram (70m threshold for metrics)
            ax10 = fig.add_subplot(gs[3, :2])
            if valid_mask_metrics.sum() > 0:
                gt_valid = gt_depth_frame[valid_mask_metrics]
                pred_valid = pred_depth_frame[valid_mask_metrics]

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

            # 11. Empty (reserved for importance distribution in Gear3/3_upgrade)
            ax11 = fig.add_subplot(gs[3, 2])
            ax11.text(0.5, 0.5, 'No Importance Map\n\n(Uniform Modulation)',
                     ha='center', va='center', transform=ax11.transAxes,
                     fontsize=14, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax11.set_title('Importance Distribution (N/A)', fontsize=14, fontweight='bold')
            ax11.axis('off')

            # Save figure
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
