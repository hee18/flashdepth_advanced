import torch
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
from .sparse_depth_visualization import create_dual_sparse_depth_vis


class Gear2Visualizer:
    """
    Visualization utilities for Gear2 training (Ablation: No FG/BG separation)
    """

    def __init__(self, save_dir="./visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logging.getLogger(__name__)

        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")

    def create_validation_summary(self, sample_batch, model_outputs, step, save_name=None, prefix="validation", fps=None, loss_dict=None):
        """
        Create a comprehensive validation summary for Gear2

        Args:
            sample_batch: Validation batch (images, gt_depth, dataset_idx)
            model_outputs: Dictionary with 'pred_depth' and 'importance_map' (importance_map will be None for Gear2)
            step: Training step number
            save_name: Optional custom save name
            prefix: Prefix for the visualization
            fps: Forward pass FPS (optional)
            loss_dict: Dictionary with loss values (optional)
                - 'depth_loss': Depth loss value

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

            # Create valid mask
            # Filter out invalid depths: GT and pred must be in [0, 200m] range
            MAX_DEPTH = 200.0  # Maximum valid depth in meters
            gt_valid_mask = (gt_depth_frame > 0) & (gt_depth_frame < MAX_DEPTH)
            pred_valid_mask = (pred_depth_frame > 0) & (pred_depth_frame < MAX_DEPTH)
            valid_mask = gt_valid_mask & pred_valid_mask

            if valid_mask.sum() > 0:
                gt_valid = gt_depth_frame[valid_mask]
                pred_valid = pred_depth_frame[valid_mask]

            # Create figure with subplots
            fig = plt.figure(figsize=(20, 12))
            gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

            # Calculate valid ratio and error BEFORE visualization
            valid_ratio = valid_mask.sum() / valid_mask.size
            abs_error = np.abs(pred_depth_frame - gt_depth_frame)
            abs_error_masked = np.where(valid_mask, abs_error, np.nan)

            # 1. Input Image
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(input_img)
            ax1.set_title('Input Image', fontsize=14, fontweight='bold')
            ax1.axis('off')

            # 2. Ground Truth Depth (with sparse depth handling)
            ax2 = fig.add_subplot(gs[0, 1])

            # Use enhanced sparse visualization if valid_ratio < 50% (e.g., Waymo LiDAR)
            if valid_ratio < 0.5:
                # Sparse depth: use dual visualization (inpainted)
                _, gt_dense_vis, gt_info = create_dual_sparse_depth_vis(
                    gt_depth_frame, valid_mask, colormap='plasma', percentile_range=(2, 98)
                )
                im2 = ax2.imshow(gt_dense_vis)
                ax2.set_title(f'GT Depth (Inpainted)\n{valid_ratio*100:.1f}% valid',
                             fontsize=14, fontweight='bold')
                vmin, vmax = gt_info['vmin'], gt_info['vmax']
            else:
                # Dense depth: use standard visualization
                gt_display = np.where(valid_mask, gt_depth_frame, np.nan)
                vmin, vmax = np.nanpercentile(gt_display, [2, 98])
                im2 = ax2.imshow(gt_display, cmap='plasma', vmin=vmin, vmax=vmax)
                ax2.set_title('Ground Truth Depth (m)', fontsize=14, fontweight='bold')

            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            # 3. Predicted Metric Depth (with sparse depth handling)
            ax3 = fig.add_subplot(gs[0, 2])

            if valid_ratio < 0.5:
                # Sparse depth: use dual visualization (inpainted)
                _, pred_dense_vis, pred_info = create_dual_sparse_depth_vis(
                    pred_depth_frame, valid_mask, colormap='plasma', percentile_range=(2, 98)
                )
                im3 = ax3.imshow(pred_dense_vis)
                ax3.set_title(f'Pred Depth (Inpainted)\nMAE: {np.nanmean(abs_error_masked):.2f}m',
                             fontsize=14, fontweight='bold')
            else:
                # Dense depth: use standard visualization
                pred_display = np.where(valid_mask, pred_depth_frame, np.nan)
                im3 = ax3.imshow(pred_display, cmap='plasma', vmin=vmin, vmax=vmax)
                ax3.set_title('Predicted Metric Depth (m)', fontsize=14, fontweight='bold')

            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

            # 4. Importance Map → N/A for Gear2
            ax4 = fig.add_subplot(gs[0, 3])
            ax4.text(0.5, 0.5, 'No Importance Map\n\nUniform Modulation',
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=16, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax4.set_title('Importance Map (N/A)', fontsize=14, fontweight='bold')
            ax4.axis('off')

            # 5. Valid Mask
            ax5 = fig.add_subplot(gs[1, 0])
            ax5.imshow(valid_mask.astype(np.uint8), cmap='gray_r', vmin=0, vmax=1)
            ax5.set_title(f'Valid Mask\n({valid_mask.sum():,} pixels)', fontsize=14, fontweight='bold')
            ax5.axis('off')

            # 6. Absolute Error Map
            ax6 = fig.add_subplot(gs[1, 1])
            # abs_error and abs_error_masked already computed above
            error_vmax = np.nanpercentile(abs_error_masked, 95)
            im6 = ax6.imshow(abs_error_masked, cmap='hot', vmin=0, vmax=error_vmax)
            ax6.set_title(f'Absolute Error (m)\nMean: {np.nanmean(abs_error_masked):.3f}',
                         fontsize=14, fontweight='bold')
            ax6.axis('off')
            plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

            # 7. Metric Evaluation
            ax7 = fig.add_subplot(gs[1, 2])
            if valid_mask.sum() > 0:
                pred_tensor = torch.from_numpy(pred_depth_frame).float()
                gt_tensor = torch.from_numpy(gt_depth_frame).float()
                valid_tensor = torch.from_numpy(valid_mask).bool()

                metrics = MetricDepthMetrics.compute_metric_depth_metrics(
                    pred_tensor, gt_tensor, valid_tensor
                )

                ax7.text(0.1, 0.85, f'AbsRel: {metrics["abs_rel"]:.4f}', fontsize=14,
                        transform=ax7.transAxes, bbox=dict(boxstyle="round", facecolor='lightcoral'))
                ax7.text(0.1, 0.7, f'Delta_1: {metrics["a1"]:.4f}', fontsize=14,
                        transform=ax7.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
                ax7.text(0.1, 0.55, f'Delta_2: {metrics["a2"]:.4f}', fontsize=14,
                        transform=ax7.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
                ax7.text(0.1, 0.4, f'Delta_3: {metrics["a3"]:.4f}', fontsize=14,
                        transform=ax7.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
                ax7.text(0.1, 0.25, f'RMSE: {metrics["rmse"]:.3f}m', fontsize=12,
                        transform=ax7.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
                ax7.text(0.1, 0.1, f'MAE: {metrics.get("mae", 0):.3f}m', fontsize=12,
                        transform=ax7.transAxes, bbox=dict(boxstyle="round", facecolor='lightblue'))
            ax7.set_title('Depth Metrics', fontsize=14, fontweight='bold')
            ax7.axis('off')

            # 8. Training Info
            ax8 = fig.add_subplot(gs[1, 3])
            y_pos = 0.9  # Start from top

            ax8.text(0.1, y_pos, f'Step: {step}', fontsize=16,
                    transform=ax8.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
            y_pos -= 0.15

            # Handle dataset_idx (can be string or tensor)
            if isinstance(dataset_idx, str):
                dataset_str = dataset_idx
            elif isinstance(dataset_idx, (list, tuple)):
                dataset_str = str(dataset_idx[0])
            elif torch.is_tensor(dataset_idx):
                dataset_str = str(dataset_idx[0].item() if dataset_idx.dim() > 0 else dataset_idx.item())
            else:
                dataset_str = str(dataset_idx)

            ax8.text(0.1, y_pos, f'Dataset: {dataset_str}', fontsize=14, transform=ax8.transAxes)
            y_pos -= 0.12

            # FG:BG → N/A for Gear2
            ax8.text(0.1, y_pos, f'FG:BG = N/A', fontsize=12,
                    transform=ax8.transAxes, bbox=dict(boxstyle="round", facecolor='lightgray'))
            y_pos -= 0.12

            # Show FPS if available
            if fps is not None:
                ax8.text(0.1, y_pos, f'FPS: {fps:.1f}', fontsize=12,
                        transform=ax8.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
                y_pos -= 0.12

            # Show loss values if available
            if loss_dict is not None:
                # Validation loss (for validation visualization)
                if 'val_loss' in loss_dict:
                    ax8.text(0.1, y_pos, f'Val Loss: {loss_dict["val_loss"]:.4f}', fontsize=11,
                            transform=ax8.transAxes, bbox=dict(boxstyle="round", facecolor='lightcoral'))
                    y_pos -= 0.10

                # Depth loss (for training visualization)
                if 'depth_loss' in loss_dict:
                    ax8.text(0.1, y_pos, f'Log L1 Loss: {loss_dict["depth_loss"]:.4f}', fontsize=11,
                            transform=ax8.transAxes, bbox=dict(boxstyle="round", facecolor='lightcoral'))
                    y_pos -= 0.10

            ax8.set_title('Training Info', fontsize=14, fontweight='bold')
            ax8.axis('off')

            # 9. Depth Distribution Histogram
            ax9 = fig.add_subplot(gs[2, :2])
            if valid_mask.sum() > 0:
                gt_valid = gt_depth_frame[valid_mask]
                pred_valid = pred_depth_frame[valid_mask]

                bins = np.linspace(min(gt_valid.min(), pred_valid.min()),
                                  max(gt_valid.max(), pred_valid.max()), 50)

                ax9.hist(gt_valid, bins=bins, alpha=0.6, label='Ground Truth',
                        color='blue', density=True)
                ax9.hist(pred_valid, bins=bins, alpha=0.6, label='Predicted',
                        color='red', density=True)
                ax9.set_xlabel('Depth (meters)', fontsize=12)
                ax9.set_ylabel('Density', fontsize=12)
                ax9.set_title('Depth Distribution', fontsize=14, fontweight='bold')
                ax9.legend(fontsize=12)
                ax9.grid(True, alpha=0.3)

            # 10. Importance Distribution → N/A for Gear2
            ax10 = fig.add_subplot(gs[2, 2:])
            ax10.text(0.5, 0.5, 'No Importance Map\n\nUniform Modulation Applied\n(Same gamma/beta for all pixels)',
                     ha='center', va='center', transform=ax10.transAxes,
                     fontsize=14, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax10.set_title('Importance Map Distribution (N/A)', fontsize=14, fontweight='bold')
            ax10.axis('off')

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
