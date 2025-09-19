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


class MetricDepthVisualizer:
    """
    Visualization utilities for metric depth estimation validation
    """

    def __init__(self, save_dir="./visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")

    def create_validation_summary(self, sample_batch, model_outputs, step, save_name=None):
        """
        Create a comprehensive validation summary visualization

        Args:
            sample_batch: Validation batch (video, gt_depth, dataset_name)
            model_outputs: Model predictions dictionary
            step: Training step number
            save_name: Optional custom save name

        Returns:
            fig: Matplotlib figure object
        """
        try:
            video, gt_depth, dataset_name = sample_batch
            pred_metric = model_outputs['metric_depth']
            pred_relative = model_outputs['relative_depth']
            scale = model_outputs['scale']
            shift = model_outputs['shift']

            # Use first batch and first frame for visualization
            input_img = video[0, 0].cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
            gt_depth_frame = gt_depth[0, 0].cpu().numpy()  # [H, W]
            pred_metric_frame = pred_metric[0, 0].cpu().numpy()  # [H, W]
            pred_relative_frame = pred_relative[0, 0].cpu().numpy()  # [H, W]

            # Normalize input image for display
            input_img = np.clip((input_img + 1) / 2, 0, 1)  # Assuming normalized input

            # Create valid mask
            valid_mask = gt_depth_frame >= 0

            # Create figure with subplots
            fig = plt.figure(figsize=(20, 12))
            gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

            # 1. Input Image
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(input_img)
            ax1.set_title('Input Image', fontsize=14, fontweight='bold')
            ax1.axis('off')

            # 2. Ground Truth Depth
            ax2 = fig.add_subplot(gs[0, 1])
            gt_display = np.where(valid_mask, gt_depth_frame, np.nan)
            vmin, vmax = np.nanpercentile(gt_display, [2, 98])
            im2 = ax2.imshow(gt_display, cmap='plasma', vmin=vmin, vmax=vmax)
            ax2.set_title('Ground Truth Depth (m)', fontsize=14, fontweight='bold')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            # 3. Predicted Metric Depth
            ax3 = fig.add_subplot(gs[0, 2])
            pred_display = np.where(valid_mask, pred_metric_frame, np.nan)
            im3 = ax3.imshow(pred_display, cmap='plasma', vmin=vmin, vmax=vmax)
            ax3.set_title('Predicted Metric Depth (m)', fontsize=14, fontweight='bold')
            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

            # 4. Predicted Relative Depth
            ax4 = fig.add_subplot(gs[0, 3])
            rel_vmin, rel_vmax = np.nanpercentile(pred_relative_frame, [2, 98])
            im4 = ax4.imshow(pred_relative_frame, cmap='plasma', vmin=rel_vmin, vmax=rel_vmax)
            ax4.set_title('Predicted Relative Depth', fontsize=14, fontweight='bold')
            ax4.axis('off')
            plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

            # 5. Valid Mask
            ax5 = fig.add_subplot(gs[1, 0])
            ax5.imshow(valid_mask, cmap='gray')
            ax5.set_title(f'Valid Mask\n({valid_mask.sum():,} pixels)', fontsize=14, fontweight='bold')
            ax5.axis('off')

            # 6. Absolute Error Map
            ax6 = fig.add_subplot(gs[1, 1])
            abs_error = np.abs(pred_metric_frame - gt_depth_frame)
            abs_error_masked = np.where(valid_mask, abs_error, np.nan)
            error_vmax = np.nanpercentile(abs_error_masked, 95)
            im6 = ax6.imshow(abs_error_masked, cmap='hot', vmin=0, vmax=error_vmax)
            ax6.set_title(f'Absolute Error (m)\nMean: {np.nanmean(abs_error_masked):.3f}',
                         fontsize=14, fontweight='bold')
            ax6.axis('off')
            plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

            # 7. Relative Error Map
            ax7 = fig.add_subplot(gs[1, 2])
            rel_error = abs_error / (gt_depth_frame + 1e-8)
            rel_error_masked = np.where(valid_mask, rel_error, np.nan)
            rel_error_vmax = np.nanpercentile(rel_error_masked, 95)
            im7 = ax7.imshow(rel_error_masked, cmap='hot', vmin=0, vmax=rel_error_vmax)
            ax7.set_title(f'Relative Error\nMean: {np.nanmean(rel_error_masked):.3f}',
                         fontsize=14, fontweight='bold')
            ax7.axis('off')
            plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)

            # 8. Scale and Shift Info
            ax8 = fig.add_subplot(gs[1, 3])
            ax8.text(0.1, 0.8, f'Scale: {scale[0, 0].item():.4f}', fontsize=16,
                    transform=ax8.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
            ax8.text(0.1, 0.6, f'Shift: {shift[0, 0].item():.4f}', fontsize=16,
                    transform=ax8.transAxes, bbox=dict(boxstyle="round", facecolor='lightblue'))
            ax8.text(0.1, 0.4, f'Dataset: {dataset_name}', fontsize=14,
                    transform=ax8.transAxes)
            ax8.text(0.1, 0.2, f'Step: {step}', fontsize=14, transform=ax8.transAxes)
            ax8.set_title('Transformation Parameters', fontsize=14, fontweight='bold')
            ax8.axis('off')

            # 9. Depth Distribution Histogram
            ax9 = fig.add_subplot(gs[2, :2])
            gt_valid = gt_depth_frame[valid_mask]
            pred_valid = pred_metric_frame[valid_mask]

            # Create histograms
            bins = np.linspace(min(gt_valid.min(), pred_valid.min()),
                              max(gt_valid.max(), pred_valid.max()), 50)

            ax9.hist(gt_valid, bins=bins, alpha=0.6, label='Ground Truth',
                    color='blue', density=True)
            ax9.hist(pred_valid, bins=bins, alpha=0.6, label='Predicted',
                    color='red', density=True)
            ax9.set_xlabel('Depth (meters)', fontsize=12)
            ax9.set_ylabel('Density', fontsize=12)
            ax9.set_title('Depth Distribution Comparison', fontsize=14, fontweight='bold')
            ax9.legend()
            ax9.grid(True, alpha=0.3)

            # 10. Scatter Plot: GT vs Predicted
            ax10 = fig.add_subplot(gs[2, 2:])
            # Subsample for visualization if too many points
            if len(gt_valid) > 10000:
                indices = np.random.choice(len(gt_valid), 10000, replace=False)
                gt_sample = gt_valid[indices]
                pred_sample = pred_valid[indices]
            else:
                gt_sample = gt_valid
                pred_sample = pred_valid

            ax10.scatter(gt_sample, pred_sample, alpha=0.5, s=1)

            # Perfect prediction line
            min_val = min(gt_sample.min(), pred_sample.min())
            max_val = max(gt_sample.max(), pred_sample.max())
            ax10.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
                     label='Perfect Prediction')

            ax10.set_xlabel('Ground Truth Depth (m)', fontsize=12)
            ax10.set_ylabel('Predicted Depth (m)', fontsize=12)
            ax10.set_title('GT vs Predicted Scatter Plot', fontsize=14, fontweight='bold')
            ax10.legend()
            ax10.grid(True, alpha=0.3)

            # Overall title
            fig.suptitle(f'Metric Depth Estimation Validation - Step {step}',
                        fontsize=16, fontweight='bold', y=0.98)

            # Save figure
            if save_name is None:
                save_name = f'validation_step_{step}.png'

            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            self.logger.info(f"Validation visualization saved to {save_path}")

            return fig

        except Exception as e:
            self.logger.error(f"Error creating validation summary: {e}")
            return None

    def create_training_progress_plot(self, metrics_history, save_name="training_progress.png"):
        """
        Create training progress visualization

        Args:
            metrics_history: Dictionary containing training metrics over time
            save_name: Name for saved figure

        Returns:
            fig: Matplotlib figure object
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Training Progress', fontsize=16, fontweight='bold')

            # Loss curves
            if 'train_loss' in metrics_history and 'val_loss' in metrics_history:
                axes[0, 0].plot(metrics_history['steps'], metrics_history['train_loss'],
                               label='Training Loss', color='blue', alpha=0.7)
                axes[0, 0].plot(metrics_history['val_steps'], metrics_history['val_loss'],
                               label='Validation Loss', color='red', alpha=0.7)
                axes[0, 0].set_xlabel('Step')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].set_title('Loss Curves')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)

            # Scale evolution
            if 'scale' in metrics_history:
                axes[0, 1].plot(metrics_history['steps'], metrics_history['scale'],
                               color='green', alpha=0.7)
                axes[0, 1].set_xlabel('Step')
                axes[0, 1].set_ylabel('Scale Parameter')
                axes[0, 1].set_title('Scale Parameter Evolution')
                axes[0, 1].grid(True, alpha=0.3)

            # Shift evolution
            if 'shift' in metrics_history:
                axes[1, 0].plot(metrics_history['steps'], metrics_history['shift'],
                               color='orange', alpha=0.7)
                axes[1, 0].set_xlabel('Step')
                axes[1, 0].set_ylabel('Shift Parameter')
                axes[1, 0].set_title('Shift Parameter Evolution')
                axes[1, 0].grid(True, alpha=0.3)

            # Validation metrics
            if 'abs_rel' in metrics_history:
                axes[1, 1].plot(metrics_history['val_steps'], metrics_history['abs_rel'],
                               label='AbsRel', color='purple', alpha=0.7)
            if 'delta_1' in metrics_history:
                axes[1, 1].plot(metrics_history['val_steps'], metrics_history['delta_1'],
                               label='Delta_1', color='brown', alpha=0.7)

            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Metric Value')
            axes[1, 1].set_title('Validation Metrics')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            # Save figure
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            self.logger.info(f"Training progress plot saved to {save_path}")

            return fig

        except Exception as e:
            self.logger.error(f"Error creating training progress plot: {e}")
            return None

    def create_error_analysis(self, gt_depths, pred_depths, valid_masks, save_name="error_analysis.png"):
        """
        Create detailed error analysis visualization

        Args:
            gt_depths: Ground truth depth maps [N, H, W]
            pred_depths: Predicted depth maps [N, H, W]
            valid_masks: Valid pixel masks [N, H, W]
            save_name: Name for saved figure

        Returns:
            fig: Matplotlib figure object
        """
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Error Analysis', fontsize=16, fontweight='bold')

            # Flatten all valid pixels across all samples
            gt_flat = []
            pred_flat = []

            for i in range(len(gt_depths)):
                mask = valid_masks[i]
                gt_flat.extend(gt_depths[i][mask].flatten())
                pred_flat.extend(pred_depths[i][mask].flatten())

            gt_flat = np.array(gt_flat)
            pred_flat = np.array(pred_flat)

            # 1. Error distribution
            abs_errors = np.abs(pred_flat - gt_flat)
            axes[0, 0].hist(abs_errors, bins=50, alpha=0.7, color='red')
            axes[0, 0].set_xlabel('Absolute Error (m)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Absolute Error Distribution')
            axes[0, 0].grid(True, alpha=0.3)

            # 2. Relative error distribution
            rel_errors = abs_errors / (gt_flat + 1e-8)
            axes[0, 1].hist(rel_errors, bins=50, alpha=0.7, color='orange')
            axes[0, 1].set_xlabel('Relative Error')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Relative Error Distribution')
            axes[0, 1].grid(True, alpha=0.3)

            # 3. Error vs depth
            axes[0, 2].scatter(gt_flat, abs_errors, alpha=0.1, s=1)
            axes[0, 2].set_xlabel('Ground Truth Depth (m)')
            axes[0, 2].set_ylabel('Absolute Error (m)')
            axes[0, 2].set_title('Error vs Depth')
            axes[0, 2].grid(True, alpha=0.3)

            # 4. Prediction accuracy by depth range
            depth_ranges = [(0, 2), (2, 5), (5, 10), (10, float('inf'))]
            range_labels = ['0-2m', '2-5m', '5-10m', '>10m']
            range_errors = []

            for depth_min, depth_max in depth_ranges:
                mask = (gt_flat >= depth_min) & (gt_flat < depth_max)
                if mask.sum() > 0:
                    range_errors.append(np.mean(abs_errors[mask]))
                else:
                    range_errors.append(0)

            axes[1, 0].bar(range_labels, range_errors, alpha=0.7, color='green')
            axes[1, 0].set_xlabel('Depth Range')
            axes[1, 0].set_ylabel('Mean Absolute Error (m)')
            axes[1, 0].set_title('Error by Depth Range')
            axes[1, 0].grid(True, alpha=0.3)

            # 5. Cumulative error distribution
            sorted_errors = np.sort(abs_errors)
            percentiles = np.arange(1, 101)
            error_percentiles = np.percentile(sorted_errors, percentiles)

            axes[1, 1].plot(percentiles, error_percentiles, color='purple', linewidth=2)
            axes[1, 1].set_xlabel('Percentile')
            axes[1, 1].set_ylabel('Absolute Error (m)')
            axes[1, 1].set_title('Cumulative Error Distribution')
            axes[1, 1].grid(True, alpha=0.3)

            # 6. Prediction vs GT correlation
            correlation = np.corrcoef(gt_flat, pred_flat)[0, 1]
            axes[1, 2].scatter(gt_flat, pred_flat, alpha=0.1, s=1, color='blue')

            # Perfect prediction line
            min_val = min(gt_flat.min(), pred_flat.min())
            max_val = max(gt_flat.max(), pred_flat.max())
            axes[1, 2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

            axes[1, 2].set_xlabel('Ground Truth Depth (m)')
            axes[1, 2].set_ylabel('Predicted Depth (m)')
            axes[1, 2].set_title(f'GT vs Pred Correlation (r={correlation:.3f})')
            axes[1, 2].grid(True, alpha=0.3)

            plt.tight_layout()

            # Save figure
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            self.logger.info(f"Error analysis saved to {save_path}")

            return fig

        except Exception as e:
            self.logger.error(f"Error creating error analysis: {e}")
            return None