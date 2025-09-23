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

    def create_validation_summary(self, sample_batch, model_outputs, step, save_name=None, prefix="validation", image_path=None):
        """
        Create a comprehensive validation summary visualization

        Args:
            sample_batch: Validation batch (video, gt_depth, dataset_name)
            model_outputs: Model predictions dictionary
            step: Training step number
            save_name: Optional custom save name
            prefix: Prefix for the visualization
            image_path: Optional image file path for display

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

            # Create valid mask considering both GT and pred ranges to prevent extreme values
            gt_valid_mask = gt_depth_frame > 0  # GT valid pixels
            pred_valid_mask = (pred_metric_frame > 0) & (pred_metric_frame < 1000.0)  # Pred in reasonable range
            valid_mask = gt_valid_mask & pred_valid_mask

            # Debug: Print statistics for troubleshooting
            gt_valid = gt_depth_frame[valid_mask]
            pred_valid = pred_metric_frame[valid_mask]
            print(f"DEBUG Step {step}:")
            print(f"  GT valid range: {gt_valid.min():.3f} - {gt_valid.max():.3f}")
            print(f"  Pred valid range: {pred_valid.min():.3f} - {pred_valid.max():.3f}")
            print(f"  GT invalid pixels: {(gt_depth_frame == -1).sum()}")
            print(f"  Valid pixels: {valid_mask.sum()}")

            # Check for extreme values that might cause issues
            extreme_pred = pred_valid[pred_valid > 1000]
            if len(extreme_pred) > 0:
                print(f"  EXTREME pred values (>1000m): {len(extreme_pred)} pixels, max: {extreme_pred.max():.1f}")

            # Create figure with subplots
            fig = plt.figure(figsize=(20, 12))
            gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

            # 1. Input Image
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(input_img)
            # Main title with 14pt font
            ax1.set_title('Input Image', fontsize=14, fontweight='bold')

            # Add file path info with smaller font (9pt) between title and image
            # if image_path:
            #     # Extract path after 'tartanair/' or similar dataset name
            #     path_parts = image_path.split('/')
            #     if 'tartanair' in path_parts:
            #         tartanair_idx = path_parts.index('tartanair')
            #         display_path = '/'.join(path_parts[tartanair_idx+1:])
            #     elif any(dataset in path_parts for dataset in ['spring', 'pointodyssey', 'dynamicreplica', 'mvs_synth']):
            #         # Find dataset name and start from next part
            #         for i, part in enumerate(path_parts):
            #             if part in ['spring', 'pointodyssey', 'dynamicreplica', 'mvs_synth']:
            #                 display_path = '/'.join(path_parts[i+1:])
            #                 break
            #         else:
            #             display_path = '/'.join(path_parts[-3:])  # Show last 3 parts as fallback
            #     else:
            #         display_path = '/'.join(path_parts[-3:])  # Show last 3 parts as fallback
            # else:
            #     # Fallback to dataset name if no image path
            #     display_path = dataset_name[0] if isinstance(dataset_name, (list, tuple)) and len(dataset_name) > 0 else str(dataset_name)

            # ax1.text(0.5, 0.95, f'{display_path}', fontsize=9, ha='center', va='top',
            #         transform=ax1.transAxes, color='gray', style='italic')
            ax1.axis('off')

            # 2. Ground Truth Depth (for TartanAir: already metric depth)
            ax2 = fig.add_subplot(gs[0, 1])
            # For TartanAir, GT is already metric depth, no conversion needed
            # For other datasets, would need: gt_metric_depth = 1.0 / (gt_depth_frame + 1e-8)
            gt_metric_depth = gt_depth_frame  # Assuming TartanAir format
            gt_display = np.where(valid_mask, gt_metric_depth, np.nan)
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

            # 4. Predicted Relative Depth (display raw values from FlashDepth final head)
            ax4 = fig.add_subplot(gs[0, 3])
            # Debug: Print relative depth statistics
            rel_mean = np.mean(pred_relative_frame)
            rel_std = np.std(pred_relative_frame)
            rel_min, rel_max = pred_relative_frame.min(), pred_relative_frame.max()
            print(f"DEBUG - Relative depth stats: mean={rel_mean:.4f}, std={rel_std:.4f}, min={rel_min:.4f}, max={rel_max:.4f}")

            rel_vmin, rel_vmax = np.nanpercentile(pred_relative_frame, [2, 98])
            im4 = ax4.imshow(pred_relative_frame, cmap='plasma', vmin=rel_vmin, vmax=rel_vmax)
            ax4.set_title(f'Relative Depth\nstd={rel_std:.3f}', fontsize=14, fontweight='bold')
            ax4.axis('off')
            plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

            # 5. Valid Mask
            ax5 = fig.add_subplot(gs[1, 0])
            # Use reversed colormap so valid pixels (True) show as black
            ax5.imshow(valid_mask.astype(np.uint8), cmap='gray_r', vmin=0, vmax=1)
            ax5.set_title(f'Valid Mask\n({valid_mask.sum():,} pixels)', fontsize=14, fontweight='bold')
            ax5.axis('off')

            # 6. Absolute Error Map (both in metric depth space)
            ax6 = fig.add_subplot(gs[1, 1])
            abs_error = np.abs(pred_metric_frame - gt_metric_depth)  # Both in metric depth space
            abs_error_masked = np.where(valid_mask, abs_error, np.nan)
            error_vmax = np.nanpercentile(abs_error_masked, 95)
            im6 = ax6.imshow(abs_error_masked, cmap='hot', vmin=0, vmax=error_vmax)
            ax6.set_title(f'Absolute Error (m)\nMean: {np.nanmean(abs_error_masked):.3f}',
                         fontsize=14, fontweight='bold')
            ax6.axis('off')
            plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

            # 7. Metric Evaluation (AbsRel and Delta_1)
            ax7 = fig.add_subplot(gs[1, 2])
            # Convert to torch tensors for metric computation
            pred_tensor = torch.from_numpy(pred_metric_frame).float()
            gt_tensor = torch.from_numpy(gt_metric_depth).float()
            valid_tensor = torch.from_numpy(valid_mask).bool()

            # Compute metrics
            metrics = MetricDepthMetrics.compute_metric_depth_metrics(
                pred_tensor, gt_tensor, valid_tensor
            )

            # Display AbsRel and Delta metrics as text
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
            ax7.text(0.1, 0.1, f'Valid: {metrics.get("mae", 0):.3f}m MAE', fontsize=12,
                    transform=ax7.transAxes, bbox=dict(boxstyle="round", facecolor='lightblue'))
            ax7.set_title('Depth Metrics', fontsize=14, fontweight='bold')
            ax7.axis('off')

            # 8. Scale and Shift Info
            ax8 = fig.add_subplot(gs[1, 3])
            ax8.text(0.1, 0.8, f'Scale: {scale[0, 0].item():.4f}', fontsize=16,
                    transform=ax8.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
            ax8.text(0.1, 0.6, f'Shift: {shift[0, 0].item():.4f}', fontsize=16,
                    transform=ax8.transAxes, bbox=dict(boxstyle="round", facecolor='lightblue'))
            # Handle dataset name (could be list or tensor)
            if isinstance(dataset_name, (list, tuple)):
                dataset_str = dataset_name[0] if len(dataset_name) > 0 else 'unknown'
            else:
                dataset_str = str(dataset_name)
            ax8.text(0.1, 0.4, f'Dataset: {dataset_str}', fontsize=14,
                    transform=ax8.transAxes)
            ax8.text(0.1, 0.2, f'Step: {step}', fontsize=14, transform=ax8.transAxes)
            ax8.set_title('Transformation Parameters', fontsize=14, fontweight='bold')
            ax8.axis('off')

            # 9. Depth Distribution Histogram (both in metric depth space)
            ax9 = fig.add_subplot(gs[2, :2])
            gt_valid = gt_metric_depth[valid_mask]  # Use converted metric GT depth
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

            # 10. Scatter Plot: GT vs Predicted (both in metric depth space)
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
            fig.suptitle(f'Metric Depth Estimation {prefix.capitalize()} - Step {step}',
                        fontsize=16, fontweight='bold', y=0.98)

            # Save figure
            if save_name is None:
                save_name = f'{prefix}_step_{step}.png'

            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            self.logger.info(f"{prefix.capitalize()} visualization saved to {save_path}")

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