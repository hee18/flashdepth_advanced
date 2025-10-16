#!/usr/bin/env python3
"""
Test script for Gear3: Feature-level Metric Depth Learning

Key differences from baseline:
    - No relative depth visualization (features are modulated)
    - Test on inverse depth metrics
    - De-canonicalize outputs for final evaluation
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import logging
import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from PIL import Image
from einops import rearrange
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from flashdepth.model import FlashDepth
from flashdepth.gear3_modules import Gear3MetricHead
from dataloaders.combined_dataset import CombinedDataset
from utils.metric_depth_metrics import MetricDepthMetrics, format_metrics
from utils.helpers import save_gifs_as_grid, save_grid_to_mp4, depth_to_np_arr, torch_batch_to_np_arr

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Gear3Tester:
    """
    Test harness for Gear3 model.

    Evaluates on:
        - Inverse depth metrics (TAE, AbsRel, δ1/δ2/δ3)
        - Metric depth visualization (no relative depth)
        - Importance map visualization
    """
    def __init__(self, config):
        self.config = config
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Setup save directory - use results_dir if provided, otherwise use eval.outfolder
        save_dir_str = config.get('results_dir', config.eval.outfolder)
        self.save_dir = Path(save_dir_str)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Save directory: {self.save_dir}")

        # Initialize model
        self.model = self._setup_model()

        # Setup test loader
        self.test_loader = self._setup_test_loader()

        # Setup metrics
        self.metrics = MetricDepthMetrics()

    def _setup_model(self):
        """Load trained Gear3 model"""
        # Create base FlashDepth model
        model_config = dict(self.config.model)
        model_config['batch_size'] = 1
        model_config['use_metric_head'] = False

        model = FlashDepth(**model_config)

        # Add Gear3 metric head
        embed_dim = 1024 if model.encoder == 'vitl' else 384
        dpt_dim = 256 if model.encoder == 'vitl' else 64
        num_heads = 16 if model.encoder == 'vitl' else 6

        model.gear3_head = Gear3MetricHead(
            embed_dim=embed_dim,
            dpt_dim=dpt_dim,
            num_heads=num_heads
        )

        # Enable attention weights storage ONLY for last block (like train_gear3)
        for i, block in enumerate(model.pretrained.blocks):
            if i == len(model.pretrained.blocks) - 1:
                block.attn.store_attn_weights = True
                logger.info(f"Enabled attention weights storage for block {i} (last block)")
            else:
                block.attn.store_attn_weights = False

        # Load checkpoint
        checkpoint_path = self.config.get('load')
        if checkpoint_path and checkpoint_path != 'true':
            if os.path.exists(checkpoint_path):
                logger.info(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location='cpu')

                # Extract state dict
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                # Load state dict
                model.load_state_dict(state_dict, strict=True)
                logger.info(f"Loaded checkpoint successfully")

                # Log training info if available
                if 'global_step' in checkpoint:
                    logger.info(f"Checkpoint step: {checkpoint['global_step']}")
                if 'phase' in checkpoint:
                    logger.info(f"Training phase: {checkpoint['phase']}")
            else:
                logger.warning(f"Checkpoint {checkpoint_path} not found")
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        else:
            logger.warning("No checkpoint specified!")

        model = model.to(self.device)
        model.eval()

        return model

    def _setup_test_loader(self):
        """Setup test data loader"""
        # Check if single sequence mode
        single_seq_path = self.config.get('single_sequence', None)

        if single_seq_path:
            # Single sequence mode: create custom dataset
            logger.info(f"Single sequence mode: {single_seq_path}")
            test_dataset = self._create_single_sequence_dataset(single_seq_path)
        else:
            # Normal mode: use CombinedDataset
            test_datasets = self.config.eval.test_datasets
            video_length = self.config.get('vid_len', 50)  # Get from config override

            logger.info(f"Test datasets: {test_datasets}")
            logger.info(f"Video length: {video_length}")

            test_dataset = CombinedDataset(
                root_dir=self.config.dataset.data_root,
                enable_dataset_flags=test_datasets,
                resolution=self.config.eval.test_dataset_resolution,
                split='val',  # Use 'val' split which returns dict format
                video_length=video_length
            )

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,  # Single process for testing
            pin_memory=True,
            collate_fn=self._collate_fn
        )

        logger.info(f"Test dataset size: {len(test_dataset)}")

        return test_loader

    def _create_single_sequence_dataset(self, sequence_path):
        """Create a dataset from a single sequence directory"""
        from torch.utils.data import Dataset
        import glob
        from PIL import Image

        class SingleSequenceDataset(Dataset):
            def __init__(self, seq_path, resolution=518):
                self.seq_path = Path(seq_path)
                # Ensure resolution is an integer (config may pass 'base' string)
                if isinstance(resolution, str):
                    if resolution == 'base':
                        self.resolution = 518
                    else:
                        self.resolution = int(resolution)
                else:
                    self.resolution = int(resolution)

                # Find images and depths
                images_dir = self.seq_path / "images"
                depths_dir = self.seq_path / "depths"

                if not images_dir.exists():
                    raise FileNotFoundError(f"Images directory not found: {images_dir}")
                if not depths_dir.exists():
                    raise FileNotFoundError(f"Depths directory not found: {depths_dir}")

                # Get sorted image files
                self.image_files = sorted(glob.glob(str(images_dir / "*.png")))

                # Determine depth file pattern
                if len(self.image_files) > 0:
                    img_name = Path(self.image_files[0]).stem
                    # Try .geometric.png pattern (dynamicreplica)
                    depth_pattern = str(depths_dir / f"{img_name}_*.geometric.png")
                    depth_files = glob.glob(depth_pattern)

                    if len(depth_files) == 0:
                        # Try direct pattern matching
                        depth_files = sorted(glob.glob(str(depths_dir / "*.geometric.png")))

                    self.depth_files = sorted(depth_files)
                else:
                    self.depth_files = []

                assert len(self.image_files) > 0, f"No images found in {images_dir}"
                assert len(self.depth_files) > 0, f"No depth files found in {depths_dir}"
                assert len(self.image_files) == len(self.depth_files), \
                    f"Mismatch: {len(self.image_files)} images vs {len(self.depth_files)} depths"

                logger.info(f"Loaded single sequence: {len(self.image_files)} frames")

            def __len__(self):
                return 1  # Single sequence

            def __getitem__(self, idx):
                # Load all frames
                images = []
                depths = []

                for img_path, depth_path in zip(self.image_files, self.depth_files):
                    # Load image
                    img = Image.open(img_path).convert('RGB')
                    # Use integer constants for PIL compatibility (2=BILINEAR, 0=NEAREST)
                    img = img.resize((self.resolution, self.resolution), 2)  # 2 = BILINEAR (positional arg)
                    img_array = np.array(img).astype(np.float32) / 255.0
                    # Normalize (ImageNet stats)
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    img_normalized = (img_array - mean) / std
                    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float()  # [3, H, W], ensure float32
                    images.append(img_tensor)

                    # Load depth (.geometric.png for dynamicreplica)
                    depth_img = Image.open(depth_path)
                    depth_array = np.array(depth_img).astype(np.float32) / 1000.0  # mm to m
                    # Resize depth
                    depth_pil = Image.fromarray(depth_array)
                    depth_resized = depth_pil.resize((self.resolution, self.resolution), 0)  # 0 = NEAREST (positional arg)
                    depth_array = np.array(depth_resized)
                    # Convert to inverse depth (1/m) like CombinedDataset
                    inverse_depth = np.zeros_like(depth_array)
                    valid_mask = depth_array > 0
                    inverse_depth[valid_mask] = 1.0 / depth_array[valid_mask]
                    depth_tensor = torch.from_numpy(inverse_depth)  # [H, W]
                    depths.append(depth_tensor)

                # Stack
                images = torch.stack(images, dim=0)  # [T, 3, H, W]
                depths = torch.stack(depths, dim=0)  # [T, H, W]

                return images, depths, "single_sequence"

        return SingleSequenceDataset(sequence_path, resolution=self.config.eval.test_dataset_resolution)

    def _collate_fn(self, batch):
        """Custom collate function to filter out None values and convert tuple to dict"""
        # Filter out None values
        batch = [item for item in batch if item is not None]

        # If all items are None, skip this batch
        if len(batch) == 0:
            return None

        # CombinedDataset returns (images, depths, dataset_name) tuple for val/test splits
        # Convert to dict format for easier access
        if len(batch) > 0 and isinstance(batch[0], tuple):
            images, depths, names = zip(*batch)
            return {
                'image': torch.stack(images, dim=0),
                'depth': torch.stack(depths, dim=0),
                'dataset_name': names
            }

        # Use default collate for dict items (training split)
        return torch.utils.data.dataloader.default_collate(batch)

    @torch.no_grad()
    def test(self):
        """Main testing loop"""
        logger.info("Starting testing...")

        all_metrics = []
        sequence_id = 0

        for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Testing")):
            try:
                metrics = self.test_sequence(batch, sequence_id)
                # Add sequence_id for tracking
                metrics['sequence_id'] = sequence_id
                all_metrics.append(metrics)
                sequence_id += 1

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Aggregate metrics
        if all_metrics:
            avg_metrics = self._aggregate_metrics(all_metrics)
            logger.info("\n" + "="*80)
            logger.info("FINAL RESULTS")
            logger.info("="*80)
            logger.info(format_metrics(avg_metrics))

            # Save overall results
            results_path = self.save_dir / "test_results.json"
            with open(results_path, 'w') as f:
                json.dump(avg_metrics, f, indent=2)
            logger.info(f"Results saved to {results_path}")

            # Save per-sequence results
            per_sequence_path = self.save_dir / "per_sequence_results.json"
            with open(per_sequence_path, 'w') as f:
                json.dump(all_metrics, f, indent=2)
            logger.info(f"Per-sequence results saved to {per_sequence_path}")

            # Find and save best sequence (lowest abs_rel)
            best_seq = min(all_metrics, key=lambda x: x['abs_rel'])
            best_seq_path = self.save_dir / "best_sequence.json"
            with open(best_seq_path, 'w') as f:
                json.dump(best_seq, f, indent=2)
            logger.info(f"\nBest sequence (lowest AbsRel): Sequence {best_seq['sequence_id']}")
            logger.info(f"  AbsRel: {best_seq['abs_rel']:.4f}")
            logger.info(f"  MAE: {best_seq['mae']:.4f}")
            logger.info(f"  RMSE: {best_seq['rmse']:.4f}")
            logger.info(f"  δ1: {best_seq['a1']:.4f}")
            logger.info(f"Best sequence saved to {best_seq_path}")

        else:
            logger.warning("No metrics computed!")

    @torch.no_grad()
    def test_sequence(self, batch, sequence_id):
        """Test on a single sequence"""
        # Debug: Check batch type and structure
        if not isinstance(batch, dict):
            logger.error(f"Batch is not a dict! Type: {type(batch)}, Content: {batch if not isinstance(batch, torch.Tensor) else 'Tensor'}")
            raise TypeError(f"Expected dict, got {type(batch)}")

        images = batch['image'].to(self.device)  # [1, T, 3, H, W]
        gt_depth = batch['depth'].to(self.device)  # [1, T, H, W] - val split has no channel dim

        # Add channel dimension if needed
        if gt_depth.ndim == 3:
            gt_depth = gt_depth.unsqueeze(2)  # [1, T, 1, H, W]
        elif gt_depth.ndim == 4 and gt_depth.shape[2] > 3:
            # Already [1, T, H, W], add channel dim
            gt_depth = gt_depth.unsqueeze(2)  # [1, T, 1, H, W]

        B, T = images.shape[:2]
        assert B == 1, "Batch size must be 1 for testing"

        # Dataloader gives inverse depth (1/m), scale to 100/m for training
        gt_depth_inverse_100 = gt_depth * 100.0  # [1, T, 1, H, W] in 100/m

        # Storage for predictions
        pred_depths = []
        importance_maps = []
        fg_features_list = []
        bg_features_list = []

        # Best frame tracking
        best_frame_idx = 0
        best_frame_abs_rel = float('inf')

        # Process each frame
        for t in range(T):
            img_t = images[0, t]  # [3, H, W]
            gt_t_inverse = gt_depth_inverse_100[0, t]  # [1, H, W] in 100/m

            # Extract features from DINOv2
            encoder_features = self.model.pretrained.get_intermediate_layers(
                img_t.unsqueeze(0), self.model.intermediate_layer_idx[self.model.encoder]
            )

            # Get attention weights from last block
            last_block = self.model.pretrained.blocks[-1]
            attention_weights = last_block.attn.attn_weights

            # Get patch tokens from last encoder layer
            patch_tokens = encoder_features[-1]

            # Get DPT features
            h, w = img_t.shape[1:]
            patch_h, patch_w = h // self.model.patch_size, w // self.model.patch_size
            dpt_features = self.model.depth_head.get_forward_features(encoder_features, patch_h, patch_w)

            # Apply Gear3 modulation
            path_1_modulated, importance_map, fg_features, bg_features = self.model.gear3_head(
                patch_tokens, attention_weights, dpt_features, patch_h, patch_w
            )

            # Get depth prediction (output is inverse depth in 100/m scale)
            # path_1_modulated is already the modulated feature, no need to index
            out = self.model.depth_head.scratch.output_conv1(path_1_modulated)
            out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
            out = self.model.depth_head.scratch.output_conv2(out)  # [1, 1, H, W]

            # Prediction is already positive (Softplus activation in output_conv2)
            pred_depth_inverse_100 = out  # [1, 1, H, W] in 100/m

            # Interpolate prediction to GT resolution (like train_gear3.py validation)
            gt_t_shape = gt_t_inverse.shape[-2:]  # GT original resolution
            if pred_depth_inverse_100.shape[-2:] != gt_t_shape:
                pred_depth_inverse_100 = F.interpolate(
                    pred_depth_inverse_100, size=gt_t_shape, mode="bilinear", align_corners=True
                )

            # Convert to metric depth: 100/m -> m
            pred_depth_metric = 100.0 / (pred_depth_inverse_100[0] + 1e-8)  # [1, H, W]

            pred_depths.append(pred_depth_metric)
            importance_maps.append(importance_map[0])
            fg_features_list.append(fg_features[0])
            bg_features_list.append(bg_features[0])

        # Stack predictions
        pred_depths = torch.stack(pred_depths, dim=0)  # [T, 1, H, W] in meters
        importance_maps = torch.stack(importance_maps, dim=0)  # [T, 1, patch_h, patch_w]
        fg_features_all = torch.stack(fg_features_list, dim=0)  # [T, C, patch_h, patch_w]
        bg_features_all = torch.stack(bg_features_list, dim=0)  # [T, C, patch_h, patch_w]

        # Convert GT to metric depth for evaluation: 100/m -> m
        gt_depth_metric = 100.0 / (gt_depth_inverse_100[0] + 1e-8)  # [T, 1, H, W] in meters

        # Compute metrics (both pred and GT are now in meters)
        # Move to CPU and compute per-frame metrics (like test_metric_head.py)
        pred_depths_cpu = pred_depths.cpu()
        gt_depth_metric_cpu = gt_depth_metric.cpu()

        frame_metrics = []
        for t in range(pred_depths.shape[0]):
            # Get individual frames (already on CPU)
            pred_frame = pred_depths_cpu[t, 0]  # [H, W]
            gt_frame = gt_depth_metric_cpu[t, 0]  # [H, W]

            # Create valid mask for this frame (like train_gear3 validation)
            # Use same MAX_DEPTH as Gear3Visualizer (200m)
            MAX_DEPTH = 200.0
            gt_valid_mask = (gt_frame > 0) & (gt_frame < MAX_DEPTH)  # GT valid pixels
            pred_valid_mask = (pred_frame > 0) & (pred_frame < MAX_DEPTH)  # Filter extreme values
            valid_mask = gt_valid_mask & pred_valid_mask  # [H, W] bool tensor

            # Debug logging for first frame of first sequence
            if t == 0 and sequence_id == 0:
                logger.info(f"DEBUG Metrics - Frame {t}")
                logger.info(f"  GT depth range: [{gt_frame.min():.2f}, {gt_frame.max():.2f}] meters")
                logger.info(f"  Pred depth range: [{pred_frame.min():.2f}, {pred_frame.max():.2f}] meters")
                logger.info(f"  Valid pixels: {valid_mask.sum()} / {valid_mask.numel()} ({100*valid_mask.sum()/valid_mask.numel():.1f}%)")
                if valid_mask.sum() > 0:
                    gt_valid_values = gt_frame[valid_mask]
                    pred_valid_values = pred_frame[valid_mask]
                    logger.info(f"  GT valid range: [{gt_valid_values.min():.2f}, {gt_valid_values.max():.2f}]")
                    logger.info(f"  Pred valid range: [{pred_valid_values.min():.2f}, {pred_valid_values.max():.2f}]")
                    mae = torch.abs(pred_valid_values - gt_valid_values).mean()
                    logger.info(f"  MAE: {mae:.4f} meters")

            if valid_mask.sum() > 0:
                frame_metric = self.metrics.compute_metric_depth_metrics(
                    pred_frame,  # [H, W]
                    gt_frame,   # [H, W]
                    valid_mask  # [H, W]
                )
                frame_metrics.append(frame_metric)

                # Track best frame (lowest abs_rel)
                if frame_metric['abs_rel'] < best_frame_abs_rel:
                    best_frame_abs_rel = frame_metric['abs_rel']
                    best_frame_idx = t

        # Average metrics across frames
        if len(frame_metrics) == 0:
            logger.warning(f"No valid frames for sequence {sequence_id}")
            return {k: 0.0 for k in ["mae", "rmse", "abs_rel", "a1"]}

        metrics = {}
        for key in frame_metrics[0].keys():
            values = [m[key] for m in frame_metrics]
            metrics[key] = np.mean(values)

        # Recreate valid_mask on GPU for visualization
        valid_mask = (gt_depth_metric > 0)  # [T, 1, H, W] on GPU

        # Visualize
        if self.config.eval.get('save_grid', True):
            self._visualize_sequence(
                images[0], pred_depths, gt_depth_metric, importance_maps,
                valid_mask, sequence_id, metrics
            )

        # Save video (GIF or MP4)
        if self.config.eval.get('out_video', True):
            # Resize images to match GT resolution for video creation
            gt_h, gt_w = gt_depth_metric.shape[-2:]
            images_resized = F.interpolate(
                images[0], size=(gt_h, gt_w), mode='bilinear', align_corners=True
            )
            self._save_video(
                images_resized, pred_depths, gt_depth_metric, valid_mask, sequence_id
            )

        # Save best frame visualizations
        if len(frame_metrics) > 0:
            logger.info(f"Best frame for sequence {sequence_id}: Frame {best_frame_idx} (AbsRel={best_frame_abs_rel:.4f})")
            self._save_best_frame_visualizations(
                images[0, best_frame_idx],  # [3, H, W]
                pred_depths[best_frame_idx, 0],  # [H, W]
                gt_depth_metric[best_frame_idx, 0],  # [H, W]
                importance_maps[best_frame_idx, 0],  # [patch_h, patch_w]
                fg_features_all[best_frame_idx],  # [C, patch_h, patch_w]
                bg_features_all[best_frame_idx],  # [C, patch_h, patch_w]
                sequence_id,
                best_frame_idx,
                best_frame_abs_rel
            )

        return metrics

    def _visualize_sequence(self, images, pred_depths, gt_depths, importance_maps,
                           valid_mask, sequence_id, metrics):
        """
        Create visualization grid for a sequence.

        Rows: Image, Metric Depth (Prediction), Metric Depth (GT), Importance Map
        """
        T = images.shape[0]
        frames_to_show = min(10, T)  # Show up to 10 frames
        interval = max(1, T // frames_to_show)
        frame_indices = list(range(0, T, interval))[:frames_to_show]

        # Create figure
        fig, axes = plt.subplots(4, frames_to_show, figsize=(frames_to_show * 3, 12))
        if frames_to_show == 1:
            axes = axes.reshape(-1, 1)

        for col, t in enumerate(frame_indices):
            # Row 0: Image (denormalize ImageNet normalization)
            img = images[t].permute(1, 2, 0).cpu().numpy()
            # Denormalize ImageNet stats
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean  # Reverse normalization
            img = np.clip(img, 0, 1)  # Clip to valid range
            img = (img * 255).astype(np.uint8)
            axes[0, col].imshow(img)
            axes[0, col].set_title(f'Frame {t}')
            axes[0, col].axis('off')

            # Row 1: Predicted metric depth
            pred = pred_depths[t, 0].cpu().numpy()
            pred_valid = (pred > 0) & (pred < 1000)
            pred_display = np.full_like(pred, np.nan)
            if pred_valid.sum() > 0:
                pred_vmin, pred_vmax = np.nanpercentile(pred[pred_valid], [2, 98])
                pred_display[pred_valid] = pred[pred_valid]
            else:
                pred_vmin, pred_vmax = 0, 1
            axes[1, col].imshow(pred_display, cmap='plasma', vmin=pred_vmin, vmax=pred_vmax)
            axes[1, col].set_title(f'Pred (m)')
            axes[1, col].axis('off')

            # Row 2: GT metric depth
            gt = gt_depths[t, 0].cpu().numpy()
            gt_valid = valid_mask[t, 0].cpu().numpy().astype(bool)
            gt_display = np.full_like(gt, np.nan)
            if gt_valid.sum() > 0:
                gt_vmin, gt_vmax = np.nanpercentile(gt[gt_valid], [2, 98])
                gt_display[gt_valid] = gt[gt_valid]
            else:
                gt_vmin, gt_vmax = 0, 1
            axes[2, col].imshow(gt_display, cmap='plasma', vmin=gt_vmin, vmax=gt_vmax)
            axes[2, col].set_title(f'GT (m)')
            axes[2, col].axis('off')

            # Row 3: Importance map (matches train_gear3.py visualization)
            importance = importance_maps[t, 0].cpu().numpy()
            axes[3, col].imshow(importance, cmap='jet', vmin=0, vmax=1)
            axes[3, col].set_title(f'Importance')
            axes[3, col].axis('off')

        # Add overall title with metrics
        fig.suptitle(
            f"Sequence {sequence_id} | "
            f"TAE: {metrics.get('tae', 0):.4f} | "
            f"AbsRel: {metrics.get('abs_rel', 0):.4f} | "
            f"δ1: {metrics.get('delta_1', 0):.4f}",
            fontsize=14
        )

        plt.tight_layout()
        save_path = self.save_dir / f"sequence_{sequence_id:04d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Saved visualization: {save_path}")

    def _save_video(self, images, pred_depths, gt_depths, valid_mask, sequence_id):
        """
        Create video files (GIF/MP4) similar to FlashDepth validation.

        Args:
            images: [T, 3, H, W] - RGB images
            pred_depths: [T, 1, H, W] - Predicted metric depth
            gt_depths: [T, 1, H, W] - GT metric depth
            valid_mask: [T, 1, H, W] - Valid mask
            sequence_id: int - Sequence index
        """
        T = images.shape[0]

        # Convert to numpy arrays for video creation
        video_frames = torch_batch_to_np_arr(images)  # [T, H, W, 3]

        # Convert depth to colorized numpy arrays (per-frame normalization like sequence.png)
        # Use same normalization as _visualize_sequence for consistency
        pred_frames = []
        gt_frames = []

        for t in range(T):
            # Process pred depth
            pred = pred_depths[t, 0].cpu().numpy()
            pred_valid = (pred > 0) & (pred < 1000)
            if pred_valid.sum() > 0:
                pred_vmin, pred_vmax = np.percentile(pred[pred_valid], [2, 98])
                pred_display = np.clip((pred - pred_vmin) / (pred_vmax - pred_vmin + 1e-8), 0, 1)
                pred_display[~pred_valid] = 0
            else:
                pred_display = np.zeros_like(pred)
            pred_colored = (plt.cm.plasma(pred_display)[:, :, :3] * 255).astype(np.uint8)
            pred_frames.append(pred_colored)

            # Process GT depth
            gt = gt_depths[t, 0].cpu().numpy()
            gt_valid = valid_mask[t, 0].cpu().numpy().astype(bool)
            if gt_valid.sum() > 0:
                gt_vmin, gt_vmax = np.percentile(gt[gt_valid], [2, 98])
                gt_display = np.clip((gt - gt_vmin) / (gt_vmax - gt_vmin + 1e-8), 0, 1)
                gt_display[~gt_valid] = 0
            else:
                gt_display = np.zeros_like(gt)
            gt_colored = (plt.cm.plasma(gt_display)[:, :, :3] * 255).astype(np.uint8)
            gt_frames.append(gt_colored)

        # Generate video paths
        base_name = f"sequence_{sequence_id:04d}"
        gif_path = self.save_dir / f"{base_name}.gif"
        mp4_path = self.save_dir / f"{base_name}.mp4"

        # Save based on config
        if self.config.eval.get('out_mp4', False):
            # Save as MP4 (with separate pred-only video)
            logger.info(f"Saving MP4 videos for sequence {sequence_id}...")
            grid = save_grid_to_mp4(
                video_frames,
                gt_frames,
                pred_frames,
                output_path=str(mp4_path),
                fixed_height=self.config.eval.get('save_res', 256),
                fps=self.config.eval.get('video_fps', 10)
            )
            logger.info(f"Saved: {mp4_path}")
            logger.info(f"Saved: {grid['pred_video_path']}")
        else:
            # Save as GIF (default)
            logger.info(f"Saving GIF for sequence {sequence_id}...")
            grid = save_gifs_as_grid(
                video_frames,
                gt_frames,
                pred_frames,
                output_path=str(gif_path),
                fixed_height=self.config.eval.get('save_res', 256),
                duration=self.config.eval.get('gif_duration', 110)
            )
            logger.info(f"Saved: {gif_path}")

        return grid

    def _save_best_frame_visualizations(self, image, pred_depth, gt_depth, importance_map,
                                        fg_features, bg_features, sequence_id, frame_idx, abs_rel):
        """
        Save best frame visualizations following visualize_attention_weights.py style.
        
        Creates 6 separate PNG files:
            1. best_frame_image.png - Original input image
            2. best_frame_gt.png - Ground truth depth
            3. best_frame_pred.png - Predicted depth
            4. best_frame_importance.png - Importance map (colorized)
            5. best_frame_fg.png - Foreground mask overlay
            6. best_frame_bg.png - Background mask overlay
        
        Args:
            image: [3, H, W] - RGB image
            pred_depth: [H, W] - Predicted metric depth
            gt_depth: [H, W] - Ground truth metric depth
            importance_map: [patch_h, patch_w] - Importance map (0-1 normalized)
            fg_features: [C, patch_h, patch_w] - Foreground features (not used for mask, just for reference)
            bg_features: [C, patch_h, patch_w] - Background features (not used for mask, just for reference)
            sequence_id: int - Sequence index
            frame_idx: int - Frame index within sequence
            abs_rel: float - AbsRel metric for this frame
        """
        # Create output directory for best frames
        best_frame_dir = self.save_dir / f"seq{sequence_id:04d}_best_frame"
        best_frame_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert tensors to numpy and move to CPU
        image_np = image.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        pred_depth_np = pred_depth.cpu().numpy()  # [H, W]
        gt_depth_np = gt_depth.cpu().numpy()  # [H, W]
        importance_map_np = importance_map.cpu().numpy()  # [patch_h, patch_w]

        # Denormalize ImageNet normalization for image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = image_np * std + mean  # Reverse normalization
        image_np = np.clip(image_np, 0, 1)  # Clip to valid range

        # Get image size
        img_h, img_w = image_np.shape[:2]

        # 1. Save original image
        img_path = best_frame_dir / "best_frame_image.png"
        img_uint8 = (image_np * 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(img_path)
        logger.info(f"  Saved best frame image: {img_path}")
        
        # 2. Save GT depth (colorized)
        gt_path = best_frame_dir / "best_frame_gt.png"
        gt_valid = gt_depth_np > 0
        if gt_valid.sum() > 0:
            gt_vmin, gt_vmax = np.percentile(gt_depth_np[gt_valid], [2, 98])
            gt_display = np.clip((gt_depth_np - gt_vmin) / (gt_vmax - gt_vmin + 1e-8), 0, 1)
            gt_display[~gt_valid] = 0
        else:
            gt_display = np.zeros_like(gt_depth_np)
        
        gt_colored = (plt.cm.plasma(gt_display)[:, :, :3] * 255).astype(np.uint8)
        Image.fromarray(gt_colored).save(gt_path)
        logger.info(f"  Saved GT depth: {gt_path}")
        
        # 3. Save predicted depth (colorized)
        pred_path = best_frame_dir / "best_frame_pred.png"
        pred_valid = (pred_depth_np > 0) & (pred_depth_np < 1000)
        if pred_valid.sum() > 0:
            pred_vmin, pred_vmax = np.percentile(pred_depth_np[pred_valid], [2, 98])
            pred_display = np.clip((pred_depth_np - pred_vmin) / (pred_vmax - pred_vmin + 1e-8), 0, 1)
            pred_display[~pred_valid] = 0
        else:
            pred_display = np.zeros_like(pred_depth_np)
        
        pred_colored = (plt.cm.plasma(pred_display)[:, :, :3] * 255).astype(np.uint8)
        Image.fromarray(pred_colored).save(pred_path)
        logger.info(f"  Saved predicted depth: {pred_path}")
        
        # 4. Save importance map (colorized, upsampled to image resolution)
        importance_path = best_frame_dir / "best_frame_importance.png"
        importance_upsampled = F.interpolate(
            torch.from_numpy(importance_map_np).unsqueeze(0).unsqueeze(0),
            size=(img_h, img_w),
            mode='bilinear',
            align_corners=True
        ).squeeze().numpy()
        
        importance_colored = (plt.cm.jet(importance_upsampled)[:, :, :3] * 255).astype(np.uint8)
        Image.fromarray(importance_colored).save(importance_path)
        logger.info(f"  Saved importance map: {importance_path}")
        
        # 5. Save FG mask overlay (red overlay on original image)
        fg_path = best_frame_dir / "best_frame_fg.png"
        mean_val = importance_map_np.mean()
        fg_mask = (importance_map_np > mean_val).astype(np.float32)
        
        # Upsample FG mask to image resolution
        fg_mask_upsampled = F.interpolate(
            torch.from_numpy(fg_mask).unsqueeze(0).unsqueeze(0),
            size=(img_h, img_w),
            mode='nearest'
        ).squeeze().numpy()
        
        # Create red overlay
        fg_overlay = img_uint8.copy()
        red_mask = np.zeros_like(fg_overlay)
        red_mask[:, :, 0] = (fg_mask_upsampled * 255).astype(np.uint8)  # Red channel
        fg_overlay = cv2.addWeighted(fg_overlay, 0.5, red_mask, 0.5, 0)
        
        Image.fromarray(fg_overlay).save(fg_path)
        fg_ratio = fg_mask.sum() / fg_mask.size * 100
        logger.info(f"  Saved FG mask (>{mean_val:.3f}): {fg_ratio:.1f}% - {fg_path}")
        
        # 6. Save BG mask overlay (blue overlay on original image)
        bg_path = best_frame_dir / "best_frame_bg.png"
        bg_mask = (importance_map_np <= mean_val).astype(np.float32)
        
        # Upsample BG mask to image resolution
        bg_mask_upsampled = F.interpolate(
            torch.from_numpy(bg_mask).unsqueeze(0).unsqueeze(0),
            size=(img_h, img_w),
            mode='nearest'
        ).squeeze().numpy()
        
        # Create blue overlay
        bg_overlay = img_uint8.copy()
        blue_mask = np.zeros_like(bg_overlay)
        blue_mask[:, :, 2] = (bg_mask_upsampled * 255).astype(np.uint8)  # Blue channel
        bg_overlay = cv2.addWeighted(bg_overlay, 0.5, blue_mask, 0.5, 0)
        
        Image.fromarray(bg_overlay).save(bg_path)
        bg_ratio = bg_mask.sum() / bg_mask.size * 100
        logger.info(f"  Saved BG mask (≤{mean_val:.3f}): {bg_ratio:.1f}% - {bg_path}")
        
        logger.info(f"Saved all best frame visualizations to: {best_frame_dir}")

    def _aggregate_metrics(self, all_metrics):
        """Aggregate metrics across sequences"""
        metric_keys = all_metrics[0].keys()
        aggregated = {}

        for key in metric_keys:
            values = [m[key] for m in all_metrics if key in m]
            if values:
                aggregated[key] = np.mean(values)

        return aggregated


@hydra.main(version_base=None, config_path="configs/gear3", config_name="config")
def main(config: DictConfig):
    """Main entry point"""
    import os

    # Override config for testing
    config.inference = True

    tester = Gear3Tester(config)
    tester.test()


if __name__ == "__main__":
    main()
