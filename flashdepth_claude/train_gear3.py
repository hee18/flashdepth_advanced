"""
Gear3 Training Script: Feature-level Metric Depth Learning

Two-phase training:
    Phase 1: Train on MVS-Synth, PointOdyssey, Spring, TartanAir, DynamicReplica
    Phase 2: Fine-tune on nuScenes only

Key differences from baseline:
    - No scale/shift operation on depth map
    - Feature-level modulation using FiLM
    - Canonical space normalization (focal_length=1000)
    - Loss on inverse depth: loss(100/pred, 100/gt)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from tqdm import tqdm
import numpy as np
from pathlib import Path
from einops import rearrange
import math

from flashdepth.model import FlashDepth
from flashdepth.gear3_modules import Gear3MetricHead
from dataloaders.combined_dataset import CombinedDataset
from utils.helpers import *
from utils.metric_depth_metrics import MetricDepthMetrics


class CanonicalSpaceNormalizer:
    """
    Canonical space normalization using fixed focal length.

    Converts metric depth to canonical space:
        depth_canonical = depth * (focal_canonical / focal_actual)

    And de-canonicalizes predictions:
        depth_actual = depth_canonical * (focal_actual / focal_canonical)
    """
    def __init__(self, focal_canonical=1000.0, enable=True):
        self.focal_canonical = focal_canonical
        self.enable = enable
        logging.info(f"Canonical space normalization: {'enabled' if enable else 'disabled'} (f={focal_canonical})")

    def canonicalize(self, depth, focal_length):
        """Convert metric depth to canonical space"""
        if not self.enable:
            return depth

        if isinstance(focal_length, (int, float)):
            focal_length = torch.tensor(focal_length, device=depth.device, dtype=depth.dtype)

        # depth_canonical = depth * (focal_canonical / focal_actual)
        scale_factor = self.focal_canonical / focal_length
        return depth * scale_factor.view(-1, 1, 1, 1)

    def decanonicalize(self, depth_canonical, focal_length):
        """Convert canonical space depth back to metric depth"""
        if not self.enable:
            return depth_canonical

        if isinstance(focal_length, (int, float)):
            focal_length = torch.tensor(focal_length, device=depth_canonical.device, dtype=depth_canonical.dtype)

        # depth_actual = depth_canonical * (focal_actual / focal_canonical)
        scale_factor = focal_length / self.focal_canonical
        return depth_canonical * scale_factor.view(-1, 1, 1, 1)


class InverseDepthLoss(nn.Module):
    """
    Inverse depth loss for metric depth learning.

    Loss = L1(100/pred_depth, 100/gt_depth)
    """
    def __init__(self, inverse_scale=100.0):
        super().__init__()
        self.inverse_scale = inverse_scale

    def forward(self, pred_depth, gt_depth, valid_mask=None):
        """
        Args:
            pred_depth: [B, 1, H, W] predicted metric depth
            gt_depth: [B, 1, H, W] ground truth metric depth
            valid_mask: [B, 1, H, W] valid pixels (optional)

        Returns:
            loss: scalar
        """
        # Convert to inverse depth: 100/depth
        pred_inverse = self.inverse_scale / (pred_depth + 1e-8)
        gt_inverse = self.inverse_scale / (gt_depth + 1e-8)

        # L1 loss
        loss = F.l1_loss(pred_inverse, gt_inverse, reduction='none')

        # Apply valid mask if provided
        if valid_mask is not None:
            loss = loss * valid_mask
            loss = loss.sum() / (valid_mask.sum() + 1e-8)
        else:
            loss = loss.mean()

        return loss


class Gear3Trainer:
    """
    Trainer for Gear3 metric depth learning.

    Frozen: DINOv2, DPT
    Fine-tuned: Mamba (LR: 1e-5)
    Trained: Gear3 modules (LR: 1e-4)
    """
    def __init__(self, config):
        self.config = config
        self.phase = config.get('phase', 1)  # Training phase (1 or 2)

        # Setup device
        gpu_id = config.get('gpu', 0)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        self.device = "cuda:0"
        logging.info(f"Training Phase {self.phase} on GPU {gpu_id}")

        # Setup results directory
        phase_suffix = f"_phase{self.phase}"
        self.results_dir = Path(config.get('results_dir', f'./train_results/gear3{phase_suffix}'))
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.results_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Results directory: {self.results_dir}")
        self.logger.info(f"Training phase: {self.phase}")

        # Setup canonical space normalizer
        self.canonical_normalizer = CanonicalSpaceNormalizer(
            focal_canonical=config.get('canonical_focal_length', 1000.0),
            enable=config.get('use_canonical_space', True)
        )

        # Initialize model
        self.model = self._setup_model()

        # Setup data loaders
        self.train_loader, self.val_loader = self._setup_data_loaders()

        # Setup optimizer and loss
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.loss_fn = InverseDepthLoss(inverse_scale=100.0)

        # Setup wandb
        if config.training.get('wandb', False):
            wandb.init(
                project="flashdepth-gear3",
                name=f"gear3_phase{self.phase}_{config.training.get('wandb_name', 'experiment')}",
                config=dict(config)
            )

        self.global_step = 0
        self.best_val_loss = float('inf')

    def _setup_model(self):
        """Initialize FlashDepth with Gear3 metric head"""
        # Create base FlashDepth model
        model_config = dict(self.config.model)
        model_config['batch_size'] = self.config.training.batch_size
        model_config['use_metric_head'] = False  # Don't use GSP head

        model = FlashDepth(**model_config)

        # Load pre-trained checkpoint (DINOv2 + DPT only, excluding Mamba)
        checkpoint_path = self.config.get('load')
        if checkpoint_path and checkpoint_path != 'true':
            if os.path.exists(checkpoint_path):
                self.logger.info(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location='cpu')

                # Extract state dict
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                # Remove module. prefix if present
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

                # Load only DINOv2 and DPT refinement layers
                # Exclude: Mamba (modulated input), output_conv1/2 (modulated features)
                loaded_dict = {}
                excluded_keys = []
                for k, v in state_dict.items():
                    # Exclude modules that receive modulated features
                    if any(x in k for x in ['mamba', 'hybrid_fusion', 'teacher_model',
                                            'output_conv1', 'output_conv2']):
                        excluded_keys.append(k)
                    else:
                        loaded_dict[k] = v

                # Load state dict (strict=False to allow missing modules)
                model.load_state_dict(loaded_dict, strict=False)
                self.logger.info(f"Loaded {len(loaded_dict)} parameters from checkpoint")
                self.logger.info(f"  - DINOv2 encoder: ✓")
                self.logger.info(f"  - DPT projects/resize/refinenet: ✓")
                self.logger.info(f"Excluded {len(excluded_keys)} parameters for training:")
                self.logger.info(f"  - Mamba (modulated input)")
                self.logger.info(f"  - output_conv1/2 (modulated features)")
                self.logger.info(f"  - hybrid_fusion, teacher_model (not used)")
            else:
                self.logger.warning(f"Checkpoint {checkpoint_path} not found")

        # Add Gear3 metric head
        embed_dim = 1024 if model.encoder == 'vitl' else 384
        dpt_dim = 256 if model.encoder == 'vitl' else 64
        num_heads = 16 if model.encoder == 'vitl' else 6

        model.gear3_head = Gear3MetricHead(
            embed_dim=embed_dim,
            dpt_dim=dpt_dim,
            num_heads=num_heads,
            num_dpt_layers=4
        )

        model = model.to(self.device)

        # Freeze and configure parameters
        self._configure_parameters(model)

        return model

    def _configure_parameters(self, model):
        """
        Freeze: DINOv2, DPT refinement layers
        Train from scratch: Mamba, output_conv1/2, Gear3 modules
        """
        frozen_params = 0
        mamba_params = 0
        output_conv_params = 0
        gear3_params = 0

        for name, param in model.named_parameters():
            if 'gear3_head' in name:
                # Gear3 modules: trainable
                param.requires_grad = True
                gear3_params += param.numel()
                self.logger.info(f"Trainable (Gear3): {name} - {param.shape}")

            elif 'mamba' in name:
                # Mamba: train from scratch (receives modulated input)
                param.requires_grad = True
                mamba_params += param.numel()

            elif 'output_conv' in name:
                # DPT output head: train from scratch (receives modulated features)
                param.requires_grad = True
                output_conv_params += param.numel()
                self.logger.info(f"Trainable (output_conv): {name} - {param.shape}")

            else:
                # Everything else (DINOv2, DPT refinement): frozen
                param.requires_grad = False
                frozen_params += param.numel()

        self.logger.info(f"Frozen (DINOv2 + DPT refinement): {frozen_params:,}")
        self.logger.info(f"Trainable (Mamba, from scratch): {mamba_params:,}")
        self.logger.info(f"Trainable (output_conv, from scratch): {output_conv_params:,}")
        self.logger.info(f"Trainable (Gear3 modules): {gear3_params:,}")

        total_trainable = mamba_params + output_conv_params + gear3_params
        self.logger.info(f"Total trainable: {total_trainable:,}")

    def _setup_data_loaders(self):
        """Setup phase-specific data loaders"""
        if self.phase == 1:
            # Phase 1: All FlashDepth datasets
            train_datasets = ['mvs-synth', 'pointodyssey', 'spring', 'tartanair', 'dynamicreplica']
            val_datasets = ['tartanair']  # Use TartanAir for validation
        else:
            # Phase 2: nuScenes only
            train_datasets = ['nuscenes']
            val_datasets = ['nuscenes']

        self.logger.info(f"Phase {self.phase} - Train datasets: {train_datasets}")
        self.logger.info(f"Phase {self.phase} - Val datasets: {val_datasets}")

        # Training dataset
        train_dataset = CombinedDataset(
            root_dir=self.config.dataset.data_root,
            enable_dataset_flags=train_datasets,
            resolution=self.config.dataset.resolution,
            split='train',
            video_length=self.config.dataset.video_length,
            color_aug=False  # No augmentation for metric training
        )

        # Validation dataset
        val_dataset = CombinedDataset(
            root_dir=self.config.dataset.data_root,
            enable_dataset_flags=val_datasets,
            resolution=self.config.dataset.resolution,
            split='val',
            video_length=self.config.dataset.video_length
        )

        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.workers,
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.workers,
            pin_memory=True,
            drop_last=False
        )

        self.logger.info(f"Train dataset size: {len(train_dataset)}")
        self.logger.info(f"Val dataset size: {len(val_dataset)}")

        return train_loader, val_loader

    def _setup_optimizer(self):
        """Setup optimizer - same LR for Mamba and Gear3 (both train from scratch)"""
        mamba_lr = self.config.training.get('mamba_lr', 1e-4)  # Same as gear3_lr
        gear3_lr = self.config.training.get('gear3_lr', 1e-4)

        # Separate parameter groups (for monitoring, both use same LR)
        mamba_params = []
        gear3_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'gear3_head' in name:
                    gear3_params.append(param)
                elif 'mamba' in name:
                    mamba_params.append(param)

        param_groups = [
            {'params': gear3_params, 'lr': gear3_lr, 'name': 'gear3'},
            {'params': mamba_params, 'lr': mamba_lr, 'name': 'mamba'}
        ]

        optimizer = torch.optim.Adam(
            param_groups,
            weight_decay=self.config.training.get('weight_decay', 1e-6)
        )

        self.logger.info(f"Optimizer: Gear3 LR={gear3_lr}, Mamba LR={mamba_lr} (both from scratch)")

        return optimizer

    def _setup_scheduler(self):
        """Setup cosine annealing scheduler with warmup"""
        total_steps = self.config.training.iterations
        warmup_steps = int(total_steps * 0.1)  # 10% warmup
        decay_start = int(total_steps * 0.3)  # Start decay at 30%

        def lr_lambda(step):
            if step < warmup_steps:
                # Warmup: 0.1x -> 1x
                return 0.1 + 0.9 * (step / warmup_steps)
            elif step < decay_start:
                # Stable phase
                return 1.0
            else:
                # Cosine decay: 1.0 -> 0.01
                progress = (step - decay_start) / (total_steps - decay_start)
                return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        return scheduler

    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")

        train_iterator = iter(self.train_loader)

        for step in range(self.config.training.iterations):
            self.global_step = step

            # Get batch
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_loader)
                batch = next(train_iterator)

            # Training step
            loss_dict = self.train_step(batch)

            # Logging
            if step % self.config.training.get('log_freq', 100) == 0:
                lr_gear3 = self.optimizer.param_groups[0]['lr']
                lr_mamba = self.optimizer.param_groups[1]['lr']
                self.logger.info(
                    f"Step {step}/{self.config.training.iterations} | "
                    f"Loss: {loss_dict['loss']:.4f} | "
                    f"LR: Gear3={lr_gear3:.6f}, Mamba={lr_mamba:.6f}"
                )

                if self.config.training.get('wandb', False):
                    wandb.log({**loss_dict, 'lr_gear3': lr_gear3, 'lr_mamba': lr_mamba}, step=step)

            # Validation
            if step % self.config.training.get('val_freq', 1000) == 0 and step > 0:
                val_metrics = self.validate()
                self.logger.info(f"Validation at step {step}: {val_metrics}")

                if self.config.training.get('wandb', False):
                    wandb.log({f'val/{k}': v for k, v in val_metrics.items()}, step=step)

                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint(f'best_phase{self.phase}.pth')

            # Save checkpoint
            if step % self.config.training.get('save_freq', 5000) == 0 and step > 0:
                self.save_checkpoint(f'iter_{step}_phase{self.phase}.pth')

        # Final save
        self.save_checkpoint(f'final_phase{self.phase}.pth')
        self.logger.info("Training completed!")

    def train_step(self, batch):
        """Single training step"""
        self.model.train()

        # Move batch to device
        images = batch['image'].to(self.device)  # [B, T, 3, H, W]
        gt_depth = batch['depth'].to(self.device)  # [B, T, 1, H, W]
        focal_length = batch.get('focal_length', 1000.0)  # Can be tensor or scalar

        B, T = images.shape[:2]

        # Canonicalize GT depth
        gt_depth_canonical = self.canonical_normalizer.canonicalize(gt_depth, focal_length)

        # Convert inverse depth to metric depth for GT
        # GT is stored as inverse depth, convert to metric
        gt_depth_metric = 1.0 / (gt_depth_canonical + 1e-8)

        # Forward pass through model
        total_loss = 0
        for t in range(T):
            img_t = images[:, t]  # [B, 3, H, W]
            gt_t = gt_depth_metric[:, t]  # [B, 1, H, W]

            # Extract features from DINOv2
            with torch.no_grad():
                features = self.model.pretrained(img_t, is_training=False)
                encoder_features = [features[idx] for idx in self.model.intermediate_layer_idx[self.model.encoder]]
                cls_token = features[-1][:, 0]  # [B, embed_dim]
                patch_tokens = features[-1][:, 1:]  # [B, num_patches, embed_dim]
                attention_weights = features[-2]  # [B, num_heads, num_patches+1, num_patches+1]

            # Get DPT features (before modulation)
            h, w = img_t.shape[2:]
            patch_h, patch_w = h // self.model.patch_size, w // self.model.patch_size

            with torch.no_grad():
                dpt_features = self.model.depth_head.get_forward_features(
                    encoder_features, patch_h, patch_w
                )  # List of 4 features: [path_4, path_3, path_2, path_1]

            # Apply Gear3 modulation
            modulated_dpt_features, importance_map = self.model.gear3_head(
                patch_tokens, attention_weights, dpt_features, patch_h, patch_w
            )

            # Use modulated path_1 (last feature) for depth prediction
            path_1_modulated = modulated_dpt_features[-1]

            # Pass through DPT output head to get depth
            # Note: output_conv1/2 are trainable (receive modulated features)
            out = self.model.depth_head.scratch.output_conv1(path_1_modulated)
            out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
            out = self.model.depth_head.scratch.output_conv2(out)  # [B, 1, H, W]

            pred_depth = out  # Metric depth in canonical space

            # Compute loss on inverse depth
            valid_mask = (gt_t > 0).float()
            loss_t = self.loss_fn(pred_depth, gt_t, valid_mask)
            total_loss += loss_t

        # Average loss over time
        loss = total_loss / T

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        return {'loss': loss.item()}

    @torch.no_grad()
    def validate(self):
        """Validation loop"""
        self.model.eval()

        total_loss = 0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            images = batch['image'].to(self.device)
            gt_depth = batch['depth'].to(self.device)
            focal_length = batch.get('focal_length', 1000.0)

            B, T = images.shape[:2]

            # Canonicalize GT
            gt_depth_canonical = self.canonical_normalizer.canonicalize(gt_depth, focal_length)
            gt_depth_metric = 1.0 / (gt_depth_canonical + 1e-8)

            # Forward pass (only first frame for validation speed)
            img_t = images[:, 0]
            gt_t = gt_depth_metric[:, 0]

            # Extract features
            features = self.model.pretrained(img_t, is_training=False)
            encoder_features = [features[idx] for idx in self.model.intermediate_layer_idx[self.model.encoder]]
            patch_tokens = features[-1][:, 1:]
            attention_weights = features[-2]

            # Get DPT features
            h, w = img_t.shape[2:]
            patch_h, patch_w = h // self.model.patch_size, w // self.model.patch_size
            dpt_features = self.model.depth_head.get_forward_features(encoder_features, patch_h, patch_w)

            # Apply modulation
            modulated_dpt_features, _ = self.model.gear3_head(
                patch_tokens, attention_weights, dpt_features, patch_h, patch_w
            )

            # Get depth
            path_1_modulated = modulated_dpt_features[-1]
            out = self.model.depth_head.scratch.output_conv1(path_1_modulated)
            out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
            out = self.model.depth_head.scratch.output_conv2(out)

            pred_depth = out

            # Compute loss
            valid_mask = (gt_t > 0).float()
            loss = self.loss_fn(pred_depth, gt_t, valid_mask)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {'loss': avg_loss}

    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint_path = self.results_dir / filename

        checkpoint = {
            'global_step': self.global_step,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': OmegaConf.to_container(self.config, resolve=True),
            'phase': self.phase
        }

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")


@hydra.main(version_base=None, config_path="configs/gear3", config_name="config")
def main(config: DictConfig):
    """Main entry point"""
    trainer = Gear3Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
