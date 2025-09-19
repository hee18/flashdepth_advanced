import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import logging
import hydra
from omegaconf import DictConfig
import wandb
from tqdm import tqdm
import numpy as np
from pathlib import Path

from flashdepth.model import FlashDepth
from flashdepth.heads import MetricDepthLoss
from dataloaders.combined_dataset import CombinedDataset
from utils.helpers import *
from utils.eval_metrics.metrics import compute_depth_metrics


class MetricHeadTrainer:
    """
    Trainer class specifically for fine-tuning the Global Scale Predictor (GSP) head
    while keeping the base FlashDepth model frozen.
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.cuda.current_device()

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize model
        self.model = self._setup_model()

        # Setup data loaders
        self.train_loader, self.val_loader = self._setup_data_loaders()

        # Setup optimizer and loss function
        self.optimizer = self._setup_optimizer()
        self.loss_fn = MetricDepthLoss(loss_type=config.training.get('loss_type', 'l1'))

        # Setup wandb if enabled
        if config.training.get('wandb', False):
            wandb.init(
                project="flashdepth-metric-head",
                name=config.training.get('wandb_name', 'metric_head_experiment'),
                config=dict(config)
            )

        self.global_step = 0
        self.best_val_loss = float('inf')

    def _setup_model(self):
        """Initialize FlashDepth model with GSP head"""
        # Create model with metric head enabled
        model_config = dict(self.config.model)
        model_config['use_metric_head'] = True

        model = FlashDepth(**model_config)

        # Load pre-trained FlashDepth checkpoint
        if self.config.load and self.config.load != 'true':
            checkpoint_path = self.config.load
            if os.path.exists(checkpoint_path):
                self.logger.info(f"Loading FlashDepth checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location='cpu')

                # Load only the base model weights (exclude GSP head)
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in checkpoint.items()
                                 if k in model_dict and not k.startswith('gsp_head')}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                self.logger.info(f"Loaded {len(pretrained_dict)} parameters from checkpoint")
            else:
                self.logger.warning(f"Checkpoint path {checkpoint_path} does not exist")

        model = model.to(self.device)

        # Freeze all parameters except GSP head
        self._freeze_base_model(model)

        return model

    def _freeze_base_model(self, model):
        """Freeze all parameters except GSP head"""
        frozen_params = 0
        trainable_params = 0

        for name, param in model.named_parameters():
            if name.startswith('gsp_head'):
                param.requires_grad = True
                trainable_params += param.numel()
                self.logger.info(f"Trainable parameter: {name} - {param.shape}")
            else:
                param.requires_grad = False
                frozen_params += param.numel()

        self.logger.info(f"Frozen parameters: {frozen_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")

        if trainable_params == 0:
            raise ValueError("No trainable parameters found! Check GSP head initialization.")

    def _setup_data_loaders(self):
        """Setup training and validation data loaders"""
        # Training dataset
        train_dataset = CombinedDataset(
            root_dir=self.config.dataset.data_root,
            enable_dataset_flags=self.config.dataset.train_datasets,
            resolution=self.config.dataset.resolution,
            split='train',
            video_length=self.config.dataset.video_length,
            color_aug=False  # Disable augmentation for metric training
        )

        # Validation dataset
        val_dataset = CombinedDataset(
            root_dir=self.config.dataset.data_root,
            enable_dataset_flags=self.config.dataset.val_datasets,
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
        """Setup optimizer for GSP head parameters only"""
        # Get only GSP head parameters
        gsp_params = [p for name, p in self.model.named_parameters()
                      if name.startswith('gsp_head') and p.requires_grad]

        if len(gsp_params) == 0:
            raise ValueError("No GSP head parameters found for optimization!")

        optimizer = torch.optim.Adam(
            gsp_params,
            lr=self.config.training.lr.get('gsp', 1e-4),
            weight_decay=1e-6
        )

        self.logger.info(f"Optimizer initialized with {len(gsp_params)} parameter groups")
        return optimizer

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()

        # Only GSP head should be in training mode
        for name, module in self.model.named_modules():
            if not name.startswith('gsp_head') and not name == '':
                module.eval()

        total_loss = 0.0
        total_samples = 0

        pbar = tqdm(self.train_loader, desc=f'Training Epoch')

        for batch_idx, batch in enumerate(pbar):
            try:
                # Forward pass and loss computation
                loss, metrics = self.model.train_metric_head(batch, self.loss_fn)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_norm=1.0
                )

                self.optimizer.step()

                # Update statistics
                total_loss += loss.item()
                total_samples += 1
                self.global_step += 1

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'scale': f'{metrics["mean_scale"]:.3f}',
                    'shift': f'{metrics["mean_shift"]:.3f}'
                })

                # Log to wandb
                if self.config.training.get('wandb', False):
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/scale': metrics['mean_scale'],
                        'train/shift': metrics['mean_shift'],
                        'global_step': self.global_step
                    })

                # Validation and checkpointing
                if self.global_step % self.config.training.get('val_freq', 1000) == 0:
                    val_metrics = self.validate()
                    self.save_checkpoint(val_metrics['val_loss'])

            except Exception as e:
                self.logger.error(f"Error in training step {batch_idx}: {e}")
                continue

        avg_loss = total_loss / max(total_samples, 1)
        self.logger.info(f"Average training loss: {avg_loss:.4f}")

        return avg_loss

    def validate(self):
        """Validate the model"""
        self.model.eval()

        total_loss = 0.0
        total_samples = 0
        all_errors = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')

            for batch_idx, batch in enumerate(pbar):
                try:
                    video, gt_depth, dataset_name = batch
                    video = video.to(self.device)
                    gt_depth = gt_depth.to(self.device)

                    # Forward pass
                    outputs = self.model.forward_with_metric_head((video, gt_depth))
                    pred_metric = outputs['metric_depth']

                    # Compute loss
                    pred_flat = rearrange(pred_metric, 'b t h w -> (b t) h w')
                    gt_flat = rearrange(gt_depth, 'b t h w -> (b t) h w')
                    valid_mask = gt_flat >= 0

                    loss = self.loss_fn(pred_flat, gt_flat, valid_mask)
                    total_loss += loss.item()
                    total_samples += 1

                    # Compute depth metrics for first frame
                    if batch_idx == 0:  # Save computation time
                        metrics = compute_depth_metrics(
                            pred_metric[0, 0].cpu(),  # First batch, first frame
                            gt_depth[0, 0].cpu(),
                            valid_mask=gt_depth[0, 0].cpu() >= 0
                        )
                        all_errors.append(metrics)

                    pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})

                except Exception as e:
                    self.logger.error(f"Error in validation step {batch_idx}: {e}")
                    continue

        avg_loss = total_loss / max(total_samples, 1)

        # Compute average metrics
        if all_errors:
            avg_metrics = {k: np.mean([e[k] for e in all_errors])
                          for k in all_errors[0].keys()}
        else:
            avg_metrics = {}

        self.logger.info(f"Validation loss: {avg_loss:.4f}")
        for k, v in avg_metrics.items():
            self.logger.info(f"  {k}: {v:.4f}")

        # Log to wandb
        if self.config.training.get('wandb', False):
            log_dict = {'val/loss': avg_loss}
            for k, v in avg_metrics.items():
                log_dict[f'val/{k}'] = v
            wandb.log(log_dict)

        return {'val_loss': avg_loss, **avg_metrics}

    def save_checkpoint(self, val_loss):
        """Save model checkpoint"""
        # Create checkpoint directory
        checkpoint_dir = Path(self.config.get('config_dir', './checkpoints'))
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint = {
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': dict(self.config)
        }

        # Save latest checkpoint
        latest_path = checkpoint_dir / 'latest_metric_head.pth'
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = checkpoint_dir / 'best_metric_head.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with val_loss: {val_loss:.4f}")

        # Save periodic checkpoint
        if self.global_step % self.config.training.get('save_freq', 5000) == 0:
            periodic_path = checkpoint_dir / f'metric_head_step_{self.global_step}.pth'
            torch.save(checkpoint, periodic_path)

    def train(self):
        """Main training loop"""
        self.logger.info("Starting GSP head fine-tuning...")
        self.logger.info(f"Total training steps: {self.config.training.total_iters}")

        epoch = 0
        while self.global_step < self.config.training.total_iters:
            self.logger.info(f"Epoch {epoch + 1}")

            train_loss = self.train_epoch()

            # Final validation
            if self.global_step >= self.config.training.total_iters:
                self.logger.info("Training completed. Running final validation...")
                final_metrics = self.validate()
                self.save_checkpoint(final_metrics['val_loss'])
                break

            epoch += 1

        self.logger.info("Training completed!")


@hydra.main(version_base=None, config_path="configs/flashdepth", config_name="config")
def main(cfg: DictConfig):
    """Main training function"""

    # Initialize distributed training if needed
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)

    # Create trainer and start training
    trainer = MetricHeadTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()