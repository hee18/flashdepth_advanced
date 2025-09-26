import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import hydra
from omegaconf import DictConfig
import wandb
from tqdm import tqdm
import numpy as np
from pathlib import Path
from einops import rearrange

from flashdepth.model import FlashDepth
from flashdepth.heads import MetricDepthLoss
from dataloaders.combined_dataset import CombinedDataset
from utils.helpers import *
from utils.metric_depth_metrics import MetricDepthMetrics
try:
    from utils.metric_visualization import MetricDepthVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Visualization not available: {e}")
    VISUALIZATION_AVAILABLE = False
    MetricDepthVisualizer = None


class MetricHeadTrainer:
    """
    Trainer class specifically for fine-tuning the Global Scale Predictor (GSP) head
    while keeping the base FlashDepth model frozen.
    """

    def __init__(self, config):
        self.config = config

        # Single GPU setup for GSP training
        gpu_id = config.get('gpu', 0)

        # Set CUDA_VISIBLE_DEVICES to make only the specified GPU visible
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        self.device = "cuda:0"  # Always cuda:0 since CUDA_VISIBLE_DEVICES maps the specified GPU
        self.use_multi_gpu = False
        logging.info("Using single GPU for GSP training")

        # Setup results directory
        self.results_dir = Path(config.get('results_dir', './train_results/results_1'))
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging with file output
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.results_dir / 'training.log'),
                logging.StreamHandler()  # Also log to console
            ]
        )
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Results will be saved to: {self.results_dir}")

        # Initialize visualizer
        if VISUALIZATION_AVAILABLE:
            self.visualizer = MetricDepthVisualizer(save_dir=self.results_dir / "visualizations")
            self.logger.info("Visualization enabled")
        else:
            self.visualizer = None
            self.logger.warning("Visualization disabled - missing dependencies")

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
        model_config['batch_size'] = self.config.training.batch_size

        model = FlashDepth(**model_config)

        # No DataParallel for single GPU training

        # Load pre-trained FlashDepth checkpoint
        checkpoint_path = self.config.get('flashdepth_checkpoint')
        if not checkpoint_path:
            # Default checkpoint path
            checkpoint_path = "configs/flashdepth-l/iter_10001.pth"
            self.logger.info(f"No flashdepth_checkpoint specified, using default: {checkpoint_path}")

        if self.config.load and self.config.load != 'true':
            # Override with explicit load path if provided
            checkpoint_path = self.config.load

        if checkpoint_path:
            if os.path.exists(checkpoint_path):
                self.logger.info(f"Loading FlashDepth checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location='cpu')

                # Extract state dict from checkpoint - handle different checkpoint formats
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                self.logger.info(f"Checkpoint keys (first 10): {list(state_dict.keys())[:10]}")

                # Get current model state dict (no DataParallel, so no module. prefix)
                model_dict = model.state_dict()

                # Simple key mapping - just exclude GSP head keys
                pretrained_dict = {}
                loaded_keys = []
                missing_keys = []

                for checkpoint_key, checkpoint_value in state_dict.items():
                    # Remove 'module.' prefix if present in checkpoint
                    clean_key = checkpoint_key.replace('module.', '')

                    # Load if key exists in model and not GSP head
                    if clean_key in model_dict and not clean_key.startswith('gsp_head'):
                        pretrained_dict[clean_key] = checkpoint_value
                        loaded_keys.append(clean_key)
                    elif not clean_key.startswith('gsp_head'):
                        missing_keys.append(clean_key)

                # Update model dict and load state
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)

                self.logger.info(f"Loaded {len(pretrained_dict)} parameters from checkpoint")
                self.logger.info(f"Successfully loaded keys (first 10): {loaded_keys[:10]}")
                if missing_keys:
                    self.logger.warning(f"Missing keys (first 10): {missing_keys[:10]}")
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

        # Debug: Print all parameter names to see the structure
        # self.logger.info("All model parameters:")
        # for name, param in model.named_parameters():
        #     self.logger.info(f"  {name} - {param.shape}")

        for name, param in model.named_parameters():
            # No DataParallel, so no module. prefix handling needed
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
            drop_last=False,
            collate_fn=self.collate_fn
        )

        self.logger.info(f"Train dataset size: {len(train_dataset)}")
        self.logger.info(f"Val dataset size: {len(val_dataset)}")

        return train_loader, val_loader

    def collate_fn(self, batch):
        """Custom collate function to filter out None values"""
        # Filter out None values
        batch = [item for item in batch if item is not None]

        # If all items are None, return None
        if len(batch) == 0:
            return None

        # Use default collate for non-None items
        from torch.utils.data.dataloader import default_collate
        return default_collate(batch)

    def _setup_optimizer(self):
        """Setup optimizer for GSP head parameters only"""
        # Get only GSP head parameters
        gsp_params = []
        for name, param in self.model.named_parameters():
            # Handle DataParallel wrapper - parameters have 'module.' prefix
            param_name = name.replace('module.', '') if name.startswith('module.') else name
            if param_name.startswith('gsp_head') and param.requires_grad:
                gsp_params.append(param)
                self.logger.info(f"Optimizer parameter: {name} - {param.shape}")

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
                # Forward pass and loss computation with mixed precision
                model_fn = self.model.module if self.use_multi_gpu else self.model
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    loss, metrics = model_fn.train_metric_head(batch, self.loss_fn)

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

                # Training visualization (every 250 steps)
                vis_steps = [1, 10, 50, 100]
                if (self.global_step in vis_steps or self.global_step % 250 == 0 ) and self.visualizer:
                    try:
                        # Create a simple visualization using current training batch
                        self.model.eval()
                        with torch.no_grad():
                            model_fn = self.model.module if self.use_multi_gpu else self.model
                            train_outputs = model_fn.forward_with_metric_head((batch[0].to(self.device), batch[1].to(self.device)))
                            self.visualizer.create_validation_summary(
                                (batch[0], batch[1], batch[2]), train_outputs, self.global_step, prefix="training"
                            )
                            self.logger.info(f"Training visualization saved at step {self.global_step}")
                        self.model.train()
                        # Keep only GSP head in training mode
                        for name, module in self.model.named_modules():
                            if not name.startswith('gsp_head') and not name == '':
                                module.eval()
                    except Exception as viz_e:
                        self.logger.warning(f"Failed to save training visualization: {viz_e}")

                # Validation (every val_freq steps)
                if self.global_step % self.config.training.get('val_freq', 1000) == 0:
                    try:
                        val_metrics = self.validate()
                        self.save_checkpoint(val_metrics['val_loss'])
                    except Exception as val_e:
                        self.logger.error(f"Validation failed: {val_e}")
                        # Save checkpoint even if validation fails
                        self.save_checkpoint(loss.item())

                # Periodic checkpoint saving (every save_freq steps, independent of validation)
                elif self.global_step % self.config.training.get('save_freq', 1000) == 0:
                    self.save_checkpoint(loss.item())

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
                    # Skip None batches (no valid pairs found)
                    if batch is None:
                        continue

                    video, gt_depth, dataset_name = batch

                    # Forward pass with mixed precision
                    model_fn = self.model.module if self.use_multi_gpu else self.model
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        outputs = model_fn.forward_with_metric_head((video, gt_depth))
                    pred_metric = outputs['metric_depth']

                    # Compute loss (convert GT to metric depth space)
                    pred_flat = rearrange(pred_metric, 'b t h w -> (b t) h w')
                    gt_flat = rearrange(gt_depth, 'b t h w -> (b t) h w')
                    # For TartanAir: GT is already metric depth, no conversion needed

                    # DEBUG: Check shapes for loss computation
                    if batch_idx == 0:  # Only log first batch to avoid spam
                        self.logger.info(f"DEBUG TRAIN LOSS - Pred flat shape: {pred_flat.shape}")
                        self.logger.info(f"DEBUG TRAIN LOSS - GT flat shape: {gt_flat.shape}")

                    # Handle shape mismatch for loss computation if needed
                    if pred_flat.shape != gt_flat.shape:
                        self.logger.warning(f"Loss computation shape mismatch! Pred: {pred_flat.shape}, GT: {gt_flat.shape}")
                        # Resize pred to match GT for loss computation
                        import torch.nn.functional as F
                        pred_flat = F.interpolate(
                            pred_flat.unsqueeze(1),  # Add channel dimension
                            size=gt_flat.shape[-2:],
                            mode='bilinear',
                            align_corners=True
                        ).squeeze(1)  # Remove channel dimension
                        self.logger.info(f"Resized pred for loss to: {pred_flat.shape}")

                    # Create valid mask considering both GT and pred ranges
                    gt_valid_mask = gt_flat > 0  # GT valid pixels
                    pred_valid_mask = (pred_flat > 0) & (pred_flat < 1000.0)  # Pred in reasonable range
                    valid_mask = gt_valid_mask & pred_valid_mask

                    loss = self.loss_fn(pred_flat, gt_flat, valid_mask)
                    total_loss += loss.item()
                    total_samples += 1

                    # Compute depth metrics for first frame
                    if batch_idx == 0:  # Save computation time
                        gt_metric = gt_depth[0, 0].cpu()
                        pred_metric_cpu = pred_metric[0, 0].cpu()

                        # Create valid mask considering both GT and pred ranges
                        gt_valid_mask = gt_metric > 0  # GT valid pixels
                        pred_valid_mask = (pred_metric_cpu > 0) & (pred_metric_cpu < 1000.0)  # Pred in reasonable range

                        # DEBUG: Check shapes before creating valid mask
                        self.logger.info(f"DEBUG TRAIN - GT valid mask shape: {gt_valid_mask.shape}")
                        self.logger.info(f"DEBUG TRAIN - Pred valid mask shape: {pred_valid_mask.shape}")

                        # Handle shape mismatch if needed
                        if gt_valid_mask.shape != pred_valid_mask.shape:
                            self.logger.warning(f"Shape mismatch in train! GT: {gt_valid_mask.shape}, Pred: {pred_valid_mask.shape}")
                            # Resize pred_valid_mask to match GT
                            import torch.nn.functional as F
                            pred_valid_mask = F.interpolate(
                                pred_valid_mask.unsqueeze(0).unsqueeze(0).float(),
                                size=gt_valid_mask.shape,
                                mode='nearest'
                            ).squeeze(0).squeeze(0).bool()
                            self.logger.info(f"Resized pred valid mask to: {pred_valid_mask.shape}")

                        valid_mask = gt_valid_mask & pred_valid_mask
                        metrics = MetricDepthMetrics.compute_metric_depth_metrics(
                            pred_metric[0, 0].cpu(),  # First batch, first frame (already in metric depth)
                            gt_metric,
                            valid_mask=valid_mask
                        )
                        all_errors.append(metrics)

                        # Save visualization every save_freq steps
                        if self.global_step % self.config.training.get('save_freq', 1000) == 0 and self.visualizer:
                            try:
                                self.visualizer.create_validation_summary(
                                    batch, outputs, self.global_step
                                )
                                self.logger.info(f"Validation visualization saved at step {self.global_step}")
                            except Exception as viz_e:
                                self.logger.warning(f"Failed to save visualization: {viz_e}")

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
        # Use the configured results directory
        checkpoint_dir = self.results_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Get model state dict (handle DataParallel wrapper)
        model_state = self.model.module.state_dict() if self.use_multi_gpu else self.model.state_dict()

        checkpoint = {
            'global_step': self.global_step,
            'model_state_dict': model_state,
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
            best_path = checkpoint_dir / f'best_metric_head_step_{self.global_step}.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved at step {self.global_step} with val_loss: {val_loss:.4f}")

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

    # Create trainer and start training
    trainer = MetricHeadTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()