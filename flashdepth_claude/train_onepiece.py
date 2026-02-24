"""
Onepiece V2 Training Script: Unified Global Mamba for Metric Depth Estimation

Architecture:
    DPT features → GAP(256) + GStdP(256) → Unified Global Mamba(512-dim, 4 layers) →
    FiLMGenerator → γ,β → modulate DPT features →
    Relative Head → relative depth
    MetricHead (Conv) → scale, shift → metric depth

Training Phases:
    Phase 1 (0 ~ 1.5K steps): Metric Alignment
        Trainable: UnifiedGlobalMamba + OnepieceMetricHead
        Frozen: DINOv2, DPT, FiLMGenerator, RelativeHead (output_conv)

    Phase 2 (1.5K+ steps): Full Video Optimization
        Additional unfreeze: FiLMGenerator + RelativeHead
        Frozen: DINOv2, DPT

Loss:
    Phase 1: L_total = L_log_l1 + L_tgm  (full graph)
    Phase 2: L_total = L_log_l1 + L_tgm + L_ofc

Data:
    Gear5 8-element batch format, video_length=8, resolution=518
    Training: TartanAir (metric GT required)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
import logging
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import wandb
from tqdm import tqdm
import numpy as np
from pathlib import Path
from einops import rearrange
import math
import time
from datetime import timedelta

from flashdepth.model import FlashDepth
from dataloaders.combined_dataset import CombinedDataset
from utils.helpers import *
from utils.metric_depth_metrics import MetricDepthMetrics
from utils.onepiece_losses import OnepieceCombinedLoss
from utils.flow_estimator import FlowEstimator
from utils.onepiece_visualization import OnepieceVisualizer


def init_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            timeout=timedelta(seconds=3600),
            rank=rank,
            world_size=world_size
        )
        dist.barrier()

    return rank, world_size, local_rank


class OnepieceTrainer:
    """
    Trainer for Onepiece V2 unified metric depth estimation.

    Phase 1 (0 ~ auto_transition_step):
        Trainable: UnifiedGlobalMamba + OnepieceMetricHead
        Frozen: DINOv2, DPT, FiLMGenerator, RelativeHead

    Phase 2 (auto_transition_step+):
        Additional unfreeze: FiLMGenerator + RelativeHead (final_head/output_conv)
        Frozen: DINOv2, DPT
    """

    def __init__(self, config, rank, world_size, local_rank):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank

        # Phase configuration
        self.auto_transition_step = config.phase.get('auto_transition_step', 5000)
        self.phase2_warmup_steps = config.phase.get('phase2_warmup_steps', 500)
        self.current_phase = 1  # Start with Phase 1

        # No-shift mode
        self.no_shift = config.get('no_shift', False)

        # Device setup
        self.device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)

        # Results directory
        self.results_dir = Path(config.get('results_dir', './train_results/onepiece'))
        if rank == 0:
            self.results_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        if rank == 0:
            self.logger.info(f"=== ONEPIECE TRAINING ===")
            self.logger.info(f"  Auto phase transition: Phase 1 → Phase 2 at step {self.auto_transition_step}")
            self.logger.info(f"  Phase 2 warmup: {self.phase2_warmup_steps} steps")
            self.logger.info(f"  No-shift (scale only): {self.no_shift}")
            self.logger.info(f"  Training on {world_size} GPU(s)")

        # Initialize model
        self.model = self._setup_model()
        self._configure_parameters_phase1()
        self._set_train_mode()

        # Setup data loaders
        self.train_loader, self.val_loader = self._setup_data_loaders()

        # Setup optimizer and scheduler (Phase 1)
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        # Loss function
        loss_config = config.get('loss', {})
        self.loss_fn = OnepieceCombinedLoss(
            log_l1_weight=loss_config.get('log_l1_weight', 1.0),
            tgm_weight=loss_config.get('tgm_weight', 1.0),
            ofc_weight=loss_config.get('ofc_weight', 0.01),
            use_log_space=loss_config.get('use_log_space', True)
        )

        # Sea-RAFT flow estimator (frozen, for feature consistency loss)
        flow_config = config.get('flow', {})
        flow_checkpoint = flow_config.get('checkpoint',
            'third_party/SEA-RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth')
        try:
            self.flow_estimator = FlowEstimator(
                checkpoint_path=flow_checkpoint,
                device=self.device
            )
            if rank == 0:
                self.logger.info(f"Sea-RAFT loaded from {flow_checkpoint}")
        except (ImportError, FileNotFoundError) as e:
            raise RuntimeError(
                f"Sea-RAFT is REQUIRED for Onepiece training. Error: {e}\n"
                f"Install: git clone https://github.com/princeton-vl/SEA-RAFT.git third_party/SEA-RAFT/"
            )

        # Setup visualizers
        if rank == 0:
            self.train_visualizer = OnepieceVisualizer(
                save_dir=self.results_dir / "visualizations" / "train"
            )
            self.val_visualizer = OnepieceVisualizer(
                save_dir=self.results_dir / "visualizations" / "valid"
            )
        else:
            self.train_visualizer = None
            self.val_visualizer = None

        # WandB
        if config.training.get('wandb', False) and rank == 0:
            wandb.init(
                project="flashdepth-onepiece",
                name=config.training.get('wandb_name', 'onepiece'),
                config=dict(config)
            )

        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_step = 0
        self.current_val_loss = None
        self.dataset_losses = None
        self.num_sequences = None

        # Per-dataset validation tracking (match Gear5)
        self.val_vis_config = {
            'sintel': {'sequences': [0, 4, 7], 'saved': []},
            'waymo_seg': {'sequences': [0, 1, 2, 3, 4, 5, 6, 7], 'saved': []}
        }

    def _setup_logging(self):
        """Setup logging with file and console handlers."""
        class FlushFileHandler(logging.FileHandler):
            def emit(self, record):
                super().emit(record)
                self.flush()

        if self.rank == 0:
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.INFO)
            root_logger.handlers.clear()

            file_handler = FlushFileHandler(self.results_dir / 'training.log', mode='a')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)
            stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

            root_logger.addHandler(file_handler)
            root_logger.addHandler(stream_handler)
        else:
            logging.basicConfig(level=logging.ERROR)

        self.logger = logging.getLogger(__name__)

    def _get_canonical_focal_length(self):
        """Get canonical focal length (fixed at 500.0 for 518x518 resolution)."""
        return 500.0

    def _setup_model(self):
        """Initialize FlashDepth with Onepiece V2 modules."""
        model_config = dict(self.config.model)
        model_config['batch_size'] = self.config.training.batch_size
        model_config['use_metric_head'] = False  # Don't use original GSP head
        model_config['use_onepiece'] = True

        model = FlashDepth(**model_config)

        # Load pre-trained FlashDepth checkpoint
        checkpoint_path = self.config.get('load')
        if checkpoint_path and os.path.exists(checkpoint_path):
            if self.rank == 0:
                self.logger.info(f"Loading FlashDepth checkpoint from {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Remove module. prefix
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            # Exclude onepiece-specific keys (train from scratch)
            loaded_dict = {}
            excluded_keys = []
            for k, v in state_dict.items():
                if any(x in k for x in ['unified_global_mamba', 'onepiece_metric_head', 'onepiece_film_generator']):
                    excluded_keys.append(k)
                else:
                    loaded_dict[k] = v

            model.load_state_dict(loaded_dict, strict=False)
            if self.rank == 0:
                self.logger.info(f"Loaded {len(loaded_dict)} parameters from FlashDepth checkpoint")
                self.logger.info(f"Excluded {len(excluded_keys)} onepiece parameters (will train from scratch)")
        elif self.rank == 0:
            self.logger.warning(f"Checkpoint not found: {checkpoint_path}")

        model = model.to(self.device)

        # Apply gradient checkpointing
        if self.config.training.get('gradient_checkpointing', False):
            if self.rank == 0:
                self.logger.info("Applying gradient checkpointing to ViT and DPT")
            apply_activation_checkpointing(
                model.pretrained,
                checkpoint_wrapper_fn=checkpoint_wrapper,
                check_fn=lambda _: True
            )
            apply_activation_checkpointing(
                model.depth_head,
                checkpoint_wrapper_fn=checkpoint_wrapper,
                check_fn=lambda _: True
            )

        # Wrap with DDP if multi-GPU
        if self.world_size > 1:
            model = DDP(
                model,
                device_ids=[self.local_rank],
                find_unused_parameters=True
            )
            if self.rank == 0:
                self.logger.info(f"Model wrapped with DDP on {self.world_size} GPUs")

        return model

    def _get_model(self):
        """Get underlying model (unwrap DDP if needed)."""
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _configure_parameters_phase1(self):
        """Phase 1: Only UnifiedGlobalMamba + OnepieceMetricHead trainable."""
        model = self._get_model()
        frozen_count = 0
        trainable_count = 0

        for name, param in model.named_parameters():
            if 'unified_global_mamba' in name or 'onepiece_metric_head' in name:
                param.requires_grad = True
                trainable_count += param.numel()
            else:
                param.requires_grad = False
                frozen_count += param.numel()

        if self.rank == 0:
            self.logger.info(f"=== Phase 1 Parameters ===")
            self.logger.info(f"  Frozen: {frozen_count:,} (ViT + DPT + FiLMGenerator + RelativeHead)")
            self.logger.info(f"  Trainable: {trainable_count:,} (UnifiedGlobalMamba + OnepieceMetricHead)")

    def _configure_parameters_phase2(self):
        """Phase 2: Additionally unfreeze FiLMGenerator + RelativeHead (final_head/output_conv)."""
        model = self._get_model()
        frozen_count = 0
        trainable_mamba = 0
        trainable_metric_head = 0
        trainable_film = 0
        trainable_rel_head = 0

        for name, param in model.named_parameters():
            if 'unified_global_mamba' in name:
                param.requires_grad = True
                trainable_mamba += param.numel()
            elif 'onepiece_metric_head' in name:
                param.requires_grad = True
                trainable_metric_head += param.numel()
            elif 'onepiece_film_generator' in name:
                param.requires_grad = True
                trainable_film += param.numel()
            elif 'output_conv' in name:
                # final_head uses output_conv1 and output_conv2
                param.requires_grad = True
                trainable_rel_head += param.numel()
            elif 'pretrained' in name or 'depth_head' in name:
                # ViT encoder and DPT always frozen
                param.requires_grad = False
                frozen_count += param.numel()
            else:
                param.requires_grad = False
                frozen_count += param.numel()

        if self.rank == 0:
            self.logger.info(f"=== Phase 2 Parameters ===")
            self.logger.info(f"  Frozen: {frozen_count:,} (ViT + DPT)")
            self.logger.info(f"  Trainable Mamba: {trainable_mamba:,}")
            self.logger.info(f"  Trainable MetricHead: {trainable_metric_head:,}")
            self.logger.info(f"  Trainable FiLMGenerator: {trainable_film:,}")
            self.logger.info(f"  Trainable RelativeHead: {trainable_rel_head:,}")
            total = trainable_mamba + trainable_metric_head + trainable_film + trainable_rel_head
            self.logger.info(f"  Total trainable: {total:,}")

    def _set_train_mode(self):
        """Set trainable parts to train mode, frozen parts to eval mode."""
        self.model.train()
        model = self._get_model()

        for name, module in model.named_modules():
            if name == '':
                continue
            # Keep onepiece modules in train mode
            if any(keyword in name for keyword in [
                'unified_global_mamba', 'onepiece_metric_head',
                'onepiece_film_generator'
            ]):
                continue
            # Phase 2: output_conv (RelativeHead) in train mode
            if self.current_phase >= 2 and 'output_conv' in name:
                continue
            module.eval()

    def _setup_data_loaders(self):
        """Setup training and validation data loaders."""
        train_datasets = self.config.dataset.get('train_datasets',
            ['mvs-synth', 'dynamicreplica', 'tartanair', 'pointodyssey', 'spring'])
        val_datasets = self.config.dataset.get('val_datasets', ['sintel', 'waymo_seg'])
        resolution = self.config.dataset.get('resolution', 'base')
        video_length = self.config.dataset.get('video_length', 8)

        if self.rank == 0:
            self.logger.info(f"Train datasets: {train_datasets}")
            self.logger.info(f"Val datasets: {val_datasets}")
            self.logger.info(f"Resolution: {resolution}, video_length: {video_length}")

        train_dataset = CombinedDataset(
            root_dir=self.config.dataset.data_root,
            enable_dataset_flags=train_datasets,
            resolution=resolution,
            split='train',
            video_length=video_length,
            color_aug=False
        )

        val_dataset = CombinedDataset(
            root_dir=self.config.dataset.data_root,
            enable_dataset_flags=val_datasets,
            resolution=resolution,
            split='val',
            video_length=video_length
        )

        # Samplers
        if self.world_size > 1:
            train_sampler = DistributedSampler(
                train_dataset, num_replicas=self.world_size,
                rank=self.rank, shuffle=True, drop_last=True
            )
            val_sampler = None
        else:
            train_sampler = None
            val_sampler = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=self.config.training.workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self._collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            sampler=val_sampler,
            shuffle=False,
            num_workers=self.config.training.workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self._collate_fn
        )

        if self.rank == 0:
            self.logger.info(f"Train dataset: {len(train_dataset)} samples")
            self.logger.info(f"Val dataset: {len(val_dataset)} samples")

        return train_loader, val_loader

    def _collate_fn(self, batch):
        """Custom collate to filter None values."""
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
        return torch.utils.data.dataloader.default_collate(batch)

    def _setup_optimizer(self):
        """Setup optimizer with parameter groups."""
        model = self._get_model()
        lr_config = self.config.training.get('lr', {})
        base_lr = lr_config.get('onepiece', 1e-4)

        if self.current_phase == 1:
            # Phase 1: Only Mamba + MetricHead
            mamba_metric_params = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    mamba_metric_params.append(param)

            param_groups = [
                {'params': mamba_metric_params, 'lr': base_lr, 'name': 'mamba_metric'}
            ]
        else:
            # Phase 2: Mamba+MetricHead at base_lr, FiLM+RelHead at 1/10
            phase2_lr = lr_config.get('dpt', base_lr / 10)

            mamba_metric_params = []
            film_rel_params = []

            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if 'unified_global_mamba' in name or 'onepiece_metric_head' in name:
                    mamba_metric_params.append(param)
                elif 'onepiece_film_generator' in name or 'output_conv' in name:
                    film_rel_params.append(param)

            param_groups = [
                {'params': mamba_metric_params, 'lr': base_lr, 'name': 'mamba_metric'},
                {'params': film_rel_params, 'lr': phase2_lr, 'name': 'film_relhead'},
            ]

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=[0.9, 0.95],
            weight_decay=self.config.training.get('weight_decay', 1e-6)
        )

        if self.rank == 0:
            for pg in param_groups:
                n_params = sum(p.numel() for p in pg['params'])
                self.logger.info(f"  Optimizer group '{pg['name']}': {n_params:,} params, LR={pg['lr']:.2e}")

        return optimizer

    def _setup_scheduler(self):
        """Setup LR scheduler with warmup and cosine decay."""
        total_steps = self.config.training.get('iterations', self.config.training.total_iters)
        warmup_steps = self.config.training.lr.get('warmup_steps', 500)
        decay_start = int(total_steps * 0.3)

        if self.current_phase == 1:
            def lr_lambda(step):
                if step < warmup_steps:
                    return 0.1 + 0.9 * (step / warmup_steps)
                elif step < decay_start:
                    return 1.0
                else:
                    progress = (step - decay_start) / (total_steps - decay_start)
                    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        else:
            transition_step = self.auto_transition_step

            def lr_lambda_onepiece(step):
                if step < warmup_steps:
                    return 0.1 + 0.9 * (step / warmup_steps)
                elif step < decay_start:
                    return 1.0
                else:
                    progress = (step - decay_start) / (total_steps - decay_start)
                    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

            def lr_lambda_phase2_new(step):
                steps_since_transition = step - transition_step
                if steps_since_transition < 0:
                    return 0.0
                elif steps_since_transition < self.phase2_warmup_steps:
                    return steps_since_transition / self.phase2_warmup_steps
                elif step < decay_start:
                    return 1.0
                else:
                    progress = (step - decay_start) / (total_steps - decay_start)
                    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

            lambdas = [lr_lambda_onepiece]
            for _ in range(len(self.optimizer.param_groups) - 1):
                lambdas.append(lr_lambda_phase2_new)

            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambdas)

    def _transition_to_phase2(self):
        """Auto-transition from Phase 1 to Phase 2."""
        if self.rank == 0:
            self.logger.info("=" * 60)
            self.logger.info("=== ONEPIECE PHASE TRANSITION: Phase 1 → Phase 2 ===")
            self.logger.info("=" * 60)

        self.current_phase = 2

        # Reconfigure parameters
        self._configure_parameters_phase2()

        # Rebuild optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        # Keep same val_vis_config as Phase 1 (visualize all sequences)
        # Phase 1 config is preserved; no need to reduce to seq0 only

        # Set proper train mode
        self._set_train_mode()

        if self.rank == 0:
            self.logger.info(f"Phase transition complete at step {self.global_step}")
            self.logger.info("=" * 60)

    def train(self):
        """Main training loop."""
        if self.rank == 0:
            self.logger.info("Starting Onepiece training...")

        train_iterator = iter(self.train_loader)
        total_iters = self.config.training.get('iterations', self.config.training.total_iters)

        pbar = tqdm(range(total_iters), desc="Onepiece Training", disable=(self.rank != 0))

        for step in pbar:
            self.global_step = step

            # Phase transition check
            if self.current_phase == 1 and step == self.auto_transition_step:
                self._transition_to_phase2()

            # Get batch
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_loader)
                batch = next(train_iterator)

            if batch is None:
                continue

            # Training step
            loss_dict = self.train_step(batch, step)
            if loss_dict is None:
                continue

            # Update scheduler
            self.scheduler.step()

            # Update progress bar
            lr_op = self.optimizer.param_groups[0]['lr']
            postfix = {
                'loss': f'{loss_dict["loss"]:.4f}',
                'lr': f'{lr_op:.2e}',
                'phase': self.current_phase
            }
            if 'log_l1_loss' in loss_dict:
                postfix['l1'] = f'{loss_dict["log_l1_loss"]:.4f}'
            if 'tgm_loss' in loss_dict:
                postfix['tgm'] = f'{loss_dict["tgm_loss"]:.4f}'
            if 'ofc_loss' in loss_dict and loss_dict['ofc_loss'] > 0:
                postfix['ofc'] = f'{loss_dict["ofc_loss"]:.4f}'
            pbar.set_postfix(postfix)

            # WandB logging
            if self.config.training.get('wandb', False) and self.rank == 0:
                wandb_dict = {
                    **loss_dict,
                    'lr_onepiece': lr_op,
                    'phase': self.current_phase
                }
                if len(self.optimizer.param_groups) > 1:
                    wandb_dict['lr_dpt'] = self.optimizer.param_groups[1]['lr']
                wandb.log(wandb_dict, step=step)

            # Console logging
            if self.rank == 0 and (step % 100 == 0 or step < 10):
                log_parts = [f"Step {step}"]
                log_parts.append(f"loss={loss_dict['loss']:.4f}")
                if 'log_l1_loss' in loss_dict:
                    log_parts.append(f"L1={loss_dict['log_l1_loss']:.4f}")
                if 'tgm_loss' in loss_dict:
                    log_parts.append(f"TGM={loss_dict['tgm_loss']:.4f}")
                if 'ofc_loss' in loss_dict and loss_dict['ofc_loss'] > 0:
                    log_parts.append(f"OFC={loss_dict['ofc_loss']:.4f}")
                if 'mean_scale' in loss_dict:
                    log_parts.append(f"scale={loss_dict['mean_scale']:.4f}")
                if 'mean_shift' in loss_dict:
                    log_parts.append(f"shift={loss_dict['mean_shift']:.6f}")
                log_parts.append(f"phase={self.current_phase}")
                self.logger.info(" | ".join(log_parts))

            # Training visualization (steps 0, 10, 50, 100, then every 250)
            vis_steps = [0, 10, 50, 100]
            if (step in vis_steps or step % 250 == 0) and self.train_visualizer and self.rank == 0:
                self._save_training_visualization(batch, loss_dict, step)

            # Validation
            if step % self.config.training.get('val_freq', 1000) == 0 and self.rank == 0:
                val_metrics = self.validate()
                val_loss = val_metrics['loss']
                self.current_val_loss = val_loss
                self.logger.info(f"Validation at step {step}: loss={val_loss:.4f}")

                # Per-dataset loss logging
                if 'dataset_losses' in val_metrics:
                    self.dataset_losses = val_metrics['dataset_losses']
                    self.num_sequences = val_metrics.get('num_sequences', {})
                    for ds_name, ds_loss in val_metrics['dataset_losses'].items():
                        self.logger.info(f"  {ds_name}: loss={ds_loss:.4f}")
                        if self.config.training.get('wandb', False):
                            wandb.log({f'val/{ds_name}_loss': ds_loss}, step=step)

                if self.config.training.get('wandb', False):
                    wandb.log({'val/loss': val_loss}, step=step)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_step = step
                    self.save_checkpoint('best.pth')
                    self.logger.info(f"New best model at step {step}: val_loss={val_loss:.4f}")

                self._set_train_mode()

            # Save checkpoint periodically
            if step % self.config.training.get('save_freq', 5000) == 0 and step > 0 and self.rank == 0:
                self.save_checkpoint(f'checkpoint_step{step}.pth')

        # Final save
        if self.rank == 0:
            self.save_checkpoint('last.pth')
            self.logger.info("Onepiece training completed!")

    def _save_training_visualization(self, batch, loss_dict, step):
        """Save training visualization at designated steps."""
        try:
            self.model.eval()
            model = self._get_model()

            images, gt_depth, focal_lengths_canonical, focal_lengths_actual, \
                actual_valid_masks, fx_ratio, resize_ratio, dataset_idx = batch

            images = images.to(self.device)
            gt_depth = gt_depth.to(self.device)
            fx_ratio = fx_ratio.to(self.device)
            resize_ratio = resize_ratio.to(self.device)

            if gt_depth.ndim == 3:
                gt_depth = gt_depth.unsqueeze(1)
            elif gt_depth.ndim == 4 and gt_depth.shape[1] != 1:
                gt_depth = gt_depth.unsqueeze(2)

            B, T, C, H, W = images.shape

            with torch.no_grad():
                outputs = model.forward_with_onepiece(
                    (images,), phase=self.current_phase, no_shift=self.no_shift
                )

            metric_depth = outputs['metric_depth']  # [B, T, H, W]
            scale = outputs['scale']  # [B, T]
            shift = outputs['shift']  # [B, T]

            # Convert GT: inverse depth (1/m) → metric depth (m)
            gt_depth_inverse_100 = gt_depth * 100.0
            gt_depth_metric = 100.0 / (gt_depth_inverse_100.squeeze(2) + 1e-8)

            # Canonical valid masks (70m threshold)
            MIN_INVERSE_DEPTH = 100.0 / 70.0
            canonical_gt_valid = (gt_depth_inverse_100.squeeze(2) > MIN_INVERSE_DEPTH)
            MAX_DEPTH_OUTLIER = 200.0
            MIN_INVERSE_OUTLIER = 100.0 / MAX_DEPTH_OUTLIER
            pred_inverse = 100.0 / (metric_depth + 1e-8)
            canonical_pred_valid = (pred_inverse > MIN_INVERSE_OUTLIER)

            # First frame for visualization
            sample_batch = (
                images[:1, :1].float().cpu(),
                gt_depth_metric[:1, :1].float().cpu(),
                dataset_idx,
                fx_ratio[:1, :1].float().cpu(),
                resize_ratio[:1, :1].float().cpu()
            )

            model_outputs_cpu = {
                'pred_depth': metric_depth[:1, :1].float().cpu(),
                'canonical_gt_valid': canonical_gt_valid[:1, :1].cpu(),
                'canonical_pred_valid': canonical_pred_valid[:1, :1].cpu(),
                'scale': scale[:1, :1].cpu(),
                'shift': shift[:1, :1].cpu()
            }

            self.train_visualizer.create_validation_summary(
                sample_batch, model_outputs_cpu, step,
                prefix="training", loss_dict=loss_dict, config=self.config
            )

            self._set_train_mode()

        except Exception as e:
            import traceback
            self.logger.error(f"Failed to save training visualization: {e}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            self._set_train_mode()

    def train_step(self, batch, step):
        """Single training step."""
        if batch is None:
            return None

        # Unpack Gear5 8-element batch
        images, gt_depth, focal_lengths_canonical, focal_lengths_actual, \
            actual_valid_masks, fx_ratio, resize_ratio, dataset_idx = batch

        # Move to device
        images = images.to(self.device)
        gt_depth = gt_depth.to(self.device)
        actual_valid_masks = actual_valid_masks.to(self.device)

        # Ensure gt_depth shape: [B, T, 1, H, W]
        if gt_depth.ndim == 3:
            gt_depth = gt_depth.unsqueeze(1)
        elif gt_depth.ndim == 4 and gt_depth.shape[1] != 1:
            gt_depth = gt_depth.unsqueeze(2)

        B, T = images.shape[:2]
        H, W = images.shape[3], images.shape[4]

        model = self._get_model()

        # Forward pass with BFloat16 autocast
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model.forward_with_onepiece(
                (images,), phase=self.current_phase, no_shift=self.no_shift
            )

            metric_depth = outputs['metric_depth']                    # [B, T, H, W]
            modulated_features = outputs['modulated_features']        # [B, T, 256, h, w]
            scale = outputs['scale']                                  # [B, T]
            shift = outputs['shift']                                  # [B, T]

        # Compute loss (outside autocast)
        with torch.amp.autocast('cuda', enabled=False):
            gt_depth_meters = 1.0 / (gt_depth.squeeze(2).clamp(min=1e-8))

            if metric_depth.shape[-2:] != gt_depth_meters.shape[-2:]:
                BT = B * T
                metric_depth = F.interpolate(
                    metric_depth.view(BT, 1, metric_depth.shape[-2], metric_depth.shape[-1]),
                    size=gt_depth_meters.shape[-2:],
                    mode='bilinear', align_corners=True
                ).squeeze(1).view(B, T, gt_depth_meters.shape[-2], gt_depth_meters.shape[-1])

            gt_valid = (gt_depth.squeeze(2) > 0)
            pred_valid = (metric_depth > 0) & (metric_depth < 1000.0)
            if actual_valid_masks.ndim == 3:
                actual_valid_masks = actual_valid_masks.unsqueeze(1)
            valid_mask = gt_valid & pred_valid & actual_valid_masks

            # Convert to inverse depth space for LogL1/TGM losses
            pred_inverse = 1.0 / metric_depth.float().clamp(min=1e-8)
            gt_inverse = gt_depth.squeeze(2).float()  # already 1/m

            total_loss, loss_components = self.loss_fn(
                metric_depth=pred_inverse,
                gt_depth=gt_inverse,
                valid_mask=valid_mask.float(),
                modulated_features=modulated_features.float() if self.current_phase == 2 else None,
                images=images.float() if self.current_phase == 2 else None,
                flow_estimator=self.flow_estimator if self.current_phase == 2 else None,
                phase=self.current_phase,
                return_components=True
            )

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Build result dict
        result = {
            'loss': total_loss.item(),
            'mean_scale': scale.mean().item(),
            'mean_shift': shift.mean().item(),
        }
        result.update(loss_components)

        return result

    @torch.no_grad()
    def validate(self):
        """Full validation loop with per-dataset tracking."""
        torch.cuda.empty_cache()
        self.model.eval()
        model = self._get_model()

        total_loss = 0.0
        total_depth_loss = 0.0
        total_tgm_loss = 0.0
        num_batches = 0

        # Per-dataset tracking
        dataset_losses = {}
        dataset_depth_losses = {}
        dataset_tgm_losses = {}

        # Reset visualization tracking
        for ds_config in self.val_vis_config.values():
            ds_config['saved'] = []

        # Track dataset-specific sequence counters (match Gear5)
        dataset_sequence_counters = {'sintel': 0, 'waymo_seg': 0}

        # Phase-specific batch limits (match Gear5)
        if self.current_phase >= 2:
            max_val_batches = 16
            dataset_max_sequences = {'sintel': 8, 'waymo_seg': 8}
        else:
            max_val_batches = None  # unlimited
            dataset_max_sequences = {}

        total_processed = 0

        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation", disable=(self.rank != 0))):
            if batch is None:
                continue

            if max_val_batches is not None and total_processed >= max_val_batches:
                break

            images, gt_depth, focal_lengths_canonical, focal_lengths_actual, \
                actual_valid_masks, fx_ratio, resize_ratio, dataset_idx = batch

            # Determine current dataset name
            if isinstance(dataset_idx, (list, tuple)):
                current_dataset = str(dataset_idx[0])
            elif torch.is_tensor(dataset_idx):
                current_dataset = str(dataset_idx[0].item() if dataset_idx.dim() > 0 else dataset_idx.item())
            else:
                current_dataset = str(dataset_idx)

            # Check dataset-specific limit (Phase 2 only, match Gear5)
            if dataset_max_sequences and current_dataset in dataset_max_sequences:
                if dataset_sequence_counters.get(current_dataset, 0) >= dataset_max_sequences[current_dataset]:
                    if self.rank == 0 and dataset_sequence_counters.get(current_dataset, 0) == dataset_max_sequences[current_dataset]:
                        self.logger.info(f"  [{current_dataset}] Reached max {dataset_max_sequences[current_dataset]} sequences, skipping further...")
                    continue

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                images = images.to(self.device)
                gt_depth = gt_depth.to(self.device)
                actual_valid_masks = actual_valid_masks.to(self.device)
                fx_ratio_dev = fx_ratio.to(self.device)
                resize_ratio_dev = resize_ratio.to(self.device)

                if gt_depth.ndim == 3:
                    gt_depth = gt_depth.unsqueeze(1)
                elif gt_depth.ndim == 4 and gt_depth.shape[1] != 1:
                    gt_depth = gt_depth.unsqueeze(2)

                B, T = images.shape[:2]
                H, W = images.shape[3], images.shape[4]

                outputs = model.forward_with_onepiece(
                    (images,), phase=self.current_phase, no_shift=self.no_shift
                )

                metric_depth = outputs['metric_depth']
                scale = outputs['scale']
                shift = outputs['shift']

            # Compute validation loss
            gt_depth_meters = 1.0 / (gt_depth.squeeze(2).float().clamp(min=1e-8))

            if metric_depth.shape[-2:] != gt_depth_meters.shape[-2:]:
                metric_depth = F.interpolate(
                    metric_depth.view(B * T, 1, metric_depth.shape[-2], metric_depth.shape[-1]),
                    size=gt_depth_meters.shape[-2:],
                    mode='bilinear', align_corners=True
                ).squeeze(1).view(B, T, gt_depth_meters.shape[-2], gt_depth_meters.shape[-1])

            # Validation uses 70m threshold to match test evaluation
            VAL_MAX_DEPTH = 70.0
            gt_valid = (gt_depth.squeeze(2) > 0) & (gt_depth_meters < VAL_MAX_DEPTH)
            pred_valid = (metric_depth > 0) & (metric_depth < VAL_MAX_DEPTH)
            if actual_valid_masks.ndim == 3:
                actual_valid_masks = actual_valid_masks.unsqueeze(1)
            valid_mask = gt_valid & pred_valid & actual_valid_masks

            if valid_mask.sum() > 0:
                # Log L1 loss
                pred_valid_vals = metric_depth.float()[valid_mask]
                gt_valid_vals = gt_depth_meters.float()[valid_mask]
                epsilon = 1e-8
                avg_depth_loss = F.l1_loss(
                    torch.log(pred_valid_vals.clamp(min=epsilon)),
                    torch.log(gt_valid_vals.clamp(min=epsilon))
                )

                # TGM loss (if temporal frames available)
                avg_tgm_loss = torch.tensor(0.0)
                if T > 1:
                    pred_inverse = 1.0 / metric_depth.float().clamp(min=1e-8)
                    gt_inverse = gt_depth.squeeze(2).float()
                    pred_diff = pred_inverse[:, 1:] - pred_inverse[:, :-1]
                    gt_diff = gt_inverse[:, 1:] - gt_inverse[:, :-1]
                    temporal_valid = valid_mask[:, 1:] & valid_mask[:, :-1]
                    temporal_valid_w = temporal_valid.float()
                    if temporal_valid_w.sum() > 0:
                        tgm_error = (pred_diff - gt_diff).abs()
                        avg_tgm_loss = (tgm_error * temporal_valid_w).sum() / temporal_valid_w.sum().clamp(min=1)

                avg_loss = avg_depth_loss + avg_tgm_loss
                total_loss += avg_loss.item()
                total_depth_loss += avg_depth_loss.item()
                total_tgm_loss += avg_tgm_loss.item()
                num_batches += 1

                # Per-dataset tracking
                if current_dataset not in dataset_losses:
                    dataset_losses[current_dataset] = []
                    dataset_depth_losses[current_dataset] = []
                    dataset_tgm_losses[current_dataset] = []
                dataset_losses[current_dataset].append(avg_loss.item())
                dataset_depth_losses[current_dataset].append(avg_depth_loss.item())
                dataset_tgm_losses[current_dataset].append(avg_tgm_loss.item() if torch.is_tensor(avg_tgm_loss) else avg_tgm_loss)

            # Validation visualization
            if self.val_visualizer and current_dataset in self.val_vis_config:
                vis_config = self.val_vis_config[current_dataset]
                seq_idx_in_dataset = dataset_sequence_counters.get(current_dataset, 0)

                if seq_idx_in_dataset in vis_config['sequences'] and seq_idx_in_dataset not in vis_config['saved']:
                    try:
                        # Compute metric depth for vis WITHOUT clamp (preserves -1.0 for non-LiDAR, like Gear5)
                        gt_depth_metric_vis = (1.0 / (gt_depth.squeeze(2)[:1, :1].float() + 1e-8)).cpu()
                        pred_metric_vis = metric_depth[:1, :1].float().cpu()

                        MIN_INVERSE_DEPTH = 100.0 / 70.0
                        gt_inv_100 = gt_depth.squeeze(2)[:1, :1] * 100.0
                        canonical_gt_valid = (gt_inv_100 > MIN_INVERSE_DEPTH)
                        pred_inv = 100.0 / (metric_depth[:1, :1] + 1e-8)
                        canonical_pred_valid = (pred_inv > 100.0 / 200.0)

                        sample_batch = (
                            images[:1, :1].float().cpu(),
                            gt_depth_metric_vis,
                            dataset_idx,
                            fx_ratio_dev[:1, :1].float().cpu(),
                            resize_ratio_dev[:1, :1].float().cpu()
                        )

                        model_outputs_cpu = {
                            'pred_depth': pred_metric_vis,
                            'canonical_gt_valid': canonical_gt_valid.cpu(),
                            'canonical_pred_valid': canonical_pred_valid.cpu(),
                            'scale': scale[:1, :1].cpu(),
                            'shift': shift[:1, :1].cpu()
                        }

                        val_loss_dict = {
                            'val_loss': avg_loss.item() if valid_mask.sum() > 0 else 0.0,
                            'depth_loss': avg_depth_loss.item() if valid_mask.sum() > 0 else 0.0,
                            'tgm_loss': avg_tgm_loss.item() if torch.is_tensor(avg_tgm_loss) else 0.0,
                        }

                        self.val_visualizer.create_validation_summary(
                            sample_batch, model_outputs_cpu, self.global_step,
                            prefix=f"val_{current_dataset}_seq{seq_idx_in_dataset}",
                            loss_dict=val_loss_dict,
                            dataset_name=current_dataset,
                            config=self.config
                        )
                        vis_config['saved'].append(seq_idx_in_dataset)
                    except Exception as e:
                        self.logger.warning(f"Validation vis failed for {current_dataset} seq {seq_idx_in_dataset}: {e}")

                dataset_sequence_counters[current_dataset] = seq_idx_in_dataset + 1
            else:
                # Update counter even when not in val_vis_config (for dataset_max_sequences tracking)
                if current_dataset not in dataset_sequence_counters:
                    dataset_sequence_counters[current_dataset] = 0
                dataset_sequence_counters[current_dataset] += 1

            total_processed += 1

            # Memory cleanup
            del images, gt_depth, metric_depth
            torch.cuda.empty_cache()

        # Compute averages
        avg_loss = total_loss / max(num_batches, 1)
        avg_depth_loss = total_depth_loss / max(num_batches, 1)
        avg_tgm_loss = total_tgm_loss / max(num_batches, 1)

        # Per-dataset summary
        per_dataset_avg = {}
        per_dataset_num = {}
        if dataset_losses:
            self.logger.info("  === Per-Dataset Validation ===")
            for ds_name in sorted(dataset_losses.keys()):
                ds_avg = np.mean(dataset_losses[ds_name])
                ds_depth = np.mean(dataset_depth_losses[ds_name])
                ds_tgm = np.mean(dataset_tgm_losses[ds_name])
                ds_count = len(dataset_losses[ds_name])
                per_dataset_avg[ds_name] = ds_avg
                per_dataset_num[ds_name] = ds_count
                self.logger.info(
                    f"    {ds_name}: loss={ds_avg:.4f} (L1={ds_depth:.4f}, TGM={ds_tgm:.4f}) [{ds_count} seqs]"
                )

        return {
            'loss': avg_loss,
            'depth_loss': avg_depth_loss,
            'tgm_loss': avg_tgm_loss,
            'dataset_losses': per_dataset_avg,
            'num_sequences': per_dataset_num
        }

    def save_checkpoint(self, filename):
        """Save checkpoint with full tracking info (rank 0 only)."""
        if self.rank != 0:
            return

        checkpoint_path = self.results_dir / filename
        model = self._get_model()

        checkpoint = {
            'global_step': self.global_step,
            'model': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_step': self.best_step,
            'current_val_loss': self.current_val_loss,
            'dataset_losses': self.dataset_losses,
            'num_sequences': self.num_sequences,
            'config': OmegaConf.to_container(self.config, resolve=True),
            'current_phase': self.current_phase,
        }

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")


@hydra.main(version_base=None, config_path="configs/onepiece", config_name="config")
def main(config: DictConfig):
    """Main entry point."""
    rank, world_size, local_rank = init_distributed()

    trainer = OnepieceTrainer(config, rank, world_size, local_rank)
    trainer.train()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
