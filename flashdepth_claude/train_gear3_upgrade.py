"""
Gear3 Upgrade Training Script: Enhanced FG/BG Separation Methods

Three-phase training:
    Phase 1: Train on 5 datasets (518×518)
    Phase 2: Train on mvs-synth, spring (2K resolution)
    Phase 3: Fine-tune on nuScenes (2K resolution)

Key features:
    - 3 FG/BG separation options:
        1. CLS-based light segmentation (~1-2ms overhead)
        2. Differentiable K-means clustering (~5-10ms overhead)
        3. Multi-layer attention fusion (~3ms overhead)
    - Feature-level modulation using FiLM
    - Canonical space normalization (focal_length=1000)
    - Loss on inverse depth: loss(100/pred, 100/gt)
    - Enhanced visualization with FG/BG masks
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from tqdm import tqdm
import numpy as np
from pathlib import Path
from einops import rearrange
import math
import time
from datetime import timedelta

from flashdepth.model import FlashDepth
from utils.gear3_upgrade_visualization import Gear3UpgradeVisualizer
from flashdepth.gear3_upgrade_modules import Gear3UpgradeMetricHead
from dataloaders.combined_dataset import CombinedDataset
from utils.helpers import *
from utils.metric_depth_metrics import MetricDepthMetrics
from utils.gear_losses import LogL1Loss


def init_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # torchrun/distributed launch
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # Single GPU fallback
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





class Gear3UpgradeTrainer:
    """
    Trainer for Gear3 metric depth learning.

    Frozen: DINOv2, DPT
    Fine-tuned: Mamba (LR: 1e-5)
    Trained: Gear3 modules (LR: 1e-4)
    """
    def __init__(self, config, rank, world_size, local_rank):
        self.config = config
        self.phase = config.get('phase', 1)  # Training phase (1 or 2)
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank

        # Setup device
        self.device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)

        if rank == 0:
            logging.info(f"Training Phase {self.phase} on {world_size} GPU(s)")

        # Setup results directory (only rank 0)
        phase_suffix = f"_phase{self.phase}"
        self.results_dir = Path(config.get('results_dir', f'./train_results/gear3{phase_suffix}'))
        if rank == 0:
            self.results_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging (only rank 0)
        # Force immediate flush after each log
        class FlushFileHandler(logging.FileHandler):
            def emit(self, record):
                super().emit(record)
                self.flush()  # Flush immediately

        if rank == 0:
            # Get root logger and clear any existing handlers
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.INFO)
            root_logger.handlers.clear()  # Clear existing handlers

            # Create file handler with immediate flushing
            file_handler = FlushFileHandler(self.results_dir / 'training.log', mode='a')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

            # Create console handler
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)
            stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

            # Add handlers to root logger
            root_logger.addHandler(file_handler)
            root_logger.addHandler(stream_handler)
        else:
            # Other ranks: only show errors
            logging.basicConfig(level=logging.ERROR)

        self.logger = logging.getLogger(__name__)

        if rank == 0:
            self.logger.info(f"Results directory: {self.results_dir}")
            self.logger.info(f"Training phase: {self.phase}")

        # Initialize model
        self.model = self._setup_model()

        # Set proper training mode (trainable parts in train mode, frozen parts in eval mode)
        self._set_train_mode()

        # Setup data loaders
        self.train_loader, self.val_loader = self._setup_data_loaders()

        # Setup optimizer and loss
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.loss_fn = LogL1Loss()

        # FPS measurement removed (only in test_gear3.py with batch=1 for fair comparison)

        # Setup wandb
        if config.training.get('wandb', False):
            wandb.init(
                project="flashdepth-gear3",
                name=f"gear3_phase{self.phase}_{config.training.get('wandb_name', 'experiment')}",
                config=dict(config)
            )

        # Setup visualizer with separate folders
        self.train_visualizer = Gear3UpgradeVisualizer(save_dir=self.results_dir / "visualizations" / "train")
        self.val_visualizer = Gear3UpgradeVisualizer(save_dir=self.results_dir / "visualizations" / "valid")

        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_step = 0  # Track which step achieved best validation loss
        self.current_val_loss = None  # Track current validation loss for checkpoint
        self.dataset_losses = None  # Track per-dataset validation losses
        self.num_sequences = None  # Track number of sequences per dataset

        # Validation visualization config: track which sequences to visualize
        # Phase 2/3: Save 1 sample per dataset at every validation step (for step-by-step comparison)
        # Sintel is smaller (1022×434), Waymo_seg is 2K (1918×1274)
        if self.phase >= 2:
            self.val_vis_config = {
                'sintel': {'sequences': [0], 'saved': []},  # seq 0 only
                'waymo_seg': {'sequences': [0], 'saved': []}    # seq 0 only
            }
        else:
            # Phase 1: Can afford more samples (518x518, no batch limit)
            self.val_vis_config = {
                'sintel': {'sequences': [0, 4, 7], 'saved': []},  # seq 0, 4, 7 (3 samples)
                'waymo_seg': {'sequences': [0, 1, 2, 3, 4, 5, 6, 7], 'saved': []}  # all 8 sequences
            }

    def _setup_model(self):
        """Initialize FlashDepth with Gear3 metric head"""
        # Create base FlashDepth model
        model_config = dict(self.config.model)
        model_config['batch_size'] = self.config.training.batch_size
        model_config['use_metric_head'] = False  # Don't use GSP head

        model = FlashDepth(**model_config)

        # Load pre-trained checkpoint
        # Phase 1: Load from config (DINOv2 + DPT only)
        # Phase 2: Load Gear-S checkpoint (for Gear modules) + Hybrid weights (for ViT-DPT)
        if self.phase == 1:
            checkpoint_path = self.config.get('load')
        else:
            # Phase 2: Load Gear-S Phase 1 checkpoint (always S for hybrid)
            checkpoint_path = self.config.get('gear_checkpoint')
            if not checkpoint_path:
                raise ValueError(
                    "Phase 2 (hybrid) requires 'gear_checkpoint' in config! "
                    "Set gear_checkpoint to your Gear-S Phase 1 checkpoint path."
                )
            if self.rank == 0:
                self.logger.info(f"Phase {self.phase}: Loading Gear-S checkpoint for Gear modules from {checkpoint_path}")

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

                if self.phase == 1:
                    # Phase 1: Load only DINOv2 and DPT refinement layers
                    # Exclude: Mamba (modulated input), output_conv1/2 (modulated features), gear3_head
                    loaded_dict = {}
                    excluded_keys = []
                    for k, v in state_dict.items():
                        # Exclude modules that receive modulated features
                        if any(x in k for x in ['mamba', 'hybrid_fusion', 'teacher_model',
                                                'output_conv1', 'output_conv2', 'gear3_upgrade_head']):
                            excluded_keys.append(k)
                        else:
                            loaded_dict[k] = v

                    # Load state dict (strict=False to allow missing modules)
                    model.load_state_dict(loaded_dict, strict=False)
                    self.logger.info(f"Phase 1: Loaded {len(loaded_dict)} parameters from checkpoint")
                    self.logger.info(f"  - DINOv2 encoder: ✓")
                    self.logger.info(f"  - DPT projects/resize/refinenet: ✓")
                    self.logger.info(f"Excluded {len(excluded_keys)} parameters (will train from scratch):")
                    self.logger.info(f"  - Mamba, output_conv1/2, gear3_head")
                else:
                    # Phase 2, 3: Load ALL parameters including gear3_head
                    # But need to add gear3_head first before loading
                    pass  # Will load after gear3_head is created
            else:
                self.logger.warning(f"Checkpoint {checkpoint_path} not found")

        # Add Gear3 Upgrade metric head
        embed_dim = 1024 if model.encoder == 'vitl' else 384
        dpt_dim = 256 if model.encoder == 'vitl' else 64
        num_heads = 16 if model.encoder == 'vitl' else 6

        # Get separation method from config (default: 'cls_seg')
        separation_method = self.config.get('separation_method', 'cls_seg')
        self.separation_method = separation_method
        self.logger.info(f"Using FG/BG separation method: {separation_method}")

        model.gear3_upgrade_head = Gear3UpgradeMetricHead(
            embed_dim=embed_dim,
            dpt_dim=dpt_dim,
            separation_method=separation_method,
            num_heads=num_heads
        )

        # Phase 2, 3: Load ALL parameters after gear3_head is created
        if self.phase > 1 and checkpoint_path and checkpoint_path != 'true' and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            # Load all parameters including gear3_head
            model.load_state_dict(state_dict, strict=False)
            self.logger.info(f"Phase {self.phase}: Loaded ALL parameters from Phase 1 checkpoint")
            self.logger.info(f"  - DINOv2, DPT, Mamba, output_conv, Gear3: ✓")

            # Phase 2 ONLY: Overwrite ViT-DPT with FlashDepth-hybrid weights
            if self.phase == 2:
                # Use 'load' config for hybrid weights (ViT-DPT overwrite)
                hybrid_path = self.config.get('load', 'configs/flashdepth/iter_43002.pth')
                if os.path.exists(hybrid_path):
                    self.logger.info(f"Phase 2: Overwriting ViT-DPT with Hybrid weights from {hybrid_path}")
                    hybrid_checkpoint = torch.load(hybrid_path, map_location='cpu')

                    # Extract state dict
                    if isinstance(hybrid_checkpoint, dict) and 'model' in hybrid_checkpoint:
                        hybrid_state_dict = hybrid_checkpoint['model']
                    elif isinstance(hybrid_checkpoint, dict) and 'state_dict' in hybrid_checkpoint:
                        hybrid_state_dict = hybrid_checkpoint['state_dict']
                    else:
                        hybrid_state_dict = hybrid_checkpoint

                    # Remove module. prefix if present
                    hybrid_state_dict = {k.replace('module.', ''): v for k, v in hybrid_state_dict.items()}

                    # Load ONLY DINOv2 and DPT parameters (overwrite Phase 1 weights)
                    loaded_hybrid = {}
                    for k, v in hybrid_state_dict.items():
                        # Include only encoder and DPT refinement (same as Phase 1 loading)
                        if not any(x in k for x in ['mamba', 'hybrid_fusion', 'teacher_model',
                                                     'output_conv1', 'output_conv2', 'gear3_upgrade_head']):
                            loaded_hybrid[k] = v

                    # Overwrite ViT-DPT parameters
                    model.load_state_dict(loaded_hybrid, strict=False)
                    self.logger.info(f"Phase 2: Overwritten {len(loaded_hybrid)} ViT-DPT parameters with Hybrid weights")
                    self.logger.info(f"  - Kept from Phase 1: Gear3, Mamba, output_conv (continue training)")
                else:
                    self.logger.warning(f"Hybrid checkpoint {hybrid_path} not found! Using Phase 1 ViT-DPT weights.")

        # Enable attention weights storage
        # - 'multi_layer': Enable for multiple blocks (encoder-specific)
        #   - ViT-L (24 blocks): [3, 10, 16, 22]
        #   - ViT-S (12 blocks): [2, 5, 8, 11]
        # - Others: Enable ONLY for last block (saves memory)
        if separation_method == 'multi_layer':
            # Use encoder-specific block indices for proportional coverage
            multi_layer_blocks = {
                'vitl': [3, 10, 16, 22],
                'vits': [2, 5, 8, 11]
            }
            target_blocks = multi_layer_blocks[model.encoder]
        else:
            target_blocks = [len(model.pretrained.blocks) - 1]

        for i, block in enumerate(model.pretrained.blocks):
            if i in target_blocks:
                block.attn.store_attn_weights = True
                self.logger.info(f"Enabled attention weights storage for block {i}")
            else:
                block.attn.store_attn_weights = False

        if separation_method == 'multi_layer':
            self.logger.info(f"Multi-layer attention fusion: storing attention from blocks {target_blocks}")

        # Store target blocks for use in training/validation steps
        self.target_blocks = target_blocks

        model = model.to(self.device)

        # Freeze and configure parameters
        self._configure_parameters(model)

        # Wrap with DDP if multi-GPU
        if self.world_size > 1:
            model = DDP(
                model,
                device_ids=[self.local_rank],
                find_unused_parameters=True  # Important for frozen parameters
            )
            if self.rank == 0:
                self.logger.info(f"Model wrapped with DDP on {self.world_size} GPUs")

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
            if 'gear3_upgrade_head' in name:
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

    def _set_train_mode(self):
        """
        Set model to training mode, but keep frozen parts in eval mode.
        This prevents BatchNorm/Dropout in frozen parts from updating.
        """
        self.model.train()

        # Keep frozen parts in eval mode (like train_metric_head)
        for name, module in self.model.named_modules():
            # Skip empty name (root module)
            if name == '':
                continue

            # Keep trainable parts in train mode
            if any(keyword in name for keyword in ['gear3_upgrade_head', 'mamba', 'output_conv']):
                continue

            # Set frozen parts to eval mode
            module.eval()

    def _setup_data_loaders(self):
        """Setup phase-specific data loaders"""
        if self.phase == 1:
            # Phase 1: 5 datasets, 518×518
            train_datasets = self.config.dataset.get('train_datasets',
                ['mvs-synth', 'dynamicreplica', 'tartanair', 'pointodyssey', 'spring'])
            val_datasets = self.config.dataset.get('val_datasets', ['sintel', 'waymo'])
            resolution = 'base'  # 518×518
        elif self.phase == 1.5:
            # Phase 1.5: nuScenes fine-tuning, 518×518 (optional)
            train_datasets = ['nuscenes']
            val_datasets = ['nuscenes']
            resolution = 'base'  # 518×518 (same as Phase 1)
        elif self.phase == 2:
            # Phase 2: mvs-synth, spring only, 2K resolution (Hybrid)
            train_datasets = ['mvs-synth', 'spring']
            val_datasets = ['sintel', 'waymo']
            resolution = '2k'
        else:
            raise ValueError(f"Invalid phase: {self.phase}. Must be 1, 1.5, or 2.")

        if self.rank == 0:
            self.logger.info(f"Phase {self.phase} - Train datasets: {train_datasets}")
            self.logger.info(f"Phase {self.phase} - Val datasets: {val_datasets}")
            self.logger.info(f"Phase {self.phase} - Resolution: {resolution}")

        # Training dataset
        train_dataset = CombinedDataset(
            root_dir=self.config.dataset.data_root,
            enable_dataset_flags=train_datasets,
            resolution=resolution,  # Use phase-specific resolution
            split='train',
            video_length=self.config.dataset.video_length,
            color_aug=False  # No augmentation for metric training
        )

        # Validation dataset: use shorter sequence for Phase 2/3 to save memory
        # Phase 2/3 use 2K resolution which requires much more memory
        val_video_length = 1 if self.phase >= 2 else self.config.dataset.video_length
        val_dataset = CombinedDataset(
            root_dir=self.config.dataset.data_root,
            enable_dataset_flags=val_datasets,
            resolution=resolution,  # Use phase-specific resolution
            split='val',
            video_length=val_video_length  # Single frame for Phase 2/3 (2K resolution)
        )

        # Setup samplers for DDP
        if self.world_size > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True
            )
            # Validation: No DistributedSampler - all ranks see the same data
            # This ensures rank 0 can save visualizations for all sequences
            val_sampler = None
        else:
            train_sampler = None
            val_sampler = None

        # Data loaders with custom collate_fn
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),  # Only shuffle if not using sampler
            num_workers=self.config.training.workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.collate_fn
        )

        # Validation: batch_size=1 like original FlashDepth to avoid memory issues
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,  # Process one video sequence at a time
            sampler=val_sampler,
            shuffle=False,
            num_workers=self.config.training.workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.collate_fn
        )

        if self.rank == 0:
            self.logger.info(f"Train dataset size: {len(train_dataset)}")
            self.logger.info(f"Val dataset size: {len(val_dataset)}")
            self.logger.info(f"Train batch size: {self.config.training.batch_size}, video_length: {self.config.dataset.video_length}")
            self.logger.info(f"Val batch size: 1, video_length: {val_video_length} {'(reduced for 2K)' if self.phase >= 2 else '(like original FlashDepth)'}")

        return train_loader, val_loader

    def collate_fn(self, batch):
        """Custom collate function to filter out None values"""
        # Filter out None values
        batch = [item for item in batch if item is not None]

        # If all items are None, skip this batch
        if len(batch) == 0:
            return None

        # Use default collate for valid items
        return torch.utils.data.dataloader.default_collate(batch)

    def _setup_optimizer(self):
        """Setup optimizer - same LR for all trainable modules (all train from scratch)"""
        base_lr = self.config.training.get('gear3_lr', 1e-4)  # Use same LR for all

        # Separate parameter groups (for monitoring and potential future tuning)
        mamba_params = []
        gear3_params = []
        output_conv_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'gear3_upgrade_head' in name:
                    gear3_params.append(param)
                elif 'mamba' in name:
                    mamba_params.append(param)
                elif 'output_conv' in name:
                    output_conv_params.append(param)
                else:
                    # Fallback: should not happen, but log for debugging
                    self.logger.warning(f"Trainable parameter not in any group: {name}")

        param_groups = [
            {'params': gear3_params, 'lr': base_lr, 'name': 'gear3'},
            {'params': mamba_params, 'lr': base_lr, 'name': 'mamba'},
            {'params': output_conv_params, 'lr': base_lr, 'name': 'output_conv'}
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=[0.9, 0.95],  # Same as original FlashDepth
            weight_decay=self.config.training.get('weight_decay', 1e-6)
        )

        self.logger.info(f"Optimizer setup:")
        self.logger.info(f"  Gear3: {len(gear3_params)} params, LR={base_lr}")
        self.logger.info(f"  Mamba: {len(mamba_params)} params, LR={base_lr}")
        self.logger.info(f"  Output_conv: {len(output_conv_params)} params, LR={base_lr}")

        return optimizer

    def _setup_scheduler(self):
        """Setup cosine annealing scheduler with warmup"""
        total_steps = self.config.training.iterations
        warmup_steps = 1000  # Same as original FlashDepth
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
        if self.rank == 0:
            self.logger.info("Starting training...")

        train_iterator = iter(self.train_loader)

        # Use tqdm for progress bar (only rank 0)
        pbar = tqdm(
            range(self.config.training.iterations),
            desc="Training",
            disable=(self.rank != 0)
        )
        
        for step in pbar:
            self.global_step = step

            # Get batch
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_loader)
                batch = next(train_iterator)

            # Skip None batches (all items were invalid)
            if batch is None:
                continue

            # Training step
            loss_dict = self.train_step(batch)
            
            # Get learning rates
            lr_gear3 = self.optimizer.param_groups[0]['lr']
            lr_mamba = self.optimizer.param_groups[1]['lr']

            # Update progress bar (every step)
            postfix_dict = {
                'loss': f'{loss_dict["loss"]:.4f}',
                'lr_g3': f'{lr_gear3:.2e}',
                'lr_mb': f'{lr_mamba:.2e}'
            }

            pbar.set_postfix(postfix_dict)

            # WandB logging (every step)
            wandb_dict = {**loss_dict, 'lr_gear3': lr_gear3, 'lr_mamba': lr_mamba}
            if self.config.training.get('wandb', False):
                wandb.log(wandb_dict, step=step)

            # Training visualization (steps 0, 10, 50, 100, then every 250) - rank 0 only
            vis_steps = [0, 10, 50, 100]
            if (step in vis_steps or step % 250 == 0) and self.train_visualizer and self.rank == 0:
                try:
                    # Use current training batch for visualization
                    self.model.eval()

                    # Unwrap DDP for visualization
                    model = self.model.module if isinstance(self.model, DDP) else self.model

                    # Unpack and move to device (batch is still on CPU from dataloader)
                    images, gt_depth, dataset_idx = batch
                    images = images.to(self.device)
                    gt_depth = gt_depth.to(self.device)

                    if gt_depth.ndim == 3:
                        gt_depth = gt_depth.unsqueeze(1)
                    elif gt_depth.ndim == 4 and gt_depth.shape[1] != 1:
                        gt_depth = gt_depth.unsqueeze(2)

                    # GT depth from dataloader is inverse depth (1/m), scale to 100/m for training
                    gt_depth_inverse_100 = gt_depth * 100.0  # (1/m) * 100 = 100/m

                    # Get batch size for Mamba initialization
                    B_orig, T_orig, C, H, W = images.shape

                    # Initialize Mamba sequence for temporal processing
                    if hasattr(model, 'mamba'):
                        model.mamba.start_new_sequence()

                    # Process entire sequence at once (full sequence processing)
                    images_flat = rearrange(images, 'b t c h w -> (b t) c h w')
                    patch_h, patch_w = H // model.patch_size, W // model.patch_size

                    with torch.no_grad():
                        # Extract features from all frames at once
                        encoder_features = model.pretrained.get_intermediate_layers(
                            images_flat, model.intermediate_layer_idx[model.encoder]
                        )
                        last_block = model.pretrained.blocks[-1]
                        attention_weights = last_block.attn.attn_weights
                        patch_tokens = encoder_features[-1]

                        # Get DPT features WITHOUT Mamba
                        dpt_features = model.depth_head.get_forward_features(
                            encoder_features, patch_h, patch_w
                        )
                        path_1 = dpt_features[-1]  # Extract path_1

                        # Prepare inputs based on separation_method
                        attention_weights_multi_layer = None
                        cls_token = None

                        if self.separation_method == 'multi_layer':
                            # Collect attention weights from multiple layers (encoder-specific)
                            attention_weights_multi_layer = [
                                model.pretrained.blocks[i].attn.attn_weights
                                for i in self.target_blocks
                            ]
                        elif self.separation_method == 'cls_seg':
                            cls_token = encoder_features[-1][:, 0]  # [B*T, embed_dim]

                        # Apply Gear modulation BEFORE Mamba
                        path_1_modulated, importance_map, fg_features, bg_features, fg_mask, bg_mask = model.gear3_upgrade_head(
                            patch_tokens, attention_weights, [path_1], patch_h, patch_w,
                            attention_weights_multi_layer=attention_weights_multi_layer,
                            cls_token=cls_token
                        )

                        # Apply Mamba temporal modeling to modulated feature
                        path_1_temporal = model.dpt_features_to_mamba(
                            input_shape=(B_orig, T_orig, None, H, W),
                            dpt_features=path_1_modulated,
                            in_dpt_layer=0
                        )

                        out = model.depth_head.scratch.output_conv1(path_1_temporal)
                        out = F.interpolate(out, (H, W), mode="bilinear", align_corners=True)
                        out = model.depth_head.scratch.output_conv2(out)

                        # Convert prediction to metric depth: 100/m -> m
                        # Output shape: (B*T, 1, H, W)
                        pred_depth_metric = 100.0 / (out + 1e-8)

                        # Reshape outputs from (B*T, ...) to (B, T, ...)
                        pred_depth_metric_seq = rearrange(pred_depth_metric, '(b t) 1 h w -> b t 1 h w', b=B_orig, t=T_orig)
                        importance_map_seq = rearrange(importance_map, '(b t) c h w -> b t c h w', b=B_orig, t=T_orig)
                        fg_mask_seq = rearrange(fg_mask, '(b t) c h w -> b t c h w', b=B_orig, t=T_orig)
                        bg_mask_seq = rearrange(bg_mask, '(b t) c h w -> b t c h w', b=B_orig, t=T_orig)

                        # Select first frame for visualization
                        pred_depth_metric_vis = pred_depth_metric_seq[:, 0]  # [B, 1, H, W]
                        importance_map_vis = importance_map_seq[:, 0]  # [B, C, H, W]
                        fg_mask_vis = fg_mask_seq[:, 0]  # [B, 1, H, W]
                        bg_mask_vis = bg_mask_seq[:, 0]  # [B, 1, H, W]

                        # GT depth for first frame
                        gt_depth_inverse_vis = gt_depth_inverse_100[:, 0]  # [B, 1, H, W]
                        gt_depth_metric = 100.0 / (gt_depth_inverse_vis + 1e-8)

                        # Resize masks to match image resolution (no need to resize pred_depth_metric_vis as it's already at H, W)
                        importance_map_resized = F.interpolate(
                            importance_map_vis, size=(H, W), mode='bilinear', align_corners=True
                        )
                        fg_mask_resized = F.interpolate(
                            fg_mask_vis, size=(H, W), mode='bilinear', align_corners=True
                        )
                        bg_mask_resized = F.interpolate(
                            bg_mask_vis, size=(H, W), mode='bilinear', align_corners=True
                        )

                        # Move tensors to CPU for visualization (only first batch, first frame)
                        sample_batch = (
                            images[:1, :1].cpu(),  # [1, 1, 3, H, W]
                            gt_depth_metric[:1].cpu(),  # [1, 1, H, W]
                            dataset_idx
                        )
                        model_outputs_cpu = {
                            'pred_depth': pred_depth_metric_vis[:1].cpu(),  # [1, 1, H, W]
                            'importance_map': importance_map_resized[:1].cpu(),  # [1, 1, H, W]
                            'fg_mask': fg_mask_resized[:1].cpu(),  # [1, 1, H, W]
                            'bg_mask': bg_mask_resized[:1].cpu()   # [1, 1, H, W]
                        }

                        # FPS removed from training (only measured in test_gear3.py)
                        current_fps = None

                        # Extract layer_weights for visualization (multi_layer separation only)
                        layer_weights = None
                        if self.separation_method == 'multi_layer':
                            try:
                                # Handle DDP wrapping
                                model = self.model.module if hasattr(self.model, 'module') else self.model
                                # Get softmax-normalized weights
                                fusion_weights = model.gear3_upgrade_head.multi_layer_fusion.fusion_weights
                                layer_weights = torch.softmax(fusion_weights, dim=0).detach().cpu().numpy()
                            except Exception as e:
                                self.logger.warning(f"Failed to extract layer_weights: {e}")

                        # Pass loss_dict and layer_weights for visualization
                        self.train_visualizer.create_validation_summary(
                            sample_batch, model_outputs_cpu, step, prefix="training", fps=current_fps, loss_dict=loss_dict, layer_weights=layer_weights
                        )

                    self._set_train_mode()

                except Exception as e:
                    self.logger.warning(f"Failed to save training visualization: {e}")
                    self._set_train_mode()

            # Validation (run at step 0 and every val_freq steps)
            if step % self.config.training.get('val_freq', 1000) == 0:
                val_metrics = self.validate()
                self.logger.info(f"Validation at step {step}: {val_metrics}")

                # Update current validation loss for checkpoint
                self.current_val_loss = val_metrics['loss']
                self.dataset_losses = val_metrics.get('dataset_losses', None)
                self.num_sequences = val_metrics.get('num_sequences', None)

                if self.config.training.get('wandb', False):
                    wandb.log({f'val/{k}': v for k, v in val_metrics.items()}, step=step)

                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.best_step = step  # Track best step
                    self.save_checkpoint(f'best.pth')
                    self.logger.info(f"New best model at step {step}: val_loss={val_metrics['loss']:.4f}")

                self._set_train_mode()

            # Save checkpoint
            if step % self.config.training.get('save_freq', 5000) == 0 and step > 0:
                self.save_checkpoint(f'checkpoint_step{step}_phase{self.phase}.pth')

        # Final save
        self.save_checkpoint('last.pth')
        self.logger.info("Training completed!")

    def train_step(self, batch):
        """Single training step with BFloat16 autocast"""
        # Unpack batch
        images, gt_depth, dataset_idx = batch
        images = images.to(self.device)
        gt_depth = gt_depth.to(self.device)

        # Add channel dimension if needed
        if gt_depth.ndim == 3:
            gt_depth = gt_depth.unsqueeze(1)
        elif gt_depth.ndim == 4 and gt_depth.shape[1] != 1:
            gt_depth = gt_depth.unsqueeze(2)

        focal_length = 1000.0
        B, T = images.shape[:2]

        # Get the actual model (unwrap DDP if needed)
        model = self.model.module if isinstance(self.model, DDP) else self.model

        # DEBUG: Check raw GT depth values
        if self.global_step < 5:  # Only log first few steps
            self.logger.info(f"DEBUG - Raw GT depth from dataloader: min={gt_depth.min():.4f}, max={gt_depth.max():.4f}, shape={gt_depth.shape}")
            self.logger.info(f"DEBUG - GT depth has {(gt_depth > 0).sum()} valid pixels out of {gt_depth.numel()} total")

        # GT depth from dataloader is inverse depth (1/m), scale to 100/m for training
        # This matches FlashDepth's relative depth scale (≈ 100/metric_depth)
        gt_depth_inverse_100 = gt_depth * 100.0  # (1/m) * 100 = 100/m

        # DEBUG: Check after scaling to 100/m
        if self.global_step < 5:
            self.logger.info(f"DEBUG - After scaling to 100/m: min={gt_depth_inverse_100.min():.4f}, max={gt_depth_inverse_100.max():.4f}")
            self.logger.info(f"DEBUG - Has {(gt_depth_inverse_100 > 0).sum()} valid pixels")

        # Forward pass following original FlashDepth pattern (whole sequence at once)
        # Initialize Mamba sequence (critical for temporal processing!)
        if hasattr(model, 'mamba'):
            model.mamba.start_new_sequence()

        # Reshape video from (B, T, C, H, W) to (B*T, C, H, W) for encoder
        B_orig, T_orig, C, H, W = images.shape
        images_flat = rearrange(images, 'b t c h w -> (b t) c h w')

        patch_h, patch_w = H // model.patch_size, W // model.patch_size

        # Use BFloat16 for forward pass
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # Extract features from DINOv2 (frozen, no grad) - all frames at once
            with torch.no_grad():
                encoder_features = model.pretrained.get_intermediate_layers(
                    images_flat, model.intermediate_layer_idx[model.encoder]
                )

                # Get attention weights from last block (B*T, num_heads, N, N)
                last_block = model.pretrained.blocks[-1]
                attention_weights = last_block.attn.attn_weights

                # Get patch tokens from last encoder layer (B*T, N, C)
                patch_tokens = encoder_features[-1]

            # Get DPT features WITHOUT Mamba (DPT frozen, no grad)
            with torch.no_grad():
                dpt_features = model.depth_head.get_forward_features(
                    encoder_features, patch_h, patch_w
                )  # Returns [path_4, path_3, path_2, path_1], each (B*T, dpt_dim, h, w)
                path_1 = dpt_features[-1]  # Extract path_1: (B*T, dpt_dim, h, w)

            # Prepare inputs based on separation_method
            attention_weights_multi_layer = None
            cls_token = None

            if self.separation_method == 'multi_layer':
                # Collect attention weights from multiple layers (encoder-specific)
                # Each has shape (B*T, num_heads, N, N)
                attention_weights_multi_layer = [
                    model.pretrained.blocks[i].attn.attn_weights.detach()
                    for i in self.target_blocks
                ]
            elif self.separation_method == 'cls_seg':
                with torch.no_grad():
                    cls_token = patch_tokens[:, 0]  # (B*T, embed_dim)

            # Apply Gear3 Upgrade modulation to path_1 (BEFORE Mamba)
            # This makes the feature metric-aware before temporal modeling
            # Inputs: (B*T, ...), Output: (B*T, ...)
            path_1_modulated, importance_map, fg_features, bg_features, fg_mask, bg_mask = model.gear3_upgrade_head(
                patch_tokens, attention_weights, [path_1], patch_h, patch_w,
                attention_weights_multi_layer=attention_weights_multi_layer,
                cls_token=cls_token
            )

            # Apply Mamba temporal modeling to modulated (metric-aware) feature
            # Mamba is trainable in Phase 1+, so gradients flow through it
            path_1_temporal = model.dpt_features_to_mamba(
                input_shape=(B_orig, T_orig, None, H, W),
                dpt_features=path_1_modulated,
                in_dpt_layer=0  # Single layer index
            )  # Returns (B*T, dpt_dim, h, w) with temporal consistency

            # Pass through DPT output head (trainable)
            out = model.depth_head.scratch.output_conv1(path_1_temporal)
            out = F.interpolate(out, (H, W), mode="bilinear", align_corners=True)
            out = model.depth_head.scratch.output_conv2(out)

            # Prediction is already positive (Softplus activation in output_conv2)
            # Shape: (B*T, 1, H, W)
            pred_depth_inverse = out

        # Compute loss (outside autocast for stability)
        # Reshape GT from (B, T, 1, H, W) to (B*T, H, W)
        gt_depth_inverse_flat = rearrange(gt_depth_inverse_100, 'b t 1 h w -> (b t) h w')

        # Remove channel dimension from prediction
        pred_depth_inverse_flat = pred_depth_inverse.squeeze(1)  # (B*T, H, W)

        # DEBUG: Check values in first few steps
        if self.global_step < 5:
            self.logger.info(f"DEBUG - Pred inverse depth: min={pred_depth_inverse_flat.min():.4f}, max={pred_depth_inverse_flat.max():.4f}, mean={pred_depth_inverse_flat.mean():.4f}")

        # Clamp prediction to reasonable range to prevent NaN
        pred_depth_inverse_flat = torch.clamp(pred_depth_inverse_flat, min=1e-3, max=1e4)

        # Compute valid mask: GT and Pred must both be valid
        # Warmup threshold for initial steps (prevent empty batches during initialization)
        if self.global_step < 100:
            MIN_INVERSE_DEPTH = 100.0 / 200.0  # Relaxed: 200m threshold for first 100 steps
        else:
            MIN_INVERSE_DEPTH = 100.0 / 70.0   # Normal: 70m threshold after warmup

        gt_valid_mask = (gt_depth_inverse_flat > MIN_INVERSE_DEPTH)
        pred_valid_mask = (pred_depth_inverse_flat > MIN_INVERSE_DEPTH)
        valid_mask = gt_valid_mask & pred_valid_mask

        if valid_mask.sum() == 0:
            self.logger.error("No valid GT & Pred pixels in batch!")
            return {'loss': 0.0}

        # Compute loss (like original FlashDepth)
        with torch.amp.autocast('cuda', enabled=False):
            loss = self.loss_fn(
                pred_depth_inverse_flat.float(),
                gt_depth_inverse_flat.float(),
                valid_mask.float()
            )

        # Backward pass (outside autocast for numerical stability)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        return {
            'loss': loss.item()
        }

    @torch.no_grad()
    def validate(self):
        """Validation loop with visualization - like original FlashDepth"""
        # Clear cache before validation to free memory from training
        torch.cuda.empty_cache()

        self.model.eval()

        # Unwrap DDP if needed
        model = self.model.module if isinstance(self.model, DDP) else self.model

        total_loss = 0
        num_batches = 0

        # Track per-dataset losses for detailed analysis
        dataset_losses = {}  # {dataset_name: [loss1, loss2, ...]}
        dataset_sequence_counts = {}  # {dataset_name: num_sequences_evaluated}

        # Reset visualization tracking for this validation run
        # This ensures we save the SAME sequences at EVERY validation step (for step-by-step comparison)
        for dataset_name in self.val_vis_config:
            self.val_vis_config[dataset_name]['saved'] = []

        # Track dataset-specific sequence counters for visualization
        dataset_sequence_counters = {'sintel': 0, 'waymo_seg': 0}

        # Phase 2/3: Limit validation batches to save memory (2K resolution)
        # 8 batches to ensure both sintel and waymo_seg are included (DDP splits across ranks)
        max_val_batches = 8 if self.phase >= 2 else None

        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation", disable=(self.rank != 0))):
            # Limit validation batches for Phase 2/3 to prevent OOM
            if max_val_batches is not None and batch_idx >= max_val_batches:
                break

            # Skip None batches (all items were invalid)
            if batch is None:
                continue

            # Unpack batch
            images, gt_depth, dataset_idx = batch
            
            # Get dataset name for this batch (before frame loop)
            if isinstance(dataset_idx, str):
                current_dataset = dataset_idx
            elif isinstance(dataset_idx, (list, tuple)):
                current_dataset = str(dataset_idx[0])
            elif torch.is_tensor(dataset_idx):
                current_dataset = str(dataset_idx[0].item() if dataset_idx.dim() > 0 else dataset_idx.item())
            else:
                current_dataset = str(dataset_idx)

            # Use BFloat16 autocast like original FlashDepth
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                images = images.to(self.device)
                gt_depth = gt_depth.to(self.device)

                # Add channel dimension if needed
                if gt_depth.ndim == 3:
                    gt_depth = gt_depth.unsqueeze(1)
                elif gt_depth.ndim == 4 and gt_depth.shape[1] != 1:
                    gt_depth = gt_depth.unsqueeze(2)

                B, T = images.shape[:2]

                # GT depth from dataloader is inverse depth (1/m), scale to 100/m for training
                gt_depth_inverse_100 = gt_depth * 100.0  # (1/m) * 100 = 100/m

                # Initialize Mamba sequence for temporal processing
                if hasattr(model, 'mamba'):
                    model.mamba.start_new_sequence()

                # Process entire sequence at once (like training)
                B_orig, T_orig, C, H, W = images.shape
                images_flat = rearrange(images, 'b t c h w -> (b t) c h w')
                patch_h, patch_w = H // model.patch_size, W // model.patch_size

                # Extract features from DINOv2 - all frames at once
                encoder_features = model.pretrained.get_intermediate_layers(
                    images_flat, model.intermediate_layer_idx[model.encoder]
                )

                # Get attention weights from last block (B*T, num_heads, N, N)
                last_block = model.pretrained.blocks[-1]
                attention_weights = last_block.attn.attn_weights
                patch_tokens = encoder_features[-1]

                # Get DPT features WITHOUT Mamba
                dpt_features = model.depth_head.get_forward_features(
                    encoder_features, patch_h, patch_w
                )  # Returns [path_4, path_3, path_2, path_1]
                path_1 = dpt_features[-1]  # Extract path_1: (B*T, dpt_dim, h, w)

                # Prepare inputs based on separation_method
                attention_weights_multi_layer = None
                cls_token = None

                if self.separation_method == 'multi_layer':
                    # Collect attention weights from multiple layers (encoder-specific)
                    attention_weights_multi_layer = [
                        model.pretrained.blocks[i].attn.attn_weights.detach()
                        for i in self.target_blocks
                    ]
                elif self.separation_method == 'cls_seg':
                    cls_token = patch_tokens[:, 0]  # (B*T, embed_dim)

                # Apply Gear modulation BEFORE Mamba (metric-aware feature)
                # Inputs/outputs: (B*T, ...)
                path_1_modulated, importance_map, fg_features, bg_features, fg_mask, bg_mask = model.gear3_upgrade_head(
                    patch_tokens, attention_weights, [path_1], patch_h, patch_w,
                    attention_weights_multi_layer=attention_weights_multi_layer,
                    cls_token=cls_token
                )

                # Apply Mamba temporal modeling to modulated feature
                path_1_temporal = model.dpt_features_to_mamba(
                    input_shape=(B_orig, T_orig, None, H, W),
                    dpt_features=path_1_modulated,
                    in_dpt_layer=0
                )  # Returns (B*T, dpt_dim, h, w) with temporal consistency

                # Get depth (at model resolution)
                out = model.depth_head.scratch.output_conv1(path_1_temporal)
                out = F.interpolate(out, (H, W), mode="bilinear", align_corners=True)
                out = model.depth_head.scratch.output_conv2(out)

                pred_depth_inverse = out  # [B*T, 1, H, W] at model resolution

                # Reshape GT from (B, T, 1, H, W) to (B*T, 1, H, W)
                gt_depth_inverse_flat = rearrange(gt_depth_inverse_100, 'b t 1 h w -> (b t) 1 h w')

                # Interpolate prediction to GT resolution (like original FlashDepth validation)
                gt_shape = gt_depth_inverse_flat.shape[-2:]
                if pred_depth_inverse.shape[-2:] != gt_shape:
                    pred_depth_inverse = F.interpolate(
                        pred_depth_inverse, size=gt_shape, mode="bilinear", align_corners=True
                    )

                # Compute loss for entire sequence
                # Apply same 70m threshold as training for consistency
                MIN_INVERSE_DEPTH = 100.0 / 70.0  # Filter out >70m depths (same as training)

                # GT and Pred must both be valid (same as metrics calculation)
                gt_valid_mask = (gt_depth_inverse_flat >= MIN_INVERSE_DEPTH)  # GT within 70m and valid
                pred_valid_mask = (pred_depth_inverse >= MIN_INVERSE_DEPTH)  # Pred within reasonable range
                valid_mask = (gt_valid_mask & pred_valid_mask).float()

                if valid_mask.sum() > 0:
                    loss_batch = self.loss_fn(pred_depth_inverse, gt_depth_inverse_flat, valid_mask)
                    frame_losses = [loss_batch.float()]
                else:
                    frame_losses = []

                # Validation visualization: save multiple sequences per dataset (rank 0 only)
                # Reshape predictions back to (B, T, 1, H, W) for visualization
                pred_depth_inverse_seq = rearrange(pred_depth_inverse, '(b t) 1 h w -> b t 1 h w', b=B_orig, t=T_orig)

                # Use first frame for visualization
                if self.val_visualizer and self.rank == 0:
                    # Check if this dataset is in our visualization config
                    if current_dataset in self.val_vis_config:
                        config = self.val_vis_config[current_dataset]
                        seq_num = dataset_sequence_counters[current_dataset]

                        # Check if we should save this sequence
                        should_save = (
                            seq_num in config['sequences'] and
                            seq_num not in config['saved']
                        )

                        if should_save:
                            try:
                                # Use first frame (t=0) for visualization
                                pred_depth_inverse_vis = pred_depth_inverse_seq[:, 0]  # [B, 1, H, W]
                                gt_depth_inverse_vis = gt_depth_inverse_100[:, 0]  # [B, 1, H, W]

                                # Convert to metric depth for visualization: 100/m -> m
                                # Convert to Float32 for CPU operations
                                pred_depth_metric = (100.0 / (pred_depth_inverse_vis.float() + 1e-8)).cpu()
                                gt_depth_metric = (100.0 / (gt_depth_inverse_vis.float() + 1e-8)).cpu()

                                # Get GT resolution for visualization
                                gt_h, gt_w = gt_depth_inverse_vis.shape[-2:]

                                # Reshape importance/mask maps to (B, T, ...) and select first frame
                                importance_map_seq = rearrange(importance_map, '(b t) c h w -> b t c h w', b=B_orig, t=T_orig)
                                fg_mask_seq = rearrange(fg_mask, '(b t) c h w -> b t c h w', b=B_orig, t=T_orig)
                                bg_mask_seq = rearrange(bg_mask, '(b t) c h w -> b t c h w', b=B_orig, t=T_orig)

                                importance_map_vis = importance_map_seq[:, 0]  # [B, C, h, w]
                                fg_mask_vis = fg_mask_seq[:, 0]
                                bg_mask_vis = bg_mask_seq[:, 0]

                                # Resize importance map to GT resolution
                                importance_map_resized = F.interpolate(
                                    importance_map_vis, size=(gt_h, gt_w), mode='bilinear', align_corners=True
                                )

                                # Resize FG/BG masks to GT resolution
                                fg_mask_resized = F.interpolate(
                                    fg_mask_vis, size=(gt_h, gt_w), mode='bilinear', align_corners=True
                                )
                                bg_mask_resized = F.interpolate(
                                    bg_mask_vis, size=(gt_h, gt_w), mode='bilinear', align_corners=True
                                )

                                # Resize images to GT resolution (for visualization consistency)
                                img_vis = images[:, 0]  # [B, C, H, W] - first frame
                                img_t_resized = F.interpolate(
                                    img_vis, size=(gt_h, gt_w), mode='bilinear', align_corners=True
                                )

                                model_outputs = {
                                    'pred_depth': pred_depth_metric,  # [B, 1, gt_h, gt_w] at GT resolution
                                    'importance_map': importance_map_resized.float().cpu(),  # [B, 1, gt_h, gt_w]
                                    'fg_mask': fg_mask_resized.float().cpu(),  # [B, 1, gt_h, gt_w]
                                    'bg_mask': bg_mask_resized.float().cpu()   # [B, 1, gt_h, gt_w]
                                }

                                # For visualization, we need [B, T, ...] format like training
                                # But we only have one frame (t=0), so unsqueeze T dimension
                                sample_batch = (
                                    img_t_resized.unsqueeze(1).float().cpu(),  # [B, 1, C, gt_h, gt_w] at GT resolution
                                    gt_depth_metric.unsqueeze(1),  # [B, 1, gt_h, gt_w]
                                    dataset_idx
                                )

                                # FPS removed from training (only measured in test_gear3.py)
                                current_fps = None

                                # Create loss_dict with current frame loss (for visualization)
                                val_loss_dict = {
                                    'val_loss': loss_batch.item() if len(frame_losses) > 0 else 0.0
                                }

                                # Extract layer_weights for visualization (multi_layer separation only)
                                layer_weights = None
                                if self.separation_method == 'multi_layer':
                                    try:
                                        # Handle DDP wrapping
                                        model = self.model.module if hasattr(self.model, 'module') else self.model
                                        # Get softmax-normalized weights
                                        fusion_weights = model.gear3_upgrade_head.multi_layer_fusion.fusion_weights
                                        layer_weights = torch.softmax(fusion_weights, dim=0).detach().cpu().numpy()
                                    except Exception as e:
                                        self.logger.warning(f"Failed to extract layer_weights: {e}")

                                # Save with dataset and sequence-specific name
                                save_name = f"validation_{current_dataset}_seq{seq_num:03d}_step_{self.global_step:06d}"
                                self.val_visualizer.create_validation_summary(
                                    sample_batch, model_outputs, self.global_step,
                                    save_name=save_name, fps=current_fps, loss_dict=val_loss_dict, dataset_name=current_dataset, layer_weights=layer_weights
                                )
                                config['saved'].append(seq_num)
                                self.logger.info(f"Saved validation visualization: {current_dataset} sequence {seq_num} ({len(config['saved'])}/{len(config['sequences'])})")
                            except Exception as e:
                                if self.rank == 0:
                                    self.logger.warning(f"Failed to save validation visualization for {current_dataset} seq {seq_num}: {e}")

                    # Clear intermediate tensors to free memory after each sequence
                    del encoder_features, attention_weights, patch_tokens, dpt_features, path_1, path_1_temporal
                    del path_1_modulated, importance_map, fg_features, bg_features, fg_mask, bg_mask, pred_depth_inverse
                    # Always clear cache after each sequence to prevent OOM during validation
                    torch.cuda.empty_cache()                # Track sequence count (regardless of valid pixels)
                if current_dataset not in dataset_sequence_counts:
                    dataset_sequence_counts[current_dataset] = 0
                dataset_sequence_counts[current_dataset] += 1

                # Track loss only if valid pixels exist
                if len(frame_losses) > 0:
                    avg_loss = sum(frame_losses) / len(frame_losses)
                    total_loss += avg_loss.item()
                    num_batches += 1

                    # Track per-dataset loss
                    if current_dataset not in dataset_losses:
                        dataset_losses[current_dataset] = []
                    dataset_losses[current_dataset].append(avg_loss.item())

            # Increment sequence counter for this dataset
            if current_dataset in dataset_sequence_counters:
                dataset_sequence_counters[current_dataset] += 1

            # Clear batch memory and GPU cache to prevent OOM
            del images, gt_depth, gt_depth_inverse_100
            torch.cuda.empty_cache()

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

        # Log detailed per-dataset statistics (rank 0 only)
        if self.rank == 0:
            self.logger.info("=" * 80)
            self.logger.info(f"VALIDATION SUMMARY (Step {self.global_step})")
            self.logger.info("=" * 80)
            self.logger.info(f"Total batches with valid pixels: {num_batches}")

            # Show all datasets (including those with no valid pixels)
            for dataset_name in dataset_sequence_counts.keys():
                total_seqs = dataset_sequence_counts[dataset_name]
                if dataset_name in dataset_losses:
                    losses = dataset_losses[dataset_name]
                    dataset_avg = np.mean(losses)
                    dataset_std = np.std(losses)
                    valid_seqs = len(losses)
                    self.logger.info(
                        f"  [{dataset_name}] {valid_seqs}/{total_seqs} sequences with valid pixels | "
                        f"Loss: {dataset_avg:.4f} ± {dataset_std:.4f}"
                    )
                else:
                    self.logger.info(
                        f"  [{dataset_name}] 0/{total_seqs} sequences with valid pixels | "
                        f"Loss: N/A (no valid pixels)"
                    )

            self.logger.info(f"Overall Average Loss: {avg_loss:.4f} (from {num_batches} sequences)")

            # WARNING: Check if validation set is too small (Phase 2/3)
            if self.phase >= 2 and num_batches < 20:
                self.logger.warning(
                    f"⚠️  SMALL VALIDATION SET: Only {num_batches} batches evaluated! "
                    f"Consider increasing max_val_batches for more reliable validation."
                )

            self.logger.info("=" * 80)

        # Clear cache after validation
        torch.cuda.empty_cache()

        # Back to training mode
        self.model.train()

        return {
            'loss': avg_loss,
            'dataset_losses': {k: float(np.mean(v)) for k, v in dataset_losses.items()},
            'num_sequences': {k: v for k, v in dataset_sequence_counts.items()}
        }

    def save_checkpoint(self, filename):
        """Save model checkpoint (only rank 0)"""
        if self.rank != 0:
            return

        checkpoint_path = self.results_dir / filename

        # Unwrap DDP for saving
        model = self.model.module if isinstance(self.model, DDP) else self.model

        checkpoint = {
            'global_step': self.global_step,
            'model': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_step': self.best_step,  # Save which step was best
            'current_val_loss': self.current_val_loss,  # Save current validation loss
            'dataset_losses': self.dataset_losses,  # Save per-dataset validation losses
            'num_sequences': self.num_sequences,  # Save number of sequences per dataset
            'config': OmegaConf.to_container(self.config, resolve=True),
            'phase': self.phase
        }

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")


@hydra.main(version_base=None, config_path="configs/gear3", config_name="config")
def main(config: DictConfig):
    """Main entry point"""
    # Initialize distributed training
    rank, world_size, local_rank = init_distributed()

    # Create trainer
    trainer = Gear3UpgradeTrainer(config, rank, world_size, local_rank)
    trainer.train()

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
