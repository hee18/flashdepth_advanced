"""
Gear2 Training Script: Ablation Study (No FG/BG Separation)

Three-phase training:
    Phase 1: Train on 5 datasets (518×518)
    Phase 2: Train on mvs-synth, spring (2K resolution)
    Phase 3: Fine-tune on nuScenes (2K resolution)

Key differences from Gear3:
    - No importance map computation
    - No FG/BG separation
    - Uses CLS token for global feature extraction
    - Uniform modulation (same gamma/beta for all pixels)
    - Feature-level modulation using FiLM
    - Loss on inverse depth: loss(100/pred, 100/gt)
"""

import os
import numpy as np
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
from pathlib import Path
from einops import rearrange
import math
import time
from datetime import timedelta

from flashdepth.model import FlashDepth
from utils.gear2_visualization import Gear2Visualizer
from flashdepth.gear2_modules import Gear2MetricHead
from flashdepth.gear3_upgrade_modules import Gear3UpgradeAblationHead
from dataloaders.combined_dataset import CombinedDataset
from utils.helpers import *
from utils.metric_depth_metrics import MetricDepthMetrics
from utils.gear_losses import LogL1Loss, DepthVariancePseudoLabelLoss, EdgeAwareLoss, ContrastiveFGBGLoss


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



class Gear2Trainer:
    """
    Trainer for Gear2 metric depth learning.

    Frozen: DINOv2, DPT
    Fine-tuned: Mamba (LR: 1e-5)
    Trained: Gear2 modules (LR: 1e-4)
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
        self.results_dir = Path(config.get('results_dir', f'./train_results/gear2{phase_suffix}'))
        if rank == 0:
            self.results_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging (only rank 0)
        # Force immediate flush after each log
        class FlushFileHandler(logging.FileHandler):
        
    def _get_canonical_focal_length(self):
        """
        Get canonical focal length (fixed at 1000.0 for all resolutions).

        Returns:
            float: Canonical focal length (always 1000.0)
        """
        return 1000.0


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

        # Regularization losses - ALL DISABLED
        # Importance map now uses raw attention directly (no trainable parameters)
        # Therefore no regularization is needed
        self.use_depth_variance_loss = False
        self.use_edge_aware_loss = False
        self.use_contrastive_fgbg_loss = False

        if rank == 0:
            self.logger.info("=== Regularization Losses ===")
            self.logger.info("✗ ALL regularization losses disabled (importance map uses raw attention)")

        # FPS measurement removed (only in test_gear2.py with batch=1 for fair comparison)

        # Setup wandb
        if config.training.get('wandb', False):
            wandb.init(
                project="flashdepth-gear2",
                name=f"gear2_phase{self.phase}_{config.training.get('wandb_name', 'experiment')}",
                config=dict(config)
            )

        # Setup visualizer with separate folders
        self.train_visualizer = Gear2Visualizer(save_dir=self.results_dir / "visualizations" / "train")
        self.val_visualizer = Gear2Visualizer(save_dir=self.results_dir / "visualizations" / "valid")

        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_step = 0  # Track which step achieved best validation loss
        self.current_val_loss = None  # Track current validation loss for checkpoint
        self.dataset_losses = None  # Track per-dataset validation losses
        self.num_sequences = None  # Track number of sequences per dataset

        # Validation visualization config: track which sequences to visualize
        # Phase 2/3: Save 1 sample per dataset at every validation step (for step-by-step comparison)
        # Sintel is smaller (1022×434), Waymo is 2K (1918×1274)
        if self.phase in [2, 3]:
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
        """Initialize FlashDepth with Gear2 metric head"""
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
                    # Exclude: Mamba (modulated input), output_conv1/2 (modulated features), gear2_head
                    loaded_dict = {}
                    excluded_keys = []
                    for k, v in state_dict.items():
                        # Exclude modules that receive modulated features
                        if any(x in k for x in ['mamba', 'hybrid_fusion', 'teacher_model',
                                                'output_conv1', 'output_conv2', 'gear2_head']):
                            excluded_keys.append(k)
                        else:
                            loaded_dict[k] = v

                    # Load state dict (strict=False to allow missing modules)
                    model.load_state_dict(loaded_dict, strict=False)
                    self.logger.info(f"Phase 1: Loaded {len(loaded_dict)} parameters from checkpoint")
                    self.logger.info(f"  - DINOv2 encoder: ✓")
                    self.logger.info(f"  - DPT projects/resize/refinenet: ✓")
                    self.logger.info(f"Excluded {len(excluded_keys)} parameters (will train from scratch):")
                    self.logger.info(f"  - Mamba, output_conv1/2, gear2_head")
                else:
                    # Phase 2, 3: Load ALL parameters including gear2_head
                    # But need to add gear2_head first before loading
                    pass  # Will load after gear2_head is created
            else:
                self.logger.warning(f"Checkpoint {checkpoint_path} not found")

        # Add Gear2 metric head (using multi-layer CLS ablation)
        embed_dim = 1024 if model.encoder == 'vitl' else 384
        dpt_dim = 256 if model.encoder == 'vitl' else 64

        # Use Gear3 Upgrade Ablation Head (multi-layer CLS, no FG/BG separation)
        model.gear2_head = Gear3UpgradeAblationHead(
            embed_dim=embed_dim,
            dpt_dim=dpt_dim
        )
        self.logger.info("Using Gear3 Upgrade Ablation Head for multi-layer CLS ablation study")

        # Phase 2, 3: Load ALL parameters after gear2_head is created
        if self.phase > 1 and checkpoint_path and checkpoint_path != 'true' and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            # Load all parameters including gear2_head
            model.load_state_dict(state_dict, strict=False)
            self.logger.info(f"Phase {self.phase}: Loaded ALL parameters from Phase 1 checkpoint")
            self.logger.info(f"  - DINOv2, DPT, Mamba, output_conv, Gear2: ✓")

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
                                                     'output_conv1', 'output_conv2', 'gear2_head']):
                            loaded_hybrid[k] = v

                    # Overwrite ViT-DPT parameters
                    model.load_state_dict(loaded_hybrid, strict=False)
                    self.logger.info(f"Phase 2: Overwritten {len(loaded_hybrid)} ViT-DPT parameters with Hybrid weights")
                    self.logger.info(f"  - Kept from Phase 1: Gear2, Mamba, output_conv (continue training)")
                else:
                    self.logger.warning(f"Hybrid checkpoint {hybrid_path} not found! Using Phase 1 ViT-DPT weights.")

        # Enable attention weights storage for multi-layer extraction (blocks 4, 11, 17, 23)
        # Aligned with DPT layers for consistent feature extraction
        target_blocks = [4, 11, 17, 23]  # Blocks 4, 11, 17, 23 (aligned with DPT layers)
        for i, block in enumerate(model.pretrained.blocks):
            if i in target_blocks:
                block.attn.store_attn_weights = True
                self.logger.info(f"Enabled attention weights storage for block {i}")
            else:
                block.attn.store_attn_weights = False

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
        Train from scratch: Mamba, output_conv1/2, Gear2 modules
        """
        frozen_params = 0
        mamba_params = 0
        output_conv_params = 0
        gear2_params = 0

        for name, param in model.named_parameters():
            if 'gear2_head' in name:
                # Gear2 modules: trainable
                param.requires_grad = True
                gear2_params += param.numel()
                self.logger.info(f"Trainable (Gear2): {name} - {param.shape}")

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
        self.logger.info(f"Trainable (Gear2 modules): {gear2_params:,}")

        total_trainable = mamba_params + output_conv_params + gear2_params
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
            if any(keyword in name for keyword in ['gear2_head', 'mamba', 'output_conv']):
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
        base_lr = self.config.training.get('gear2_lr', 1e-4)  # Use same LR for all

        # Separate parameter groups (for monitoring and potential future tuning)
        mamba_params = []
        gear2_params = []
        output_conv_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'gear2_head' in name:
                    gear2_params.append(param)
                elif 'mamba' in name:
                    mamba_params.append(param)
                elif 'output_conv' in name:
                    output_conv_params.append(param)
                else:
                    # Fallback: should not happen, but log for debugging
                    self.logger.warning(f"Trainable parameter not in any group: {name}")

        param_groups = [
            {'params': gear2_params, 'lr': base_lr, 'name': 'gear2'},
            {'params': mamba_params, 'lr': base_lr, 'name': 'mamba'},
            {'params': output_conv_params, 'lr': base_lr, 'name': 'output_conv'}
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=[0.9, 0.95],  # Same as original FlashDepth
            weight_decay=self.config.training.get('weight_decay', 1e-6)
        )

        self.logger.info(f"Optimizer setup:")
        self.logger.info(f"  Gear2: {len(gear2_params)} params, LR={base_lr}")
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
            lr_gear2 = self.optimizer.param_groups[0]['lr']
            lr_mamba = self.optimizer.param_groups[1]['lr']

            # Update progress bar (every step)
            postfix_dict = {
                'loss': f'{loss_dict["loss"]:.4f}',
                'depth': f'{loss_dict["depth_loss"]:.4f}',
                'lr_g3': f'{lr_gear2:.2e}',
                'lr_mb': f'{lr_mamba:.2e}'
            }

            pbar.set_postfix(postfix_dict)

            # WandB logging (every step)
            wandb_dict = {**loss_dict, 'lr_gear2': lr_gear2, 'lr_mamba': lr_mamba}
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
                    images, gt_depth, focal_lengths, dataset_idx = batch
                    images = images.to(self.device)
                    gt_depth = gt_depth.to(self.device)
                    focal_lengths = focal_lengths.to(self.device)

                    if gt_depth.ndim == 3:
                        gt_depth = gt_depth.unsqueeze(1)
                    elif gt_depth.ndim == 4 and gt_depth.shape[1] != 1:
                        gt_depth = gt_depth.unsqueeze(2)

                    # GT depth from dataloader is inverse depth (1/m), scale to 100/m
                    gt_depth_inverse_100 = gt_depth * 100.0  # (1/m) * 100 = 100/m

                    # Get batch size for Mamba initialization
                    B, T = images.shape[:2]

                    # Apply canonical space transformation if enabled (for visualization consistency)
                    if self.config.get('use_canonical_space', False):
                        CANONICAL_FX = self._get_canonical_focal_length()

                        # Transform inverse depth directly to canonical space
                        # inverse_canonical = inverse_actual * (fx_actual / CANONICAL_FX)
                        fx_actual = focal_lengths.view(B, T, 1, 1, 1)
                        gt_depth_inverse_100 = gt_depth_inverse_100 * (fx_actual / CANONICAL_FX)

                    # Initialize Mamba sequence for temporal processing
                    if hasattr(model, 'mamba'):
                        model.mamba.start_new_sequence()

                    img_t = images[:, 0]
                    gt_t_inverse_100 = gt_depth_inverse_100[:, 0]

                    with torch.no_grad():
                        encoder_features = model.pretrained.get_intermediate_layers(
                            img_t, model.intermediate_layer_idx[model.encoder]
                        )

                        # Extract multi-layer CLS tokens
                        cls_tokens_multi_layer = [
                            encoder_features[i][:, 0] for i in range(len(encoder_features))
                        ]

                        h, w = img_t.shape[2:]
                        patch_h, patch_w = h // model.patch_size, w // model.patch_size

                        # Get DPT features WITHOUT Mamba
                        dpt_features = model.depth_head.get_forward_features(
                            encoder_features, patch_h, patch_w
                        )
                        path_1 = dpt_features[-1]  # Extract path_1

                        # Apply Gear2 modulation BEFORE Mamba
                        path_1_modulated, importance_map, fg_features, bg_features, fg_mask, bg_mask = model.gear2_head(
                            cls_tokens_multi_layer, [path_1]
                        )

                        # Apply Mamba temporal modeling to modulated feature
                        T_vis = 1
                        path_1_temporal = model.dpt_features_to_mamba(
                            input_shape=(B, T_vis, None, h, w),
                            dpt_features=path_1_modulated,
                            in_dpt_layer=0
                        )

                        out = model.depth_head.scratch.output_conv1(path_1_temporal)
                        out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
                        out = model.depth_head.scratch.output_conv2(out)

                        # Save prediction inverse depth for mask calculation
                        pred_depth_inverse = out  # [B, 1, H, W]

                        # Convert to metric depth for visualization: 100/m -> m
                        # Already positive (Softplus activation in output_conv2)
                        pred_depth_metric = 100.0 / (out + 1e-8)  # 100 / (100/m) = m
                        gt_depth_metric = 100.0 / (gt_t_inverse_100 + 1e-8)  # 100 / (100/m) = m

                        # Compute canonical masks for visualization (70m threshold in canonical space)
                        # Use same logic as training loss calculation (but no warmup)
                        MIN_INVERSE_DEPTH_VIS = 100.0 / 70.0  # Canonical space 70m
                        canonical_gt_valid_vis = (gt_t_inverse_100 > MIN_INVERSE_DEPTH_VIS)  # [B, 1, H, W]

                        # Training: Pred outlier filtering (200m, like training loss)
                        MAX_DEPTH_OUTLIER_VIS = 200.0
                        MIN_INVERSE_OUTLIER_VIS = 100.0 / MAX_DEPTH_OUTLIER_VIS
                        canonical_pred_valid_vis = (pred_depth_inverse > MIN_INVERSE_OUTLIER_VIS)  # [B, 1, H, W]

                        # Handle importance_map (None for Gear2)
                        if importance_map is not None:
                            importance_map_resized = F.interpolate(
                                importance_map, size=(h, w), mode='bilinear', align_corners=True
                            )
                            importance_map_cpu = importance_map_resized[:1].cpu()
                        else:
                            importance_map_cpu = None

                        # Move tensors to CPU for visualization (only first batch, first frame)
                        sample_batch = (
                            images[:1, :1].float().cpu(),  # [1, 1, 3, H, W] - convert BFloat16 to Float32 first
                            gt_depth_metric[:1].float().cpu(),  # [1, 1, H, W] (already has channel dim)
                            dataset_idx,
                            focal_lengths[:1, :1].float().cpu()  # [1, 1] - resized focal length
                        )
                        model_outputs_cpu = {
                            'pred_depth': pred_depth_metric[:1].cpu(),  # [1, 1, H, W]
                            'importance_map': importance_map_cpu,  # [1, 1, H, W] or None
                            'canonical_gt_valid': canonical_gt_valid_vis[:1].cpu(),  # [1, 1, H, W] - canonical space mask
                            'canonical_pred_valid': canonical_pred_valid_vis[:1].cpu()  # [1, 1, H, W] - canonical space mask
                        }

                        # FPS removed from training (only measured in test_gear2.py)
                        current_fps = None

                        # Pass loss_dict for visualization
                        self.train_visualizer.create_validation_summary(
                            sample_batch, model_outputs_cpu, step, prefix="training", fps=current_fps, loss_dict=loss_dict, config=self.config
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
        # Unpack batch (updated to include focal_lengths)
        images, gt_depth, focal_lengths, dataset_idx = batch
        images = images.to(self.device)
        gt_depth = gt_depth.to(self.device)
        focal_lengths = focal_lengths.to(self.device)  # Shape: (B, T)

        # Add channel dimension if needed
        if gt_depth.ndim == 3:
            gt_depth = gt_depth.unsqueeze(1)
        elif gt_depth.ndim == 4 and gt_depth.shape[1] != 1:
            gt_depth = gt_depth.unsqueeze(2)

        B, T = images.shape[:2]

        # Get the actual model (unwrap DDP if needed)
        model = self.model.module if isinstance(self.model, DDP) else self.model

        # DEBUG: Check raw GT depth values
        if self.global_step < 5:  # Only log first few steps
            self.logger.info(f"DEBUG - Raw GT from dataloader (inverse depth 1/m): min={gt_depth.min():.4f}, max={gt_depth.max():.4f}, shape={gt_depth.shape}")
            self.logger.info(f"DEBUG - GT has {(gt_depth > 0).sum()} valid pixels out of {gt_depth.numel()} total")

        # GT depth from dataloader is inverse depth (1/m), scale to 100/m for training
        # This matches FlashDepth's relative depth scale (≈ 100/metric_depth)
        gt_depth_inverse_100 = gt_depth * 100.0  # (1/m) * 100 = 100/m

        # DEBUG: Check after scaling
        if self.global_step < 5:
            self.logger.info(f"DEBUG - After scaling to 100/m: min={gt_depth_inverse_100.min():.4f}, max={gt_depth_inverse_100.max():.4f}")
            self.logger.info(f"DEBUG - Valid pixels: {(gt_depth_inverse_100 > 0).sum()}")

        # Apply canonical space transformation if enabled
        if self.config.get('use_canonical_space', False):
            CANONICAL_FX = self._get_canonical_focal_length()

            # Transform inverse depth directly to canonical space
            # inverse_canonical = inverse_actual * (fx_actual / CANONICAL_FX)
            # Math: depth_canonical = depth_actual * (CANONICAL_FX / fx_actual)
            #       1/depth_canonical = (fx_actual / CANONICAL_FX) * (1/depth_actual)
            fx_actual = focal_lengths.view(B, T, 1, 1, 1)
            gt_depth_inverse_100 = gt_depth_inverse_100 * (fx_actual / CANONICAL_FX)

            if self.global_step < 5:
                self.logger.info(f"DEBUG - Canonical space enabled (CANONICAL_FX={CANONICAL_FX})")
                self.logger.info(f"DEBUG - fx_actual range: {fx_actual.min():.1f} - {fx_actual.max():.1f}")
                self.logger.info(f"DEBUG - After canonical transform: min={gt_depth_inverse_100.min():.4f}, max={gt_depth_inverse_100.max():.4f}")

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

            # Get DPT features WITHOUT Mamba
            dpt_features = model.depth_head.get_forward_features(
                encoder_features, patch_h, patch_w
            )  # Returns [path_4, path_3, path_2, path_1]
            path_1 = dpt_features[-1]  # Extract path_1: (B*T, dpt_dim, h, w)

            # Extract multi-layer CLS tokens for ablation study
            # Get CLS tokens from layers 4, 11, 17, 23 (all encoder features)
            cls_tokens_multi_layer = [
                encoder_features[i][:, 0] for i in range(len(encoder_features))
            ]

            # Apply Gear2 modulation BEFORE Mamba (metric-aware feature)
            # Inputs: (B*T, ...), Output: (B*T, ...)
            path_1_modulated, importance_map, fg_features, bg_features, fg_mask, bg_mask = model.gear2_head(
                cls_tokens_multi_layer, [path_1]
            )

            # Apply Mamba temporal modeling to modulated feature
            path_1_temporal = model.dpt_features_to_mamba(
                input_shape=(B_orig, T_orig, None, H, W),
                dpt_features=path_1_modulated,
                in_dpt_layer=0
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

        # Compute valid mask: GT valid + Pred outlier filtering
        # GT valid: Only compute loss where GT is valid (70m threshold)
        # Pred outlier: Filter out extreme predictions (>200m outliers)
        if self.global_step < 100:
            MIN_INVERSE_DEPTH = 100.0 / 200.0  # Relaxed: 200m threshold for first 100 steps
        else:
            MIN_INVERSE_DEPTH = 100.0 / 70.0   # Normal: 70m threshold after warmup

        # GT valid mask: where GT depth is within valid range
        gt_valid_mask = (gt_depth_inverse_flat > MIN_INVERSE_DEPTH)

        # Pred outlier mask: filter extreme predictions (>200m is outlier)
        MAX_DEPTH_OUTLIER = 200.0
        MIN_INVERSE_OUTLIER = 100.0 / MAX_DEPTH_OUTLIER
        pred_outlier_mask = (pred_depth_inverse_flat > MIN_INVERSE_OUTLIER)

        # Final mask: GT valid AND pred not outlier
        valid_mask = gt_valid_mask & pred_outlier_mask

        if valid_mask.sum() == 0:
            self.logger.error("No valid GT & Pred pixels in batch!")
            return {'loss': 0.0, 'depth_loss': 0.0}

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
            'loss': loss.item(),
            'depth_loss': loss.item()
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

        # Phase 2: Limit validation batches to save memory (2K resolution)
        # 8 batches to ensure both sintel and waymo_seg are included (DDP splits across ranks)
        max_val_batches = 8 if self.phase >= 2 else None

        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation", disable=(self.rank != 0))):
            # Limit validation batches for Phase 2/3 to prevent OOM
            if max_val_batches is not None and batch_idx >= max_val_batches:
                break

            # Skip None batches (all items were invalid)
            if batch is None:
                continue

            # Unpack batch (updated to include focal_lengths)
            images, gt_depth, focal_lengths, dataset_idx = batch

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
                focal_lengths = focal_lengths.to(self.device)  # Shape: (B, T)

                # Add channel dimension if needed
                if gt_depth.ndim == 3:
                    gt_depth = gt_depth.unsqueeze(1)
                elif gt_depth.ndim == 4 and gt_depth.shape[1] != 1:
                    gt_depth = gt_depth.unsqueeze(2)

                B, T = images.shape[:2]

                # GT depth from dataloader is inverse depth (1/m), scale to 100/m
                gt_depth_inverse_100 = gt_depth * 100.0  # (1/m) * 100 = 100/m

                # Apply canonical space transformation if enabled
                if self.config.get('use_canonical_space', False):
                    CANONICAL_FX = self._get_canonical_focal_length()

                    # Transform inverse depth directly to canonical space
                    # inverse_canonical = inverse_actual * (fx_actual / CANONICAL_FX)
                    fx_actual = focal_lengths.view(B, T, 1, 1, 1)
                    gt_depth_inverse_100 = gt_depth_inverse_100 * (fx_actual / CANONICAL_FX)

                # Process all frames in sequence (like original FlashDepth)
                frame_losses = []

                # Initialize Mamba sequence for validation
                if hasattr(model, 'mamba'):
                    model.mamba.start_new_sequence()

                for t in range(T):
                    img_t = images[:, t]
                    gt_t_inverse_100 = gt_depth_inverse_100[:, t]

                    # Extract features from DINOv2
                    encoder_features = model.pretrained.get_intermediate_layers(
                        img_t, model.intermediate_layer_idx[model.encoder]
                    )

                    # Extract multi-layer CLS tokens
                    cls_tokens_multi_layer = [
                        encoder_features[i][:, 0] for i in range(len(encoder_features))
                    ]

                    # Get DPT features WITHOUT Mamba
                    h, w = img_t.shape[2:]
                    patch_h, patch_w = h // model.patch_size, w // model.patch_size

                    dpt_features = model.depth_head.get_forward_features(
                        encoder_features, patch_h, patch_w
                    )
                    path_1 = dpt_features[-1]  # Extract path_1

                    # Apply Gear2 modulation BEFORE Mamba
                    path_1_modulated, importance_map, fg_features, bg_features, fg_mask, bg_mask = model.gear2_head(
                        cls_tokens_multi_layer, [path_1]
                    )

                    # Apply Mamba temporal modeling to modulated feature
                    C = img_t.shape[1]
                    path_1_temporal = model.dpt_features_to_mamba(
                        input_shape=(B, 1, C, h, w),  # Single frame
                        dpt_features=path_1_modulated,
                        in_dpt_layer=0
                    )

                    # Get depth (at model resolution)
                    out = model.depth_head.scratch.output_conv1(path_1_temporal)
                    out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
                    out = model.depth_head.scratch.output_conv2(out)

                    pred_depth_inverse = out  # [B, 1, h, w] at model resolution

                    # Interpolate prediction to GT resolution (like original FlashDepth)
                    gt_t_shape = gt_t_inverse.shape[-2:]  # GT original resolution
                    if pred_depth_inverse.shape[-2:] != gt_t_shape:
                        pred_depth_inverse = F.interpolate(
                            pred_depth_inverse, size=gt_t_shape, mode="bilinear", align_corners=True
                        )

                    # Compute loss in inverse depth space (100/m)
                    # Validation: Use same threshold (70m) for both GT and Pred for fair evaluation
                    MIN_INVERSE_DEPTH = 100.0 / 70.0  # 70m threshold (consistent with test)

                    # GT valid mask: where GT depth is within valid range
                    gt_valid_mask = (gt_t_inverse_100 >= MIN_INVERSE_DEPTH)

                    # Pred valid mask: same threshold as GT for fair evaluation
                    pred_valid_mask = (pred_depth_inverse >= MIN_INVERSE_DEPTH)

                    # Final mask: GT valid AND pred valid (both use 70m threshold)
                    valid_mask = (gt_valid_mask & pred_valid_mask).float()

                    # Save canonical masks for visualization
                    canonical_gt_valid = gt_valid_mask.cpu()  # [B, 1, H, W]
                    canonical_pred_valid = pred_valid_mask.cpu()  # [B, 1, H, W]

                    # Ensure shapes match: pred [B, 1, H, W], gt [B, 1, H, W], mask [B, 1, H, W]
                    if gt_t_inverse_100.dim() == 3:  # [B, H, W]
                        gt_t_inverse_100 = gt_t_inverse_100.unsqueeze(1)  # [B, 1, H, W]
                    if valid_mask.dim() == 3:  # [B, H, W]
                        valid_mask = valid_mask.unsqueeze(1)  # [B, 1, H, W]

                    if valid_mask.sum() > 0:
                        loss_t = self.loss_fn(pred_depth_inverse, gt_t_inverse_100, valid_mask)
                        frame_losses.append(loss_t.float())  # Convert to Float32 for accumulation

                    # Validation visualization: save multiple sequences per dataset (rank 0 only)
                    # Sintel: 3 samples (sequence 0, 5, 10)
                    # Waymo: 8 samples (sequence 0, 5, 10, 15, 20, 25, 30, 35)
                    if t == 0 and self.val_visualizer and self.rank == 0:
                        # Check if this dataset is in our visualization config
                        if current_dataset in self.val_vis_config:
                            config = self.val_vis_config[current_dataset]
                            seq_num = dataset_sequence_counters[current_dataset]

                            # Check if we should save this sequence
                            # Save only specified sequences (e.g., [0, 4, 7] for sintel, [0,1,2,3,4,5,6,7] for waymo)
                            # We only save if this seq_num hasn't been saved yet in THIS validation run
                            should_save = (
                                seq_num in config['sequences'] and
                                seq_num not in config['saved']
                            )

                            if should_save:
                                try:
                                    # Convert to metric depth for visualization: 100/m -> m
                                    # Convert to Float32 for CPU operations
                                    pred_depth_metric = (100.0 / (pred_depth_inverse.float() + 1e-8)).cpu()
                                    gt_depth_metric = (100.0 / (gt_t_inverse_100.float() + 1e-8)).cpu()

                                    # Get GT resolution for visualization
                                    gt_h, gt_w = gt_t_inverse_100.shape[-2:]

                                    # Resize importance map to GT resolution (None for Gear2)
                                    if importance_map is not None:
                                        importance_map_resized = F.interpolate(
                                            importance_map, size=(gt_h, gt_w), mode='bilinear', align_corners=True
                                        )
                                        importance_map_cpu = importance_map_resized.float().cpu()
                                    else:
                                        importance_map_cpu = None

                                    # Resize images to GT resolution (for visualization consistency)
                                    img_t_resized = F.interpolate(
                                        img_t, size=(gt_h, gt_w), mode='bilinear', align_corners=True
                                    )

                                    model_outputs = {
                                        'pred_depth': pred_depth_metric,  # [B, 1, gt_h, gt_w] at GT resolution
                                        'importance_map': importance_map_cpu,  # [B, 1, gt_h, gt_w] or None
                                        'canonical_gt_valid': canonical_gt_valid,  # [B, 1, H, W] - canonical space mask
                                        'canonical_pred_valid': canonical_pred_valid  # [B, 1, H, W] - canonical space mask
                                    }

                                    # For visualization, we need [B, T, ...] format like training
                                    # But we only have one frame (t=0), so unsqueeze T dimension
                                    sample_batch = (
                                        img_t_resized.unsqueeze(1).float().cpu(),  # [B, 1, C, gt_h, gt_w] at GT resolution
                                        gt_depth_metric.unsqueeze(1).float().cpu(),  # [B, 1, gt_h, gt_w]
                                        dataset_idx,
                                        focal_lengths[:, 0:1].float().cpu()  # [B, 1] - resized focal length for first frame
                                    )

                                    # FPS removed from training (only measured in test_gear2.py)
                                    current_fps = None

                                    # Create loss_dict with current frame loss (for visualization)
                                    val_loss_dict = {
                                        'val_loss': loss_t.item() if len(frame_losses) > 0 else 0.0
                                    }

                                    # Save with dataset and sequence-specific name
                                    save_name = f"validation_{current_dataset}_seq{seq_num:03d}_step_{self.global_step:06d}"
                                    self.val_visualizer.create_validation_summary(
                                        sample_batch, model_outputs, self.global_step,
                                        save_name=save_name, fps=current_fps, loss_dict=val_loss_dict, dataset_name=current_dataset, config=self.config
                                    )
                                    config['saved'].append(seq_num)
                                    self.logger.info(f"Saved validation visualization: {current_dataset} sequence {seq_num} ({len(config['saved'])}/{len(config['sequences'])})")
                                except Exception as e:
                                    if self.rank == 0:
                                        self.logger.warning(f"Failed to save validation visualization for {current_dataset} seq {seq_num}: {e}")

                    # Clear intermediate tensors to free memory after each frame
                    del encoder_features, cls_tokens_multi_layer, dpt_features, path_1
                    del path_1_modulated, importance_map, fg_features, bg_features, fg_mask, bg_mask, pred_depth_inverse
                    # Always clear cache after each frame to prevent OOM during validation
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

            # WARNING: Check if validation set is too small (Phase 2)
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


@hydra.main(version_base=None, config_path="configs/gear2", config_name="config")
def main(config: DictConfig):
    """Main entry point"""
    # Initialize distributed training
    rank, world_size, local_rank = init_distributed()

    # Create trainer
    trainer = Gear2Trainer(config, rank, world_size, local_rank)
    trainer.train()

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
