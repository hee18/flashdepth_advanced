"""
Onepiece V3 FSDP2 Training: Hybrid 2K High-Resolution

Key changes from train_onepiece.py (DDP):
  - DDP → fully_shard (FSDP2 composable, torch.distributed._composable.fsdp)
  - autocast → MixedPrecisionPolicy (param/output bf16, reduce fp32)
  - checkpoint_wrapper → NO_REENTRANT mode (required for FSDP2 composable)
  - save/load → get/set_model_state_dict with full_state_dict=True (FSDP-aware)
  - validate / training_vis: ALL ranks participate (FSDP all-gather requires this)
  - _get_model(): returns self.model directly (FSDP2 composable does not wrap)

Original DDP script: train_onepiece.py (untouched)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from functools import partial
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
from dataloaders.combined_dataset import CombinedDataset
from utils.helpers import *
from utils.metric_depth_metrics import MetricDepthMetrics
from utils.onepiece_losses import OnepieceCombinedLoss
from utils.flow_estimator import FlowEstimator
from utils.onepiece_visualization import OnepieceVisualizer


def init_distributed():
    """Initialize distributed training."""
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
            world_size=world_size,
        )
        dist.barrier()

    return rank, world_size, local_rank


# FSDP2 requires NO_REENTRANT activation checkpointing to avoid hook conflicts.
_ac_wrapper = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)


class OnepieceTrainer:
    """
    Onepiece V3 trainer using FSDP2 (fully_shard composable API).

    Phase 1 (0 → auto_transition_step):
        Trainable: CLSMetricHead + CLS projection only.
    Phase 2 (auto_transition_step+):
        Trainable: SpatialMamba + CLSMetricHead + CLS projection + DPT + HybridFusion.

    FSDP wrapping order (inner → outer):
        student ViT blocks → teacher ViT blocks → DPT refinenets
        → SpatialMamba → HybridFusion → model root
    """

    def __init__(self, config, rank, world_size, local_rank):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank

        self.auto_transition_step = config.phase.get('auto_transition_step', 1500)
        self.phase2_warmup_steps = config.phase.get('phase2_warmup_steps', 500)
        self.current_phase = 1

        self.train_mode = config.get('train_mode', 'metric')

        self.device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)

        self.results_dir = Path(config.get('results_dir', './train_results/onepiece_fsdp'))
        if rank == 0:
            self.results_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        if rank == 0:
            self.logger.info("=== ONEPIECE V3 FSDP2 TRAINING ===")
            self.logger.info(f"  World size: {world_size} GPU(s)")
            self.logger.info(f"  Phase transition at step {self.auto_transition_step}")
            self.logger.info(f"  Train mode: {self.train_mode}")

        # DeviceMesh: 1-D fully-shard across all GPUs
        self.mesh = init_device_mesh("cuda", (world_size,)) if world_size > 1 else None

        # FSDP mixed precision: params/outputs bf16, all-reduce in fp32
        self.mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        )

        # FSDP per-module reshard flags from config
        fsdp_cfg = config.get('fsdp', {})
        self.fsdp_reshard_student = fsdp_cfg.get('reshard_after_forward_student', False)
        self.fsdp_reshard_teacher = fsdp_cfg.get('reshard_after_forward_teacher', True)

        self.model = self._setup_model()
        self._configure_parameters_phase1()
        self._set_train_mode()

        self.train_loader, self.val_loader = self._setup_data_loaders()

        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        loss_config = config.get('loss', {})
        self.loss_fn = OnepieceCombinedLoss(
            log_l1_weight=loss_config.get('log_l1_weight', 1.0),
            tgm_weight=loss_config.get('tgm_weight', 1.0),
            ofc_weight=loss_config.get('ofc_weight', loss_config.get('feat_cons_weight', 1.0)),
            use_log_space=loss_config.get('use_log_space', True),
        )

        flow_cfg = config.get('flow', {})
        flow_ckpt = flow_cfg.get('checkpoint',
            'third_party/SEA-RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth')
        try:
            self.flow_estimator = FlowEstimator(checkpoint_path=flow_ckpt, device=self.device)
            if rank == 0:
                self.logger.info(f"Sea-RAFT loaded: {flow_ckpt}")
        except (ImportError, FileNotFoundError) as e:
            raise RuntimeError(f"Sea-RAFT required. Error: {e}")

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

        if config.training.get('wandb', False) and rank == 0:
            wandb.init(
                project="flashdepth-onepiece",
                name=config.training.get('wandb_name', 'onepiece_fsdp'),
                config=dict(config),
            )

        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_step = 0
        self.current_val_loss = None
        self.dataset_losses = None
        self.num_sequences = None

        self.val_vis_config = {
            'sintel': {'sequences': [0, 4, 7], 'saved': []},
            'waymo_seg': {'sequences': [0, 1, 2, 3, 4, 5, 6, 7], 'saved': []},
        }

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _setup_logging(self):
        class FlushFileHandler(logging.FileHandler):
            def emit(self, record):
                super().emit(record)
                self.flush()

        if self.rank == 0:
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.INFO)
            root_logger.handlers.clear()
            fh = FlushFileHandler(self.results_dir / 'training.log', mode='a')
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            sh = logging.StreamHandler()
            sh.setLevel(logging.INFO)
            sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            root_logger.addHandler(fh)
            root_logger.addHandler(sh)
        else:
            logging.basicConfig(level=logging.ERROR)
        self.logger = logging.getLogger(__name__)

    # -------------------------------------------------------------------------
    # Model setup: load checkpoint → AC (NO_REENTRANT) → FSDP2 wrap
    # -------------------------------------------------------------------------

    def _setup_model(self):
        """Build model, load checkpoint, apply AC, apply FSDP2 sharding."""
        model_config = dict(self.config.model)
        model_config['batch_size'] = self.config.training.batch_size
        model_config['use_metric_head'] = False
        model_config['use_onepiece'] = True
        model_config['spatial_mamba_layers'] = self.config.model.get('spatial_mamba_layers', 4)
        model_config['spatial_mamba_d_state'] = self.config.model.get('spatial_mamba_d_state', 256)
        model_config['spatial_mamba_d_conv'] = self.config.model.get('spatial_mamba_d_conv', 4)
        model_config['spatial_mamba_downsample'] = self.config.model.get('spatial_mamba_downsample', 0.1)
        model_config['onepiece_train_mode'] = self.train_mode
        sc = self.config.get('scene_cut', {})
        model_config['scene_cut_tau'] = sc.get('tau', 0.05)
        model_config['scene_cut_k'] = sc.get('k', 80)
        model_config['hybrid_configs'] = self.config.get('hybrid_configs', None)

        model = FlashDepth(**model_config)

        # --- Load checkpoint BEFORE FSDP sharding (plain state dict load) ---
        ckpt_path = self.config.get('load')
        resume_path = self.config.get('resume')  # FSDP checkpoint to resume from

        if resume_path and os.path.exists(resume_path):
            # Resume from a previously saved FSDP checkpoint (fp32 full state dict)
            if self.rank == 0:
                self.logger.info(f"Resuming from checkpoint: {resume_path}")
            ckpt = torch.load(resume_path, map_location='cpu')
            model.load_state_dict(ckpt['model'])  # full state dict saved by get_model_state_dict
            self._resume_meta = ckpt  # store for optimizer/scheduler restore after init
        elif ckpt_path and os.path.exists(ckpt_path):
            # Initial FlashDepth / Onepiece-S pretrained checkpoint (student weights)
            if self.rank == 0:
                self.logger.info(f"Loading student checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location='cpu')
            if isinstance(ckpt, dict) and 'model' in ckpt:
                state_dict = ckpt['model']
            elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            # Exclude only keys whose shape differs between non-hybrid and hybrid.
            # spatial_mamba / onepiece_metric_head: identical architecture in S and Hybrid → load them.
            # cls_projection: Linear(384→64) in S vs Linear(1024→64) in Hybrid → shape mismatch, exclude.
            # unified_global_mamba: legacy module not present in current model → exclude.
            loaded, excluded = {}, []
            for k, v in state_dict.items():
                if any(x in k for x in ['cls_projection', 'unified_global_mamba']):
                    excluded.append(k)
                else:
                    loaded[k] = v
            model.load_state_dict(loaded, strict=False)
            if self.rank == 0:
                self.logger.info(f"Loaded {len(loaded)} params, excluded {len(excluded)} keys "
                                 f"(cls_projection shape mismatch / legacy)")
        elif self.rank == 0:
            self.logger.warning(f"No checkpoint found: {ckpt_path}")

        # --- Optional: load teacher weights from a separate Onepiece-L / FlashDepth-L checkpoint ---
        # Supports two checkpoint formats:
        #   (A) Onepiece-L checkpoint: keys are pretrained.*, depth_head.*  (no teacher_model. prefix)
        #   (B) FlashDepth-L checkpoint: same key structure as (A)
        # In both cases we remap  pretrained.* → teacher_model.pretrained.*
        #                         depth_head.* → teacher_model.depth_head.*
        teacher_path = self.config.get('load_teacher')
        if teacher_path and os.path.exists(teacher_path) and hasattr(model, 'teacher_model'):
            if self.rank == 0:
                self.logger.info(f"Loading teacher checkpoint: {teacher_path}")
            t_ckpt = torch.load(teacher_path, map_location='cpu')
            if isinstance(t_ckpt, dict) and 'model' in t_ckpt:
                t_sd = t_ckpt['model']
            elif isinstance(t_ckpt, dict) and 'state_dict' in t_ckpt:
                t_sd = t_ckpt['state_dict']
            else:
                t_sd = t_ckpt
            t_sd = {k.replace('module.', ''): v for k, v in t_sd.items()}

            # Remap student-style keys into teacher_model namespace
            t_remapped = {}
            for k, v in t_sd.items():
                if k.startswith('pretrained.') or k.startswith('depth_head.'):
                    t_remapped[k] = v  # loaded directly into model.teacher_model submodule

            missing, unexpected = model.teacher_model.load_state_dict(t_remapped, strict=False)
            if self.rank == 0:
                self.logger.info(
                    f"Teacher loaded: {len(t_remapped)} keys, "
                    f"missing={len(missing)}, unexpected={len(unexpected)}"
                )
            for param in model.teacher_model.parameters():
                param.requires_grad = False
        elif teacher_path and not os.path.exists(teacher_path) and self.rank == 0:
            self.logger.warning(f"load_teacher path not found: {teacher_path}")

        model = model.to(self.device)

        # --- Step 1: Activation checkpointing (NO_REENTRANT, must come before FSDP2 wrap) ---
        # IMPORTANT: wrap only at the same granularity as FSDP2 sharding.
        # check_fn=lambda _: True would also wrap patch_embed.proj (Conv2d) and other non-block
        # submodules. After FSDP2 root-shards them, their weights become DTensors, but AC
        # re-runs forward with a plain Tensor input → "got mixed Tensor and DTensor" crash.
        if self.config.training.get('gradient_checkpointing', False):
            if self.rank == 0:
                self.logger.info("Applying NO_REENTRANT activation checkpointing (block-level only)")
            # Student ViT blocks only — same units individually sharded by FSDP2 below
            ViTBlockType = type(model.pretrained.blocks[0])
            apply_activation_checkpointing(
                model.pretrained, checkpoint_wrapper_fn=_ac_wrapper,
                check_fn=lambda m: isinstance(m, ViTBlockType),
            )
            # DPT refinenets only — same units individually sharded by FSDP2 below
            DPTRefinenetType = type(model.depth_head.scratch.refinenet1)
            apply_activation_checkpointing(
                model.depth_head, checkpoint_wrapper_fn=_ac_wrapper,
                check_fn=lambda m: isinstance(m, DPTRefinenetType),
            )
            # Teacher is frozen (no_grad) → AC saves no memory, skip it entirely.
            # hybrid_fusion is sharded as a single unit → no sub-module AC needed.

        # --- Step 2: FSDP2 sharding (inner → outer) ---
        if self.world_size > 1:
            fkw = dict(mesh=self.mesh, mp_policy=self.mp_policy)

            # Student ViT-S blocks
            for block in model.pretrained.blocks:
                fully_shard(block, reshard_after_forward=self.fsdp_reshard_student, **fkw)

            # Teacher ViT-L blocks (frozen, reshard immediately to minimize peak memory)
            if hasattr(model, 'teacher_model'):
                for block in model.teacher_model.pretrained.blocks:
                    fully_shard(block, reshard_after_forward=self.fsdp_reshard_teacher, **fkw)

            # DPT refinenet blocks
            for i in range(1, 5):
                fully_shard(getattr(model.depth_head.scratch, f'refinenet{i}'), **fkw)

            # SpatialMamba: keep unsharded between frame iterations (reshard_after_forward=False)
            fully_shard(model.spatial_mamba, reshard_after_forward=False, **fkw)

            # HybridFusion
            if hasattr(model, 'hybrid_fusion'):
                fully_shard(model.hybrid_fusion, **fkw)

            # Root model wrap (covers all remaining params not wrapped above)
            fully_shard(model, **fkw)

            if self.rank == 0:
                self.logger.info(f"FSDP2 fully_shard applied across {self.world_size} GPUs")

        # Route model.__call__ through forward_with_onepiece so FSDP2 pre/post_forward
        # hooks (registered on __call__) fire correctly.
        # Calling model.forward_with_onepiece(...) directly bypasses nn.Module.__call__
        # → FSDP2 pre_forward hook never runs → root-sharded params (e.g. patch_embed.proj.weight)
        # remain as DTensors while inputs are plain Tensors → conv crash.
        _fwop = FlashDepth.forward_with_onepiece
        model.forward = lambda *a, **kw: _fwop(model, *a, **kw)

        return model

    def _get_model(self):
        """FSDP2 composable: model is not wrapped in a container."""
        return self.model

    # -------------------------------------------------------------------------
    # Parameter phase configuration (identical to DDP version)
    # FSDP2 is transparent to requires_grad changes.
    # -------------------------------------------------------------------------

    def _configure_parameters_phase1(self):
        """Phase 1: CLSMetricHead + CLS projection trainable only."""
        model = self._get_model()
        frozen = trainable = 0
        for name, param in model.named_parameters():
            if 'onepiece_metric_head' in name or 'cls_projection' in name:
                param.requires_grad = True
                trainable += param.numel()
            else:
                param.requires_grad = False
                frozen += param.numel()
        if self.rank == 0:
            self.logger.info("=== Phase 1 Parameters ===")
            self.logger.info(f"  Frozen:    {frozen:,}")
            self.logger.info(f"  Trainable: {trainable:,} (CLSMetricHead + CLS proj)")

    def _configure_parameters_phase2(self):
        """Phase 2: SpatialMamba + CLSMetricHead + DPT + HybridFusion trainable."""
        model = self._get_model()
        frozen = t_onepiece = t_dpt = t_output_conv = t_fusion = 0
        for name, param in model.named_parameters():
            if 'spatial_mamba' in name or 'onepiece_metric_head' in name or 'cls_projection' in name:
                param.requires_grad = True
                t_onepiece += param.numel()
            elif 'hybrid_fusion' in name:
                param.requires_grad = True
                t_fusion += param.numel()
            elif 'depth_head' in name and 'output_conv' not in name:
                param.requires_grad = True
                t_dpt += param.numel()
            elif 'output_conv' in name:
                param.requires_grad = True
                t_output_conv += param.numel()
            elif 'pretrained' in name or 'teacher_model' in name:
                param.requires_grad = False
                frozen += param.numel()
            else:
                param.requires_grad = False
                frozen += param.numel()
        if self.rank == 0:
            self.logger.info("=== Phase 2 Parameters ===")
            self.logger.info(f"  Frozen:          {frozen:,}")
            self.logger.info(f"  Onepiece:        {t_onepiece:,}")
            self.logger.info(f"  DPT:             {t_dpt:,}")
            self.logger.info(f"  output_conv:     {t_output_conv:,}")
            if t_fusion > 0:
                self.logger.info(f"  HybridFusion:    {t_fusion:,}")
            self.logger.info(f"  Total trainable: {t_onepiece + t_dpt + t_output_conv + t_fusion:,}")

    def _set_train_mode(self):
        """Trainable modules → train(); frozen modules → eval()."""
        self.model.train()
        model = self._get_model()
        for name, module in model.named_modules():
            if name == '':
                continue
            if any(kw in name for kw in ['spatial_mamba', 'onepiece_metric_head',
                                          'scene_cut_detector', 'cls_projection']):
                continue
            if self.current_phase >= 2 and any(kw in name for kw in [
                'depth_head', 'output_conv', 'hybrid_fusion'
            ]):
                continue
            module.eval()

    # -------------------------------------------------------------------------
    # Data loaders
    # -------------------------------------------------------------------------

    def _setup_data_loaders(self):
        train_dsets = self.config.dataset.get(
            'train_datasets', ['mvs-synth', 'dynamicreplica', 'tartanair', 'pointodyssey', 'spring'])
        val_dsets = self.config.dataset.get('val_datasets', ['sintel', 'waymo_seg'])
        resolution = self.config.dataset.get('resolution', 'base')
        vlen = self.config.dataset.get('video_length', 8)

        if self.rank == 0:
            self.logger.info(f"Train: {train_dsets}, Val: {val_dsets}")
            self.logger.info(f"Resolution: {resolution}, video_length: {vlen}")

        train_ds = CombinedDataset(
            root_dir=self.config.dataset.data_root,
            enable_dataset_flags=train_dsets,
            resolution=resolution,
            split='train',
            video_length=vlen,
            color_aug=False,
        )
        val_ds = CombinedDataset(
            root_dir=self.config.dataset.data_root,
            enable_dataset_flags=val_dsets,
            resolution=resolution,
            split='val',
            video_length=vlen,
        )

        train_sampler = DistributedSampler(
            train_ds, num_replicas=self.world_size,
            rank=self.rank, shuffle=True, drop_last=True,
        ) if self.world_size > 1 else None

        # Val sampler: None → all ranks see same data (required for FSDP all-gather sync)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.training.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=self.config.training.workers,
            pin_memory=True, drop_last=True,
            collate_fn=self._collate_fn,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1, sampler=None, shuffle=False,
            num_workers=self.config.training.workers,
            pin_memory=True, drop_last=False,
            collate_fn=self._collate_fn,
        )
        if self.rank == 0:
            self.logger.info(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
        return train_loader, val_loader

    def _collate_fn(self, batch):
        batch = [item for item in batch if item is not None]
        if not batch:
            return None
        return torch.utils.data.dataloader.default_collate(batch)

    # -------------------------------------------------------------------------
    # Optimizer / Scheduler (identical to DDP version)
    # -------------------------------------------------------------------------

    def _setup_optimizer(self):
        model = self._get_model()
        lr_cfg = self.config.training.get('lr', {})
        base_lr = lr_cfg.get('onepiece', 1e-4)

        if self.current_phase == 1:
            params = [p for p in model.parameters() if p.requires_grad]
            param_groups = [{'params': params, 'lr': base_lr, 'name': 'onepiece'}]
        else:
            dpt_lr = lr_cfg.get('dpt', base_lr / 10)
            fusion_lr = lr_cfg.get('fusion', base_lr)
            onepiece_p, dpt_p, output_conv_p, fusion_p = [], [], [], []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if 'spatial_mamba' in name or 'onepiece_metric_head' in name or 'cls_projection' in name:
                    onepiece_p.append(param)
                elif 'hybrid_fusion' in name:
                    fusion_p.append(param)
                elif 'output_conv' in name:
                    output_conv_p.append(param)
                elif 'depth_head' in name:
                    dpt_p.append(param)
            param_groups = [
                {'params': onepiece_p,    'lr': base_lr,  'name': 'onepiece'},
                {'params': dpt_p,         'lr': dpt_lr,   'name': 'dpt'},
                {'params': output_conv_p, 'lr': dpt_lr,   'name': 'output_conv'},
            ]
            if fusion_p:
                param_groups.append({'params': fusion_p, 'lr': fusion_lr, 'name': 'fusion'})

        optimizer = torch.optim.AdamW(
            param_groups, betas=[0.9, 0.95],
            weight_decay=self.config.training.get('weight_decay', 1e-6),
        )
        if self.rank == 0:
            for pg in param_groups:
                n = sum(p.numel() for p in pg['params'])
                self.logger.info(f"  Optimizer group '{pg['name']}': {n:,} params, LR={pg['lr']:.2e}")
        return optimizer

    def _setup_scheduler(self):
        total_steps = self.config.training.get('iterations', self.config.training.total_iters)
        warmup = self.config.training.lr.get('warmup_steps', 500)
        decay_start = int(total_steps * 0.3)

        if self.current_phase == 1:
            def lr_lambda(step):
                if step < warmup:
                    return 0.1 + 0.9 * (step / warmup)
                elif step < decay_start:
                    return 1.0
                else:
                    p = (step - decay_start) / (total_steps - decay_start)
                    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        else:
            t = self.auto_transition_step

            def lr_onepiece(step):
                if step < warmup:
                    return 0.1 + 0.9 * (step / warmup)
                elif step < decay_start:
                    return 1.0
                else:
                    p = (step - decay_start) / (total_steps - decay_start)
                    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))

            def lr_new(step):
                since = step - t
                if since < 0:
                    return 0.0
                elif since < self.phase2_warmup_steps:
                    return since / self.phase2_warmup_steps
                elif step < decay_start:
                    return 1.0
                else:
                    p = (step - decay_start) / (total_steps - decay_start)
                    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))

            lambdas = [lr_onepiece] + [lr_new] * (len(self.optimizer.param_groups) - 1)
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambdas)

    def _transition_to_phase2(self):
        if self.rank == 0:
            self.logger.info("=" * 60)
            self.logger.info("=== PHASE TRANSITION: Phase 1 → Phase 2 ===")
            self.logger.info("=" * 60)
        self.current_phase = 2
        self._configure_parameters_phase2()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self._set_train_mode()
        if self.rank == 0:
            self.logger.info(f"Phase transition complete at step {self.global_step}")

    # -------------------------------------------------------------------------
    # Checkpoint I/O (FSDP2-aware)
    # -------------------------------------------------------------------------

    def save_checkpoint(self, filename):
        """
        Gather full fp32 state dict from FSDP shards (rank 0 only).
        Output format is compatible with train_onepiece.py load_state_dict.
        """
        opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        model_sd = get_model_state_dict(self.model, options=opts)
        optim_sd = get_optimizer_state_dict(self.model, self.optimizer, options=opts)

        if self.rank == 0:
            checkpoint_path = self.results_dir / filename
            torch.save({
                'global_step': self.global_step,
                'model': model_sd,
                'optimizer': optim_sd,
                'scheduler': self.scheduler.state_dict(),
                'best_val_loss': self.best_val_loss,
                'best_step': self.best_step,
                'current_val_loss': self.current_val_loss,
                'dataset_losses': self.dataset_losses,
                'num_sequences': self.num_sequences,
                'config': OmegaConf.to_container(self.config, resolve=True),
                'current_phase': self.current_phase,
            }, checkpoint_path)
            self.logger.info(f"Saved FSDP2 checkpoint: {checkpoint_path}")

    # -------------------------------------------------------------------------
    # Training step
    # No autocast wrapper — FSDP MixedPrecisionPolicy handles bf16 casting.
    # -------------------------------------------------------------------------

    def train_step(self, batch, step):
        if batch is None:
            return None

        images, gt_depth, focal_lengths_canonical, focal_lengths_actual, \
            actual_valid_masks, fx_ratio, resize_ratio, dataset_idx = batch

        images = images.to(self.device)
        gt_depth = gt_depth.to(self.device)
        actual_valid_masks = actual_valid_masks.to(self.device)

        if gt_depth.ndim == 3:
            gt_depth = gt_depth.unsqueeze(1)
        elif gt_depth.ndim == 4 and gt_depth.shape[1] != 1:
            gt_depth = gt_depth.unsqueeze(2)

        B, T = images.shape[:2]

        # model((images,), phase=...) goes through __call__ → FSDP2 pre/post hooks fire
        outputs = self.model((images,), phase=self.current_phase)

        # Cast outputs to fp32 for loss computation
        metric_depth = outputs['metric_depth'].float()
        scale = outputs['scale'].float()
        shift = outputs['shift'].float()
        post_mamba_features = outputs['post_mamba_features'].float()

        # Loss in fp32
        gt_depth_meters = 1.0 / (gt_depth.squeeze(2).clamp(min=1e-8))

        if metric_depth.shape[-2:] != gt_depth_meters.shape[-2:]:
            metric_depth = F.interpolate(
                metric_depth.view(B * T, 1, *metric_depth.shape[-2:]),
                size=gt_depth_meters.shape[-2:],
                mode='bilinear', align_corners=True,
            ).squeeze(1).view(B, T, *gt_depth_meters.shape[-2:])

        gt_valid = (gt_depth.squeeze(2) > 0)
        pred_valid = (metric_depth > 0) & (metric_depth < 1000.0)
        if actual_valid_masks.ndim == 3:
            actual_valid_masks = actual_valid_masks.unsqueeze(1)
        valid_mask = gt_valid & pred_valid & actual_valid_masks

        pred_inverse = 100.0 / metric_depth.clamp(min=1e-8)
        gt_inverse = gt_depth.squeeze(2).float() * 100.0

        if self.current_phase == 1:
            total_loss, loss_components = self.loss_fn(
                pred_depth=pred_inverse, gt_depth=gt_inverse,
                valid_mask=valid_mask.float(),
                post_mamba_features=None, images=None, flow_estimator=None,
                return_components=True,
            )
        else:
            total_loss, loss_components = self.loss_fn(
                pred_depth=pred_inverse, gt_depth=gt_inverse,
                valid_mask=valid_mask.float(),
                post_mamba_features=post_mamba_features,
                images=images.float(),
                flow_estimator=self.flow_estimator,
                return_components=True,
            )

        self.optimizer.zero_grad()
        total_loss.backward()
        # clip_grad_norm_ operates on local shards; suitable for stable training
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            'loss': total_loss.item(),
            'mean_scale': scale.mean().item(),
            'mean_shift': shift.mean().item(),
            **loss_components,
        }

    # -------------------------------------------------------------------------
    # Training visualization
    # ALL ranks must call this (FSDP forward requires all ranks).
    # Only rank 0 saves the output.
    # -------------------------------------------------------------------------

    def _save_training_visualization(self, batch, loss_dict, step):
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

            with torch.no_grad():
                outputs = model((images,), phase=self.current_phase)

            if self.rank == 0 and self.train_visualizer:
                metric_depth = outputs['metric_depth'].float()
                scale = outputs['scale'].float()
                shift = outputs['shift'].float()

                gt_depth_inverse_100 = gt_depth * 100.0
                gt_depth_metric = 100.0 / (gt_depth_inverse_100.squeeze(2) + 1e-8)
                MIN_INVERSE_DEPTH = 100.0 / 70.0
                canonical_gt_valid = (gt_depth_inverse_100.squeeze(2) > MIN_INVERSE_DEPTH)
                pred_inverse = 100.0 / (metric_depth + 1e-8)
                canonical_pred_valid = (pred_inverse > 100.0 / 200.0)

                sample_batch = (
                    images[:1, :1].float().cpu(),
                    gt_depth_metric[:1, :1].float().cpu(),
                    dataset_idx,
                    fx_ratio[:1, :1].float().cpu(),
                    resize_ratio[:1, :1].float().cpu(),
                )
                model_outputs_cpu = {
                    'pred_depth': metric_depth[:1, :1].float().cpu(),
                    'canonical_gt_valid': canonical_gt_valid[:1, :1].cpu(),
                    'canonical_pred_valid': canonical_pred_valid[:1, :1].cpu(),
                    'scale': scale[:1, :1].cpu(),
                    'shift': shift[:1, :1].cpu(),
                }
                self.train_visualizer.create_validation_summary(
                    sample_batch, model_outputs_cpu, step,
                    prefix="training", loss_dict=loss_dict, config=self.config,
                )

            self._set_train_mode()

        except Exception as e:
            import traceback
            if self.rank == 0:
                self.logger.error(f"Training vis failed: {e}\n{traceback.format_exc()}")
            self._set_train_mode()

    # -------------------------------------------------------------------------
    # Validation
    # ALL ranks participate (FSDP all-gather requires synchronization).
    # Only rank 0 logs/visualizes.
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def validate(self):
        torch.cuda.empty_cache()
        self.model.eval()
        model = self._get_model()

        total_loss = total_depth_loss = total_tgm_loss = 0.0
        num_batches = 0
        dataset_losses: dict = {}
        dataset_depth_losses: dict = {}
        dataset_tgm_losses: dict = {}

        for ds_config in self.val_vis_config.values():
            ds_config['saved'] = []

        dataset_seq_counters: dict = {'sintel': 0, 'waymo_seg': 0}

        if self.current_phase >= 2:
            max_val_batches = 16
            dataset_max_seqs = {'sintel': 8, 'waymo_seg': 8}
        else:
            max_val_batches = None
            dataset_max_seqs = {}

        total_processed = 0

        # All ranks iterate the same val_loader (no DistributedSampler for val)
        for batch_idx, batch in enumerate(self.val_loader):
            if batch is None:
                continue
            if max_val_batches is not None and total_processed >= max_val_batches:
                break

            images, gt_depth, focal_lengths_canonical, focal_lengths_actual, \
                actual_valid_masks, fx_ratio, resize_ratio, dataset_idx = batch

            if isinstance(dataset_idx, (list, tuple)):
                current_dataset = str(dataset_idx[0])
            elif torch.is_tensor(dataset_idx):
                current_dataset = str(
                    dataset_idx[0].item() if dataset_idx.dim() > 0 else dataset_idx.item()
                )
            else:
                current_dataset = str(dataset_idx)

            if dataset_max_seqs and current_dataset in dataset_max_seqs:
                if dataset_seq_counters.get(current_dataset, 0) >= dataset_max_seqs[current_dataset]:
                    continue

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

            # All ranks run forward; model.__call__ triggers FSDP2 all-gather hooks
            outputs = model((images,), phase=self.current_phase)

            metric_depth = outputs['metric_depth'].float()
            scale = outputs['scale'].float()
            shift = outputs['shift'].float()

            # Loss (fp32) — rank 0 accumulates
            if self.rank == 0:
                gt_depth_meters = 1.0 / (gt_depth.squeeze(2).float().clamp(min=1e-8))

                if metric_depth.shape[-2:] != gt_depth_meters.shape[-2:]:
                    metric_depth = F.interpolate(
                        metric_depth.view(B * T, 1, *metric_depth.shape[-2:]),
                        size=gt_depth_meters.shape[-2:],
                        mode='bilinear', align_corners=True,
                    ).squeeze(1).view(B, T, *gt_depth_meters.shape[-2:])

                VAL_MAX_DEPTH = 80.0
                gt_valid = (gt_depth.squeeze(2) > 0) & (gt_depth_meters < VAL_MAX_DEPTH)
                pred_valid = (metric_depth > 0) & (metric_depth < VAL_MAX_DEPTH)
                if actual_valid_masks.ndim == 3:
                    actual_valid_masks = actual_valid_masks.unsqueeze(1)
                valid_mask = gt_valid & pred_valid & actual_valid_masks

                avg_loss = avg_depth_loss = avg_tgm_loss = 0.0
                if valid_mask.sum() > 0:
                    pred_inverse = 100.0 / metric_depth.clamp(min=1e-8)
                    gt_inverse = gt_depth.squeeze(2).float() * 100.0

                    val_loss, val_components = self.loss_fn(
                        pred_depth=pred_inverse, gt_depth=gt_inverse,
                        valid_mask=valid_mask.float(),
                        post_mamba_features=None, images=None, flow_estimator=None,
                        return_components=True,
                    )
                    avg_depth_loss = val_components['log_l1_loss']
                    avg_tgm_loss = val_components['tgm_loss']
                    avg_loss = val_loss.item()

                    total_loss += avg_loss
                    total_depth_loss += avg_depth_loss
                    total_tgm_loss += avg_tgm_loss
                    num_batches += 1

                    if current_dataset not in dataset_losses:
                        dataset_losses[current_dataset] = []
                        dataset_depth_losses[current_dataset] = []
                        dataset_tgm_losses[current_dataset] = []
                    dataset_losses[current_dataset].append(avg_loss)
                    dataset_depth_losses[current_dataset].append(avg_depth_loss)
                    dataset_tgm_losses[current_dataset].append(avg_tgm_loss)

                # Visualization (rank 0 only)
                if self.val_visualizer and current_dataset in self.val_vis_config:
                    vis_cfg = self.val_vis_config[current_dataset]
                    seq_idx = dataset_seq_counters.get(current_dataset, 0)
                    if seq_idx in vis_cfg['sequences'] and seq_idx not in vis_cfg['saved']:
                        try:
                            gt_vis = (1.0 / (gt_depth.squeeze(2)[:1, :1].float() + 1e-8)).cpu()
                            pred_vis = metric_depth[:1, :1].float().cpu()
                            MIN_INV = 100.0 / 70.0
                            gt_inv_100 = gt_depth.squeeze(2)[:1, :1] * 100.0
                            can_gt = (gt_inv_100 > MIN_INV)
                            can_pred = (100.0 / (metric_depth[:1, :1] + 1e-8) > 100.0 / 200.0)
                            sample_batch = (
                                images[:1, :1].float().cpu(), gt_vis, dataset_idx,
                                fx_ratio_dev[:1, :1].float().cpu(),
                                resize_ratio_dev[:1, :1].float().cpu(),
                            )
                            model_out = {
                                'pred_depth': pred_vis,
                                'canonical_gt_valid': can_gt.cpu(),
                                'canonical_pred_valid': can_pred.cpu(),
                                'scale': scale[:1, :1].cpu(),
                                'shift': shift[:1, :1].cpu(),
                            }
                            self.val_visualizer.create_validation_summary(
                                sample_batch, model_out, self.global_step,
                                prefix=f"val_{current_dataset}_seq{seq_idx}",
                                loss_dict={'val_loss': avg_loss, 'depth_loss': avg_depth_loss,
                                           'tgm_loss': avg_tgm_loss},
                                dataset_name=current_dataset, config=self.config,
                            )
                            vis_cfg['saved'].append(seq_idx)
                        except Exception as e:
                            self.logger.warning(f"Val vis failed {current_dataset} seq{seq_idx}: {e}")

                if current_dataset in dataset_seq_counters:
                    dataset_seq_counters[current_dataset] += 1
                else:
                    dataset_seq_counters[current_dataset] = 1

            total_processed += 1

            del images, gt_depth, metric_depth
            torch.cuda.empty_cache()

        avg_loss = total_loss / max(num_batches, 1)
        avg_depth_loss = total_depth_loss / max(num_batches, 1)
        avg_tgm_loss = total_tgm_loss / max(num_batches, 1)

        if self.rank == 0 and dataset_losses:
            self.logger.info("  === Per-Dataset Validation ===")
            for ds_name in sorted(dataset_losses.keys()):
                ds_avg = np.mean(dataset_losses[ds_name])
                ds_d = np.mean(dataset_depth_losses[ds_name])
                ds_t = np.mean(dataset_tgm_losses[ds_name])
                ds_n = len(dataset_losses[ds_name])
                self.logger.info(
                    f"    {ds_name}: loss={ds_avg:.4f} (L1={ds_d:.4f}, TGM={ds_t:.4f}) [{ds_n} seqs]"
                )

        return {
            'loss': avg_loss,
            'depth_loss': avg_depth_loss,
            'tgm_loss': avg_tgm_loss,
            'dataset_losses': {k: float(np.mean(v)) for k, v in dataset_losses.items()},
            'num_sequences': {k: len(v) for k, v in dataset_losses.items()},
        }

    # -------------------------------------------------------------------------
    # Main training loop
    # -------------------------------------------------------------------------

    def train(self):
        if self.rank == 0:
            self.logger.info("Starting FSDP2 Onepiece training...")

        train_iterator = iter(self.train_loader)
        total_iters = self.config.training.get('iterations', self.config.training.total_iters)

        pbar = tqdm(range(total_iters), desc="Onepiece FSDP2", disable=(self.rank != 0))

        for step in pbar:
            self.global_step = step

            # Phase transition — FSDP2 handles requires_grad changes transparently
            if self.current_phase == 1 and step == self.auto_transition_step:
                self._transition_to_phase2()

            try:
                batch = next(train_iterator)
            except StopIteration:
                if self.world_size > 1:
                    self.train_loader.sampler.set_epoch(step)
                train_iterator = iter(self.train_loader)
                batch = next(train_iterator)

            if batch is None:
                continue

            loss_dict = self.train_step(batch, step)
            if loss_dict is None:
                continue

            self.scheduler.step()

            if self.rank == 0:
                lr_op = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f'{loss_dict["loss"]:.4f}',
                    'lr': f'{lr_op:.2e}',
                    'phase': self.current_phase,
                })

                if self.config.training.get('wandb', False):
                    wandb_dict = {**loss_dict, 'lr_onepiece': lr_op, 'phase': self.current_phase}
                    if len(self.optimizer.param_groups) > 1:
                        wandb_dict['lr_dpt'] = self.optimizer.param_groups[1]['lr']
                    wandb.log(wandb_dict, step=step)

                if step % 100 == 0 or step < 10:
                    parts = [f"Step {step}",
                             f"loss={loss_dict['loss']:.4f}",
                             f"phase={self.current_phase}"]
                    if 'log_l1_loss' in loss_dict:
                        parts.append(f"L1={loss_dict['log_l1_loss']:.4f}")
                    if 'tgm_loss' in loss_dict:
                        parts.append(f"TGM={loss_dict['tgm_loss']:.4f}")
                    if 'ofc_loss' in loss_dict:
                        parts.append(f"OFC={loss_dict['ofc_loss']:.4f}")
                    self.logger.info(" | ".join(parts))

            # Training visualization: ALL ranks enter (FSDP forward required)
            vis_steps = [0, 10, 50, 100]
            if step in vis_steps or step % 250 == 0:
                self._save_training_visualization(batch, loss_dict, step)

            # Validation: ALL ranks enter (FSDP forward required)
            if step % self.config.training.get('val_freq', 1000) == 0:
                val_metrics = self.validate()

                # validate() only accumulates loss on rank 0; broadcast so all ranks agree.
                val_loss_t = torch.tensor(
                    [val_metrics['loss']], device=self.device, dtype=torch.float32
                )
                if self.world_size > 1:
                    dist.broadcast(val_loss_t, src=0)
                val_loss = val_loss_t.item()

                self.current_val_loss = val_loss
                is_new_best = val_loss < self.best_val_loss
                if is_new_best:
                    self.best_val_loss = val_loss
                    self.best_step = step

                if self.rank == 0:
                    self.logger.info(f"Validation step {step}: loss={val_loss:.4f}")
                    if 'dataset_losses' in val_metrics:
                        self.dataset_losses = val_metrics['dataset_losses']
                        self.num_sequences = val_metrics.get('num_sequences', {})
                        for ds_name, ds_loss in val_metrics['dataset_losses'].items():
                            self.logger.info(f"  {ds_name}: loss={ds_loss:.4f}")
                        if self.config.training.get('wandb', False):
                            for ds_name, ds_loss in val_metrics['dataset_losses'].items():
                                wandb.log({f'val/{ds_name}_loss': ds_loss}, step=step)
                    if self.config.training.get('wandb', False):
                        wandb.log({'val/loss': val_loss}, step=step)
                    if is_new_best:
                        self.logger.info(f"New best at step {step}: val_loss={val_loss:.4f}")

                # save_checkpoint uses get_model_state_dict (collective NCCL op).
                # MUST be called on ALL ranks — never inside an if self.rank == 0: block.
                if is_new_best:
                    self.save_checkpoint('best.pth')

                self._set_train_mode()

            # Periodic checkpoint: save on ALL ranks (FSDP gather), write on rank 0
            if step % self.config.training.get('save_freq', 5000) == 0 and step > 0:
                self.save_checkpoint(f'checkpoint_step{step}.pth')

        # Final save
        self.save_checkpoint('last.pth')
        if self.rank == 0:
            self.logger.info("FSDP2 Onepiece training completed!")


@hydra.main(version_base=None, config_path="configs/onepiece", config_name="config_hybrid_fsdp")
def main(config: DictConfig):
    rank, world_size, local_rank = init_distributed()
    trainer = OnepieceTrainer(config, rank, world_size, local_rank)
    trainer.train()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
