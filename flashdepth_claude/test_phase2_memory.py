"""
Phase 2 Comprehensive Test Script for Onepiece Training.

Skips Phase 1 entirely and directly tests all Phase 2 components:

  [Test 1] Parameter Configuration
      - Correct params frozen/unfrozen
      - FiLMGenerator + RelativeHead (output_conv) trainable
      - DINOv2 + DPT frozen

  [Test 2] Forward Pass
      - Phase 2 forward outputs correct shapes
      - relative_depth_isolated is not None (Phase 2 specific)
      - modulated_features available

  [Test 3] Loss Computation
      - All 4 losses computed: LogL1, TGM, WFC, SSIL
      - WFC > 0 (Sea-RAFT flow working)
      - SSIL > 0 (Scale-Shift-Invariant loss working)

  [Test 4] Backward + Gradient Flow
      - backward() succeeds without OOM
      - Gradients flow to correct parameters only
      - Frozen params have no gradients

  [Test 5] Optimizer Step
      - Phase 2 optimizer with 2 param groups
      - Parameters actually update after step

  [Test 6] Memory Report
      - Peak memory at each stage
      - Headroom analysis

Usage (Docker):
    CUDA_VISIBLE_DEVICES=1 docker compose run --rm \\
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        flashdepth python test_phase2_memory.py \\
        --config-path configs/onepiece \\
        --config-name config_l \\
        dataset.data_root=/data/datasets \\
        training.batch_size=8

    # Test with smaller batch
    CUDA_VISIBLE_DEVICES=1 docker compose run --rm \\
        flashdepth python test_phase2_memory.py \\
        --config-path configs/onepiece \\
        --config-name config_l \\
        dataset.data_root=/data/datasets \\
        training.batch_size=4
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import time
from pathlib import Path
from einops import rearrange

from flashdepth.model import FlashDepth
from utils.onepiece_losses import OnepieceCombinedLoss
from utils.flow_estimator import FlowEstimator

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fmt_mem(bytes_val):
    """Format bytes to GiB string."""
    return f"{bytes_val / (1024 ** 3):.2f} GiB"


def log_memory(label):
    """Log current GPU memory stats."""
    alloc = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    logger.info(f"    [{label}] allocated={fmt_mem(alloc)}, peak={fmt_mem(peak)}")


class Phase2Tester:
    """Comprehensive Phase 2 tester."""

    def __init__(self, config):
        self.config = config
        self.device = 'cuda:0'
        torch.cuda.set_device(0)

        self.batch_size = config.training.batch_size
        self.video_length = config.dataset.get('video_length', 8)
        self.resolution = 518
        self.passed = []
        self.failed = []

    def _print_header(self):
        logger.info("=" * 70)
        logger.info("  ONEPIECE PHASE 2 COMPREHENSIVE TEST")
        logger.info("=" * 70)
        logger.info(f"  Batch size:    {self.batch_size}")
        logger.info(f"  Video length:  {self.video_length}")
        logger.info(f"  Resolution:    {self.resolution}x{self.resolution}")
        logger.info(f"  GPU:           {torch.cuda.get_device_name(0)}")
        logger.info(f"  GPU memory:    {fmt_mem(torch.cuda.get_device_properties(0).total_memory)}")
        logger.info("=" * 70)

    def _build_model(self):
        """Build model and load checkpoint."""
        logger.info("\n[Setup] Building model...")
        model_config = dict(self.config.model)
        model_config['batch_size'] = self.batch_size
        model_config['use_metric_head'] = False
        model_config['use_onepiece'] = True

        model = FlashDepth(**model_config)

        checkpoint_path = self.config.get('load')
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"  Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            loaded_dict = {k: v for k, v in state_dict.items()
                           if not any(x in k for x in ['unified_global_mamba', 'onepiece_metric_head', 'onepiece_film_generator'])}
            model.load_state_dict(loaded_dict, strict=False)
            logger.info(f"  Loaded {len(loaded_dict)} parameters")
        else:
            logger.warning(f"  Checkpoint not found: {checkpoint_path}, using random weights")

        model = model.to(self.device)

        if self.config.training.get('gradient_checkpointing', False):
            logger.info("  Applying gradient checkpointing to ViT and DPT")
            apply_activation_checkpointing(
                model.pretrained, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=lambda _: True
            )
            apply_activation_checkpointing(
                model.depth_head, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=lambda _: True
            )

        return model

    def _build_flow_estimator(self):
        """Load Sea-RAFT flow estimator."""
        flow_config = self.config.get('flow', {})
        flow_checkpoint = flow_config.get('checkpoint',
            'third_party/SEA-RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth')
        flow_estimator = FlowEstimator(checkpoint_path=flow_checkpoint, device=self.device)
        logger.info(f"  Sea-RAFT loaded from {flow_checkpoint}")
        return flow_estimator

    def _create_dummy_data(self):
        """Create dummy training data."""
        B, T, H, W = self.batch_size, self.video_length, self.resolution, self.resolution
        images = torch.randn(B, T, 3, H, W, device=self.device)
        gt_depth = torch.rand(B, T, 1, H, W, device=self.device) * 0.1 + 0.001
        return images, gt_depth

    def _record(self, test_name, passed, msg=""):
        if passed:
            self.passed.append(test_name)
            logger.info(f"  >> PASS: {test_name}" + (f" ({msg})" if msg else ""))
        else:
            self.failed.append(test_name)
            logger.error(f"  >> FAIL: {test_name}" + (f" ({msg})" if msg else ""))

    # =========================================================================
    # Test 1: Parameter Configuration
    # =========================================================================
    def test_parameter_config(self, model):
        logger.info("\n" + "=" * 70)
        logger.info("[Test 1] Parameter Configuration (Phase 2)")
        logger.info("=" * 70)

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
                param.requires_grad = True
                trainable_rel_head += param.numel()
            else:
                param.requires_grad = False
                frozen_count += param.numel()

        total_trainable = trainable_mamba + trainable_metric_head + trainable_film + trainable_rel_head
        logger.info(f"  Frozen:             {frozen_count:>12,}  (ViT + DPT)")
        logger.info(f"  Trainable Mamba:    {trainable_mamba:>12,}")
        logger.info(f"  Trainable MetricH:  {trainable_metric_head:>12,}")
        logger.info(f"  Trainable FiLM:     {trainable_film:>12,}")
        logger.info(f"  Trainable RelHead:  {trainable_rel_head:>12,}")
        logger.info(f"  Total trainable:    {total_trainable:>12,}")

        # Checks
        self._record("Mamba trainable", trainable_mamba > 0, f"{trainable_mamba:,}")
        self._record("MetricHead trainable", trainable_metric_head > 0, f"{trainable_metric_head:,}")
        self._record("FiLM trainable (Phase 2)", trainable_film > 0, f"{trainable_film:,}")
        self._record("RelHead trainable (Phase 2)", trainable_rel_head > 0, f"{trainable_rel_head:,}")
        self._record("ViT+DPT frozen", frozen_count > 100_000_000, f"{frozen_count:,}")

        # Verify specific frozen params
        # Note: output_conv lives under depth_head.scratch but is trainable in Phase 2
        for name, param in model.named_parameters():
            if 'pretrained' in name and param.requires_grad:
                self._record("DINOv2 should be frozen", False, f"{name} has requires_grad=True")
                return
            if 'depth_head' in name and 'output_conv' not in name and param.requires_grad:
                self._record("DPT should be frozen", False, f"{name} has requires_grad=True")
                return
        self._record("DINOv2 all frozen", True)
        self._record("DPT all frozen", True)

        # Set train/eval modes
        model.train()
        for name, module in model.named_modules():
            if name == '':
                continue
            if any(kw in name for kw in ['unified_global_mamba', 'onepiece_metric_head', 'onepiece_film_generator']):
                continue
            if 'output_conv' in name:
                continue
            module.eval()

    # =========================================================================
    # Test 2: Forward Pass
    # =========================================================================
    def test_forward(self, model, images):
        logger.info("\n" + "=" * 70)
        logger.info("[Test 2] Forward Pass (Phase 2)")
        logger.info("=" * 70)

        torch.cuda.reset_peak_memory_stats()
        B, T = self.batch_size, self.video_length

        try:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model.forward_with_onepiece(
                    (images,), phase=2, no_shift=self.config.get('no_shift', False)
                )

            log_memory("after forward")

            # Check output keys
            expected_keys = ['metric_depth', 'relative_depth', 'metric_depth_isolated',
                           'relative_depth_isolated', 'modulated_features', 'scale', 'shift', 'dpt_features']
            for key in expected_keys:
                self._record(f"output key '{key}' exists", key in outputs)

            # Check shapes
            md = outputs['metric_depth']
            self._record("metric_depth shape", md.shape == (B, T, md.shape[2], md.shape[3]),
                         f"{list(md.shape)}")

            rd_iso = outputs['relative_depth_isolated']
            self._record("relative_depth_isolated not None (Phase 2)",
                         rd_iso is not None, f"shape={list(rd_iso.shape) if rd_iso is not None else 'None'}")

            mf = outputs['modulated_features']
            self._record("modulated_features shape",
                         mf.shape[0] == B and mf.shape[1] == T and mf.shape[2] == 256,
                         f"{list(mf.shape)}")

            scale = outputs['scale']
            self._record("scale shape", scale.shape == (B, T), f"{list(scale.shape)}")
            self._record("scale positive", (scale > 0).all().item(), f"min={scale.min().item():.4f}")

            return outputs

        except torch.cuda.OutOfMemoryError as e:
            self._record("Forward pass OOM", False, str(e))
            return None
        except Exception as e:
            self._record("Forward pass error", False, str(e))
            return None

    # =========================================================================
    # Test 3: Loss Computation
    # =========================================================================
    def test_loss(self, outputs, gt_depth, images, flow_estimator, loss_fn):
        logger.info("\n" + "=" * 70)
        logger.info("[Test 3] Loss Computation (Phase 2: LogL1 + TGM + WFC + SSIL)")
        logger.info("=" * 70)

        B, T = self.batch_size, self.video_length

        metric_depth = outputs['metric_depth']
        metric_depth_isolated = outputs['metric_depth_isolated']
        relative_depth_isolated = outputs['relative_depth_isolated']
        modulated_features = outputs['modulated_features']

        try:
            with torch.amp.autocast('cuda', enabled=False):
                gt_depth_meters = 1.0 / (gt_depth.squeeze(2).clamp(min=1e-8))

                # Interpolate if needed
                if metric_depth.shape[-2:] != gt_depth_meters.shape[-2:]:
                    BT = B * T
                    metric_depth = F.interpolate(
                        metric_depth.view(BT, 1, metric_depth.shape[-2], metric_depth.shape[-1]),
                        size=gt_depth_meters.shape[-2:], mode='bilinear', align_corners=True
                    ).squeeze(1).view(B, T, gt_depth_meters.shape[-2], gt_depth_meters.shape[-1])
                    metric_depth_isolated = F.interpolate(
                        metric_depth_isolated.view(BT, 1, metric_depth_isolated.shape[-2], metric_depth_isolated.shape[-1]),
                        size=gt_depth_meters.shape[-2:], mode='bilinear', align_corners=True
                    ).squeeze(1).view(B, T, gt_depth_meters.shape[-2], gt_depth_meters.shape[-1])
                    if relative_depth_isolated is not None:
                        relative_depth_isolated = F.interpolate(
                            relative_depth_isolated.view(BT, 1, relative_depth_isolated.shape[-2], relative_depth_isolated.shape[-1]),
                            size=gt_depth_meters.shape[-2:], mode='bilinear', align_corners=True
                        ).squeeze(1).view(B, T, gt_depth_meters.shape[-2], gt_depth_meters.shape[-1])

                gt_valid = (gt_depth.squeeze(2) > 0)
                pred_valid = (metric_depth > 0) & (metric_depth < 1000.0)
                valid_mask = gt_valid & pred_valid

                pred_inverse = 1.0 / metric_depth.float().clamp(min=1e-8)
                pred_inverse_isolated = 1.0 / metric_depth_isolated.float().clamp(min=1e-8)
                gt_inverse = gt_depth.squeeze(2).float()
                gt_inverse_for_ssil = gt_depth.squeeze(2).float() * 100.0

                total_loss, loss_components = loss_fn(
                    metric_depth=pred_inverse,
                    gt_depth=gt_inverse,
                    valid_mask=valid_mask.float(),
                    metric_depth_isolated=pred_inverse_isolated,
                    relative_depth_isolated=relative_depth_isolated,
                    gt_depth_for_ssil=gt_inverse_for_ssil,
                    modulated_features=modulated_features.float(),
                    images=images.float(),
                    flow_estimator=flow_estimator,
                    phase=2,
                    return_components=True
                )

            log_memory("after loss")

            # Check losses
            logger.info(f"  Total loss: {total_loss.item():.6f}")
            for name, val in loss_components.items():
                logger.info(f"    {name}: {val:.6f}")

            self._record("total_loss is finite", torch.isfinite(total_loss).item(),
                         f"{total_loss.item():.6f}")
            self._record("total_loss > 0", total_loss.item() > 0, f"{total_loss.item():.6f}")
            self._record("LogL1 loss computed", loss_components['log_l1_loss'] > 0,
                         f"{loss_components['log_l1_loss']:.6f}")
            self._record("TGM loss computed", loss_components['tgm_loss'] > 0,
                         f"{loss_components['tgm_loss']:.6f}")
            self._record("WFC loss computed (Phase 2)", loss_components['wfc_loss'] > 0,
                         f"{loss_components['wfc_loss']:.6f}")
            self._record("SSIL loss computed (Phase 2)", loss_components['ssil_loss'] > 0,
                         f"{loss_components['ssil_loss']:.6f}")
            self._record("total_loss requires grad", total_loss.requires_grad)

            return total_loss

        except torch.cuda.OutOfMemoryError as e:
            self._record("Loss computation OOM", False, str(e))
            return None
        except Exception as e:
            self._record("Loss computation error", False, str(e))
            import traceback
            traceback.print_exc()
            return None

    # =========================================================================
    # Test 4: Backward + Gradient Flow
    # =========================================================================
    def test_backward(self, model, total_loss, optimizer):
        logger.info("\n" + "=" * 70)
        logger.info("[Test 4] Backward Pass + Gradient Flow")
        logger.info("=" * 70)

        # Save param snapshots before backward
        param_snapshots = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_snapshots[name] = param.data.clone()

        try:
            optimizer.zero_grad()
            total_loss.backward()
            log_memory("after backward")

            self._record("backward() succeeded", True)

            # Check gradients
            grad_stats = {
                'mamba': {'has_grad': False, 'grad_norm': 0.0, 'count': 0},
                'metric_head': {'has_grad': False, 'grad_norm': 0.0, 'count': 0},
                'film': {'has_grad': False, 'grad_norm': 0.0, 'count': 0},
                'rel_head': {'has_grad': False, 'grad_norm': 0.0, 'count': 0},
            }

            frozen_with_grad = []
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if 'unified_global_mamba' in name:
                        grad_stats['mamba']['has_grad'] = True
                        grad_stats['mamba']['grad_norm'] += grad_norm
                        grad_stats['mamba']['count'] += 1
                    elif 'onepiece_metric_head' in name:
                        grad_stats['metric_head']['has_grad'] = True
                        grad_stats['metric_head']['grad_norm'] += grad_norm
                        grad_stats['metric_head']['count'] += 1
                    elif 'onepiece_film_generator' in name:
                        grad_stats['film']['has_grad'] = True
                        grad_stats['film']['grad_norm'] += grad_norm
                        grad_stats['film']['count'] += 1
                    elif 'output_conv' in name:
                        grad_stats['rel_head']['has_grad'] = True
                        grad_stats['rel_head']['grad_norm'] += grad_norm
                        grad_stats['rel_head']['count'] += 1

                if not param.requires_grad and param.grad is not None:
                    frozen_with_grad.append(name)

            for group_name, stats in grad_stats.items():
                avg_norm = stats['grad_norm'] / max(stats['count'], 1)
                self._record(f"Gradients flow to {group_name}",
                             stats['has_grad'],
                             f"avg_norm={avg_norm:.6f}, n_params={stats['count']}")

            self._record("No gradients on frozen params",
                         len(frozen_with_grad) == 0,
                         f"{len(frozen_with_grad)} frozen params have grads" if frozen_with_grad else "clean")

            # Gradient clipping
            total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self._record("Grad clipping succeeded",
                         torch.isfinite(torch.tensor(total_grad_norm)).item(),
                         f"grad_norm={total_grad_norm:.4f}")

            return param_snapshots

        except torch.cuda.OutOfMemoryError as e:
            self._record("backward() OOM", False, str(e))
            logger.error(f"\n  *** OOM during backward! ***")
            logger.error(f"  Peak memory: {fmt_mem(torch.cuda.max_memory_allocated())}")
            logger.error(f"  GPU total:   {fmt_mem(torch.cuda.get_device_properties(0).total_memory)}")
            return None
        except Exception as e:
            self._record("backward() error", False, str(e))
            import traceback
            traceback.print_exc()
            return None

    # =========================================================================
    # Test 5: Optimizer Step
    # =========================================================================
    def test_optimizer_step(self, model, optimizer, param_snapshots):
        logger.info("\n" + "=" * 70)
        logger.info("[Test 5] Optimizer Step (Phase 2: 2 param groups)")
        logger.info("=" * 70)

        try:
            optimizer.step()
            self._record("optimizer.step() succeeded", True)

            # Check parameters actually changed
            changed_count = 0
            unchanged_count = 0
            for name, param in model.named_parameters():
                if name in param_snapshots:
                    if not torch.equal(param.data, param_snapshots[name]):
                        changed_count += 1
                    else:
                        unchanged_count += 1

            self._record("Trainable params updated",
                         changed_count > 0,
                         f"{changed_count} changed, {unchanged_count} unchanged")

            # Verify frozen params didn't change
            frozen_changed = []
            for name, param in model.named_parameters():
                if not param.requires_grad and name in param_snapshots:
                    if not torch.equal(param.data, param_snapshots[name]):
                        frozen_changed.append(name)

            self._record("Frozen params unchanged",
                         len(frozen_changed) == 0,
                         f"{len(frozen_changed)} frozen params changed" if frozen_changed else "clean")

            log_memory("after optimizer step")

        except Exception as e:
            self._record("optimizer.step() error", False, str(e))

    # =========================================================================
    # Test 6: Memory Report
    # =========================================================================
    def test_memory_report(self):
        logger.info("\n" + "=" * 70)
        logger.info("[Test 6] Memory Report")
        logger.info("=" * 70)

        total_memory = torch.cuda.get_device_properties(0).total_memory
        peak_mem = torch.cuda.max_memory_allocated()
        headroom = total_memory - peak_mem

        logger.info(f"  Peak memory used: {fmt_mem(peak_mem)}")
        logger.info(f"  GPU total:        {fmt_mem(total_memory)}")
        logger.info(f"  Headroom:         {fmt_mem(headroom)} ({100 * headroom / total_memory:.1f}%)")
        logger.info(f"  Batch size:       {self.batch_size}")
        logger.info(f"  Video length:     {self.video_length}")

        if headroom < 0:
            self._record("Memory sufficient", False, f"OVER by {fmt_mem(-headroom)}")
        elif headroom < 2 * (1024 ** 3):
            self._record("Memory sufficient", True,
                         f"WARNING: tight headroom ({fmt_mem(headroom)}), real data may OOM")
        else:
            self._record("Memory sufficient", True, f"headroom={fmt_mem(headroom)}")

        # Estimate max batch size
        per_sample_mem = peak_mem / self.batch_size  # rough estimate
        max_bs = int(total_memory * 0.90 / per_sample_mem)
        logger.info(f"\n  Estimated max batch_size: ~{max_bs} (rough, assumes linear scaling)")

    # =========================================================================
    # Run All Tests
    # =========================================================================
    def run(self):
        self._print_header()

        torch.cuda.reset_peak_memory_stats()

        # Setup
        model = self._build_model()
        log_memory("model loaded")

        flow_estimator = self._build_flow_estimator()
        log_memory("flow estimator loaded")

        loss_config = self.config.get('loss', {})
        loss_fn = OnepieceCombinedLoss(
            log_l1_weight=loss_config.get('log_l1_weight', 1.0),
            tgm_weight=loss_config.get('tgm_weight', 1.0),
            wfc_weight=loss_config.get('wfc_weight', 0.01),
            ssil_weight=loss_config.get('ssil_weight', 1.0),
            use_log_space=loss_config.get('use_log_space', True)
        )

        # Test 1: Parameter config
        self.test_parameter_config(model)

        # Setup optimizer (Phase 2: 2 param groups)
        lr_config = self.config.training.get('lr', {})
        base_lr = lr_config.get('onepiece', 1e-4)
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

        optimizer = torch.optim.AdamW(
            [
                {'params': mamba_metric_params, 'lr': base_lr, 'name': 'mamba_metric'},
                {'params': film_rel_params, 'lr': phase2_lr, 'name': 'film_relhead'},
            ],
            betas=[0.9, 0.95],
            weight_decay=self.config.training.get('weight_decay', 1e-6)
        )
        logger.info(f"\n  Optimizer: 2 param groups (mamba_metric @ {base_lr:.1e}, film_relhead @ {phase2_lr:.1e})")

        # Create dummy data
        images, gt_depth = self._create_dummy_data()
        logger.info(f"  Dummy data: images={list(images.shape)}, gt_depth={list(gt_depth.shape)}")

        # Test 2: Forward pass
        outputs = self.test_forward(model, images)
        if outputs is None:
            self._print_summary()
            return

        # Test 3: Loss
        total_loss = self.test_loss(outputs, gt_depth, images, flow_estimator, loss_fn)
        if total_loss is None:
            self._print_summary()
            return

        # Test 4: Backward + gradients
        param_snapshots = self.test_backward(model, total_loss, optimizer)
        if param_snapshots is None:
            self._print_summary()
            return

        # Test 5: Optimizer step
        self.test_optimizer_step(model, optimizer, param_snapshots)

        # Test 6: Memory report
        self.test_memory_report()

        # Bonus: Run 2 more steps to verify stability
        logger.info("\n" + "=" * 70)
        logger.info("[Bonus] Stability Check (2 more forward+backward steps)")
        logger.info("=" * 70)
        for extra_step in range(2):
            try:
                torch.cuda.empty_cache()
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model.forward_with_onepiece(
                        (images,), phase=2, no_shift=self.config.get('no_shift', False)
                    )

                with torch.amp.autocast('cuda', enabled=False):
                    gt_depth_meters = 1.0 / (gt_depth.squeeze(2).clamp(min=1e-8))
                    md = outputs['metric_depth']
                    md_iso = outputs['metric_depth_isolated']
                    rd_iso = outputs['relative_depth_isolated']
                    mf = outputs['modulated_features']

                    if md.shape[-2:] != gt_depth_meters.shape[-2:]:
                        BT = self.batch_size * self.video_length
                        md = F.interpolate(
                            md.view(BT, 1, md.shape[-2], md.shape[-1]),
                            size=gt_depth_meters.shape[-2:], mode='bilinear', align_corners=True
                        ).squeeze(1).view(self.batch_size, self.video_length, gt_depth_meters.shape[-2], gt_depth_meters.shape[-1])
                        md_iso = F.interpolate(
                            md_iso.view(BT, 1, md_iso.shape[-2], md_iso.shape[-1]),
                            size=gt_depth_meters.shape[-2:], mode='bilinear', align_corners=True
                        ).squeeze(1).view(self.batch_size, self.video_length, gt_depth_meters.shape[-2], gt_depth_meters.shape[-1])
                        if rd_iso is not None:
                            rd_iso = F.interpolate(
                                rd_iso.view(BT, 1, rd_iso.shape[-2], rd_iso.shape[-1]),
                                size=gt_depth_meters.shape[-2:], mode='bilinear', align_corners=True
                            ).squeeze(1).view(self.batch_size, self.video_length, gt_depth_meters.shape[-2], gt_depth_meters.shape[-1])

                    gt_valid = (gt_depth.squeeze(2) > 0)
                    pred_valid = (md > 0) & (md < 1000.0)
                    valid_mask = gt_valid & pred_valid

                    pred_inv = 1.0 / md.float().clamp(min=1e-8)
                    pred_inv_iso = 1.0 / md_iso.float().clamp(min=1e-8)
                    gt_inv = gt_depth.squeeze(2).float()
                    gt_inv_ssil = gt_depth.squeeze(2).float() * 100.0

                    loss, comps = loss_fn(
                        metric_depth=pred_inv, gt_depth=gt_inv, valid_mask=valid_mask.float(),
                        metric_depth_isolated=pred_inv_iso,
                        relative_depth_isolated=rd_iso,
                        gt_depth_for_ssil=gt_inv_ssil,
                        modulated_features=mf.float(),
                        images=images.float(),
                        flow_estimator=flow_estimator,
                        phase=2, return_components=True
                    )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                self._record(f"Stability step {extra_step + 1}", True,
                             f"loss={loss.item():.4f}")

            except torch.cuda.OutOfMemoryError:
                self._record(f"Stability step {extra_step + 1}", False, "OOM")
                break
            except Exception as e:
                self._record(f"Stability step {extra_step + 1}", False, str(e))
                break

        self._print_summary()

    def _print_summary(self):
        logger.info("\n" + "=" * 70)
        logger.info("  SUMMARY")
        logger.info("=" * 70)
        logger.info(f"  Passed: {len(self.passed)}/{len(self.passed) + len(self.failed)}")

        if self.failed:
            logger.info(f"\n  FAILED tests:")
            for name in self.failed:
                logger.info(f"    - {name}")
        else:
            logger.info(f"\n  All tests passed! Phase 2 is ready.")

        logger.info("=" * 70)


@hydra.main(version_base=None, config_path="configs/onepiece", config_name="config")
def main(config: DictConfig):
    """Main entry point for Phase 2 comprehensive test."""
    tester = Phase2Tester(config)
    tester.run()


if __name__ == "__main__":
    main()
