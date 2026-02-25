#!/usr/bin/env python3
"""
DPT Feature Flickering Analysis Script

Analyzes where depth flickering originates in the FlashDepth pipeline and
provides quantitative evidence for FiLM modulation approach.

Captures Pre-Mamba and Post-Mamba features via monkey-patching, then performs:
  Part A: Temporal stability (cosine similarity frame-to-frame)
  Part B: FiLM validity (affine alignment, channel stats drift, variance decomposition)

Supports both:
  - FlashDepth-L  (mamba_in_dpt_layer=[3], path_1)
  - FlashDepth Full (mamba_in_dpt_layer=[1], path_3)

Usage:
  python analyze_dpt_features.py \
    --config-path ../FlashDepth/configs/flashdepth-l \
    --checkpoint ../FlashDepth/configs/flashdepth-l/checkpoint.pth \
    --data-root /home/cvlab/hsy/Datasets \
    --dataset sintel \
    --seq-idx 0 \
    --video-length 50 \
    --results-dir analysis_results/flashdepth_l_sintel_seq0 \
    --gpu 0
"""

import os
import sys
import argparse
import json
import logging
import yaml
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── paths ──────────────────────────────────────────────────────────────
project_root = Path(__file__).parent

# Detect Docker vs host: ../FlashDepth on host, /FlashDepth in Docker
flashdepth_root_host = project_root.parent / 'FlashDepth'
flashdepth_root_docker = Path('/FlashDepth')
if flashdepth_root_docker.exists():
    flashdepth_root = flashdepth_root_docker
else:
    flashdepth_root = flashdepth_root_host

sys.path.insert(0, str(project_root))

from flashdepth.model import FlashDepth
from dataloaders.combined_dataset import CombinedDataset
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# 1. CLI
# ════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="DPT Feature Flickering Analysis")
    p.add_argument('--config-path', required=True, help='Path to FlashDepth config dir (e.g. ../FlashDepth/configs/flashdepth-l)')
    p.add_argument('--checkpoint', required=True, help='Path to FlashDepth checkpoint .pth')
    p.add_argument('--data-root', required=True, help='Dataset root directory')
    p.add_argument('--dataset', default='sintel', help='Dataset name for testing')
    p.add_argument('--seq-idx', type=int, default=0, help='Sequence index to analyze')
    p.add_argument('--video-length', type=int, default=50, help='Number of frames')
    p.add_argument('--results-dir', default='analysis_results/default', help='Output directory')
    p.add_argument('--flicker-threshold', type=float, default=3.0, help='MAD multiplier for flicker detection')
    p.add_argument('--flicker-frames', type=str, default=None, help='Manual flicker frame indices, comma-separated (e.g. 10,25)')
    p.add_argument('--gpu', type=int, default=0, help='GPU id')
    p.add_argument('--top-k-channels', type=int, default=10, help='Number of top unstable channels to highlight')
    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════
# 2. Model loading
# ════════════════════════════════════════════════════════════════════════

def load_config(config_path):
    """Load YAML config from FlashDepth config directory."""
    cfg_file = os.path.join(config_path, 'config.yaml')
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_model(cfg, checkpoint_path, device):
    """Construct FlashDepth model from config and load weights."""
    model_cfg = cfg['model']

    kwargs = {
        'vit_size': model_cfg['vit_size'],
        'patch_size': model_cfg.get('patch_size', 14),
        'use_mamba': model_cfg.get('use_mamba', True),
        'mamba_type': model_cfg.get('mamba_type', 'add'),
        'num_mamba_layers': model_cfg.get('num_mamba_layers', 4),
        'downsample_mamba': model_cfg.get('downsample_mamba', [0.1]),
        'mamba_pos_embed': model_cfg.get('mamba_pos_embed', None),
        'mamba_in_dpt_layer': model_cfg.get('mamba_in_dpt_layer', [3]),
        'mamba_d_conv': model_cfg.get('mamba_d_conv', 4),
        'mamba_d_state': model_cfg.get('mamba_d_state', 256),
        'use_hydra': model_cfg.get('use_hydra', False),
        'use_transformer_rnn': model_cfg.get('use_transformer_rnn', False),
        'use_xlstm': model_cfg.get('use_xlstm', False),
        'batch_size': 1,
        'training': False,
    }

    # Hybrid configs — model expects DictConfig-like objects with attribute access
    hybrid_cfg = cfg.get('hybrid_configs', {})
    if hybrid_cfg and hybrid_cfg.get('use_hybrid', False):
        from omegaconf import OmegaConf
        kwargs['hybrid_configs'] = OmegaConf.create(hybrid_cfg)
    else:
        kwargs['hybrid_configs'] = None

    model = FlashDepth(**kwargs)

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    model = model.to(device)
    model.eval()

    mamba_layer = model_cfg.get('mamba_in_dpt_layer', [3])
    logger.info(f"Model loaded. vit={model_cfg['vit_size']}, mamba_in_dpt_layer={mamba_layer}")
    return model


# ════════════════════════════════════════════════════════════════════════
# 3. Monkey-patch for feature capture
# ════════════════════════════════════════════════════════════════════════

def patch_model_for_capture(model, feature_store):
    """
    Monkey-patch model.dpt_features_to_mamba to capture input/output features.
    Each call appends (pre_mamba, post_mamba) for one frame.
    """
    original_fn = model.dpt_features_to_mamba

    def patched_fn(input_shape, dpt_features, in_dpt_layer):
        # Capture pre-Mamba (clone to avoid mutation)
        pre_mamba = dpt_features.detach().clone().cpu()  # (B*T, c, h, w), T=1 during inference
        feature_store['pre_mamba'].append(pre_mamba)

        # Run original
        mamba_out = original_fn(input_shape, dpt_features, in_dpt_layer)

        # Capture post-Mamba
        post_mamba = mamba_out.detach().clone().cpu()
        feature_store['post_mamba'].append(post_mamba)

        return mamba_out

    model.dpt_features_to_mamba = patched_fn
    logger.info("Monkey-patched dpt_features_to_mamba for feature capture")


# ════════════════════════════════════════════════════════════════════════
# 4. Data loading
# ════════════════════════════════════════════════════════════════════════

def load_test_sequence(args):
    """Load a single test sequence using CombinedDataset."""
    dataset = CombinedDataset(
        root_dir=args.data_root,
        enable_dataset_flags=[args.dataset],
        resolution='base',
        split='test',
        video_length=args.video_length,
        skip_gt_canonicalization=True,
    )

    if args.seq_idx >= len(dataset):
        logger.warning(f"seq_idx={args.seq_idx} >= dataset length {len(dataset)}, clamping to 0")
        args.seq_idx = 0

    sample = dataset[args.seq_idx]
    if sample is None:
        raise RuntimeError(f"Sequence {args.seq_idx} returned None")

    # Unpack test tuple (9 elements)
    if len(sample) == 9:
        images, depths, fl_canon, fl_actual, valid_masks, fx_ratios, resize_ratios, name, img_paths = sample
    elif len(sample) == 8:
        images, depths, fl_canon, fl_actual, valid_masks, fx_ratios, resize_ratios, name = sample
        img_paths = None
    else:
        raise RuntimeError(f"Unexpected sample length: {len(sample)}")

    logger.info(f"Loaded sequence '{name}': {images.shape[0]} frames, resolution {images.shape[-2]}x{images.shape[-1]}")
    return {
        'images': images,           # (T, 3, H, W)
        'depths': depths,           # (T, H, W) inverse depth
        'valid_masks': valid_masks,  # (T, H, W)
        'fx_ratios': fx_ratios,     # (T,)
        'resize_ratios': resize_ratios,  # (T,)
        'name': name,
    }


# ════════════════════════════════════════════════════════════════════════
# 5. Inference
# ════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(model, sequence, feature_store, device):
    """Run frame-by-frame inference, collecting features and depth predictions."""
    images = sequence['images']  # (T, 3, H, W)
    T = images.shape[0]

    feature_store['pre_mamba'] = []
    feature_store['post_mamba'] = []

    model.mamba.start_new_sequence()

    depths = []
    images_gpu = images.unsqueeze(0).to(device)  # (1, T, 3, H, W)

    for i in range(T):
        frame = images_gpu[:, i]  # (1, 3, H, W)
        B, C, H, W = frame.shape
        patch_h, patch_w = H // model.patch_size, W // model.patch_size

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            dpt_features = model.get_dpt_features(frame, input_shape=(B, C, H, W))
            pred_depth = model.final_head(dpt_features, patch_h, patch_w)
            pred_depth = torch.clip(pred_depth, min=0)

        depths.append(pred_depth.squeeze(0).float().cpu())  # (H, W)

    depths = torch.stack(depths)  # (T, H, W)
    logger.info(f"Inference complete: {T} frames, depth shape {depths.shape}")
    return depths


# ════════════════════════════════════════════════════════════════════════
# 6. Flickering frame detection
# ════════════════════════════════════════════════════════════════════════

def detect_flicker_frames(depths, args):
    """
    Detect flickering frames from depth output.
    Uses median + k*MAD threshold on frame-to-frame L1 differences.
    """
    T = depths.shape[0]

    # Frame-to-frame L1 difference
    diffs = []
    for t in range(1, T):
        diff = (depths[t] - depths[t - 1]).abs().mean().item()
        diffs.append(diff)
    diffs = np.array(diffs)

    # MAD-based outlier detection
    median_diff = np.median(diffs)
    mad = np.median(np.abs(diffs - median_diff))
    threshold = median_diff + args.flicker_threshold * max(mad, 1e-8)

    auto_flicker = [t + 1 for t, d in enumerate(diffs) if d > threshold]

    # Manual override
    manual_flicker = []
    if args.flicker_frames:
        manual_flicker = [int(x.strip()) for x in args.flicker_frames.split(',') if x.strip()]

    # Combine (unique, sorted)
    all_flicker = sorted(set(auto_flicker + manual_flicker))
    # Remove invalid indices
    all_flicker = [f for f in all_flicker if 1 <= f < T]

    logger.info(f"Depth temporal diffs: median={median_diff:.4f}, MAD={mad:.4f}, threshold={threshold:.4f}")
    logger.info(f"Auto-detected flicker frames: {auto_flicker}")
    if manual_flicker:
        logger.info(f"Manual flicker frames: {manual_flicker}")
    logger.info(f"Combined flicker frames: {all_flicker}")

    return all_flicker, diffs


# ════════════════════════════════════════════════════════════════════════
# 7. Part A: Temporal Stability Analysis
# ════════════════════════════════════════════════════════════════════════

def cosine_sim_global(a, b):
    """Global cosine similarity: flatten (C,H,W) → single scalar."""
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def cosine_sim_pixelwise(a, b):
    """Pixel-wise cosine similarity: (C,H,W) → (H,W)."""
    # a, b: (C, H, W)
    a = a.float()
    b = b.float()
    dot = (a * b).sum(dim=0)  # (H, W)
    norm_a = a.norm(dim=0).clamp(min=1e-8)
    norm_b = b.norm(dim=0).clamp(min=1e-8)
    return dot / (norm_a * norm_b)  # (H, W)


def analyze_temporal_stability(feature_store, depths):
    """
    Part A: Temporal stability analysis.
    1. Pre-Mamba frame-to-frame cosine similarity
    2. Post-Mamba frame-to-frame cosine similarity
    3. Per-frame pre vs post Mamba cosine similarity (Mamba effect)
    """
    pre_list = feature_store['pre_mamba']   # list of (1, C, H, W)
    post_list = feature_store['post_mamba']  # list of (1, C, H, W)
    T = len(pre_list)

    # Squeeze batch dim
    pre_feats = [f.squeeze(0) for f in pre_list]   # list of (C, h, w)
    post_feats = [f.squeeze(0) for f in post_list]

    # 1. Pre-Mamba temporal cosine similarity
    pre_temporal_sim = []
    for t in range(1, T):
        sim = cosine_sim_global(pre_feats[t], pre_feats[t - 1])
        pre_temporal_sim.append(sim)

    # 2. Post-Mamba temporal cosine similarity
    post_temporal_sim = []
    for t in range(1, T):
        sim = cosine_sim_global(post_feats[t], post_feats[t - 1])
        post_temporal_sim.append(sim)

    # 3. Mamba effect: pre vs post cosine similarity per frame
    mamba_effect = []
    for t in range(T):
        sim = cosine_sim_global(pre_feats[t], post_feats[t])
        mamba_effect.append(sim)

    results = {
        'pre_temporal_cosine_sim': pre_temporal_sim,
        'post_temporal_cosine_sim': post_temporal_sim,
        'mamba_effect_cosine_sim': mamba_effect,
        'pre_temporal_mean': float(np.mean(pre_temporal_sim)),
        'post_temporal_mean': float(np.mean(post_temporal_sim)),
        'mamba_effect_mean': float(np.mean(mamba_effect)),
    }

    logger.info(f"Part A — Pre-Mamba temporal sim:  mean={results['pre_temporal_mean']:.6f}")
    logger.info(f"Part A — Post-Mamba temporal sim: mean={results['post_temporal_mean']:.6f}")
    logger.info(f"Part A — Mamba effect sim:        mean={results['mamba_effect_mean']:.6f}")

    return results


# ════════════════════════════════════════════════════════════════════════
# 8. Part B: FiLM Validity Analysis
# ════════════════════════════════════════════════════════════════════════

def compute_affine_alignment(feat_flicker, feat_stable):
    """
    Channel-wise affine alignment test.
    For each channel c: y ≈ gamma_c * x + beta_c (least squares)

    Args:
        feat_flicker: (C, H, W) — the "source" frame (flickering)
        feat_stable:  (C, H, W) — the "target" frame (stable, typically t-1)

    Returns:
        aligned: (C, H, W)
        residual: (C, H, W)
        r_affine: scalar — fraction of variance explained by affine
        gammas: (C,)
        betas: (C,)
    """
    C, H, W = feat_flicker.shape
    feat_flicker = feat_flicker.float()
    feat_stable = feat_stable.float()

    aligned = torch.zeros_like(feat_stable)
    gammas = torch.zeros(C)
    betas = torch.zeros(C)

    for c in range(C):
        x = feat_flicker[c].flatten()  # (H*W,)
        y = feat_stable[c].flatten()   # (H*W,)
        A = torch.stack([x, torch.ones_like(x)], dim=1)  # (H*W, 2)
        solution = torch.linalg.lstsq(A, y).solution      # (2,)
        gamma_c, beta_c = solution[0].item(), solution[1].item()
        gammas[c] = gamma_c
        betas[c] = beta_c
        aligned[c] = gamma_c * feat_flicker[c] + beta_c

    residual = aligned - feat_stable
    total_diff = feat_flicker - feat_stable
    total_var = total_diff.norm() ** 2
    residual_var = residual.norm() ** 2

    if total_var > 1e-10:
        r_affine = 1.0 - (residual_var / total_var).item()
    else:
        r_affine = 1.0  # Identical features

    return aligned, residual, r_affine, gammas, betas


def compute_channel_stats(features_list):
    """
    Extract per-channel mean/std across spatial dims for each frame.
    Args:
        features_list: list of (C, H, W) tensors, length T
    Returns:
        means: (T, C)
        stds: (T, C)
    """
    means = torch.stack([f.float().mean(dim=(-2, -1)) for f in features_list])  # (T, C)
    stds = torch.stack([f.float().std(dim=(-2, -1)) for f in features_list])    # (T, C)
    return means, stds


def analyze_film_validity(feature_store, flicker_frames):
    """
    Part B: FiLM validity analysis.
    4. Affine alignment test (per flicker frame)
    5. Channel statistics drift
    6. Variance decomposition
    """
    pre_list = feature_store['pre_mamba']
    post_list = feature_store['post_mamba']
    T = len(pre_list)

    pre_feats = [f.squeeze(0) for f in pre_list]   # (C, h, w)
    post_feats = [f.squeeze(0) for f in post_list]

    results = {
        'affine_alignment': {},
        'channel_stats': {},
        'variance_decomposition': {},
    }

    # ── 4. Affine Alignment Test ──
    for t in flicker_frames:
        if t < 1 or t >= T:
            continue
        stable_t = t - 1

        # Pre-Mamba
        _, residual_pre, r_pre, gammas_pre, betas_pre = compute_affine_alignment(pre_feats[t], pre_feats[stable_t])
        # Post-Mamba
        _, residual_post, r_post, gammas_post, betas_post = compute_affine_alignment(post_feats[t], post_feats[stable_t])

        # L2 distance before/after alignment
        pre_l2_before = (pre_feats[t].float() - pre_feats[stable_t].float()).norm().item()
        pre_l2_after = residual_pre.norm().item()
        post_l2_before = (post_feats[t].float() - post_feats[stable_t].float()).norm().item()
        post_l2_after = residual_post.norm().item()

        results['affine_alignment'][str(t)] = {
            'pre_mamba': {
                'r_affine': r_pre,
                'l2_before': pre_l2_before,
                'l2_after': pre_l2_after,
                'l2_ratio': pre_l2_after / max(pre_l2_before, 1e-10),
            },
            'post_mamba': {
                'r_affine': r_post,
                'l2_before': post_l2_before,
                'l2_after': post_l2_after,
                'l2_ratio': post_l2_after / max(post_l2_before, 1e-10),
            },
        }

    # ── 5. Channel Statistics Drift ──
    pre_means, pre_stds = compute_channel_stats(pre_feats)
    post_means, post_stds = compute_channel_stats(post_feats)

    results['channel_stats'] = {
        'pre_means': pre_means.numpy().tolist(),
        'pre_stds': pre_stds.numpy().tolist(),
        'post_means': post_means.numpy().tolist(),
        'post_stds': post_stds.numpy().tolist(),
    }

    # ── 6. Variance Decomposition ──
    for t in flicker_frames:
        if t < 1 or t >= T:
            continue
        stable_t = t - 1

        # Pre-Mamba decomposition
        _, residual_pre, r_pre, _, _ = compute_affine_alignment(pre_feats[t], pre_feats[stable_t])
        # Post-Mamba decomposition
        _, residual_post, r_post, _, _ = compute_affine_alignment(post_feats[t], post_feats[stable_t])

        results['variance_decomposition'][str(t)] = {
            'pre_mamba_r_affine': r_pre,
            'post_mamba_r_affine': r_post,
        }

    # Log summary
    if flicker_frames:
        pre_r_vals = [results['variance_decomposition'][str(t)]['pre_mamba_r_affine']
                      for t in flicker_frames if str(t) in results['variance_decomposition']]
        post_r_vals = [results['variance_decomposition'][str(t)]['post_mamba_r_affine']
                       for t in flicker_frames if str(t) in results['variance_decomposition']]
        if pre_r_vals:
            logger.info(f"Part B — R_affine (pre-Mamba):  mean={np.mean(pre_r_vals):.4f}")
        if post_r_vals:
            logger.info(f"Part B — R_affine (post-Mamba): mean={np.mean(post_r_vals):.4f}")

    return results


# ════════════════════════════════════════════════════════════════════════
# 9. Visualization
# ════════════════════════════════════════════════════════════════════════

def _save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved: {path}")


def visualize_part_a(temporal_results, diffs, flicker_frames, results_dir):
    """Part A visualizations."""
    out_dir = os.path.join(results_dir, 'part_a_temporal')

    pre_sim = temporal_results['pre_temporal_cosine_sim']
    post_sim = temporal_results['post_temporal_cosine_sim']
    mamba_eff = temporal_results['mamba_effect_cosine_sim']
    T_minus1 = len(pre_sim)

    # ── temporal_cosine_similarity.png ──
    fig, ax = plt.subplots(figsize=(14, 5))
    frames = list(range(1, T_minus1 + 1))
    ax.plot(frames, pre_sim, 'b-o', markersize=3, label='Pre-Mamba (frame-to-frame)', alpha=0.8)
    ax.plot(frames, post_sim, 'r-o', markersize=3, label='Post-Mamba (frame-to-frame)', alpha=0.8)
    for f in flicker_frames:
        if 1 <= f <= T_minus1:
            ax.axvline(x=f, color='orange', alpha=0.4, linestyle='--')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Temporal Cosine Similarity (frame t vs t-1)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(min(min(pre_sim), min(post_sim)) - 0.01, 1.001)
    _save_fig(fig, os.path.join(out_dir, 'temporal_cosine_similarity.png'))

    # ── mamba_effect.png ──
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(range(len(mamba_eff)), mamba_eff, 'g-o', markersize=3, label='Pre vs Post Mamba (same frame)', alpha=0.8)
    for f in flicker_frames:
        if 0 <= f < len(mamba_eff):
            ax.axvline(x=f, color='orange', alpha=0.4, linestyle='--')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Mamba Effect: Pre-Mamba vs Post-Mamba per frame')
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, os.path.join(out_dir, 'mamba_effect.png'))

    # ── depth_temporal_diff.png ──
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(range(1, len(diffs) + 1), diffs, 'k-o', markersize=3, alpha=0.8)
    median_diff = np.median(diffs)
    ax.axhline(y=median_diff, color='gray', linestyle=':', alpha=0.5, label=f'Median={median_diff:.4f}')
    for f in flicker_frames:
        if 1 <= f <= len(diffs):
            ax.axvline(x=f, color='red', alpha=0.5, linestyle='--')
            ax.annotate(f'F{f}', (f, diffs[f - 1]), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=8, color='red')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Mean L1 Depth Diff')
    ax.set_title('Depth Temporal Difference (flickering intensity)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, os.path.join(out_dir, 'depth_temporal_diff.png'))


def visualize_part_b(film_results, feature_store, flicker_frames, results_dir, top_k):
    """Part B visualizations."""
    out_dir = os.path.join(results_dir, 'part_b_film_validity')

    pre_feats = [f.squeeze(0) for f in feature_store['pre_mamba']]
    post_feats = [f.squeeze(0) for f in feature_store['post_mamba']]
    T = len(pre_feats)

    # ── affine_alignment_test.png ──
    if flicker_frames and film_results['affine_alignment']:
        valid_frames = [t for t in flicker_frames if str(t) in film_results['affine_alignment']]
        if valid_frames:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Left: L2 distance before/after
            x_pos = np.arange(len(valid_frames))
            width = 0.2
            pre_before = [film_results['affine_alignment'][str(t)]['pre_mamba']['l2_before'] for t in valid_frames]
            pre_after = [film_results['affine_alignment'][str(t)]['pre_mamba']['l2_after'] for t in valid_frames]
            post_before = [film_results['affine_alignment'][str(t)]['post_mamba']['l2_before'] for t in valid_frames]
            post_after = [film_results['affine_alignment'][str(t)]['post_mamba']['l2_after'] for t in valid_frames]

            axes[0].bar(x_pos - 1.5 * width, pre_before, width, label='Pre-Mamba before', color='steelblue', alpha=0.8)
            axes[0].bar(x_pos - 0.5 * width, pre_after, width, label='Pre-Mamba after', color='lightblue', alpha=0.8)
            axes[0].bar(x_pos + 0.5 * width, post_before, width, label='Post-Mamba before', color='firebrick', alpha=0.8)
            axes[0].bar(x_pos + 1.5 * width, post_after, width, label='Post-Mamba after', color='lightsalmon', alpha=0.8)
            axes[0].set_xticks(x_pos)
            axes[0].set_xticklabels([f'F{t}' for t in valid_frames])
            axes[0].set_ylabel('L2 Distance')
            axes[0].set_title('Affine Alignment: L2 before/after')
            axes[0].legend(fontsize=8)
            axes[0].grid(True, alpha=0.3)

            # Right: R_affine ratio
            r_pre = [film_results['affine_alignment'][str(t)]['pre_mamba']['r_affine'] for t in valid_frames]
            r_post = [film_results['affine_alignment'][str(t)]['post_mamba']['r_affine'] for t in valid_frames]
            axes[1].bar(x_pos - 0.2, r_pre, 0.35, label='Pre-Mamba R_affine', color='steelblue', alpha=0.8)
            axes[1].bar(x_pos + 0.2, r_post, 0.35, label='Post-Mamba R_affine', color='firebrick', alpha=0.8)
            axes[1].axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='R=0.8 threshold')
            axes[1].set_xticks(x_pos)
            axes[1].set_xticklabels([f'F{t}' for t in valid_frames])
            axes[1].set_ylabel('R_affine')
            axes[1].set_title('Variance Explained by Affine Transform')
            axes[1].legend(fontsize=8)
            axes[1].set_ylim(0, 1.05)
            axes[1].grid(True, alpha=0.3)

            fig.tight_layout()
            _save_fig(fig, os.path.join(out_dir, 'affine_alignment_test.png'))

    # ── channel_stats_drift.png ──
    pre_means = torch.tensor(film_results['channel_stats']['pre_means'])   # (T, C)
    pre_stds = torch.tensor(film_results['channel_stats']['pre_stds'])
    post_means = torch.tensor(film_results['channel_stats']['post_means'])
    post_stds = torch.tensor(film_results['channel_stats']['post_stds'])
    C = pre_means.shape[1]

    # Find top-k unstable channels (by std of the temporal mean series)
    pre_mean_volatility = pre_means.std(dim=0)   # (C,)
    post_mean_volatility = post_means.std(dim=0)
    combined_volatility = pre_mean_volatility + post_mean_volatility
    topk_channels = combined_volatility.argsort(descending=True)[:top_k].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    for ch_idx in topk_channels:
        axes[0, 0].plot(pre_means[:, ch_idx].numpy(), alpha=0.6, linewidth=1)
    axes[0, 0].set_title(f'Pre-Mamba Channel Mean (top-{top_k} volatile)')
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].grid(True, alpha=0.3)

    for ch_idx in topk_channels:
        axes[0, 1].plot(pre_stds[:, ch_idx].numpy(), alpha=0.6, linewidth=1)
    axes[0, 1].set_title(f'Pre-Mamba Channel Std (top-{top_k} volatile)')
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].grid(True, alpha=0.3)

    for ch_idx in topk_channels:
        axes[1, 0].plot(post_means[:, ch_idx].numpy(), alpha=0.6, linewidth=1)
    axes[1, 0].set_title(f'Post-Mamba Channel Mean (top-{top_k} volatile)')
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].grid(True, alpha=0.3)

    for ch_idx in topk_channels:
        axes[1, 1].plot(post_stds[:, ch_idx].numpy(), alpha=0.6, linewidth=1)
    axes[1, 1].set_title(f'Post-Mamba Channel Std (top-{top_k} volatile)')
    axes[1, 1].set_xlabel('Frame')
    axes[1, 1].grid(True, alpha=0.3)

    # Mark flicker frames
    for ax in axes.flatten():
        for f in flicker_frames:
            ax.axvline(x=f, color='red', alpha=0.3, linestyle='--')

    fig.suptitle(f'Channel Statistics Drift (top-{top_k} channels: {topk_channels})', fontsize=12)
    fig.tight_layout()
    _save_fig(fig, os.path.join(out_dir, 'channel_stats_drift.png'))

    # ── variance_decomposition.png ──
    if flicker_frames and film_results['variance_decomposition']:
        valid_frames = [t for t in flicker_frames if str(t) in film_results['variance_decomposition']]
        if valid_frames:
            fig, ax = plt.subplots(figsize=(12, 6))
            x_pos = np.arange(len(valid_frames))
            r_pre = [film_results['variance_decomposition'][str(t)]['pre_mamba_r_affine'] for t in valid_frames]
            r_post = [film_results['variance_decomposition'][str(t)]['post_mamba_r_affine'] for t in valid_frames]

            ax.bar(x_pos - 0.2, r_pre, 0.35, label='Pre-Mamba', color='steelblue', alpha=0.8)
            ax.bar(x_pos + 0.2, r_post, 0.35, label='Post-Mamba', color='firebrick', alpha=0.8)
            ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='R=0.8 (strong FiLM support)')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'Frame {t}' for t in valid_frames])
            ax.set_ylabel('R_affine (variance explained)')
            ax.set_title('Variance Decomposition: Affine vs Residual')
            ax.legend()
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)

            # Add value labels on bars
            for i, (rp, rpo) in enumerate(zip(r_pre, r_post)):
                ax.text(i - 0.2, rp + 0.02, f'{rp:.3f}', ha='center', fontsize=8, color='steelblue')
                ax.text(i + 0.2, rpo + 0.02, f'{rpo:.3f}', ha='center', fontsize=8, color='firebrick')

            fig.tight_layout()
            _save_fig(fig, os.path.join(out_dir, 'variance_decomposition.png'))


def visualize_flicker_details(feature_store, flicker_frames, depths, results_dir):
    """Per-flicker-frame heatmaps."""
    out_dir = os.path.join(results_dir, 'flicker_analysis')

    pre_feats = [f.squeeze(0) for f in feature_store['pre_mamba']]
    post_feats = [f.squeeze(0) for f in feature_store['post_mamba']]
    T = len(pre_feats)

    for t in flicker_frames:
        if t < 1 or t >= T:
            continue

        # ── pre-mamba pixel-wise cosine sim heatmap ──
        pre_sim_map = cosine_sim_pixelwise(pre_feats[t], pre_feats[t - 1])
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(pre_sim_map.numpy(), cmap='RdYlGn', vmin=0.8, vmax=1.0)
        plt.colorbar(im, ax=ax)
        ax.set_title(f'Frame {t}: Pre-Mamba pixel-wise cosine sim (vs t-1)')
        _save_fig(fig, os.path.join(out_dir, f'frame_{t:03d}_pre_mamba_diff.png'))

        # ── post-mamba pixel-wise cosine sim heatmap ──
        post_sim_map = cosine_sim_pixelwise(post_feats[t], post_feats[t - 1])
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(post_sim_map.numpy(), cmap='RdYlGn', vmin=0.8, vmax=1.0)
        plt.colorbar(im, ax=ax)
        ax.set_title(f'Frame {t}: Post-Mamba pixel-wise cosine sim (vs t-1)')
        _save_fig(fig, os.path.join(out_dir, f'frame_{t:03d}_post_mamba_diff.png'))

        # ── mamba effect heatmap (pre vs post, same frame) ──
        mamba_sim_map = cosine_sim_pixelwise(pre_feats[t], post_feats[t])
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(mamba_sim_map.numpy(), cmap='RdYlGn', vmin=0.8, vmax=1.0)
        plt.colorbar(im, ax=ax)
        ax.set_title(f'Frame {t}: Mamba Effect (pre vs post cosine sim)')
        _save_fig(fig, os.path.join(out_dir, f'frame_{t:03d}_mamba_effect.png'))

        # ── affine residual heatmap ──
        _, residual_post, _, _, _ = compute_affine_alignment(post_feats[t], post_feats[t - 1])
        residual_norm = residual_post.float().norm(dim=0)  # (h, w)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(residual_norm.numpy(), cmap='hot')
        plt.colorbar(im, ax=ax)
        ax.set_title(f'Frame {t}: Affine Residual (post-Mamba, L2 norm per pixel)')
        _save_fig(fig, os.path.join(out_dir, f'frame_{t:03d}_affine_residual.png'))

        # ── depth context strip: [t-1, t, t+1] ──
        num_context = min(3, T - t + 1)
        context_frames = list(range(max(0, t - 1), min(T, t + 2)))
        fig, axes = plt.subplots(1, len(context_frames), figsize=(5 * len(context_frames), 4))
        if len(context_frames) == 1:
            axes = [axes]
        for i, ct in enumerate(context_frames):
            depth_np = depths[ct].numpy()
            vmin, vmax = np.percentile(depth_np[depth_np > 0], [2, 98]) if (depth_np > 0).any() else (0, 1)
            axes[i].imshow(depth_np, cmap='inferno', vmin=vmin, vmax=vmax)
            label = f't={ct}'
            if ct == t:
                label += ' (flicker)'
            axes[i].set_title(label)
            axes[i].axis('off')
        fig.suptitle(f'Depth Context around Frame {t}')
        fig.tight_layout()
        _save_fig(fig, os.path.join(out_dir, f'frame_{t:03d}_context.png'))


# ════════════════════════════════════════════════════════════════════════
# 10. Summary & JSON output
# ════════════════════════════════════════════════════════════════════════

def write_summary(temporal_results, film_results, flicker_frames, diffs, results_dir):
    """Write summary.txt and feature_analysis.json."""

    # ── JSON ──
    json_data = {
        'temporal_stability': {
            'pre_temporal_mean_cosine_sim': temporal_results['pre_temporal_mean'],
            'post_temporal_mean_cosine_sim': temporal_results['post_temporal_mean'],
            'mamba_effect_mean_cosine_sim': temporal_results['mamba_effect_mean'],
            'pre_temporal_cosine_sim': temporal_results['pre_temporal_cosine_sim'],
            'post_temporal_cosine_sim': temporal_results['post_temporal_cosine_sim'],
            'mamba_effect_cosine_sim': temporal_results['mamba_effect_cosine_sim'],
        },
        'film_validity': {
            'affine_alignment': film_results['affine_alignment'],
            'variance_decomposition': film_results['variance_decomposition'],
        },
        'flicker_frames': flicker_frames,
        'depth_temporal_diffs': diffs.tolist(),
    }

    json_path = os.path.join(results_dir, 'feature_analysis.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    logger.info(f"Saved: {json_path}")

    # ── Summary text ──
    lines = []
    lines.append("=" * 60)
    lines.append("DPT Feature Flickering Analysis — Summary")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Part A: Temporal Stability")
    lines.append(f"  Pre-Mamba temporal cosine sim (mean):  {temporal_results['pre_temporal_mean']:.6f}")
    lines.append(f"  Post-Mamba temporal cosine sim (mean): {temporal_results['post_temporal_mean']:.6f}")
    lines.append(f"  Mamba effect cosine sim (mean):        {temporal_results['mamba_effect_mean']:.6f}")
    lines.append("")

    diff_improvement = temporal_results['post_temporal_mean'] - temporal_results['pre_temporal_mean']
    if diff_improvement > 0.001:
        lines.append(f"  → Mamba IMPROVES temporal stability (+{diff_improvement:.6f})")
    elif diff_improvement < -0.001:
        lines.append(f"  → Mamba DEGRADES temporal stability ({diff_improvement:.6f})")
    else:
        lines.append(f"  → Mamba has MINIMAL effect on temporal stability")
    lines.append("")

    lines.append(f"Detected flicker frames: {flicker_frames}")
    lines.append("")

    lines.append("Part B: FiLM Validity")
    if film_results['variance_decomposition']:
        for t_str, vd in film_results['variance_decomposition'].items():
            lines.append(f"  Frame {t_str}:")
            lines.append(f"    Pre-Mamba  R_affine = {vd['pre_mamba_r_affine']:.4f}")
            lines.append(f"    Post-Mamba R_affine = {vd['post_mamba_r_affine']:.4f}")

        all_r_pre = [v['pre_mamba_r_affine'] for v in film_results['variance_decomposition'].values()]
        all_r_post = [v['post_mamba_r_affine'] for v in film_results['variance_decomposition'].values()]
        avg_r_pre = np.mean(all_r_pre)
        avg_r_post = np.mean(all_r_post)

        lines.append("")
        lines.append(f"  Average R_affine (pre-Mamba):  {avg_r_pre:.4f}")
        lines.append(f"  Average R_affine (post-Mamba): {avg_r_post:.4f}")
        lines.append("")

        if avg_r_post >= 0.8:
            lines.append("  → STRONG support for FiLM modulation (R_affine ≥ 0.8)")
            lines.append("    Channel-wise affine transform explains >80% of feature variance between frames.")
        elif avg_r_post >= 0.5:
            lines.append("  → MODERATE support for FiLM modulation (0.5 ≤ R_affine < 0.8)")
            lines.append("    FiLM can partially correct flickering, but spatial conditioning may also help.")
        else:
            lines.append("  → WEAK support for FiLM alone (R_affine < 0.5)")
            lines.append("    Spatial or more complex modulation may be needed.")
    else:
        lines.append("  (No flicker frames detected for FiLM analysis)")

    lines.append("")
    lines.append("Interpretation:")
    lines.append("  ① Pre-Mamba stable (sim~1.0) → DPT features are per-frame consistent → DPT freeze justified")
    lines.append("  ② Post-Mamba flickering or unresolved → temporal correction needed after Mamba")
    lines.append("  ③ R_affine high → frame differences are mostly channel-wise scale+shift → FiLM is the right tool")
    lines.append("=" * 60)

    summary_path = os.path.join(results_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines))
    logger.info(f"Saved: {summary_path}")

    # Print to console
    print('\n'.join(lines))


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # GPU setup
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device} (GPU {args.gpu})")

    # Output dir
    os.makedirs(args.results_dir, exist_ok=True)

    # 1. Load model
    cfg = load_config(args.config_path)
    model = build_model(cfg, args.checkpoint, device)

    # 2. Monkey-patch for feature capture
    feature_store = {'pre_mamba': [], 'post_mamba': []}
    patch_model_for_capture(model, feature_store)

    # 3. Load data
    sequence = load_test_sequence(args)

    # 4. Inference
    logger.info("Running inference...")
    t0 = time.time()
    depths = run_inference(model, sequence, feature_store, device)
    logger.info(f"Inference done in {time.time() - t0:.1f}s")

    # 5. Detect flickering
    flicker_frames, diffs = detect_flicker_frames(depths, args)

    # If no flicker frames detected, pick top-3 highest-diff frames as fallback
    if not flicker_frames:
        logger.info("No flicker frames detected by threshold. Using top-3 highest-diff frames.")
        top3_idx = np.argsort(diffs)[-3:][::-1]
        flicker_frames = sorted([int(i + 1) for i in top3_idx])
        logger.info(f"Fallback flicker frames: {flicker_frames}")

    # 6. Part A: Temporal stability
    logger.info("Analyzing temporal stability (Part A)...")
    temporal_results = analyze_temporal_stability(feature_store, depths)

    # 7. Part B: FiLM validity
    logger.info("Analyzing FiLM validity (Part B)...")
    film_results = analyze_film_validity(feature_store, flicker_frames)

    # 8. Visualize & save
    logger.info("Generating visualizations...")
    visualize_part_a(temporal_results, diffs, flicker_frames, args.results_dir)
    visualize_part_b(film_results, feature_store, flicker_frames, args.results_dir, args.top_k_channels)
    visualize_flicker_details(feature_store, flicker_frames, depths, args.results_dir)

    # 9. Summary
    write_summary(temporal_results, film_results, flicker_frames, diffs, args.results_dir)

    logger.info(f"All results saved to: {args.results_dir}")


if __name__ == '__main__':
    main()
