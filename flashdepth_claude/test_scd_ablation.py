#!/usr/bin/env python3
"""
SCD Ablation Test: Temporal consistency at scene cut boundaries.

Creates "host + insert + host" test videos where an alien segment is inserted
at the midpoint of a host sequence. The model should detect and reset at the two
known scene cuts (insert start / insert end).

Two scenarios:
  1. Same-dataset: for each host seq i, insert = seq (i+2) % N (from same dataset)
  2. Cross-dataset: host from one dataset, insert from another
       - dataset2=sintel  → fixed insert: Sintel seq 13, frames 21-30 (for all host seqs)
       - dataset=sintel, dataset2=eth3d → fixed insert: ETH3D seq 1, frames 0-9 (for all Sintel seqs)
       - other combos    → generic (insert seq k for host seq k)

Host video length is dataset-specific (matches test_onepiece defaults):
  eth3d=30, sintel=50, waymo(_seg)=200, vkitti=200, unreal4k=500

Models compared (all configured via CLI flags):
  1. Metric-FlashDepth (SCD on)  — Mamba state resets at detected cuts
  2. Metric-FlashDepth (SCD off) — tau=inf, state carries over cuts
  3. FlashDepth                  — no SCD, Mamba state crosses all cuts (optional)
  4. VDA                         — sliding-window (optional)

Docker usage:
  # Same-dataset (ETH3D sequences with cyclic i+2 inserts)
  ./run_docker.sh test_scd_ablation \\
    --dataset eth3d \\
    --config-variant l \\
    --gear-checkpoint train_results/results_34/onepiece/large/best.pth \\
    --gpu 0

  # Cross-dataset: ETH3D base + fixed Sintel insert
  ./run_docker.sh test_scd_ablation \\
    --dataset eth3d --dataset2 sintel \\
    --config-variant l \\
    --gear-checkpoint train_results/results_34/onepiece/large/best.pth \\
    --gpu 0

  # Cross-dataset: Sintel base + fixed ETH3D seq1 insert
  ./run_docker.sh test_scd_ablation \\
    --dataset sintel --dataset2 eth3d \\
    --config-variant hybrid \\
    --gear-checkpoint train_results/results_34/onepiece/hybrid/best.pth \\
    --gpu 0
"""

import os
import sys
import json
import math
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from torch.utils.data import DataLoader

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dataloaders.combined_dataset import CombinedDataset
from flashdepth.model import FlashDepth

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Per-dataset host video length (matches test_onepiece / _onepiece_vid_len defaults)
DATASET_VID_LEN = {
    'eth3d':     30,
    'sintel':    50,
    'waymo':     200,
    'waymo_seg': 200,
    'vkitti':    200,
    'unreal4k':  500,
}

# Fixed Sintel insert: used when dataset2=sintel (any non-Sintel base)
SINTEL_FIXED_INSERT_SEQ   = 13   # Sintel sequence index (0-based)
SINTEL_FIXED_INSERT_START = 21   # Start frame (inclusive)
SINTEL_FIXED_INSERT_LEN   = 10   # Number of frames to insert

# Fixed ETH3D insert: used when dataset=sintel, dataset2=eth3d
ETH3D_FIXED_INSERT_SEQ   = 1    # ETH3D sequence index (0-based)
ETH3D_FIXED_INSERT_START = 0    # Start frame
ETH3D_FIXED_INSERT_LEN   = 10   # Number of frames to insert

# Onepiece config files per variant
_VARIANT_CONFIG_FILES = {
    'l':      'configs/onepiece/config_l.yaml',
    's':      'configs/onepiece/config_s.yaml',
    'hybrid': 'configs/onepiece/config_hybrid.yaml',
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def colorize_depth(depth_np, vmin=None, vmax=None):
    """Convert float depth (H,W) → colorized BGR uint8 (H,W,3) with MAGMA cmap."""
    valid = depth_np > 0
    if valid.sum() == 0:
        return np.zeros((*depth_np.shape, 3), dtype=np.uint8)
    v0 = vmin if vmin is not None else float(np.percentile(depth_np[valid], 2))
    v1 = vmax if vmax is not None else float(np.percentile(depth_np[valid], 98))
    norm = np.clip((depth_np - v0) / (v1 - v0 + 1e-8), 0, 1)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)


def compute_rtc(d1, d2, threshold=1.25):
    """rTC between two depth maps: fraction of valid pixels where max(d1/d2, d2/d1) < threshold."""
    if isinstance(d1, np.ndarray):
        d1 = torch.from_numpy(d1.astype(np.float32))
    if isinstance(d2, np.ndarray):
        d2 = torch.from_numpy(d2.astype(np.float32))
    valid = (d1 > 0) & (d2 > 0)
    if valid.sum() == 0:
        return float('nan')
    a, b = d1[valid].float(), d2[valid].float()
    ratio = torch.maximum(a / b, b / a)
    return float((ratio < threshold).float().mean())


def analyze_insertion_rtc(depths, cut_frames, pre_win=3, post_win=3, threshold=1.25):
    """Compute rTC metrics for host+insert+host structure.

    cut_frames must be [mid, mid+ins_len] (exactly 2 cuts):
      - Position 0..mid-1        : host frames (first half)
      - Position mid..mid+ins-1  : insert frames
      - Position mid+ins..T-1    : host frames (second half)

    Key metrics:
      cut_in_rtc          rTC(depth[mid-1], depth[mid])           host→insert boundary
      cut_out_rtc         rTC(depth[mid+ins-1], depth[mid+ins])   insert→host boundary
      host_continuity_rtc rTC(depth[mid-1], depth[mid+ins])       adjacent host frames
                          across the insert (e.g. ETH3D frame 14 vs ETH3D frame 15)
      host_within_rtc     avg rTC of consecutive pairs within host segments
      insert_within_rtc   avg rTC of consecutive pairs within insert segment
      pre_cut_rtc         avg rTC of pre_win pairs before cut_in (host baseline)
      post_cut_rtc        avg rTC of post_win pairs after cut_out (host baseline)
    """
    assert len(cut_frames) == 2, "analyze_insertion_rtc expects exactly 2 cut frames"
    mid     = cut_frames[0]   # index of first insert frame
    cut_out = cut_frames[1]   # index of first host frame after insert
    T       = len(depths)

    def _r(v):
        return round(float(v), 6) if not math.isnan(float(v)) else None

    cut_in_rtc          = compute_rtc(depths[mid - 1],     depths[mid],     threshold)
    cut_out_rtc         = compute_rtc(depths[cut_out - 1], depths[cut_out], threshold)
    host_continuity_rtc = compute_rtc(depths[mid - 1],     depths[cut_out], threshold)

    # Within-host pairs (both host halves, excluding the two cut boundaries)
    host_pairs = []
    for t in range(1, mid):
        host_pairs.append(compute_rtc(depths[t - 1], depths[t], threshold))
    for t in range(cut_out + 1, T):
        host_pairs.append(compute_rtc(depths[t - 1], depths[t], threshold))

    # Within-insert pairs
    ins_pairs = [compute_rtc(depths[t - 1], depths[t], threshold)
                 for t in range(mid + 1, cut_out)]

    # Baseline: pre_win pairs just before cut_in (within host)
    pre_pairs = [compute_rtc(depths[t - 1], depths[t], threshold)
                 for t in range(max(1, mid - pre_win), mid)]

    # Baseline: post_win pairs just after cut_out (within host)
    post_pairs = [compute_rtc(depths[t - 1], depths[t], threshold)
                  for t in range(cut_out + 1, min(T, cut_out + post_win + 1))]

    def _mean(lst):
        vals = [v for v in lst if not math.isnan(v)]
        return float(np.mean(vals)) if vals else float('nan')

    return {
        'cut_in_rtc':           _r(cut_in_rtc),
        'cut_out_rtc':          _r(cut_out_rtc),
        'host_continuity_rtc':  _r(host_continuity_rtc),
        'host_within_rtc':      _r(_mean(host_pairs)),
        'insert_within_rtc':    _r(_mean(ins_pairs)),
        'pre_cut_rtc':          _r(_mean(pre_pairs)),
        'post_cut_rtc':         _r(_mean(post_pairs)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def load_all_sequences(dataset_name, data_root, vid_len, resolution='base', max_seqs=None):
    """Load unique sequences from dataset using test split; de-normalized to 0-1.

    Uses split='test' to match test_onepiece behavior and get full sequences.
    Accepts sequences shorter than vid_len (uses all available frames).

    Returns list of [1, T, 3, H, W] tensors where T <= vid_len (CPU, float32).
    """
    dataset = CombinedDataset(
        root_dir=data_root,
        enable_dataset_flags=[dataset_name],
        resolution=resolution,
        split='test',
        video_length=vid_len,
        skip_gt_canonicalization=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_skip_none,
    )

    mean = torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)

    seqs = []
    seen_names = set()
    for batch in loader:
        if batch is None:
            continue
        images = batch[0].float()  # [1, T, 3, H, W] ImageNet-normalized
        name = batch[7] if len(batch) > 7 else ('',)
        seq_name = name[0] if isinstance(name, (list, tuple)) else str(name)
        if seq_name in seen_names:
            continue
        seen_names.add(seq_name)
        T = images.shape[1]
        if T < 4:  # need at least 2 frames on each side of midpoint
            continue
        images_01 = (images * std + mean).clamp(0, 1)  # keep all available frames
        seqs.append(images_01)
        if max_seqs is not None and len(seqs) >= max_seqs:
            break

    logger.info(f"Loaded {len(seqs)} sequences from {dataset_name} "
                f"(vid_len={vid_len}, split=test)")
    return seqs


def build_insertion_video(host_seq, insert_seq, insert_start=0, insert_len=10):
    """Build: [host[:mid]] + [insert[start:start+len]] + [host[mid:]].

    Args:
        host_seq:   [1, T, 3, H, W] — host video (0-1)
        insert_seq: [1, T2, 3, H2, W2] — insert source; resized to host dims if needed
        insert_start: first frame index in insert_seq to use
        insert_len:   number of insert frames

    Returns:
        video [1, T+insert_len, 3, H, W], cut_frames [mid, mid+insert_len]
    """
    H, W = host_seq.shape[3], host_seq.shape[4]
    mid  = host_seq.shape[1] // 2

    insert = insert_seq[:, insert_start:insert_start + insert_len]
    if insert.shape[3] != H or insert.shape[4] != W:
        B, T, C, ih, iw = insert.shape
        insert = F.interpolate(
            insert.view(B * T, C, ih, iw), size=(H, W),
            mode='bilinear', align_corners=False
        ).view(B, T, C, H, W)

    video = torch.cat([host_seq[:, :mid], insert, host_seq[:, mid:]], dim=1)
    return video, [mid, mid + insert_len]


def imagenet_normalize(video_01):
    """Apply ImageNet normalization to [1, T, 3, H, W] tensor (0-1 → normalized)."""
    mean = torch.tensor(IMAGENET_MEAN, device=video_01.device).view(1, 1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=video_01.device).view(1, 1, 3, 1, 1)
    return (video_01 - mean) / std


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_metric_flashdepth(config_variant, checkpoint_path, device,
                           scene_cut_tau=0.10, train_mode='metric'):
    """Load Metric-FlashDepth (Onepiece V3) from onepiece config + checkpoint."""
    from omegaconf import OmegaConf

    cfg_path = project_root / _VARIANT_CONFIG_FILES[config_variant]
    if not cfg_path.exists():
        raise FileNotFoundError(f"Onepiece config not found: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)

    # Build model kwargs from config, with test-time overrides
    model_kwargs = dict(cfg.model)
    model_kwargs.update({
        'batch_size':          1,
        'training':            False,
        'use_metric_head':     False,
        'onepiece_train_mode': train_mode,
        'scene_cut_tau':       scene_cut_tau,
        'scene_cut_k':         int(cfg.scene_cut.k),
    })

    model = FlashDepth(hybrid_configs=cfg.hybrid_configs, **model_kwargs)

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        sd = ckpt.get('model', ckpt.get('state_dict', ckpt))
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        logger.info(f"Loaded Metric-FlashDepth ({config_variant}) from {checkpoint_path}")
    else:
        logger.warning(f"Metric-FlashDepth checkpoint not found: {checkpoint_path}")

    return model.to(device).eval()


# ─────────────────────────────────────────────────────────────────────────────
# Inference runners
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_metric_fd(model, video_01, use_scd, device):
    """Run Metric-FlashDepth frame-by-frame on test video.

    Returns (depths: list of [H,W] CPU tensors, detected_cuts: list of int).
    """
    orig_tau = model.scene_cut_detector.tau
    if not use_scd:
        model.scene_cut_detector.tau = float('inf')

    video_norm = imagenet_normalize(video_01.to(device))
    model.spatial_mamba.start_new_sequence()
    depths, detected_cuts = [], []
    prev_patch_mean = None

    d_patch_log = []  # (t, d_patch_mean_val) for SCD-on debug
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for t in range(video_norm.shape[1]):
            frame = video_norm[:, t]
            out = model.forward_onepiece_single_frame(frame, prev_patch_mean=prev_patch_mean)
            prev_patch_mean = out['patch_mean']
            if out.get('is_reset', False):
                detected_cuts.append(t)
            if use_scd and out.get('d_patch_mean', 0.0) > 0:
                d_patch_log.append((t, round(float(out['d_patch_mean']), 4)))
            depths.append(out['metric_depth'].float().squeeze(0).cpu())

    if use_scd and d_patch_log:
        top = sorted(d_patch_log, key=lambda x: -x[1])[:5]
        logger.info(f"  [SCD d_patch_mean top-5] {top}  tau={orig_tau}  detected={detected_cuts}")

    model.scene_cut_detector.tau = orig_tau
    return depths, detected_cuts


@torch.no_grad()
def run_flashdepth(adapter, video_01, device):
    """Run FlashDepth (no SCD). Returns list of [H,W] depth tensors."""
    pred = adapter.inference(video_01.to(device))  # [1, T, H, W] inverse*100
    depths = []
    for t in range(pred.shape[1]):
        inv100 = pred[0, t].float().cpu()
        d = torch.where(inv100 > 1e-3, 100.0 / inv100, torch.zeros_like(inv100))
        depths.append(d)
    return depths


@torch.no_grad()
def run_vda(adapter, video_01, device):
    """Run VDA (no SCD). Returns list of [H,W] depth tensors."""
    pred = adapter.inference(video_01.to(device))
    return [pred[0, t].float().cpu() for t in range(pred.shape[1])]


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def save_cut_pngs(all_depths_by_model, model_keys, video_01, cut_frames, out_dir, window=2):
    """Save depth PNGs for window frames around each cut, for all models."""
    T = video_01.shape[1]
    out_dir = Path(out_dir)

    for cut_t in cut_frames:
        cut_dir = out_dir / f'cut_{cut_t:04d}'
        cut_dir.mkdir(parents=True, exist_ok=True)
        frame_indices = list(range(max(0, cut_t - window), min(T, cut_t + window + 1)))

        for t in frame_indices:
            label = 'CUT' if t == cut_t else ('BEFORE' if t < cut_t else 'AFTER')
            img_np = video_01[0, t].permute(1, 2, 0).cpu().numpy()
            img_bgr = (np.clip(img_np[..., ::-1], 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(str(cut_dir / f'frame_{t:04d}_image_{label}.png'), img_bgr)
            for key, depths in zip(model_keys, all_depths_by_model):
                d_np = depths[t].numpy() if isinstance(depths[t], torch.Tensor) else np.array(depths[t])
                cv2.imwrite(str(cut_dir / f'frame_{t:04d}_{key}_{label}.png'), colorize_depth(d_np))

    logger.info(f"Cut PNGs → {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Single test runner
# ─────────────────────────────────────────────────────────────────────────────

def run_single_test(test_id, video, cut_frames, metric_fd,
                    flashdepth_adapter, vda_adapter, args, out_dir):
    """Run all models on one test video, save per-test JSON + cut PNGs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = next(metric_fd.parameters()).device

    all_model_depths = []
    model_keys       = []
    results = {
        'test_id':      test_id,
        'cut_frames':   cut_frames,
        'total_frames': int(video.shape[1]),
        'rtc_threshold': args.rtc_threshold,
        'models':       {},
    }

    def _record(key, depths, extra=None):
        stats = analyze_insertion_rtc(depths, cut_frames,
                                      pre_win=args.pre_post_window,
                                      post_win=args.pre_post_window,
                                      threshold=args.rtc_threshold)
        if extra:
            stats.update(extra)
        results['models'][key] = stats
        all_model_depths.append(depths)
        model_keys.append(key)

        def _fmt(v):
            return f'{v:.4f}' if v is not None else '  nan'

        logger.info(
            f"  [{key:<22}] "
            f"cut_in={_fmt(stats['cut_in_rtc'])}  "
            f"cut_out={_fmt(stats['cut_out_rtc'])}  "
            f"host_cont={_fmt(stats['host_continuity_rtc'])}  "
            f"host_within={_fmt(stats['host_within_rtc'])}"
        )

    logger.info("Running Metric-FlashDepth (SCD on)...")
    d_scd, scd_cuts = run_metric_fd(metric_fd, video, use_scd=True, device=device)
    _record('metric_fd_scd', d_scd, {'detected_cuts': scd_cuts})

    logger.info("Running Metric-FlashDepth (SCD off)...")
    d_noscd, _ = run_metric_fd(metric_fd, video, use_scd=False, device=device)
    _record('metric_fd_noscd', d_noscd)

    if flashdepth_adapter:
        logger.info("Running FlashDepth (no SCD)...")
        d_fd = run_flashdepth(flashdepth_adapter, video, device=device)
        _record('flashdepth', d_fd)

    if vda_adapter:
        logger.info("Running VDA (no SCD)...")
        d_vda = run_vda(vda_adapter, video, device=device)
        _record('vda', d_vda)

    with open(out_dir / 'result.json', 'w') as f:
        json.dump(results, f, indent=2)

    save_cut_pngs(all_model_depths, model_keys, video, cut_frames,
                  out_dir / 'cuts', window=args.vis_window)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation & summary
# ─────────────────────────────────────────────────────────────────────────────

_METRIC_FIELDS = [
    'cut_in_rtc', 'cut_out_rtc', 'host_continuity_rtc',
    'host_within_rtc', 'insert_within_rtc', 'pre_cut_rtc', 'post_cut_rtc',
]

_MODEL_KEYS = ['metric_fd_scd', 'metric_fd_noscd', 'flashdepth', 'vda']


def aggregate_results(per_test_results, scenario):
    """Average insertion rTC metrics across all test cases."""
    agg = {'scenario': scenario, 'n_tests': len(per_test_results), 'models': {}}

    for key in _MODEL_KEYS:
        stats_list = [r['models'][key] for r in per_test_results
                      if key in r.get('models', {})]
        if not stats_list:
            continue

        def _nanmean(field):
            vals = [s[field] for s in stats_list
                    if s.get(field) is not None and not math.isnan(float(s[field]))]
            return round(float(np.mean(vals)), 6) if vals else float('nan')

        agg['models'][key] = {f: _nanmean(f) for f in _METRIC_FIELDS}

    return agg


def print_summary(agg):
    """Print results table with insertion-specific metrics."""
    W = 100
    print()
    print('=' * W)
    print(f"SCD ABLATION  —  {agg['scenario']}  ({agg['n_tests']} tests)")
    print('=' * W)
    header = (f"{'Model':<22} {'cut_in':>8} {'cut_out':>8} "
              f"{'host_cont':>10} {'host_w':>8} {'ins_w':>8} "
              f"{'pre_cut':>8} {'post_cut':>9}")
    print(header)
    print('-' * W)
    for key in _MODEL_KEYS:
        if key not in agg['models']:
            continue
        r = agg['models'][key]

        def _f(k):
            v = r.get(k)
            return f'{v:.4f}' if v is not None and not math.isnan(float(v)) else '   nan'

        print(f"{key:<22} {_f('cut_in_rtc'):>8} {_f('cut_out_rtc'):>8} "
              f"{_f('host_continuity_rtc'):>10} {_f('host_within_rtc'):>8} "
              f"{_f('insert_within_rtc'):>8} {_f('pre_cut_rtc'):>8} "
              f"{_f('post_cut_rtc'):>9}")
    print('=' * W)
    print()
    print("Key:  cut_in = host→insert  |  cut_out = insert→host  |  "
          "host_cont = adjacent host frames across insert")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='SCD Ablation Test')
    p.add_argument('--dataset', default='sintel',
                   help='Base/host dataset (eth3d, sintel, waymo_seg, vkitti, unreal4k)')
    p.add_argument('--dataset2', default=None,
                   help='Insert dataset for cross-dataset mode. Omit for same-dataset mode.')
    p.add_argument('--data-root', default='/data/datasets')
    p.add_argument('--config-variant', default='l', choices=['l', 's', 'hybrid'],
                   help='Metric-FlashDepth variant (l=ViT-L, s=ViT-S, hybrid=ViT-S+L)')
    p.add_argument('--gear-checkpoint', required=True,
                   help='Metric-FlashDepth (Onepiece) checkpoint path')
    p.add_argument('--flashdepth-checkpoint', default=None,
                   help='Base FlashDepth checkpoint for comparison (optional)')
    p.add_argument('--vda', action='store_true',
                   help='Enable VDA comparison (auto-loads from refer_test/)')
    p.add_argument('--insert-frames', type=int, default=10,
                   help='Number of frames inserted at host midpoint (default: 10)')
    p.add_argument('--max-seqs', type=int, default=None,
                   help='Limit host sequences per scenario (default: all)')
    p.add_argument('--pre-post-window', type=int, default=3,
                   help='Within-scene pairs to avg for pre/post-cut rTC')
    p.add_argument('--vis-window', type=int, default=2,
                   help='Frames around each cut to save as PNG (cut ± N)')
    p.add_argument('--results-dir', default='test_results/scd_ablation')
    p.add_argument('--resolution', default='base', choices=['base', '2k'])
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--scene-cut-tau', type=float, default=0.10,
                   help='SCD patch-mean distance threshold (default: 0.10, matches config)')
    p.add_argument('--rtc-threshold', type=float, default=1.25,
                   help='rTC ratio threshold')
    return p.parse_args()


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset   = args.dataset.lower()
    dataset2  = args.dataset2.lower() if args.dataset2 else None
    ins_len   = args.insert_frames
    host_vlen = DATASET_VID_LEN.get(dataset, 50)

    # ── 1. Load models ─────────────────────────────────────────────────────
    logger.info(f"Loading Metric-FlashDepth ({args.config_variant})...")
    metric_fd = load_metric_flashdepth(
        args.config_variant, args.gear_checkpoint, device,
        scene_cut_tau=args.scene_cut_tau,
    )

    flashdepth_adapter = None
    if args.flashdepth_checkpoint:
        try:
            from adapters.flashdepth_adapter import FlashDepthAdapter
            _fd = FlashDepthAdapter(
                config_variant=args.config_variant,
                checkpoint_path=args.flashdepth_checkpoint,
            )
            _fd.load_model()
            _fd.model = _fd.model.to(device).eval()
            flashdepth_adapter = _fd
            logger.info("Loaded FlashDepth adapter")
        except Exception as e:
            logger.warning(f"FlashDepth failed to load: {e}")
            flashdepth_adapter = None

    vda_adapter = None
    if args.vda:
        try:
            from adapters.video_depth_anything_adapter import VideoDepthAnythingAdapter
            _vda = VideoDepthAnythingAdapter(metric=True)
            _vda.load_model()
            _vda.model = _vda.model.to(device).eval()
            _vda.device = device
            vda_adapter = _vda
            logger.info("Loaded VDA")
        except Exception as e:
            logger.warning(f"VDA failed to load: {e}")
            vda_adapter = None

    per_test_results = []

    # ── 2. Same-dataset mode ────────────────────────────────────────────────
    if dataset2 is None:
        scenario = f'same_dataset_{dataset}'
        logger.info(f"Scenario: {scenario}  (host_vlen={host_vlen}, insert_len={ins_len})")

        seqs = load_all_sequences(dataset, args.data_root, host_vlen,
                                  args.resolution, max_seqs=args.max_seqs)
        if len(seqs) < 3:
            logger.error(f"Need ≥3 sequences for same-dataset test (got {len(seqs)})")
            return
        N = len(seqs)

        for i, host_seq in enumerate(seqs):
            insert_idx = (i + 2) % N
            video, cut_frames = build_insertion_video(
                host_seq, seqs[insert_idx], insert_start=0, insert_len=ins_len
            )
            logger.info(
                f"Test {i+1}/{N}: host=seq{i:02d} insert=seq{insert_idx:02d}  "
                f"frames={video.shape[1]} cuts={cut_frames}"
            )
            result = run_single_test(
                i, video, cut_frames, metric_fd, flashdepth_adapter, vda_adapter,
                args, results_dir / f'seq{i:04d}'
            )
            per_test_results.append(result)

    # ── 3. Cross-dataset: any-base + Sintel insert (fixed seq 13 frames 21-30) ─
    elif dataset2 == 'sintel' and dataset != 'sintel':
        scenario = f'cross_{dataset}_base_sintel_insert'
        logger.info(
            f"Scenario: {scenario}  "
            f"insert=Sintel seq{SINTEL_FIXED_INSERT_SEQ} "
            f"frames {SINTEL_FIXED_INSERT_START}-{SINTEL_FIXED_INSERT_START+ins_len-1}"
        )

        host_seqs = load_all_sequences(dataset, args.data_root, host_vlen,
                                       args.resolution, max_seqs=args.max_seqs)
        sintel_vlen = max(SINTEL_FIXED_INSERT_START + ins_len, DATASET_VID_LEN['sintel'])
        sintel_seqs = load_all_sequences('sintel', args.data_root, sintel_vlen,
                                         args.resolution,
                                         max_seqs=SINTEL_FIXED_INSERT_SEQ + 1)
        if len(sintel_seqs) <= SINTEL_FIXED_INSERT_SEQ:
            logger.error(
                f"Could not load Sintel seq {SINTEL_FIXED_INSERT_SEQ} "
                f"(only {len(sintel_seqs)} available)"
            )
            return
        sintel_insert = sintel_seqs[SINTEL_FIXED_INSERT_SEQ]

        N = len(host_seqs)
        for i, host_seq in enumerate(host_seqs):
            video, cut_frames = build_insertion_video(
                host_seq, sintel_insert,
                insert_start=SINTEL_FIXED_INSERT_START,
                insert_len=ins_len,
            )
            logger.info(
                f"Test {i+1}/{N}: host={dataset} seq{i:02d} + Sintel insert  "
                f"frames={video.shape[1]} cuts={cut_frames}"
            )
            result = run_single_test(
                i, video, cut_frames, metric_fd, flashdepth_adapter, vda_adapter,
                args, results_dir / f'seq{i:04d}'
            )
            per_test_results.append(result)

    # ── 4. Cross-dataset: Sintel base + fixed ETH3D seq1 insert ───────────
    elif dataset == 'sintel' and dataset2 == 'eth3d':
        scenario = 'cross_sintel_base_eth3d_insert'
        logger.info(
            f"Scenario: {scenario}  "
            f"insert=ETH3D seq{ETH3D_FIXED_INSERT_SEQ} "
            f"frames {ETH3D_FIXED_INSERT_START}-{ETH3D_FIXED_INSERT_START+ins_len-1} (fixed)"
        )

        sintel_seqs = load_all_sequences('sintel', args.data_root, host_vlen,
                                         args.resolution, max_seqs=args.max_seqs)
        eth3d_seqs  = load_all_sequences('eth3d', args.data_root, DATASET_VID_LEN['eth3d'],
                                         args.resolution,
                                         max_seqs=ETH3D_FIXED_INSERT_SEQ + 1)
        if len(eth3d_seqs) <= ETH3D_FIXED_INSERT_SEQ:
            logger.error(
                f"Could not load ETH3D seq {ETH3D_FIXED_INSERT_SEQ} "
                f"(only {len(eth3d_seqs)} available)"
            )
            return
        eth3d_insert = eth3d_seqs[ETH3D_FIXED_INSERT_SEQ]

        N = len(sintel_seqs)
        for i, host_seq in enumerate(sintel_seqs):
            video, cut_frames = build_insertion_video(
                host_seq, eth3d_insert,
                insert_start=ETH3D_FIXED_INSERT_START,
                insert_len=ins_len,
            )
            logger.info(
                f"Test {i+1}/{N}: Sintel seq{i:02d} + ETH3D seq{ETH3D_FIXED_INSERT_SEQ} insert  "
                f"frames={video.shape[1]} cuts={cut_frames}"
            )
            result = run_single_test(
                i, video, cut_frames, metric_fd, flashdepth_adapter, vda_adapter,
                args, results_dir / f'seq{i:04d}'
            )
            per_test_results.append(result)

    # ── 5. Generic cross-dataset ────────────────────────────────────────────
    else:
        scenario = f'cross_{dataset}_base_{dataset2}_insert'
        logger.info(f"Scenario: {scenario}  (generic pairing)")

        ins_vlen  = DATASET_VID_LEN.get(dataset2, 50)
        host_seqs = load_all_sequences(dataset,  args.data_root, host_vlen,
                                       args.resolution, max_seqs=args.max_seqs)
        ins_seqs  = load_all_sequences(dataset2, args.data_root, ins_vlen,
                                       args.resolution, max_seqs=len(host_seqs))
        if not host_seqs or not ins_seqs:
            logger.error("Could not load sequences from one or both datasets")
            return
        N = len(host_seqs)

        for i, host_seq in enumerate(host_seqs):
            ins_seq = ins_seqs[i % len(ins_seqs)]
            video, cut_frames = build_insertion_video(
                host_seq, ins_seq, insert_start=0, insert_len=ins_len
            )
            logger.info(
                f"Test {i+1}/{N}: {dataset} seq{i:02d} + {dataset2} seq{i%len(ins_seqs):02d}  "
                f"frames={video.shape[1]} cuts={cut_frames}"
            )
            result = run_single_test(
                i, video, cut_frames, metric_fd, flashdepth_adapter, vda_adapter,
                args, results_dir / f'seq{i:04d}'
            )
            per_test_results.append(result)

    # ── 6. Aggregate & save ─────────────────────────────────────────────────
    if not per_test_results:
        logger.error("No test results to aggregate.")
        return

    agg = aggregate_results(per_test_results, scenario)

    summary_path = results_dir / 'scd_ablation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(agg, f, indent=2)
    logger.info(f"Summary → {summary_path}")

    all_path = results_dir / 'scd_ablation_all_tests.json'
    with open(all_path, 'w') as f:
        json.dump({'scenario': scenario, 'tests': per_test_results}, f, indent=2)
    logger.info(f"All tests → {all_path}")

    print_summary(agg)


if __name__ == '__main__':
    main()
