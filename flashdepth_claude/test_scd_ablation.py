#!/usr/bin/env python3
"""
SCD Ablation Test: Temporal consistency at scene cut boundaries.

Stitches N segments (--frames-per-seg frames each) from different sequences/datasets
into one video, with known cut points every --frames-per-seg frames.
Runs 4 models and measures rTC (ratio temporal consistency) at cut boundaries vs
within-scene pairs.

Models compared:
  1. Metric-FlashDepth (SCD on)    — Mamba state resets at detected cuts
  2. Metric-FlashDepth (SCD off)   — tau=inf, state carries over cuts
  3. FlashDepth                    — no SCD, Mamba state crosses all cuts
  4. VDA                           — sliding-window, processes full stitched video

Expected result: SCD-on should show lowest rTC AT cut boundaries
(correct behavior: depth changes dramatically at scene cut).

Usage (same-dataset cuts):
  python test_scd_ablation.py \\
    --dataset sintel \\
    --config-variant l \\
    --gear-checkpoint train_results/best.pth \\
    --flashdepth-checkpoint configs/flashdepth-l/iter_10001.pth \\
    --results-dir test_results/scd_ablation \\
    --gpu 0

Usage (cross-dataset cuts):
  python test_scd_ablation.py \\
    --dataset sintel --dataset2 waymo_seg \\
    --config-variant l \\
    --gear-checkpoint train_results/best.pth \\
    --results-dir test_results/scd_ablation_cross \\
    --gpu 0

Docker:
  ./run_docker.sh test_scd_ablation \\
    --dataset sintel --dataset2 waymo_seg \\
    --config-variant l --gear-checkpoint train_results/best.pth \\
    --flashdepth-checkpoint configs/flashdepth-l/iter_10001.pth \\
    --vda-checkpoint path/to/vda.pth --gpu 0
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
    """
    rTC between two depth maps (ratio temporal consistency).
    rTC = fraction of valid pixels where max(d1/d2, d2/d1) < threshold.
    Scale-invariant: works for both metric and inverse depth.
    """
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


def analyze_rtc(depths, cut_frames, pre_win=3, post_win=3, threshold=1.25):
    """
    Compute rTC statistics around cut boundaries.

    Returns dict with:
      per_cut_rtc       — rTC at each cut pair (frame t-1 → t)
      per_cut_pre_rtc   — avg rTC of pre_win pairs before each cut (within-scene)
      per_cut_post_rtc  — avg rTC of post_win pairs after each cut (within-scene)
      avg_cut_rtc       — mean of per_cut_rtc
      avg_pre_cut_rtc   — mean of per_cut_pre_rtc (ignoring NaN)
      avg_post_cut_rtc  — mean of per_cut_post_rtc (ignoring NaN)
      within_scene_rtc  — avg rTC for all non-cut pairs
    """
    T = len(depths)
    cut_set = set(cut_frames)

    cut_rtc, pre_rtc, post_rtc, within_rtc = [], [], [], []

    for t in range(1, T):
        rtc = compute_rtc(depths[t - 1], depths[t], threshold)
        if t in cut_set:
            cut_rtc.append(rtc)
            # Pre: pairs strictly inside the previous segment
            pv = [compute_rtc(depths[t - k - 1], depths[t - k], threshold)
                  for k in range(1, pre_win + 1)
                  if t - k - 1 >= 0 and t - k not in cut_set]
            pre_rtc.append(float(np.nanmean(pv)) if pv else float('nan'))
            # Post: pairs strictly inside the next segment
            pv2 = [compute_rtc(depths[t + k - 1], depths[t + k], threshold)
                   for k in range(1, post_win + 1)
                   if t + k < T and t + k not in cut_set]
            post_rtc.append(float(np.nanmean(pv2)) if pv2 else float('nan'))
        else:
            if not math.isnan(rtc):
                within_rtc.append(rtc)

    def _mean(lst):
        vals = [v for v in lst if not math.isnan(v)]
        return float(np.mean(vals)) if vals else float('nan')

    return {
        'per_cut_rtc':      [round(v, 6) if not math.isnan(v) else None for v in cut_rtc],
        'per_cut_pre_rtc':  [round(v, 6) if not math.isnan(v) else None for v in pre_rtc],
        'per_cut_post_rtc': [round(v, 6) if not math.isnan(v) else None for v in post_rtc],
        'avg_cut_rtc':      round(_mean(cut_rtc), 6),
        'avg_pre_cut_rtc':  round(_mean(pre_rtc), 6),
        'avg_post_cut_rtc': round(_mean(post_rtc), 6),
        'within_scene_rtc': round(_mean(within_rtc), 6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def load_dataset_sequences(dataset_name, data_root, n_seqs, frames_per_seg,
                           resolution='base', strict_focal=False):
    """
    Load n_seqs independent sequences (each frames_per_seg frames) from dataset.
    Returns list of image tensors: each [1, frames_per_seg, 3, H, W], range 0-1 RGB.
    Note: images from CombinedDataset are ImageNet-normalized; we de-normalize here
    so that VDA (expecting 0-1) and visualization are correct.
    """
    dataset = CombinedDataset(
        root_dir=data_root,
        enable_dataset_flags=[dataset_name],
        resolution=resolution,
        split='val',
        video_length=frames_per_seg,
        skip_gt_canonicalization=True,
        strict_focal_length=strict_focal,
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
        images = batch[0].float()         # [1, T, 3, H, W] ImageNet-normalized
        name   = batch[7] if len(batch) > 7 else ('',)
        seq_name = name[0] if isinstance(name, (list, tuple)) else str(name)

        if seq_name in seen_names:        # skip duplicate sequences
            continue
        seen_names.add(seq_name)

        if images.shape[1] < frames_per_seg:
            continue

        # De-normalize to 0-1 for storage (models re-normalize internally or get 0-1)
        images_01 = (images[:, :frames_per_seg] * std + mean).clamp(0, 1)
        seqs.append(images_01)            # [1, fps, 3, H, W]
        if len(seqs) >= n_seqs:
            break

    return seqs


def build_stitched_video(segments):
    """
    Concatenate segments into one stitched video.
    Resizes all segments to the spatial size of the first segment.
    Returns (video [1, T, 3, H, W], cut_frames list).
    """
    H, W = segments[0].shape[3], segments[0].shape[4]
    fps   = segments[0].shape[1]
    resized = []
    for seg in segments:
        if seg.shape[3] != H or seg.shape[4] != W:
            B, T, C, sh, sw = seg.shape
            seg = F.interpolate(
                seg.view(B * T, C, sh, sw), size=(H, W),
                mode='bilinear', align_corners=False
            ).view(B, T, C, H, W)
        resized.append(seg)

    video = torch.cat(resized, dim=1)                   # [1, N*fps, 3, H, W]
    cut_frames = [i * fps for i in range(1, len(segments))]
    return video, cut_frames


def imagenet_normalize(video_01):
    """Apply ImageNet normalization to 0-1 video tensor [1, T, 3, H, W]."""
    mean = torch.tensor(IMAGENET_MEAN, device=video_01.device).view(1, 1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=video_01.device).view(1, 1, 3, 1, 1)
    return (video_01 - mean) / std


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

_VARIANT_CFGS = {
    'l': {'vit_size': 'vitl', 'patch_size': 14, 'use_mamba': False,
          'attn_class': 'MemEffAttention'},
    's': {'vit_size': 'vits', 'patch_size': 14, 'use_mamba': False,
          'attn_class': 'MemEffAttention'},
}


def load_metric_flashdepth(config_variant, checkpoint_path, device,
                           scene_cut_tau=0.05, train_mode='metric'):
    """Load Metric-FlashDepth (Onepiece V3) from checkpoint."""
    cfg = _VARIANT_CFGS.get(config_variant, _VARIANT_CFGS['l'])
    model = FlashDepth(
        **cfg,
        use_metric_head=False,
        use_onepiece=True,
        spatial_mamba_layers=4,
        spatial_mamba_d_state=256,
        spatial_mamba_d_conv=4,
        spatial_mamba_downsample=0.1,
        onepiece_train_mode=train_mode,
        scene_cut_tau=scene_cut_tau,
        scene_cut_k=80,
        batch_size=1,
    )
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
    """
    Run Metric-FlashDepth frame-by-frame on stitched video.

    Args:
        video_01: [1, T, 3, H, W] tensor, range 0-1
        use_scd: if False, set tau=inf to disable scene cut detection
    Returns:
        depths (list of [H,W] tensors), detected_cuts (list of int)
    """
    B, T, C, H, W = video_01.shape

    orig_tau = model.scene_cut_detector.tau
    if not use_scd:
        model.scene_cut_detector.tau = float('inf')

    # ImageNet-normalize for model input
    video_norm = imagenet_normalize(video_01.to(device))

    model.spatial_mamba.start_new_sequence()
    depths, detected_cuts = [], []
    prev_cls = None

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for t in range(T):
            frame = video_norm[:, t]               # [1, 3, H, W]
            out   = model.forward_onepiece_single_frame(frame, prev_cls=prev_cls)
            prev_cls = out['cls_token']
            if out.get('is_reset', False):
                detected_cuts.append(t)
            depths.append(out['metric_depth'].float().squeeze(0).cpu())  # [H, W]

    model.scene_cut_detector.tau = orig_tau
    return depths, detected_cuts


@torch.no_grad()
def run_flashdepth(adapter, video_01, device):
    """
    Run FlashDepth on the full stitched video (Mamba state carries over cuts).
    FlashDepth adapter resets state once at start, then processes frame-by-frame.
    Returns list of [H,W] depth tensors (positive metric depth in meters).
    """
    pred = adapter.inference(video_01.to(device))  # [1, T, H, W] inverse*100
    depths = []
    for t in range(pred.shape[1]):
        inv100 = pred[0, t].float().cpu()
        # Convert inverse*100 → metric depth (meters)
        d = torch.where(inv100 > 1e-3, 100.0 / inv100, torch.zeros_like(inv100))
        depths.append(d)
    return depths


@torch.no_grad()
def run_vda(adapter, video_01, device):
    """
    Run VDA on the full stitched video (sliding window, no explicit cut handling).
    Returns list of [H,W] depth tensors.
    """
    pred = adapter.inference(video_01.to(device))  # [1, T, H, W]
    return [pred[0, t].float().cpu() for t in range(pred.shape[1])]


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def save_cut_pngs(all_depths_by_model, model_keys, video_01, cut_frames, out_dir, window=2):
    """
    For each cut frame, save depth PNG for all models for frames [cut-window, cut+window].

    Output structure:
      out_dir/cut_TTTT/frame_FFFF_<model_key>.png
      out_dir/cut_TTTT/frame_FFFF_image.png
    """
    T = video_01.shape[1]
    out_dir = Path(out_dir)

    for cut_t in cut_frames:
        cut_dir = out_dir / f'cut_{cut_t:04d}'
        cut_dir.mkdir(parents=True, exist_ok=True)

        frame_indices = list(range(max(0, cut_t - window), min(T, cut_t + window + 1)))

        for t in frame_indices:
            label = 'CUT' if t == cut_t else ('BEFORE' if t < cut_t else 'AFTER')

            # Input image
            img_np = video_01[0, t].permute(1, 2, 0).cpu().numpy()  # [H, W, 3] 0-1
            img_bgr = (np.clip(img_np[..., ::-1], 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(str(cut_dir / f'frame_{t:04d}_image_{label}.png'), img_bgr)

            # Depth for each model
            for key, depths in zip(model_keys, all_depths_by_model):
                d_np = depths[t].numpy() if isinstance(depths[t], torch.Tensor) else np.array(depths[t])
                d_vis = colorize_depth(d_np)
                cv2.imwrite(str(cut_dir / f'frame_{t:04d}_{key}_{label}.png'), d_vis)

    logger.info(f"Depth PNGs saved → {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='SCD Ablation Test')
    p.add_argument('--dataset',  default='sintel', help='Primary dataset')
    p.add_argument('--dataset2', default=None,
                   help='Second dataset for cross-dataset cuts (interleaved with --dataset)')
    p.add_argument('--data-root', default='/data/datasets')
    p.add_argument('--config-variant', default='l', choices=['l', 's'],
                   help='Metric-FlashDepth variant (l=ViT-L, s=ViT-S)')
    p.add_argument('--gear-checkpoint', required=True,
                   help='Metric-FlashDepth checkpoint path')
    p.add_argument('--flashdepth-checkpoint', default=None,
                   help='FlashDepth checkpoint path (if omitted, FlashDepth is skipped)')
    p.add_argument('--vda', action='store_true',
                   help='Enable VDA comparison (auto-loads from refer_test/Video-Depth-Anything/checkpoints/)')
    p.add_argument('--frames-per-seg', type=int, default=10,
                   help='Frames per segment (cut every N frames)')
    p.add_argument('--n-segs', type=int, default=5,
                   help='Total number of segments to stitch')
    p.add_argument('--pre-post-window', type=int, default=3,
                   help='Number of within-scene pairs to avg for pre/post-cut rTC')
    p.add_argument('--vis-window', type=int, default=2,
                   help='Frames around each cut to save as PNG (cut ± N)')
    p.add_argument('--results-dir', default='test_results/scd_ablation')
    p.add_argument('--resolution', default='base', choices=['base', '2k'])
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--scene-cut-tau', type=float, default=0.05,
                   help='SCD cosine-distance threshold for SCD-on model')
    p.add_argument('--rtc-threshold', type=float, default=1.25,
                   help='rTC ratio threshold')
    return p.parse_args()


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    fps   = args.frames_per_seg
    n_seg = args.n_segs

    # ── 1. Load segments ─────────────────────────────────────────────────────
    logger.info(f"Loading segments: {n_seg} × {fps} frames")

    if args.dataset2:
        # Interleave: ds1, ds2, ds1, ds2, ...
        half1 = math.ceil(n_seg / 2)
        half2 = n_seg // 2
        segs1 = load_dataset_sequences(args.dataset,  args.data_root, half1, fps, args.resolution)
        segs2 = load_dataset_sequences(args.dataset2, args.data_root, half2, fps, args.resolution)
        if not segs1 or not segs2:
            logger.error("Could not load enough sequences from one or both datasets.")
            return
        segments = []
        for a, b in zip(segs1, segs2):
            segments.extend([a, b])
        if len(segs1) > len(segs2):
            segments.append(segs1[-1])
        scenario = f"cross_dataset_{args.dataset}+{args.dataset2}"
    else:
        segments = load_dataset_sequences(args.dataset, args.data_root, n_seg, fps, args.resolution)
        scenario = f"same_dataset_{args.dataset}"

    if len(segments) < 2:
        logger.error(f"Need ≥2 segments, loaded {len(segments)}. Check dataset and data_root.")
        return

    logger.info(f"Scenario: {scenario} | {len(segments)} segments → {len(segments)*fps} frames")
    video, cut_frames = build_stitched_video(segments)
    logger.info(f"Stitched video: {video.shape} | cut frames: {cut_frames}")

    # ── 2. Load models ───────────────────────────────────────────────────────
    logger.info("Loading Metric-FlashDepth...")
    metric_fd = load_metric_flashdepth(
        args.config_variant, args.gear_checkpoint, device,
        scene_cut_tau=args.scene_cut_tau
    )

    flashdepth_adapter = None
    if args.flashdepth_checkpoint:
        try:
            from adapters.flashdepth_adapter import FlashDepthAdapter
            fd_variant = 'flashdepth-l' if args.config_variant == 'l' else 'flashdepth-s'
            flashdepth_adapter = FlashDepthAdapter(
                config_variant=fd_variant,
                checkpoint_path=args.flashdepth_checkpoint
            )
            flashdepth_adapter.load_model()
            flashdepth_adapter.model = flashdepth_adapter.model.to(device).eval()
            logger.info("Loaded FlashDepth")
        except Exception as e:
            logger.warning(f"FlashDepth failed to load: {e}")
            flashdepth_adapter = None

    vda_adapter = None
    if args.vda:
        try:
            from adapters.video_depth_anything_adapter import VideoDepthAnythingAdapter
            # Auto-loads from refer_test/Video-Depth-Anything/checkpoints/metric_video_depth_anything_vitl.pth
            vda_adapter = VideoDepthAnythingAdapter(metric=True)
            vda_adapter.load_model()
            logger.info("Loaded VDA (metric, vitl, auto-checkpoint)")
        except Exception as e:
            logger.warning(f"VDA failed to load: {e}")
            vda_adapter = None

    # ── 3. Run inference ─────────────────────────────────────────────────────
    all_model_depths = []
    model_keys       = []
    results = {
        'scenario':       scenario,
        'cut_frames':     cut_frames,
        'frames_per_seg': fps,
        'n_segments':     len(segments),
        'total_frames':   int(video.shape[1]),
        'rtc_threshold':  args.rtc_threshold,
        'models':         {},
    }

    def _run_and_record(key, depths, extra=None):
        stats = analyze_rtc(depths, cut_frames,
                            pre_win=args.pre_post_window,
                            post_win=args.pre_post_window,
                            threshold=args.rtc_threshold)
        if extra:
            stats.update(extra)
        results['models'][key] = stats
        all_model_depths.append(depths)
        model_keys.append(key)
        logger.info(
            f"  [{key:<22}] cut rTC={stats['avg_cut_rtc']:.4f} | "
            f"within rTC={stats['within_scene_rtc']:.4f} | "
            f"pre={stats['avg_pre_cut_rtc']:.4f} | post={stats['avg_post_cut_rtc']:.4f}"
        )

    # 1) Metric-FlashDepth SCD on
    logger.info("Running Metric-FlashDepth (SCD on)...")
    d_scd, scd_cuts = run_metric_fd(metric_fd, video, use_scd=True, device=device)
    _run_and_record('metric_fd_scd', d_scd, {'detected_cuts': scd_cuts})

    # 2) Metric-FlashDepth SCD off
    logger.info("Running Metric-FlashDepth (SCD off)...")
    d_noscd, _ = run_metric_fd(metric_fd, video, use_scd=False, device=device)
    _run_and_record('metric_fd_noscd', d_noscd)

    # 3) FlashDepth
    if flashdepth_adapter:
        logger.info("Running FlashDepth (no SCD)...")
        d_fd = run_flashdepth(flashdepth_adapter, video, device=device)
        _run_and_record('flashdepth', d_fd)

    # 4) VDA
    if vda_adapter:
        logger.info("Running VDA (no SCD)...")
        d_vda = run_vda(vda_adapter, video, device=device)
        _run_and_record('vda', d_vda)

    # ── 4. Save JSON ─────────────────────────────────────────────────────────
    json_path = results_dir / 'scd_ablation_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results → {json_path}")

    # ── 5. Save depth PNGs ───────────────────────────────────────────────────
    logger.info("Saving depth PNGs around cut frames...")
    save_cut_pngs(
        all_model_depths, model_keys, video, cut_frames,
        results_dir / 'cuts', window=args.vis_window
    )

    # ── 6. Print summary table ───────────────────────────────────────────────
    print()
    print('=' * 72)
    print(f'SCD ABLATION  —  {scenario}')
    print(f'Cut frames: {cut_frames}')
    print('=' * 72)
    print(f"{'Model':<24} {'Cut rTC':>9} {'Within rTC':>11} {'Pre-cut':>9} {'Post-cut':>9}")
    print('-' * 72)
    for key in ['metric_fd_scd', 'metric_fd_noscd', 'flashdepth', 'vda']:
        if key in results['models']:
            r = results['models'][key]
            print(f"{key:<24} {r['avg_cut_rtc']:>9.4f} {r['within_scene_rtc']:>11.4f} "
                  f"{r['avg_pre_cut_rtc']:>9.4f} {r['avg_post_cut_rtc']:>9.4f}")
    print('=' * 72)
    print()


if __name__ == '__main__':
    main()
