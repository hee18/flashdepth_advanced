#!/usr/bin/env python3
"""
Measure multiple SCD signals at cut-in / cut-out boundaries.

Signals compared:
  d_deep       : CLS cosine dist, layers [17, 23]  (current method)
  d_patch_mean : cosine dist of mean patch tokens, layer 4  (candidate 2)
  d_shallow_cls: CLS cosine dist, layer 4  (candidate 3 component)
  d_multiscale : 0.5 * d_shallow_cls + 0.5 * d_deep  (candidate 3)

Does NOT run full inference — only DINOv2 encoder forward on 4 frames per test.

Usage:
  python measure_boundary_dcls.py \
    --gear-checkpoint train_results/results_34/onepiece/large/best.pth \
    --config-variant l \
    --data-root /data/datasets \
    --gpu 1
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from test_scd_ablation import (
    load_all_sequences, build_insertion_video,
    DATASET_VID_LEN, SINTEL_FIXED_INSERT_SEQ, SINTEL_FIXED_INSERT_START,
    ETH3D_FIXED_INSERT_SEQ, ETH3D_FIXED_INSERT_START,
    _VARIANT_CONFIG_FILES,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# d_deep: CLS layers [17,23] — current baseline
# d_patch_lN: mean patch tokens from layer N — candidates
SIGNAL_KEYS = ['d_deep', 'd_patch_l4', 'd_patch_l11', 'd_patch_l17', 'd_patch_l23']


# ─────────────────────────────────────────────────────────────────────────────

def load_encoder_only(config_variant, checkpoint_path, device):
    from omegaconf import OmegaConf
    from flashdepth.model import FlashDepth

    cfg_path = project_root / _VARIANT_CONFIG_FILES[config_variant]
    cfg = OmegaConf.load(cfg_path)

    model_kwargs = dict(cfg.model)
    model_kwargs.update({
        'batch_size': 1,
        'training': False,
        'use_metric_head': False,
        'scene_cut_tau': 0.10,
        'scene_cut_k': int(cfg.scene_cut.k),
    })
    model = FlashDepth(hybrid_configs=cfg.hybrid_configs, **model_kwargs)

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        sd = ckpt.get('model', ckpt.get('state_dict', ckpt))
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path} — using random weights")

    return model.to(device).eval()


@torch.no_grad()
def get_features(model, frame_01, device):
    """Extract SCD signals from a single frame [1,3,H,W] (0-1 range).

    For ViT-L (config_l): extracted layers = [4, 11, 17, 23]
      normed_outputs[0] → layer 4,  [1] → layer 11,  [2] → layer 17,  [3] → layer 23

    Returns dict with:
      cls_deep        : [B, D] — avg CLS from model.cls_layer_indices (layers 17, 23)
      patch_mean_lN   : [B, D] — mean patch tokens from layer N (N=4,11,17,23)
    """
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=device).view(1, 3, 1, 1)
    frame_norm = (frame_01.to(device) - mean) / std

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        if model.use_onepiece_hybrid and model.use_teacher_cls:
            pretrained = model.teacher_model.pretrained
            layer_idx  = model.intermediate_layer_idx['vitl']
        else:
            pretrained = model.pretrained
            layer_idx  = model.intermediate_layer_idx[model.encoder]

        raw_outputs    = pretrained._get_intermediate_layers_not_chunked(frame_norm, layer_idx)
        normed_outputs = [pretrained.norm(o) for o in raw_outputs]
        num_reg        = pretrained.num_register_tokens

        # deep CLS: average over cls_layer_indices (e.g. [2,3] → layers [17,23])
        selected_deep = [normed_outputs[i][:, 0] for i in model.cls_layer_indices]
        cls_deep      = torch.stack(selected_deep, dim=0).mean(dim=0).float()

        # patch means for all 4 extracted layers
        patch_means = {}
        layer_names = [4, 11, 17, 23]  # ViT-L; ViT-S would be [2,5,8,11]
        for idx, lname in enumerate(layer_names):
            if idx < len(normed_outputs):
                patches = normed_outputs[idx][:, 1 + num_reg:]    # [B, N, D]
                patch_means[f'patch_mean_l{lname}'] = patches.mean(dim=1).float()

    return {'cls_deep': cls_deep, **patch_means}


def _cdist(a, b):
    """Cosine distance between two [B, D] tensors."""
    return float(1.0 - F.cosine_similarity(
        F.normalize(a, dim=-1), F.normalize(b, dim=-1), dim=-1
    ).mean())


def measure_video(model, video_01, cut_frames, device):
    """Compute all SCD signals at cut_in and cut_out boundaries."""
    mid, cut_out = cut_frames

    feats = {}
    for t in [mid - 1, mid, cut_out - 1, cut_out]:
        feats[t] = get_features(model, video_01[0, t].unsqueeze(0), device)

    def _signals(fa, fb):
        return {
            'd_deep':        round(_cdist(fa['cls_deep'],         fb['cls_deep']),         4),
            'd_patch_l4':    round(_cdist(fa['patch_mean_l4'],   fb['patch_mean_l4']),    4),
            'd_patch_l11':   round(_cdist(fa['patch_mean_l11'],  fb['patch_mean_l11']),   4),
            'd_patch_l17':   round(_cdist(fa['patch_mean_l17'],  fb['patch_mean_l17']),   4),
            'd_patch_l23':   round(_cdist(fa['patch_mean_l23'],  fb['patch_mean_l23']),   4),
        }

    return {
        'cut_in':  _signals(feats[mid - 1],     feats[mid]),
        'cut_out': _signals(feats[cut_out - 1], feats[cut_out]),
    }


# ─────────────────────────────────────────────────────────────────────────────

def _scenario_summary_dict(name, results, taus=(0.10, 0.15, 0.20)):
    """Build summary dict for JSON saving."""
    summary = {'scenario': name, 'n_tests': len(results), 'signals': {}}
    for key in SIGNAL_KEYS:
        ins  = [r['cut_in'][key]  for r in results]
        outs = [r['cut_out'][key] for r in results]
        summary['signals'][key] = {
            'cut_in_avg':  round(sum(ins) / len(ins), 4),
            'cut_in_max':  round(max(ins), 4),
            'cut_out_avg': round(sum(outs) / len(outs), 4),
            'cut_out_max': round(max(outs), 4),
            'detectable': {
                f'tau_{tau:.2f}': sum(
                    (r['cut_in'][key] > tau) or (r['cut_out'][key] > tau)
                    for r in results
                )
                for tau in taus
            },
        }
    return summary


def _print_scenario_summary(name, results, taus=(0.10, 0.15, 0.20)):
    print(f"\n{'='*72}")
    print(f"Scenario : {name}  (N={len(results)})")
    print(f"{'─'*72}")

    header = f"  {'signal':<16}  {'cut_in avg':>10}  {'cut_in max':>10}  "
    header += f"{'cut_out avg':>11}  {'cut_out max':>11}"
    print(header)
    print(f"  {'─'*66}")

    for key in SIGNAL_KEYS:
        ins  = [r['cut_in'][key]  for r in results]
        outs = [r['cut_out'][key] for r in results]
        print(f"  {key:<16}  {sum(ins)/len(ins):>10.4f}  {max(ins):>10.4f}  "
              f"{sum(outs)/len(outs):>11.4f}  {max(outs):>11.4f}")

    print(f"{'─'*72}")
    print(f"  Detectability (either boundary exceeds tau):")
    for tau in taus:
        row = f"  tau={tau:.2f} : "
        for key in SIGNAL_KEYS:
            n = sum(
                (r['cut_in'][key] > tau) or (r['cut_out'][key] > tau)
                for r in results
            )
            row += f"  {key}={n}/{len(results)}"
        print(row)
    print('='*72)


def measure_consecutive_pairs(model, seq, device):
    """Compute all signals for every consecutive frame pair within a single sequence.

    Args:
        seq: [1, T, 3, H, W] tensor (0-1)
    Returns:
        list of signal dicts, one per pair (t-1, t)
    """
    T = seq.shape[1]
    pairs = []
    prev_feat = get_features(model, seq[0, 0].unsqueeze(0), device)
    for t in range(1, T):
        curr_feat = get_features(model, seq[0, t].unsqueeze(0), device)
        pairs.append({
            'd_deep':      round(_cdist(prev_feat['cls_deep'],        curr_feat['cls_deep']),        4),
            'd_patch_l4':  round(_cdist(prev_feat['patch_mean_l4'],  curr_feat['patch_mean_l4']),  4),
            'd_patch_l11': round(_cdist(prev_feat['patch_mean_l11'], curr_feat['patch_mean_l11']), 4),
            'd_patch_l17': round(_cdist(prev_feat['patch_mean_l17'], curr_feat['patch_mean_l17']), 4),
            'd_patch_l23': round(_cdist(prev_feat['patch_mean_l23'], curr_feat['patch_mean_l23']), 4),
        })
        prev_feat = curr_feat
    return pairs


def _consecutive_summary_dict(name, all_pairs, taus=(0.10, 0.15, 0.20)):
    import statistics
    summary = {'scenario': name, 'n_pairs': len(all_pairs), 'signals': {}}
    for key in SIGNAL_KEYS:
        vals = [p[key] for p in all_pairs]
        sorted_vals = sorted(vals)
        n = len(sorted_vals)
        summary['signals'][key] = {
            'avg':  round(sum(vals) / n, 4),
            'max':  round(max(vals), 4),
            'p90':  round(sorted_vals[int(n * 0.90)], 4),
            'p95':  round(sorted_vals[int(n * 0.95)], 4),
            'p99':  round(sorted_vals[int(n * 0.99)], 4),
            'exceed_tau': {
                f'tau_{tau:.2f}': sum(v > tau for v in vals)
                for tau in taus
            },
        }
    return summary


def _print_consecutive_summary(name, all_pairs, taus=(0.10, 0.15, 0.20)):
    import statistics
    n = len(all_pairs)
    print(f"\n{'='*72}")
    print(f"Consecutive frames : {name}  (N pairs={n})")
    print(f"{'─'*72}")
    print(f"  {'signal':<16}  {'avg':>8}  {'p90':>8}  {'p95':>8}  {'p99':>8}  {'max':>8}")
    print(f"  {'─'*60}")
    for key in SIGNAL_KEYS:
        vals = sorted(p[key] for p in all_pairs)
        avg = sum(vals) / n
        print(f"  {key:<16}  {avg:>8.4f}  "
              f"{vals[int(n*0.90)]:>8.4f}  "
              f"{vals[int(n*0.95)]:>8.4f}  "
              f"{vals[int(n*0.99)]:>8.4f}  "
              f"{max(vals):>8.4f}")
    print(f"{'─'*72}")
    print(f"  False positive rate (consecutive frames exceeding tau):")
    for tau in taus:
        row = f"  tau={tau:.2f} : "
        for key in SIGNAL_KEYS:
            cnt = sum(p[key] > tau for p in all_pairs)
            row += f"  {key}={cnt}/{n} ({100*cnt/n:.1f}%)"
        print(row)
    print('='*72)


def run_consecutive_scenario(name, seqs, model, device):
    """Measure consecutive-frame signals across all sequences."""
    all_pairs = []
    for i, seq in enumerate(seqs):
        pairs = measure_consecutive_pairs(model, seq, device)
        all_pairs.extend(pairs)
        logger.info(f"  [consec {name[:20]}] seq{i:02d}  "
                    f"T={seq.shape[1]}  pairs={len(pairs)}  "
                    f"l4_max={max(p['d_patch_l4'] for p in pairs):.3f}")
    _print_consecutive_summary(name, all_pairs)
    return all_pairs, _consecutive_summary_dict(name, all_pairs)



def run_scenario(name, host_seqs, insert_seqs, ins_start, ins_len, model, device):
    results = []
    for i, host_seq in enumerate(host_seqs):
        ins_seq = insert_seqs[i % len(insert_seqs)]
        video, cut_frames = build_insertion_video(
            host_seq, ins_seq, insert_start=ins_start, insert_len=ins_len
        )
        r = measure_video(model, video, cut_frames, device)
        results.append(r)
        logger.info(
            f"  [{name[:28]}] seq{i:02d}"
            f"  deep=({r['cut_in']['d_deep']:.3f},{r['cut_out']['d_deep']:.3f})"
            f"  l4=({r['cut_in']['d_patch_l4']:.3f},{r['cut_out']['d_patch_l4']:.3f})"
            f"  l11=({r['cut_in']['d_patch_l11']:.3f},{r['cut_out']['d_patch_l11']:.3f})"
            f"  l17=({r['cut_in']['d_patch_l17']:.3f},{r['cut_out']['d_patch_l17']:.3f})"
            f"  l23=({r['cut_in']['d_patch_l23']:.3f},{r['cut_out']['d_patch_l23']:.3f})"
        )

    if results:
        _print_scenario_summary(name, results)
    return results, _scenario_summary_dict(name, results)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gear-checkpoint', required=True)
    p.add_argument('--config-variant', default='l', choices=['l', 's', 'hybrid'])
    p.add_argument('--data-root', default='/data/datasets')
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--max-seqs', type=int, default=None)
    p.add_argument('--insert-frames', type=int, default=10)
    p.add_argument('--resolution', default='base')
    p.add_argument('--output-dir', default='test_results/boundary_dcls',
                   help='Directory to save JSON results (default: test_results/boundary_dcls)')
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda:0')
    ins_len = args.insert_frames

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model encoder...")
    model = load_encoder_only(args.config_variant, args.gear_checkpoint, device)

    all_summaries = []

    # ── 1. cross: ETH3D host + fixed Sintel insert ───────────────────────────
    logger.info("\n[Loading ETH3D sequences]")
    eth3d_seqs = load_all_sequences('eth3d', args.data_root,
                                    DATASET_VID_LEN['eth3d'], args.resolution,
                                    max_seqs=args.max_seqs)
    sintel_vlen = max(SINTEL_FIXED_INSERT_START + ins_len, DATASET_VID_LEN['sintel'])
    logger.info("[Loading Sintel sequences]")
    sintel_seqs = load_all_sequences('sintel', args.data_root,
                                     sintel_vlen, args.resolution,
                                     max_seqs=SINTEL_FIXED_INSERT_SEQ + 1)

    if eth3d_seqs and len(sintel_seqs) > SINTEL_FIXED_INSERT_SEQ:
        _, s = run_scenario(
            'cross: ETH3D host + Sintel insert',
            eth3d_seqs, [sintel_seqs[SINTEL_FIXED_INSERT_SEQ]],
            SINTEL_FIXED_INSERT_START, ins_len, model, device,
        )
        all_summaries.append(s)

    # ── 2. cross: Sintel host + fixed ETH3D insert ───────────────────────────
    eth3d_all = load_all_sequences('eth3d', args.data_root,
                                   DATASET_VID_LEN['eth3d'], args.resolution,
                                   max_seqs=ETH3D_FIXED_INSERT_SEQ + 1)
    sintel_host_seqs = load_all_sequences('sintel', args.data_root,
                                          DATASET_VID_LEN['sintel'], args.resolution,
                                          max_seqs=args.max_seqs)

    if sintel_host_seqs and len(eth3d_all) > ETH3D_FIXED_INSERT_SEQ:
        _, s = run_scenario(
            'cross: Sintel host + ETH3D insert',
            sintel_host_seqs, [eth3d_all[ETH3D_FIXED_INSERT_SEQ]],
            ETH3D_FIXED_INSERT_START, ins_len, model, device,
        )
        all_summaries.append(s)

    # ── 3. same: ETH3D cyclic (baseline) ─────────────────────────────────────
    if len(eth3d_seqs) >= 3:
        N = len(eth3d_seqs)
        results = []
        for i, h in enumerate(eth3d_seqs):
            ins = eth3d_seqs[(i + 2) % N]
            video, cut_frames = build_insertion_video(h, ins, insert_start=0, insert_len=ins_len)
            r = measure_video(model, video, cut_frames, device)
            results.append(r)
        _print_scenario_summary('same ETH3D (cyclic i+2)', results)
        all_summaries.append(_scenario_summary_dict('same ETH3D (cyclic i+2)', results))

    # ── 4. same: Sintel cyclic (reference) ───────────────────────────────────
    if len(sintel_host_seqs) >= 3:
        N = len(sintel_host_seqs)
        results = []
        for i, h in enumerate(sintel_host_seqs):
            ins = sintel_host_seqs[(i + 2) % N]
            video, cut_frames = build_insertion_video(h, ins, insert_start=0, insert_len=ins_len)
            r = measure_video(model, video, cut_frames, device)
            results.append(r)
        _print_scenario_summary('same Sintel (cyclic i+2) — reference', results)
        all_summaries.append(_scenario_summary_dict('same Sintel (cyclic i+2) — reference', results))

    # ── 5. consecutive frames within ETH3D (false positive baseline) ─────────
    logger.info("\n[Consecutive frames — ETH3D (no scene cut)]")
    if eth3d_seqs:
        _, s = run_consecutive_scenario('ETH3D consecutive', eth3d_seqs, model, device)
        all_summaries.append(s)

    # ── 6. consecutive frames within Sintel (false positive baseline) ─────────
    logger.info("\n[Consecutive frames — Sintel (no scene cut)]")
    if sintel_host_seqs:
        _, s = run_consecutive_scenario('Sintel consecutive', sintel_host_seqs, model, device)
        all_summaries.append(s)

    # ── 7. consecutive frames within Unreal4K — 1 sequence ───────────────────
    logger.info("\n[Consecutive frames — Unreal4K seq0 (no scene cut)]")
    unreal4k_seqs = load_all_sequences('unreal4k', args.data_root,
                                       DATASET_VID_LEN['unreal4k'], args.resolution,
                                       max_seqs=1)
    if unreal4k_seqs:
        _, s = run_consecutive_scenario('Unreal4K consecutive (seq0)', unreal4k_seqs, model, device)
        all_summaries.append(s)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = out_dir / 'boundary_dcls_summary.json'
    with open(out_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    logger.info(f"Results saved → {out_path}")
    logger.info("Done.")


if __name__ == '__main__':
    main()
