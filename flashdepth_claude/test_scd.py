#!/usr/bin/env python3
"""
Quick SCD (Scene Cut Detection) D_cls analysis.

Runs inference on specified sequences and outputs per-frame D_cls values
and statistics to help tune the tau threshold.

Usage (via run_docker.sh):
    ./run_docker.sh test_onepiece --dataset all --config-variant l
    (but use this script instead of test_onepiece.py)

    CUDA_VISIBLE_DEVICES=0 docker compose run --rm flashdepth python test_scd.py \
        --config-path configs/onepiece --config-name config_l \
        dataset.data_root=/data/datasets \
        +results_dir=test_results/scd_analysis
"""

import json
import logging
import sys

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf, ListConfig
from pathlib import Path
from torch.utils.data import DataLoader

from flashdepth.model import FlashDepth
from dataloaders.combined_dataset import CombinedDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Test targets: (dataset, seq_indices, vid_len)
TEST_TARGETS = [
    ('eth3d', [0, 1], 30),
    ('sintel', [1, 22], 50),
    ('waymo_seg', [0], 200),
    ('vkitti', [0], 200),
    ('unreal4k', [5], 500),
]


def setup_model(config):
    model_config = dict(config.model)
    model_config['batch_size'] = 1
    model_config['use_metric_head'] = False
    model_config['use_onepiece'] = True
    model_config['spatial_mamba_layers'] = config.model.get('spatial_mamba_layers', 4)
    model_config['spatial_mamba_d_state'] = config.model.get('spatial_mamba_d_state', 256)
    model_config['spatial_mamba_d_conv'] = config.model.get('spatial_mamba_d_conv', 4)
    model_config['spatial_mamba_downsample'] = config.model.get('spatial_mamba_downsample', 0.1)
    model_config['onepiece_train_mode'] = config.get('train_mode', 'metric')
    scene_cut_config = config.get('scene_cut', {})
    model_config['scene_cut_tau'] = scene_cut_config.get('tau', 0.05)
    model_config['scene_cut_k'] = scene_cut_config.get('k', 80)

    model = FlashDepth(**model_config)
    ckpt_path = config.get('load', None)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt.get('model', ckpt)
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded checkpoint: {ckpt_path}")

    model = model.cuda().eval()
    return model


def collate_fn(batch):
    """Same collate as test_onepiece.py"""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    if isinstance(batch[0], tuple):
        if len(batch[0]) == 9:
            images, depths, fl_c, fl_a, masks, fx, rr, names, paths = zip(*batch)
            return {
                'image': torch.stack(images, dim=0),
                'depth': torch.stack(depths, dim=0),
                'dataset_name': names,
            }
        elif len(batch[0]) == 8:
            images, depths, fl_c, fl_a, masks, fx, rr, names = zip(*batch)
            return {
                'image': torch.stack(images, dim=0),
                'depth': torch.stack(depths, dim=0),
                'dataset_name': names,
            }
        elif len(batch[0]) == 5:
            images, depths, fl, masks, names = zip(*batch)
            return {
                'image': torch.stack(images, dim=0),
                'depth': torch.stack(depths, dim=0),
                'dataset_name': names,
            }
        else:
            images, depths, fl, names = zip(*batch)
            return {
                'image': torch.stack(images, dim=0),
                'depth': torch.stack(depths, dim=0),
                'dataset_name': names,
            }

    if isinstance(batch[0], dict):
        return batch[0]

    return None


def analyze_sequence(model, images):
    """Run inference and collect D_cls values. images: [1, T, 3, H, W]"""
    T = images.shape[1]
    d_cls_values = []
    reset_frames = []
    prev_cls = None

    model.spatial_mamba.start_new_sequence()

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for t in range(T):
            frame = images[0, t].unsqueeze(0)
            outputs = model.forward_onepiece_single_frame(frame, prev_cls=prev_cls)
            prev_cls = outputs['cls_token']
            d_cls_val = outputs.get('d_cls', 0.0)
            d_cls_values.append(d_cls_val)

            if outputs.get('is_reset', False):
                reset_frames.append(t)

    return d_cls_values, reset_frames


@hydra.main(version_base=None, config_path="configs/onepiece", config_name="config")
def main(config: DictConfig):
    save_dir = Path(config.get('results_dir', 'test_results/scd_analysis'))
    save_dir.mkdir(parents=True, exist_ok=True)

    model = setup_model(config)
    tau = model.scene_cut_detector.tau
    logger.info(f"Current SCD tau: {tau}")

    all_results = {}

    for dataset_name, seq_indices, vid_len in TEST_TARGETS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {dataset_name}, sequences: {seq_indices}, vid_len: {vid_len}")
        logger.info(f"{'='*60}")

        try:
            data_root = config.dataset.get('data_root', '/home/cvlab/hsy/Datasets')
            resolution = config.get('resolution',
                config.eval.get('test_dataset_resolution', 'base'))

            dataset = CombinedDataset(
                root_dir=data_root,
                enable_dataset_flags=[dataset_name],
                resolution=resolution,
                split='test',
                video_length=vid_len,
                skip_gt_canonicalization=True
            )

            loader = DataLoader(
                dataset, batch_size=1, shuffle=False,
                num_workers=1, pin_memory=True,
                collate_fn=collate_fn
            )
            logger.info(f"Dataset loaded: {len(dataset)} sequences")
        except Exception as e:
            logger.error(f"Failed to setup loader for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        max_target = max(seq_indices)
        for seq_id, batch in enumerate(loader):
            if batch is None:
                continue
            if seq_id not in seq_indices:
                if seq_id > max_target:
                    break
                continue

            # Get images
            if 'images' in batch:
                images = batch['images'].cuda()
            elif 'image' in batch:
                images = batch['image'].cuda()
            else:
                logger.error(f"No images in batch for {dataset_name} seq {seq_id}")
                continue

            T = images.shape[1]
            logger.info(f"\n--- {dataset_name} seq {seq_id} ({T} frames) ---")

            d_cls_values, reset_frames = analyze_sequence(model, images)

            # Statistics (skip frame 0, always 0.0)
            d_cls_arr = np.array(d_cls_values)
            d_cls_nonzero = d_cls_arr[1:]

            stats = {
                'mean': float(np.mean(d_cls_nonzero)) if len(d_cls_nonzero) > 0 else 0,
                'std': float(np.std(d_cls_nonzero)) if len(d_cls_nonzero) > 0 else 0,
                'min': float(np.min(d_cls_nonzero)) if len(d_cls_nonzero) > 0 else 0,
                'max': float(np.max(d_cls_nonzero)) if len(d_cls_nonzero) > 0 else 0,
                'p50': float(np.percentile(d_cls_nonzero, 50)) if len(d_cls_nonzero) > 0 else 0,
                'p90': float(np.percentile(d_cls_nonzero, 90)) if len(d_cls_nonzero) > 0 else 0,
                'p95': float(np.percentile(d_cls_nonzero, 95)) if len(d_cls_nonzero) > 0 else 0,
                'p99': float(np.percentile(d_cls_nonzero, 99)) if len(d_cls_nonzero) > 0 else 0,
            }

            # Threshold analysis
            thresholds = [0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
            for thr in thresholds:
                stats[f'exceed_{thr}'] = int(np.sum(d_cls_nonzero > thr))

            result = {
                'dataset': dataset_name,
                'seq_id': seq_id,
                'num_frames': T,
                'tau': tau,
                'reset_frames': reset_frames,
                'num_resets': len(reset_frames),
                'd_cls_values': [float(v) for v in d_cls_values],
                'stats': stats,
            }

            key = f"{dataset_name}_seq{seq_id}"
            all_results[key] = result

            s = stats
            logger.info(f"  D_cls: mean={s['mean']:.4f}, std={s['std']:.4f}, "
                       f"max={s['max']:.4f}, p95={s['p95']:.4f}, p99={s['p99']:.4f}")
            logger.info(f"  Resets (tau={tau}): {len(reset_frames)} at frames {reset_frames}")

            logger.info(f"  Frames exceeding threshold:")
            for thr in thresholds:
                cnt = s[f'exceed_{thr}']
                pct = 100 * cnt / len(d_cls_nonzero) if len(d_cls_nonzero) > 0 else 0
                marker = " <-- current tau" if abs(thr - tau) < 0.001 else ""
                logger.info(f"    tau={thr:.2f}: {cnt:4d} frames ({pct:5.1f}%){marker}")

        torch.cuda.empty_cache()

    # Save results
    out_path = save_dir / "scd_analysis.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")

    # Summary table
    logger.info(f"\n{'='*90}")
    logger.info(f"{'Dataset/Seq':<25} {'Frames':>6} {'Mean':>7} {'Std':>7} {'Max':>7} {'P95':>7} {'P99':>7} {'Resets':>7}")
    logger.info(f"{'-'*90}")
    for key, r in all_results.items():
        s = r['stats']
        logger.info(f"{key:<25} {r['num_frames']:>6} {s['mean']:>7.4f} {s['std']:>7.4f} {s['max']:>7.4f} "
                   f"{s['p95']:>7.4f} {s['p99']:>7.4f} {r['num_resets']:>7}")
    logger.info(f"{'='*90}")
    logger.info(f"Current tau: {tau}")


if __name__ == "__main__":
    main()
