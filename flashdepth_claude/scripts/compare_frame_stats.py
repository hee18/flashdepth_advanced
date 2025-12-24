#!/usr/bin/env python3
"""
Compare per-frame statistics (abs_rel, a1) from per_sequence_results.json files.

This script compares results between:
- gear5_mamba (test_results/results_20/gear_5_mamba/{large,hybrid}/...)
- original FlashDepth (test_results/original/{large,hybrid}/.../eval_aligned/)

Output: CSV file with dataset-wise comparison of per-frame min, max, std statistics.
"""

import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


def load_gear5_results(base_dir: Path, datasets: List[str], configs: List[str]) -> List[Dict]:
    """Load results from gear5_mamba test results."""
    results = []

    for config in configs:
        for dataset in datasets:
            # Handle different dataset naming (unreal4k_fix for gear5)
            if dataset == 'unreal4k':
                dataset_dir = 'unreal4k_fix'
            else:
                dataset_dir = dataset

            json_path = base_dir / config / dataset_dir / "per_sequence_results.json"

            if not json_path.exists():
                print(f"Warning: {json_path} not found, skipping...")
                continue

            try:
                with open(json_path) as f:
                    data = json.load(f)

                # Aggregate across all sequences
                abs_rel_mins, abs_rel_maxs, abs_rel_stds = [], [], []
                a1_mins, a1_maxs, a1_stds = [], [], []
                abs_rel_means, a1_means = [], []

                for seq in data:
                    # Check if per-frame stats exist
                    if 'abs_rel_min' in seq:
                        abs_rel_mins.append(seq['abs_rel_min'])
                        abs_rel_maxs.append(seq['abs_rel_max'])
                        abs_rel_stds.append(seq['abs_rel_std'])
                    if 'a1_min' in seq:
                        a1_mins.append(seq['a1_min'])
                        a1_maxs.append(seq['a1_max'])
                        a1_stds.append(seq['a1_std'])

                    # Also collect mean values
                    if 'abs_rel' in seq:
                        abs_rel_means.append(seq['abs_rel'])
                    if 'a1' in seq:
                        a1_means.append(seq['a1'])

                # Create result entry
                result = {
                    'dataset': dataset,
                    'model_type': 'gear5_mamba',
                    'config': config,
                    'num_sequences': len(data)
                }

                # Add abs_rel stats
                if abs_rel_means:
                    result['abs_rel_mean'] = round(np.mean(abs_rel_means), 3)
                if abs_rel_mins:
                    result['abs_rel_min'] = round(np.min(abs_rel_mins), 3)
                if abs_rel_maxs:
                    result['abs_rel_max'] = round(np.max(abs_rel_maxs), 3)
                if abs_rel_stds:
                    result['abs_rel_std'] = round(np.mean(abs_rel_stds), 3)

                # Add a1 stats
                if a1_means:
                    result['a1_mean'] = round(np.mean(a1_means), 3)
                if a1_mins:
                    result['a1_min'] = round(np.min(a1_mins), 3)
                if a1_maxs:
                    result['a1_max'] = round(np.max(a1_maxs), 3)
                if a1_stds:
                    result['a1_std'] = round(np.mean(a1_stds), 3)

                results.append(result)

            except Exception as e:
                print(f"Error loading {json_path}: {e}")

    return results


def load_original_results(base_dir: Path, datasets: List[str], configs: List[str]) -> List[Dict]:
    """Load results from original FlashDepth test results (eval_aligned)."""
    results = []

    for config in configs:
        for dataset in datasets:
            json_path = base_dir / config / dataset / "eval_aligned" / "per_sequence_results.json"

            if not json_path.exists():
                print(f"Warning: {json_path} not found, skipping...")
                continue

            try:
                with open(json_path) as f:
                    data = json.load(f)

                # Aggregate across all sequences
                abs_rel_mins, abs_rel_maxs, abs_rel_stds = [], [], []
                a1_mins, a1_maxs, a1_stds = [], [], []
                abs_rel_means, a1_means = [], []

                for seq in data:
                    # Original format has nested 'metrics' dict
                    metrics = seq.get('metrics', seq)

                    # Check if per-frame stats exist
                    if 'abs_rel_min' in metrics:
                        abs_rel_mins.append(metrics['abs_rel_min'])
                        abs_rel_maxs.append(metrics['abs_rel_max'])
                        abs_rel_stds.append(metrics['abs_rel_std'])
                    if 'a1_min' in metrics:
                        a1_mins.append(metrics['a1_min'])
                        a1_maxs.append(metrics['a1_max'])
                        a1_stds.append(metrics['a1_std'])

                    # Also collect mean values
                    if 'abs_rel' in metrics:
                        abs_rel_means.append(metrics['abs_rel'])
                    if 'a1' in metrics:
                        a1_means.append(metrics['a1'])

                # Create result entry
                result = {
                    'dataset': dataset,
                    'model_type': 'original',
                    'config': config,
                    'num_sequences': len(data)
                }

                # Add abs_rel stats
                if abs_rel_means:
                    result['abs_rel_mean'] = round(np.mean(abs_rel_means), 3)
                if abs_rel_mins:
                    result['abs_rel_min'] = round(np.min(abs_rel_mins), 3)
                if abs_rel_maxs:
                    result['abs_rel_max'] = round(np.max(abs_rel_maxs), 3)
                if abs_rel_stds:
                    result['abs_rel_std'] = round(np.mean(abs_rel_stds), 3)

                # Add a1 stats
                if a1_means:
                    result['a1_mean'] = round(np.mean(a1_means), 3)
                if a1_mins:
                    result['a1_min'] = round(np.min(a1_mins), 3)
                if a1_maxs:
                    result['a1_max'] = round(np.max(a1_maxs), 3)
                if a1_stds:
                    result['a1_std'] = round(np.mean(a1_stds), 3)

                results.append(result)

            except Exception as e:
                print(f"Error loading {json_path}: {e}")

    return results


def write_csv(results: List[Dict], output_path: Path):
    """Write results to CSV file."""
    if not results:
        print("No results to write!")
        return

    # Define column order
    columns = [
        'dataset', 'model_type', 'config', 'num_sequences',
        'abs_rel_mean', 'abs_rel_min', 'abs_rel_max', 'abs_rel_std',
        'a1_mean', 'a1_min', 'a1_max', 'a1_std'
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()

        # Sort by dataset, then config, then model_type
        sorted_results = sorted(results, key=lambda x: (x['dataset'], x['config'], x['model_type']))

        for row in sorted_results:
            writer.writerow(row)

    print(f"Results saved to: {output_path}")


def print_summary(results: List[Dict]):
    """Print summary to console."""
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    # Group by dataset and config
    datasets = sorted(set(r['dataset'] for r in results))
    configs = sorted(set(r['config'] for r in results))

    for dataset in datasets:
        print(f"\n[{dataset.upper()}]")
        for config in configs:
            gear5 = next((r for r in results if r['dataset'] == dataset and r['config'] == config and r['model_type'] == 'gear5_mamba'), None)
            original = next((r for r in results if r['dataset'] == dataset and r['config'] == config and r['model_type'] == 'original'), None)

            if gear5 or original:
                print(f"  Config: {config}")

                if gear5:
                    abs_rel_str = f"mean={gear5.get('abs_rel_mean', 'N/A')}"
                    if 'abs_rel_min' in gear5:
                        abs_rel_str += f", min={gear5['abs_rel_min']}, max={gear5['abs_rel_max']}, std={gear5['abs_rel_std']}"

                    a1_str = f"mean={gear5.get('a1_mean', 'N/A')}"
                    if 'a1_min' in gear5:
                        a1_str += f", min={gear5['a1_min']}, max={gear5['a1_max']}, std={gear5['a1_std']}"

                    print(f"    gear5_mamba: abs_rel({abs_rel_str}), a1({a1_str})")

                if original:
                    abs_rel_str = f"mean={original.get('abs_rel_mean', 'N/A')}"
                    if 'abs_rel_min' in original:
                        abs_rel_str += f", min={original['abs_rel_min']}, max={original['abs_rel_max']}, std={original['abs_rel_std']}"

                    a1_str = f"mean={original.get('a1_mean', 'N/A')}"
                    if 'a1_min' in original:
                        a1_str += f", min={original['a1_min']}, max={original['a1_max']}, std={original['a1_std']}"

                    print(f"    original:    abs_rel({abs_rel_str}), a1({a1_str})")


def main():
    parser = argparse.ArgumentParser(description="Compare per-frame statistics between gear5_mamba and original FlashDepth")
    parser.add_argument('--gear5-dir', type=str, default='test_results/results_20/gear_5_mamba',
                        help='Base directory for gear5_mamba results')
    parser.add_argument('--original-dir', type=str, default='test_results/original',
                        help='Base directory for original FlashDepth results')
    parser.add_argument('--output', type=str, default='comparison_frame_stats.csv',
                        help='Output CSV file path')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['eth3d', 'sintel', 'waymo_seg', 'vkitti', 'unreal4k'],
                        help='Datasets to compare')
    parser.add_argument('--configs', type=str, nargs='+',
                        default=['large', 'hybrid'],
                        help='Config variants to compare')

    args = parser.parse_args()

    # Convert to Path objects
    gear5_dir = Path(args.gear5_dir)
    original_dir = Path(args.original_dir)
    output_path = Path(args.output)

    print(f"Loading gear5_mamba results from: {gear5_dir}")
    gear5_results = load_gear5_results(gear5_dir, args.datasets, args.configs)
    print(f"  Found {len(gear5_results)} result entries")

    print(f"\nLoading original FlashDepth results from: {original_dir}")
    original_results = load_original_results(original_dir, args.datasets, args.configs)
    print(f"  Found {len(original_results)} result entries")

    # Combine results
    all_results = gear5_results + original_results

    # Write CSV
    write_csv(all_results, output_path)

    # Print summary
    print_summary(all_results)


if __name__ == '__main__':
    main()
