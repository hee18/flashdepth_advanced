#!/usr/bin/env python3
"""
Aggregate UnrealStereo4K test results from multiple sequences

Usage:
    python scripts/aggregate_unreal4k_results.py refer_test/test_results/metric3d/v1/unreal4k
    python scripts/aggregate_unreal4k_results.py refer_test/test_results/unidepth/v2/unreal4k
"""

import json
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def aggregate_results(base_dir):
    """Aggregate results from seq0-seq8 directories"""
    base_path = Path(base_dir)

    if not base_path.exists():
        logger.error(f"Directory not found: {base_dir}")
        return False

    # Collect all sequence results
    all_seq_results = []
    missing_sequences = []

    for seq_idx in range(9):  # seq0 to seq8
        seq_dir = base_path / f"seq{seq_idx}"
        results_file = seq_dir / "test_results.json"

        if results_file.exists():
            logger.info(f"Reading seq{seq_idx}/test_results.json")
            with open(results_file, 'r') as f:
                seq_results = json.load(f)
                all_seq_results.append({
                    'sequence': seq_idx,
                    'metrics': seq_results.get('metrics', seq_results)
                })
        else:
            logger.warning(f"Missing: seq{seq_idx}/test_results.json")
            missing_sequences.append(seq_idx)

    if not all_seq_results:
        logger.error("No sequence results found!")
        return False

    logger.info(f"\nFound {len(all_seq_results)}/9 sequences")
    if missing_sequences:
        logger.warning(f"Missing sequences: {missing_sequences}")

    # Extract method and dataset info from first result
    first_result = all_seq_results[0]['metrics']
    method_name = first_result.get('method', 'unknown')
    dataset_name = first_result.get('dataset', 'unreal4k')

    # Compute average metrics
    logger.info("\nComputing averaged metrics...")
    avg_metrics = {}

    # Get all metric keys from first sequence
    metric_keys = [k for k in first_result.keys()
                   if k not in ['method', 'dataset', 'num_sequences', 'processing_resolution']]

    for key in metric_keys:
        values = []
        for seq_result in all_seq_results:
            metrics = seq_result['metrics']
            if key in metrics:
                value = metrics[key]
                if isinstance(value, (int, float)):
                    values.append(value)

        if values:
            avg_metrics[key] = sum(values) / len(values)
            logger.info(f"  {key}: {avg_metrics[key]:.4f}")

    # Get processing resolution from first result
    processing_resolution = first_result.get('processing_resolution', 'unknown')

    # Create aggregated results
    aggregated_results = {
        'method': method_name,
        'dataset': dataset_name,
        'num_sequences': len(all_seq_results),
        'sequences_tested': [r['sequence'] for r in all_seq_results],
        'processing_resolution': processing_resolution,
        'average_metrics': avg_metrics,
        'per_sequence_results': all_seq_results
    }

    # Save aggregated results
    output_path = base_path / "aggregated_results.json"
    logger.info(f"\nSaving to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(aggregated_results, f, indent=2)

    logger.info("\n" + "="*80)
    logger.info("AGGREGATED RESULTS (Average across all sequences)")
    logger.info("="*80)
    logger.info(f"Method: {method_name}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Sequences: {len(all_seq_results)}/9")
    logger.info(f"Processing Resolution: {processing_resolution}")
    logger.info("-"*80)

    # Print key metrics in order
    key_metrics = ['abs_rel', 'a1', 'a2', 'a3', 'fps', 'tae', 'boundary_f1', 'mae', 'rmse']
    for key in key_metrics:
        if key in avg_metrics:
            logger.info(f"{key:15s}: {avg_metrics[key]:.4f}")

    logger.info("="*80)
    logger.info(f"✓ Aggregated results saved to: {output_path}")

    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/aggregate_unreal4k_results.py <base_dir>")
        print("\nExample:")
        print("  python scripts/aggregate_unreal4k_results.py refer_test/test_results/metric3d/v1/unreal4k")
        print("  python scripts/aggregate_unreal4k_results.py refer_test/test_results/unidepth/v2/unreal4k")
        print("\nThis script will:")
        print("  1. Read test_results.json from seq0/ to seq8/")
        print("  2. Compute averaged metrics across all sequences")
        print("  3. Save aggregated_results.json in the base directory")
        sys.exit(1)

    base_dir = sys.argv[1]
    success = aggregate_results(base_dir)

    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
