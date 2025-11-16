#!/usr/bin/env python3
"""Debug script to test object-wise evaluation flow"""
import sys
sys.path.insert(0, '.')

import argparse
import torch
from pathlib import Path

print("="*80)
print("Testing Object-Wise Configuration and Data Loading")
print("="*80)

# Simulate argument parsing like in test_comparison.py
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='vkitti')
parser.add_argument('--objwise', action='store_true', default=True)
args = parser.parse_args([])

# Create config like in main()
config = {
    'dataset': 'vkitti',
    'data_root': '/home/cvlab/hsy/Datasets',
    'workers': 4,
    'video_length': 50,
    'object_wise': {
        'enabled': True,  # Simulating --objwise flag
        'dataset': 'vkitti'
    },
    'only_clone': True
}

print("\nConfig:")
print(f"  dataset: {config['dataset']}")
print(f"  object_wise: {config['object_wise']}")
print(f"  object_wise.enabled: {config.get('object_wise', {}).get('enabled', False)}")

# Test ComparisonTester setup
print("\n" + "="*80)
print("Simulating ComparisonTester Initialization")
print("="*80)

object_wise_enabled = config.get('object_wise', {}).get('enabled', False)
object_wise_dataset = config.get('object_wise', {}).get('dataset', 'waymo')
dataset_name = config.get('dataset', 'waymo')

print(f"\nobject_wise_enabled: {object_wise_enabled}")
print(f"object_wise_dataset: {object_wise_dataset}")
print(f"dataset_name: {dataset_name}")

if object_wise_enabled:
    print(f"✓ Object-wise evaluation ENABLED for {object_wise_dataset}")
else:
    print("✗ Object-wise evaluation DISABLED")

# Test dataset loading
print("\n" + "="*80)
print("Testing Dataset Loading")
print("="*80)

from dataloaders.comparison_dataset import ComparisonDataset, comparison_collate_fn

dataset = ComparisonDataset(
    dataset_name=dataset_name,
    data_root=config['data_root'],
    split='test',
    video_length=config['video_length'],
    objwise_enabled=object_wise_enabled,
    only_clone=config.get('only_clone', False)
)

print(f"\nDataset created:")
print(f"  Total sequences: {len(dataset)}")
print(f"  dataset.objwise_enabled: {dataset.objwise_enabled}")

# Load first batch
if len(dataset) > 0:
    print("\nLoading first sequence...")
    batch = dataset[0]

    print(f"\nBatch keys: {list(batch.keys())}")

    has_segmentations = 'segmentations' in batch
    print(f"\nhas_segmentations: {has_segmentations}")

    if has_segmentations:
        print(f"  ✓ Segmentations shape: {batch['segmentations'].shape}")
        print(f"  ✓ Unique classes: {torch.unique(batch['segmentations'])}")
    else:
        print("  ✗ No segmentations in batch!")

    # Test the condition used in test_sequence
    compute_regular_metrics = not object_wise_enabled or (object_wise_enabled and not has_segmentations)
    compute_objwise_metrics = object_wise_enabled and has_segmentations

    print(f"\n" + "="*80)
    print("Evaluation Path Selection")
    print("="*80)
    print(f"compute_regular_metrics: {compute_regular_metrics}")
    print(f"compute_objwise_metrics: {compute_objwise_metrics}")

    if compute_objwise_metrics:
        print("\n✓ Object-wise metrics WILL BE COMPUTED")
    else:
        print("\n✗ Regular metrics will be computed instead")
        if not object_wise_enabled:
            print("   Reason: object_wise_enabled is False")
        elif not has_segmentations:
            print("   Reason: segmentations not in batch")

print("\n" + "="*80)
