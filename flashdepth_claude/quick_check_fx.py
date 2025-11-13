#!/usr/bin/env python3
"""
Quick check of fx_actual from dataloader vs _get_actual_focal_length
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.dataset_intrinsics import get_intrinsics_info, get_fallback_fx

def check_dataset(dataset_name, width):
    """Check what _get_actual_focal_length would return"""
    dataset_name = dataset_name.lower().replace('-', '_')

    intrinsics_info = get_intrinsics_info(dataset_name)

    if intrinsics_info is None:
        fx = get_fallback_fx(width)
        print(f"{dataset_name}: No intrinsics → fallback fx={fx:.1f}")
        return fx

    if intrinsics_info['type'] == 'fixed':
        fx = intrinsics_info['fx']
        print(f"{dataset_name}: Fixed fx={fx:.1f}")
        return fx

    if intrinsics_info['type'] == 'computed':
        if dataset_name in ['dynamicreplica', 'replica']:
            fx = width / 2.0
            print(f"{dataset_name}: Computed fx={fx:.1f}")
            return fx
        else:
            fx = get_fallback_fx(width)
            print(f"{dataset_name}: Computed fallback fx={fx:.1f}")
            return fx

    if 'typical_fx' in intrinsics_info:
        fx = intrinsics_info['typical_fx']
        print(f"{dataset_name}: Typical fx={fx:.1f}")
        return fx

    fx = get_fallback_fx(width)
    print(f"{dataset_name}: No typical_fx → fallback fx={fx:.1f}")
    return fx


print("="*80)
print("Checking what _get_actual_focal_length returns for each dataset")
print("="*80)

# Sintel (val-base target: 1022×434)
print("\nSINTEL (width=1022):")
sintel_fx = check_dataset('sintel', 1022)
print(f"  BUT: Actual values are per-frame (seq0=688, seq4=1120, seq7=800)")
print(f"  Discrepancy for seq0: |{sintel_fx:.1f} - 688| = {abs(sintel_fx - 688):.1f}")
print(f"  Discrepancy for seq4: |{sintel_fx:.1f} - 1120| = {abs(sintel_fx - 1120):.1f}")

# Waymo_seg (val-base target: 784×518)
print("\nWAYMO_SEG (width=784):")
waymo_fx = check_dataset('waymo_seg', 784)
print(f"  Using typical_fx from registry")
print(f"  BUT: Actual values are per-sequence (may vary)")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("For Sintel:")
print(f"  - _get_actual_focal_length returns: {sintel_fx:.1f} (fallback)")
print(f"  - Actual values range: 688 - 1120")
print(f"  - Max error: {abs(sintel_fx - 1120):.1f} pixels ({abs(sintel_fx - 1120) / 1120 * 100:.1f}%)")
print("")
print("For Waymo_seg:")
print(f"  - _get_actual_focal_length returns: {waymo_fx:.1f} (typical)")
print(f"  - Actual values: per-sequence (unknown if 2059 is always accurate)")
print("")
print("RECOMMENDATION:")
print("  ✅ Use batch['focal_lengths_actual'] or compute from batch['fx_ratio']")
print("  ❌ DON'T use _get_actual_focal_length(dataset_name) for these datasets")
