"""
Verify DPT/ViT/output_conv freeze strategy by comparing checkpoint parameters.

Compares:
1. best.pth (step 0) vs last.pth: depth_head (DPT) and pretrained (ViT) params
2. best.pth vs checkpoint_step5000.pth: output_conv params (should be frozen in Phase 1)
3. checkpoint_step5000.pth vs last.pth: output_conv params (should change in Phase 2)
"""

import torch
import sys
import os
from collections import defaultdict


def load_checkpoint(path):
    """Load checkpoint and return model state dict."""
    print(f"Loading: {path}")
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if 'model' in ckpt:
        return ckpt['model']
    elif 'state_dict' in ckpt:
        return ckpt['state_dict']
    else:
        return ckpt


def categorize_params(state_dict):
    """Categorize parameters into groups."""
    groups = defaultdict(list)
    for key in state_dict.keys():
        if key.startswith('pretrained.'):
            groups['pretrained (ViT encoder)'].append(key)
        elif key.startswith('depth_head.'):
            groups['depth_head (DPT)'].append(key)
        elif key.startswith('output_conv.') or key.startswith('output_conv1.') or key.startswith('output_conv2.'):
            groups['output_conv'].append(key)
        elif 'mamba' in key.lower():
            groups['mamba (temporal)'].append(key)
        elif key.startswith('gsp_head.') or key.startswith('metric_head.'):
            groups['metric head (GSP)'].append(key)
        elif key.startswith('onepiece') or key.startswith('scale_predictor') or key.startswith('shift_predictor'):
            groups['onepiece modules'].append(key)
        else:
            groups['other'].append(key)
    return groups


def compare_params(state_dict_a, state_dict_b, param_keys, name_a, name_b):
    """Compare parameters between two state dicts."""
    changed = []
    unchanged = []
    missing_in_a = []
    missing_in_b = []
    
    for key in param_keys:
        if key not in state_dict_a:
            missing_in_a.append(key)
            continue
        if key not in state_dict_b:
            missing_in_b.append(key)
            continue
        
        param_a = state_dict_a[key]
        param_b = state_dict_b[key]
        
        if param_a.shape != param_b.shape:
            changed.append((key, f"Shape mismatch: {param_a.shape} vs {param_b.shape}"))
            continue
        
        if torch.equal(param_a, param_b):
            unchanged.append(key)
        else:
            max_diff = (param_a.float() - param_b.float()).abs().max().item()
            mean_diff = (param_a.float() - param_b.float()).abs().mean().item()
            changed.append((key, f"max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}"))
    
    return changed, unchanged, missing_in_a, missing_in_b


def print_comparison(group_name, changed, unchanged, missing_a, missing_b, name_a, name_b, show_details=False):
    """Print comparison results for a parameter group."""
    total = len(changed) + len(unchanged)
    
    if total == 0 and not missing_a and not missing_b:
        print(f"  [SKIP] {group_name}: No parameters found")
        return
    
    if len(changed) == 0 and total > 0:
        status = "FROZEN (all identical)"
        symbol = "FROZEN"
    elif len(unchanged) == 0 and total > 0:
        status = "ALL CHANGED"
        symbol = "CHANGED"
    else:
        status = f"PARTIALLY CHANGED ({len(changed)}/{total} changed)"
        symbol = "PARTIAL"
    
    print(f"  [{symbol}] {group_name}: {status}")
    print(f"           Parameters: {len(unchanged)} unchanged, {len(changed)} changed (out of {total})")
    
    if missing_a:
        print(f"           Missing in {name_a}: {len(missing_a)}")
        if len(missing_a) <= 5:
            for k in missing_a:
                print(f"             - {k}")
    if missing_b:
        print(f"           Missing in {name_b}: {len(missing_b)}")
        if len(missing_b) <= 5:
            for k in missing_b:
                print(f"             - {k}")
    
    if show_details and changed:
        print(f"           Changed parameters (first 10):")
        for key, info in changed[:10]:
            short_key = key if len(key) < 80 else "..." + key[-77:]
            print(f"             - {short_key}: {info}")
        if len(changed) > 10:
            print(f"             ... and {len(changed) - 10} more")
    
    if show_details and unchanged and len(unchanged) <= 5:
        print(f"           Unchanged parameters:")
        for key in unchanged:
            print(f"             - {key}")


def main():
    # Auto-detect base directory (works in both Docker /app and host paths)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "train_results/results_29/onepiece/large")
    
    print(f"Script directory: {script_dir}")
    print(f"Checkpoint directory: {base_dir}")
    
    # Verify files exist
    for fname in ['best.pth', 'checkpoint_step5000.pth', 'last.pth']:
        fpath = os.path.join(base_dir, fname)
        if not os.path.exists(fpath):
            print(f"ERROR: {fpath} not found!")
            sys.exit(1)
    
    print("=" * 80)
    print("LOADING CHECKPOINTS")
    print("=" * 80)
    
    best = load_checkpoint(f"{base_dir}/best.pth")
    step5000 = load_checkpoint(f"{base_dir}/checkpoint_step5000.pth")
    last = load_checkpoint(f"{base_dir}/last.pth")
    
    print(f"\nbest.pth keys: {len(best)}")
    print(f"checkpoint_step5000.pth keys: {len(step5000)}")
    print(f"last.pth keys: {len(last)}")
    
    # Show parameter groups
    print("\n" + "=" * 80)
    print("PARAMETER GROUP OVERVIEW (from last.pth)")
    print("=" * 80)
    
    groups = categorize_params(last)
    for group_name, keys in sorted(groups.items()):
        total_params = sum(last[k].numel() for k in keys)
        print(f"  {group_name}: {len(keys)} tensors, {total_params:,} parameters")
    
    groups_best = categorize_params(best)
    print("\nParameter groups in best.pth:")
    for group_name, keys in sorted(groups_best.items()):
        total_params = sum(best[k].numel() for k in keys)
        print(f"  {group_name}: {len(keys)} tensors, {total_params:,} parameters")
    
    # Use the union of all keys across checkpoints for each group
    all_keys = set(best.keys()) | set(step5000.keys()) | set(last.keys())
    all_groups = categorize_params({k: torch.tensor(0) for k in all_keys})
    
    # ============================================================
    # Comparison 1: best.pth vs last.pth (full training span)
    # ============================================================
    print("\n" + "=" * 80)
    print("COMPARISON 1: best.pth vs last.pth (full training: step 0 -> final)")
    print("Expected: pretrained (ViT) and depth_head (DPT) should be FROZEN")
    print("=" * 80)
    
    for group_name in ['pretrained (ViT encoder)', 'depth_head (DPT)', 'output_conv', 
                        'mamba (temporal)', 'onepiece modules', 'metric head (GSP)', 'other']:
        keys = all_groups.get(group_name, [])
        if not keys:
            continue
        changed, unchanged, miss_a, miss_b = compare_params(best, last, keys, 'best.pth', 'last.pth')
        print_comparison(group_name, changed, unchanged, miss_a, miss_b, 'best.pth', 'last.pth', show_details=True)
    
    # ============================================================
    # Comparison 2: best.pth vs checkpoint_step5000.pth (Phase 1)
    # ============================================================
    print("\n" + "=" * 80)
    print("COMPARISON 2: best.pth vs checkpoint_step5000.pth (Phase 1: step 0 -> 5000)")
    print("Expected: output_conv should be FROZEN in Phase 1")
    print("=" * 80)
    
    for group_name in ['pretrained (ViT encoder)', 'depth_head (DPT)', 'output_conv',
                        'mamba (temporal)', 'onepiece modules', 'metric head (GSP)', 'other']:
        keys = all_groups.get(group_name, [])
        if not keys:
            continue
        changed, unchanged, miss_a, miss_b = compare_params(best, step5000, keys, 'best.pth', 'step5000.pth')
        print_comparison(group_name, changed, unchanged, miss_a, miss_b, 'best.pth', 'step5000.pth', show_details=True)
    
    # ============================================================
    # Comparison 3: checkpoint_step5000.pth vs last.pth (Phase 2)
    # ============================================================
    print("\n" + "=" * 80)
    print("COMPARISON 3: checkpoint_step5000.pth vs last.pth (Phase 2: step 5000 -> final)")
    print("Expected: output_conv should CHANGE in Phase 2 (unfrozen)")
    print("=" * 80)
    
    for group_name in ['pretrained (ViT encoder)', 'depth_head (DPT)', 'output_conv',
                        'mamba (temporal)', 'onepiece modules', 'metric head (GSP)', 'other']:
        keys = all_groups.get(group_name, [])
        if not keys:
            continue
        changed, unchanged, miss_a, miss_b = compare_params(step5000, last, keys, 'step5000.pth', 'last.pth')
        print_comparison(group_name, changed, unchanged, miss_a, miss_b, 'step5000.pth', 'last.pth', show_details=True)
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 80)
    print("SUMMARY: FREEZE STRATEGY VERIFICATION")
    print("=" * 80)
    
    checks = []
    
    # Check 1: ViT frozen across full training
    keys = all_groups.get('pretrained (ViT encoder)', [])
    if keys:
        changed, unchanged, _, _ = compare_params(best, last, keys, '', '')
        frozen = len(changed) == 0
        checks.append(('ViT encoder frozen (best -> last)', frozen, len(changed), len(unchanged)))
    
    # Check 2: DPT frozen across full training
    keys = all_groups.get('depth_head (DPT)', [])
    if keys:
        changed, unchanged, _, _ = compare_params(best, last, keys, '', '')
        frozen = len(changed) == 0
        checks.append(('DPT head frozen (best -> last)', frozen, len(changed), len(unchanged)))
    
    # Check 3: output_conv frozen in Phase 1
    keys = all_groups.get('output_conv', [])
    if keys:
        changed, unchanged, _, _ = compare_params(best, step5000, keys, '', '')
        frozen = len(changed) == 0
        checks.append(('output_conv frozen in Phase 1 (best -> step5000)', frozen, len(changed), len(unchanged)))
    
    # Check 4: output_conv changed in Phase 2
    keys = all_groups.get('output_conv', [])
    if keys:
        changed, unchanged, _, _ = compare_params(step5000, last, keys, '', '')
        did_change = len(changed) > 0
        checks.append(('output_conv unfrozen in Phase 2 (step5000 -> last)', did_change, len(changed), len(unchanged)))
    
    # Check 5: Mamba changed (should be trainable)
    keys = all_groups.get('mamba (temporal)', [])
    if keys:
        changed, unchanged, _, _ = compare_params(best, last, keys, '', '')
        did_change = len(changed) > 0
        checks.append(('Mamba modules trained (best -> last)', did_change, len(changed), len(unchanged)))
    
    # Check 6: Onepiece modules changed
    keys = all_groups.get('onepiece modules', [])
    if keys:
        changed, unchanged, _, _ = compare_params(best, last, keys, '', '')
        did_change = len(changed) > 0
        checks.append(('Onepiece modules trained (best -> last)', did_change, len(changed), len(unchanged)))
    
    print()
    all_pass = True
    for name, passed, n_changed, n_unchanged in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}")
        print(f"          -> {n_changed} changed, {n_unchanged} unchanged")
    
    print()
    if all_pass:
        print("  *** ALL CHECKS PASSED - Freeze strategy is working correctly ***")
    else:
        print("  *** SOME CHECKS FAILED - Investigate freeze strategy ***")
    
    print()


if __name__ == '__main__':
    main()
