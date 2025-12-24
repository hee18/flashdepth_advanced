"""
Compare Instance Depth Results - Depth Variation Analysis

GT와 각 depth 모델의 prediction을 비교합니다.
정확도 비교가 아닌 **뎁스 변동량**만 비교합니다.

- 프레임별 변동량: variation[t] = depth[t+1] - depth[t]
- GT 변동량과 Prediction 변동량 비교
- 통계: max, mean, min, std (절대값 + 퍼센트값)

Usage:
    python scripts/compare_instance_results.py \
        --gt-path test_results/crosswalk_gt/crosswalk_sample/instance_tracking_results.json \
        --pred-paths test_results/instance_gear5_l/crosswalk_sample/instance_tracking_results.json \
                     test_results/instance_comparison/metric3d/crosswalk_sample/instance_tracking_results.json \
        --output-dir test_results/crosswalk_comparison
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_results(json_path: Path) -> Dict[str, Any]:
    """결과 JSON 로드"""
    with open(json_path, 'r') as f:
        return json.load(f)


def compute_depth_variations(trajectory: List[Dict]) -> List[Dict]:
    """
    Trajectory에서 프레임별 depth 변동량 계산

    Args:
        trajectory: [{'frame': 0, 'depth_m': 8.5, ...}, ...]

    Returns:
        variations: [{'frame': 0, 'variation': 0.12}, ...]
    """
    if len(trajectory) < 2:
        return []

    # Sort by frame
    sorted_traj = sorted(trajectory, key=lambda x: x['frame'])

    variations = []
    for i in range(len(sorted_traj) - 1):
        curr = sorted_traj[i]
        next_frame = sorted_traj[i + 1]

        # Check consecutive frames
        if next_frame['frame'] != curr['frame'] + 1:
            # Non-consecutive frames - skip
            continue

        depth_curr = curr['depth_m']
        depth_next = next_frame['depth_m']

        # Skip invalid depths
        if depth_curr >= 1000 or depth_next >= 1000:
            continue
        if depth_curr <= 0 or depth_next <= 0:
            continue

        variation = depth_next - depth_curr
        variations.append({
            'frame': curr['frame'],
            'variation': variation,
            'depth_curr': depth_curr,
            'depth_next': depth_next
        })

    return variations


def compute_variation_statistics(variations: List[float]) -> Dict[str, float]:
    """
    변동량 통계 계산

    Args:
        variations: list of variation values

    Returns:
        {'max': ..., 'mean': ..., 'min': ..., 'std': ...}
    """
    if len(variations) == 0:
        return {
            'max': 0.0,
            'mean': 0.0,
            'min': 0.0,
            'std': 0.0
        }

    arr = np.array(variations)
    return {
        'max': float(np.max(np.abs(arr))),
        'mean': float(np.mean(np.abs(arr))),
        'min': float(np.min(np.abs(arr))),
        'std': float(np.std(arr))
    }


def compare_instance_variations(
    gt_trajectory: List[Dict],
    pred_trajectory: List[Dict]
) -> Dict[str, Any]:
    """
    단일 인스턴스의 GT vs Prediction 변동량 비교

    Args:
        gt_trajectory: GT trajectory
        pred_trajectory: Prediction trajectory

    Returns:
        comparison dict with frame-wise variations and statistics
    """
    # Compute variations
    gt_vars = compute_depth_variations(gt_trajectory)
    pred_vars = compute_depth_variations(pred_trajectory)

    # Build frame -> variation maps
    gt_var_map = {v['frame']: v for v in gt_vars}
    pred_var_map = {v['frame']: v for v in pred_vars}

    # Find common frames
    common_frames = sorted(set(gt_var_map.keys()) & set(pred_var_map.keys()))

    if len(common_frames) == 0:
        return {
            'frame_variations': [],
            'statistics': {
                'abs': {'max': 0.0, 'mean': 0.0, 'min': 0.0, 'std': 0.0},
                'pct': {'max': 0.0, 'mean': 0.0, 'min': 0.0, 'std': 0.0}
            },
            'num_common_frames': 0
        }

    # Frame-wise comparison
    frame_variations = []
    diff_abs_list = []
    diff_pct_list = []

    for frame in common_frames:
        gt_var = gt_var_map[frame]['variation']
        pred_var = pred_var_map[frame]['variation']

        diff = pred_var - gt_var
        diff_abs = abs(diff)

        # Percentage difference (avoid division by zero)
        if abs(gt_var) > 1e-6:
            diff_pct = abs(diff / gt_var) * 100
        else:
            diff_pct = 0.0 if abs(diff) < 1e-6 else 100.0

        frame_variations.append({
            'frame': frame,
            'gt_var': round(gt_var, 6),
            'pred_var': round(pred_var, 6),
            'diff': round(diff, 6),
            'diff_abs': round(diff_abs, 6),
            'diff_pct': round(diff_pct, 2)
        })

        diff_abs_list.append(diff_abs)
        diff_pct_list.append(diff_pct)

    # Statistics
    abs_stats = compute_variation_statistics(diff_abs_list)
    pct_stats = {
        'max': float(np.max(diff_pct_list)) if diff_pct_list else 0.0,
        'mean': float(np.mean(diff_pct_list)) if diff_pct_list else 0.0,
        'min': float(np.min(diff_pct_list)) if diff_pct_list else 0.0,
        'std': float(np.std(diff_pct_list)) if diff_pct_list else 0.0
    }

    return {
        'frame_variations': frame_variations,
        'statistics': {
            'abs': {k: round(v, 6) for k, v in abs_stats.items()},
            'pct': {k: round(v, 2) for k, v in pct_stats.items()}
        },
        'num_common_frames': len(common_frames)
    }


def match_instances(
    gt_instances: Dict[str, Any],
    pred_instances: Dict[str, Any]
) -> List[Tuple[str, str]]:
    """
    GT와 Prediction의 인스턴스 매칭 (track_id 기준)

    Returns:
        List of (gt_track_id, pred_track_id) tuples
    """
    gt_ids = set(gt_instances.keys())
    pred_ids = set(pred_instances.keys())

    # 동일한 track_id 매칭
    matched = [(tid, tid) for tid in gt_ids & pred_ids]

    logger.info(f"Matched {len(matched)} instances out of GT:{len(gt_ids)}, Pred:{len(pred_ids)}")

    return matched


def compare_results(
    gt_path: Path,
    pred_path: Path,
    model_name: str
) -> Dict[str, Any]:
    """
    GT와 단일 Prediction 결과 비교

    Args:
        gt_path: GT JSON 경로
        pred_path: Prediction JSON 경로
        model_name: 모델 이름

    Returns:
        comparison results dict
    """
    logger.info(f"Comparing GT vs {model_name}")
    logger.info(f"  GT: {gt_path}")
    logger.info(f"  Pred: {pred_path}")

    # Load results
    gt_results = load_results(gt_path)
    pred_results = load_results(pred_path)

    gt_instances = gt_results.get('instances', {})
    pred_instances = pred_results.get('instances', {})

    # Match instances
    matched_pairs = match_instances(gt_instances, pred_instances)

    if len(matched_pairs) == 0:
        logger.warning("No matched instances found!")
        return {
            'model_name': model_name,
            'gt_path': str(gt_path),
            'pred_path': str(pred_path),
            'instances': {},
            'overall': {
                'abs': {'max': 0.0, 'mean': 0.0, 'min': 0.0, 'std': 0.0},
                'pct': {'max': 0.0, 'mean': 0.0, 'min': 0.0, 'std': 0.0}
            },
            'num_matched_instances': 0,
            'total_comparison_frames': 0
        }

    # Compare each matched instance
    instance_comparisons = {}
    all_diff_abs = []
    all_diff_pct = []

    for gt_id, pred_id in matched_pairs:
        gt_traj = gt_instances[gt_id]['trajectory']
        pred_traj = pred_instances[pred_id]['trajectory']

        comparison = compare_instance_variations(gt_traj, pred_traj)
        instance_comparisons[gt_id] = comparison

        # Collect all diffs for overall statistics
        for fv in comparison['frame_variations']:
            all_diff_abs.append(fv['diff_abs'])
            all_diff_pct.append(fv['diff_pct'])

    # Overall statistics
    overall_abs = compute_variation_statistics(all_diff_abs)
    overall_pct = {
        'max': float(np.max(all_diff_pct)) if all_diff_pct else 0.0,
        'mean': float(np.mean(all_diff_pct)) if all_diff_pct else 0.0,
        'min': float(np.min(all_diff_pct)) if all_diff_pct else 0.0,
        'std': float(np.std(all_diff_pct)) if all_diff_pct else 0.0
    }

    return {
        'model_name': model_name,
        'gt_path': str(gt_path),
        'pred_path': str(pred_path),
        'instances': instance_comparisons,
        'overall': {
            'abs': {k: round(v, 6) for k, v in overall_abs.items()},
            'pct': {k: round(v, 2) for k, v in overall_pct.items()}
        },
        'num_matched_instances': len(matched_pairs),
        'total_comparison_frames': len(all_diff_abs)
    }


def create_variation_comparison_plot(
    comparisons: Dict[str, Dict],
    output_path: Path
):
    """
    변동량 비교 타임라인 플롯 생성

    여러 모델의 변동량을 GT와 함께 시각화
    """
    fig, axes = plt.subplots(len(comparisons) + 1, 1, figsize=(14, 4 * (len(comparisons) + 1)))
    if len(comparisons) == 0:
        plt.close()
        return

    if len(comparisons) == 0:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # For each model, plot GT vs Pred variations
    for idx, (model_name, comparison) in enumerate(comparisons.items()):
        ax = axes[idx] if len(comparisons) > 0 else axes

        for inst_id, inst_comp in comparison['instances'].items():
            if len(inst_comp['frame_variations']) == 0:
                continue

            frames = [fv['frame'] for fv in inst_comp['frame_variations']]
            gt_vars = [fv['gt_var'] for fv in inst_comp['frame_variations']]
            pred_vars = [fv['pred_var'] for fv in inst_comp['frame_variations']]

            ax.plot(frames, gt_vars, 'o-', label=f'GT (Instance {inst_id})', alpha=0.7)
            ax.plot(frames, pred_vars, 's--', label=f'{model_name} (Instance {inst_id})', alpha=0.7)

        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Depth Variation (m)')
        ax.set_title(f'{model_name} vs GT - Depth Variation')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Last subplot: Overall statistics bar chart
    ax_stats = axes[-1] if len(comparisons) > 0 else axes
    model_names = list(comparisons.keys())
    mean_abs = [comparisons[m]['overall']['abs']['mean'] for m in model_names]
    max_abs = [comparisons[m]['overall']['abs']['max'] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    ax_stats.bar(x - width/2, mean_abs, width, label='Mean |diff|', color='steelblue')
    ax_stats.bar(x + width/2, max_abs, width, label='Max |diff|', color='coral')
    ax_stats.set_ylabel('Variation Difference (m)')
    ax_stats.set_title('Overall Variation Difference Statistics')
    ax_stats.set_xticks(x)
    ax_stats.set_xticklabels(model_names, rotation=45, ha='right')
    ax_stats.legend()
    ax_stats.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved variation comparison plot: {output_path}")


def create_histogram_plot(
    comparisons: Dict[str, Dict],
    output_path: Path
):
    """
    변동량 차이 히스토그램 생성
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Collect all diffs per model
    model_diffs_abs = {}
    model_diffs_pct = {}

    for model_name, comparison in comparisons.items():
        diffs_abs = []
        diffs_pct = []
        for inst_comp in comparison['instances'].values():
            for fv in inst_comp['frame_variations']:
                diffs_abs.append(fv['diff_abs'])
                diffs_pct.append(fv['diff_pct'])
        model_diffs_abs[model_name] = diffs_abs
        model_diffs_pct[model_name] = diffs_pct

    # Absolute difference histogram
    ax1 = axes[0]
    for model_name, diffs in model_diffs_abs.items():
        if len(diffs) > 0:
            ax1.hist(diffs, bins=30, alpha=0.6, label=model_name)
    ax1.set_xlabel('|Prediction Var - GT Var| (m)')
    ax1.set_ylabel('Count')
    ax1.set_title('Variation Difference Distribution (Absolute)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Percentage difference histogram
    ax2 = axes[1]
    for model_name, diffs in model_diffs_pct.items():
        if len(diffs) > 0:
            # Cap at 200% for visualization
            diffs_capped = [min(d, 200) for d in diffs]
            ax2.hist(diffs_capped, bins=30, alpha=0.6, label=model_name)
    ax2.set_xlabel('|Prediction Var - GT Var| / |GT Var| (%)')
    ax2.set_ylabel('Count')
    ax2.set_title('Variation Difference Distribution (Percentage)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved histogram plot: {output_path}")


def create_per_instance_stats_plot(
    comparisons: Dict[str, Dict],
    output_path: Path
):
    """
    인스턴스별 통계 막대 그래프
    """
    # Collect instance IDs across all models
    all_instances = set()
    for comparison in comparisons.values():
        all_instances.update(comparison['instances'].keys())

    if len(all_instances) == 0:
        return

    all_instances = sorted(all_instances, key=lambda x: int(x))
    model_names = list(comparisons.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Mean absolute diff per instance
    ax1 = axes[0]
    x = np.arange(len(all_instances))
    width = 0.8 / len(model_names)

    for i, model_name in enumerate(model_names):
        means = []
        for inst_id in all_instances:
            if inst_id in comparisons[model_name]['instances']:
                means.append(comparisons[model_name]['instances'][inst_id]['statistics']['abs']['mean'])
            else:
                means.append(0)
        ax1.bar(x + i * width, means, width, label=model_name)

    ax1.set_ylabel('Mean |diff| (m)')
    ax1.set_title('Per-Instance Mean Variation Difference (Absolute)')
    ax1.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax1.set_xticklabels([f'Inst {i}' for i in all_instances])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Mean percentage diff per instance
    ax2 = axes[1]
    for i, model_name in enumerate(model_names):
        means = []
        for inst_id in all_instances:
            if inst_id in comparisons[model_name]['instances']:
                means.append(comparisons[model_name]['instances'][inst_id]['statistics']['pct']['mean'])
            else:
                means.append(0)
        ax2.bar(x + i * width, means, width, label=model_name)

    ax2.set_ylabel('Mean |diff| (%)')
    ax2.set_title('Per-Instance Mean Variation Difference (Percentage)')
    ax2.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax2.set_xticklabels([f'Inst {i}' for i in all_instances])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved per-instance stats plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare Instance Depth Results - Variation Analysis')
    parser.add_argument('--gt-path', type=str, required=True,
                        help='Path to GT JSON file')
    parser.add_argument('--pred-paths', type=str, nargs='+', required=True,
                        help='Paths to prediction JSON files')
    parser.add_argument('--model-names', type=str, nargs='*',
                        help='Model names (optional, derived from path if not provided)')
    parser.add_argument('--output-dir', type=str, default='test_results/crosswalk_comparison',
                        help='Output directory')

    args = parser.parse_args()

    gt_path = Path(args.gt_path)
    pred_paths = [Path(p) for p in args.pred_paths]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive model names if not provided
    if args.model_names and len(args.model_names) == len(pred_paths):
        model_names = args.model_names
    else:
        model_names = []
        for p in pred_paths:
            # Try to extract model name from path
            parts = p.parts
            for i, part in enumerate(parts):
                if 'instance' in part.lower():
                    if i + 1 < len(parts):
                        model_names.append(parts[i].replace('instance_', ''))
                        break
            else:
                model_names.append(p.parent.name)

    # Compare each prediction with GT
    all_comparisons = {}
    for pred_path, model_name in zip(pred_paths, model_names):
        if not pred_path.exists():
            logger.warning(f"Prediction file not found: {pred_path}")
            continue

        comparison = compare_results(gt_path, pred_path, model_name)
        all_comparisons[model_name] = comparison

        logger.info(f"\n{model_name} Results:")
        logger.info(f"  Matched instances: {comparison['num_matched_instances']}")
        logger.info(f"  Total comparison frames: {comparison['total_comparison_frames']}")
        logger.info(f"  Overall abs stats: {comparison['overall']['abs']}")
        logger.info(f"  Overall pct stats: {comparison['overall']['pct']}")

    # Save combined results JSON
    combined_results = {
        'gt_path': str(gt_path),
        'comparisons': all_comparisons
    }
    json_path = output_dir / 'comparison_results.json'
    with open(json_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    logger.info(f"\nSaved comparison results: {json_path}")

    # Create visualizations
    if len(all_comparisons) > 0:
        create_variation_comparison_plot(
            all_comparisons,
            output_dir / 'variation_comparison.png'
        )
        create_histogram_plot(
            all_comparisons,
            output_dir / 'variation_histogram.png'
        )
        create_per_instance_stats_plot(
            all_comparisons,
            output_dir / 'per_instance_stats.png'
        )

    # Print summary table
    print("\n" + "=" * 80)
    print("DEPTH VARIATION COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Model':<25} {'Instances':<10} {'Frames':<10} {'Mean|diff|(m)':<15} {'Max|diff|(m)':<15} {'Mean|diff|(%)':<15}")
    print("-" * 80)
    for model_name, comparison in all_comparisons.items():
        print(f"{model_name:<25} "
              f"{comparison['num_matched_instances']:<10} "
              f"{comparison['total_comparison_frames']:<10} "
              f"{comparison['overall']['abs']['mean']:<15.6f} "
              f"{comparison['overall']['abs']['max']:<15.6f} "
              f"{comparison['overall']['pct']['mean']:<15.2f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
