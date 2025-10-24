"""
Visualize object-wise depth evaluation results.

Generates plots showing per-class improvements in depth estimation accuracy.

Usage:
    python scripts/visualize_object_wise_results.py \
        --results-json test_results/object_wise/kitti_object_wise_results.json \
        --output-dir test_results/object_wise/visualizations
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_results(json_path: Path) -> Dict:
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        results = json.load(f)
    return results


def plot_per_class_metrics(
    results: Dict,
    output_dir: Path,
    metrics: list = ['mae', 'rmse', 'abs_rel', 'a1']
):
    """
    Plot metrics for each class.

    Args:
        results: Results dictionary from JSON
        output_dir: Output directory for plots
        metrics: Metrics to plot
    """
    per_class = results['per_class_metrics']

    # Sort classes by number of pixels (most common first)
    classes = sorted(
        per_class.items(),
        key=lambda x: x[1].get('num_pixels', 0),
        reverse=True
    )

    class_names = [c[0] for c in classes]
    class_metrics = {metric: [] for metric in metrics}

    for class_name, class_data in classes:
        for metric in metrics:
            class_metrics[metric].append(class_data.get(metric, 0))

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Per-Class Depth Metrics', fontsize=16, fontweight='bold')

    metric_titles = {
        'mae': 'Mean Absolute Error (m)',
        'rmse': 'Root Mean Square Error (m)',
        'abs_rel': 'Absolute Relative Error',
        'a1': 'δ1 Accuracy (higher is better)'
    }

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        # Bar plot
        bars = ax.bar(range(len(class_names)), class_metrics[metric], color='steelblue', alpha=0.7)

        # Highlight best class
        if metric == 'a1':
            best_idx = np.argmax(class_metrics[metric])
        else:
            best_idx = np.argmin(class_metrics[metric])
        bars[best_idx].set_color('green')
        bars[best_idx].set_alpha(0.9)

        ax.set_xlabel('Object Class', fontsize=12)
        ax.set_ylabel(metric_titles[metric], fontsize=12)
        ax.set_title(metric_titles[metric], fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, class_metrics[metric])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_path = output_dir / 'per_class_metrics.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved per-class metrics plot to {output_path}")


def plot_model_comparison(
    results: Dict,
    output_dir: Path,
    metrics: list = ['mae', 'rmse', 'abs_rel', 'a1']
):
    """
    Plot model comparison showing improvements.

    Args:
        results: Results dictionary from JSON
        output_dir: Output directory for plots
        metrics: Metrics to plot
    """
    if 'model_comparison' not in results or results['model_comparison'] is None:
        logger.warning("No model comparison available, skipping comparison plot")
        return

    comparison = results['model_comparison']

    # Sort classes by improvement (highest first)
    classes = sorted(
        comparison.items(),
        key=lambda x: x[1].get('mae_improvement', 0),
        reverse=True
    )

    class_names = [c[0] for c in classes]

    # Create improvement plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Per-Class Model Improvements (Baseline → Gear3)', fontsize=16, fontweight='bold')

    metric_titles = {
        'mae': 'MAE Improvement (%)',
        'rmse': 'RMSE Improvement (%)',
        'abs_rel': 'AbsRel Improvement (%)',
        'a1': 'δ1 Improvement (%)'
    }

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        # Get improvements
        improvements = [comparison[c][f'{metric}_improvement'] for c in class_names]

        # Bar plot with color based on positive/negative
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = ax.bar(range(len(class_names)), improvements, color=colors, alpha=0.7)

        ax.set_xlabel('Object Class', fontsize=12)
        ax.set_ylabel(metric_titles[metric], fontsize=12)
        ax.set_title(metric_titles[metric], fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, improvements):
            height = bar.get_height()
            va = 'bottom' if height > 0 else 'top'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:+.1f}%',
                   ha='center', va=va, fontsize=9)

    plt.tight_layout()
    output_path = output_dir / 'model_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved model comparison plot to {output_path}")


def plot_baseline_vs_gear3(
    results: Dict,
    output_dir: Path,
    metric: str = 'mae'
):
    """
    Plot baseline vs Gear3 side-by-side comparison.

    Args:
        results: Results dictionary from JSON
        output_dir: Output directory for plots
        metric: Metric to compare
    """
    if 'model_comparison' not in results or results['model_comparison'] is None:
        logger.warning("No model comparison available, skipping side-by-side plot")
        return

    comparison = results['model_comparison']

    # Sort by baseline performance (worst first)
    classes = sorted(
        comparison.items(),
        key=lambda x: x[1].get(f'Baseline_{metric}', 0),
        reverse=(metric == 'a1')  # Reverse for accuracy (higher is better)
    )

    class_names = [c[0] for c in classes]
    baseline_values = [comparison[c][f'Baseline_{metric}'] for c in class_names]
    gear3_values = [comparison[c][f'Gear3_{metric}'] for c in class_names]

    # Create side-by-side comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(class_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', color='coral', alpha=0.7)
    bars2 = ax.bar(x + width/2, gear3_values, width, label='Gear3', color='steelblue', alpha=0.7)

    metric_titles = {
        'mae': 'Mean Absolute Error (m)',
        'rmse': 'Root Mean Square Error (m)',
        'abs_rel': 'Absolute Relative Error',
        'a1': 'δ1 Accuracy'
    }

    ax.set_xlabel('Object Class', fontsize=12)
    ax.set_ylabel(metric_titles.get(metric, metric), fontsize=12)
    ax.set_title(f'Baseline vs Gear3: {metric_titles.get(metric, metric)}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Add improvement annotations
    for i, (b_val, g_val) in enumerate(zip(baseline_values, gear3_values)):
        if metric == 'a1':
            improvement = (g_val - b_val) / b_val * 100
        else:
            improvement = (b_val - g_val) / b_val * 100

        color = 'green' if improvement > 0 else 'red'
        ax.annotate(f'{improvement:+.1f}%',
                   xy=(i, max(b_val, g_val)),
                   xytext=(0, 5),
                   textcoords='offset points',
                   ha='center',
                   fontsize=8,
                   color=color,
                   fontweight='bold')

    plt.tight_layout()
    output_path = output_dir / f'baseline_vs_gear3_{metric}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved baseline vs Gear3 plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize object-wise evaluation results')

    parser.add_argument('--results-json', type=Path, required=True,
                        help='Path to results JSON file')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Output directory for plots')
    parser.add_argument('--metrics', type=str, nargs='+',
                        default=['mae', 'rmse', 'abs_rel', 'a1'],
                        help='Metrics to plot (default: mae rmse abs_rel a1)')

    args = parser.parse_args()

    # Check input
    if not args.results_json.exists():
        logger.error(f"Results file not found: {args.results_json}")
        return

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    logger.info(f"Loading results from {args.results_json}")
    results = load_results(args.results_json)

    # Generate plots
    logger.info("Generating per-class metrics plot...")
    plot_per_class_metrics(results, args.output_dir, metrics=args.metrics)

    logger.info("Generating model comparison plot...")
    plot_model_comparison(results, args.output_dir, metrics=args.metrics)

    logger.info("Generating side-by-side comparison plots...")
    for metric in args.metrics:
        plot_baseline_vs_gear3(results, args.output_dir, metric=metric)

    logger.info(f"Complete! All plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
