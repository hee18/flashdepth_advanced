"""
Object-wise depth evaluation utilities.

Evaluates depth estimation accuracy per segmentation class to demonstrate
improvements on specific object types (e.g., vehicles, pedestrians, cyclists).

Supports multiple dataset formats:
- KITTI: Instance segmentation (car, pedestrian, cyclist)
- Cityscapes: Semantic/instance segmentation (19 classes)
- NYU Depth V2: Semantic segmentation (40 classes)
- ScanNet: Semantic/instance segmentation (20 classes)
- VKITTI2: Semantic/instance segmentation (13 classes)
"""

import numpy as np
import torch
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Import boundary metrics for F1 score computation
from utils.eval_metrics.boundary_metrics import SI_boundary_F1

logger = logging.getLogger(__name__)


class ObjectWiseMetrics:
    """Compute depth metrics per segmentation class."""

    # KITTI class IDs (from instance segmentation)
    KITTI_CLASSES = {
        0: 'background',
        1: 'car',
        2: 'pedestrian',
        3: 'cyclist'
    }

    # Cityscapes class IDs (trainId format)
    CITYSCAPES_CLASSES = {
        0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence',
        5: 'pole', 6: 'traffic_light', 7: 'traffic_sign', 8: 'vegetation',
        9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car',
        14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle',
        255: 'ignore'
    }

    # NYU Depth V2 class IDs (40 classes)
    NYU_CLASSES = {
        0: 'unknown', 1: 'bed', 2: 'books', 3: 'ceiling', 4: 'chair',
        5: 'floor', 6: 'furniture', 7: 'objects', 8: 'picture', 9: 'sofa',
        10: 'table', 11: 'tv', 12: 'wall', 13: 'window'
        # ... (40 classes total, abbreviated for brevity)
    }

    # VKITTI2 class IDs (15 classes total)
    # Source: https://www.changjiangcai.com/studynotes/2020-05-16-Virtual-KITTI-2-Dataset/
    VKITTI2_CLASSES = {
        0: 'undefined', 1: 'terrain', 2: 'sky', 3: 'tree', 4: 'vegetation',
        5: 'building', 6: 'road', 7: 'guard_rail', 8: 'traffic_sign',
        9: 'traffic_light', 10: 'pole', 11: 'misc', 12: 'truck', 13: 'car', 14: 'van'
    }

    # Waymo Open Dataset class IDs (Semantic Segmentation v2.0)
    # Based on Waymo Open Dataset 2.0 semantic segmentation labels
    WAYMO_CLASSES = {
        0: 'undefined',
        1: 'vehicle',
        2: 'pedestrian',
        3: 'sign',
        4: 'cyclist',
        5: 'traffic_light',
        6: 'pole',
        7: 'construction_cone',
        8: 'bicycle',
        9: 'motorcycle',
        10: 'building',
        11: 'vegetation',
        12: 'tree_trunk',
        13: 'curb',
        14: 'road',
        15: 'lane_marker',
        16: 'other_ground',
        17: 'walkable',
        18: 'sidewalk',
        255: 'ignore'
    }

    # UrbanSyn class IDs (uses Cityscapes 19-class standard)
    # Same as Cityscapes trainId format
    URBANSYN_CLASSES = {
        0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence',
        5: 'pole', 6: 'traffic_light', 7: 'traffic_sign', 8: 'vegetation',
        9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car',
        14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle',
        255: 'ignore'
    }


    # Object class names (dynamic, movable objects) for visualization
    # These classes will be visualized in test_object_wise

    KITTI_OBJECT_CLASSES = {
        'pedestrian', 'car', 'cyclist'
    }

    CITYSCAPES_OBJECT_CLASSES = {
        'person', 'rider', 'car', 'truck', 'bus', 'train',
        'motorcycle', 'bicycle'
    }

    NYU_OBJECT_CLASSES = {
        'chair', 'sofa', 'bed', 'table', 'tv', 'book'
    }

    VKITTI2_OBJECT_CLASSES = {
        'truck', 'car', 'van'
    }

    WAYMO_OBJECT_CLASSES = {
        'vehicle', 'pedestrian', 'cyclist', 'bicycle', 'motorcycle'
    }

    URBANSYN_OBJECT_CLASSES = {
        'person', 'rider', 'car', 'truck', 'bus', 'train',
        'motorcycle', 'bicycle'
    }


    def __init__(self, dataset_type: str = 'kitti', depth_mode: str = 'metric'):
        """
        Initialize object-wise metrics calculator.

        Args:
            dataset_type: Dataset type ('kitti', 'cityscapes', 'nyu', 'vkitti2', 'waymo', 'urbansyn')
                         Also accepts '_seg' variants (e.g., 'waymo_seg')
            depth_mode: Depth type ('metric' or 'relative')
        """
        # Normalize dataset type: remove _seg suffix and handle aliases
        self.dataset_type = dataset_type.lower().replace('_seg', '')
        self.depth_mode = depth_mode

        # Handle dataset aliases
        if self.dataset_type == 'vkitti':
            self.dataset_type = 'vkitti2'  # VKITTI uses VKITTI2 class definitions

        if self.dataset_type == 'kitti':
            self.classes = self.KITTI_CLASSES
            self.object_classes = self.KITTI_OBJECT_CLASSES
        elif self.dataset_type == 'cityscapes':
            self.classes = self.CITYSCAPES_CLASSES
            self.object_classes = self.CITYSCAPES_OBJECT_CLASSES
        elif self.dataset_type == 'nyu':
            self.classes = self.NYU_CLASSES
            self.object_classes = self.NYU_OBJECT_CLASSES
        elif self.dataset_type == 'vkitti2':
            self.classes = self.VKITTI2_CLASSES
            self.object_classes = self.VKITTI2_OBJECT_CLASSES
        elif self.dataset_type == 'waymo':
            self.classes = self.WAYMO_CLASSES
            self.object_classes = self.WAYMO_OBJECT_CLASSES
        elif self.dataset_type == 'urbansyn':
            self.classes = self.URBANSYN_CLASSES
            self.object_classes = self.URBANSYN_OBJECT_CLASSES
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type} (original: {dataset_type})")

        logger.info(f"Initialized object-wise metrics for {self.dataset_type} ({len(self.classes)} classes)")
        logger.info(f"Object classes for visualization: {len(self.object_classes)} classes")

    def compute_metrics_per_class(
        self,
        pred_depth: np.ndarray,
        gt_depth: np.ndarray,
        seg_mask: np.ndarray,
        min_pixels: int = 100,
        max_depth: float = 1000.0
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute depth metrics for each segmentation class.

        Args:
            pred_depth: Predicted depth map (H, W)
            gt_depth: Ground truth depth map (H, W)
            seg_mask: Segmentation mask (H, W) with class IDs
            min_pixels: Minimum pixels required to compute metrics for a class
            max_depth: Maximum valid depth value (default: 1000m)

        Returns:
            Dictionary mapping class names to metrics dictionaries
        """
        results = {}

        # Standard per-class processing
        # Get unique classes in this frame
        unique_classes = np.unique(seg_mask)

        for class_id in unique_classes:
            if class_id not in self.classes:
                continue

            class_name = self.classes[class_id]

            # Skip ignore classes
            if class_name in ['ignore', 'unknown', 'undefined']:
                continue

            # IMPORTANT: Only compute metrics for object classes (dynamic, movable objects)
            # Skip static/scene classes like sky, road, building, vegetation, etc.
            if class_name not in self.object_classes:
                continue

            # Create mask for this class
            class_mask = (seg_mask == class_id)

            # Combine with valid depth mask (filter invalid and out-of-range depths)
            valid_mask = (gt_depth > 0) & (pred_depth > 0) & (pred_depth < max_depth) & class_mask

            # Skip if too few pixels
            num_pixels = np.sum(valid_mask)
            if num_pixels < min_pixels:
                continue

            # Extract valid depths
            pred_valid = pred_depth[valid_mask]
            gt_valid = gt_depth[valid_mask]

            # Compute metrics (pass full maps for F1 computation in relative mode)
            metrics = self._compute_depth_metrics(
                pred_valid, gt_valid,
                pred_full=pred_depth,
                gt_full=gt_depth
            )
            metrics['num_pixels'] = int(num_pixels)

            results[class_name] = metrics

        return results

    def _compute_depth_metrics(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        pred_full: np.ndarray = None,
        gt_full: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Compute standard depth estimation metrics.

        Args:
            pred: Predicted depth values (N,) - masked pixels only
            gt: Ground truth depth values (N,) - masked pixels only
            pred_full: Full predicted depth map (H, W) - for F1 computation
            gt_full: Full ground truth depth map (H, W) - for F1 computation

        Returns:
            Dictionary of metrics
        """
        # Absolute error metrics
        abs_diff = np.abs(pred - gt)
        mae = np.mean(abs_diff)
        rmse = np.sqrt(np.mean((pred - gt) ** 2))

        # Relative error metrics
        abs_rel = np.mean(abs_diff / gt)
        sq_rel = np.mean(((pred - gt) ** 2) / gt)

        # Threshold accuracy metrics
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = np.mean(thresh < 1.25)
        a2 = np.mean(thresh < 1.25 ** 2)
        a3 = np.mean(thresh < 1.25 ** 3)

        # Compute boundary F1 (edge accuracy / depth discontinuity detection)
        # Works for both metric and relative depth (scale-invariant)
        if pred_full is not None and gt_full is not None:
            try:
                boundary_f1 = SI_boundary_F1(pred_full, gt_full, t_min=1.05, t_max=1.25, N=10)
            except Exception as e:
                logger.warning(f"Failed to compute boundary F1: {e}")
                boundary_f1 = 0.0
        else:
            boundary_f1 = 0.0

        # Order metrics: abs_rel, a1, a2, a3, fps, tae, f1, mae, rmse
        # Note: fps and tae are not computed here, will be added later if needed
        metrics = {
            'abs_rel': float(abs_rel),
            'a1': float(a1),
            'a2': float(a2),
            'a3': float(a3),
            'boundary_f1': float(boundary_f1),
            'mae': float(mae),
            'rmse': float(rmse),
            'sq_rel': float(sq_rel)
        }

        return metrics

    def aggregate_metrics(
        self,
        class_metrics_list: List[Dict[str, Dict[str, float]]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate metrics across multiple frames.

        Args:
            class_metrics_list: List of per-frame class metrics

        Returns:
            Aggregated metrics per class (mean across frames)
        """
        # Collect all metrics per class
        class_aggregated = {}

        for frame_metrics in class_metrics_list:
            for class_name, metrics in frame_metrics.items():
                if class_name not in class_aggregated:
                    class_aggregated[class_name] = []
                class_aggregated[class_name].append(metrics)

        # Compute mean for each class
        results = {}
        for class_name, metrics_list in class_aggregated.items():
            if not metrics_list:
                continue

            # Average all metrics
            avg_metrics = {}
            metric_keys = metrics_list[0].keys()
            for key in metric_keys:
                if key == 'num_pixels':
                    # Sum total pixels
                    avg_metrics[key] = sum(m[key] for m in metrics_list)
                else:
                    # Average metrics
                    avg_metrics[key] = np.mean([m[key] for m in metrics_list])

            avg_metrics['num_frames'] = len(metrics_list)
            results[class_name] = avg_metrics

        return results

    def compare_models(
        self,
        model_a_metrics: Dict[str, Dict[str, float]],
        model_b_metrics: Dict[str, Dict[str, float]],
        model_a_name: str = "Baseline",
        model_b_name: str = "Gear3"
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare two models' per-class metrics.

        Args:
            model_a_metrics: Metrics from model A (baseline)
            model_b_metrics: Metrics from model B (Gear3)
            model_a_name: Name of model A
            model_b_name: Name of model B

        Returns:
            Dictionary showing improvement/degradation per class
        """
        comparison = {}

        # Get all classes present in either model
        all_classes = set(model_a_metrics.keys()) | set(model_b_metrics.keys())

        for class_name in all_classes:
            if class_name not in model_a_metrics or class_name not in model_b_metrics:
                logger.warning(f"Class '{class_name}' not in both models, skipping comparison")
                continue

            metrics_a = model_a_metrics[class_name]
            metrics_b = model_b_metrics[class_name]

            comparison[class_name] = {
                f'{model_a_name}_mae': metrics_a['mae'],
                f'{model_b_name}_mae': metrics_b['mae'],
                'mae_improvement': (metrics_a['mae'] - metrics_b['mae']) / metrics_a['mae'] * 100,

                f'{model_a_name}_rmse': metrics_a['rmse'],
                f'{model_b_name}_rmse': metrics_b['rmse'],
                'rmse_improvement': (metrics_a['rmse'] - metrics_b['rmse']) / metrics_a['rmse'] * 100,

                f'{model_a_name}_abs_rel': metrics_a['abs_rel'],
                f'{model_b_name}_abs_rel': metrics_b['abs_rel'],
                'abs_rel_improvement': (metrics_a['abs_rel'] - metrics_b['abs_rel']) / metrics_a['abs_rel'] * 100,

                f'{model_a_name}_a1': metrics_a['a1'],
                f'{model_b_name}_a1': metrics_b['a1'],
                'a1_improvement': (metrics_b['a1'] - metrics_a['a1']) / metrics_a['a1'] * 100,

                'num_pixels': metrics_b['num_pixels'],
                'num_frames': metrics_b.get('num_frames', 1)
            }

        return comparison

    def save_results(
        self,
        results: Dict[str, Dict[str, float]],
        output_path: Path,
        comparison: Optional[Dict[str, Dict[str, float]]] = None
    ):
        """
        Save metrics to JSON file.

        Args:
            results: Per-class metrics
            output_path: Output JSON path
            comparison: Optional model comparison results
        """
        # Reorder metrics for each class: abs_rel, a1, a2, a3, fps, tae, f1, mae, rmse
        metric_order = ['abs_rel', 'a1', 'a2', 'a3', 'fps', 'tae', 'boundary_f1', 'mae', 'rmse']

        reordered_results = {}
        for class_name, class_metrics in results.items():
            reordered = {}
            # Add metrics in the desired order
            for key in metric_order:
                if key in class_metrics:
                    reordered[key] = class_metrics[key]
            # Add any remaining metrics not in the order list
            for key, value in class_metrics.items():
                if key not in reordered:
                    reordered[key] = value
            reordered_results[class_name] = reordered

        output_data = {
            'dataset_type': self.dataset_type,
            'per_class_metrics': reordered_results
        }

        if comparison is not None:
            # Also reorder comparison metrics if provided
            reordered_comparison = {}
            for class_name, class_metrics in comparison.items():
                reordered = {}
                for key in metric_order:
                    if key in class_metrics:
                        reordered[key] = class_metrics[key]
                for key, value in class_metrics.items():
                    if key not in reordered:
                        reordered[key] = value
                reordered_comparison[class_name] = reordered
            output_data['model_comparison'] = reordered_comparison

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Saved object-wise metrics to {output_path}")

    def print_summary(
        self,
        results: Dict[str, Dict[str, float]],
        comparison: Optional[Dict[str, Dict[str, float]]] = None
    ):
        """
        Print formatted summary of results.

        Args:
            results: Per-class metrics
            comparison: Optional model comparison results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Object-wise Depth Evaluation ({self.dataset_type.upper()})")
        logger.info(f"{'='*80}")

        # Sort classes by number of pixels (most common first)
        sorted_classes = sorted(
            results.items(),
            key=lambda x: x[1].get('num_pixels', 0),
            reverse=True
        )

        for class_name, metrics in sorted_classes:
            logger.info(f"\n{class_name.upper()}")
            logger.info(f"  Pixels: {metrics.get('num_pixels', 0):,} ({metrics.get('num_frames', 1)} frames)")
            logger.info(f"  MAE: {metrics['mae']:.4f}m")
            logger.info(f"  RMSE: {metrics['rmse']:.4f}m")
            logger.info(f"  AbsRel: {metrics['abs_rel']:.4f}")
            logger.info(f"  delta1: {metrics['a1']:.4f}")

        if comparison is not None:
            logger.info(f"\n{'='*80}")
            logger.info("MODEL COMPARISON (% improvement, positive = better)")
            logger.info(f"{'='*80}")

            for class_name, comp_metrics in comparison.items():
                logger.info(f"\n{class_name.upper()}")
                logger.info(f"  MAE: {comp_metrics['mae_improvement']:+.2f}%")
                logger.info(f"  RMSE: {comp_metrics['rmse_improvement']:+.2f}%")
                logger.info(f"  AbsRel: {comp_metrics['abs_rel_improvement']:+.2f}%")
                logger.info(f"  delta1: {comp_metrics['a1_improvement']:+.2f}%")


def load_segmentation_mask(seg_path: Path, dataset_type: str) -> np.ndarray:
    """
    Load segmentation mask from file.

    Args:
        seg_path: Path to segmentation file
        dataset_type: Dataset type ('kitti', 'cityscapes', 'nyu', 'vkitti2')

    Returns:
        Segmentation mask as numpy array (H, W)
    """
    if dataset_type == 'kitti':
        # KITTI instance segmentation typically in PNG format
        from PIL import Image
        seg = np.array(Image.open(seg_path))

    elif dataset_type == 'cityscapes':
        # Cityscapes uses PNG with trainId encoding
        from PIL import Image
        seg = np.array(Image.open(seg_path))

    elif dataset_type == 'nyu':
        # NYU Depth V2 uses MAT or PNG format
        if seg_path.suffix == '.mat':
            import scipy.io
            seg = scipy.io.loadmat(seg_path)['segmentation']
        else:
            from PIL import Image
            seg = np.array(Image.open(seg_path))

    elif dataset_type == 'vkitti2':
        # VKITTI2 uses PNG format
        from PIL import Image
        seg = np.array(Image.open(seg_path))

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return seg


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Initialize for KITTI dataset
    evaluator = ObjectWiseMetrics(dataset_type='kitti')

    # Dummy example (replace with actual data)
    H, W = 518, 518
    pred_depth = np.random.rand(H, W) * 50  # Predicted depth
    gt_depth = np.random.rand(H, W) * 50    # GT depth
    seg_mask = np.random.randint(0, 4, (H, W))  # Segmentation (0-3 for KITTI)

    # Compute per-class metrics
    class_metrics = evaluator.compute_metrics_per_class(pred_depth, gt_depth, seg_mask)

    # Print summary
    evaluator.print_summary(class_metrics)
