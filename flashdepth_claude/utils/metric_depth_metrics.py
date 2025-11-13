import torch
import numpy as np
from typing import Dict, Tuple, Optional
import logging

# Import boundary metrics for F1 score computation
from utils.eval_metrics.boundary_metrics import SI_boundary_F1


class MetricDepthMetrics:
    """
    Comprehensive metrics for metric depth estimation evaluation
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def compute_scale_shift_invariant_metrics(pred: torch.Tensor, gt: torch.Tensor,
                                             valid_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Compute scale and shift invariant metrics (as used in MiDaS)

        Args:
            pred: Predicted depth [H, W] or [N, H, W]
            gt: Ground truth depth [H, W] or [N, H, W]
            valid_mask: Valid pixels mask [H, W] or [N, H, W]

        Returns:
            Dictionary of metrics
        """
        if valid_mask is None:
            valid_mask = (gt > 0)

        pred_valid = pred[valid_mask]
        gt_valid = gt[valid_mask]

        if len(pred_valid) == 0:
            return {"abs_rel": float('inf'), "rmse": float('inf'), "a1": 0.0}

        # Compute optimal scale and shift
        A = torch.stack([pred_valid, torch.ones_like(pred_valid)], dim=1)
        try:
            solution = torch.linalg.lstsq(A, gt_valid, rcond=None).solution
            scale, shift = solution[0], solution[1]
            pred_aligned = scale * pred_valid + shift
        except:
            # Fallback: use median scaling
            scale = torch.median(gt_valid / (pred_valid + 1e-8))
            pred_aligned = scale * pred_valid
            shift = 0.0

        # Compute metrics on aligned prediction
        abs_rel = torch.mean(torch.abs(pred_aligned - gt_valid) / gt_valid)
        sq_rel = torch.mean(((pred_aligned - gt_valid) ** 2) / gt_valid)
        rmse = torch.sqrt(torch.mean((pred_aligned - gt_valid) ** 2))
        rmse_log = torch.sqrt(torch.mean((torch.log(pred_aligned + 1e-8) - torch.log(gt_valid + 1e-8)) ** 2))

        # Threshold accuracies
        thresh = torch.maximum(
            (gt_valid / (pred_aligned + 1e-8)),
            (pred_aligned / (gt_valid + 1e-8))
        )
        a1 = torch.mean((thresh < 1.25).float())
        a2 = torch.mean((thresh < 1.25 ** 2).float())
        a3 = torch.mean((thresh < 1.25 ** 3).float())

        return {
            "abs_rel": abs_rel.item(),
            "sq_rel": sq_rel.item(),
            "rmse": rmse.item(),
            "rmse_log": rmse_log.item(),
            "a1": a1.item(),
            "a2": a2.item(),
            "a3": a3.item(),
            "scale": scale.item() if torch.is_tensor(scale) else scale,
            "shift": shift.item() if torch.is_tensor(shift) else shift,
        }

    @staticmethod
    def compute_metric_depth_metrics(pred: torch.Tensor, gt: torch.Tensor,
                                   valid_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Compute metrics for metric depth estimation (absolute depth values matter)

        Args:
            pred: Predicted depth in meters [H, W] or [N, H, W]
            gt: Ground truth depth in meters [H, W] or [N, H, W]
            valid_mask: Valid pixels mask [H, W] or [N, H, W]

        Returns:
            Dictionary of metrics
        """
        if valid_mask is None:
            valid_mask = (gt > 0)

        pred_valid = pred[valid_mask]
        gt_valid = gt[valid_mask]

        if len(pred_valid) == 0:
            return {k: float('inf') for k in ["mae", "rmse", "abs_rel", "sq_rel", "rmse_log", "a1", "a2", "a3"]}

        # Mean Absolute Error (L1 loss)
        mae = torch.mean(torch.abs(pred_valid - gt_valid))

        # Root Mean Square Error (L2 loss)
        rmse = torch.sqrt(torch.mean((pred_valid - gt_valid) ** 2))

        # Absolute Relative Error
        abs_rel = torch.mean(torch.abs(pred_valid - gt_valid) / gt_valid)
        
        # Squared Relative Error
        sq_rel = torch.mean(((pred_valid - gt_valid) ** 2) / gt_valid)

        # print(f"@@@@@Pred Max {torch.max(pred_valid)} Min {torch.min(pred_valid)} Mean {torch.mean(pred_valid)}@@@@@")

        # Log RMSE
        pred_log = torch.log(torch.clamp(pred_valid, min=1e-8))
        gt_log = torch.log(torch.clamp(gt_valid, min=1e-8))
        rmse_log = torch.sqrt(torch.mean((pred_log - gt_log) ** 2))

        # Threshold accuracies (delta_i)
        thresh = torch.maximum(
            (gt_valid / torch.clamp(pred_valid, min=1e-8)),
            (pred_valid / torch.clamp(gt_valid, min=1e-8))
        )
        a1 = torch.mean((thresh < 1.25).float())
        a2 = torch.mean((thresh < 1.25 ** 2).float())
        a3 = torch.mean((thresh < 1.25 ** 3).float())

        # Additional metrics
        mre = torch.mean((pred_valid - gt_valid) / gt_valid)  # Mean Relative Error (signed)
        log_mae = torch.mean(torch.abs(pred_log - gt_log))

        # Boundary F1 score (edge accuracy / depth discontinuity detection)
        # Convert to numpy for boundary_metrics computation
        pred_np = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else pred
        gt_np = gt.cpu().numpy() if isinstance(gt, torch.Tensor) else gt

        # Compute scale-invariant boundary F1 score
        # (weighted average across thresholds from 5% to 25% depth changes)
        try:
            boundary_f1 = SI_boundary_F1(pred_np, gt_np, t_min=1.05, t_max=1.25, N=10)
        except Exception as e:
            # Fallback if computation fails
            boundary_f1 = 0.0

        return {
            "mae": mae.item(),
            "rmse": rmse.item(),
            "abs_rel": abs_rel.item(),
            "sq_rel": sq_rel.item(),
            "rmse_log": rmse_log.item(),
            "mre": mre.item(),
            "log_mae": log_mae.item(),
            "a1": a1.item(),
            "a2": a2.item(),
            "a3": a3.item(),
            "boundary_f1": float(boundary_f1),  # Edge accuracy (depth discontinuity F1 score)
        }

    @staticmethod
    def compute_depth_range_metrics(pred: torch.Tensor, gt: torch.Tensor,
                                  valid_mask: Optional[torch.Tensor] = None,
                                  depth_ranges: Optional[list] = None) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for different depth ranges

        Args:
            pred: Predicted depth [H, W] or [N, H, W]
            gt: Ground truth depth [H, W] or [N, H, W]
            valid_mask: Valid pixels mask
            depth_ranges: List of (min, max) depth ranges

        Returns:
            Dictionary of metrics for each depth range
        """
        if depth_ranges is None:
            depth_ranges = [(0, 2), (2, 5), (5, 10), (10, float('inf'))]

        if valid_mask is None:
            valid_mask = (gt > 0)

        results = {}
        range_names = [f"{r[0]}-{r[1]}m" if r[1] != float('inf') else f">{r[0]}m"
                      for r in depth_ranges]

        for i, (depth_min, depth_max) in enumerate(depth_ranges):
            # Create range mask
            if depth_max == float('inf'):
                range_mask = (gt >= depth_min) & valid_mask
            else:
                range_mask = (gt >= depth_min) & (gt < depth_max) & valid_mask

            if range_mask.sum() == 0:
                results[range_names[i]] = {"mae": 0.0, "abs_rel": 0.0, "a1": 0.0, "count": 0}
                continue

            # Compute metrics for this range
            range_metrics = MetricDepthMetrics.compute_metric_depth_metrics(
                pred, gt, range_mask
            )
            range_metrics["count"] = range_mask.sum().item()
            results[range_names[i]] = range_metrics

        return results

    @staticmethod
    def compute_boundary_metrics(pred: torch.Tensor, gt: torch.Tensor,
                               valid_mask: Optional[torch.Tensor] = None,
                               threshold: float = 0.1) -> Dict[str, float]:
        """
        Compute metrics specifically at depth boundaries (discontinuities)

        Args:
            pred: Predicted depth [H, W]
            gt: Ground truth depth [H, W]
            valid_mask: Valid pixels mask
            threshold: Threshold for defining boundaries (relative depth change)

        Returns:
            Dictionary of boundary metrics
        """
        if valid_mask is None:
            valid_mask = (gt > 0)

        # Compute depth gradients
        gt_grad_x = torch.abs(gt[1:, :] - gt[:-1, :])
        gt_grad_y = torch.abs(gt[:, 1:] - gt[:, :-1])

        pred_grad_x = torch.abs(pred[1:, :] - pred[:-1, :])
        pred_grad_y = torch.abs(pred[:, 1:] - pred[:, :-1])

        # Find boundary pixels (high gradient)
        boundary_x = gt_grad_x > threshold * gt[:-1, :]
        boundary_y = gt_grad_y > threshold * gt[:, :-1]

        # Combine valid masks
        valid_x = valid_mask[1:, :] & valid_mask[:-1, :]
        valid_y = valid_mask[:, 1:] & valid_mask[:, :-1]

        boundary_x = boundary_x & valid_x
        boundary_y = boundary_y & valid_y

        if boundary_x.sum() == 0 and boundary_y.sum() == 0:
            return {"boundary_mae": 0.0, "boundary_count": 0}

        # Compute errors at boundaries
        errors_x = torch.abs(pred_grad_x - gt_grad_x)[boundary_x]
        errors_y = torch.abs(pred_grad_y - gt_grad_y)[boundary_y]

        all_errors = torch.cat([errors_x, errors_y])
        boundary_mae = torch.mean(all_errors).item()

        return {
            "boundary_mae": boundary_mae,
            "boundary_count": len(all_errors)
        }

    @staticmethod
    def compute_comprehensive_metrics(pred: torch.Tensor, gt: torch.Tensor,
                                    valid_mask: Optional[torch.Tensor] = None) -> Dict[str, any]:
        """
        Compute all available metrics

        Args:
            pred: Predicted depth [H, W] or [N, H, W]
            gt: Ground truth depth [H, W] or [N, H, W]
            valid_mask: Valid pixels mask

        Returns:
            Dictionary containing all metrics
        """
        # Handle batch dimension
        if pred.dim() == 3:
            # Compute metrics for each sample and average
            batch_metrics = []
            for i in range(pred.shape[0]):
                p = pred[i]
                g = gt[i]
                v = valid_mask[i] if valid_mask is not None else None
                batch_metrics.append(
                    MetricDepthMetrics.compute_comprehensive_metrics(p, g, v)
                )

            # Average metrics across batch
            result = {}
            for key in batch_metrics[0].keys():
                if isinstance(batch_metrics[0][key], dict):
                    result[key] = {}
                    for subkey in batch_metrics[0][key].keys():
                        values = [m[key][subkey] for m in batch_metrics]
                        result[key][subkey] = np.mean(values)
                else:
                    values = [m[key] for m in batch_metrics]
                    result[key] = np.mean(values)

            return result

        # Single sample metrics
        result = {}

        # Basic metric depth metrics
        basic_metrics = MetricDepthMetrics.compute_metric_depth_metrics(pred, gt, valid_mask)
        for k, v in basic_metrics.items():
            result[k] = v

        # Scale-shift invariant metrics (for comparison)
        result["scale_invariant"] = MetricDepthMetrics.compute_scale_shift_invariant_metrics(
            pred, gt, valid_mask
        )

        # Depth range metrics
        result["depth_ranges"] = MetricDepthMetrics.compute_depth_range_metrics(
            pred, gt, valid_mask
        )

        # Boundary metrics
        boundary_metrics = MetricDepthMetrics.compute_boundary_metrics(pred, gt, valid_mask)
        for k, v in boundary_metrics.items():
            result[k] = v

        # Additional statistics
        if valid_mask is None:
            valid_mask = (gt > 0)

        pred_valid = pred[valid_mask]
        gt_valid = gt[valid_mask]

        if len(pred_valid) > 0:
            additional_stats = {
                "pred_mean": torch.mean(pred_valid).item(),
                "pred_std": torch.std(pred_valid).item(),
                "gt_mean": torch.mean(gt_valid).item(),
                "gt_std": torch.std(gt_valid).item(),
                "pred_min": torch.min(pred_valid).item(),
                "pred_max": torch.max(pred_valid).item(),
                "gt_min": torch.min(gt_valid).item(),
                "gt_max": torch.max(gt_valid).item(),
                "valid_pixels": len(pred_valid),
                "total_pixels": pred.numel(),
                "valid_ratio": len(pred_valid) / pred.numel(),
            }
            for k, v in additional_stats.items():
                result[k] = v

        return result


def format_metrics(metrics: Dict, precision: int = 4) -> str:
    """
    Format metrics dictionary for readable output

    Args:
        metrics: Dictionary of metrics
        precision: Number of decimal places

    Returns:
        Formatted string
    """
    lines = []

    # Basic metrics
    basic_keys = ["mae", "rmse", "abs_rel", "sq_rel", "rmse_log", "a1", "a2", "a3"]
    lines.append("=== Basic Metric Depth Metrics ===")
    for key in basic_keys:
        if key in metrics:
            lines.append(f"{key.upper():>10}: {metrics[key]:.{precision}f}")

    # Scale invariant metrics
    if "scale_invariant" in metrics:
        lines.append("\n=== Scale-Shift Invariant Metrics ===")
        si_metrics = metrics["scale_invariant"]
        for key in ["abs_rel", "rmse", "a1", "scale", "shift"]:
            if key in si_metrics:
                lines.append(f"{key.upper():>10}: {si_metrics[key]:.{precision}f}")

    # Depth range metrics
    if "depth_ranges" in metrics:
        lines.append("\n=== Depth Range Metrics ===")
        for range_name, range_metrics in metrics["depth_ranges"].items():
            lines.append(f"\n{range_name}:")
            for key in ["mae", "abs_rel", "a1", "count"]:
                if key in range_metrics:
                    lines.append(f"  {key:>8}: {range_metrics[key]:.{precision}f}")

    # Boundary metrics
    boundary_keys = ["boundary_mae", "boundary_count"]
    if any(key in metrics for key in boundary_keys):
        lines.append("\n=== Boundary Metrics ===")
        for key in boundary_keys:
            if key in metrics:
                if key == "boundary_count":
                    lines.append(f"{key.upper():>15}: {int(metrics[key])}")
                else:
                    lines.append(f"{key.upper():>15}: {metrics[key]:.{precision}f}")

    # Statistics
    stat_keys = ["valid_pixels", "valid_ratio", "pred_mean", "gt_mean"]
    if any(key in metrics for key in stat_keys):
        lines.append("\n=== Statistics ===")
        for key in stat_keys:
            if key in metrics:
                if "pixels" in key:
                    lines.append(f"{key.upper():>12}: {int(metrics[key])}")
                else:
                    lines.append(f"{key.upper():>12}: {metrics[key]:.{precision}f}")

    return "\n".join(lines)


class RelativeDepthMetrics:
    """
    Metrics for relative (scale-invariant) depth estimation evaluation
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def compute_relative_depth_metrics(pred: torch.Tensor, gt: torch.Tensor,
                                       valid_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Compute metrics for relative depth estimation

        For relative depth, only scale-invariant metrics are meaningful:
        - Boundary F1: Edge-aware metric (scale-invariant)
        - Scale-invariant AbsRel and δ1 (optional, for comparison)

        Args:
            pred: Predicted relative depth [H, W] or [N, H, W]
            gt: Ground truth depth [H, W] or [N, H, W]
            valid_mask: Valid pixels mask [H, W] or [N, H, W]

        Returns:
            Dictionary containing:
            - boundary_f1: Scale-invariant boundary F1 score
            - abs_rel_si: Scale-invariant absolute relative error
            - a1: Scale-invariant δ1 threshold accuracy
        """
        if valid_mask is None:
            valid_mask = (gt > 0)

        # Compute scale-shift invariant alignment first
        si_metrics = MetricDepthMetrics.compute_scale_shift_invariant_metrics(
            pred, gt, valid_mask
        )

        # Convert to numpy for boundary F1 computation
        pred_np = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else pred
        gt_np = gt.cpu().numpy() if isinstance(gt, torch.Tensor) else gt

        # Compute scale-invariant boundary F1 score
        try:
            boundary_f1 = SI_boundary_F1(pred_np, gt_np, t_min=1.05, t_max=1.25, N=10)
        except Exception as e:
            boundary_f1 = 0.0

        return {
            "boundary_f1": float(boundary_f1),
            "abs_rel_si": si_metrics["abs_rel"],  # Scale-invariant AbsRel
            "a1": si_metrics["a1"],  # Scale-invariant δ1
        }

    @staticmethod
    def compute_tae_scale_invariant(pred_t: torch.Tensor, pred_t_next: torch.Tensor,
                                    gt_t: torch.Tensor, gt_t_next: torch.Tensor,
                                    valid_mask_t: torch.Tensor, valid_mask_t_next: torch.Tensor) -> float:
        """
        Compute scale-invariant Temporal Alignment Error (TAE)

        For relative depth, we need to align scale per frame pair before computing TAE,
        since the absolute scale of predictions may vary frame-to-frame.

        Args:
            pred_t: Predicted depth at frame t [H, W]
            pred_t_next: Predicted depth at frame t+1 [H, W]
            gt_t: Ground truth depth at frame t [H, W]
            gt_t_next: Ground truth depth at frame t+1 [H, W]
            valid_mask_t: Valid pixels at frame t [H, W]
            valid_mask_t_next: Valid pixels at frame t+1 [H, W]

        Returns:
            tae_si: Scale-invariant TAE value
        """
        # Combine valid masks for both frames
        valid_both = valid_mask_t & valid_mask_t_next

        if valid_both.sum() == 0:
            return float('inf')

        # Compute optimal scale per frame using median scaling
        scale_t = torch.median(gt_t[valid_both] / (pred_t[valid_both] + 1e-8))
        scale_t_next = torch.median(gt_t_next[valid_both] / (pred_t_next[valid_both] + 1e-8))

        # Align predictions to GT scale
        pred_t_aligned = pred_t * scale_t
        pred_t_next_aligned = pred_t_next * scale_t_next

        # Compute temporal changes (frame-to-frame differences)
        pred_change = pred_t_next_aligned - pred_t_aligned
        gt_change = gt_t_next - gt_t

        # Compute TAE on the changes (using valid mask)
        tae_si = torch.abs(pred_change[valid_both] - gt_change[valid_both]).mean()

        return tae_si.item()