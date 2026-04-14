"""
Flow-based Temporal Consistency (rTC) metric.

Implements rTC from "Enforcing Temporal Consistency in Video Depth Estimation":
    rTC_i = (1/sum(M_i)) * sum(M_i * [max(D_i/D_hat_{i+1}, D_hat_{i+1}/D_i) < thr])

Occlusion mask M_i follows the original paper's implementation:
    M_i = exp(-sigma * ||X_i - X_hat_{i+1}||^2) * flow_magnitude_mask * depth_validity
where X_hat is the warped color frame (detects occluded/invalid flow regions).

Uses SEA-RAFT optical flow to warp depth maps between consecutive frames and
measures the ratio of temporally consistent pixels.

Multi-threshold analysis: computes rTC at thresholds 1.05-1.25 (step 0.05)
and counts flickering frames (rTC < flicker_cutoff) per threshold.

Complements reprojection-based TAE (which requires camera poses).
"""

import json
import torch
import torch.nn.functional as F
import numpy as np
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Multi-threshold analysis settings
MULTI_THRESHOLDS = [1.05, 1.10, 1.15, 1.20, 1.25]
FLICKER_CUTOFFS = [0.3, 0.5, 0.7]  # rTC < cutoff → flickering frame

# ImageNet normalization constants (used by DINOv2 / FlashDepth)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


class FlowTemporalConsistency:
    """
    Computes flow-based temporal consistency (rTC) between consecutive depth frames.

    Lazy-loads SEA-RAFT on first use (~200MB GPU, ~1-2s loading time).
    """

    # SEA-RAFT was trained at 540x960; limit long edge to avoid OOM on high-res inputs
    FLOW_MAX_LONG_EDGE = 960
    # Cap visualization resolution to avoid slow matplotlib rendering on high-res datasets
    VIS_MAX_LONG_EDGE = 1024

    def __init__(self, device='cuda:0', thr=1.1, max_depth=70.0, checkpoint_path=None,
                 occlusion_sigma=50.0, occlusion_hard_thr=0.01, flow_max_magnitude=250.0,
                 flicker_cutoffs=None):
        """
        Args:
            device: torch device
            thr: ratio threshold for rTC (default 1.1)
            max_depth: maximum valid depth in meters
            checkpoint_path: path to SEA-RAFT weights (auto-detected if None)
            occlusion_sigma: sigma for soft color occlusion mask (default 50.0, from TCMonoDepth)
            occlusion_hard_thr: hard threshold on occlusion mask product (default 0.01)
            flow_max_magnitude: maximum flow magnitude in pixels (default 250.0)
            flicker_cutoffs: list of rTC cutoffs for flickering detection (default [0.3, 0.5, 0.7])
        """
        self.device = device
        self.thr = thr
        self.max_depth = max_depth
        self.checkpoint_path = checkpoint_path
        self.occlusion_sigma = occlusion_sigma
        self.occlusion_hard_thr = occlusion_hard_thr
        self.flow_max_magnitude = flow_max_magnitude
        self.flicker_cutoffs = flicker_cutoffs or FLICKER_CUTOFFS
        self._flow_estimator = None  # Lazy loaded

    def offload_to_cpu(self):
        """Move SEA-RAFT to CPU to free GPU memory between sequences."""
        if self._flow_estimator is not None:
            self._flow_estimator.model.cpu()
            torch.cuda.empty_cache()

    def _get_flow_estimator(self):
        """Lazy-load SEA-RAFT flow estimator on first use."""
        if self._flow_estimator is not None:
            self._flow_estimator.model.to(self.device)
            return self._flow_estimator

        import os
        from utils.flow_estimator import FlowEstimator

        if self.checkpoint_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.checkpoint_path = os.path.join(
                project_root, 'third_party', 'SEA-RAFT', 'models',
                'Tartan-C-T-TSKH-spring540x960-M.pth'
            )

        logger.info(f"Loading SEA-RAFT for temporal consistency from {self.checkpoint_path}")
        self._flow_estimator = FlowEstimator(self.checkpoint_path, device=self.device)
        return self._flow_estimator

    def _denormalize_images(self, images):
        """
        De-normalize ImageNet-normalized images to 0-1 range for SEA-RAFT.

        Args:
            images: [T, 3, H, W] ImageNet-normalized tensor

        Returns:
            images_01: [T, 3, H, W] in 0-1 range
        """
        mean = IMAGENET_MEAN.to(images.device).view(1, 3, 1, 1)
        std = IMAGENET_STD.to(images.device).view(1, 3, 1, 1)
        images_01 = images * std + mean
        return images_01.clamp(0, 1)

    @torch.no_grad()
    def compute_rtc(self, images, pred_depths, gt_depths=None):
        """
        Compute rTC for a sequence of frames.

        Args:
            images: [T, 3, H, W] ImageNet-normalized images (from dataloader)
            pred_depths: [T, 1, H, W] predicted metric depth (meters)
            gt_depths: [T, 1, H, W] ground truth metric depth (meters), optional

        Returns:
            dict with keys:
                rtc: float - mean rTC across frame pairs (pred)
                rtc_gt: float - mean rTC across frame pairs (GT, oracle upper bound)
                per_frame_rtc: list[float] - rTC for each pair (i, i+1)
                per_frame_rtc_gt: list[float] - rTC_gt for each pair
                per_frame_ratio_stats: list[dict] - ratio statistics per pair
                ratio_stats: dict - aggregated ratio statistics
                best_frame_idx: int - pair index with highest rTC
                worst_frame_idx: int - pair index with lowest rTC
        """
        flow_estimator = self._get_flow_estimator()
        T = images.shape[0]

        if T < 2:
            return {
                'rtc': 0.0, 'rtc_gt': 0.0,
                'per_frame_rtc': [], 'per_frame_rtc_gt': [],
                'per_frame_ratio_stats': [],
                'ratio_stats': {'avg': 0.0, 'min': 0.0, 'max': 0.0, 'p90': 0.0, 'p95': 0.0},
                'best_frame_idx': 0, 'worst_frame_idx': 0
            }

        # Downscale on CPU BEFORE moving to GPU to avoid OOM on high-res inputs
        _, _, H_orig, W_orig = images.shape
        long_edge = max(H_orig, W_orig)
        if long_edge > self.FLOW_MAX_LONG_EDGE:
            scale_factor = self.FLOW_MAX_LONG_EDGE / long_edge
            new_H = (int(H_orig * scale_factor) // 8) * 8
            new_W = (int(W_orig * scale_factor) // 8) * 8
            logger.info(f"Downscaling for flow estimation: {H_orig}x{W_orig} -> {new_H}x{new_W}")
            images = F.interpolate(images.float(), size=(new_H, new_W), mode='bilinear', align_corners=False)
            pred_depths = F.interpolate(pred_depths.float(), size=(new_H, new_W), mode='bilinear', align_corners=False)
            if gt_depths is not None:
                gt_depths = F.interpolate(gt_depths.float(), size=(new_H, new_W), mode='bilinear', align_corners=False)

        # De-normalize images for SEA-RAFT (expects 0-1) and move to GPU
        images_01 = self._denormalize_images(images.to(self.device))

        # Ensure depths are on device and float32
        pred_d = pred_depths.to(self.device).float()  # [T, 1, H, W]
        if gt_depths is not None:
            gt_d = gt_depths.to(self.device).float()
        else:
            gt_d = None

        per_frame_rtc = []
        per_frame_rtc_gt = []
        per_frame_ratio_stats = []
        all_valid_ratios = []  # For aggregated stats
        # Multi-threshold: store per-frame ratios for post-hoc analysis
        per_frame_ratios_list = []  # list of (ratio_tensor, num_valid) per pair

        for t in range(T - 1):
            # Estimate forward flow: frame_t -> frame_{t+1}
            frame_t = images_01[t:t+1]    # [1, 3, H, W]
            frame_tp1 = images_01[t+1:t+2]  # [1, 3, H, W]
            flow, _ = flow_estimator.estimate_flow(frame_t, frame_tp1)  # [1, 2, H, W]

            # === Pred rTC (with color occlusion mask) ===
            rtc_val, ratio_stats = self._compute_pair_rtc(
                pred_d[t:t+1], pred_d[t+1:t+2], flow,
                img_i=frame_t, img_ip1=frame_tp1
            )
            per_frame_rtc.append(rtc_val)
            per_frame_ratio_stats.append(ratio_stats)
            if ratio_stats.get('_valid_ratios') is not None:
                all_valid_ratios.append(ratio_stats['_valid_ratios'])
                per_frame_ratios_list.append(ratio_stats['_valid_ratios'])
            else:
                per_frame_ratios_list.append(None)

            # === GT rTC (oracle) ===
            if gt_d is not None:
                rtc_gt_val, _ = self._compute_pair_rtc(
                    gt_d[t:t+1], gt_d[t+1:t+2], flow,
                    img_i=frame_t, img_ip1=frame_tp1
                )
                per_frame_rtc_gt.append(rtc_gt_val)
            else:
                per_frame_rtc_gt.append(0.0)

        # Clean ratio stats (remove internal _valid_ratios)
        clean_ratio_stats = []
        for rs in per_frame_ratio_stats:
            clean_rs = {k: v for k, v in rs.items() if not k.startswith('_')}
            clean_ratio_stats.append(clean_rs)

        # Aggregated ratio statistics
        if all_valid_ratios:
            all_ratios_cat = np.concatenate(all_valid_ratios)
            agg_ratio_stats = {
                'avg': float(np.mean(all_ratios_cat)),
                'min': float(np.min(all_ratios_cat)),
                'max': float(np.max(all_ratios_cat)),
                'p90': float(np.percentile(all_ratios_cat, 90)),
                'p95': float(np.percentile(all_ratios_cat, 95))
            }
        else:
            agg_ratio_stats = {'avg': 0.0, 'min': 0.0, 'max': 0.0, 'p90': 0.0, 'p95': 0.0}

        # Best/worst frame pair
        valid_rtc = [r for r in per_frame_rtc if r > 0]
        if valid_rtc:
            best_idx = int(np.argmax(per_frame_rtc))
            worst_idx = int(np.argmin(per_frame_rtc))
        else:
            best_idx = 0
            worst_idx = 0

        mean_rtc = float(np.mean(per_frame_rtc)) if per_frame_rtc else 0.0
        mean_rtc_gt = float(np.mean(per_frame_rtc_gt)) if per_frame_rtc_gt else 0.0

        # === Multi-threshold analysis ===
        multi_threshold_result = self._compute_multi_threshold(
            per_frame_ratios_list, per_frame_rtc_gt
        )

        return {
            'rtc': mean_rtc,
            'rtc_gt': mean_rtc_gt,
            'per_frame_rtc': [float(x) for x in per_frame_rtc],
            'per_frame_rtc_gt': [float(x) for x in per_frame_rtc_gt],
            'per_frame_ratio_stats': clean_ratio_stats,
            'ratio_stats': agg_ratio_stats,
            'best_frame_idx': best_idx,
            'worst_frame_idx': worst_idx,
            'multi_threshold': multi_threshold_result,
        }

    def _compute_pair_rtc(self, depth_i, depth_ip1, flow, img_i=None, img_ip1=None):
        """
        Compute rTC for a single frame pair (i, i+1).

        Occlusion mask follows "Enforcing Temporal Consistency in Video Depth Estimation":
            M_i = exp(-sigma * ||X_i - warp(X_{i+1})||^2) * flow_mask * depth_validity > hard_thr

        Args:
            depth_i: [1, 1, H, W] depth at frame i
            depth_ip1: [1, 1, H, W] depth at frame i+1
            flow: [1, 2, H, W] forward optical flow from i to i+1
            img_i: [1, 3, H, W] RGB image at frame i (0-1 range), for occlusion mask
            img_ip1: [1, 3, H, W] RGB image at frame i+1 (0-1 range), for occlusion mask

        Returns:
            rtc: float - ratio of consistent pixels
            ratio_stats: dict - statistics of depth ratios
        """
        _, _, H, W = depth_i.shape

        # Create sampling grid: pixel coordinates + flow
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=flow.device, dtype=flow.dtype),
            torch.arange(W, device=flow.device, dtype=flow.dtype),
            indexing='ij'
        )
        grid_x = grid_x.unsqueeze(0)  # [1, H, W]
        grid_y = grid_y.unsqueeze(0)  # [1, H, W]

        # Apply flow: where does pixel (x, y) in frame i go in frame i+1?
        warped_x = grid_x + flow[:, 0]  # [1, H, W]
        warped_y = grid_y + flow[:, 1]  # [1, H, W]

        # Normalize to [-1, 1] for grid_sample
        norm_x = 2.0 * warped_x / (W - 1) - 1.0
        norm_y = 2.0 * warped_y / (H - 1) - 1.0
        grid = torch.stack([norm_x, norm_y], dim=-1)  # [1, H, W, 2]

        # Warp D_{i+1} to frame i's coordinate system
        warped_depth = F.grid_sample(
            depth_ip1, grid, mode='bilinear', padding_mode='zeros', align_corners=True
        )  # [1, 1, H, W]

        # === Validity mask M_i (following TCMonoDepth) ===
        # 1. In-bounds check
        in_bounds = (warped_x >= 0) & (warped_x < W) & (warped_y >= 0) & (warped_y < H)
        in_bounds = in_bounds.unsqueeze(1)  # [1, 1, H, W]

        # 2. Flow magnitude filter (reject extremely large flows)
        flow_mag = torch.sqrt(flow[:, 0:1] ** 2 + flow[:, 1:2] ** 2)  # [1, 1, H, W]
        flow_valid = flow_mag <= self.flow_max_magnitude

        # 3. Depth validity
        d_i = depth_i
        d_hat = warped_depth
        depth_valid = (d_i > 0) & (d_i < self.max_depth) & (d_hat > 0) & (d_hat < self.max_depth)

        # 4. Color-based occlusion mask (soft): exp(-sigma * ||X_i - warp(X_{i+1})||^2)
        if img_i is not None and img_ip1 is not None:
            warped_img = F.grid_sample(
                img_ip1, grid, mode='bilinear', padding_mode='zeros', align_corners=True
            )  # [1, 3, H, W]
            color_diff = torch.sqrt(torch.sum((img_i - warped_img) ** 2, dim=1, keepdim=True))  # [1, 1, H, W]
            occlusion_mask = torch.exp(-self.occlusion_sigma * color_diff)  # [1, 1, H, W]

            # Combined mask: soft occlusion * hard filters > threshold
            combined_soft = occlusion_mask * in_bounds.float() * flow_valid.float() * depth_valid.float()
            valid_mask = combined_soft > self.occlusion_hard_thr
        else:
            # Fallback: geometric mask only (no color info)
            valid_mask = in_bounds & flow_valid & depth_valid

        num_valid = valid_mask.sum().item()
        if num_valid == 0:
            return 0.0, {'avg': 0.0, 'min': 0.0, 'max': 0.0, 'p90': 0.0, 'p95': 0.0, '_valid_ratios': None}

        # Compute depth ratio: max(D_i/D_hat, D_hat/D_i)
        d_i_valid = d_i[valid_mask]
        d_hat_valid = d_hat[valid_mask]
        ratio = torch.maximum(d_i_valid / d_hat_valid, d_hat_valid / d_i_valid)

        # rTC = fraction of pixels with ratio < threshold
        consistent = (ratio < self.thr).float()
        rtc = consistent.mean().item()

        # Ratio statistics
        ratio_np = ratio.cpu().numpy()
        ratio_stats = {
            'avg': float(np.mean(ratio_np)),
            'min': float(np.min(ratio_np)),
            'max': float(np.max(ratio_np)),
            'p90': float(np.percentile(ratio_np, 90)),
            'p95': float(np.percentile(ratio_np, 95)),
            '_valid_ratios': ratio_np  # Internal, removed before JSON output
        }

        return rtc, ratio_stats

    def _compute_multi_threshold(self, per_frame_ratios_list, per_frame_rtc_gt):
        """
        Compute rTC at multiple thresholds and count flickering frames at multiple cutoffs.

        Args:
            per_frame_ratios_list: list of numpy arrays (per-pixel ratios per frame pair)
            per_frame_rtc_gt: list of GT rTC values per frame pair

        Returns:
            dict with per-threshold rTC values, flickering counts at each cutoff
        """
        result = {}

        for thr in MULTI_THRESHOLDS:
            thr_key = f"{thr:.2f}"
            per_frame_rtc_at_thr = []

            for ratios in per_frame_ratios_list:
                if ratios is not None and len(ratios) > 0:
                    rtc_val = float(np.mean(ratios < thr))
                else:
                    rtc_val = 0.0
                per_frame_rtc_at_thr.append(rtc_val)

            mean_rtc = float(np.mean(per_frame_rtc_at_thr)) if per_frame_rtc_at_thr else 0.0
            total_pairs = len(per_frame_rtc_at_thr)

            # Flickering detection at multiple cutoffs
            flickering = {}
            for cutoff in self.flicker_cutoffs:
                cutoff_key = f"{cutoff:.1f}"
                flicker_frames = [i for i, rtc in enumerate(per_frame_rtc_at_thr)
                                  if rtc < cutoff]
                flickering[cutoff_key] = {
                    'count': len(flicker_frames),
                    'rate': len(flicker_frames) / max(total_pairs, 1),
                    'frames': flicker_frames,
                }

            result[thr_key] = {
                'rtc': mean_rtc,
                'per_frame_rtc': [float(x) for x in per_frame_rtc_at_thr],
                'total_pairs': total_pairs,
                'flickering': flickering,
            }

        # Add metadata
        result['_meta'] = {
            'thresholds': MULTI_THRESHOLDS,
            'flicker_cutoffs': self.flicker_cutoffs,
            'occlusion_sigma': self.occlusion_sigma,
            'occlusion_hard_thr': self.occlusion_hard_thr,
            'flow_max_magnitude': self.flow_max_magnitude,
        }

        return result

    def get_ratio_heatmap(self, images, pred_depths, frame_idx):
        """
        Get pixel-wise ratio heatmap for a specific frame pair.

        Args:
            images: [T, 3, H, W] ImageNet-normalized
            pred_depths: [T, 1, H, W] predicted depths
            frame_idx: int - frame pair index k (pair k, k+1)

        Returns:
            ratio_map: [H, W] numpy array of depth ratios (0 where invalid)
        """
        flow_estimator = self._get_flow_estimator()

        # Downscale on CPU BEFORE moving to GPU to avoid OOM
        _, _, H_orig, W_orig = images.shape
        long_edge = max(H_orig, W_orig)
        if long_edge > self.FLOW_MAX_LONG_EDGE:
            scale_factor = self.FLOW_MAX_LONG_EDGE / long_edge
            new_H = (int(H_orig * scale_factor) // 8) * 8
            new_W = (int(W_orig * scale_factor) // 8) * 8
            images = F.interpolate(images.float(), size=(new_H, new_W), mode='bilinear', align_corners=False)
            pred_depths = F.interpolate(pred_depths.float(), size=(new_H, new_W), mode='bilinear', align_corners=False)

        images_01 = self._denormalize_images(images.to(self.device))
        pred_d = pred_depths.to(self.device).float()

        t = frame_idx
        img_t = images_01[t:t+1]
        img_tp1 = images_01[t+1:t+2]
        flow, _ = flow_estimator.estimate_flow(img_t, img_tp1)

        _, _, H, W = pred_d.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=flow.device, dtype=flow.dtype),
            torch.arange(W, device=flow.device, dtype=flow.dtype),
            indexing='ij'
        )
        warped_x = grid_x.unsqueeze(0) + flow[:, 0]
        warped_y = grid_y.unsqueeze(0) + flow[:, 1]

        norm_x = 2.0 * warped_x / (W - 1) - 1.0
        norm_y = 2.0 * warped_y / (H - 1) - 1.0
        grid = torch.stack([norm_x, norm_y], dim=-1)

        warped_depth = F.grid_sample(
            pred_d[t+1:t+2], grid, mode='bilinear', padding_mode='zeros', align_corners=True
        )

        in_bounds = (warped_x >= 0) & (warped_x < W) & (warped_y >= 0) & (warped_y < H)

        d_i = pred_d[t, 0]
        d_hat = warped_depth[0, 0]

        # Occlusion mask (consistent with _compute_pair_rtc)
        flow_mag = torch.sqrt(flow[:, 0] ** 2 + flow[:, 1] ** 2)  # [1, H, W]
        flow_valid = flow_mag[0] <= self.flow_max_magnitude
        depth_valid = (d_i > 0) & (d_i < self.max_depth) & (d_hat > 0) & (d_hat < self.max_depth)

        warped_img = F.grid_sample(
            img_tp1, grid, mode='bilinear', padding_mode='zeros', align_corners=True
        )
        color_diff = torch.sqrt(torch.sum((img_t - warped_img) ** 2, dim=1))  # [1, H, W]
        occlusion_mask = torch.exp(-self.occlusion_sigma * color_diff[0])  # [H, W]

        combined_soft = occlusion_mask * in_bounds[0].float() * flow_valid.float() * depth_valid.float()
        valid = combined_soft > self.occlusion_hard_thr

        ratio_map = torch.zeros(H, W, device=self.device)
        if valid.sum() > 0:
            ratio_map[valid] = torch.maximum(d_i[valid] / d_hat[valid], d_hat[valid] / d_i[valid])

        return ratio_map.cpu().numpy()

    # === Visualization methods ===

    def save_visualization(self, pred_depths, gt_depths, frame_idx, sequence_id,
                           save_dir, rtc_value, label='worst', dataset_name=''):
        """
        Save 2-row 4-column depth grid visualization.

        Row 1 (GT):   [frame k-1] [frame k] [frame k+1] [frame k+2]
        Row 2 (Pred):  same frame indices

        Args:
            pred_depths: [T, 1, H, W] tensor
            gt_depths: [T, 1, H, W] tensor
            frame_idx: int - frame pair index k (pair k, k+1)
            sequence_id: int
            save_dir: Path
            rtc_value: float
            label: 'worst' or 'best'
            dataset_name: str - dataset name for title
        """
        T = pred_depths.shape[0]

        # Context frames: k-1, k, k+1, k+2 (clamped to valid range)
        context_indices = [
            max(0, frame_idx - 1),
            frame_idx,
            min(T - 1, frame_idx + 1),
            min(T - 1, frame_idx + 2)
        ]

        # Downscale for visualization if high-res (e.g. ETH3D 4135x6205)
        _, _, H_orig, W_orig = pred_depths.shape
        long_edge = max(H_orig, W_orig)
        if long_edge > self.VIS_MAX_LONG_EDGE:
            scale = self.VIS_MAX_LONG_EDGE / long_edge
            new_H = int(H_orig * scale)
            new_W = int(W_orig * scale)
            pred_depths = F.interpolate(pred_depths.float(), size=(new_H, new_W),
                                        mode='bilinear', align_corners=False)
            gt_depths = F.interpolate(gt_depths.float(), size=(new_H, new_W),
                                      mode='bilinear', align_corners=False)

        pred_np = pred_depths.cpu().float().numpy()
        gt_np = gt_depths.cpu().float().numpy()

        # Compute shared colormap range from GT (2nd-98th percentile)
        # Also build per-frame GT valid masks for unified pred masking
        gt_valid_all = []
        gt_valid_masks = {}
        for idx in context_indices:
            gt_frame = gt_np[idx, 0]
            valid = (gt_frame > 0) & (gt_frame < self.max_depth)
            gt_valid_masks[idx] = valid
            if valid.sum() > 0:
                gt_valid_all.append(gt_frame[valid])
        if gt_valid_all:
            gt_valid_concat = np.concatenate(gt_valid_all)
            vmin = float(np.percentile(gt_valid_concat, 2))
            vmax = float(np.percentile(gt_valid_concat, 98))
        else:
            vmin, vmax = 0, self.max_depth

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.set_facecolor('white')

        cmap = plt.cm.plasma_r.copy()
        cmap.set_bad(color='black')  # Invalid (NaN) pixels shown as black

        for col, idx in enumerate(context_indices):
            # Row 0: GT
            gt_frame = gt_np[idx, 0].copy()
            gt_valid = gt_valid_masks[idx]
            gt_frame[~gt_valid] = np.nan
            axes[0, col].imshow(gt_frame, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[0, col].set_title(f'GT frame {idx}', fontsize=10)
            axes[0, col].axis('off')

            # Row 1: Pred (masked by GT valid region + pred valid range)
            pred_frame = pred_np[idx, 0].copy()
            pred_invalid = (pred_frame <= 0) | (pred_frame >= self.max_depth)
            pred_frame[~gt_valid | pred_invalid] = np.nan
            axes[1, col].imshow(pred_frame, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[1, col].set_title(f'Pred frame {idx}', fontsize=10)
            axes[1, col].axis('off')

        ds_prefix = f'{dataset_name} | ' if dataset_name else ''
        fig.suptitle(
            f'{ds_prefix}Seq {sequence_id} | {label.upper()} TC (frames {frame_idx}\u2192{frame_idx+1}) | rTC={rtc_value:.4f}',
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout()

        filename = f'tc_{label}_seq{sequence_id:04d}.png'
        fig.savefig(save_dir / filename, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        logger.info(f"Saved TC {label} visualization: {filename}")

    def save_ratio_heatmap(self, images, pred_depths, frame_idx, sequence_id,
                           save_dir, rtc_value, label='worst', dataset_name=''):
        """
        Save pixel-wise depth ratio heatmap.

        For each pixel, computes max(D_k / D_hat, D_hat / D_k) where D_hat is
        frame k+1's depth warped to frame k via optical flow. Ratio=1 means
        perfect temporal consistency; higher values indicate inconsistency.

        Args:
            images: [T, 3, H, W] ImageNet-normalized
            pred_depths: [T, 1, H, W]
            frame_idx: int - frame pair index
            sequence_id: int
            save_dir: Path
            rtc_value: float
            label: 'worst' or 'best'
            dataset_name: str - dataset name for title
        """
        ratio_map = self.get_ratio_heatmap(images, pred_depths, frame_idx)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.set_facecolor('white')

        # Mask invalid pixels
        display_map = ratio_map.copy()
        display_map[display_map == 0] = np.nan

        cmap_hot = plt.cm.hot.copy()
        cmap_hot.set_bad(color='black')  # Invalid pixels shown as black
        im = ax.imshow(display_map, cmap=cmap_hot, vmin=1.0, vmax=2.0)
        ds_prefix = f'{dataset_name} | ' if dataset_name else ''
        ax.set_title(
            f'{ds_prefix}Seq {sequence_id} | {label.upper()} Ratio Heatmap '
            f'(frames {frame_idx}\u2192{frame_idx+1}) | rTC={rtc_value:.4f}',
            fontsize=11
        )
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(r'max($D_t / \hat{D}_{t+1}$, $\hat{D}_{t+1} / D_t$)', fontsize=10)
        plt.tight_layout()

        filename = f'tc_ratio_{label}_seq{sequence_id:04d}.png'
        fig.savefig(save_dir / filename, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        logger.info(f"Saved ratio heatmap: {filename}")

    def save_rtc_plot(self, per_frame_rtc, per_frame_rtc_gt, best_idx, worst_idx,
                      sequence_id, save_dir, dataset_name=''):
        """
        Save per-frame rTC line plot.

        Args:
            per_frame_rtc: list[float] - pred rTC per pair
            per_frame_rtc_gt: list[float] - GT rTC per pair
            best_idx: int
            worst_idx: int
            sequence_id: int
            save_dir: Path
            dataset_name: str - dataset name for title
        """
        n = len(per_frame_rtc)
        if n == 0:
            return

        x = list(range(n))

        fig, ax = plt.subplots(1, 1, figsize=(12, 5))

        ax.plot(x, per_frame_rtc, 'b-o', markersize=3, label='Pred rTC', linewidth=1.5)
        if per_frame_rtc_gt and any(v > 0 for v in per_frame_rtc_gt):
            ax.plot(x, per_frame_rtc_gt, 'g--s', markersize=3, label='GT rTC', linewidth=1.0, alpha=0.7)

        # Mark best/worst
        ax.plot(best_idx, per_frame_rtc[best_idx], 'g^', markersize=12, label=f'Best ({best_idx})', zorder=5)
        ax.plot(worst_idx, per_frame_rtc[worst_idx], 'rv', markersize=12, label=f'Worst ({worst_idx})', zorder=5)

        mean_rtc = np.mean(per_frame_rtc)
        ax.axhline(y=mean_rtc, color='b', linestyle=':', alpha=0.5, label=f'Mean={mean_rtc:.4f}')

        ax.set_xlabel('Frame Pair Index')
        ax.set_ylabel('rTC')
        ds_prefix = f'{dataset_name} | ' if dataset_name else ''
        ax.set_title(f'{ds_prefix}Seq {sequence_id} | Per-Frame rTC (thr={self.thr})')
        ax.legend(loc='lower left', fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = f'tc_rtc_plot_seq{sequence_id:04d}.png'
        fig.savefig(save_dir / filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved rTC plot: {filename}")

    @staticmethod
    def save_multi_threshold_json(all_results, save_dir):
        """
        Save multi-threshold rTC analysis + flickering counts as JSON.

        Aggregates per-sequence multi_threshold results into:
        - Per-sequence: rTC, flicker_count, flicker_rate at each (threshold, cutoff)
        - Dataset aggregate: mean rTC, total flicker_count, mean flicker_rate

        Args:
            all_results: list of dicts, each with 'multi_threshold' key from compute_rtc()
            save_dir: Path to save directory
        """
        from pathlib import Path
        save_dir = Path(save_dir)

        # Collect per-sequence data
        per_sequence = []
        for i, r in enumerate(all_results):
            mt = r.get('multi_threshold') or r.get('_multi_threshold')
            if mt is None or mt == {}:
                continue
            seq_entry = {'sequence_idx': i}
            for thr_key, thr_data in mt.items():
                if thr_key.startswith('_'):
                    continue
                seq_entry[f'thr_{thr_key}'] = {
                    'rtc': thr_data['rtc'],
                    'total_pairs': thr_data['total_pairs'],
                    'flickering': thr_data['flickering'],
                }
            # SCD-excluded multi_threshold (if available)
            mt_excl = r.get('_multi_threshold_excl_scd')
            if mt_excl:
                for thr_key, thr_data in mt_excl.items():
                    if thr_key.startswith('_'):
                        continue
                    seq_entry[f'thr_{thr_key}_excl_scd'] = {
                        'rtc': thr_data['rtc'],
                        'total_pairs': thr_data['total_pairs'],
                        'flickering': thr_data['flickering'],
                    }
            per_sequence.append(seq_entry)

        if not per_sequence:
            logger.warning("No multi-threshold results to save")
            return

        # Get metadata from first result
        meta = all_results[0].get('multi_threshold', {}).get('_meta', {})
        thresholds = meta.get('thresholds', MULTI_THRESHOLDS)
        flicker_cutoffs = meta.get('flicker_cutoffs', FLICKER_CUTOFFS)

        # Aggregate across sequences
        aggregate = {}
        for thr in thresholds:
            thr_key = f"{thr:.2f}"
            rtc_values = []
            flicker_data = {f"{c:.1f}": {'counts': [], 'rates': []} for c in flicker_cutoffs}

            for seq in per_sequence:
                entry = seq.get(f'thr_{thr_key}')
                if entry is None:
                    continue
                rtc_values.append(entry['rtc'])
                for cutoff_key, cutoff_data in entry['flickering'].items():
                    if cutoff_key in flicker_data:
                        flicker_data[cutoff_key]['counts'].append(cutoff_data['count'])
                        flicker_data[cutoff_key]['rates'].append(cutoff_data['rate'])

            flicker_agg = {}
            for cutoff_key, fd in flicker_data.items():
                flicker_agg[cutoff_key] = {
                    'total_count': int(sum(fd['counts'])),
                    'mean_count': float(np.mean(fd['counts'])) if fd['counts'] else 0.0,
                    'mean_rate': float(np.mean(fd['rates'])) if fd['rates'] else 0.0,
                }

            aggregate[f'thr_{thr_key}'] = {
                'mean_rtc': float(np.mean(rtc_values)) if rtc_values else 0.0,
                'num_sequences': len(rtc_values),
                'flickering': flicker_agg,
            }

        # Aggregate excl_scd across sequences
        aggregate_excl_scd = {}
        has_excl_scd = any(f'thr_{thresholds[0]:.2f}_excl_scd' in seq for seq in per_sequence) if thresholds else False
        if has_excl_scd:
            for thr in thresholds:
                thr_key = f"{thr:.2f}"
                rtc_values_ex = []
                flicker_data_ex = {f"{c:.1f}": {'counts': [], 'rates': []} for c in flicker_cutoffs}

                for seq in per_sequence:
                    entry = seq.get(f'thr_{thr_key}_excl_scd')
                    if entry is None:
                        continue
                    rtc_values_ex.append(entry['rtc'])
                    for cutoff_key, cutoff_data in entry['flickering'].items():
                        if cutoff_key in flicker_data_ex:
                            flicker_data_ex[cutoff_key]['counts'].append(cutoff_data['count'])
                            flicker_data_ex[cutoff_key]['rates'].append(cutoff_data['rate'])

                flicker_agg_ex = {}
                for cutoff_key, fd in flicker_data_ex.items():
                    flicker_agg_ex[cutoff_key] = {
                        'total_count': int(sum(fd['counts'])),
                        'mean_count': float(np.mean(fd['counts'])) if fd['counts'] else 0.0,
                        'mean_rate': float(np.mean(fd['rates'])) if fd['rates'] else 0.0,
                    }

                aggregate_excl_scd[f'thr_{thr_key}'] = {
                    'mean_rtc': float(np.mean(rtc_values_ex)) if rtc_values_ex else 0.0,
                    'num_sequences': len(rtc_values_ex),
                    'flickering': flicker_agg_ex,
                }

        output = {
            'meta': meta,
            'aggregate': aggregate,
            'per_sequence': per_sequence,
        }
        if aggregate_excl_scd:
            output['aggregate_excl_scd'] = aggregate_excl_scd

        output_path = save_dir / 'multi_threshold_rtc.json'
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved multi-threshold rTC analysis: {output_path}")

        # Log summary table
        logger.info("=== Multi-threshold rTC & Flickering Summary ===")
        header = f"{'thr':>6}"
        for cutoff in flicker_cutoffs:
            header += f" | rTC<{cutoff:.1f} count"
        logger.info(f"{header} | mean_rTC")
        for thr in thresholds:
            thr_key = f"thr_{thr:.2f}"
            agg = aggregate.get(thr_key, {})
            line = f"{thr:>6.2f}"
            for cutoff in flicker_cutoffs:
                cutoff_key = f"{cutoff:.1f}"
                fc = agg.get('flickering', {}).get(cutoff_key, {})
                line += f" | {fc.get('total_count', 0):>12d}"
            line += f" | {agg.get('mean_rtc', 0):.4f}"
            logger.info(line)
