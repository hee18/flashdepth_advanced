"""
Flow-based Temporal Consistency (rTC) metric.

Implements rTC from "Enforcing Temporal Consistency in Video Depth Estimation":
    rTC_i = (1/sum(M_i)) * sum(M_i * [max(D_i/D_hat_{i+1}, D_hat_{i+1}/D_i) < thr])

Uses SEA-RAFT optical flow to warp depth maps between consecutive frames and
measures the ratio of temporally consistent pixels.

Complements reprojection-based TAE (which requires camera poses).
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

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

    def __init__(self, device='cuda:0', thr=1.1, max_depth=70.0, checkpoint_path=None):
        """
        Args:
            device: torch device
            thr: ratio threshold for rTC (default 1.1)
            max_depth: maximum valid depth in meters
            checkpoint_path: path to SEA-RAFT weights (auto-detected if None)
        """
        self.device = device
        self.thr = thr
        self.max_depth = max_depth
        self.checkpoint_path = checkpoint_path
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

        for t in range(T - 1):
            # Estimate forward flow: frame_t -> frame_{t+1}
            frame_t = images_01[t:t+1]    # [1, 3, H, W]
            frame_tp1 = images_01[t+1:t+2]  # [1, 3, H, W]
            flow, _ = flow_estimator.estimate_flow(frame_t, frame_tp1)  # [1, 2, H, W]

            # === Pred rTC ===
            rtc_val, ratio_stats = self._compute_pair_rtc(
                pred_d[t:t+1], pred_d[t+1:t+2], flow
            )
            per_frame_rtc.append(rtc_val)
            per_frame_ratio_stats.append(ratio_stats)
            if ratio_stats.get('_valid_ratios') is not None:
                all_valid_ratios.append(ratio_stats['_valid_ratios'])

            # === GT rTC (oracle) ===
            if gt_d is not None:
                rtc_gt_val, _ = self._compute_pair_rtc(
                    gt_d[t:t+1], gt_d[t+1:t+2], flow
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

        return {
            'rtc': mean_rtc,
            'rtc_gt': mean_rtc_gt,
            'per_frame_rtc': [float(x) for x in per_frame_rtc],
            'per_frame_rtc_gt': [float(x) for x in per_frame_rtc_gt],
            'per_frame_ratio_stats': clean_ratio_stats,
            'ratio_stats': agg_ratio_stats,
            'best_frame_idx': best_idx,
            'worst_frame_idx': worst_idx
        }

    def _compute_pair_rtc(self, depth_i, depth_ip1, flow):
        """
        Compute rTC for a single frame pair (i, i+1).

        Args:
            depth_i: [1, 1, H, W] depth at frame i
            depth_ip1: [1, 1, H, W] depth at frame i+1
            flow: [1, 2, H, W] forward optical flow from i to i+1

        Returns:
            rtc: float - ratio of consistent pixels
            ratio_stats: dict - statistics of depth ratios
        """
        _, _, H, W = depth_i.shape

        # Create sampling grid: pixel coordinates + flow
        # grid_sample expects normalized coordinates in [-1, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=flow.device, dtype=flow.dtype),
            torch.arange(W, device=flow.device, dtype=flow.dtype),
            indexing='ij'
        )
        grid_x = grid_x.unsqueeze(0)  # [1, H, W]
        grid_y = grid_y.unsqueeze(0)  # [1, H, W]

        # Apply flow: where does pixel (x, y) in frame i go in frame i+1?
        # flow[0] = horizontal (x), flow[1] = vertical (y)
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

        # Validity mask M_i
        in_bounds = (warped_x >= 0) & (warped_x < W) & (warped_y >= 0) & (warped_y < H)
        in_bounds = in_bounds.unsqueeze(1)  # [1, 1, H, W]

        d_i = depth_i
        d_hat = warped_depth

        valid_mask = (
            (d_i > 0) & (d_i < self.max_depth) &
            (d_hat > 0) & (d_hat < self.max_depth) &
            in_bounds
        )

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
        flow, _ = flow_estimator.estimate_flow(images_01[t:t+1], images_01[t+1:t+2])

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

        valid = (d_i > 0) & (d_i < self.max_depth) & (d_hat > 0) & (d_hat < self.max_depth) & in_bounds[0]

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
