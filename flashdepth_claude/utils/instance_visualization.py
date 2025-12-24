"""
Instance Visualization Utilities

인스턴스 depth 추적 결과 시각화를 위한 유틸리티 함수들.
프레임 시각화, trajectory plot, 비디오 저장 기능 제공.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless servers
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import json
import logging

logger = logging.getLogger(__name__)


def colorize_depth(depth: np.ndarray,
                   cmap_name: str = 'Spectral',
                   apply_mask: Optional[np.ndarray] = None,
                   vmin: Optional[float] = None,
                   vmax: Optional[float] = None) -> np.ndarray:
    """
    Depth map을 컬러맵으로 시각화합니다.

    Args:
        depth: Depth map (H, W)
        cmap_name: Matplotlib colormap name (default: 'Spectral')
        apply_mask: Optional binary mask to apply (H, W)
        vmin, vmax: Optional depth range for normalization

    Returns:
        Colorized depth image (H, W, 3) RGB uint8
    """
    # Depth normalization
    if vmin is None:
        vmin = np.min(depth[depth > 0]) if np.any(depth > 0) else 0
    if vmax is None:
        vmax = np.max(depth[depth < 1000]) if np.any(depth < 1000) else depth.max()

    # Avoid division by zero
    depth_range = vmax - vmin
    if depth_range < 1e-6:
        depth_range = 1.0

    depth_normalized = np.clip((depth - vmin) / depth_range, 0, 1)

    # Apply colormap
    cmap = matplotlib.colormaps[cmap_name]
    depth_colored = (cmap(depth_normalized)[:, :, :3] * 255).astype(np.uint8)

    # Apply mask if provided
    if apply_mask is not None:
        depth_colored[apply_mask == 0] = 0

    return depth_colored


def create_frame_visualization(frame: np.ndarray,
                                depth_map: np.ndarray,
                                instances_info: List[Dict[str, Any]],
                                alpha: float = 0.6,
                                show_depth_colorbar: bool = False,
                                show_depth_values: bool = True) -> np.ndarray:
    """
    프레임에 depth overlay와 instance 정보를 시각화합니다.

    Args:
        frame: Original BGR frame (H, W, 3)
        depth_map: Depth map (H, W) in meters
        instances_info: List of dicts with track_id, depth, lateral_pos, box, mask
        alpha: Depth overlay alpha (default: 0.6)
        show_depth_colorbar: Whether to add colorbar (default: False)
        show_depth_values: Whether to show Z/X depth values on labels (default: True)
                          If False, only shows track ID

    Returns:
        Visualization frame (H, W, 3) BGR
    """
    vis_frame = frame.copy()

    # Create combined mask for all instances
    combined_mask = np.zeros(depth_map.shape, dtype=np.uint8)
    for inst in instances_info:
        if 'mask' in inst and inst['mask'] is not None:
            combined_mask = np.maximum(combined_mask, inst['mask'])

    # Colorize depth and blend if there are instances
    if combined_mask.sum() > 0:
        depth_colored = colorize_depth(depth_map, apply_mask=combined_mask)
        # Convert RGB to BGR for OpenCV
        depth_colored_bgr = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)

        # Create blend mask (3 channels)
        blend_mask = np.stack([combined_mask] * 3, axis=-1).astype(np.float32)
        vis_frame = (vis_frame * (1 - blend_mask * alpha) +
                     depth_colored_bgr * blend_mask * alpha).astype(np.uint8)

    # Draw instance info
    for inst in instances_info:
        track_id = inst.get('track_id', -1)
        depth = inst.get('depth', 0)
        lat_pos = inst.get('lateral_pos', 0)
        box = inst.get('box', None)

        # Draw bounding box
        if box is not None:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            color = get_track_color(track_id)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

            # Draw ID and depth info with background
            if show_depth_values:
                label = f"ID:{track_id} Z:{depth:.1f}m X:{lat_pos:.1f}m"
            else:
                label = f"ID:{track_id}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # Background rectangle
            cv2.rectangle(vis_frame,
                         (x1, y1 - label_h - 10),
                         (x1 + label_w + 4, y1 - 2),
                         color, -1)

            # Text
            cv2.putText(vis_frame, label,
                       (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis_frame


def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """
    Track ID에 따른 고유 색상을 반환합니다.

    Args:
        track_id: Track ID

    Returns:
        BGR color tuple
    """
    # Tab10 colormap colors in BGR
    colors = [
        (255, 127, 14),   # Orange
        (44, 160, 44),    # Green
        (214, 39, 40),    # Red
        (148, 103, 189),  # Purple
        (140, 86, 75),    # Brown
        (227, 119, 194),  # Pink
        (127, 127, 127),  # Gray
        (188, 189, 34),   # Olive
        (23, 190, 207),   # Cyan
        (31, 119, 180),   # Blue
    ]
    return colors[track_id % len(colors)]


def create_trajectory_plot(track_trajectories: Dict[int, List[Dict]],
                           output_path: Path,
                           title: str = 'Instance Depth Trajectories',
                           figsize: Tuple[int, int] = (12, 8),
                           min_frames: int = 30,
                           is_metric: bool = True) -> None:
    """
    Instance trajectory를 depth vs lateral position 그래프로 시각화합니다.

    Args:
        track_trajectories: Dict[track_id -> list of trajectory points]
        output_path: Path to save plot
        title: Plot title
        figsize: Figure size
        min_frames: Minimum number of frames required for a track (default: 30)
        is_metric: If True, use "(m)" units; if False, use "(rel.)" for relative depth
    """
    plt.figure(figsize=figsize)
    cmap = plt.get_cmap('tab10')

    # Unit label based on depth type
    unit_label = "(m)" if is_metric else "(rel.)"

    valid_tracks = 0
    for idx, (track_id, traj) in enumerate(sorted(track_trajectories.items())):
        # Skip tracks with fewer than min_frames
        if len(traj) < min_frames:
            continue

        # Filter valid depths
        valid_traj = [p for p in traj if p.get('depth_m', 1000) < 1000]
        if len(valid_traj) < min_frames:
            continue

        depths = [p['depth_m'] for p in valid_traj]
        laterals = [p['lateral_m'] for p in valid_traj]
        frames = [p.get('frame', i) for i, p in enumerate(valid_traj)]

        # Plot trajectory
        color = cmap(idx % 10)
        plt.plot(laterals, depths, 'o-',
                label=f'Person {track_id} ({len(valid_traj)} frames)',
                color=color, markersize=4, linewidth=1.5)

        # Mark start and end
        plt.scatter([laterals[0]], [depths[0]], s=100, c=[color],
                   marker='s', edgecolors='black', linewidths=1.5, zorder=5)
        plt.scatter([laterals[-1]], [depths[-1]], s=100, c=[color],
                   marker='^', edgecolors='black', linewidths=1.5, zorder=5)

        valid_tracks += 1

    if valid_tracks == 0:
        plt.text(0.5, 0.5, 'No valid trajectories',
                ha='center', va='center', transform=plt.gca().transAxes)

    plt.xlabel(f'Lateral Position {unit_label}', fontsize=12)
    plt.ylabel(f'Depth {unit_label}', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    # Y-axis: smaller depth at bottom, larger depth at top (default matplotlib behavior)

    # Add legend for markers
    if valid_tracks > 0:
        plt.plot([], [], 's', color='gray', markersize=8, label='Start')
        plt.plot([], [], '^', color='gray', markersize=8, label='End')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Trajectory plot saved to {output_path}")


def create_depth_timeline_plot(track_trajectories: Dict[int, List[Dict]],
                                output_path: Path,
                                title: str = 'Depth Over Time',
                                figsize: Tuple[int, int] = (14, 6),
                                min_frames: int = 30,
                                is_metric: bool = True) -> None:
    """
    시간에 따른 depth 변화를 그래프로 시각화합니다.

    Args:
        track_trajectories: Dict[track_id -> list of trajectory points]
        output_path: Path to save plot
        title: Plot title
        figsize: Figure size
        min_frames: Minimum number of frames required for a track (default: 30)
        is_metric: If True, use "(m)" units; if False, use "(rel.)" for relative depth
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    cmap = plt.get_cmap('tab10')

    # Unit label based on depth type
    unit_label = "(m)" if is_metric else "(rel.)"

    for idx, (track_id, traj) in enumerate(sorted(track_trajectories.items())):
        # Skip tracks with fewer than min_frames
        if len(traj) < min_frames:
            continue

        valid_traj = [p for p in traj if p.get('depth_m', 1000) < 1000]
        if len(valid_traj) < min_frames:
            continue

        frames = [p.get('frame', i) for i, p in enumerate(valid_traj)]
        depths = [p['depth_m'] for p in valid_traj]
        laterals = [p['lateral_m'] for p in valid_traj]

        color = cmap(idx % 10)

        # Depth over time
        ax1.plot(frames, depths, 'o-', label=f'Person {track_id}',
                color=color, markersize=3, linewidth=1.5)

        # Lateral position over time
        ax2.plot(frames, laterals, 'o-', label=f'Person {track_id}',
                color=color, markersize=3, linewidth=1.5)

    ax1.set_xlabel('Frame', fontsize=11)
    ax1.set_ylabel(f'Depth {unit_label}', fontsize=11)
    ax1.set_title('Depth Over Time', fontsize=12)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Frame', fontsize=11)
    ax2.set_ylabel(f'Lateral Position {unit_label}', fontsize=11)
    ax2.set_title('Lateral Position Over Time', fontsize=12)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Timeline plot saved to {output_path}")


def create_scale_shift_timeline_plot(scale_shift_history: List[Dict],
                                      output_path: Path,
                                      title: str = 'Scale/Shift Over Time',
                                      figsize: Tuple[int, int] = (14, 6)) -> None:
    """
    Gear5 모델의 프레임별 scale/shift 변화를 그래프로 시각화합니다.

    Args:
        scale_shift_history: List of {frame, scale, shift} dicts
        output_path: Path to save plot
        title: Plot title
        figsize: Figure size
    """
    if len(scale_shift_history) == 0:
        logger.warning("No scale/shift data to plot")
        return

    # Check if this is sparse GT alignment mode (no per-frame data)
    if any(p.get('type') == 'sparse_gt_alignment' for p in scale_shift_history):
        logger.info("Sparse GT alignment mode - skipping timeline plot (single alignment, not per-frame)")
        return

    frames = [p['frame'] for p in scale_shift_history]
    scales = [p['scale'] for p in scale_shift_history]
    shifts = [p['shift'] for p in scale_shift_history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Scale over time
    ax1.plot(frames, scales, 'b-', linewidth=1.5, alpha=0.7)
    ax1.scatter(frames, scales, c='blue', s=10, alpha=0.5)
    ax1.set_xlabel('Frame', fontsize=11)
    ax1.set_ylabel('Scale', fontsize=11)
    ax1.set_title('Scale Over Time', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Add statistics
    scale_mean = np.mean(scales)
    scale_std = np.std(scales)
    ax1.axhline(y=scale_mean, color='red', linestyle='--', alpha=0.7,
                label=f'Mean: {scale_mean:.4f}')
    ax1.fill_between(frames, scale_mean - scale_std, scale_mean + scale_std,
                     color='red', alpha=0.1, label=f'Std: {scale_std:.4f}')
    ax1.legend(loc='best', fontsize=9)

    # Shift over time
    ax2.plot(frames, shifts, 'g-', linewidth=1.5, alpha=0.7)
    ax2.scatter(frames, shifts, c='green', s=10, alpha=0.5)
    ax2.set_xlabel('Frame', fontsize=11)
    ax2.set_ylabel('Shift', fontsize=11)
    ax2.set_title('Shift Over Time', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Add statistics
    shift_mean = np.mean(shifts)
    shift_std = np.std(shifts)
    ax2.axhline(y=shift_mean, color='red', linestyle='--', alpha=0.7,
                label=f'Mean: {shift_mean:.4f}')
    ax2.fill_between(frames, shift_mean - shift_std, shift_mean + shift_std,
                     color='red', alpha=0.1, label=f'Std: {shift_std:.4f}')
    ax2.legend(loc='best', fontsize=9)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Scale/shift timeline plot saved to {output_path}")


def compute_depth_variation(depth_maps: List[np.ndarray]) -> Dict:
    """
    프레임별 depth 변동량(이전 프레임과의 절대/상대 차이)을 계산합니다.

    Args:
        depth_maps: List of depth maps (H, W)

    Returns:
        Dict containing:
            - per_frame: List of {frame, mean_variation, max_variation, mean_variation_pct, max_variation_pct, valid_ratio}
            - statistics_abs: {max, min, mean, std, sum} for absolute variation
            - statistics_pct: {max, min, mean, std, sum} for relative variation (%)
    """
    if len(depth_maps) < 2:
        empty_stats = {'max': 0.0, 'min': 0.0, 'mean': 0.0, 'std': 0.0, 'sum': 0.0}
        return {
            'per_frame': [],
            'statistics_abs': empty_stats,
            'statistics_pct': empty_stats,
            'statistics': empty_stats  # backward compatibility
        }

    per_frame = []
    all_mean_variations_abs = []
    all_mean_variations_pct = []

    for i in range(1, len(depth_maps)):
        prev_depth = depth_maps[i - 1]
        curr_depth = depth_maps[i]

        # Compute absolute difference
        diff_abs = np.abs(curr_depth - prev_depth)

        # Valid mask: exclude very large depths and zeros
        valid_mask = (prev_depth > 0) & (prev_depth < 200) & (curr_depth > 0) & (curr_depth < 200)
        valid_ratio = np.sum(valid_mask) / valid_mask.size if valid_mask.size > 0 else 0

        if np.sum(valid_mask) > 0:
            valid_diff_abs = diff_abs[valid_mask]
            valid_prev = prev_depth[valid_mask]
            
            # Absolute variation
            mean_var_abs = float(np.mean(valid_diff_abs))
            max_var_abs = float(np.max(valid_diff_abs))
            
            # Relative variation (%) = |curr - prev| / prev * 100
            relative_diff = valid_diff_abs / (valid_prev + 1e-8) * 100.0
            mean_var_pct = float(np.mean(relative_diff))
            max_var_pct = float(np.max(relative_diff))
        else:
            mean_var_abs = 0.0
            max_var_abs = 0.0
            mean_var_pct = 0.0
            max_var_pct = 0.0

        per_frame.append({
            'frame': i,
            'mean_variation': mean_var_abs,
            'max_variation': max_var_abs,
            'mean_variation_pct': mean_var_pct,
            'max_variation_pct': max_var_pct,
            'valid_ratio': float(valid_ratio)
        })
        all_mean_variations_abs.append(mean_var_abs)
        all_mean_variations_pct.append(mean_var_pct)

    # Compute overall statistics for absolute variation
    variations_abs = np.array(all_mean_variations_abs)
    statistics_abs = {
        'max': float(np.max(variations_abs)),
        'min': float(np.min(variations_abs)),
        'mean': float(np.mean(variations_abs)),
        'std': float(np.std(variations_abs)),
        'sum': float(np.sum(variations_abs))
    }

    # Compute overall statistics for relative variation (%)
    variations_pct = np.array(all_mean_variations_pct)
    statistics_pct = {
        'max': float(np.max(variations_pct)),
        'min': float(np.min(variations_pct)),
        'mean': float(np.mean(variations_pct)),
        'std': float(np.std(variations_pct)),
        'sum': float(np.sum(variations_pct))
    }

    return {
        'per_frame': per_frame,
        'statistics_abs': statistics_abs,
        'statistics_pct': statistics_pct,
        'statistics': statistics_abs  # backward compatibility
    }


def create_depth_variation_plot(depth_variation: Dict,
                                 output_path: Path,
                                 title: str = 'Depth Variation Over Time',
                                 figsize: Tuple[int, int] = (14, 8),
                                 is_metric: bool = True) -> None:
    """
    프레임별 depth 변동량(절대/상대)을 그래프로 시각화합니다.

    Args:
        depth_variation: Output from compute_depth_variation()
        output_path: Path to save plot
        title: Plot title
        figsize: Figure size
        is_metric: If True, use "(m)" units; if False, use "(rel.)"
    """
    per_frame = depth_variation.get('per_frame', [])
    stats_abs = depth_variation.get('statistics_abs', depth_variation.get('statistics', {}))
    stats_pct = depth_variation.get('statistics_pct', {})

    if len(per_frame) == 0:
        logger.warning("No depth variation data to plot")
        return

    frames = [p['frame'] for p in per_frame]
    mean_vars_abs = [p['mean_variation'] for p in per_frame]
    mean_vars_pct = [p.get('mean_variation_pct', 0) for p in per_frame]

    unit_label = "(m)" if is_metric else "(rel.)"

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

    # Top left: Mean absolute variation over time
    ax1.plot(frames, mean_vars_abs, 'b-', linewidth=1.5, alpha=0.7)
    ax1.scatter(frames, mean_vars_abs, c='blue', s=10, alpha=0.5)
    ax1.set_xlabel('Frame', fontsize=10)
    ax1.set_ylabel(f'Mean Variation {unit_label}', fontsize=10)
    ax1.set_title('Absolute Variation', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Add absolute statistics annotation
    stat_text_abs = (f"Max: {stats_abs.get('max', 0):.4f}\n"
                     f"Min: {stats_abs.get('min', 0):.4f}\n"
                     f"Mean: {stats_abs.get('mean', 0):.4f}\n"
                     f"Std: {stats_abs.get('std', 0):.4f}\n"
                     f"Sum: {stats_abs.get('sum', 0):.2f}")
    ax1.text(0.02, 0.98, stat_text_abs, transform=ax1.transAxes, fontsize=8,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    mean_val_abs = stats_abs.get('mean', 0)
    ax1.axhline(y=mean_val_abs, color='red', linestyle='--', alpha=0.7)

    # Top right: Mean relative variation (%) over time
    ax2.plot(frames, mean_vars_pct, 'g-', linewidth=1.5, alpha=0.7)
    ax2.scatter(frames, mean_vars_pct, c='green', s=10, alpha=0.5)
    ax2.set_xlabel('Frame', fontsize=10)
    ax2.set_ylabel('Mean Variation (%)', fontsize=10)
    ax2.set_title('Relative Variation (%)', fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Add relative statistics annotation
    stat_text_pct = (f"Max: {stats_pct.get('max', 0):.2f}%\n"
                     f"Min: {stats_pct.get('min', 0):.2f}%\n"
                     f"Mean: {stats_pct.get('mean', 0):.2f}%\n"
                     f"Std: {stats_pct.get('std', 0):.2f}%\n"
                     f"Sum: {stats_pct.get('sum', 0):.1f}%")
    ax2.text(0.02, 0.98, stat_text_pct, transform=ax2.transAxes, fontsize=8,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    mean_val_pct = stats_pct.get('mean', 0)
    ax2.axhline(y=mean_val_pct, color='red', linestyle='--', alpha=0.7)

    # Bottom left: Histogram of absolute variations
    ax3.hist(mean_vars_abs, bins=30, color='blue', alpha=0.7, edgecolor='black')
    ax3.set_xlabel(f'Variation {unit_label}', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title('Absolute Variation Distribution', fontsize=11)
    ax3.axvline(x=mean_val_abs, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val_abs:.4f}')
    ax3.legend(fontsize=8)

    # Bottom right: Histogram of relative variations (%)
    ax4.hist(mean_vars_pct, bins=30, color='green', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Variation (%)', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.set_title('Relative Variation Distribution', fontsize=11)
    ax4.axvline(x=mean_val_pct, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val_pct:.2f}%')
    ax4.legend(fontsize=8)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Depth variation plot saved to {output_path}")


def save_video_result(frames: List[np.ndarray],
                      output_path: Path,
                      fps: int = 9,
                      frame_interval: int = 1) -> bool:
    """
    결과 프레임들을 MP4 비디오로 저장합니다.

    Args:
        frames: List of BGR frames
        output_path: Output video path
        fps: Frames per second
        frame_interval: Save every Nth frame (default: 1 = all frames)

    Returns:
        True if successful, False otherwise
    """
    if len(frames) == 0:
        logger.warning("No frames to save")
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Select frames based on interval
    selected_frames = [frames[i] for i in range(0, len(frames), frame_interval)]
    if len(selected_frames) == 0:
        logger.warning("No frames selected after interval filtering")
        return False

    h, w = selected_frames[0].shape[:2]

    # Try MP4 first
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        if not writer.isOpened():
            raise RuntimeError("VideoWriter failed to open")

        for frame in selected_frames:
            writer.write(frame)

        writer.release()
        logger.info(f"Video saved to {output_path} ({len(selected_frames)} frames)")
        return True

    except Exception as e:
        logger.warning(f"MP4 export failed: {e}")

        # Fallback: save as individual frames
        frames_dir = output_path.parent / 'frames'
        frames_dir.mkdir(exist_ok=True)

        for i, frame in enumerate(selected_frames):
            frame_idx = i * frame_interval
            frame_path = frames_dir / f'frame_{frame_idx:04d}.png'
            cv2.imwrite(str(frame_path), frame)

        logger.info(f"Frames saved to {frames_dir} ({len(selected_frames)} frames)")
        return False


def save_json_results(track_trajectories: Dict[int, List[Dict]],
                      video_info: Dict[str, Any],
                      output_path: Path,
                      depth_model: str = 'unknown',
                      processing_time: float = 0.0,
                      min_frames: int = 30) -> None:
    """
    Instance tracking 결과를 JSON 파일로 저장합니다.

    Args:
        track_trajectories: Dict[track_id -> list of trajectory points]
        video_info: Dict with video metadata (name, fps, resolution, etc.)
        output_path: Output JSON path
        depth_model: Name of depth model used
        processing_time: Total processing time in seconds
        min_frames: Minimum number of frames required for a track (default: 30)
    """
    from utils.instance_depth_utils import compute_instance_statistics

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build instances dict with statistics (filter by min_frames)
    instances = {}
    for track_id, traj in track_trajectories.items():
        # Skip tracks with fewer than min_frames
        if len(traj) < min_frames:
            continue

        # Also check valid depth frames
        valid_frames = [p['frame'] for p in traj if p.get('depth_m', 1000) < 1000]
        if len(valid_frames) < min_frames:
            continue

        stats = compute_instance_statistics(traj)
        first_frame = min(valid_frames) if valid_frames else 0
        last_frame = max(valid_frames) if valid_frames else 0

        instances[str(track_id)] = {
            'class': 'person',
            'first_frame': first_frame,
            'last_frame': last_frame,
            'trajectory': traj,
            'statistics': stats
        }

    result = {
        'video_name': video_info.get('name', 'unknown'),
        'total_frames': video_info.get('total_frames', 0),
        'fps': video_info.get('fps', 9),
        'resolution': video_info.get('resolution', [0, 0]),
        'camera_intrinsics': video_info.get('intrinsics', {}),
        'segmentation_model': 'yolo11x-seg.pt',
        'tracker': 'botsort',
        'depth_model': depth_model,
        'instances': instances,
        'num_instances': len(instances),
        'processing_time_sec': round(processing_time, 2)
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {output_path}")


def export_frame_images(frames: List[np.ndarray],
                        output_dir: Path,
                        frame_interval: int = 10,
                        prefix: str = 'frame') -> None:
    """
    개별 프레임들을 이미지로 저장합니다.

    Args:
        frames: List of BGR frames
        output_dir: Output directory
        frame_interval: Save every Nth frame
        prefix: Filename prefix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for i in range(0, len(frames), frame_interval):
        frame_path = output_dir / f'{prefix}_{i:04d}.png'
        cv2.imwrite(str(frame_path), frames[i])
        saved_count += 1

    logger.info(f"Exported {saved_count} frames to {output_dir}")


def colorize_full_depth(depth: np.ndarray,
                         max_depth: float = 80.0,
                         cmap_name: str = 'plasma_r',
                         percentile_low: float = 2.0,
                         percentile_high: float = 98.0) -> np.ndarray:
    """
    전체 프레임의 depth를 colormap으로 시각화합니다 (max_depth 미만인 픽셀만).

    Args:
        depth: Depth map (H, W) in meters
        max_depth: Maximum depth threshold (default: 80m)
        cmap_name: Matplotlib colormap name (default: 'plasma_r')
        percentile_low: Lower percentile for normalization (default: 2)
        percentile_high: Upper percentile for normalization (default: 98)

    Returns:
        Colorized depth image (H, W, 3) RGB uint8
    """
    # Create valid mask: depth > 0 and depth < max_depth
    valid_mask = (depth > 0) & (depth < max_depth)

    if not np.any(valid_mask):
        # Return black image if no valid pixels
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    # Get valid depths for percentile calculation
    valid_depths = depth[valid_mask]

    # Calculate percentile bounds
    vmin = np.percentile(valid_depths, percentile_low)
    vmax = np.percentile(valid_depths, percentile_high)

    # Avoid division by zero
    depth_range = vmax - vmin
    if depth_range < 1e-6:
        depth_range = 1.0

    # Normalize depth (only valid pixels)
    depth_normalized = np.zeros_like(depth)
    depth_normalized[valid_mask] = np.clip(
        (depth[valid_mask] - vmin) / depth_range, 0, 1
    )

    # Apply colormap
    cmap = matplotlib.colormaps[cmap_name]
    depth_colored = (cmap(depth_normalized)[:, :, :3] * 255).astype(np.uint8)

    # Set invalid pixels (>= max_depth or <= 0) to black
    depth_colored[~valid_mask] = 0

    return depth_colored


def create_depth_colormap_frame(frame: np.ndarray,
                                 depth_map: np.ndarray,
                                 max_depth: float = 80.0,
                                 alpha: float = 0.0) -> np.ndarray:
    """
    프레임에 depth colormap을 오버레이하거나 depth만 표시합니다.

    Args:
        frame: Original BGR frame (H, W, 3)
        depth_map: Depth map (H, W) in meters
        max_depth: Maximum depth threshold (default: 80m)
        alpha: Blend factor (0=depth only, 1=frame only, 0.5=blended)

    Returns:
        Visualization frame (H, W, 3) BGR
    """
    # Colorize depth
    depth_colored = colorize_full_depth(depth_map, max_depth=max_depth)

    # Convert RGB to BGR for OpenCV
    depth_colored_bgr = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)

    if alpha <= 0:
        return depth_colored_bgr
    elif alpha >= 1:
        return frame.copy()
    else:
        # Blend frame with depth colormap
        return cv2.addWeighted(frame, alpha, depth_colored_bgr, 1 - alpha, 0)


def save_depth_colormap_video(depth_maps: List[np.ndarray],
                               output_path: Path,
                               fps: int = 9,
                               max_depth: float = 80.0,
                               frames: Optional[List[np.ndarray]] = None,
                               alpha: float = 0.0,
                               frame_interval: int = 1) -> bool:
    """
    Depth map 시퀀스를 colormap 비디오로 저장합니다.

    Args:
        depth_maps: List of depth maps (H, W) in meters
        output_path: Output video path
        fps: Frames per second
        max_depth: Maximum depth threshold (default: 80m)
        frames: Optional original frames for blending
        alpha: Blend factor (0=depth only, 1=frame only)
        frame_interval: Save every Nth frame (default: 1 = all frames)

    Returns:
        True if successful, False otherwise
    """
    if len(depth_maps) == 0:
        logger.warning("No depth maps to save")
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Select frames based on interval
    indices = list(range(0, len(depth_maps), frame_interval))
    selected_depths = [depth_maps[i] for i in indices]
    selected_frames = [frames[i] for i in indices] if frames else None

    if len(selected_depths) == 0:
        logger.warning("No depth maps selected after interval filtering")
        return False

    h, w = selected_depths[0].shape[:2]

    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        if not writer.isOpened():
            raise RuntimeError("VideoWriter failed to open")

        for i, depth in enumerate(selected_depths):
            if selected_frames is not None:
                vis_frame = create_depth_colormap_frame(
                    selected_frames[i], depth, max_depth=max_depth, alpha=alpha
                )
            else:
                # Depth only
                depth_colored = colorize_full_depth(depth, max_depth=max_depth)
                vis_frame = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)

            writer.write(vis_frame)

        writer.release()
        logger.info(f"Depth colormap video saved to {output_path} ({len(selected_depths)} frames)")
        return True

    except Exception as e:
        logger.warning(f"Depth colormap video export failed: {e}")
        return False
