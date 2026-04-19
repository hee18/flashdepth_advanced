"""
Visualization utilities for pedestrian tracking + depth estimation.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def colorize_depth(depth, cmap_name='Spectral', mask=None):
    """
    Colorize a depth map using a matplotlib colormap.

    Args:
        depth: [H, W] depth values
        cmap_name: matplotlib colormap name
        mask: optional [H, W] binary mask to zero out background

    Returns:
        colored: [H, W, 3] uint8 RGB image
    """
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min < 1e-6:
        d_max = d_min + 1.0
    normalized = (depth - d_min) / (d_max - d_min)

    cmap = matplotlib.colormaps[cmap_name]
    colored = (cmap(normalized)[:, :, :3] * 255).astype(np.uint8)

    if mask is not None:
        colored[mask == 0] = 0

    return colored


def draw_detections(frame, detections_info, depth_overlay=None,
                    overlay_alpha=0.6, show_bbox=True, show_text=True):
    """
    Draw detection results on a frame.

    Args:
        frame: [H, W, 3] BGR uint8 image
        detections_info: list of dicts with 'track_id', 'bbox', 'depth', 'lateral'
        depth_overlay: optional [H, W, 3] BGR depth colormap to overlay
        overlay_alpha: alpha for depth overlay blending
        show_bbox: draw bounding boxes
        show_text: draw ID/depth/lateral text

    Returns:
        vis: [H, W, 3] BGR uint8 annotated image
    """
    vis = frame.copy()

    # Overlay depth colormap if provided
    if depth_overlay is not None:
        vis = cv2.addWeighted(vis, 1.0, depth_overlay, overlay_alpha, 0)

    for det in detections_info:
        track_id = det['track_id']
        box = det['bbox']
        depth = det.get('depth', 0)
        lateral = det.get('lateral', 0)

        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        if show_bbox:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if show_text:
            text = f"ID:{track_id} Z:{depth:.1f}m X:{lateral:.1f}m"
            cv2.putText(vis, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return vis


def plot_trajectories(trajectories, save_path, title='Estimated Pedestrian Trajectories'):
    """
    Plot estimated trajectories (lateral vs depth, bird's-eye view).

    Args:
        trajectories: dict {track_id: [(depth, lateral), ...]}
        save_path: path to save the plot
        title: plot title
    """
    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap('tab10')

    sorted_ids = sorted(trajectories.keys())
    for idx, track_id in enumerate(sorted_ids):
        traj = trajectories[track_id]
        if len(traj) < 2:
            continue

        depths = [t[0] for t in traj]
        laterals = [t[1] for t in traj]
        color = cmap(idx % 10)

        plt.plot(laterals, depths, 'o-', label=f'ID {track_id}',
                 color=color, markersize=3, linewidth=1.5)

        # Mark start and end
        plt.plot(laterals[0], depths[0], 's', color=color, markersize=8)
        plt.plot(laterals[-1], depths[-1], '^', color=color, markersize=8)

    plt.xlabel('Lateral Position (m)')
    plt.ylabel('Depth (m)')
    plt.title(title)
    plt.legend(loc='best', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150)
    plt.close()
