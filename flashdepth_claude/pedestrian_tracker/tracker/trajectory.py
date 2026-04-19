"""
Trajectory management for tracked pedestrians.

Maintains per-ID trajectory history with EMA smoothing
and provides JSON/TXT export.
"""

import json
import numpy as np
from collections import defaultdict


class TrajectoryManager:
    """
    Manages pedestrian trajectories with EMA-smoothed depth.

    Each trajectory stores (depth_m, lateral_m) pairs per frame.
    """

    def __init__(self, ema_alpha=0.3, max_track_length=300):
        """
        Args:
            ema_alpha: EMA smoothing factor (0=full smooth, 1=no smooth)
            max_track_length: max points per trajectory
        """
        self.ema_alpha = ema_alpha
        self.max_track_length = max_track_length

        self.trajectories = defaultdict(list)  # track_id -> [(depth, lateral), ...]
        self.prev_depth = {}  # track_id -> last smoothed depth
        self.frame_data = defaultdict(dict)  # frame_idx -> {track_id: {depth, lateral}}

    def update(self, track_id, raw_depth, lateral_pos, frame_idx):
        """
        Add a new observation for a tracked pedestrian.

        Args:
            track_id: int tracking ID
            raw_depth: raw metric depth (meters)
            lateral_pos: lateral position (meters)
            frame_idx: current frame index
        """
        # EMA smoothing on depth
        if track_id in self.prev_depth:
            smoothed = self.ema_alpha * raw_depth + (1 - self.ema_alpha) * self.prev_depth[track_id]
        else:
            smoothed = raw_depth

        self.prev_depth[track_id] = smoothed

        self.trajectories[track_id].append((smoothed, lateral_pos))
        self.frame_data[frame_idx][track_id] = {
            'depth': smoothed,
            'lateral': lateral_pos,
            'raw_depth': raw_depth,
        }

        # Trim to max length
        if len(self.trajectories[track_id]) > self.max_track_length:
            self.trajectories[track_id] = self.trajectories[track_id][-self.max_track_length:]

    def get_trajectory(self, track_id):
        """Get trajectory for a specific track ID."""
        return self.trajectories.get(track_id, [])

    def get_all_trajectories(self):
        """Get all trajectories as dict {track_id: [(depth, lateral), ...]}."""
        return dict(self.trajectories)

    def get_last_depth(self, track_id):
        """Get last smoothed depth for a track ID, or None."""
        return self.prev_depth.get(track_id)

    def save_json(self, path):
        """Save all trajectories to JSON."""
        data = {}
        for track_id, traj in self.trajectories.items():
            depths = [t[0] for t in traj]
            laterals = [t[1] for t in traj]
            data[str(track_id)] = {
                'trajectory': [{'depth': d, 'lateral': l} for d, l in traj],
                'num_frames': len(traj),
                'depth_range': [float(min(depths)), float(max(depths))] if depths else [0, 0],
                'lateral_range': [float(min(laterals)), float(max(laterals))] if laterals else [0, 0],
                'avg_depth': float(np.mean(depths)) if depths else 0,
            }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def save_txt(self, path):
        """Save trajectories to text file (compatible with DAv2 format)."""
        with open(path, 'w') as f:
            f.write("ID\tMin\tMax\tAvg\tTrajectory\n")
            for track_id, traj in sorted(self.trajectories.items()):
                depths = [t[0] for t in traj]
                if not depths:
                    continue
                traj_str = ', '.join([f"[{z:.2f},{x:.2f}]" for z, x in traj])
                f.write(f"{track_id}\t{min(depths):.2f}\t{max(depths):.2f}\t"
                        f"{np.mean(depths):.2f}\t{traj_str}\n")
