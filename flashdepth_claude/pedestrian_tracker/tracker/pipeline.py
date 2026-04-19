"""
Main pipeline: YOLO tracking + Onepiece depth estimation.

Phase 1: Sequential execution (YOLO → Depth).
Phase 2 (TRT): CUDA multi-stream parallel execution.
"""

import cv2
import os
import time
import logging
import numpy as np

from tracker.depth_estimator import OnepieceDepthEstimator
from tracker.pedestrian_tracker import PedestrianTracker
from tracker.trajectory import TrajectoryManager
from tracker.visualization import colorize_depth, draw_detections, plot_trajectories
from ped_utils.camera import calculate_lateral_position
from ped_utils.mask_ops import create_mask_from_polygon, erode_mask, get_mask_center, extract_depth_from_mask

logger = logging.getLogger(__name__)


class PedestrianPipeline:
    """
    Full pipeline: video → YOLO seg+track → Onepiece depth → trajectories.
    """

    def __init__(self, config):
        """
        Args:
            config: dict with all configuration parameters
        """
        self.config = config
        paths = config['paths']
        cam = config['camera']
        depth_cfg = config['depth']
        yolo_cfg = config['yolo']
        traj_cfg = config['trajectory']
        vis_cfg = config['visualization']

        # Depth settings
        self.input_height = depth_cfg['input_size']  # 518 (must be multiple of 14)
        self.max_depth = depth_cfg['max_depth']

        # Camera intrinsics (will be rescaled in run() after computing target resolution)
        self.orig_fx = cam['fx']
        self.orig_fy = cam['fy']
        self.orig_cx = cam['cx']
        self.orig_cy = cam['cy']

        # Trajectory settings
        self.use_eroded_mask = traj_cfg['use_eroded_mask']
        self.erode_kernel = traj_cfg['erode_kernel_size']
        self.erode_iters = traj_cfg['erode_iterations']

        # Visualization settings
        self.vis_config = vis_cfg

        # Initialize components
        logger.info("Initializing depth estimator...")
        self.depth_estimator = OnepieceDepthEstimator(
            config_path=paths['onepiece_config'],
            checkpoint_path=paths['onepiece_checkpoint'],
            use_bfloat16=depth_cfg.get('use_bfloat16', True),
        )

        logger.info("Initializing pedestrian tracker...")
        self.tracker = PedestrianTracker(
            model_path=paths['yolo_model'],
            confidence=yolo_cfg['confidence'],
            iou_threshold=yolo_cfg['iou_threshold'],
            tracker=yolo_cfg['tracker'],
            person_only=yolo_cfg['person_only'],
            # imgsz set later in run() after computing target resolution
        )

        self.trajectory_mgr = TrajectoryManager(
            ema_alpha=traj_cfg['ema_alpha'],
            max_track_length=traj_cfg['max_track_length'],
        )

    def run(self, video_path, output_dir, start_frame=0, max_frames=0):
        """
        Process a video file end-to-end.

        Args:
            video_path: path to input video
            output_dir: directory to save outputs
            start_frame: skip to this frame index before processing
            max_frames: max frames to process (0=all)
        """
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video: {orig_w}x{orig_h} @ {fps}fps, {total_frames} frames")

        # Compute target resolution: height=input_height, width=aspect-preserving multiple of 14
        patch_size = 14
        target_h = self.input_height  # 518
        target_w = round(orig_w / orig_h * target_h / patch_size) * patch_size  # nearest multiple of 14
        logger.info(f"Processing resolution: {target_w}x{target_h} "
                     f"(original {orig_w}x{orig_h})")

        # Rescale camera intrinsics to target resolution
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        self.fx = self.orig_fx * scale_x
        self.fy = self.orig_fy * scale_y
        self.cx = self.orig_cx * scale_x
        self.cy = self.orig_cy * scale_y

        # YOLO receives the resized frame directly.
        # No need to set imgsz — YOLO internally pads to stride-32 multiples.

        # Skip to start frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            logger.info(f"Skipping to frame {start_frame}")

        # Video writer at target resolution
        video_writer = None
        if self.vis_config['save_video']:
            out_path = os.path.join(output_dir, "tracked_depth.mp4")
            video_writer = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (target_w, target_h)
            )

        # Frames dir
        frames_dir = None
        if self.vis_config.get('save_frames', False):
            frames_dir = os.path.join(output_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)

        # Reset depth estimator for new sequence
        self.depth_estimator.reset()

        frame_idx = 0
        total_time = 0
        end_frame = max_frames if max_frames > 0 else float('inf')

        while cap.isOpened():
            if frame_idx >= end_frame:
                break
            ret, frame_orig = cap.read()
            if not ret:
                break

            t_start = time.time()

            # --- Resize once: shared by YOLO and Onepiece ---
            frame = cv2.resize(frame_orig, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            # --- YOLO tracking ---
            detections = self.tracker.track(frame)

            # --- Depth estimation (no internal resize) ---
            depth_result = self.depth_estimator.estimate_from_bgr(frame)
            depth_map = depth_result['metric_depth']  # [target_h, target_w]

            # --- Process each detection ---
            detections_info = []
            combined_mask = np.zeros((target_h, target_w), dtype=np.uint8)

            for det in detections:
                track_id = det['track_id']
                bbox = det['bbox']
                mask_points = det['mask_points']

                # Create mask
                obj_mask = create_mask_from_polygon(mask_points, (target_h, target_w))

                # Optionally erode mask
                if self.use_eroded_mask:
                    depth_mask = erode_mask(obj_mask, self.erode_kernel, self.erode_iters)
                    # Fall back to original if erosion removes everything
                    if not np.any(depth_mask):
                        depth_mask = obj_mask
                else:
                    depth_mask = obj_mask

                # Get mask center for lateral position
                center = get_mask_center(obj_mask)
                if center is None:
                    continue
                center_x, center_y = center

                # Extract depth
                raw_depth = extract_depth_from_mask(depth_map, depth_mask)
                if raw_depth is None or raw_depth <= 0 or raw_depth > self.max_depth:
                    # Use last known depth if available
                    raw_depth = self.trajectory_mgr.get_last_depth(track_id)
                    if raw_depth is None:
                        continue

                # Lateral position
                lateral = calculate_lateral_position(raw_depth, center_x, self.fx, self.cx)

                # Update trajectory
                self.trajectory_mgr.update(track_id, raw_depth, lateral, frame_idx)
                smoothed_depth = self.trajectory_mgr.get_last_depth(track_id)

                # Recalculate lateral with smoothed depth
                lateral_smooth = calculate_lateral_position(smoothed_depth, center_x, self.fx, self.cx)

                detections_info.append({
                    'track_id': track_id,
                    'bbox': bbox,
                    'depth': smoothed_depth,
                    'lateral': lateral_smooth,
                })

                combined_mask = np.maximum(combined_mask, depth_mask)

            # --- Visualization ---
            depth_colored = colorize_depth(depth_map, self.vis_config['depth_colormap'], mask=combined_mask)
            depth_overlay_bgr = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)

            vis_frame = draw_detections(
                frame, detections_info,
                depth_overlay=depth_overlay_bgr,
                overlay_alpha=self.vis_config['depth_overlay_alpha'],
                show_bbox=self.vis_config['show_bbox'],
                show_text=self.vis_config['show_depth_text'],
            )

            t_end = time.time()
            frame_time = (t_end - t_start) * 1000
            total_time += frame_time

            # Log progress
            if frame_idx % 30 == 0:
                avg_ms = total_time / (frame_idx + 1)
                logger.info(f"[Frame {frame_idx:04d}/{total_frames}] "
                            f"{frame_time:.1f}ms (avg {avg_ms:.1f}ms), "
                            f"{len(detections_info)} pedestrians")

            # Write outputs
            if video_writer:
                video_writer.write(vis_frame)

            if frames_dir:
                cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_idx:04d}.png"), vis_frame)

            if not self.vis_config['headless']:
                cv2.imshow("Pedestrian Tracker", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1

        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        if not self.vis_config['headless']:
            cv2.destroyAllWindows()

        avg_fps = frame_idx / (total_time / 1000) if total_time > 0 else 0
        logger.info(f"Processing done: {frame_idx} frames, avg {avg_fps:.1f} FPS")

        # Save trajectory outputs
        if self.vis_config.get('save_trajectory_json', True):
            json_path = os.path.join(output_dir, "trajectories.json")
            self.trajectory_mgr.save_json(json_path)
            logger.info(f"Trajectories saved to {json_path}")

        txt_path = os.path.join(output_dir, "trajectories.txt")
        self.trajectory_mgr.save_txt(txt_path)
        logger.info(f"Trajectories (txt) saved to {txt_path}")

        if self.vis_config.get('save_trajectory_plot', True):
            plot_path = os.path.join(output_dir, "trajectory_plot.png")
            plot_trajectories(
                self.trajectory_mgr.get_all_trajectories(),
                plot_path,
                title='Estimated Pedestrian Trajectories (Onepiece + YOLO)'
            )
            logger.info(f"Trajectory plot saved to {plot_path}")

        return {
            'total_frames': frame_idx,
            'avg_fps': avg_fps,
            'num_tracks': len(self.trajectory_mgr.get_all_trajectories()),
        }
