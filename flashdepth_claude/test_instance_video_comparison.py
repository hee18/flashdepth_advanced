"""
Instance Segmentation + Video Depth Models Testing

YOLOv11 인스턴스 세그멘테이션 + 트래킹과 VIDEO depth 모델들을 결합하여
각 객체의 depth를 프레임별로 추적합니다.

VIDEO 모델은 전체 시퀀스를 한번에 처리하므로:
1. 먼저 전체 프레임 로드
2. VIDEO 모델로 전체 시퀀스 depth 추정 (한번에)
3. 각 프레임에 YOLOv11 segmentation + tracking 적용

지원 모델: vda (Video-Depth-Anything), depthcrafter

Usage:
    python test_instance_video_comparison.py --method vda --video-path /path/to/video.mp4
    python test_instance_video_comparison.py --method vda --metric --video-path /path/to/videos
    python test_instance_video_comparison.py --method depthcrafter --video-path /path/to/video.mp4
"""

import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
import logging
import time

# YOLOv11 for instance segmentation + tracking
from ultralytics import YOLO

# Video depth adapters
from adapters.video_depth_anything_adapter import VideoDepthAnythingAdapter
from adapters.depthcrafter_adapter import DepthCrafterAdapter

# Shared utilities
from utils.instance_depth_utils import (
    get_eroded_mask_and_center,
    get_circle_mask_and_center,
    get_center_mask,
    get_mask_center,
    calculate_mask_depth,
    calculate_lateral_position,
    create_mask_from_yolo_result,
    resize_depth_to_frame,
    get_default_intrinsics,
    NUSCENES_INTRINSICS
)
from utils.instance_visualization import (
    create_frame_visualization,
    create_trajectory_plot,
    create_depth_timeline_plot,
    compute_depth_variation,
    create_depth_variation_plot,
    save_video_result,
    save_json_results,
    save_depth_colormap_video
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InstanceVideoComparisonTester:
    """
    Instance Segmentation + Video Depth Model Tester

    VIDEO 모델(vda, depthcrafter)은 전체 시퀀스를 한번에 처리하므로,
    depth 추정 후 YOLOv11 트래킹 적용.
    """

    def __init__(self, method_name: str, config: dict, adapter):
        """
        Args:
            method_name: Depth method identifier (vda, depthcrafter)
            config: Configuration dictionary
            adapter: VideoDepthAnythingAdapter or DepthCrafterAdapter instance
        """
        self.method_name = method_name
        self.config = config
        self.adapter = adapter
        self.device = torch.device(f"cuda:{config.get('gpu', 0)}"
                                   if torch.cuda.is_available() else "cpu")

        # Depth mode tracking
        self.is_metric = config.get('metric', False)
        if method_name == 'depthcrafter':
            self.is_metric = False  # DepthCrafter always outputs relative depth
            logger.info("DepthCrafter always outputs relative depth (0-1 normalized)")

        # Results directory
        depth_mode_suffix = '_metric' if self.is_metric else '_relative'
        self.save_dir = Path(config.get('results_dir',
                            f'test_results/instance_video_comparison/{method_name}{depth_mode_suffix}'))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # YOLOv11 setup
        seg_model_name = config.get('seg_model', 'yolo11x-seg.pt')
        self.yolo = YOLO(seg_model_name)
        self.tracker_config = config.get('tracker', 'botsort.yaml')
        self.person_only = config.get('person_only', True)
        self.center_mask = config.get('center_mask', True)

        # Camera intrinsics (NuScenes defaults)
        self.fx = config.get('fx', NUSCENES_INTRINSICS['fx'])
        self.fy = config.get('fy', NUSCENES_INTRINSICS['fy'])
        self.cx = config.get('cx', NUSCENES_INTRINSICS['cx'])
        self.cy = config.get('cy', NUSCENES_INTRINSICS['cy'])

        # Video settings
        self.frame_interval = config.get('frame_interval', 1)
        self.max_frames = config.get('max_frames', 500)  # Limit frames for memory

        # Load depth model
        logger.info(f"Loading depth model: {method_name}")
        self.model = self._setup_model()

        logger.info(f"InstanceVideoComparisonTester initialized")
        logger.info(f"  Method: {method_name}")
        logger.info(f"  Metric mode: {self.is_metric}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Save dir: {self.save_dir}")
        logger.info(f"  YOLO model: {seg_model_name}")
        logger.info(f"  Max frames per video: {self.max_frames}")

    def _setup_model(self):
        """Load video depth model using adapter"""
        checkpoint_path = self.config.get('checkpoint_path', None)
        model = self.adapter.load_model(checkpoint_path)

        # Move adapter to device
        if hasattr(self.adapter, 'to'):
            self.adapter.to(self.device)
        elif hasattr(model, 'to'):
            model.to(self.device)

        self.adapter.device = self.device
        logger.info(f"Video depth model loaded successfully")
        return model

    def _load_all_frames(self, video_path: Path) -> tuple:
        """
        Load all frames from video

        Args:
            video_path: Path to video file

        Returns:
            frames: List of BGR frames
            fps: Video FPS
            width, height: Frame dimensions
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Limit frames
        frames_to_load = min(total_frames, self.max_frames)
        logger.info(f"Loading {frames_to_load} frames "
                   f"(total: {total_frames}, max: {self.max_frames})")

        frames = []
        while cap.isOpened() and len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        logger.info(f"Loaded {len(frames)} frames at {width}x{height}, {fps} FPS")

        return frames, fps, width, height

    def _preprocess_sequence(self, frames: list) -> torch.Tensor:
        """
        Preprocess video sequence for depth model

        Args:
            frames: List of BGR images (H, W, 3)

        Returns:
            Tensor [1, T, 3, H, W] in [0, 1] range
        """
        processed = []
        for frame in frames:
            # BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # HWC to CHW
            frame_chw = np.transpose(frame_rgb, (2, 0, 1))

            # Normalize to [0, 1]
            frame_normalized = frame_chw.astype(np.float32) / 255.0

            processed.append(frame_normalized)

        # Stack to [T, 3, H, W]
        sequence = np.stack(processed, axis=0)

        # To tensor and add batch dim [1, T, 3, H, W]
        tensor = torch.from_numpy(sequence).unsqueeze(0)

        return tensor.to(self.device)

    def _estimate_depth_sequence(self, frames: list) -> np.ndarray:
        """
        Run video depth estimation on entire sequence

        Args:
            frames: List of BGR images

        Returns:
            Depth maps [T, H, W] in meters (metric) or normalized (relative)
        """
        # Preprocess entire sequence
        frames_tensor = self._preprocess_sequence(frames)  # [1, T, 3, H, W]
        T = frames_tensor.shape[1]

        logger.info(f"Running video depth estimation on {T} frames...")
        start_time = time.time()

        # Inference on entire sequence
        with torch.no_grad():
            depths = self.adapter.inference(frames_tensor)  # [1, T, H, W]

        inference_time = time.time() - start_time
        logger.info(f"Depth estimation completed in {inference_time:.1f}s "
                   f"({T/inference_time:.1f} FPS)")

        # Convert to numpy [T, H, W]
        depths_np = depths.squeeze(0).cpu().numpy()

        # Log depth statistics
        logger.info(f"Depth stats: min={depths_np.min():.3f}, "
                   f"max={depths_np.max():.3f}, mean={depths_np.mean():.3f}")

        return depths_np

    def _process_instances(self, seg_result, depth_map: np.ndarray,
                           frame_shape: tuple) -> list:
        """
        Extract depth info for each tracked instance

        Args:
            seg_result: YOLO segmentation result
            depth_map: Depth map (H, W)
            frame_shape: Original frame shape (H, W, C)

        Returns:
            List of instance info dicts
        """
        instances_info = []

        # Check if we have valid results
        if seg_result.masks is None or seg_result.boxes.id is None:
            return instances_info

        # Process each detected instance
        for mask_xy, track_id, box in zip(
            seg_result.masks.xy,
            seg_result.boxes.id.int().cpu().tolist(),
            seg_result.boxes.xyxy.cpu().numpy()
        ):
            # Create binary mask from polygon
            mask = create_mask_from_yolo_result(mask_xy, frame_shape)

            # Resize depth map if needed
            target_shape = (frame_shape[0], frame_shape[1])
            if depth_map.shape != target_shape:
                depth_map_resized = resize_depth_to_frame(depth_map, target_shape)
            else:
                depth_map_resized = depth_map

            # Get center mask for robust depth extraction (erosion + circle)
            if self.center_mask:
                depth_mask, center_x = get_center_mask(mask)
            else:
                depth_mask = mask
                center_x, _ = get_mask_center(mask)

            # Calculate depth from mask
            depth = calculate_mask_depth(depth_mask, depth_map_resized, method='mean')

            # For relative depth, scale might not be meaningful
            # But we still compute lateral position for tracking visualization
            if self.is_metric:
                lateral_pos = calculate_lateral_position(
                    depth, center_x, self.fx, self.cx
                )
            else:
                # For relative depth, lateral position is not meaningful in meters
                # Just use pixel-based position normalized
                lateral_pos = (center_x - self.cx) / self.cx  # Normalized

            # Get center_y
            _, center_y = get_mask_center(mask)

            instances_info.append({
                'track_id': track_id,
                'depth': depth,
                'lateral_pos': lateral_pos,
                'center_x': center_x,
                'center_y': center_y,
                'box': box,
                'mask': depth_mask
            })

        return instances_info

    def process_video(self, video_path: Path) -> tuple:
        """
        Process single video with video depth + segmentation tracking

        Args:
            video_path: Path to video file

        Returns:
            track_trajectories: Dict[track_id -> list of trajectory points]
            result_frames: List of visualization frames
            depth_maps_list: List of depth maps for colormap video
            original_frames: List of original BGR frames
            fps: Video FPS
            processing_time: Total processing time
        """
        logger.info(f"Processing video: {video_path}")
        start_time = time.time()

        # 1. Load all frames first
        frames, fps, width, height = self._load_all_frames(video_path)

        # Update intrinsics based on video resolution
        if width != 1600 or height != 900:
            self.cx = width / 2
            self.cy = height / 2
            logger.info(f"Updated intrinsics for resolution: cx={self.cx}, cy={self.cy}")

        # 2. Run video depth estimation on entire sequence
        depth_maps = self._estimate_depth_sequence(frames)  # [T, H, W]

        # 3. Run YOLOv11 tracking on each frame with pre-computed depth
        logger.info("Running YOLOv11 tracking on frames...")
        track_trajectories = defaultdict(list)
        result_frames = []

        # Reset YOLO tracker for new video (prevents track_id from carrying over)
        if hasattr(self.yolo, 'predictor') and self.yolo.predictor is not None:
            if hasattr(self.yolo.predictor, 'trackers'):
                for tracker in self.yolo.predictor.trackers:
                    tracker.reset()
                logger.info("YOLO tracker reset for new video")

        for frame_idx, (frame, depth_map) in enumerate(zip(frames, depth_maps)):
            # YOLOv11 segmentation + tracking
            classes = [0] if self.person_only else None  # 0 = person
            seg_results = self.yolo.track(
                frame,
                persist=True,
                classes=classes,
                tracker=self.tracker_config,
                verbose=False
            )

            # Process each instance
            instances_info = self._process_instances(
                seg_results[0], depth_map, frame.shape
            )

            # Update trajectories
            for inst in instances_info:
                track_trajectories[inst['track_id']].append({
                    'frame': frame_idx,
                    'depth_m': inst['depth'],
                    'lateral_m': inst['lateral_pos'],
                    'center_x': inst['center_x'],
                    'center_y': inst['center_y']
                })

            # Create visualization frame
            vis_frame = create_frame_visualization(
                frame, depth_map, instances_info
            )
            result_frames.append(vis_frame)

            # Progress logging
            if frame_idx % 50 == 0:
                logger.info(f"Tracking frame {frame_idx}/{len(frames)} "
                           f"({100*frame_idx/len(frames):.1f}%)")

        processing_time = time.time() - start_time
        logger.info(f"Video processing completed in {processing_time:.1f}s")
        logger.info(f"Tracked {len(track_trajectories)} instances")

        # Convert depth_maps array to list for colormap video
        depth_maps_list = [depth_maps[i] for i in range(len(depth_maps))]

        return dict(track_trajectories), result_frames, depth_maps_list, frames, fps, processing_time

    def test(self):
        """Main test loop - process all videos"""
        video_path = Path(self.config.get('video_path', '.'))

        # Find videos
        if video_path.is_file():
            videos = [video_path]
        else:
            videos = sorted(list(video_path.glob('*.mp4')))
            if not videos:
                videos = sorted(list(video_path.glob('*.avi')))

        if not videos:
            logger.error(f"No video files found at {video_path}")
            return {}

        logger.info(f"Found {len(videos)} video(s) to process")

        all_results = {}

        for video in videos:
            try:
                # Process video
                trajectories, frames, depth_maps, original_frames, fps, processing_time = self.process_video(video)

                # Create output directory for this video
                video_name = video.stem
                video_save_dir = self.save_dir / video_name
                video_save_dir.mkdir(exist_ok=True)

                # Save JSON results
                video_info = {
                    'name': video_name,
                    'total_frames': len(frames),
                    'fps': fps,
                    'resolution': [frames[0].shape[1], frames[0].shape[0]] if frames else [0, 0],
                    'intrinsics': {
                        'fx': self.fx, 'fy': self.fy,
                        'cx': self.cx, 'cy': self.cy
                    }
                }

                depth_model_name = f"{self.method_name}"
                if self.method_name == 'vda':
                    depth_model_name += "_metric" if self.is_metric else "_relative"

                save_json_results(
                    trajectories, video_info,
                    video_save_dir / 'instance_tracking_results.json',
                    depth_model=depth_model_name,
                    processing_time=processing_time
                )

                # Save trajectory plot
                title = f'{video_name} - {self.method_name}'
                if self.method_name == 'vda':
                    title += ' (metric)' if self.is_metric else ' (relative)'

                create_trajectory_plot(
                    trajectories,
                    video_save_dir / 'trajectory_plot.png',
                    title=title
                )

                # Save depth timeline plot
                create_depth_timeline_plot(
                    trajectories,
                    video_save_dir / 'depth_timeline.png',
                    title=title
                )

                # Save video result
                save_video_result(
                    frames,
                    video_save_dir / 'result_video.mp4',
                    fps,
                    self.frame_interval
                )

                # Save depth colormap video (80m threshold, plasma_r, 2-98 percentile)
                if len(depth_maps) > 0:
                    save_depth_colormap_video(
                        depth_maps,
                        video_save_dir / 'depth_colormap_video.mp4',
                        fps=fps,
                        max_depth=80.0,
                        frames=original_frames,
                        alpha=0.0,  # depth only
                        frame_interval=1  # 매 프레임 저장
                    )

                    # Compute and save depth variation (frame-to-frame)
                    depth_variation = compute_depth_variation(depth_maps)
                    depth_var_path = video_save_dir / 'depth_variation.json'
                    with open(depth_var_path, 'w') as f:
                        json.dump({
                            'video': video.name,
                            'depth_model': f"{self.method_name}_{'metric' if self.is_metric else 'relative'}",
                            'statistics_abs': depth_variation['statistics_abs'],
                            'statistics_pct': depth_variation['statistics_pct']
                        }, f, indent=2)
                    logger.info(f"Saved depth variation to {depth_var_path}")

                    # Create depth variation plot
                    create_depth_variation_plot(
                        depth_variation,
                        video_save_dir / 'depth_variation.png',
                        title=f'{video_name} - {self.method_name} Depth Variation',
                        is_metric=self.is_metric
                    )

                all_results[video_name] = trajectories
                logger.info(f"Results saved to {video_save_dir}")

            except Exception as e:
                logger.error(f"Error processing {video}: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Clear GPU cache between videos
            torch.cuda.empty_cache()

        logger.info(f"Completed testing on {len(all_results)} videos")
        return all_results


def create_adapter(method: str, config: dict = None):
    """
    Create appropriate video adapter for the specified method

    Args:
        method: Method name (vda, depthcrafter)
        config: Configuration dict

    Returns:
        Adapter instance
    """
    if method == 'vda':
        metric = config.get('metric', False) if config else False
        return VideoDepthAnythingAdapter(metric=metric)

    elif method == 'depthcrafter':
        max_res = config.get('max_res', 1024) if config else 1024
        return DepthCrafterAdapter(max_res=max_res)

    else:
        raise ValueError(f"Unknown video method: {method}. "
                        f"Supported: vda, depthcrafter")


def main():
    parser = argparse.ArgumentParser(
        description='Instance Segmentation + Video Depth Models Testing'
    )

    # Required arguments
    parser.add_argument('--method', required=True,
                        choices=['vda', 'depthcrafter'],
                        help='Video depth estimation method')
    parser.add_argument('--video-path', required=True,
                        help='Path to video file or directory containing videos')

    # Optional arguments
    parser.add_argument('--results-dir', default=None,
                        help='Output directory for results')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--checkpoint-path', default=None,
                        help='Path to model checkpoint')

    # Video Depth Anything specific
    parser.add_argument('--metric', action='store_true',
                        help='Use metric depth mode for VDA (default: relative)')

    # DepthCrafter specific
    parser.add_argument('--max-res', type=int, default=1024,
                        help='Maximum resolution for DepthCrafter')

    # Segmentation settings
    parser.add_argument('--seg-model', default='yolo11x-seg.pt',
                        help='YOLOv11 segmentation model')
    parser.add_argument('--tracker', default='botsort.yaml',
                        help='Tracker config file')
    parser.add_argument('--no-person-only', action='store_true',
                        help='Track all classes, not just person')
    parser.add_argument('--no-center-mask', action='store_true',
                        help='Use full mask instead of center mask (erosion + circle)')

    # Camera intrinsics
    parser.add_argument('--fx', type=float, default=NUSCENES_INTRINSICS['fx'],
                        help='Focal length X')
    parser.add_argument('--fy', type=float, default=NUSCENES_INTRINSICS['fy'],
                        help='Focal length Y')
    parser.add_argument('--cx', type=float, default=NUSCENES_INTRINSICS['cx'],
                        help='Principal point X')
    parser.add_argument('--cy', type=float, default=NUSCENES_INTRINSICS['cy'],
                        help='Principal point Y')

    # Visualization
    parser.add_argument('--frame-interval', type=int, default=1,
                        help='Save every Nth frame to video')

    # Memory management
    parser.add_argument('--max-frames', type=int, default=500,
                        help='Maximum frames per video (for memory)')

    args = parser.parse_args()

    # Build config dictionary
    config = {
        'video_path': args.video_path,
        'gpu': args.gpu,
        'checkpoint_path': args.checkpoint_path,
        'metric': args.metric,
        'max_res': args.max_res,
        'seg_model': args.seg_model,
        'tracker': args.tracker,
        'person_only': not args.no_person_only,
        'center_mask': not args.no_center_mask,
        'fx': args.fx,
        'fy': args.fy,
        'cx': args.cx,
        'cy': args.cy,
        'frame_interval': args.frame_interval,
        'max_frames': args.max_frames,
        'method': args.method,
    }

    # Set results directory
    if args.results_dir:
        config['results_dir'] = args.results_dir
    else:
        depth_mode_suffix = '_metric' if args.metric else '_relative'
        if args.method == 'depthcrafter':
            depth_mode_suffix = '_relative'  # Always relative
        config['results_dir'] = f'test_results/instance_video_comparison/{args.method}{depth_mode_suffix}'

    # Create adapter
    adapter = create_adapter(args.method, config)

    # Create tester and run
    tester = InstanceVideoComparisonTester(args.method, config, adapter)
    results = tester.test()

    logger.info("Testing completed!")
    return results


if __name__ == "__main__":
    main()
