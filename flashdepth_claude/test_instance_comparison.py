"""
Instance Segmentation + Comparison Depth Models Testing

YOLOv11 인스턴스 세그멘테이션 + 트래킹과 IMAGE depth 모델들을 결합하여
각 객체의 depth를 프레임별로 추적합니다.

지원 모델: metric3d, unidepth, zoedepth, depthpro, depthanythingv2, cut3r

Usage:
    python test_instance_comparison.py --method metric3d --video-path /path/to/video.mp4
    python test_instance_comparison.py --method unidepth --version v2 --video-path /path/to/videos
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

# Adapters for different depth methods
from adapters.metric3d_adapter import Metric3DAdapter
from adapters.unidepth_adapter import UniDepthAdapter
from adapters.zoedepth_adapter import ZoeDepthAdapter
from adapters.depthpro_adapter import DepthProAdapter
from adapters.depth_anything_v2_adapter import DepthAnythingV2Adapter

# Try to import CUT3R adapter (may not be available in all environments)
try:
    from adapters.cut3r_adapter import CUT3RAdapter
    CUT3R_AVAILABLE = True
except ImportError:
    CUT3R_AVAILABLE = False

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


class InstanceComparisonTester:
    """
    Instance Segmentation + Comparison Depth Model Tester

    YOLOv11으로 인스턴스 세그멘테이션/트래킹 후,
    adapter 패턴으로 다양한 depth 모델 inference 수행.
    """

    def __init__(self, method_name: str, config: dict, adapter):
        """
        Args:
            method_name: Depth method identifier (e.g., 'metric3d', 'unidepth')
            config: Configuration dictionary
            adapter: MethodAdapter instance
        """
        self.method_name = method_name
        self.config = config
        self.adapter = adapter
        self.device = torch.device(f"cuda:{config.get('gpu', 0)}"
                                   if torch.cuda.is_available() else "cpu")

        # Results directory
        self.save_dir = Path(config.get('results_dir',
                            f'test_results/instance_comparison/{method_name}'))
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

        # Load depth model
        logger.info(f"Loading depth model: {method_name}")
        self.model = self._setup_model()

        logger.info(f"InstanceComparisonTester initialized")
        logger.info(f"  Method: {method_name}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Save dir: {self.save_dir}")
        logger.info(f"  YOLO model: {seg_model_name}")
        logger.info(f"  Tracker: {self.tracker_config}")
        logger.info(f"  Person only: {self.person_only}")

    def _setup_model(self):
        """Load depth model using adapter"""
        checkpoint_path = self.config.get('checkpoint_path', None)
        model = self.adapter.load_model(checkpoint_path)
        model = model.to(self.device)
        model.eval()
        self.adapter.device = self.device
        logger.info(f"Depth model loaded successfully")
        return model

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess frame for depth model

        Args:
            frame: BGR image (H, W, 3) from cv2

        Returns:
            Tensor [1, 3, H, W] in [0, 1] range
        """
        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # HWC to CHW
        frame_chw = np.transpose(frame_rgb, (2, 0, 1))

        # Normalize to [0, 1]
        frame_normalized = frame_chw.astype(np.float32) / 255.0

        # To tensor
        frame_tensor = torch.from_numpy(frame_normalized).unsqueeze(0)

        return frame_tensor.to(self.device)

    def _estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """
        Run depth estimation on single frame

        Args:
            frame: BGR image (H, W, 3)

        Returns:
            Depth map (H, W) in meters
        """
        # Preprocess
        frame_tensor = self._preprocess_frame(frame)

        # Inference
        with torch.no_grad():
            depth = self.adapter.inference(frame_tensor)  # [1, H, W]

        # Convert to numpy
        depth_np = depth.squeeze().cpu().numpy()

        # Resize to original frame size if needed
        target_shape = (frame.shape[0], frame.shape[1])
        if depth_np.shape != target_shape:
            depth_np = resize_depth_to_frame(depth_np, target_shape)

        return depth_np

    def _process_instances(self, seg_result, depth_map: np.ndarray,
                           frame_shape: tuple) -> list:
        """
        Extract depth info for each tracked instance

        Args:
            seg_result: YOLO segmentation result
            depth_map: Depth map (H, W) in meters
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

            # Get center mask for robust depth extraction (erosion + circle)
            if self.center_mask:
                depth_mask, center_x = get_center_mask(mask)
            else:
                depth_mask = mask
                center_x, _ = get_mask_center(mask)

            # Calculate depth from mask
            depth = calculate_mask_depth(depth_mask, depth_map, method='mean')

            # Calculate lateral position using camera intrinsics
            lateral_pos = calculate_lateral_position(
                depth, center_x, self.fx, self.cx
            )

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
        Process single video with segmentation + depth

        Args:
            video_path: Path to video file

        Returns:
            track_trajectories: Dict[track_id -> list of trajectory points]
            result_frames: List of visualization frames
            depth_maps: List of depth maps for colormap video
            original_frames: List of original BGR frames
            fps: Video FPS
            processing_time: Total processing time
        """
        logger.info(f"Processing video: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")

        # Update intrinsics based on video resolution
        if width != 1600 or height != 900:
            self.cx = width / 2
            self.cy = height / 2
            logger.info(f"Updated intrinsics for resolution: cx={self.cx}, cy={self.cy}")

        track_trajectories = defaultdict(list)
        result_frames = []
        depth_maps = []
        original_frames = []
        frame_idx = 0

        # Reset YOLO tracker for new video (prevents track_id from carrying over)
        if hasattr(self.yolo, 'predictor') and self.yolo.predictor is not None:
            if hasattr(self.yolo.predictor, 'trackers'):
                for tracker in self.yolo.predictor.trackers:
                    tracker.reset()
                logger.info("YOLO tracker reset for new video")

        # Timing
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # YOLOv11 segmentation + tracking
            classes = [0] if self.person_only else None  # 0 = person
            seg_results = self.yolo.track(
                frame,
                persist=True,
                classes=classes,
                tracker=self.tracker_config,
                verbose=False
            )

            # Depth estimation
            depth_map = self._estimate_depth(frame)

            # Store depth map and original frame for colormap video
            depth_maps.append(depth_map.copy())
            original_frames.append(frame.copy())

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
                elapsed = time.time() - start_time
                fps_actual = frame_idx / elapsed if elapsed > 0 else 0
                logger.info(f"Frame {frame_idx}/{total_frames} "
                           f"({100*frame_idx/total_frames:.1f}%) "
                           f"- {fps_actual:.1f} FPS")

            frame_idx += 1

        cap.release()

        processing_time = time.time() - start_time
        logger.info(f"Video processing completed in {processing_time:.1f}s "
                   f"({frame_idx/processing_time:.1f} FPS)")
        logger.info(f"Tracked {len(track_trajectories)} instances")

        return dict(track_trajectories), result_frames, depth_maps, original_frames, fps, processing_time

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
                save_json_results(
                    trajectories, video_info,
                    video_save_dir / 'instance_tracking_results.json',
                    depth_model=self.method_name,
                    processing_time=processing_time
                )

                # Save trajectory plot
                create_trajectory_plot(
                    trajectories,
                    video_save_dir / 'trajectory_plot.png',
                    title=f'{video_name} - {self.method_name}'
                )

                # Save depth timeline plot
                create_depth_timeline_plot(
                    trajectories,
                    video_save_dir / 'depth_timeline.png',
                    title=f'{video_name} - {self.method_name}'
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
                            'depth_model': self.method_name,
                            'statistics_abs': depth_variation['statistics_abs'],
                            'statistics_pct': depth_variation['statistics_pct']
                        }, f, indent=2)
                    logger.info(f"Saved depth variation to {depth_var_path}")

                    # Create depth variation plot (comparison methods are all metric)
                    create_depth_variation_plot(
                        depth_variation,
                        video_save_dir / 'depth_variation.png',
                        title=f'{video_name} - {self.method_name} Depth Variation',
                        is_metric=True
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


def create_adapter(method: str, version: str = None, config: dict = None):
    """
    Create appropriate adapter for the specified method

    Args:
        method: Method name (metric3d, unidepth, zoedepth, depthpro, depthanythingv2, cut3r)
        version: Optional version (v1, v2)
        config: Configuration dict

    Returns:
        MethodAdapter instance
    """
    if method == 'metric3d':
        return Metric3DAdapter(version=version or 'v2')

    elif method == 'unidepth':
        return UniDepthAdapter(version=version or 'v2')

    elif method == 'zoedepth':
        return ZoeDepthAdapter()

    elif method == 'depthpro':
        return DepthProAdapter()

    elif method == 'depthanythingv2':
        encoder = config.get('encoder', 'vitl') if config else 'vitl'
        return DepthAnythingV2Adapter(encoder=encoder)

    elif method == 'cut3r':
        if not CUT3R_AVAILABLE:
            raise ImportError("CUT3R adapter not available. "
                            "Make sure cut3r is installed in this environment.")
        return CUT3RAdapter()

    else:
        raise ValueError(f"Unknown method: {method}. "
                        f"Supported: metric3d, unidepth, zoedepth, depthpro, "
                        f"depthanythingv2, cut3r")


def main():
    parser = argparse.ArgumentParser(
        description='Instance Segmentation + Comparison Depth Models Testing'
    )

    # Required arguments
    parser.add_argument('--method', required=True,
                        choices=['metric3d', 'unidepth', 'zoedepth',
                                'depthpro', 'depthanythingv2', 'cut3r'],
                        help='Depth estimation method')
    parser.add_argument('--video-path', required=True,
                        help='Path to video file or directory containing videos')

    # Optional arguments
    parser.add_argument('--version', default=None,
                        help='Model version (v1 or v2 for metric3d, unidepth)')
    parser.add_argument('--results-dir', default=None,
                        help='Output directory for results')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--checkpoint-path', default=None,
                        help='Path to model checkpoint')

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

    # DepthAnythingV2 specific
    parser.add_argument('--encoder', default='vitl',
                        choices=['vits', 'vitb', 'vitl'],
                        help='Encoder for DepthAnythingV2')

    args = parser.parse_args()

    # Build config dictionary
    config = {
        'video_path': args.video_path,
        'results_dir': args.results_dir or f'test_results/instance_comparison/{args.method}',
        'gpu': args.gpu,
        'checkpoint_path': args.checkpoint_path,
        'seg_model': args.seg_model,
        'tracker': args.tracker,
        'person_only': not args.no_person_only,
        'center_mask': not args.no_center_mask,
        'fx': args.fx,
        'fy': args.fy,
        'cx': args.cx,
        'cy': args.cy,
        'frame_interval': args.frame_interval,
        'encoder': args.encoder,
        'method': args.method,
    }

    # Create adapter
    adapter = create_adapter(args.method, args.version, config)

    # Create tester and run
    tester = InstanceComparisonTester(args.method, config, adapter)
    results = tester.test()

    logger.info("Testing completed!")
    return results


if __name__ == "__main__":
    main()
