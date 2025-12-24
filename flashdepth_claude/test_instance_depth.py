"""
Test Instance Depth (Gear5 + YOLOv11)

YOLOv11 인스턴스 세그멘테이션 + 트래킹과 Gear5 metric depth 추정을 결합하여
각 객체의 depth를 프레임별로 추적합니다.

Usage:
    python test_instance_depth.py --config-path configs/gear5 \
        +video_path=/path/to/video.mp4 \
        +results_dir=test_results/instance_depth \
        load=train_results/gear5/best.pth
"""

import sys
import os
import time
import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from flashdepth.model import FlashDepth
from flashdepth.gear5_modules import Gear5MetricHead
from utils.instance_depth_utils import (
    get_eroded_mask_and_center,
    get_circle_mask_and_center,
    get_center_mask,
    get_mask_center,
    calculate_mask_depth,
    calculate_lateral_position,
    create_mask_from_yolo_result,
    resize_depth_to_frame,
    compute_instance_statistics,
    get_default_intrinsics
)
from utils.instance_visualization import (
    create_frame_visualization,
    create_trajectory_plot,
    create_depth_timeline_plot,
    create_scale_shift_timeline_plot,
    compute_depth_variation,
    create_depth_variation_plot,
    save_video_result,
    save_json_results,
    export_frame_images,
    save_depth_colormap_video
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InstanceDepthTester:
    """
    Gear5 + YOLOv11 Instance Segmentation + Tracking 테스터

    비디오 파일을 입력받아 각 프레임에서:
    1. YOLOv11로 인스턴스 세그멘테이션 + 트래킹
    2. Gear5로 metric depth 추정
    3. 각 인스턴스의 depth 및 lateral position 계산
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.device = f"cuda:{config.get('gpu', 0)}" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Results directory
        self.save_dir = Path(config.get('results_dir', 'test_results/instance_depth'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Results directory: {self.save_dir}")

        # Video path
        self.video_path = config.get('video_path', '/data/datasets/videos_mfdepth')
        logger.info(f"Video path: {self.video_path}")

        # Visualization settings
        self.frame_interval = config.get('frame_interval', 1)
        self.enable_visualization = config.get('visualization', True)
        self.show_depth_values = config.get('show_depth_values', True)  # Show Z/X on labels

        # Camera intrinsics (NuScenes defaults)
        self.fx = config.get('fx', 1266.4)
        self.fy = config.get('fy', 1266.4)
        self.cx = config.get('cx', 816.3)  # ~half of 1600
        self.cy = config.get('cy', 450.0)  # ~half of 900

        # Canonical focal length for de-canonicalization (Metric3D style)
        # The model outputs depth in canonical space (fx=500), needs to be converted to actual space
        self.canonical_fx = config.get('canonical_fx', 500.0)
        logger.info(f"Canonical focal length: {self.canonical_fx}")

        # Instance segmentation settings
        self.person_only = config.get('person_only', True)
        self.center_mask = config.get('center_mask', True)
        self.seg_model_name = config.get('seg_model', 'yolo11x-seg.pt')
        self.tracker_config = config.get('tracker', 'botsort.yaml')

        # Original FlashDepth mode (relative depth, no Gear5 metric head)
        self.use_original_flashdepth = config.get('use_original_flashdepth', False)
        if self.use_original_flashdepth:
            logger.info("Using Original FlashDepth mode (relative depth output)")

        # Sparse GT alignment for Original FlashDepth
        self.sparse_gt_dir = config.get('sparse_gt_dir', None)
        self.sparse_gt_data = None
        self.gt_frame_to_depth = {}

        if self.use_original_flashdepth and self.sparse_gt_dir:
            self._load_sparse_gt()

        # Initialize YOLOv11
        self._setup_yolo()

        # Initialize model (Gear5 or Original FlashDepth)
        self.model = self._setup_model()

        # Model input resolution (width, height)
        # Resolution options depend on video source:
        #   nusc (NuScenes): base=(924, 518), 2k=(1596, 896)
        #   avante: base=(756, 518), 2k=(1596, 1092)
        resolution_mode = config.get('resolution', 'base')
        video_source = config.get('video_source', 'nusc')

        if video_source == 'avante':
            # Avante: orig 1600x1100, fx=900
            if resolution_mode == '2k':
                self.input_width = 1596
                self.input_height = 1092
            else:  # 'base'
                self.input_width = 756
                self.input_height = 518
        else:
            # NuScenes (default): orig 1600x900, fx=1266.4
            if resolution_mode == '2k':
                self.input_width = 1596
                self.input_height = 896
            else:  # 'base'
                self.input_width = 924
                self.input_height = 518

        logger.info(f"Input resolution: {self.input_width}x{self.input_height} ({resolution_mode}, source={video_source})")

        # Scale/shift tracking for Gear5 mode
        self.last_scale = None
        self.last_shift = None

        # Alignment scale/shift for Original FlashDepth (computed from sparse GT)
        self.alignment_scale = None
        self.alignment_shift = None

        # Video original size for de-canonicalization (set in process_video)
        self.video_original_width = None
        self.video_original_height = None
        self.de_canonical_ratio = None  # Set in process_video()

    def _load_sparse_gt(self):
        """Load sparse GT depth maps from pre-generated files."""
        gt_dir = Path(self.sparse_gt_dir)
        json_path = gt_dir / 'sparse_depth_gt.json'

        if not json_path.exists():
            logger.warning(f"Sparse GT JSON not found: {json_path}")
            return

        with open(json_path, 'r') as f:
            self.sparse_gt_data = json.load(f)

        depth_maps_dir = gt_dir / 'depth_maps'

        # Build frame_idx → depth_map mapping
        for sample_info in self.sparse_gt_data.get('samples', []):
            sweep_idx = sample_info['sweep_idx']
            depth_map_file = sample_info['depth_map_file']
            depth_map_path = depth_maps_dir / depth_map_file

            if depth_map_path.exists():
                self.gt_frame_to_depth[sweep_idx] = depth_map_path
            else:
                logger.warning(f"Depth map not found: {depth_map_path}")

        logger.info(f"Loaded sparse GT: {len(self.gt_frame_to_depth)} frames with GT")
        logger.info(f"Sample → Sweep mapping: {self.sparse_gt_data.get('sample_to_sweep', {})}")

    def _compute_alignment_from_sparse_gt(
        self,
        pred_depths: List[np.ndarray],
        frame_indices: List[int],
        max_depth: float = 70.0
    ) -> Tuple[float, float]:
        """Compute scale/shift alignment from sparse GT frames.

        Uses lstsq in DISPARITY space (same as FlashDepth convention).

        Args:
            pred_depths: List of predicted depth maps for GT frames
            frame_indices: Corresponding frame indices (sweep-based)
            max_depth: Maximum valid depth threshold

        Returns:
            scale, shift for alignment: aligned_depth = 1 / (scale * (1/pred) + shift)
        """
        all_pred_disp = []
        all_gt_disp = []

        for pred, frame_idx in zip(pred_depths, frame_indices):
            if frame_idx not in self.gt_frame_to_depth:
                continue

            # Load sparse GT
            gt = np.load(self.gt_frame_to_depth[frame_idx])

            # Resize GT to match prediction if needed
            if gt.shape != pred.shape:
                gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Valid mask (sparse GT > 0, pred valid, within depth range)
            valid_mask = (gt > 0) & (gt < max_depth) & (pred > 0.1) & (pred < max_depth) & np.isfinite(pred)

            if valid_mask.sum() < 10:
                continue

            # Convert to disparity
            pred_disp = 1.0 / pred[valid_mask]
            gt_disp = 1.0 / gt[valid_mask]

            all_pred_disp.append(pred_disp)
            all_gt_disp.append(gt_disp)

        if len(all_pred_disp) == 0:
            logger.warning("No valid pixels for alignment")
            return 1.0, 0.0

        # Concatenate all valid pixels
        all_pred_disp = np.concatenate(all_pred_disp)
        all_gt_disp = np.concatenate(all_gt_disp)

        logger.info(f"Alignment using {len(all_pred_disp)} valid pixels from {len(pred_depths)} GT frames")

        # Solve lstsq: gt_disp ≈ scale * pred_disp + shift
        A = np.vstack([all_pred_disp, np.ones_like(all_pred_disp)]).T
        result = np.linalg.lstsq(A, all_gt_disp, rcond=None)
        scale, shift = result[0]

        logger.info(f"Alignment result: scale={scale:.4f}, shift={shift:.4f}")

        return float(scale), float(shift)

    def _apply_alignment(self, depth: np.ndarray) -> np.ndarray:
        """Apply alignment scale/shift to depth map.

        Alignment in disparity space:
            aligned_disp = scale * pred_disp + shift
            aligned_depth = 1 / aligned_disp
        """
        if self.alignment_scale is None or self.alignment_shift is None:
            return depth

        # Convert to disparity, apply alignment, convert back to depth
        pred_disp = 1.0 / (depth + 1e-8)
        aligned_disp = self.alignment_scale * pred_disp + self.alignment_shift
        aligned_depth = 1.0 / (aligned_disp + 1e-8)

        # Clamp to valid range
        aligned_depth = np.clip(aligned_depth, 0.1, 1000.0)

        return aligned_depth

    def _setup_yolo(self):
        """YOLOv11 segmentation model 초기화"""
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLOv11 model: {self.seg_model_name}")
            self.yolo = YOLO(self.seg_model_name)
            logger.info("YOLOv11 loaded successfully")
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"Failed to load YOLOv11: {e}")
            raise

    def _setup_model(self) -> FlashDepth:
        """Model 초기화 (Gear5 또는 Original FlashDepth)"""
        if self.use_original_flashdepth:
            logger.info("Setting up Original FlashDepth model (relative depth)...")
        else:
            logger.info("Setting up Gear5 model (metric depth)...")

        model_config = dict(self.config.model)
        model_config['batch_size'] = 1
        model_config['use_metric_head'] = False

        model = FlashDepth(**model_config)

        model_embed_dim = 1024 if model.encoder == 'vitl' else 384
        use_mamba_temporal = self.config.model.get('use_mamba_temporal', False)

        logger.info(f"Encoder: {model.encoder}, embed_dim: {model_embed_dim}")
        logger.info(f"Temporal backend: {'Mamba2' if use_mamba_temporal else 'GRU'}")

        # Gear5 metric head 추가 (only for Gear5 mode)
        if not self.use_original_flashdepth:
            model.gear5_metric_head = Gear5MetricHead(
                embed_dim=model_embed_dim,
                feature_dim=256,
                hidden_dim=128,
                use_mamba=use_mamba_temporal
            )

            # CLS token extraction layers 설정
            cls_layers = self.config.get('cls_layers', [2, 4])
            if isinstance(cls_layers, ListConfig):
                cls_layers = OmegaConf.to_container(cls_layers)
            if isinstance(cls_layers, str):
                cls_layers = cls_layers.strip('[]').split(',')
                cls_layers = [int(x.strip()) for x in cls_layers if x.strip()]

            intermediate_idx = model.intermediate_layer_idx[model.encoder]
            encoder_indices = [layer - 1 for layer in cls_layers]
            target_blocks = [intermediate_idx[idx] for idx in encoder_indices]

            # Enable attention weights storage
            for i, block in enumerate(model.pretrained.blocks):
                if i in target_blocks:
                    block.attn.store_attn_weights = True

            self.encoder_indices = encoder_indices
            self.target_blocks = target_blocks
            logger.info(f"CLS layers: {cls_layers} -> target_blocks: {target_blocks}")
        else:
            # Original FlashDepth doesn't need CLS token extraction
            self.encoder_indices = None
            self.target_blocks = None

        # Load checkpoint
        checkpoint_path = self.config.get('load')
        if checkpoint_path and checkpoint_path != 'true' and os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys[:5]}...")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys[:5]}...")

            logger.info("Checkpoint loaded successfully")
        else:
            logger.warning(f"No checkpoint found at {checkpoint_path}")

        model = model.to(self.device)
        model.eval()

        return model

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        프레임 전처리: BGR -> RGB, normalize, resize

        Args:
            frame: BGR frame (H, W, 3)

        Returns:
            Preprocessed tensor (1, 3, input_height, input_width)
        """
        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to model input size (width, height)
        frame_resized = cv2.resize(frame_rgb, (self.input_width, self.input_height))

        # Normalize to [0, 1] and then ImageNet normalize
        frame_tensor = torch.from_numpy(frame_resized).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1)  # (3, H, W)

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        frame_tensor = (frame_tensor - mean) / std

        return frame_tensor.unsqueeze(0).to(self.device)  # (1, 3, H, W)

    @torch.no_grad()
    def _estimate_depth(self, frame_tensor: torch.Tensor, original_shape: Tuple[int, int]) -> np.ndarray:
        """
        Depth 추정 (Gear5 metric depth 또는 Original FlashDepth relative depth)

        Args:
            frame_tensor: Preprocessed frame tensor (1, 3, H, W)
            original_shape: Original frame shape (H, W)

        Returns:
            Depth map (H, W) at original resolution
            - Gear5 mode: metric depth in meters (de-canonicalized)
            - Original FlashDepth mode: relative depth (inverse depth, larger = closer)
        """
        # Reset scale/shift for this frame
        self.last_scale = None
        self.last_shift = None

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _, _, h, w = frame_tensor.shape
            patch_h = h // self.model.patch_size
            patch_w = w // self.model.patch_size

            # Extract encoder features
            encoder_features = self.model.pretrained.get_intermediate_layers(
                frame_tensor, self.model.intermediate_layer_idx[self.model.encoder]
            )

            # Get DPT features
            dpt_features = self.model.depth_head.get_forward_features(
                encoder_features, patch_h, patch_w
            )
            path_1 = dpt_features[-1]

            # Apply Mamba temporal processing
            path_1_temporal = self.model.dpt_features_to_mamba(
                input_shape=(1, 1, None, h, w),
                dpt_features=path_1,
                in_dpt_layer=0
            )

            # Get relative depth (inverse depth)
            out = self.model.depth_head.scratch.output_conv1(path_1_temporal)
            out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
            relative_depth = self.model.depth_head.scratch.output_conv2(out)

            if self.use_original_flashdepth:
                # Original FlashDepth: convert inverse depth to depth
                # FlashDepth DPT head outputs inverse depth (larger value = closer)
                # Convert to depth format (larger value = farther) for consistency
                # Multiply by 100 to match Gear5's scale range (100/inverse_depth)
                # Note: still not metric, but relative ordering and scale range are similar
                # depth_output = 100.0 / (relative_depth + 1e-8)
                depth_output = 1 / (relative_depth + 1e-8)
            else:
                # Gear5: Apply metric scale/shift
                # Extract CLS tokens
                cls_tokens_list = [
                    encoder_features[i][:, 0]
                    for i in self.encoder_indices
                ]
                cls_tokens_averaged = torch.stack(cls_tokens_list, dim=1).mean(dim=1)
                cls_tokens = cls_tokens_averaged.view(1, 1, -1)

                # Get attention weights
                attention_weights_list = [
                    self.model.pretrained.blocks[block_idx].attn.attn_weights
                    for block_idx in self.target_blocks
                ]

                # Get scale/shift from Gear5MetricHead
                gear5_outputs = self.model.gear5_metric_head(
                    cls_tokens=cls_tokens,
                    attention_weights_list=attention_weights_list,
                    patch_h=patch_h,
                    patch_w=patch_w
                )

                scale = gear5_outputs['scale']
                shift = gear5_outputs['shift']

                # Store scale/shift for logging
                self.last_scale = scale.item()
                self.last_shift = shift.item()

                # Apply scale/shift to get metric depth
                scale_expanded = scale.view(1, 1, 1, 1)
                shift_expanded = shift.view(1, 1, 1, 1)

                # pred_inverse_100 = scale * relative_depth + shift (in 100/m canonical space)
                pred_depth_inverse_100 = scale_expanded * relative_depth + shift_expanded

                # De-canonicalization: convert from canonical space to actual space
                # Formula: inverse_actual = inverse_canonical * (fx_ratio / resize_ratio)
                # where fx_ratio = canonical_fx / fx_actual
                #       resize_ratio = model_input_size / original_size
                # This is pre-computed in process_video() as self.de_canonical_ratio
                if hasattr(self, 'de_canonical_ratio') and self.de_canonical_ratio is not None:
                    pred_depth_inverse_100 = pred_depth_inverse_100 * self.de_canonical_ratio
                else:
                    # Fallback: no de-canonicalization (shouldn't happen in normal usage)
                    logger.warning("de_canonical_ratio not set - depth may be incorrect!")

                # Convert to metric depth (meters)
                depth_output = 100.0 / (pred_depth_inverse_100 + 1e-8)

        # Resize to original resolution
        depth_output = F.interpolate(
            depth_output,
            size=original_shape,
            mode='bilinear',
            align_corners=True
        )

        return depth_output[0, 0].cpu().float().numpy()

    def _process_instances(self, seg_result, depth_map: np.ndarray,
                           frame_shape: Tuple[int, int, int]) -> List[Dict[str, Any]]:
        """
        YOLOv11 결과에서 각 인스턴스의 depth 정보 추출

        Args:
            seg_result: YOLO segmentation result
            depth_map: Depth map (H, W) in meters
            frame_shape: Original frame shape (H, W, C)

        Returns:
            List of instance info dicts
        """
        instances_info = []

        if seg_result.masks is None or seg_result.boxes.id is None:
            return instances_info

        h, w = frame_shape[:2]

        for mask_xy, track_id, box, cls_id in zip(
            seg_result.masks.xy,
            seg_result.boxes.id.int().cpu().tolist(),
            seg_result.boxes.xyxy.cpu().numpy(),
            seg_result.boxes.cls.int().cpu().tolist()
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
            depth = calculate_mask_depth(depth_mask, depth_map)

            # Skip invalid depths
            if depth >= 1000:
                continue

            # Calculate lateral position
            lateral_pos = calculate_lateral_position(depth, center_x, self.fx, self.cx)

            # Get full center
            _, center_y = get_mask_center(mask)

            instances_info.append({
                'track_id': track_id,
                'class_id': cls_id,
                'depth': float(depth),
                'lateral_pos': float(lateral_pos),
                'center_x': int(center_x),
                'center_y': int(center_y),
                'box': box.tolist(),
                'mask': depth_mask
            })

        return instances_info

    def process_video(self, video_path: Path) -> Tuple[Dict[int, List], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[Dict], int, float]:
        """
        단일 비디오 처리

        Args:
            video_path: Path to video file

        Returns:
            track_trajectories: Dict[track_id -> list of trajectory points]
            result_frames: List of visualization frames
            depth_maps: List of depth maps for colormap video
            original_frames: List of original BGR frames
            scale_shift_history: List of {frame, scale, shift} dicts (Gear5 only)
            fps: Video FPS
            processing_time: Total processing time
        """
        logger.info(f"Processing video: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")

        # Store original video size for de-canonicalization
        self.video_original_width = width
        self.video_original_height = height

        # Compute de-canonicalization ratios (matching CombinedDataset formula)
        # fx_ratio = canonical_fx / fx_actual (how much focal length was scaled)
        # resize_ratio = max(target_w/orig_w, target_h/orig_h) (shorter side fills target)
        # De-canon formula: inverse_actual = inverse_canonical * (fx_ratio / resize_ratio)
        fx_ratio = self.canonical_fx / self.fx
        resize_ratio_w = self.input_width / width
        resize_ratio_h = self.input_height / height
        # Use max like CombinedDataset (consistent with center crop resize)
        resize_ratio = max(resize_ratio_w, resize_ratio_h)
        self.de_canonical_ratio = fx_ratio / resize_ratio

        logger.info(f"De-canonicalization setup:")
        logger.info(f"  fx_actual={self.fx:.1f}, canonical_fx={self.canonical_fx:.1f}")
        logger.info(f"  fx_ratio={fx_ratio:.4f}")
        logger.info(f"  resize: {width}x{height} -> {self.input_width}x{self.input_height}")
        logger.info(f"  resize_ratio={resize_ratio:.4f}")
        logger.info(f"  de_canonical_ratio={self.de_canonical_ratio:.4f}")

        # Update camera intrinsics based on video resolution
        if self.cx == 816.3 and width != 1600:
            self.cx = width / 2
            self.cy = height / 2
            logger.info(f"Updated intrinsics for {width}x{height}: cx={self.cx}, cy={self.cy}")

        track_trajectories = defaultdict(list)
        result_frames = []
        depth_maps = []
        original_frames = []
        scale_shift_history = []  # Gear5 scale/shift per frame
        frame_idx = 0

        # Reset alignment for new video
        self.alignment_scale = None
        self.alignment_shift = None

        # Initialize Mamba sequence for temporal consistency
        if hasattr(self.model, 'mamba'):
            self.model.mamba.start_new_sequence()

        # Reset YOLO tracker for new video (prevents track_id from carrying over)
        if hasattr(self.yolo, 'predictor') and self.yolo.predictor is not None:
            if hasattr(self.yolo.predictor, 'trackers'):
                for tracker in self.yolo.predictor.trackers:
                    tracker.reset()
                logger.info("YOLO tracker reset for new video")

        # Processing loop
        pbar = tqdm(total=total_frames, desc="Processing frames")
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
            frame_tensor = self._preprocess_frame(frame)
            depth_map = self._estimate_depth(frame_tensor, (height, width))

            # Store scale/shift for Gear5 mode
            if self.last_scale is not None and self.last_shift is not None:
                scale_shift_history.append({
                    'frame': frame_idx,
                    'scale': self.last_scale,
                    'shift': self.last_shift
                })

            # Store depth map and original frame for colormap video
            depth_maps.append(depth_map.copy())
            original_frames.append(frame.copy())

            # Process instances
            instances_info = self._process_instances(seg_results[0], depth_map, frame.shape)

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
            if self.enable_visualization:
                vis_frame = create_frame_visualization(
                    frame, depth_map, instances_info,
                    show_depth_values=self.show_depth_values
                )
                result_frames.append(vis_frame)

            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()

        processing_time = time.time() - start_time
        logger.info(f"Processed {frame_idx} frames in {processing_time:.2f}s ({frame_idx/processing_time:.2f} fps)")

        # =====================================================
        # Sparse GT Alignment for Original FlashDepth
        # =====================================================
        if self.use_original_flashdepth and len(self.gt_frame_to_depth) > 0:
            logger.info("Computing alignment from sparse GT...")

            # Collect GT frame predictions
            gt_frame_indices = sorted(self.gt_frame_to_depth.keys())
            gt_pred_depths = []
            valid_gt_indices = []

            for gt_idx in gt_frame_indices:
                if gt_idx < len(depth_maps):
                    gt_pred_depths.append(depth_maps[gt_idx])
                    valid_gt_indices.append(gt_idx)

            if len(gt_pred_depths) > 0:
                # Compute alignment scale/shift
                self.alignment_scale, self.alignment_shift = self._compute_alignment_from_sparse_gt(
                    gt_pred_depths, valid_gt_indices
                )

                # Apply alignment to all depth maps
                logger.info(f"Applying alignment to {len(depth_maps)} depth maps...")
                aligned_depth_maps = []
                for dm in depth_maps:
                    aligned_dm = self._apply_alignment(dm)
                    aligned_depth_maps.append(aligned_dm)

                depth_maps = aligned_depth_maps

                # Update trajectory depths and lateral positions with aligned values
                logger.info("Updating trajectory depths and lateral positions with aligned values...")
                for track_id, traj_points in track_trajectories.items():
                    for point in traj_points:
                        frame_num = point['frame']
                        if frame_num < len(depth_maps):
                            # Re-calculate depth from aligned depth map
                            # Note: This uses the original center position
                            cx, cy = point['center_x'], point['center_y']
                            aligned_depth = depth_maps[frame_num]
                            h, w = aligned_depth.shape

                            # Simple point query (could use mask if stored)
                            if 0 <= cy < h and 0 <= cx < w:
                                new_depth = float(aligned_depth[cy, cx])
                                point['depth_m'] = new_depth
                                # Re-calculate lateral position with aligned depth
                                point['lateral_m'] = calculate_lateral_position(
                                    new_depth, cx, self.fx, self.cx
                                )

                # Store alignment info
                scale_shift_history.append({
                    'type': 'sparse_gt_alignment',
                    'scale': self.alignment_scale,
                    'shift': self.alignment_shift,
                    'num_gt_frames': len(valid_gt_indices),
                    'gt_frame_indices': valid_gt_indices
                })

                logger.info(f"Alignment applied: scale={self.alignment_scale:.4f}, shift={self.alignment_shift:.4f}")
            else:
                logger.warning("No valid GT frames found for alignment")

        return dict(track_trajectories), result_frames, depth_maps, original_frames, scale_shift_history, fps, processing_time

    def test(self):
        """메인 테스트 루프"""
        video_path = Path(self.video_path)

        # Get video files
        if video_path.is_file():
            videos = [video_path]
        elif video_path.is_dir():
            videos = sorted(list(video_path.glob('*.mp4')) + list(video_path.glob('*.avi')))
        else:
            raise ValueError(f"Invalid video path: {video_path}")

        logger.info(f"Found {len(videos)} video(s) to process")

        all_results = {}

        for video in videos:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {video.name}")
            logger.info(f"{'='*60}")

            try:
                trajectories, frames, depth_maps, original_frames, scale_shift_history, fps, proc_time = self.process_video(video)

                # Create output directory for this video
                video_name = video.stem
                video_save_dir = self.save_dir / video_name
                video_save_dir.mkdir(exist_ok=True)

                # Video info for JSON
                cap = cv2.VideoCapture(str(video))
                video_info = {
                    'name': video.name,
                    'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    'fps': fps,
                    'resolution': [
                        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    ],
                    'intrinsics': {
                        'fx': self.fx, 'fy': self.fy,
                        'cx': self.cx, 'cy': self.cy
                    }
                }
                cap.release()

                # Save JSON results
                depth_model_name = 'original_flashdepth' if self.use_original_flashdepth else 'gear5'
                is_metric = not self.use_original_flashdepth
                save_json_results(
                    trajectories,
                    video_info,
                    video_save_dir / 'instance_tracking_results.json',
                    depth_model=depth_model_name,
                    processing_time=proc_time
                )

                # Save scale/shift history for Gear5 mode
                if len(scale_shift_history) > 0:
                    scale_shift_path = video_save_dir / 'scale_shift_history.json'
                    with open(scale_shift_path, 'w') as f:
                        json.dump({
                            'video': video.name,
                            'depth_model': depth_model_name,
                            'frames': scale_shift_history
                        }, f, indent=2)
                    logger.info(f"Saved scale/shift history to {scale_shift_path}")

                    # Create scale/shift timeline plot
                    create_scale_shift_timeline_plot(
                        scale_shift_history,
                        video_save_dir / 'scale_shift_timeline.png',
                        title=f'{video_name} - Scale/Shift Over Time'
                    )

                # Save trajectory plot
                if len(trajectories) > 0:
                    create_trajectory_plot(
                        trajectories,
                        video_save_dir / 'trajectory_plot.png',
                        title=f'{video_name} - Instance Depth Trajectories',
                        is_metric=is_metric
                    )

                    create_depth_timeline_plot(
                        trajectories,
                        video_save_dir / 'depth_timeline.png',
                        title=f'{video_name} - Depth Over Time',
                        is_metric=is_metric
                    )

                # Save video result
                if self.enable_visualization and len(frames) > 0:
                    save_video_result(
                        frames,
                        video_save_dir / 'result_video.mp4',
                        fps,
                        self.frame_interval
                    )

                    # Also export some frames as images
                    export_frame_images(
                        frames,
                        video_save_dir / 'frames',
                        frame_interval=1  # 매 프레임 저장
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
                            'depth_model': depth_model_name,
                            'statistics_abs': depth_variation['statistics_abs'],
                            'statistics_pct': depth_variation['statistics_pct']
                        }, f, indent=2)
                    logger.info(f"Saved depth variation to {depth_var_path}")

                    # Create depth variation plot
                    create_depth_variation_plot(
                        depth_variation,
                        video_save_dir / 'depth_variation.png',
                        title=f'{video_name} - Depth Variation',
                        is_metric=is_metric
                    )

                all_results[video_name] = {
                    'num_instances': len(trajectories),
                    'processing_time': proc_time
                }

                # Print summary for this video
                logger.info(f"\nResults for {video_name}:")
                logger.info(f"  - Tracked instances: {len(trajectories)}")
                for track_id, traj in trajectories.items():
                    if len(traj) > 0:
                        stats = compute_instance_statistics(traj)
                        logger.info(f"  - Person {track_id}: {stats['total_frames']} frames, "
                                  f"depth [{stats['min_depth']:.1f}, {stats['max_depth']:.1f}]m")

                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error processing {video}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Save overall summary
        summary_path = self.save_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\nOverall summary saved to {summary_path}")

        return all_results


@hydra.main(version_base=None, config_path="configs/gear5", config_name="config")
def main(config: DictConfig):
    """Main entry point"""
    # Override config for testing
    config.inference = True

    logger.info("="*60)
    logger.info("Instance Depth Test (Gear5 + YOLOv11)")
    logger.info("="*60)
    logger.info(f"Video path: {config.get('video_path', 'not specified')}")
    logger.info(f"Results dir: {config.get('results_dir', 'not specified')}")
    logger.info(f"Checkpoint: {config.get('load', 'not specified')}")

    tester = InstanceDepthTester(config)
    tester.test()


if __name__ == "__main__":
    main()
