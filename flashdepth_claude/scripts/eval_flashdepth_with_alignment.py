"""
Evaluate FlashDepth predictions with scale/shift alignment

FlashDepth 원본 추론 결과(.npy)에 대해 scale/shift alignment를 적용하고 정량평가를 수행합니다.
각 시퀀스별로 단일 scale/shift를 least squares로 계산합니다.

Usage:
    python scripts/eval_flashdepth_with_alignment.py \
        --pred-dir test_results/nuscenes_original_flashdepth-l \
        --dataset nuscenes \
        --data-root /home/cvlab/hsy/Datasets \
        --output-dir test_results/nuscenes_original_flashdepth-l/eval_aligned
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from glob import glob

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import reprojection TAE calculator
from utils.reprojection_tae import ReprojectionTAECalculator, compute_reprojection_tae

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def read_pfm(file_path: str) -> np.ndarray:
    """Read PFM file format (used by ETH3D, Sintel)."""
    with open(file_path, 'rb') as f:
        header = f.readline().decode('utf-8').rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise ValueError(f'Invalid PFM header: {header}')

        dims = f.readline().decode('utf-8').rstrip()
        width, height = map(int, dims.split())

        scale = float(f.readline().decode('utf-8').rstrip())
        endian = '<' if scale < 0 else '>'
        scale = abs(scale)

        data = np.frombuffer(f.read(), endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data * scale


def read_dsp5(file_path: str) -> np.ndarray:
    """Read DSP5 file format (Spring dataset disparity)."""
    with open(file_path, 'rb') as f:
        # Read header
        header = f.read(4)
        if header != b'DSP5':
            raise ValueError(f'Invalid DSP5 header: {header}')

        # Read dimensions
        width = int.from_bytes(f.read(4), 'little')
        height = int.from_bytes(f.read(4), 'little')

        # Read data
        data = np.frombuffer(f.read(), dtype=np.float32)
        data = data.reshape((height, width))

        return data


def align_depths_lstsq(pred: np.ndarray, gt: np.ndarray, max_depth: float = 70.0) -> Tuple[np.ndarray, float, float]:
    """
    Align predicted depth to ground truth using least squares in DISPARITY space.

    Same convention as FlashDepth:
    - Align in disparity (inverse depth): gt_disp ≈ s * pred_disp + t
    - Then convert back to depth for metrics

    Args:
        pred: Predicted depth [T, H, W] or [H, W] (meters)
        gt: Ground truth depth [T, H, W] or [H, W] (meters)
        max_depth: Maximum valid depth threshold

    Returns:
        aligned_pred: Aligned prediction in depth space (meters)
        s: Scale factor (in disparity space)
        t: Shift factor (in disparity space)
    """
    # Create valid mask
    valid_mask = (gt > 0) & (gt < max_depth) & (pred > 0) & (pred < max_depth) & np.isfinite(pred) & np.isfinite(gt)

    if valid_mask.sum() < 100:
        logger.warning(f"Too few valid pixels ({valid_mask.sum()}), returning unaligned")
        return pred, 1.0, 0.0

    # Convert to disparity (inverse depth) for alignment
    pred_disp = 1.0 / pred
    gt_disp = 1.0 / gt

    pred_disp_valid = pred_disp[valid_mask].reshape(-1, 1)
    gt_disp_valid = gt_disp[valid_mask].reshape(-1, 1)

    # Solve least squares in disparity space: gt_disp = s * pred_disp + t
    A = np.hstack([pred_disp_valid, np.ones_like(pred_disp_valid)])
    result, _, _, _ = np.linalg.lstsq(A, gt_disp_valid, rcond=None)
    s, t = result.flatten()

    # Apply alignment in disparity space
    aligned_disp = s * pred_disp + t

    # Convert back to depth, handling edge cases
    aligned_disp = np.clip(aligned_disp, 1e-8, None)  # Avoid division by zero
    aligned_pred = 1.0 / aligned_disp

    return aligned_pred, float(s), float(t)


def compute_depth_metrics(pred: np.ndarray, gt: np.ndarray, max_depth: float = 70.0) -> Dict[str, float]:
    """
    Compute depth evaluation metrics.

    Args:
        pred: Predicted depth [H, W] (meters)
        gt: Ground truth depth [H, W] (meters)
        max_depth: Maximum valid depth threshold

    Returns:
        Dictionary of metrics
    """
    # Create valid mask
    valid_mask = (gt > 0) & (gt < max_depth) & (pred > 0) & (pred < max_depth)

    if valid_mask.sum() == 0:
        return {'abs_rel': float('nan'), 'mae': float('nan'), 'rmse': float('nan'),
                'a1': 0.0, 'a2': 0.0, 'a3': 0.0, 'valid_pixels': 0}

    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]

    # Absolute relative error
    abs_rel = np.mean(np.abs(pred_valid - gt_valid) / gt_valid)

    # MAE
    mae = np.mean(np.abs(pred_valid - gt_valid))

    # RMSE
    rmse = np.sqrt(np.mean((pred_valid - gt_valid) ** 2))

    # Threshold accuracy (δ < 1.25^n)
    ratio = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
    a1 = np.mean(ratio < 1.25)
    a2 = np.mean(ratio < 1.25 ** 2)
    a3 = np.mean(ratio < 1.25 ** 3)

    return {
        'abs_rel': float(abs_rel),
        'mae': float(mae),
        'rmse': float(rmse),
        'a1': float(a1),
        'a2': float(a2),
        'a3': float(a3),
        'valid_pixels': int(valid_mask.sum())
    }


class FlashDepthEvaluator:
    """FlashDepth 결과에 대한 정량평가 수행"""

    def __init__(
        self,
        pred_dir: Path,
        dataset: str,
        data_root: Path,
        output_dir: Path,
        max_depth: float = 70.0
    ):
        self.pred_dir = Path(pred_dir)
        self.dataset = dataset
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_depth = max_depth

        # Initialize reprojection TAE calculator
        self.reproj_tae_calculator = ReprojectionTAECalculator(str(data_root))
        self.tae_supported = self.reproj_tae_calculator.is_supported(dataset)

        logger.info(f"Prediction directory: {self.pred_dir}")
        logger.info(f"Dataset: {self.dataset}")
        logger.info(f"Data root: {self.data_root}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Max depth: {self.max_depth}m")
        logger.info(f"Reprojection TAE supported: {self.tae_supported}")

    def find_sequences(self) -> List[Path]:
        """Find all sequence directories with depth_npy_files (recursive search)."""
        sequences = []

        # Use glob to find all depth_npy_files directories at any depth
        for npy_dir in sorted(self.pred_dir.glob('**/depth_npy_files')):
            if npy_dir.is_dir():
                # The sequence directory is the parent of depth_npy_files
                seq_dir = npy_dir.parent
                sequences.append(seq_dir)

        return sequences

    def load_predictions(self, seq_dir: Path) -> np.ndarray:
        """Load all prediction .npy files from a sequence."""
        npy_dir = seq_dir / 'depth_npy_files'
        npy_files = sorted(npy_dir.glob('frame_*.npy'))

        if len(npy_files) == 0:
            raise ValueError(f"No .npy files found in {npy_dir}")

        preds = []
        for npy_file in npy_files:
            pred = np.load(npy_file)
            # FlashDepth outputs inverse depth (1/m) * 100
            # Convert to metric depth (meters)
            pred_depth = 100.0 / (pred + 1e-8)
            preds.append(pred_depth)

        return np.stack(preds, axis=0)  # [T, H, W]

    def load_ground_truth(self, seq_dir: Path, num_frames: int, pred_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Load ground truth for a sequence."""
        seq_name = seq_dir.name

        if self.dataset == 'tartanair':
            return self._load_tartanair_gt(seq_name, num_frames, pred_shape)
        elif self.dataset == 'unrealstereo4k' or self.dataset == 'unreal4k':
            return self._load_unreal4k_gt(seq_name, num_frames, pred_shape)
        elif self.dataset == 'eth3d':
            return self._load_eth3d_gt(seq_name, num_frames, pred_shape)
        elif self.dataset == 'sintel':
            return self._load_sintel_gt(seq_name, num_frames, pred_shape)
        elif self.dataset == 'waymo_seg' or self.dataset == 'waymo':
            return self._load_waymo_gt(seq_name, num_frames, pred_shape)
        elif self.dataset == 'vkitti':
            return self._load_vkitti_gt(seq_dir, num_frames, pred_shape)
        elif self.dataset == 'spring':
            return self._load_spring_gt(seq_name, num_frames, pred_shape)
        elif self.dataset == 'nuscenes':
            logger.warning(f"NuScenes has sparse LiDAR depth only, skipping GT comparison")
            return None
        else:
            logger.warning(f"Unknown dataset: {self.dataset}")
            return None

    def _load_tartanair_gt(self, seq_name: str, num_frames: int, pred_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Load TartanAir depth GT (npy format)."""
        gt_dir = self.data_root / 'tartanair'
        parts = seq_name.split('_')
        if len(parts) >= 2:
            scene = parts[0]
            difficulty = parts[1] if parts[1] in ['Easy', 'Hard'] else 'Easy'
            depth_dir = gt_dir / scene / difficulty / 'depth_left'
            if depth_dir.exists():
                depth_files = sorted(depth_dir.glob('*.npy'))
                if len(depth_files) >= num_frames:
                    gts = []
                    for i in range(num_frames):
                        gt = np.load(depth_files[i])
                        if gt.shape != pred_shape:
                            gt = cv2.resize(gt, (pred_shape[1], pred_shape[0]), interpolation=cv2.INTER_NEAREST)
                        gts.append(gt)
                    return np.stack(gts, axis=0)
        return None

    def _load_unreal4k_gt(self, seq_name: str, num_frames: int, pred_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Load UnrealStereo4K depth GT (disparity .npy format, converted to depth).

        UnrealStereo4K stores DISPARITY maps in .npy files.
        Convert to depth: depth = (baseline × fx) / disparity

        Baselines:
        - Indoor scenes (seq 4, 6): 0.2m
        - Outdoor scenes (seq 0, 1, 2, 3, 5, 7, 8): 0.5m

        Focal length (downsampled 2112×1188): fx = 1056
        """
        gt_dir = self.data_root / 'unreal4k'
        if not gt_dir.exists():
            gt_dir = self.data_root / 'UnrealStereo4K'

        # Baselines
        INDOOR_SEQS = [4, 6]
        BASELINE_INDOOR = 0.2   # 20cm
        BASELINE_OUTDOOR = 0.5  # 50cm
        FX_ORIGINAL = 1056.0  # for 2112×1188 resolution
        ORIGINAL_WIDTH = 2112

        # Find matching scene
        for scene_dir in gt_dir.iterdir():
            if scene_dir.is_dir() and scene_dir.name in seq_name:
                # Try Disp0 first (actual structure), then depth0/depth
                disp_dir = scene_dir / 'Disp0'
                if not disp_dir.exists():
                    disp_dir = scene_dir / 'depth0'
                if not disp_dir.exists():
                    disp_dir = scene_dir / 'depth'
                if disp_dir.exists():
                    disp_files = sorted(disp_dir.glob('*.npy'))
                    if len(disp_files) == 0:
                        disp_files = sorted(disp_dir.glob('*.exr'))
                    if len(disp_files) >= num_frames:
                        # Determine baseline from sequence ID
                        try:
                            seq_id = int(scene_dir.name.split('_')[-1])
                        except ValueError:
                            seq_id = 0
                        baseline = BASELINE_INDOOR if seq_id in INDOOR_SEQS else BASELINE_OUTDOOR

                        # Scale focal length to prediction resolution
                        fx = FX_ORIGINAL * (pred_shape[1] / ORIGINAL_WIDTH)

                        gts = []
                        for i in range(num_frames):
                            if disp_files[i].suffix == '.npy':
                                disparity = np.load(disp_files[i])
                                # Convert disparity to depth
                                with np.errstate(divide='ignore', invalid='ignore'):
                                    gt = (baseline * fx) / disparity
                                # Handle invalid values
                                invalid_mask = np.logical_or.reduce((
                                    np.isinf(gt), np.isnan(gt),
                                    gt <= 0, gt > 1000.0, disparity <= 0
                                ))
                                gt[invalid_mask] = 0
                            else:
                                gt = cv2.imread(str(disp_files[i]), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                                if gt is not None and len(gt.shape) == 3:
                                    gt = gt[:, :, 0]
                            if gt is not None:
                                if gt.shape != pred_shape:
                                    gt = cv2.resize(gt, (pred_shape[1], pred_shape[0]), interpolation=cv2.INTER_NEAREST)
                                gts.append(gt)
                        if len(gts) == num_frames:
                            return np.stack(gts, axis=0)
        return None

    def _load_eth3d_gt(self, seq_name: str, num_frames: int, pred_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Load ETH3D depth GT (raw float32 binary format).

        Same as test_gear5/eth3d_dataset.py:
        - Uses ground_truth_depth (NOT ground_truth_depth_completed)
        - Fixed resolution: 6048×4032
        - Invalid pixels marked as inf → 0
        """
        gt_dir = self.data_root / 'eth3d'

        # ETH3D fixed resolution (same as eth3d_dataset.py)
        ETH3D_WIDTH, ETH3D_HEIGHT = 6048, 4032

        # ETH3D structure: eth3d/{scene}/ground_truth_depth/dslr_images/
        for scene_dir in gt_dir.iterdir():
            if scene_dir.is_dir() and scene_dir.name in seq_name:
                # Use ground_truth_depth only (same as test_gear5)
                depth_dir = scene_dir / 'ground_truth_depth' / 'dslr_images'
                if depth_dir.exists():
                    # ETH3D depth files have .JPG extension but are raw float32
                    depth_files = sorted(depth_dir.glob('*.JPG'))
                    if len(depth_files) >= num_frames:
                        gts = []
                        for i in range(num_frames):
                            try:
                                depth_file = depth_files[i]
                                # Read raw float32 binary with fixed resolution
                                data = np.fromfile(str(depth_file), dtype=np.float32)
                                assert data.size == ETH3D_HEIGHT * ETH3D_WIDTH, \
                                    f"Mismatch: {data.size} vs {ETH3D_HEIGHT}x{ETH3D_WIDTH}"
                                gt = data.reshape(ETH3D_HEIGHT, ETH3D_WIDTH)
                                # Mark inf as invalid (0)
                                gt[np.isinf(gt)] = 0
                                gt[gt < 0] = 0
                                # Resize to prediction shape
                                if gt.shape != pred_shape:
                                    gt = cv2.resize(gt, (pred_shape[1], pred_shape[0]), interpolation=cv2.INTER_NEAREST)
                                gts.append(gt)
                            except Exception as e:
                                logger.warning(f"Failed to read {depth_files[i]}: {e}")
                                return None
                        if len(gts) == num_frames:
                            return np.stack(gts, axis=0)
        return None

    def _load_sintel_gt(self, seq_name: str, num_frames: int, pred_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Load Sintel depth GT (dpt format or pfm)."""
        gt_dir = self.data_root / 'sintel'

        # Sintel structure: sintel/depth/training/depth/{scene}/ (actual layout)
        # or sintel/training/depth/{scene}/ (alternative)
        possible_paths = [
            gt_dir / 'depth' / 'training' / 'depth',  # actual layout
            gt_dir / 'training' / 'depth',
            gt_dir / 'depth',
        ]
        depth_base = None
        for p in possible_paths:
            if p.exists():
                depth_base = p
                break
        if depth_base is None:
            return None

        for scene_dir in depth_base.iterdir() if depth_base.exists() else []:
            if scene_dir.is_dir() and scene_dir.name in seq_name:
                depth_files = sorted(scene_dir.glob('*.dpt'))
                if len(depth_files) == 0:
                    depth_files = sorted(scene_dir.glob('*.pfm'))
                if len(depth_files) >= num_frames:
                    gts = []
                    for i in range(num_frames):
                        try:
                            if depth_files[i].suffix == '.pfm':
                                gt = read_pfm(str(depth_files[i]))
                            else:
                                # .dpt format
                                with open(depth_files[i], 'rb') as f:
                                    # Skip header
                                    _ = np.frombuffer(f.read(4), dtype=np.float32)
                                    dims = np.frombuffer(f.read(8), dtype=np.int32)
                                    w, h = dims[0], dims[1]
                                    gt = np.frombuffer(f.read(), dtype=np.float32).reshape(h, w)
                            if gt.shape != pred_shape:
                                gt = cv2.resize(gt, (pred_shape[1], pred_shape[0]), interpolation=cv2.INTER_NEAREST)
                            gts.append(gt)
                        except Exception as e:
                            logger.warning(f"Failed to read {depth_files[i]}: {e}")
                            return None
                    if len(gts) == num_frames:
                        return np.stack(gts, axis=0)
        return None

    def _load_waymo_gt(self, seq_name: str, num_frames: int, pred_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Load Waymo depth GT (sparse npy format from LiDAR projection).

        Waymo structure: waymo_seg/val/{segment}/FRONT/depth/*.npy
        Each .npy is sparse format: (N, 3) [x, y, depth_meters]
        Original resolution: 1920x1280
        """
        gt_dir = self.data_root / 'waymo_seg'
        if not gt_dir.exists():
            gt_dir = self.data_root / 'waymo'

        # Original Waymo resolution
        orig_w, orig_h = 1920, 1280

        # Waymo structure: waymo_seg/val/{segment}/FRONT/depth/
        for split in ['val', 'training', 'validation', 'test']:
            split_dir = gt_dir / split
            if not split_dir.exists():
                continue
            for seg_dir in split_dir.iterdir():
                if seg_dir.is_dir() and seg_dir.name in seq_name:
                    # Try FRONT camera path first (preprocessed structure)
                    depth_dir = seg_dir / 'FRONT' / 'depth'
                    if not depth_dir.exists():
                        depth_dir = seg_dir / 'depth'
                    if depth_dir.exists():
                        depth_files = sorted(depth_dir.glob('*.npy'))
                        if len(depth_files) >= num_frames:
                            gts = []
                            for i in range(num_frames):
                                sparse_depth = np.load(depth_files[i])  # (N, 3) [x, y, depth_meters]

                                # Convert sparse to dense
                                gt = np.zeros(pred_shape, dtype=np.float32)

                                if len(sparse_depth) > 0:
                                    # Scale coordinates from original to target resolution
                                    scale_x = pred_shape[1] / orig_w
                                    scale_y = pred_shape[0] / orig_h

                                    x_coords = (sparse_depth[:, 0] * scale_x).astype(np.int32)
                                    y_coords = (sparse_depth[:, 1] * scale_y).astype(np.int32)
                                    depth_values = sparse_depth[:, 2]

                                    # Filter valid coordinates
                                    valid_mask = (
                                        (x_coords >= 0) & (x_coords < pred_shape[1]) &
                                        (y_coords >= 0) & (y_coords < pred_shape[0]) &
                                        (depth_values > 0)
                                    )

                                    x_coords = x_coords[valid_mask]
                                    y_coords = y_coords[valid_mask]
                                    depth_values = depth_values[valid_mask]

                                    if len(x_coords) > 0:
                                        gt[y_coords, x_coords] = depth_values

                                gts.append(gt)
                            if len(gts) == num_frames:
                                return np.stack(gts, axis=0)
        return None

    def _load_vkitti_gt(self, seq_dir: Path, num_frames: int, pred_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Load Virtual KITTI depth GT (png format).

        Args:
            seq_dir: Full path to sequence dir (e.g., .../Scene01/clone/frames/)
        """
        gt_dir = self.data_root / 'vkitti'
        if not gt_dir.exists():
            gt_dir = self.data_root / 'vkitti2'

        # Extract scene and variant from seq_dir path
        # seq_dir structure: .../Scene01/clone/frames/
        # seq_dir.parent.name = variant (e.g., 'clone')
        # seq_dir.parent.parent.name = scene (e.g., 'Scene01')
        seq_parts = seq_dir.parts
        scene_name = None
        variant_name = None

        # Find Scene* in path parts
        for i, part in enumerate(seq_parts):
            if part.startswith('Scene'):
                scene_name = part
                if i + 1 < len(seq_parts):
                    variant_name = seq_parts[i + 1]
                break

        if scene_name is None:
            logger.warning(f"Could not extract scene name from {seq_dir}")
            return None

        # Build GT path: vkitti/{scene}/{variant}/frames/depth/Camera_0/
        if variant_name:
            depth_dir = gt_dir / scene_name / variant_name / 'frames' / 'depth' / 'Camera_0'
            if not depth_dir.exists():
                depth_dir = gt_dir / scene_name / variant_name / 'depth' / 'Camera_0'
        else:
            # Fallback: search all variants
            scene_gt_dir = gt_dir / scene_name
            if not scene_gt_dir.exists():
                return None
            for variant_dir in scene_gt_dir.iterdir():
                depth_dir = variant_dir / 'frames' / 'depth' / 'Camera_0'
                if depth_dir.exists():
                    break
                depth_dir = variant_dir / 'depth' / 'Camera_0'
                if depth_dir.exists():
                    break

        if not depth_dir.exists():
            logger.warning(f"VKITTI depth dir not found: {depth_dir}")
            return None

        depth_files = sorted(depth_dir.glob('*.png'))
        if len(depth_files) < num_frames:
            logger.warning(f"Not enough GT files: {len(depth_files)} < {num_frames}")
            return None

        gts = []
        for i in range(num_frames):
            gt = cv2.imread(str(depth_files[i]), cv2.IMREAD_UNCHANGED)
            # VKITTI depth is in cm, convert to meters
            gt = gt.astype(np.float32) / 100.0
            if gt.shape != pred_shape:
                gt = cv2.resize(gt, (pred_shape[1], pred_shape[0]), interpolation=cv2.INTER_NEAREST)
            gts.append(gt)

        if len(gts) == num_frames:
            return np.stack(gts, axis=0)
        return None

    def _load_spring_gt(self, seq_name: str, num_frames: int, pred_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Load Spring depth GT (dsp5 disparity format, need to convert)."""
        gt_dir = self.data_root / 'spring'

        # Spring structure: spring/{split}/{scene}/disp1_left/
        for split in ['train', 'test']:
            split_dir = gt_dir / split
            if not split_dir.exists():
                continue
            for scene_dir in split_dir.iterdir():
                if scene_dir.is_dir() and scene_dir.name in seq_name:
                    disp_dir = scene_dir / 'disp1_left'
                    if disp_dir.exists():
                        disp_files = sorted(disp_dir.glob('*.dsp5'))
                        if len(disp_files) >= num_frames:
                            gts = []
                            # Spring baseline and focal length for depth conversion
                            baseline = 0.065  # meters
                            focal = 1920 / 2  # approximate
                            for i in range(num_frames):
                                try:
                                    disp = read_dsp5(str(disp_files[i]))
                                    # Convert disparity to depth: depth = baseline * focal / disparity
                                    depth = baseline * focal / (disp + 1e-8)
                                    depth[disp <= 0] = 0  # Invalid disparity
                                    if depth.shape != pred_shape:
                                        depth = cv2.resize(depth, (pred_shape[1], pred_shape[0]), interpolation=cv2.INTER_NEAREST)
                                    gts.append(depth)
                                except Exception as e:
                                    logger.warning(f"Failed to read {disp_files[i]}: {e}")
                                    return None
                            if len(gts) == num_frames:
                                return np.stack(gts, axis=0)
        return None

    def _get_image_paths(self, seq_dir: Path, num_frames: int) -> Optional[List[str]]:
        """Get image paths for a sequence (needed for reprojection TAE computation).

        Args:
            seq_dir: Path to sequence directory (from predictions)
            num_frames: Number of frames to get paths for

        Returns:
            List of image paths or None if not available
        """
        seq_name = seq_dir.name

        if self.dataset == 'sintel':
            # Sintel: sintel/images/training/clean/{scene}/frame_XXXX.png
            img_dir = self.data_root / 'sintel' / 'images' / 'training' / 'clean'
            for scene_dir in img_dir.iterdir() if img_dir.exists() else []:
                if scene_dir.is_dir() and scene_dir.name in seq_name:
                    img_files = sorted(scene_dir.glob('*.png'))
                    if len(img_files) >= num_frames:
                        return [str(f) for f in img_files[:num_frames]]
            return None

        elif self.dataset == 'eth3d':
            # ETH3D: eth3d/{scene}/images/dslr_images/*.JPG
            for scene_dir in (self.data_root / 'eth3d').iterdir() if (self.data_root / 'eth3d').exists() else []:
                if scene_dir.is_dir() and scene_dir.name in seq_name:
                    img_dir = scene_dir / 'images' / 'dslr_images'
                    if not img_dir.exists():
                        img_dir = scene_dir / 'dslr_images'
                    if img_dir.exists():
                        img_files = sorted(img_dir.glob('*.JPG'))
                        if len(img_files) >= num_frames:
                            return [str(f) for f in img_files[:num_frames]]
            return None

        elif self.dataset == 'bonn':
            # Bonn: bonn/rgbd_bonn_{name}/rgb/{timestamp}.png
            bonn_dir = self.data_root / 'bonn'
            for scene_dir in bonn_dir.iterdir() if bonn_dir.exists() else []:
                if scene_dir.is_dir() and scene_dir.name.startswith('rgbd_bonn_') and scene_dir.name in seq_name:
                    rgb_dir = scene_dir / 'rgb'
                    if rgb_dir.exists():
                        img_files = sorted(rgb_dir.glob('*.png'))
                        if len(img_files) >= num_frames:
                            return [str(f) for f in img_files[:num_frames]]
            return None

        elif self.dataset == 'vkitti':
            # VKitti: vkitti/{scene}/{variant}/frames/rgb/Camera_0/rgb_XXXXX.jpg
            parts = seq_dir.parts
            scene_name = None
            variant_name = None
            for i, part in enumerate(parts):
                if part.startswith('Scene'):
                    scene_name = part
                    if i + 1 < len(parts):
                        variant_name = parts[i + 1]
                    break

            if scene_name is None:
                return None

            vkitti_dir = self.data_root / 'vkitti'
            if not vkitti_dir.exists():
                vkitti_dir = self.data_root / 'vkitti2'

            if variant_name:
                rgb_dir = vkitti_dir / scene_name / variant_name / 'frames' / 'rgb' / 'Camera_0'
            else:
                # Search for first variant with rgb
                scene_gt_dir = vkitti_dir / scene_name
                rgb_dir = None
                for variant_dir in scene_gt_dir.iterdir() if scene_gt_dir.exists() else []:
                    test_dir = variant_dir / 'frames' / 'rgb' / 'Camera_0'
                    if test_dir.exists():
                        rgb_dir = test_dir
                        break

            if rgb_dir and rgb_dir.exists():
                img_files = sorted(rgb_dir.glob('*.jpg')) + sorted(rgb_dir.glob('*.png'))
                if len(img_files) >= num_frames:
                    return [str(f) for f in img_files[:num_frames]]
            return None

        return None

    def _get_display_name(self, seq_dir: Path) -> str:
        """Get a meaningful display name for a sequence directory."""
        # For nested structures like vkitti (Scene01/clone/frames), use parent names
        parts = seq_dir.parts
        for i, part in enumerate(parts):
            if part.startswith('Scene'):
                # vkitti: Scene01/clone
                if i + 1 < len(parts) and parts[i + 1] != 'frames':
                    return f"{part}_{parts[i + 1]}"
                return part
            if part.startswith('segment-'):
                # waymo: segment-xxxxx
                return part[:25] + '...' if len(part) > 25 else part
        # Default: use directory name
        return seq_dir.name

    def evaluate_sequence(self, seq_dir: Path, frame_export: int = None) -> Optional[Dict]:
        """Evaluate a single sequence with scale/shift alignment.

        Args:
            seq_dir: Path to the sequence directory
            frame_export: If specified, export this frame ±4 (9 total) as individual images
        """
        seq_name = self._get_display_name(seq_dir)
        logger.info(f"\nEvaluating sequence: {seq_name}")

        # Load predictions
        try:
            preds = self.load_predictions(seq_dir)
            logger.info(f"  Loaded {preds.shape[0]} prediction frames, shape: {preds.shape[1:]}")
        except Exception as e:
            logger.error(f"  Failed to load predictions: {e}")
            return None

        # Load ground truth
        gts = self.load_ground_truth(seq_dir, preds.shape[0], preds.shape[1:])
        if gts is None:
            logger.warning(f"  No ground truth found, skipping sequence")
            return None

        logger.info(f"  Loaded {gts.shape[0]} GT frames")

        # Apply scale/shift alignment on the entire sequence
        aligned_preds, scale, shift = align_depths_lstsq(preds, gts, self.max_depth)
        logger.info(f"  Alignment: scale={scale:.4f}, shift={shift:.4f}")

        # Compute per-frame metrics
        frame_metrics = []
        for t in range(preds.shape[0]):
            metrics = compute_depth_metrics(aligned_preds[t], gts[t], self.max_depth)
            frame_metrics.append(metrics)

        # Aggregate metrics
        valid_metrics = [m for m in frame_metrics if not np.isnan(m['abs_rel'])]
        if len(valid_metrics) == 0:
            logger.warning(f"  No valid frames for metrics")
            return None

        avg_metrics = {
            'abs_rel': np.mean([m['abs_rel'] for m in valid_metrics]),
            'mae': np.mean([m['mae'] for m in valid_metrics]),
            'rmse': np.mean([m['rmse'] for m in valid_metrics]),
            'a1': np.mean([m['a1'] for m in valid_metrics]),
            'a2': np.mean([m['a2'] for m in valid_metrics]),
            'a3': np.mean([m['a3'] for m in valid_metrics]),
        }

        # Add per-frame statistics for abs_rel and a1
        for key in ['abs_rel', 'a1']:
            values = [m[key] for m in valid_metrics]
            avg_metrics[f'{key}_min'] = float(np.min(values))
            avg_metrics[f'{key}_max'] = float(np.max(values))
            avg_metrics[f'{key}_std'] = float(np.std(values))

        logger.info(f"  Results: AbsRel={avg_metrics['abs_rel']:.4f}, MAE={avg_metrics['mae']:.2f}, "
                   f"δ1={avg_metrics['a1']:.4f}")

        # Compute Reprojection TAE (for datasets with camera poses)
        tae_reproj = 0.0
        tae_reproj_gt = 0.0
        if self.tae_supported and preds.shape[0] > 1:
            try:
                # Build image paths for TAE computation
                image_paths = self._get_image_paths(seq_dir, preds.shape[0])
                if image_paths is not None and len(image_paths) == preds.shape[0]:
                    # Convert to torch tensors
                    pred_tensor = torch.from_numpy(aligned_preds).float()
                    gt_tensor = torch.from_numpy(gts).float()

                    tae_result = self.reproj_tae_calculator.compute_tae(
                        pred_tensor,
                        gt_tensor,
                        self.dataset,
                        image_paths
                    )
                    tae_reproj = tae_result.get('tae_reproj', 0.0)
                    tae_reproj_gt = tae_result.get('tae_reproj_gt', 0.0)
                    logger.info(f"  TAE_reproj: {tae_reproj:.4f} (GT ref: {tae_reproj_gt:.4f})")
                else:
                    logger.warning(f"  Could not build image paths for TAE computation")
            except Exception as e:
                logger.warning(f"  Failed to compute reprojection TAE: {e}")

        avg_metrics['tae_reproj'] = tae_reproj
        avg_metrics['tae_reproj_gt'] = tae_reproj_gt

        # Create visualization
        self._visualize_sequence(seq_name, preds, aligned_preds, gts, scale, shift, avg_metrics)

        # Export specific frames if requested
        if frame_export is not None:
            self._export_frame_range(seq_name, frame_export, preds, aligned_preds, gts, frame_metrics)

        return {
            'sequence_name': seq_name,
            'num_frames': preds.shape[0],
            'scale': scale,
            'shift': shift,
            'metrics': avg_metrics,
            'per_frame_metrics': frame_metrics,
            'tae_reproj': tae_reproj,
            'tae_reproj_gt': tae_reproj_gt
        }

    def _visualize_sequence(
        self,
        seq_name: str,
        preds: np.ndarray,
        aligned_preds: np.ndarray,
        gts: np.ndarray,
        scale: float,
        shift: float,
        metrics: Dict
    ):
        """Create visualization for a sequence."""
        T = preds.shape[0]
        frames_to_show = min(5, T)
        indices = np.linspace(0, T-1, frames_to_show, dtype=int)

        fig, axes = plt.subplots(3, frames_to_show, figsize=(frames_to_show * 4, 10))
        if frames_to_show == 1:
            axes = axes.reshape(-1, 1)

        # Calculate percentile-based vmin/vmax from valid GT pixels
        valid_gt = gts[(gts > 0) & (gts < self.max_depth)]
        if len(valid_gt) > 0:
            vmin = np.percentile(valid_gt, 2)
            vmax = np.percentile(valid_gt, 98)
        else:
            vmin, vmax = 0, self.max_depth

        for col, t in enumerate(indices):
            # Row 0: Original prediction
            im0 = axes[0, col].imshow(preds[t], cmap='plasma_r', vmin=vmin, vmax=vmax)
            axes[0, col].set_title(f'Pred (frame {t})')
            axes[0, col].axis('off')

            # Row 1: Aligned prediction
            im1 = axes[1, col].imshow(aligned_preds[t], cmap='plasma_r', vmin=vmin, vmax=vmax)
            axes[1, col].set_title(f'Aligned')
            axes[1, col].axis('off')

            # Row 2: Ground truth
            im2 = axes[2, col].imshow(gts[t], cmap='plasma_r', vmin=vmin, vmax=vmax)
            axes[2, col].set_title(f'GT')
            axes[2, col].axis('off')

        # Add colorbar
        fig.colorbar(im0, ax=axes.ravel().tolist(), shrink=0.6, label='Depth (m)')

        # Add title with metrics
        fig.suptitle(f'{seq_name}\nScale={scale:.4f}, Shift={shift:.4f}\n'
                    f'AbsRel={metrics["abs_rel"]:.4f}, MAE={metrics["mae"]:.2f}m, δ1={metrics["a1"]:.4f}',
                    fontsize=12)

        plt.tight_layout()

        # Save
        vis_path = self.output_dir / f'{seq_name}_visualization.png'
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved visualization to {vis_path}")

    def _export_frame_range(
        self,
        seq_name: str,
        center_frame: int,
        preds: np.ndarray,
        aligned_preds: np.ndarray,
        gts: np.ndarray,
        frame_metrics: List[Dict]
    ):
        """Export center_frame ±4 (9 total) as individual aligned depth images.

        Args:
            seq_name: Sequence name
            center_frame: Center frame index
            preds: Predicted depth [T, H, W]
            aligned_preds: Aligned predicted depth [T, H, W]
            gts: Ground truth depth [T, H, W]
            frame_metrics: Per-frame metrics list
        """
        T = preds.shape[0]

        # Calculate frame range (center ±4 = 9 frames)
        start_frame = max(0, center_frame - 4)
        end_frame = min(T - 1, center_frame + 4)

        # Create export directory
        export_dir = self.output_dir / f'{seq_name}_frame_{center_frame}'
        export_dir.mkdir(parents=True, exist_ok=True)

        # Calculate percentile-based vmin/vmax from valid GT pixels
        valid_gt = gts[(gts > 0) & (gts < self.max_depth)]
        if len(valid_gt) > 0:
            vmin = np.percentile(valid_gt, 2)
            vmax = np.percentile(valid_gt, 98)
        else:
            vmin, vmax = 0, self.max_depth

        logger.info(f"  Exporting frames {start_frame}-{end_frame} to {export_dir}")

        for t in range(start_frame, end_frame + 1):
            # Save aligned depth as PNG (no colorbar)
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(aligned_preds[t], cmap='plasma_r', vmin=vmin, vmax=vmax)
            ax.axis('off')
            plt.tight_layout()

            fig_path = export_dir / f'frame_{t:04d}_aligned.png'
            plt.savefig(fig_path, dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close()

        logger.info(f"  Exported {end_frame - start_frame + 1} aligned frames to {export_dir}")

    def run(self, seq_filter: List[int] = None, frame_export: int = None):
        """Run evaluation on all sequences.

        Args:
            seq_filter: List of sequence indices to evaluate (e.g., [0, 4])
            frame_export: If specified, export this frame ±4 (9 total) as individual images
        """
        sequences = self.find_sequences()
        logger.info(f"Found {len(sequences)} sequences")

        if len(sequences) == 0:
            logger.error("No sequences found!")
            return

        # Filter sequences if specified
        if seq_filter is not None:
            filtered_sequences = []
            for idx in seq_filter:
                if 0 <= idx < len(sequences):
                    filtered_sequences.append(sequences[idx])
                else:
                    logger.warning(f"Sequence index {idx} out of range (0-{len(sequences)-1})")
            sequences = filtered_sequences
            logger.info(f"Filtered to {len(sequences)} sequences")

        all_results = []
        all_scale_shift = []

        for seq_dir in tqdm(sequences, desc="Evaluating sequences"):
            result = self.evaluate_sequence(seq_dir, frame_export=frame_export)
            if result is not None:
                all_results.append(result)
                seq_result = {
                    'sequence_name': result['sequence_name'],
                    'scale': result['scale'],
                    'shift': result['shift'],
                    'abs_rel': result['metrics']['abs_rel'],
                    'tae_reproj': result.get('tae_reproj', 0.0),
                    'tae_reproj_gt': result.get('tae_reproj_gt', 0.0)
                }
                all_scale_shift.append(seq_result)

        if len(all_results) == 0:
            logger.error("No sequences were successfully evaluated!")
            return

        # Aggregate overall metrics
        overall_metrics = {
            'abs_rel': np.mean([r['metrics']['abs_rel'] for r in all_results]),
            'mae': np.mean([r['metrics']['mae'] for r in all_results]),
            'rmse': np.mean([r['metrics']['rmse'] for r in all_results]),
            'a1': np.mean([r['metrics']['a1'] for r in all_results]),
            'a2': np.mean([r['metrics']['a2'] for r in all_results]),
            'a3': np.mean([r['metrics']['a3'] for r in all_results]),
        }

        # Aggregate TAE metrics (if computed)
        tae_reproj_values = [r.get('tae_reproj', 0.0) for r in all_results if r.get('tae_reproj', 0.0) > 0]
        tae_reproj_gt_values = [r.get('tae_reproj_gt', 0.0) for r in all_results if r.get('tae_reproj_gt', 0.0) > 0]
        if tae_reproj_values:
            overall_metrics['tae_reproj'] = float(np.mean(tae_reproj_values))
            overall_metrics['tae_reproj_gt'] = float(np.mean(tae_reproj_gt_values)) if tae_reproj_gt_values else 0.0
        else:
            overall_metrics['tae_reproj'] = 0.0
            overall_metrics['tae_reproj_gt'] = 0.0

        # Scale/shift statistics
        scales = [r['scale'] for r in all_results]
        shifts = [r['shift'] for r in all_results]
        scale_shift_stats = {
            'scale_mean': float(np.mean(scales)),
            'scale_std': float(np.std(scales)),
            'scale_max': float(np.max(scales)),
            'scale_min': float(np.min(scales)),
            'shift_mean': float(np.mean(shifts)),
            'shift_std': float(np.std(shifts)),
            'shift_max': float(np.max(shifts)),
            'shift_min': float(np.min(shifts)),
        }

        logger.info("\n" + "="*60)
        logger.info("OVERALL RESULTS")
        logger.info("="*60)
        logger.info(f"AbsRel: {overall_metrics['abs_rel']:.4f}")
        logger.info(f"MAE: {overall_metrics['mae']:.2f}m")
        logger.info(f"RMSE: {overall_metrics['rmse']:.2f}m")
        logger.info(f"δ1: {overall_metrics['a1']:.4f}")
        logger.info(f"δ2: {overall_metrics['a2']:.4f}")
        logger.info(f"δ3: {overall_metrics['a3']:.4f}")
        if overall_metrics['tae_reproj'] > 0:
            logger.info(f"TAE_reproj: {overall_metrics['tae_reproj']:.4f} (GT ref: {overall_metrics['tae_reproj_gt']:.4f})")
        logger.info(f"\nScale/Shift Statistics:")
        logger.info(f"  Scale: mean={scale_shift_stats['scale_mean']:.4f}, std={scale_shift_stats['scale_std']:.4f}")
        logger.info(f"  Shift: mean={scale_shift_stats['shift_mean']:.4f}, std={scale_shift_stats['shift_std']:.4f}")

        # Save results
        results_path = self.output_dir / 'eval_results.json'
        with open(results_path, 'w') as f:
            json.dump({
                'dataset': self.dataset,
                'max_depth': self.max_depth,
                'num_sequences': len(all_results),
                'overall_metrics': overall_metrics,
                'scale_shift_statistics': scale_shift_stats
            }, f, indent=2)
        logger.info(f"\nSaved results to {results_path}")

        # Save per-sequence scale/shift
        scale_shift_path = self.output_dir / 'per_sequence_scale_shift.json'
        with open(scale_shift_path, 'w') as f:
            json.dump(all_scale_shift, f, indent=2)
        logger.info(f"Saved per-sequence scale/shift to {scale_shift_path}")

        # Save detailed per-sequence results
        detailed_path = self.output_dir / 'per_sequence_results.json'
        with open(detailed_path, 'w') as f:
            # Remove per_frame_metrics for cleaner output
            clean_results = [{k: v for k, v in r.items() if k != 'per_frame_metrics'} for r in all_results]
            json.dump(clean_results, f, indent=2)
        logger.info(f"Saved detailed results to {detailed_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate FlashDepth predictions with scale/shift alignment')
    parser.add_argument('--pred-dir', type=str, required=True,
                       help='Directory containing FlashDepth predictions (with depth_npy_files)')
    parser.add_argument('--dataset', type=str, default='tartanair',
                       choices=['tartanair', 'unrealstereo4k', 'unreal4k', 'eth3d', 'sintel',
                               'waymo', 'waymo_seg', 'vkitti', 'spring', 'nuscenes'],
                       help='Dataset name for GT loading')
    parser.add_argument('--data-root', type=str, default='/home/cvlab/hsy/Datasets',
                       help='Root directory of datasets')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: pred-dir/eval_aligned)')
    parser.add_argument('--max-depth', type=float, default=70.0,
                       help='Maximum valid depth threshold (meters)')
    parser.add_argument('--seq', type=str, default=None,
                       help='Sequence selection (e.g., "0,4" for sequences 0 and 4, or single "3")')
    parser.add_argument('--frame', type=int, default=None,
                       help='Export specific frame ±4 frames (9 total) as individual images')

    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output_dir) if args.output_dir else pred_dir / 'eval_aligned'

    # Parse sequence selection
    seq_filter = None
    if args.seq:
        seq_filter = [int(s.strip()) for s in args.seq.split(',')]
        logger.info(f"Filtering to sequences: {seq_filter}")

    evaluator = FlashDepthEvaluator(
        pred_dir=pred_dir,
        dataset=args.dataset,
        data_root=Path(args.data_root),
        output_dir=output_dir,
        max_depth=args.max_depth
    )

    evaluator.run(seq_filter=seq_filter, frame_export=args.frame)


if __name__ == '__main__':
    main()
