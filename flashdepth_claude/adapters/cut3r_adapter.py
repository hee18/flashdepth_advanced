"""
Adapter for CUT3R (Continuous 3D Perception Model)
Reference: refer_test/CUT3R/

CUT3R는 멀티뷰 3D reconstruction 방법이지만, monocular depth estimation도 지원합니다.
eval/monodepth/launch.py 참고
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import cv2
from .base_adapter import MethodAdapter


class CUT3RAdapter(MethodAdapter):
    """
    Adapter for CUT3R monocular depth estimation

    CUT3R은 원래 multi-view 3D reconstruction 방법이지만,
    단일 이미지로도 depth 예측이 가능합니다.
    """

    def __init__(self, size=512, model_name='cut3r_512_dpt_4_64'):
        """
        Args:
            size: 입력 이미지 크기 (512 for DPT model, 224 for linear model)
            model_name: 모델 체크포인트 이름
        """
        super().__init__()
        self.size = size
        self.model_name = model_name

        # CUT3R path 추가
        self.cut3r_path = Path(__file__).parent.parent / 'refer_test' / 'CUT3R'
        if not self.cut3r_path.exists():
            raise ValueError(f"CUT3R repository not found at {self.cut3r_path}")

        # sys.path에 추가
        cut3r_src = str(self.cut3r_path / 'src')
        if cut3r_src not in sys.path:
            sys.path.insert(0, cut3r_src)

        self.model = None
        self.device = None

    def load_model(self, checkpoint_path=None):
        """
        Load CUT3R model (ARCroco3DStereo)

        Args:
            checkpoint_path: 체크포인트 경로 (None이면 기본 경로 사용)
        """
        if checkpoint_path is None:
            checkpoint_path = self.cut3r_path / 'checkpoints' / f'{self.model_name}.pth'

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"CUT3R checkpoint not found: {checkpoint_path}\n"
                f"Download from: https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view"
            )

        # add_ckpt_path를 통해 dust3r path 추가
        sys.path.insert(0, str(self.cut3r_path))
        from add_ckpt_path import add_path_to_dust3r
        add_path_to_dust3r(str(checkpoint_path))

        # Import CUT3R modules
        from dust3r.model import ARCroco3DStereo

        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ARCroco3DStereo.from_pretrained(str(checkpoint_path)).to(self.device)
        self.model.eval()

        print(f"CUT3R model loaded from {checkpoint_path}")
        return self.model

    def inference(self, images, intrinsics=None):
        """
        Run CUT3R monocular depth inference

        CUT3R의 monocular depth evaluation 방식을 따릅니다:
        - 단일 이미지를 view dictionary로 변환
        - ray_map은 NaN으로 설정 (intrinsics 불필요)
        - camera_pose는 identity matrix
        - pts3d_in_self_view의 z좌표를 depth로 사용

        Args:
            images: torch.Tensor [B, T, 3, H, W] - 입력 이미지 시퀀스
            intrinsics: 사용 안함 (CUT3R은 intrinsics 없이도 작동)

        Returns:
            depths: torch.Tensor [B, T, H, W] - Depth maps
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Import inference function
        from dust3r.inference import inference

        B, T, C, H, W = images.shape

        # 전체 시퀀스에 대한 depth 저장
        all_depths = []

        # 각 배치에 대해 처리
        for b in range(B):
            batch_depths = []

            # 각 프레임을 개별적으로 처리 (monocular depth)
            for t in range(T):
                # [3, H, W] -> [H, W, 3]
                img = images[b, t].permute(1, 2, 0).cpu().numpy()

                # 0-1 range로 정규화
                if img.max() > 1.0:
                    img = img / 255.0

                # Record processing resolution on first inference
                if self.processing_resolution is None:
                    self.processing_resolution = (self.size, self.size)
                    print(f"[CUT3R] Processing resolution: {self.size}×{self.size}")

                # Resize to model input size
                img_resized = cv2.resize(img, (self.size, self.size))

                # [H, W, 3] -> [3, H, W]
                img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()

                # View dictionary 생성 (monocular depth evaluation 방식)
                view = {
                    "img": img_tensor.unsqueeze(0),  # [1, 3, H, W]
                    "ray_map": torch.full(
                        (1, 6, self.size, self.size),
                        torch.nan
                    ),  # intrinsics 없음
                    "true_shape": torch.tensor([[H, W]]),
                    "idx": 0,
                    "instance": "0",
                    "camera_pose": torch.eye(4).float().unsqueeze(0),  # identity
                    "img_mask": torch.tensor([True]),
                    "ray_mask": torch.tensor([False]),
                    "update": torch.tensor([True]),
                    "reset": torch.tensor([False]),
                }

                # Inference
                outputs, state_args = inference([view], self.model, self.device, verbose=False)

                # Extract depth from pts3d_in_self_view
                # pts3d_in_self_view: [1, H, W, 3] (x, y, z)
                # z좌표가 depth
                pts3d = outputs["pred"][0]["pts3d_in_self_view"]  # [1, H, W, 3]
                depth_map = pts3d[0, ..., 2]  # [H, W] - z좌표

                # Resize back to original resolution
                depth_map_np = depth_map.cpu().numpy()
                depth_resized = cv2.resize(depth_map_np, (W, H), interpolation=cv2.INTER_LINEAR)
                depth_tensor = torch.from_numpy(depth_resized).float()

                batch_depths.append(depth_tensor)

            # [T, H, W]
            batch_depths_tensor = torch.stack(batch_depths, dim=0)
            all_depths.append(batch_depths_tensor)

        # [B, T, H, W]
        depths = torch.stack(all_depths, dim=0).to(images.device)

        return depths

    def get_required_env(self):
        """Required conda environment name"""
        return "cut3r"

    def get_output_type(self):
        """
        CUT3R outputs metric depth (in arbitrary scale)

        Note: CUT3R의 depth는 scale ambiguity가 있을 수 있으므로
        실제 평가 시 scale alignment가 필요할 수 있습니다.
        """
        return "metric"  # 또는 "relative" - 평가 방식에 따라 달라질 수 있음
