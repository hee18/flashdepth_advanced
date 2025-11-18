"""
Adapter for DepthCrafter (Diffusion-based Video Depth Estimation)
Reference: refer_test/DepthCrafter/

Note: DepthCrafter is designed for video sequences but can process single frames.
For best results with temporal consistency, consider batch processing.
"""

import sys
from pathlib import Path
import torch
import numpy as np
from .base_adapter import MethodAdapter


class DepthCrafterAdapter(MethodAdapter):
    """Adapter for DepthCrafter (Diffusion-based)"""

    def __init__(self, max_res=1024):
        super().__init__()
        self.max_res = max_res  # Maximum resolution for resizing

        dc_path = Path(__file__).parent.parent / 'refer_test' / 'DepthCrafter'
        if str(dc_path) not in sys.path:
            sys.path.insert(0, str(dc_path))

    def load_model(self, checkpoint_path=None):
        """
        Load DepthCrafter diffusion pipeline

        DepthCrafter resizes inputs based on max_res while keeping aspect ratio
        """
        from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
        from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter

        # Check for local UNet checkpoint
        repo_path = Path(__file__).parent.parent / 'refer_test'
        local_unet_paths = [
            repo_path / 'DepthCrafter' / 'checkpoints' / 'depthcrafter_diffusion_pytorch_model.safetensors',
        ]

        unet_checkpoint = None
        for ckpt_path in local_unet_paths:
            if ckpt_path.exists():
                unet_checkpoint = ckpt_path.parent
                break

        # Check for local SVD pipeline
        local_svd_paths = [
            repo_path / 'DepthCrafter' / 'checkpoints' / 'stable-video-diffusion',
            repo_path / 'stable-video-diffusion-img2vid-xt',
        ]

        svd_path = None
        for path in local_svd_paths:
            if path.exists() and (path / 'model_index.json').exists():
                svd_path = path
                break

        # Load UNet
        if unet_checkpoint:
            print(f"Loading DepthCrafter UNet from local checkpoint...")
            print(f"Using local UNet checkpoint: {unet_checkpoint}")
            unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
                str(unet_checkpoint),
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            )
        else:
            print(f"Loading DepthCrafter UNet from tencent/DepthCrafter...")
            print(f"Local UNet checkpoint not found. Will download from HuggingFace...")
            unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
                "tencent/DepthCrafter",
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            )

        # Load pipeline
        if svd_path:
            print(f"Loading SVD pipeline from local checkpoint: {svd_path}")
            pre_train_path = str(svd_path)
        else:
            print(f"Loading SVD pipeline from stabilityai/stable-video-diffusion-img2vid-xt...")
            print(f"Local SVD pipeline not found. Will download from HuggingFace...")
            pre_train_path = "stabilityai/stable-video-diffusion-img2vid-xt"

        self.pipe = DepthCrafterPipeline.from_pretrained(
            pre_train_path,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        # Enable optimizations
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"Xformers not enabled: {e}")

        self.pipe.enable_attention_slicing()

        # Set default inference parameters
        self.num_inference_steps = 5
        self.guidance_scale = 1.0

        print(f"DepthCrafter model loaded successfully")
        return self.pipe

    def to(self, device):
        """Move model to device"""
        super().to(device)
        if hasattr(self, 'pipe'):
            if device.type == 'cuda':
                self.pipe.to(device)
            else:
                # For CPU, use model CPU offload
                self.pipe.enable_model_cpu_offload()
        return self

    def inference(self, image, intrinsics=None):
        """
        Run DepthCrafter inference on video sequence

        Args:
            image: torch.Tensor [1, T, 3, H, W] - Input video sequence (0-1 normalized, RGB)
            intrinsics: Optional camera intrinsics (not used)

        Returns:
            depth: torch.Tensor [1, T, H, W] - Relative depth (0-1 normalized)
        """
        # Handle both single frame [1, 3, H, W] and sequence [1, T, 3, H, W]
        if image.ndim == 4:
            image = image.unsqueeze(1)  # [1, 1, 3, H, W]

        B, T, C, H_orig, W_orig = image.shape
        assert B == 1, "Batch size must be 1"

        # Resize while keeping aspect ratio
        scale = min(self.max_res / H_orig, self.max_res / W_orig)
        if scale < 1.0:
            new_H = int(H_orig * scale)
            new_W = int(W_orig * scale)
            new_H = (new_H // 64) * 64
            new_W = (new_W // 64) * 64

            import torch.nn.functional as F
            image_resized = F.interpolate(
                image.view(T, C, H_orig, W_orig),
                size=(new_H, new_W),
                mode='bilinear',
                align_corners=False
            )
        else:
            image_resized = image.squeeze(0)
            new_H, new_W = H_orig, W_orig

        # Record processing resolution on first inference
        if self.processing_resolution is None:
            self.processing_resolution = (new_H, new_W)
            print(f"[DepthCrafter] Processing resolution: {new_H}×{new_W} (max {self.max_res}, aspect-preserved)")

        # Convert to [T, H, W, 3] numpy format
        image_uint8 = (image_resized * 255.0).to(torch.uint8)
        frames_np = image_uint8.cpu().numpy().transpose(0, 2, 3, 1)

        # Run diffusion pipeline on entire sequence
        with torch.inference_mode():
            res = self.pipe(
                frames_np,
                height=new_H,
                width=new_W,
                output_type="np",
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                window_size=min(T, self.window_size),
                overlap=min(T//4, 25),
            ).frames[0]

        # Convert RGB output to single channel and normalize
        depth_np = res.sum(-1) / res.shape[-1]  # [T, H, W]
        depth_list = []
        for t in range(T):
            depth_t = depth_np[t]
            depth_t = (depth_t - depth_t.min()) / (depth_t.max() - depth_t.min() + 1e-8)

            if scale < 1.0:
                import cv2
                depth_t = cv2.resize(depth_t, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)

            depth_list.append(depth_t)

        depth = np.stack(depth_list, axis=0)
        depth = torch.from_numpy(depth).unsqueeze(0)  # [1, T, H, W]

        if self.device is not None:
            depth = depth.to(self.device)

        return depth

    def get_required_env(self):
        return "depthcrafter"
