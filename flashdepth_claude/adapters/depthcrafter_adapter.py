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
            repo_path / 'configs' / 'depthcrafter' / 'depthcrafter_diffusion_pytorch_model.safetensors',
            repo_path / 'DepthCrafter' / 'checkpoints' / 'depthcrafter_diffusion_pytorch_model.safetensors',
        ]

        unet_checkpoint = None
        for ckpt_path in local_unet_paths:
            if ckpt_path.exists():
                unet_checkpoint = ckpt_path.parent
                break

        # Check for local SVD pipeline
        local_svd_paths = [
            repo_path / 'configs' / 'depthcrafter' / 'stable-video-diffusion',
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
        Run DepthCrafter inference on single frame

        Args:
            image: torch.Tensor [1, 3, H, W] - Input image (0-1 normalized, RGB)
            intrinsics: Optional camera intrinsics (not used)

        Returns:
            depth: torch.Tensor [1, H, W] - Relative depth (0-1 normalized)

        Note: DepthCrafter is designed for video sequences. For single frame,
        we process it as a 1-frame video, which may not utilize temporal features.
        """
        # Convert to numpy for DepthCrafter
        image_np = image[0].cpu().numpy()  # [3, H, W]
        image_np = image_np.transpose(1, 2, 0)  # [H, W, 3]
        image_np = (image_np * 255).astype(np.uint8)  # 0-1 -> 0-255

        H_orig, W_orig = image_np.shape[:2]

        # Resize while keeping aspect ratio
        scale = min(self.max_res / H_orig, self.max_res / W_orig)
        if scale < 1.0:
            new_H = int(H_orig * scale)
            new_W = int(W_orig * scale)
            # Ensure dimensions are divisible by 64 (required by diffusion model)
            new_H = (new_H // 64) * 64
            new_W = (new_W // 64) * 64
            import cv2
            image_np = cv2.resize(image_np, (new_W, new_H), interpolation=cv2.INTER_LINEAR)

        # Convert to frames format (add temporal dimension)
        frames = np.expand_dims(image_np, axis=0)  # [1, H, W, 3]

        # Run diffusion pipeline
        with torch.inference_mode():
            res = self.pipe(
                frames,
                height=frames.shape[1],
                width=frames.shape[2],
                output_type="np",
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                window_size=1,  # Single frame
                overlap=0,
            ).frames[0]  # [T, H, W, 3]

        # Convert RGB output to single channel
        depth_np = res.sum(-1) / res.shape[-1]  # [T, H, W]
        depth_np = depth_np[0]  # [H, W] - first (and only) frame

        # Normalize to [0, 1]
        depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)

        # Resize back to original resolution if needed
        if scale < 1.0:
            import cv2
            depth_np = cv2.resize(depth_np, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)

        # Convert to torch
        depth = torch.from_numpy(depth_np).unsqueeze(0)  # [1, H, W]

        if self.device is not None:
            depth = depth.to(self.device)

        return depth

    def get_required_env(self):
        return "depthcrafter"
