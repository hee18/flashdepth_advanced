import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


@torch.no_grad()
def predict_depth_sequence(model, video: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Runs the FlashDepth model on a single video batch and returns predictions.

    Args:
        model: FlashDepth module (not wrapped in DDP).
        video: Tensor of shape (1, T, C, H, W) on the target device.

    Returns:
        Dictionary with keys 'relative' and 'metric'. The 'metric' entry is None when
        the metric head is disabled.
    """
    was_training = model.training
    model.eval()

    if getattr(model, 'use_mamba', False):
        model.mamba.start_new_sequence()

    b, t, c, h, w = video.shape
    assert b == 1, "Only batch size 1 is supported for visualization"

    preds_relative = []
    preds_metric = [] if getattr(model, 'metric_head_enabled', False) else None

    for idx in range(t):
        frame = video[:, idx]
        patch_h, patch_w = frame.shape[-2] // model.patch_size, frame.shape[-1] // model.patch_size

        if getattr(model, 'metric_head_enabled', False):
            dpt_features, cls_token = model.get_dpt_features(
                frame,
                input_shape=(b, c, h, w),
                return_cls_token=True,
            )
        else:
            dpt_features = model.get_dpt_features(frame, input_shape=(b, c, h, w))
            cls_token = None

        rel_depth = model.final_head(dpt_features, patch_h, patch_w)
        rel_depth = torch.clip(rel_depth, min=0)

        preds_relative.append(rel_depth.squeeze(0).detach().cpu())

        if preds_metric is not None:
            metric_depth, _, _ = model.apply_global_scale(rel_depth, cls_token)
            preds_metric.append(torch.clip(metric_depth, min=0).squeeze(0).detach().cpu())

    if was_training:
        model.train()

    return {
        'relative': torch.stack(preds_relative),
        'metric': torch.stack(preds_metric) if preds_metric is not None else None,
    }


def _to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().float()
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
    tensor = tensor.numpy()
    tensor = np.transpose(tensor, (1, 2, 0))
    return np.clip(tensor, 0.0, 1.0)


def _to_numpy_depth(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().float().numpy()


def create_metric_visualization(
    image: torch.Tensor,
    gt_depth: torch.Tensor,
    pred_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    save_path: str,
    title: Optional[str] = None,
) -> None:
    """Creates a composite visualization comparing predicted and ground-truth depth."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    image_np = _to_numpy_image(image)
    gt_np = _to_numpy_depth(gt_depth)
    pred_np = _to_numpy_depth(pred_depth)
    mask_np = valid_mask.detach().cpu().numpy().astype(float)

    valid_gt = gt_np[mask_np.astype(bool)]
    valid_pred = pred_np[mask_np.astype(bool)]

    if valid_gt.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = np.percentile(valid_gt, 1.0)
        vmax = np.percentile(valid_gt, 99.0)
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-3

    abs_error = np.zeros_like(gt_np)
    abs_error[mask_np.astype(bool)] = np.abs(valid_pred - valid_gt) if valid_gt.size else 0.0

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].imshow(image_np)
    axes[0, 0].axis('off')
    axes[0, 0].set_title('Input RGB')

    im_gt = axes[0, 1].imshow(gt_np, cmap='inferno', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('GT Metric Depth (m)')
    axes[0, 1].axis('off')
    fig.colorbar(im_gt, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im_pred = axes[0, 2].imshow(pred_np, cmap='inferno', vmin=vmin, vmax=vmax)
    axes[0, 2].set_title('Pred Metric Depth (m)')
    axes[0, 2].axis('off')
    fig.colorbar(im_pred, ax=axes[0, 2], fraction=0.046, pad=0.04)

    axes[1, 0].imshow(mask_np, cmap='gray')
    axes[1, 0].set_title('Valid Mask')
    axes[1, 0].axis('off')

    im_err = axes[1, 1].imshow(abs_error, cmap='magma')
    axes[1, 1].set_title('Absolute Error (m)')
    axes[1, 1].axis('off')
    fig.colorbar(im_err, ax=axes[1, 1], fraction=0.046, pad=0.04)

    axes[1, 2].hist(valid_gt, bins=50, alpha=0.6, label='GT')
    axes[1, 2].hist(valid_pred, bins=50, alpha=0.6, label='Pred')
    axes[1, 2].set_title('Depth Distribution (Valid Pixels)')
    axes[1, 2].set_xlabel('Depth (m)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].legend()

    if title:
        fig.suptitle(title, fontsize=16)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def generate_checkpoint_visualization(
    model,
    sample: Optional[Dict[str, torch.Tensor]],
    cfg,
    train_step: int,
) -> None:
    if sample is None:
        return

    device = torch.cuda.current_device()

    video = sample['video'].to(device)
    gt_depth = sample['depth'].to(device)
    frame_idx = min(sample.get('frame_idx', video.shape[1] // 2), video.shape[1] - 1)

    preds = predict_depth_sequence(model, video)
    pred_metric = preds['metric'] if preds['metric'] is not None else preds['relative']

    pred_frame = pred_metric[frame_idx]
    gt_frame = gt_depth[0, frame_idx].detach().cpu()
    image_frame = video[0, frame_idx].detach().cpu()

    valid_mask = gt_frame > 0

    save_dir = os.path.join(cfg.config_dir, 'val', 'visualizations')
    save_name = f'step_{train_step:06d}.png'
    title = sample.get('name', '')
    create_metric_visualization(image_frame, gt_frame, pred_frame, valid_mask, os.path.join(save_dir, save_name), title=title)
