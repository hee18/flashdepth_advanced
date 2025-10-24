"""
Test script for object-wise depth evaluation.

Evaluates depth estimation accuracy per segmentation class to demonstrate
improvements on specific object types.

Usage:
    # Evaluate single model on KITTI
    python test_object_wise.py --model-checkpoint train_results/results_14/gear_3/best_checkpoint.pth \
                                --config-path configs/gear3 \
                                --dataset kitti \
                                --data-root /home/cvlab/hsy/Datasets/KITTI \
                                --results-dir test_results/object_wise/gear3_kitti

    # Compare two models
    python test_object_wise.py --model-checkpoint train_results/results_14/gear_3/best_checkpoint.pth \
                                --baseline-checkpoint train_results/baseline/best_checkpoint.pth \
                                --config-path configs/gear3 \
                                --dataset kitti \
                                --data-root /home/cvlab/hsy/Datasets/KITTI \
                                --results-dir test_results/object_wise/comparison
"""

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.object_wise_evaluation import ObjectWiseMetrics, load_segmentation_mask
from flashdepth.model import FlashDepth

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ObjectWiseEvaluator:
    """Evaluator for object-wise depth metrics."""

    def __init__(
        self,
        model_checkpoint: Path,
        config_path: Path,
        dataset_type: str,
        device: str = 'cuda',
        baseline_checkpoint: Path = None
    ):
        """
        Initialize evaluator.

        Args:
            model_checkpoint: Path to model checkpoint
            config_path: Path to model config directory
            dataset_type: Dataset type ('kitti', 'cityscapes', 'nyu', 'vkitti2')
            device: Device to run on
            baseline_checkpoint: Optional baseline model checkpoint for comparison
        """
        self.device = device
        self.dataset_type = dataset_type

        # Load config
        config_file = config_path / 'config.yaml'
        self.config = OmegaConf.load(config_file)

        # Initialize metrics calculator
        self.metrics_calculator = ObjectWiseMetrics(dataset_type=dataset_type)

        # Load main model
        logger.info(f"Loading model from {model_checkpoint}")
        self.model = self._load_model(model_checkpoint)
        self.model_name = "Gear3"

        # Load baseline model if provided
        self.baseline_model = None
        self.baseline_name = None
        if baseline_checkpoint is not None:
            logger.info(f"Loading baseline model from {baseline_checkpoint}")
            self.baseline_model = self._load_model(baseline_checkpoint)
            self.baseline_name = "Baseline"

    def _load_model(self, checkpoint_path: Path) -> FlashDepth:
        """Load FlashDepth model from checkpoint."""
        # Initialize model
        model = FlashDepth(self.config)
        model = model.to(self.device)
        model.eval()

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present (from DDP)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

        return model

    def _predict_depth(
        self,
        model: FlashDepth,
        images: torch.Tensor
    ) -> np.ndarray:
        """
        Predict depth for a sequence of images.

        Args:
            model: FlashDepth model
            images: Input images (B, T, C, H, W)

        Returns:
            Predicted depth map for last frame (H, W)
        """
        B, T, C, H, W = images.shape

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Initialize hidden states
                hidden_states = None

                # Process sequence frame by frame
                for t in range(T):
                    frame = images[:, t]  # (B, C, H, W)

                    # Extract features from DINOv2
                    encoder_features = model.pretrained.get_intermediate_layers(
                        frame, n=model.intermediate_layers, return_class_token=True
                    )

                    # Get patch features and CLS token
                    patch_features = [feat[0] for feat in encoder_features]
                    cls_tokens = [feat[1] for feat in encoder_features]

                    # Pass through DPT decoder with Mamba
                    if hasattr(model, 'depth_head'):
                        depth_pred, hidden_states = model.depth_head(
                            patch_features,
                            cls_tokens,
                            hidden_states=hidden_states,
                            patch_h=H // model.patch_size,
                            patch_w=W // model.patch_size
                        )
                    else:
                        depth_pred = model.forward_features(patch_features, cls_tokens)

        # Return last frame prediction (convert to numpy)
        depth_pred = depth_pred.float().cpu().numpy()[0, 0]  # (H, W)
        return depth_pred

    def evaluate_sequence(
        self,
        images: torch.Tensor,
        gt_depth: np.ndarray,
        seg_mask: np.ndarray
    ) -> dict:
        """
        Evaluate a single sequence.

        Args:
            images: Input images (B, T, C, H, W)
            gt_depth: Ground truth depth (H, W)
            seg_mask: Segmentation mask (H, W)

        Returns:
            Dictionary of per-class metrics
        """
        # Predict depth with main model
        pred_depth = self._predict_depth(self.model, images)

        # Compute per-class metrics
        class_metrics = self.metrics_calculator.compute_metrics_per_class(
            pred_depth, gt_depth, seg_mask
        )

        results = {self.model_name: class_metrics}

        # Evaluate baseline if available
        if self.baseline_model is not None:
            baseline_pred_depth = self._predict_depth(self.baseline_model, images)
            baseline_class_metrics = self.metrics_calculator.compute_metrics_per_class(
                baseline_pred_depth, gt_depth, seg_mask
            )
            results[self.baseline_name] = baseline_class_metrics

        return results

    def evaluate_dataset(
        self,
        dataloader: DataLoader,
        max_sequences: int = None
    ) -> dict:
        """
        Evaluate entire dataset.

        Args:
            dataloader: PyTorch dataloader
            max_sequences: Maximum number of sequences to evaluate (None = all)

        Returns:
            Dictionary with aggregated metrics and comparison
        """
        all_metrics = {self.model_name: []}
        if self.baseline_model is not None:
            all_metrics[self.baseline_name] = []

        num_sequences = 0

        for batch in tqdm(dataloader, desc="Evaluating sequences"):
            images = batch['images'].to(self.device)  # (B, T, C, H, W)
            gt_depth = batch['depth'].cpu().numpy()[0, -1]  # Last frame (H, W)
            seg_mask = batch['segmentation'].cpu().numpy()[0]  # (H, W)

            # Evaluate this sequence
            seq_metrics = self.evaluate_sequence(images, gt_depth, seg_mask)

            # Accumulate metrics
            for model_name, class_metrics in seq_metrics.items():
                all_metrics[model_name].append(class_metrics)

            num_sequences += 1
            if max_sequences is not None and num_sequences >= max_sequences:
                break

            # Clear GPU cache
            torch.cuda.empty_cache()

        # Aggregate metrics across all sequences
        logger.info("Aggregating metrics across all sequences...")
        aggregated_metrics = {}
        for model_name, metrics_list in all_metrics.items():
            aggregated_metrics[model_name] = self.metrics_calculator.aggregate_metrics(
                metrics_list
            )

        # Compare models if baseline available
        comparison = None
        if self.baseline_model is not None:
            comparison = self.metrics_calculator.compare_models(
                aggregated_metrics[self.baseline_name],
                aggregated_metrics[self.model_name],
                model_a_name=self.baseline_name,
                model_b_name=self.model_name
            )

        return {
            'per_model_metrics': aggregated_metrics,
            'comparison': comparison,
            'num_sequences': num_sequences
        }


def create_dataloader(
    dataset_type: str,
    data_root: Path,
    batch_size: int = 1,
    video_length: int = 5
):
    """
    Create dataloader for specified dataset.

    Args:
        dataset_type: Dataset type
        data_root: Root directory of dataset
        batch_size: Batch size
        video_length: Number of frames in sequence

    Returns:
        PyTorch DataLoader
    """
    if dataset_type == 'kitti':
        from dataloaders.kitti_segmentation_dataset import KITTISegmentationDataset, collate_fn

        dataset = KITTISegmentationDataset(
            data_root=str(data_root),
            split='val',
            video_length=video_length,
            resolution=518
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True
        )

        logger.info(f"Created KITTI dataloader with {len(dataset)} sequences")
        return dataloader

    elif dataset_type in ['cityscapes', 'nyu', 'vkitti2']:
        # TODO: Implement other dataset loaders
        logger.warning(f"Dataset loader for {dataset_type} not yet implemented!")
        logger.info(f"Please implement dataset loader in dataloaders/{dataset_type}_dataset.py")
        logger.info("Dataset should return: images (B,T,C,H,W), depth (B,T,H,W), segmentation (B,H,W)")
        return None

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def main():
    parser = argparse.ArgumentParser(description='Object-wise depth evaluation')

    # Model arguments
    parser.add_argument('--model-checkpoint', type=Path, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--baseline-checkpoint', type=Path, default=None,
                        help='Path to baseline checkpoint for comparison')
    parser.add_argument('--config-path', type=Path, required=True,
                        help='Path to model config directory')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['kitti', 'cityscapes', 'nyu', 'vkitti2'],
                        help='Dataset type')
    parser.add_argument('--data-root', type=Path, required=True,
                        help='Root directory of dataset')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size (default: 1)')
    parser.add_argument('--video-length', type=int, default=5,
                        help='Video sequence length (default: 5)')
    parser.add_argument('--max-sequences', type=int, default=None,
                        help='Maximum sequences to evaluate (default: all)')

    # Output arguments
    parser.add_argument('--results-dir', type=Path, required=True,
                        help='Directory to save results')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (default: 0)')

    args = parser.parse_args()

    # Set device
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Create results directory
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize evaluator
    evaluator = ObjectWiseEvaluator(
        model_checkpoint=args.model_checkpoint,
        config_path=args.config_path,
        dataset_type=args.dataset,
        device=device,
        baseline_checkpoint=args.baseline_checkpoint
    )

    # Create dataloader
    dataloader = create_dataloader(
        dataset_type=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        video_length=args.video_length
    )

    if dataloader is None:
        logger.error("Dataloader not implemented yet!")
        logger.info("Please implement dataset-specific loader before running evaluation.")
        sys.exit(1)

    # Run evaluation
    logger.info("Starting evaluation...")
    results = evaluator.evaluate_dataset(
        dataloader=dataloader,
        max_sequences=args.max_sequences
    )

    # Print summary
    for model_name, metrics in results['per_model_metrics'].items():
        logger.info(f"\n{model_name} Results:")
        evaluator.metrics_calculator.print_summary(metrics)

    if results['comparison'] is not None:
        logger.info("\nModel Comparison:")
        evaluator.metrics_calculator.print_summary(
            results['per_model_metrics'][evaluator.model_name],
            comparison=results['comparison']
        )

    # Save results
    output_file = args.results_dir / f"{args.dataset}_object_wise_results.json"
    evaluator.metrics_calculator.save_results(
        results['per_model_metrics'][evaluator.model_name],
        output_file,
        comparison=results['comparison']
    )

    logger.info(f"\nEvaluation complete! Results saved to {output_file}")


if __name__ == "__main__":
    main()
