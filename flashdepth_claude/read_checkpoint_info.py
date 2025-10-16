#!/usr/bin/env python3
"""
Script to read training step information from GSP checkpoint files.
"""

import argparse
from pathlib import Path


def read_checkpoint_step(checkpoint_path):
    """
    Read checkpoint file and extract training step information.

    Args:
        checkpoint_path: Path to the checkpoint file (.pth)

    Returns:
        Dictionary containing checkpoint metadata
    """
    # Import torch only when needed (avoids typing_extensions issues at import time)
    import torch

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract metadata
    info = {
        'checkpoint_path': str(checkpoint_path),
        'step': checkpoint.get('step', checkpoint.get('global_step', 'Unknown')),
        'best_step': checkpoint.get('best_step', 'Unknown'),
        'epoch': checkpoint.get('epoch', 'Unknown'),
        'best_val_loss': checkpoint.get('best_val_loss', 'Unknown'),
        'current_val_loss': checkpoint.get('val_loss', checkpoint.get('current_val_loss', 'Unknown')),
        'phase': checkpoint.get('phase', 'Unknown'),
    }

    # Check for optimizer state (indicates if full checkpoint or just model)
    info['has_optimizer'] = 'optimizer_state_dict' in checkpoint
    info['has_scheduler'] = 'scheduler_state_dict' in checkpoint

    # List all keys in checkpoint
    info['checkpoint_keys'] = list(checkpoint.keys())

    return info


def main():
    parser = argparse.ArgumentParser(description='Read checkpoint step information')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show all checkpoint keys')

    args = parser.parse_args()

    try:
        info = read_checkpoint_step(args.checkpoint)

        print(f"\n{'='*60}")
        print(f"Checkpoint: {info['checkpoint_path']}")
        print(f"{'='*60}")
        print(f"Current Step: {info['step']}")
        print(f"Current Val Loss: {info['current_val_loss']}")
        print(f"Best Step: {info['best_step']}")
        print(f"Best Val Loss: {info['best_val_loss']}")
        print(f"Phase: {info['phase']}")

        # Only show if resumable checkpoint
        if info['has_optimizer'] or info['has_scheduler']:
            print(f"\nResumable Checkpoint:")
            print(f"  Optimizer: {'✓' if info['has_optimizer'] else '✗'}")
            print(f"  Scheduler: {'✓' if info['has_scheduler'] else '✗'}")

        if args.verbose:
            print(f"\nCheckpoint Keys:")
            for key in info['checkpoint_keys']:
                print(f"  - {key}")

        print(f"{'='*60}\n")

    except Exception as e:
        print(f"Error reading checkpoint: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
