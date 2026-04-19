#!/usr/bin/env python3
"""
Pedestrian Tracker: Onepiece Depth + YOLO 11m-seg

Usage:
    python run_tracker.py
    python run_tracker.py --video /path/to/video.mp4
    python run_tracker.py --config custom_config.yaml
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path

# Ensure both flashdepth root and pedestrian_tracker are in path
PROJECT_ROOT = Path(__file__).parent.parent  # flashdepth_claude/
TRACKER_ROOT = Path(__file__).parent          # pedestrian_tracker/
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TRACKER_ROOT))

from tracker.pipeline import PedestrianPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Pedestrian Tracker: Onepiece + YOLO')
    parser.add_argument('--config', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'config.yaml'),
                        help='Path to config file')
    parser.add_argument('--video', type=str, default=None,
                        help='Override video path from config')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Override Onepiece checkpoint path')
    parser.add_argument('--headless', action='store_true', default=None,
                        help='Run without GUI display')
    parser.add_argument('--start-frame', type=int, default=0,
                        help='Start processing from this frame index')
    parser.add_argument('--max-frames', type=int, default=0,
                        help='Max frames to process (0=all)')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.video:
        config['paths']['video'] = args.video
    if args.output_dir:
        config['paths']['output_dir'] = args.output_dir
    if args.checkpoint:
        config['paths']['onepiece_checkpoint'] = args.checkpoint
    if args.headless is not None:
        config['visualization']['headless'] = args.headless

    video_path = config['paths']['video']
    output_dir = config['paths']['output_dir']

    logger.info(f"Video: {video_path}")
    logger.info(f"Output: {output_dir}")

    # Run pipeline
    pipeline = PedestrianPipeline(config)
    result = pipeline.run(video_path, output_dir,
                          start_frame=args.start_frame,
                          max_frames=args.max_frames)

    logger.info(f"=== Results ===")
    logger.info(f"Total frames: {result['total_frames']}")
    logger.info(f"Average FPS: {result['avg_fps']:.1f}")
    logger.info(f"Unique tracks: {result['num_tracks']}")


if __name__ == '__main__':
    main()
