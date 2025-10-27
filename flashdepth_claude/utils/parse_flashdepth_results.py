"""
Parse original FlashDepth test logs and save FPS as JSON.

Original FlashDepth outputs relative depth (not metric depth),
so we only extract FPS performance metrics.

Usage:
    python utils/parse_flashdepth_results.py <log_file> <output_dir>
"""

import re
import json
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_flashdepth_log(log_path):
    """
    Parse FlashDepth log to extract FPS performance.

    Expected log format (new):
        2025-10-25 09:29:44,998 - INFO - Inference FPS (data pre-loaded): 9.41 | wall time: 4.78s | num frames: 45

    Also supports legacy format:
        2025-10-25 09:29:44,998 - INFO - wall time taken: 4.78; fps: 9.41; num frames: 45
    """
    with open(log_path, 'r') as f:
        log_content = f.read()

    # New FPS pattern (preferred)
    fps_pattern_new = r'Inference FPS \(data pre-loaded\):\s*([\d.]+)\s*\|\s*wall time:\s*([\d.]+)s\s*\|\s*num frames:\s*(\d+)'

    # Legacy FPS pattern (for backward compatibility)
    fps_pattern_legacy = r'wall time taken:\s*([\d.]+);\s*fps:\s*([\d.]+);\s*num frames:\s*(\d+)'

    fps_results = []
    total_fps = 0
    total_time = 0
    total_frames = 0

    # Try new format first
    measurement_type = None
    for match in re.finditer(fps_pattern_new, log_content):
        fps = float(match.group(1))
        wall_time = float(match.group(2))
        num_frames = int(match.group(3))

        fps_results.append({
            'wall_time': wall_time,
            'fps': fps,
            'num_frames': num_frames
        })

        total_time += wall_time
        total_fps += fps
        total_frames += num_frames
        measurement_type = 'inference_only'

    # If no new format found, try legacy format
    if not fps_results:
        for match in re.finditer(fps_pattern_legacy, log_content):
            wall_time = float(match.group(1))
            fps = float(match.group(2))
            num_frames = int(match.group(3))

            fps_results.append({
                'wall_time': wall_time,
                'fps': fps,
                'num_frames': num_frames
            })

            total_time += wall_time
            total_fps += fps
            total_frames += num_frames
            measurement_type = 'legacy_with_data_transfer'

    if not fps_results:
        return {}, []

    # Exclude first sequence from statistics (warmup)
    # But keep it in the full results for reference
    fps_results_for_stats = fps_results[1:] if len(fps_results) > 1 else fps_results

    # Recompute statistics excluding first sequence
    if len(fps_results) > 1:
        total_fps_excl_first = sum(r['fps'] for r in fps_results_for_stats)
        total_time_excl_first = sum(r['wall_time'] for r in fps_results_for_stats)
        total_frames_excl_first = sum(r['num_frames'] for r in fps_results_for_stats)

        avg_fps = total_fps_excl_first / len(fps_results_for_stats) if fps_results_for_stats else 0
        weighted_avg_fps = total_frames_excl_first / total_time_excl_first if total_time_excl_first > 0 else 0
    else:
        # Only one sequence, use it
        avg_fps = total_fps / len(fps_results)
        weighted_avg_fps = total_frames / total_time if total_time > 0 else 0
        total_frames_excl_first = total_frames
        total_time_excl_first = total_time

    summary = {
        'measurement_type': measurement_type,
        'avg_fps': round(avg_fps, 2),
        'weighted_avg_fps': round(weighted_avg_fps, 2),
        'total_sequences': len(fps_results),
        'sequences_used_for_stats': len(fps_results_for_stats),
        'total_frames': total_frames_excl_first,
        'total_time': round(total_time_excl_first, 2),
        'note': 'First sequence excluded from statistics (warmup). inference_only = data pre-loaded to GPU'
    }

    return summary, fps_results


def save_results(summary, fps_results, output_dir):
    """Save FPS results to JSON files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary (like test_results.json)
    if summary:
        results_path = output_dir / "fps_results.json"
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✓ Saved FPS summary to {results_path}")

        # Print summary
        measurement_label = "Inference-only (data pre-loaded)" if summary.get('measurement_type') == 'inference_only' else "Legacy (includes data transfer)"
        logger.info("\n" + "="*50)
        logger.info(f"FPS PERFORMANCE SUMMARY - {measurement_label}")
        logger.info("="*50)
        logger.info(f"  Average FPS: {summary['avg_fps']:.2f}")
        logger.info(f"  Weighted Avg FPS: {summary['weighted_avg_fps']:.2f}")
        logger.info(f"  Total Sequences: {summary['total_sequences']}")
        logger.info(f"  Sequences Used for Stats: {summary['sequences_used_for_stats']} (first excluded for warmup)")
        logger.info(f"  Total Frames: {summary['total_frames']}")
        logger.info(f"  Total Time: {summary['total_time']:.2f}s")

    # Save per-sequence FPS results
    if fps_results:
        # Mark first sequence as warmup
        fps_results_with_labels = []
        for idx, result in enumerate(fps_results):
            labeled_result = result.copy()
            labeled_result['sequence_id'] = idx
            if idx == 0:
                labeled_result['note'] = 'warmup - excluded from statistics'
            fps_results_with_labels.append(labeled_result)

        per_sequence_path = output_dir / "per_sequence_fps.json"
        with open(per_sequence_path, 'w') as f:
            json.dump(fps_results_with_labels, f, indent=2)
        logger.info(f"✓ Saved per-sequence FPS to {per_sequence_path}")
        logger.info(f"  Total sequences: {len(fps_results)}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python parse_flashdepth_results.py <log_file> <output_dir>")
        sys.exit(1)

    log_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    if not log_path.exists():
        logger.error(f"Log file not found: {log_path}")
        sys.exit(1)

    logger.info(f"Parsing log file: {log_path}")
    summary, fps_results = parse_flashdepth_log(log_path)

    if not summary and not fps_results:
        logger.warning("No FPS data found in log file!")
        logger.warning("Make sure the log contains inference results with FPS measurements.")
        sys.exit(1)

    save_results(summary, fps_results, output_dir)
    logger.info("\n✓ Done!")


if __name__ == "__main__":
    main()
