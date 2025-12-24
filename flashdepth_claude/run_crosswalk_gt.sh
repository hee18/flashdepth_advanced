#!/bin/bash
#
# Generate Pedestrian Depth GT from NuScenes Annotations
#
# Uses nuScenes 3D annotations (not YOLO segmentation) to generate
# depth ground truth for pedestrians. Frame numbers are mapped to
# sweep indices for comparison with depth estimation results.
#
# Usage: ./run_crosswalk_gt.sh [options]
#
# Examples:
#   ./run_crosswalk_gt.sh
#   ./run_crosswalk_gt.sh --output-dir test_results/my_gt

set -e

# Default values (Docker paths)
DATA_DIR="/data/datasets/v1.0-mini"
OUTPUT_DIR="test_results/crosswalk_gt"
PERSON_ONLY=""  # Default: person only
FRAME_INTERVAL=1

# Help function
show_help() {
    cat << EOF
Generate Pedestrian Depth GT from NuScenes Annotations

Usage: ./run_crosswalk_gt.sh [options]

This script generates Ground Truth depth trajectories from nuScenes 3D
annotations for the crosswalk_sample. Unlike the YOLO+LiDAR approach,
this uses official annotation bounding boxes.

Key features:
- Uses nuScenes sample_annotation.json (3D bboxes)
- Depth = nearest face of bbox (center depth - length/2)
- Frame numbers mapped to sweep indices for comparison

Options:
  --data-dir <path>       NuScenes data directory (default: /data/datasets/v1.0-mini)
  --output-dir <path>     Output directory (default: test_results/crosswalk_gt)
  --no-person-only        Include all annotated objects, not just pedestrians
  --frame-interval <n>    Save every Nth frame image (default: 1)
  --help                  Show this help message

Output:
  Creates in <output-dir>/crosswalk_sample/:
  - instance_tracking_results.json  Annotation-based GT trajectories
  - trajectory_plot.png             Depth vs lateral position plot
  - depth_timeline.png              Depth over time plot
  - frames/                         Visualization frames (sweep-indexed)

JSON output format:
  - frame_index_type: "sweep" (frame numbers are sweep indices)
  - frame_mapping: sample_idx → sweep_idx mapping
  - depth_type: "nearest_face" (depth calculation method)

Example workflow:
  1. Generate GT:
     ./run_crosswalk_gt.sh

  2. Run depth models on crosswalk_sweep.mp4:
     # (use sweep video for matching frame indices)

  3. Compare results:
     python scripts/compare_instance_results.py \\
         --gt-path test_results/crosswalk_gt/crosswalk_sample/instance_tracking_results.json \\
         --pred-paths test_results/instance_gear5_l/crosswalk_sample/instance_tracking_results.json \\
         --output-dir test_results/crosswalk_comparison

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --no-person-only)
            PERSON_ONLY="--no-person-only"
            shift
            ;;
        --frame-interval)
            FRAME_INTERVAL="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

echo "Generate Pedestrian Depth GT from NuScenes Annotations"
echo "======================================================="
echo "Configuration:"
echo "  - Data directory: $DATA_DIR"
echo "  - Output directory: $OUTPUT_DIR"
echo "  - Frame interval: $FRAME_INTERVAL"
echo "  - Person only: $([ -z "$PERSON_ONLY" ] && echo 'yes' || echo 'no')"
echo ""

# Build command
CMD="python scripts/generate_nuscenes_annotation_gt.py \
    --data-dir $DATA_DIR \
    --output-dir $OUTPUT_DIR \
    --frame-interval $FRAME_INTERVAL"

# Add optional flags
if [ -n "$PERSON_ONLY" ]; then
    CMD="$CMD $PERSON_ONLY"
fi

echo "Running in Docker container..."
echo ""

# Execute in Docker (flashdepth container)
docker compose run --rm flashdepth $CMD

echo ""
echo "GT generation complete!"
echo "Results saved to: $OUTPUT_DIR/crosswalk_sample/"
echo ""
echo "Frame indices are sweep-based for comparison with sweep video results."
