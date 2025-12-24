#!/bin/bash
#
# Instance Segmentation + IMAGE Depth Models Testing
# YOLOv11 instance segmentation + tracking + comparison depth models
#
# Usage: ./run_instance_comparison.sh <method> [options]
#
# Methods: metric3d, unidepth, zoedepth, depthpro, depthanythingv2, cut3r
#
# Examples:
#   ./run_instance_comparison.sh metric3d --video-path /data/datasets/videos_mfdepth --gpu 0
#   ./run_instance_comparison.sh unidepth --version v2 --video-path /data/datasets/videos_mfdepth
#

set -e

# Default values
METHOD=""
VERSION=""
VIDEO_PATH="/data/datasets/videos_mfdepth"
RESULTS_DIR=""
GPU_ID=0
FRAME_INTERVAL=1
CHECKPOINT=""
SEG_MODEL="yolo11x-seg.pt"
TRACKER="botsort.yaml"
PERSON_ONLY=true
CENTER_MASK=true
ENCODER="vitl"  # For DepthAnythingV2

# Help function
show_help() {
    cat << EOF
Instance Segmentation + IMAGE Depth Models Testing Script

Usage: ./run_instance_comparison.sh <method> [options]

This script combines YOLOv11 instance segmentation + BoTSORT tracking
with IMAGE-based depth estimation models to track per-instance depth over time.

Methods (IMAGE MODELS - process one frame at a time):
  metric3d         Metric3D (specify --version v1 or v2)
  unidepth         UniDepth (specify --version v1 or v2)
  zoedepth         ZoeDepth
  depthpro         DepthPro (Apple ML)
  depthanythingv2  Depth-Anything-V2 (metric depth)
  cut3r            CUT3R

Options:
  --video-path <path>       Video file or directory (default: /data/datasets/videos_mfdepth)
  --results-dir <path>      Results directory (default: test_results/instance_comparison/<method>)
  --version <v1|v2>         Model version (for metric3d, unidepth)
  --gpu <id>                GPU device ID (default: 0)
  --frame-interval <n>      Save every Nth frame to video (default: 1)
  --checkpoint <path>       Model checkpoint path
  --seg-model <name>        YOLOv11 segmentation model (default: yolo11x-seg.pt)
  --tracker <name>          Tracker config (default: botsort.yaml)
  --no-person-only          Track all classes, not just person
  --no-center-mask          Use full mask instead of center mask (erosion + circle)
  --encoder <vits|vitb|vitl> Encoder for DepthAnythingV2 (default: vitl)
  --help                    Show this help message

Output:
  For each video, creates:
  - instance_tracking_results.json  Per-instance trajectories with depth
  - trajectory_plot.png             Depth vs lateral position plot
  - depth_timeline.png              Depth over time plot
  - result_video.mp4                Video with bounding boxes and depth overlay

Examples:
  # Test Metric3D v2 on all videos
  ./run_instance_comparison.sh metric3d --version v2 \\
      --video-path /data/datasets/videos_mfdepth --gpu 0

  # Test UniDepth v1 on single video
  ./run_instance_comparison.sh unidepth --version v1 \\
      --video-path /data/datasets/videos_mfdepth/nusc_peds6.mp4 --gpu 1

  # Test ZoeDepth
  ./run_instance_comparison.sh zoedepth \\
      --video-path /data/datasets/videos_mfdepth --gpu 0

  # Test DepthPro
  ./run_instance_comparison.sh depthpro \\
      --video-path /data/datasets/videos_mfdepth --gpu 0

EOF
}

# Parse arguments
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

METHOD=$1
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        --video-path)
            VIDEO_PATH="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --frame-interval)
            FRAME_INTERVAL="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --seg-model)
            SEG_MODEL="$2"
            shift 2
            ;;
        --tracker)
            TRACKER="$2"
            shift 2
            ;;
        --no-person-only)
            PERSON_ONLY=false
            shift
            ;;
        --no-center-mask)
            CENTER_MASK=false
            shift
            ;;
        --encoder)
            ENCODER="$2"
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

# Validate method
case $METHOD in
    metric3d|unidepth|zoedepth|depthpro|depthanythingv2|cut3r)
        ;;
    *)
        echo "Error: Unknown method '$METHOD'"
        echo "Supported methods: metric3d, unidepth, zoedepth, depthpro, depthanythingv2, cut3r"
        exit 1
        ;;
esac

# Map method to conda environment
case $METHOD in
    metric3d)
        CONDA_ENV="metric3d"
        ;;
    unidepth)
        CONDA_ENV="unidepth"
        ;;
    zoedepth)
        CONDA_ENV="zoedepth"
        ;;
    depthpro)
        CONDA_ENV="depthpro"
        ;;
    depthanythingv2)
        CONDA_ENV="depthanythingv2"
        ;;
    cut3r)
        CONDA_ENV="cut3r"
        ;;
esac

# Set default results directory
if [ -z "$RESULTS_DIR" ]; then
    RESULTS_DIR="test_results/instance_comparison/$METHOD"
fi

echo "Instance Segmentation + $METHOD Depth Testing"
echo "=============================================="
echo "Configuration:"
echo "  - Method: $METHOD"
echo "  - Video path: $VIDEO_PATH"
echo "  - Results dir: $RESULTS_DIR"
echo "  - GPU: $GPU_ID"
echo "  - Conda env: $CONDA_ENV"
echo "  - Segmentation model: $SEG_MODEL"
echo "  - Tracker: $TRACKER"
if [ -n "$VERSION" ]; then
    echo "  - Version: $VERSION"
fi
if [ "$METHOD" = "depthanythingv2" ]; then
    echo "  - Encoder: $ENCODER"
fi
echo ""

# Build command with version and other options
TEST_CMD="python test_instance_comparison.py \
    --method $METHOD \
    --video-path '$VIDEO_PATH' \
    --results-dir '$RESULTS_DIR' \
    --gpu $GPU_ID \
    --seg-model $SEG_MODEL \
    --tracker $TRACKER \
    --frame-interval $FRAME_INTERVAL"

# Add version if specified
if [ -n "$VERSION" ]; then
    TEST_CMD="$TEST_CMD --version $VERSION"
fi

# Add checkpoint if specified
if [ -n "$CHECKPOINT" ]; then
    TEST_CMD="$TEST_CMD --checkpoint-path '$CHECKPOINT'"
fi

# Add encoder for DepthAnythingV2
if [ "$METHOD" = "depthanythingv2" ]; then
    TEST_CMD="$TEST_CMD --encoder $ENCODER"
fi

# Add person-only flag
if [ "$PERSON_ONLY" = "false" ]; then
    TEST_CMD="$TEST_CMD --no-person-only"
fi

# Add center-mask flag
if [ "$CENTER_MASK" = "false" ]; then
    TEST_CMD="$TEST_CMD --no-center-mask"
fi

# Run with comparison Docker container
# Note: CUDA_VISIBLE_DEVICES selects the GPU for Docker, but inside the container
# the selected GPU appears as GPU 0, so we always use --gpu 0 inside
echo "Running test with $METHOD in $CONDA_ENV environment..."

# Replace --gpu $GPU_ID with --gpu 0 for Docker internal use
TEST_CMD_DOCKER=$(echo "$TEST_CMD" | sed "s/--gpu $GPU_ID/--gpu 0/")

CUDA_VISIBLE_DEVICES=$GPU_ID docker compose run --rm comparison bash -c "
    source /opt/conda/etc/profile.d/conda.sh && \
    conda activate $CONDA_ENV && \
    $TEST_CMD_DOCKER
"

echo ""
echo "Test completed! Results saved to: $RESULTS_DIR"
