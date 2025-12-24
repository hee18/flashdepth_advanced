#!/bin/bash
#
# Instance Segmentation + VIDEO Depth Models Testing
# YOLOv11 instance segmentation + tracking + video depth models
#
# Usage: ./run_instance_video_comparison.sh <method> [options]
#
# Methods: vda (Video-Depth-Anything), depthcrafter
#
# Examples:
#   ./run_instance_video_comparison.sh vda --video-path /data/datasets/videos_mfdepth --gpu 0
#   ./run_instance_video_comparison.sh vda --metric --video-path /data/datasets/videos_mfdepth
#   ./run_instance_video_comparison.sh depthcrafter --video-path /data/datasets/videos_mfdepth
#

set -e

# Default values
METHOD=""
VIDEO_PATH="/data/datasets/videos_mfdepth"
RESULTS_DIR=""
GPU_ID=0
FRAME_INTERVAL=1
CHECKPOINT=""
SEG_MODEL="yolo11x-seg.pt"
TRACKER="botsort.yaml"
PERSON_ONLY=true
CENTER_MASK=true
METRIC_FLAG=""  # For VDA metric mode
MAX_RES=1024    # For DepthCrafter
MAX_FRAMES=500  # Memory limit

# Help function
show_help() {
    cat << EOF
Instance Segmentation + VIDEO Depth Models Testing Script

Usage: ./run_instance_video_comparison.sh <method> [options]

This script combines YOLOv11 instance segmentation + BoTSORT tracking
with VIDEO-based depth estimation models to track per-instance depth over time.

VIDEO models process the entire sequence at once for temporal consistency.

Methods (VIDEO MODELS - process entire sequences):
  vda              Video-Depth-Anything (supports --metric for metric depth)
  depthcrafter     DepthCrafter (always relative depth)

Options:
  --video-path <path>       Video file or directory (default: /data/datasets/videos_mfdepth)
  --results-dir <path>      Results directory (default: test_results/instance_video_comparison/<method>)
  --gpu <id>                GPU device ID (default: 0)
  --frame-interval <n>      Save every Nth frame to video (default: 1)
  --checkpoint <path>       Model checkpoint path
  --seg-model <name>        YOLOv11 segmentation model (default: yolo11x-seg.pt)
  --tracker <name>          Tracker config (default: botsort.yaml)
  --no-person-only          Track all classes, not just person
  --no-center-mask          Use full mask instead of center mask (erosion + circle)
  --metric                  Use metric depth mode for VDA (default: relative)
  --max-res <n>             Maximum resolution for DepthCrafter (default: 1024)
  --max-frames <n>          Maximum frames per video (default: 500)
  --help                    Show this help message

Output:
  For each video, creates:
  - instance_tracking_results.json  Per-instance trajectories with depth
  - trajectory_plot.png             Depth vs lateral position plot
  - depth_timeline.png              Depth over time plot
  - result_video.mp4                Video with bounding boxes and depth overlay

Note:
  - VDA with --metric flag outputs metric depth (meters)
  - VDA without --metric outputs relative depth (0-1 normalized)
  - DepthCrafter always outputs relative depth (0-1 normalized)
  - For relative depth, lateral position values are normalized pixel offsets

Examples:
  # Test VDA with relative depth (default)
  ./run_instance_video_comparison.sh vda \\
      --video-path /data/datasets/videos_mfdepth --gpu 0

  # Test VDA with metric depth
  ./run_instance_video_comparison.sh vda --metric \\
      --video-path /data/datasets/videos_mfdepth --gpu 0

  # Test DepthCrafter (always relative)
  ./run_instance_video_comparison.sh depthcrafter \\
      --video-path /data/datasets/videos_mfdepth --gpu 0

  # Test with limited frames for memory
  ./run_instance_video_comparison.sh vda \\
      --video-path /data/datasets/videos_mfdepth --max-frames 200 --gpu 0

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
        --metric)
            METRIC_FLAG="--metric"
            shift
            ;;
        --max-res)
            MAX_RES="$2"
            shift 2
            ;;
        --max-frames)
            MAX_FRAMES="$2"
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
    vda|depthcrafter)
        ;;
    *)
        echo "Error: Unknown method '$METHOD'"
        echo "Supported methods: vda, depthcrafter"
        echo ""
        echo "For IMAGE models (frame-by-frame), use ./run_instance_comparison.sh"
        exit 1
        ;;
esac

# Map method to conda environment
case $METHOD in
    vda)
        CONDA_ENV="vda"
        ;;
    depthcrafter)
        CONDA_ENV="depthcrafter"
        ;;
esac

# Set default results directory
if [ -z "$RESULTS_DIR" ]; then
    if [ "$METHOD" = "vda" ]; then
        if [ -n "$METRIC_FLAG" ]; then
            RESULTS_DIR="test_results/instance_video_comparison/vda_metric"
        else
            RESULTS_DIR="test_results/instance_video_comparison/vda_relative"
        fi
    else
        RESULTS_DIR="test_results/instance_video_comparison/${METHOD}_relative"
    fi
fi

# Determine depth mode description
if [ "$METHOD" = "depthcrafter" ]; then
    DEPTH_MODE="relative (always)"
elif [ -n "$METRIC_FLAG" ]; then
    DEPTH_MODE="metric"
else
    DEPTH_MODE="relative"
fi

echo "Instance Segmentation + $METHOD Depth Testing"
echo "=============================================="
echo "Configuration:"
echo "  - Method: $METHOD"
echo "  - Depth mode: $DEPTH_MODE"
echo "  - Video path: $VIDEO_PATH"
echo "  - Results dir: $RESULTS_DIR"
echo "  - GPU: $GPU_ID"
echo "  - Conda env: $CONDA_ENV"
echo "  - Segmentation model: $SEG_MODEL"
echo "  - Tracker: $TRACKER"
echo "  - Max frames: $MAX_FRAMES"
if [ "$METHOD" = "depthcrafter" ]; then
    echo "  - Max resolution: $MAX_RES"
fi
echo ""

# Build command
TEST_CMD="python test_instance_video_comparison.py \
    --method $METHOD \
    --video-path '$VIDEO_PATH' \
    --results-dir '$RESULTS_DIR' \
    --gpu $GPU_ID \
    --seg-model $SEG_MODEL \
    --tracker $TRACKER \
    --frame-interval $FRAME_INTERVAL \
    --max-frames $MAX_FRAMES"

# Add metric flag for VDA
if [ -n "$METRIC_FLAG" ]; then
    TEST_CMD="$TEST_CMD $METRIC_FLAG"
fi

# Add max-res for DepthCrafter
if [ "$METHOD" = "depthcrafter" ]; then
    TEST_CMD="$TEST_CMD --max-res $MAX_RES"
fi

# Add checkpoint if specified
if [ -n "$CHECKPOINT" ]; then
    TEST_CMD="$TEST_CMD --checkpoint-path '$CHECKPOINT'"
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
