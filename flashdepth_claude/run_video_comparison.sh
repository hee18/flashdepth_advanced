#!/bin/bash
#
# Run VIDEO depth estimation methods evaluation
#
# Usage: ./run_video_comparison.sh <method> [options]
#
# Methods: vda, depthcrafter (video models only)
#
# Examples:
#   ./run_video_comparison.sh vda --dataset waymo --gpu 0
#   ./run_video_comparison.sh depthcrafter --dataset sintel --gpu 1
#
# Note: For image models (metric3d, unidepth, depthpro, etc.), use ./run_comparison.sh
#

set -e

# Default values
METHOD=""
VERSION=""
DATASET="waymo"
DATA_ROOT="/data/datasets"  # Docker internal path
GPU_ID=0
WORKERS=4
VID_LEN=50
OBJWISE=false
CHECKPOINT=""
RESULTS_DIR=""
# New options for depth mode and model-specific settings
DEPTH_MODE="metric"
INDOOR=false
METRIC_MODE=false
FRAME_INTERVAL=""
ONLY_CLONE=true  # For VKITTI: only use 'clone' condition by default
VISUALIZATION="true"  # Enable visualizations by default

# Help function
show_help() {
    cat << EOF
VIDEO Depth Estimation Methods Evaluation Script

Usage: ./run_video_comparison.sh <method> [options]

Methods (VIDEO MODELS ONLY):
  vda              Video-Depth-Anything (processes entire sequences)
  depthcrafter     DepthCrafter (diffusion-based video model)

Note: For image models (metric3d, unidepth, depthpro, etc.), use ./run_comparison.sh instead

Options:
  --dataset <name>         Dataset name (default: waymo)
                           Options: waymo, sintel, kitti, scannet, tartanair, bonn, nyu, vkitti
                           For object-wise: waymo_seg, urbansyn_seg, vkitti_seg
  --version <v1|v2>        Method version (for metric3d, unidepth)
  --gpu <id>               GPU device ID (default: 0)
  --workers <n>            Number of data loading workers (default: 4)
  --vid-len <n>            Video sequence length (default: 50)
  --objwise                Enable object-wise evaluation
  --only-clone <true|false> For VKITTI: use only 'clone' condition (default: true)
  --checkpoint <path>      Model checkpoint path
  --results-dir <path>     Results directory
  --depth-mode <mode>      Depth evaluation mode: metric or relative (default: metric)
  --indoor                 Use indoor checkpoint (for depthanythingv2 only)
  --metric                 Use metric mode (for vda only)
  --frame-interval <n>     Frame interval for sequence.png visualization
  --visualization <true|false> Enable/disable visualizations (sequence.png, best_frame.png, etc.). Default: true
  --help                   Show this help message

Examples:
  # Test Video-Depth-Anything on Waymo
  ./run_comparison.sh vda --dataset waymo --gpu 0

  # Test Metric3D v2 on Sintel
  ./run_comparison.sh metric3d --version v2 --dataset sintel --gpu 1

  # Test UniDepth v1 with object-wise evaluation
  ./run_comparison.sh unidepth --version v1 --dataset waymo_seg --gpu 0 --objwise

  # Test DepthPro on KITTI
  ./run_comparison.sh depthpro --dataset kitti --gpu 0

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
        --dataset)
            DATASET="$2"
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
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --vid-len)
            VID_LEN="$2"
            shift 2
            ;;
        --objwise)
            OBJWISE=true
            shift
            ;;
        --only-clone)
            ONLY_CLONE="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --depth-mode)
            DEPTH_MODE="$2"
            shift 2
            ;;
        --indoor)
            INDOOR=true
            shift
            ;;
        --metric)
            METRIC_MODE=true
            shift
            ;;
        --frame-interval)
            FRAME_INTERVAL="$2"
            shift 2
            ;;
        --visualization)
            VISUALIZATION="$2"
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

# Validate method (VIDEO MODELS ONLY)
case $METHOD in
    vda|depthcrafter)
        ;;
    *)
        echo "❌ Error: '$METHOD' is not a video model"
        echo "   This script supports VIDEO MODELS ONLY: vda, depthcrafter"
        echo "   For image models (metric3d, unidepth, depthpro, etc.), use ./run_comparison.sh instead"
        exit 1
        ;;
esac

# Build method name with version
METHOD_NAME=$METHOD
if [ -n "$VERSION" ]; then
    METHOD_NAME="${METHOD}_${VERSION}"
fi

# Set default results directory
if [ -z "$RESULTS_DIR" ]; then
    RESULTS_DIR="refer_test/test_results/${METHOD_NAME}/${DATASET}"
fi

# Print configuration
echo "========================================"
echo "Comparison Method Evaluation"
echo "========================================"
echo "Method: $METHOD"
if [ -n "$VERSION" ]; then
    echo "Version: $VERSION"
fi
echo "Dataset: $DATASET"
echo "Depth Mode: $DEPTH_MODE"
echo "GPU: $GPU_ID"
echo "Workers: $WORKERS"
echo "Video Length: $VID_LEN"
echo "Object-wise: $OBJWISE"
if [[ "$DATASET" == *"vkitti"* ]]; then
    echo "Only Clone (VKITTI): $ONLY_CLONE"
fi
if [ "$INDOOR" = true ]; then
    echo "Indoor Mode: ENABLED"
fi
if [ "$METRIC_MODE" = true ]; then
    echo "Metric Mode: ENABLED"
fi
if [ -n "$FRAME_INTERVAL" ]; then
    echo "Frame Interval: $FRAME_INTERVAL"
fi
if [ -n "$CHECKPOINT" ]; then
    echo "Checkpoint: $CHECKPOINT"
fi
echo "Visualization: $VISUALIZATION"
echo "Results Dir: $RESULTS_DIR"
echo "========================================"

# Build command
# Note: When using Docker, CUDA_VISIBLE_DEVICES maps host GPU to container GPU 0
# So we always use --gpu 0 inside the container
if command -v docker &> /dev/null && [ -f "docker-compose.yml" ]; then
    CONTAINER_GPU=0
else
    CONTAINER_GPU=$GPU_ID
fi

CMD="python test_video_comparison.py"
CMD="$CMD --method $METHOD"
CMD="$CMD --dataset $DATASET"
CMD="$CMD --data-root $DATA_ROOT"
CMD="$CMD --gpu $CONTAINER_GPU"
CMD="$CMD --workers $WORKERS"
CMD="$CMD --video-length $VID_LEN"
CMD="$CMD --results-dir $RESULTS_DIR"
CMD="$CMD --depth-mode $DEPTH_MODE"

if [ -n "$VERSION" ]; then
    CMD="$CMD --version $VERSION"
fi

if [ "$OBJWISE" = true ]; then
    CMD="$CMD --objwise"
fi

if [[ "$DATASET" == *"vkitti"* ]]; then
    if [ "$ONLY_CLONE" = "true" ]; then
        CMD="$CMD --only-clone true"
    else
        CMD="$CMD --only-clone false"
    fi
fi

if [ "$INDOOR" = true ]; then
    CMD="$CMD --indoor"
fi

if [ "$METRIC_MODE" = true ]; then
    CMD="$CMD --metric"
fi

if [ -n "$FRAME_INTERVAL" ]; then
    CMD="$CMD --frame-interval $FRAME_INTERVAL"
fi

if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint $CHECKPOINT"
fi

CMD="$CMD --visualization $VISUALIZATION"

# Get required conda environment
case $METHOD in
    vda)
        CONDA_ENV="vda"
        ;;
    depthanythingv2)
        CONDA_ENV="depthanythingv2"
        ;;
    depthcrafter)
        CONDA_ENV="depthcrafter"
        ;;
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
    cut3r)
        CONDA_ENV="cut3r"
        ;;
esac

echo "Using conda environment: $CONDA_ENV"
echo ""

# Check if using Docker or direct execution
if command -v docker &> /dev/null && [ -f "docker-compose.yml" ]; then
    echo "Running with Docker..."
    echo "Command: $CMD"
    echo ""

    # Run with Docker
    CUDA_VISIBLE_DEVICES=$GPU_ID docker compose run --rm comparison \
        bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate $CONDA_ENV && $CMD"
else
    echo "Running directly (no Docker)..."
    echo "Command: $CMD"
    echo ""

    # Check if conda environment exists
    if ! conda env list | grep -q "^$CONDA_ENV "; then
        echo "Error: Conda environment '$CONDA_ENV' not found"
        echo "Please create the environment first:"
        echo "  conda create -n $CONDA_ENV python=3.10 -y"
        echo "  conda activate $CONDA_ENV"
        echo "  # Install required packages"
        exit 1
    fi

    # Run directly with conda
    eval "$(conda shell.bash hook)"
    conda activate $CONDA_ENV
    CUDA_VISIBLE_DEVICES=$GPU_ID $CMD
fi

echo ""
echo "========================================"
echo "Evaluation completed!"
echo "Results saved to: $RESULTS_DIR"
echo "========================================"
