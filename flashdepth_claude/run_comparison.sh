#!/bin/bash
#
# Run IMAGE depth estimation methods evaluation (frame-by-frame)
#
# Usage: ./run_comparison.sh <method> [options]
#
# Methods: metric3d, unidepth, zoedepth, depthpro, cut3r, depthanythingv2
#
# Examples:
#   ./run_comparison.sh metric3d --version v2 --dataset sintel --gpu 1
#   ./run_comparison.sh unidepth --version v1 --dataset kitti --gpu 0
#   ./run_comparison.sh depthpro --dataset waymo --gpu 0
#
# For VIDEO models (vda, depthcrafter), use ./run_video_comparison.sh
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
CHECKPOINT=""
RESULTS_DIR=""
# New options for depth mode and model-specific settings
DEPTH_MODE="metric"
INDOOR=false
METRIC_MODE=false
FRAME_INTERVAL=""
ONLY_CLONE=true  # For VKITTI: only use 'clone' condition by default
VISUALIZATION="true"  # Enable visualizations by default
SEQ=""  # Sequence selection for UnrealStereo4K
BEST_FIGURE=false  # Export best_frame ±4 frames (9 total) as individual images/depth maps
FRAME=""  # Specific frame to export ±4 frames
AMP=false
AMP_DTYPE="bf16"
LIMIT_SCENES="" # New: Limit NuScenes scenes
TEST_MODE=""  # Test mode: empty (full), tc (temporal consistency only)
TC_THRESHOLD=""  # Threshold for rTC metric (default: 1.25)
MAX_DEPTH=""  # Max depth for evaluation (default: 80)

# Help function
show_help() {
    cat << EOF
IMAGE Depth Models Evaluation Script (frame-by-frame processing)

Usage: ./run_comparison.sh <method> [options]

Methods (IMAGE MODELS - process one frame at a time):
  depthanythingv2  Depth-Anything-V2 (metric depth)
  metric3d         Metric3D (specify --version v1 or v2)
  unidepth         UniDepth (specify --version v1 or v2)
  zoedepth         ZoeDepth
  depthpro         DepthPro (Apple ML)
  cut3r            CUT3R

For VIDEO models (process entire sequences):
  Use ./run_video_comparison.sh instead
  Supported: vda (Video-Depth-Anything), depthcrafter

Options:
  --dataset <name>         Dataset name (default: waymo)
                           Options: waymo, sintel, kitti, scannet, tartanair, bonn, nyu, vkitti
                           For object-wise: waymo_seg, urbansyn_seg, vkitti_seg
  --version <v1|v2>        Method version (for metric3d, unidepth)
  --gpu <id>               GPU device ID (default: 0)
  --workers <n>            Number of data loading workers (default: 4)
  --vid-len <n>            Video sequence length (default: 50)
  --only-clone <true|false> For VKITTI: use only 'clone' condition (default: true)
  --checkpoint <path>      Model checkpoint path
  --results-dir <path>     Results directory
  --depth-mode <mode>      Depth evaluation mode: metric or relative (default: metric)
  --indoor                 Use indoor checkpoint (for depthanythingv2 only)
  --metric                 Use metric mode (for vda only)
  --frame-interval <n>     Frame interval for sequence.png visualization
  --visualization <true|false> Enable/disable visualizations (sequence.png, best_frame.png, etc.). Default: true
  --seq <n>                Sequence number(s) for UnrealStereo4K (0-8). Examples: 0, 2,5, 0,3,7
  --best-figure            Export best_frame ±4 frames (9 total) as individual images/depth maps
  --frame <n>              Export frame N ±4 frames (9 total) as individual images/depth maps (e.g., --seq 6 --frame 459)
  --amp                    Enable Automatic Mixed Precision (AMP) for inference
  --amp-dtype <bf16|fp16>  Data type for AMP (default: bf16)
  --limit-scenes <n>       For NuScenes, limit the number of scenes to process (e.g., 50)
  --test-mode <mode>       Test mode: tc (temporal consistency only)
  --tc-threshold <float>   Threshold for rTC metric (default: 1.25)
  --max-depth <float>      Max depth for evaluation in meters (default: 80)
  --save-depth-maps        Save predicted depth maps as .npy files
  --help                   Show this help message

Examples:
  # Test UniDepth v2 on Unreal4K with AMP to reduce memory usage
  ./run_comparison.sh unidepth --version v2 --dataset unreal4k --gpu 0 --amp

  # Test Metric3D v2 on Sintel
  ./run_comparison.sh metric3d --version v2 --dataset sintel --gpu 1

  # Test with specific sequences and export best figure frames
  ./run_comparison.sh depthanythingv2 --dataset sintel --seq 0,3,7 --best-figure --gpu 0

  # Test specific frame export
  ./run_comparison.sh depthanythingv2 --dataset unreal4k --seq 6 --frame 459 --gpu 0

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
        --seq)
            SEQ="$2"
            shift 2
            ;;
        --best-figure)
            BEST_FIGURE=true
            shift
            ;;
        --frame)
            FRAME="$2"
            shift 2
            ;;
        --amp)
            AMP=true
            shift
            ;;
        --amp-dtype)
            AMP_DTYPE="$2"
            shift 2
            ;;
        --limit-scenes)
            LIMIT_SCENES="$2"
            shift 2
            ;;
        --test-mode)
            TEST_MODE="$2"
            shift 2
            ;;
        --tc-threshold)
            TC_THRESHOLD="$2"
            shift 2
            ;;
        --max-depth)
            MAX_DEPTH="$2"
            shift 2
            ;;
        --save-depth-maps)
            SAVE_DEPTH_MAPS=true
            shift
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

# === Dataset vid-len defaults ===
get_vid_len_for_dataset() {
    local ds="$1"
    case "$ds" in
        eth3d)       echo 30 ;;
        sintel)      echo 50 ;;
        bonn)        echo 50 ;;
        waymo_seg|waymo) echo 200 ;;
        vkitti)      echo 200 ;;
        unreal4k)    echo 500 ;;
        urbansyn)    echo 50 ;;
        *)           echo 50 ;;
    esac
}

get_workers_for_dataset() {
    local ds="$1"
    case "$ds" in
        eth3d|unreal4k) echo 1 ;;
        waymo_seg|waymo) echo 2 ;;
        sintel|vkitti)  echo 4 ;;
        *)              echo 4 ;;
    esac
}

# === Model lists ===
ALL_IMAGE_MODELS="depthanythingv2 metric3d unidepth zoedepth depthpro"
ALL_DATASETS="eth3d sintel waymo_seg vkitti unreal4k"

# === "all" expansion for models ===
# metric3d and unidepth have versions — expand them
expand_models() {
    local models=""
    for m in $ALL_IMAGE_MODELS; do
        case "$m" in
            metric3d)
                models="$models metric3d:v1 metric3d:v2"
                ;;
            unidepth)
                models="$models unidepth:v1 unidepth:v2"
                ;;
            *)
                models="$models $m"
                ;;
        esac
    done
    echo "$models"
}

# Validate method
# Reject VIDEO models (they need test_video_comparison.py)
case $METHOD in
    vda|depthcrafter)
        echo "❌ Error: '$METHOD' is a video model"
        echo "   Video models process entire sequences, not frame-by-frame"
        echo "   Use ./run_video_comparison.sh instead for video models"
        echo ""
        echo "   Example: ./run_video_comparison.sh $METHOD --dataset $DATASET --gpu $GPU_ID"
        exit 1
        ;;
esac

# Validate IMAGE models (including "all")
case $METHOD in
    all|depthanythingv2|metric3d|unidepth|zoedepth|depthpro|cut3r)
        ;;
    *)
        echo "Error: Unknown method '$METHOD'"
        echo "Valid IMAGE methods: all, depthanythingv2, metric3d, unidepth, zoedepth, depthpro, cut3r"
        echo ""
        echo "For VIDEO methods (vda, depthcrafter), use ./run_video_comparison.sh"
        exit 1
        ;;
esac

# Build method name with version
METHOD_NAME=$METHOD
if [ -n "$VERSION" ]; then
    METHOD_NAME="${METHOD}_${VERSION}"
fi

# Determine max-depth suffix for results dir
if [ -n "$MAX_DEPTH" ]; then
    MAX_DEPTH_SUFFIX="_${MAX_DEPTH}"
else
    MAX_DEPTH_SUFFIX=""
fi

# === Handle "all" mode (models and/or datasets) ===
if [ "$METHOD" = "all" ] || [ "$DATASET" = "all" ]; then
    # Determine model list
    if [ "$METHOD" = "all" ]; then
        MODEL_LIST=$(expand_models)
    else
        if [ -n "$VERSION" ]; then
            MODEL_LIST="${METHOD}:${VERSION}"
        else
            MODEL_LIST="$METHOD"
        fi
    fi

    # Determine dataset list
    if [ "$DATASET" = "all" ]; then
        DATASET_LIST="$ALL_DATASETS"
    else
        DATASET_LIST="$DATASET"
    fi

    USER_VID_LEN_SET=false
    # Check if user explicitly set --vid-len
    for arg in "$@"; do
        if [ "$arg" = "--vid-len" ]; then
            USER_VID_LEN_SET=true
            break
        fi
    done

    echo "========================================"
    echo "Batch Evaluation Mode"
    echo "========================================"
    echo "Models: $MODEL_LIST"
    echo "Datasets: $DATASET_LIST"
    echo "GPU: $GPU_ID"
    if [ -n "$MAX_DEPTH" ]; then
        echo "Max Depth: $MAX_DEPTH"
    fi
    echo "========================================"
    echo "Press Ctrl+C twice quickly to abort all runs"
    echo ""

    # Trap SIGINT (Ctrl+C) to abort entire batch
    BATCH_ABORT=false
    trap 'echo ""; echo "⚠️  Ctrl+C received — aborting batch..."; BATCH_ABORT=true' INT

    TOTAL_RUNS=0
    COMPLETED_RUNS=0
    FAILED_RUNS=0

    # Count total runs
    for model_entry in $MODEL_LIST; do
        for ds in $DATASET_LIST; do
            TOTAL_RUNS=$((TOTAL_RUNS + 1))
        done
    done

    for model_entry in $MODEL_LIST; do
        # Parse model:version format
        CUR_METHOD="${model_entry%%:*}"
        CUR_VERSION="${model_entry#*:}"
        if [ "$CUR_VERSION" = "$CUR_METHOD" ]; then
            CUR_VERSION=""
        fi

        CUR_METHOD_NAME="$CUR_METHOD"
        if [ -n "$CUR_VERSION" ]; then
            CUR_METHOD_NAME="${CUR_METHOD}_${CUR_VERSION}"
        fi

        for ds in $DATASET_LIST; do
            # Check abort flag
            if [ "$BATCH_ABORT" = true ]; then
                break 2
            fi

            COMPLETED_RUNS=$((COMPLETED_RUNS + 1))

            # Auto vid-len and workers per dataset (unless user explicitly set)
            if [ "$USER_VID_LEN_SET" = false ]; then
                CUR_VID_LEN=$(get_vid_len_for_dataset "$ds")
            else
                CUR_VID_LEN=$VID_LEN
            fi
            CUR_WORKERS=$(get_workers_for_dataset "$ds")

            # Auto results directory: model/version/dataset or model/dataset
            if [ -n "$CUR_VERSION" ]; then
                CUR_RESULTS_DIR="refer_test/test_results/${CUR_METHOD}/${CUR_VERSION}/${ds}${MAX_DEPTH_SUFFIX}"
            else
                CUR_RESULTS_DIR="refer_test/test_results/${CUR_METHOD}/${ds}${MAX_DEPTH_SUFFIX}"
            fi

            echo ""
            echo "========================================"
            echo "[$COMPLETED_RUNS/$TOTAL_RUNS] $CUR_METHOD_NAME on $ds (vid-len=$CUR_VID_LEN)"
            echo "  Results: $CUR_RESULTS_DIR"
            echo "========================================"

            # Build per-run command
            RUN_ARGS="--dataset $ds --gpu $GPU_ID --workers $CUR_WORKERS --vid-len $CUR_VID_LEN --results-dir $CUR_RESULTS_DIR"
            if [ -n "$CUR_VERSION" ]; then
                RUN_ARGS="$RUN_ARGS --version $CUR_VERSION"
            fi
            if [ -n "$MAX_DEPTH" ]; then
                RUN_ARGS="$RUN_ARGS --max-depth $MAX_DEPTH"
            fi
            if [ -n "$TEST_MODE" ]; then
                RUN_ARGS="$RUN_ARGS --test-mode $TEST_MODE"
            fi
            if [ -n "$TC_THRESHOLD" ]; then
                RUN_ARGS="$RUN_ARGS --tc-threshold $TC_THRESHOLD"
            fi
            if [ "$VISUALIZATION" != "true" ]; then
                RUN_ARGS="$RUN_ARGS --visualization $VISUALIZATION"
            fi
            if [ "$AMP" = true ]; then
                RUN_ARGS="$RUN_ARGS --amp --amp-dtype $AMP_DTYPE"
            fi
            if [ "$METRIC_MODE" = true ]; then
                RUN_ARGS="$RUN_ARGS --metric"
            fi
            if [ "$INDOOR" = true ]; then
                RUN_ARGS="$RUN_ARGS --indoor"
            fi
            if [ "$SAVE_DEPTH_MAPS" = true ]; then
                RUN_ARGS="$RUN_ARGS --save-depth-maps"
            fi

            # Recursive call (single model + single dataset)
            if bash "$0" "$CUR_METHOD" $RUN_ARGS; then
                echo "✅ [$COMPLETED_RUNS/$TOTAL_RUNS] $CUR_METHOD_NAME on $ds completed"
            else
                echo "❌ [$COMPLETED_RUNS/$TOTAL_RUNS] $CUR_METHOD_NAME on $ds FAILED"
                FAILED_RUNS=$((FAILED_RUNS + 1))
            fi
        done
    done

    # Restore default signal handling
    trap - INT

    echo ""
    echo "========================================"
    if [ "$BATCH_ABORT" = true ]; then
        echo "Batch ABORTED by user (Ctrl+C)"
    fi
    echo "Batch Evaluation Summary"
    echo "  Total: $TOTAL_RUNS, Completed: $((TOTAL_RUNS - FAILED_RUNS)), Failed: $FAILED_RUNS"
    echo "========================================"
    exit 0
fi

# Set default results directory (single model + single dataset)
# model/version/dataset or model/dataset
if [ -z "$RESULTS_DIR" ]; then
    if [ -n "$VERSION" ]; then
        RESULTS_DIR="refer_test/test_results/${METHOD}/${VERSION}/${DATASET}${MAX_DEPTH_SUFFIX}"
    else
        RESULTS_DIR="refer_test/test_results/${METHOD}/${DATASET}${MAX_DEPTH_SUFFIX}"
    fi
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
if [ "$AMP" = true ]; then
    echo "AMP: ENABLED (dtype: $AMP_DTYPE)"
fi
if [ -n "$LIMIT_SCENES" ]; then
    echo "Limit Scenes (NuScenes): $LIMIT_SCENES"
fi
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

CMD="python test_comparison.py"
CMD="$CMD --method $METHOD"
CMD="$CMD --dataset $DATASET"
CMD="$CMD --data-root /data/datasets"
CMD="$CMD --gpu $CONTAINER_GPU"
CMD="$CMD --workers $WORKERS"
CMD="$CMD --video-length $VID_LEN"
CMD="$CMD --results-dir $RESULTS_DIR"
CMD="$CMD --depth-mode $DEPTH_MODE"

if [ -n "$VERSION" ]; then
    CMD="$CMD --version $VERSION"
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

if [ -n "$SEQ" ]; then
    CMD="$CMD --seq $SEQ"
fi

if [ "$BEST_FIGURE" = true ]; then
    CMD="$CMD --best-figure"
fi

if [ -n "$FRAME" ]; then
    CMD="$CMD --frame $FRAME"
fi

if [ "$AMP" = true ]; then
    CMD="$CMD --amp --amp-dtype $AMP_DTYPE"
fi

if [ -n "$LIMIT_SCENES" ]; then
    CMD="$CMD --limit-scenes $LIMIT_SCENES"
fi

if [ -n "$TEST_MODE" ]; then
    CMD="$CMD --test-mode $TEST_MODE"
fi

if [ -n "$TC_THRESHOLD" ]; then
    CMD="$CMD --tc-threshold $TC_THRESHOLD"
fi

if [ -n "$MAX_DEPTH" ]; then
    CMD="$CMD --max-depth $MAX_DEPTH"
fi

if [ "$SAVE_DEPTH_MAPS" = true ]; then
    CMD="$CMD --save-depth-maps"
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

    # Run with Docker (ensure pyarrow for Waymo parquet support)
    CUDA_VISIBLE_DEVICES=$GPU_ID docker compose run --rm comparison \
        bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate $CONDA_ENV && pip install -q pyarrow 2>/dev/null; exec $CMD"
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
