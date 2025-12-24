#!/bin/bash
#
# Instance Segmentation + Depth Testing
# YOLOv11 instance segmentation + tracking + depth estimation (Gear5 or Original FlashDepth)
#
# Usage: ./run_instance_test.sh [options]
#
# Examples:
#   ./run_instance_test.sh --video-path /data/datasets/videos_mfdepth --gpu 0
#   ./run_instance_test.sh --video-path /data/datasets/videos_mfdepth/nusc_peds6.mp4 --gpu 1
#   ./run_instance_test.sh --original-flashdepth --video-path /data/datasets/videos_mfdepth --gpu 0
#

set -e

# Default values
VIDEO_PATH="/data/datasets/videos_mfdepth"
RESULTS_DIR=""
GPU_ID=0
FRAME_INTERVAL=1
CONFIG_VARIANT="l"
CHECKPOINT=""
SEG_MODEL="yolo11x-seg.pt"
TRACKER="botsort.yaml"
PERSON_ONLY="--person-only"
CENTER_MASK="--center-mask"
MAMBA="false"
CLS_LAYERS="2,4"
ORIGINAL_FLASHDEPTH="false"
RESOLUTION="base"  # base or 2k (resolution depends on video-source)
SHOW_DEPTH_VALUES="true"  # Show Z/X values on tracking labels (use --no-depth-viz to disable)
SPARSE_GT_DIR=""   # Sparse LiDAR GT directory for Original FlashDepth alignment
FX=""              # Auto-set based on video-source (override with --fx)
CANONICAL_FX="500.0"  # Canonical focal length (Metric3D/Gear5 standard)
VIDEO_SOURCE="nusc"  # nusc (NuScenes) or avante

# Help function
show_help() {
    cat << EOF
Instance Segmentation + Depth Testing Script

Usage: ./run_instance_test.sh [options]

This script combines YOLOv11 instance segmentation + BoTSORT tracking
with depth estimation to track per-instance depth over time.

Depth Models:
  - Gear5 (default): Metric depth estimation in meters
  - Original FlashDepth: Relative depth (use --original-flashdepth)

Options:
  --video-path <path>       Video file or directory (default: /data/datasets/videos_mfdepth)
  --results-dir <path>      Results directory (default: auto-generated based on model)
  --gpu <id>                GPU device ID (default: 0)
  --frame-interval <n>      Save every Nth frame to video (default: 1)
  --config-variant <l|s|hybrid>  Config variant (default: l)
  --checkpoint <path>       Model checkpoint path
  --seg-model <name>        YOLOv11 segmentation model (default: yolo11x-seg.pt)
  --tracker <name>          Tracker config (default: botsort.yaml)
  --no-person-only          Track all classes, not just person
  --no-center-mask          Use full mask instead of center mask (erosion + circle)
  --mamba                   Use Mamba2 for Gear5 TSP temporal modeling (Gear5 only)
  --cls-layers <layers>     CLS token extraction layers for Gear5 (default: 2,4)
  --original-flashdepth     Use Original FlashDepth (relative depth) instead of Gear5
  --resolution <base|2k>    Processing resolution (default: base)
                            Resolution depends on --video-source:
                            nusc:   base=924x518,  2k=1596x896
                            avante: base=756x518,  2k=1596x1092
  --video-source <nusc|avante>  Video source type (default: nusc)
                            Sets default fx and resolution mapping:
                            - nusc: fx=1266.4, orig 1600x900
                            - avante: fx=900, orig 1600x1100
  --sparse-gt-dir <path>    Sparse LiDAR GT directory for alignment (Original FlashDepth only)
                            Use with --original-flashdepth to align relative depth to metric
  --fx <float>              Actual focal length of video in pixels (auto-set from video-source)
                            Override this to use custom focal length
  --canonical-fx <float>    Canonical focal length (default: 500.0, Metric3D standard)
  --no-depth-viz            Hide Z/X depth values on tracking labels (show only track ID)
  --help                    Show this help message

Output:
  For each video, creates:
  - instance_tracking_results.json  Per-instance trajectories with depth
  - trajectory_plot.png             Depth vs lateral position plot
  - depth_timeline.png              Depth over time plot
  - result_video.mp4                Video with bounding boxes and depth overlay
  - depth_colormap_video.mp4        Depth colormap visualization

Examples:
  # Test with Gear5 (metric depth) on all videos
  ./run_instance_test.sh --video-path /data/datasets/videos_mfdepth --gpu 0

  # Test with Original FlashDepth (relative depth)
  ./run_instance_test.sh --original-flashdepth --video-path /data/datasets/videos_mfdepth --gpu 0

  # Test on single video with custom checkpoint
  ./run_instance_test.sh --video-path /data/datasets/videos_mfdepth/nusc_peds6.mp4 \\
      --checkpoint train_results/gear5/best.pth --gpu 1

  # Using Mamba2 temporal backend
  ./run_instance_test.sh --mamba --video-path /data/datasets/videos_mfdepth --gpu 0

  # Original FlashDepth with FlashDepth-S model
  ./run_instance_test.sh --original-flashdepth --config-variant s \\
      --video-path /data/datasets/videos_mfdepth --gpu 0

EOF
}

# Parse arguments
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
        --config-variant)
            CONFIG_VARIANT="$2"
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
            PERSON_ONLY=""
            shift
            ;;
        --no-center-mask)
            CENTER_MASK=""
            shift
            ;;
        --mamba)
            MAMBA="true"
            shift
            ;;
        --cls-layers)
            CLS_LAYERS="$2"
            shift 2
            ;;
        --original-flashdepth)
            ORIGINAL_FLASHDEPTH="true"
            shift
            ;;
        --resolution)
            RESOLUTION="$2"
            shift 2
            ;;
        --video-source)
            VIDEO_SOURCE="$2"
            shift 2
            ;;
        --sparse-gt-dir)
            SPARSE_GT_DIR="$2"
            shift 2
            ;;
        --fx)
            FX="$2"
            shift 2
            ;;
        --canonical-fx)
            CANONICAL_FX="$2"
            shift 2
            ;;
        --no-depth-viz)
            SHOW_DEPTH_VALUES="false"
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

# Set default results directory based on model type
if [ -z "$RESULTS_DIR" ]; then
    if [ "$ORIGINAL_FLASHDEPTH" = "true" ]; then
        RESULTS_DIR="test_results/instance_original_flashdepth_$CONFIG_VARIANT"
    else
        RESULTS_DIR="test_results/instance_gear5_$CONFIG_VARIANT"
    fi
fi

# Set default checkpoint based on model type
if [ -z "$CHECKPOINT" ]; then
    if [ "$ORIGINAL_FLASHDEPTH" = "true" ]; then
        # Original FlashDepth checkpoints
        case "$CONFIG_VARIANT" in
            l)
                CHECKPOINT="configs/flashdepth-l/iter_10001.pth"
                ;;
            s)
                CHECKPOINT="configs/flashdepth-s/iter_14001.pth"
                ;;
            hybrid)
                CHECKPOINT="configs/flashdepth/iter_43002.pth"
                ;;
        esac
    fi
    # Gear5: checkpoint is optional (uses default from config)
fi

# Set default FX based on video source (if not overridden)
if [ -z "$FX" ]; then
    case "$VIDEO_SOURCE" in
        avante)
            FX="900.0"
            ;;
        nusc|*)
            FX="1266.4"
            ;;
    esac
fi

# Determine config path based on model type
if [ "$ORIGINAL_FLASHDEPTH" = "true" ]; then
    # Original FlashDepth uses flashdepth configs
    case "$CONFIG_VARIANT" in
        l)
            CONFIG_PATH="configs/flashdepth-l"
            CONFIG_NAME="config"
            ;;
        s)
            CONFIG_PATH="configs/flashdepth-s"
            CONFIG_NAME="config"
            ;;
        hybrid)
            CONFIG_PATH="configs/flashdepth"
            CONFIG_NAME="config"
            ;;
    esac
    MODEL_NAME="Original FlashDepth-${CONFIG_VARIANT^^}"
    DEPTH_TYPE="relative"
else
    # Gear5 uses gear5 configs
    CONFIG_PATH="configs/gear5"
    CONFIG_NAME="config_$CONFIG_VARIANT"
    MODEL_NAME="Gear5-${CONFIG_VARIANT^^}"
    DEPTH_TYPE="metric"
fi

# Compute resolution string based on video source
if [ "$VIDEO_SOURCE" = "avante" ]; then
    if [ "$RESOLUTION" = "2k" ]; then
        RES_STR="1596x1092"
    else
        RES_STR="756x518"
    fi
else
    # nusc (default)
    if [ "$RESOLUTION" = "2k" ]; then
        RES_STR="1596x896"
    else
        RES_STR="924x518"
    fi
fi

echo "Instance Segmentation + Depth Testing"
echo "============================================"
echo "Configuration:"
echo "  - Model: $MODEL_NAME"
echo "  - Depth type: $DEPTH_TYPE"
echo "  - Video source: $VIDEO_SOURCE"
echo "  - Resolution: $RESOLUTION ($RES_STR)"
echo "  - Video path: $VIDEO_PATH"
echo "  - Results dir: $RESULTS_DIR"
echo "  - GPU: $GPU_ID"
echo "  - Config: $CONFIG_PATH/$CONFIG_NAME"
echo "  - Segmentation model: $SEG_MODEL"
echo "  - Tracker: $TRACKER"
if [ "$ORIGINAL_FLASHDEPTH" != "true" ]; then
    echo "  - TSP temporal backend: $([ "$MAMBA" = "true" ] && echo "Mamba2" || echo "GRU")"
    echo "  - CLS layers: $CLS_LAYERS"
    echo "  - Focal length (fx): $FX pixels"
    echo "  - Canonical fx: $CANONICAL_FX pixels"
    echo "  - De-canon ratio: fx_ratio / resize_ratio (computed at runtime)"
fi
if [ -n "$CHECKPOINT" ]; then
    echo "  - Checkpoint: $CHECKPOINT"
fi
echo ""

# Build command
TEST_CMD="python test_instance_depth.py \
    --config-path $CONFIG_PATH \
    --config-name $CONFIG_NAME \
    +video_path=$VIDEO_PATH \
    +results_dir=$RESULTS_DIR \
    +frame_interval=$FRAME_INTERVAL \
    +seg_model=$SEG_MODEL \
    +tracker=$TRACKER \
    +use_original_flashdepth=$ORIGINAL_FLASHDEPTH \
    +resolution=$RESOLUTION \
    +video_source=$VIDEO_SOURCE \
    +fx=$FX \
    +canonical_fx=$CANONICAL_FX"

# Add Gear5-specific options (TSP temporal backend, CLS layers)
if [ "$ORIGINAL_FLASHDEPTH" != "true" ]; then
    TEST_CMD="$TEST_CMD model.use_mamba_temporal=$MAMBA +cls_layers='[$CLS_LAYERS]'"
fi

# Add person-only flag
if [ -n "$PERSON_ONLY" ]; then
    TEST_CMD="$TEST_CMD +person_only=true"
else
    TEST_CMD="$TEST_CMD +person_only=false"
fi

# Add center-mask flag
if [ -n "$CENTER_MASK" ]; then
    TEST_CMD="$TEST_CMD +center_mask=true"
else
    TEST_CMD="$TEST_CMD +center_mask=false"
fi

# Add checkpoint if specified
if [ -n "$CHECKPOINT" ]; then
    TEST_CMD="$TEST_CMD load=$CHECKPOINT"
fi

# Add sparse GT directory for Original FlashDepth alignment
if [ -n "$SPARSE_GT_DIR" ]; then
    TEST_CMD="$TEST_CMD +sparse_gt_dir=$SPARSE_GT_DIR"
    echo "  - Sparse GT dir: $SPARSE_GT_DIR"
fi

# Add show_depth_values flag
TEST_CMD="$TEST_CMD +show_depth_values=$SHOW_DEPTH_VALUES"

# Run with flashdepth Docker
echo "Running test..."
CUDA_VISIBLE_DEVICES=$GPU_ID docker compose run --rm flashdepth $TEST_CMD

echo ""
echo "Test completed! Results saved to: $RESULTS_DIR"
