#!/bin/bash

# FlashDepth Docker Runner Script
# This script helps you run the FlashDepth training in Docker environment

set -e

echo "FlashDepth Docker Training Runner"
echo "================================="

# Check if docker compose is available
if ! docker compose version &> /dev/null; then
    echo "Error: docker compose not found. Please install Docker Compose."
    exit 1
fi

# Check if NVIDIA Docker runtime is available
if ! docker info | grep -q nvidia; then
    echo "Warning: NVIDIA Docker runtime not detected. GPU may not be accessible."
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p train_results
mkdir -p checkpoints

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build       Build the Docker image"
    echo "  train       Start training with default settings"
    echo "  test        Start testing with default settings"
    echo "  train_gear2 Start Gear2 training (single GPU) - Ablation study"
    echo "  train_gear2_ddp Start Gear2 training with 2 GPUs (GPU 0,1) - Ablation study"
    echo "  test_gear2  Start Gear2 testing with default settings"
    echo "  train_gear3 Start Gear3 training (single GPU)"
    echo "  train_gear3_ddp Start Gear3 training with 2 GPUs (GPU 0,1)"
    echo "  test_gear3  Start Gear3 testing with default settings"
    echo "  train_gear4 Start Gear4 training (single GPU) - Enhanced FG/BG separation"
    echo "  train_gear4_ddp Start Gear4 training with 2 GPUs (GPU 0,1)"
    echo "  test_gear4  Start Gear4 testing"
    echo "  train_gear5     Start Gear5 training - Two-stage Global + FG modulation"
    echo "  train_gear5_ddp Start Gear5 training with 2 GPUs (GPU 0,1) - Two-stage Global + FG modulation"
    echo "  test_gear5      Start Gear5 testing"
    echo "  train_gear5_film     Start Gear5 FiLM training - Channel-wise FiLM modulation before Mamba"
    echo "  train_gear5_film_ddp Start Gear5 FiLM training with 2 GPUs (GPU 0,1)"
    echo "  test_gear5_film      Start Gear5 FiLM testing"
    echo "  test_gear2_objwise  Start Gear2 object-wise evaluation (Waymo segmentation)"
    echo "  test_gear3_objwise  Start Gear3 object-wise evaluation (Waymo segmentation)"
    echo "  test_gear4_objwise  Start Gear4 object-wise evaluation"
    echo "  test_gear5_objwise  Start Gear5 object-wise evaluation"
    echo "  test_original_flashdepth  Test original FlashDepth (without Gear modules) for comparison"
    echo "  shell       Start interactive shell in container"
    echo "  clean       Remove containers and images"
    echo "  logs        Show container logs"
    echo ""
    echo "Training Options:"
    echo "  --batch-size SIZE     Set batch size per GPU (default: 20, effective 40 with DDP)"
    echo "  --workers NUM         Set number of DataLoader workers (default: 8)"
    echo "  --epochs NUM          Set number of training iterations (default: 60001)"
    echo "  --gpu ID              Set GPU ID (default: 0)"
    echo "  --results-dir PATH    Set results directory (default: train_results/results_1)"
    echo "  --flashdepth-checkpoint PATH  Set FlashDepth pretrained weights path"
    echo "  --gear-checkpoint PATH  Set Gear Phase 1 checkpoint path (required for --config-variant hybrid)"
    echo "  --frame-interval NUM  Set frame interval for sequence visualization (default: 1)"
    echo "  --vid-len NUM         Set video sequence length for testing (default: 50)"
    echo "  --single-sequence PATH Test on a single sequence directory (e.g., /path/to/dynamicreplica/seq)"
    echo "  --measure-fps BOOL    Enable/disable FPS measurement (default: true)"
    echo "  --config-variant VARIANT  Set Gear config variant: l, s, hybrid (default: l for Stage 1)"
    echo "  --nuscenes            Enable nuScenes fine-tuning mode (Stage 3)"
    echo "  --dataset DATASET     Set object-wise evaluation dataset: waymo (default: waymo)"
    echo "  --resolution MODE    Set resolution mode for testing: base (518x518), 2k (1918x1078) (default: base)"
    echo "  --config VARIANT     Set FlashDepth config: flashdepth, flashdepth-l, flashdepth-s (default: flashdepth-l)"
    echo "  --inverse BOOL       Inverse colormap for depth (original FlashDepth only, default: false)"
    echo "  --no-video           Skip video (GIF/MP4) generation for faster testing"
    echo "  --whole-seq-test BOOL    Use all sequences in dataset (true) or first 8 sequences only (false, default: false)"
    echo "  --canon BOOL         Use canonical focal length normalization (default: true)"
    echo "  --loss TYPE          Set loss type for Gear5: log_l1 (default), importance (importance-weighted)"
    echo "  --visualization BOOL Enable/disable visualizations (sequence.png, best_frame.png, etc.). Default: true"
    echo "  --wandb BOOL         Enable/disable WandB logging (default: true)"
    echo "  --wandb-name NAME    Set WandB experiment name (default: auto-generated)"
    echo "  --mamba              Use Mamba2 instead of GRU for Gear5 TemporalScalePredictor (default: false/GRU)"
    echo "  --seq N              Sequence selection (ignored by test_original_flashdepth, always uses first sequence)"
    echo "  --limit-scenes N     For NuScenes, limit the number of scenes to process (e.g., 50)"
    echo ""
    echo "Note: Regularization losses are deprecated. Importance map now uses raw DINOv2 attention (frozen)."
    echo "Note: test_original_flashdepth always tests only the first sequence for FPS measurement."
    echo ""
    echo "Examples:"
    echo "  $0 build                              # Build the image"
    echo "  $0 train                              # Start training with defaults"
    echo "  $0 train --batch-size 4 --gpu 1      # Train with custom settings"
    echo "  $0 test                               # Start testing with defaults"
    echo "  $0 test --vid-len 25 --frame-interval 2  # Test with custom video length and frame interval"
    echo "  $0 train_gear2                        # Start Gear2 training with defaults"
    echo "  $0 train_gear2_ddp                    # Start Gear2 DDP training (2 GPUs)"
    echo "  $0 test_gear2                         # Start Gear2 testing with defaults"
    echo "  $0 train_gear3                        # Start Gear3 training with defaults"
    echo "  $0 train_gear3 --batch-size 8 --gpu 1 # Train Gear3 with custom settings"
    echo "  $0 train_gear3_ddp                    # Start Gear3 DDP training (2 GPUs)"
    echo "  $0 test_gear3                         # Start Gear3 testing with defaults"
    echo "  $0 test_gear3 --vid-len 25 --gpu 1    # Test Gear3 with custom settings"
    echo "  $0 test_gear3 --single-sequence /data/datasets/dynamicreplica/train/0b10c6-3_obj_source_left  # Test single sequence"
    echo "  $0 train_gear4                # Start Gear4 training"
    echo "  $0 train_gear4_ddp            # DDP training with 2 GPUs"
    echo "  $0 test_gear4                 # Start Gear4 testing"
    echo "  $0 test_gear4 --vid-len 25 --frame-interval 5  # Custom video length and interval"
    echo "  $0 test_gear4 --dataset waymo --gpu 2  # Test on Waymo dataset"
    echo "  $0 test_gear4 --vid-len 200 --no-video --gpu 2  # Fast testing without video generation"
    echo "  $0 test_gear2_objwise --dataset waymo_seg --config-variant l --gpu 0  # Object-wise evaluation on Waymo"
    echo "  $0 test_gear3_objwise --dataset waymo_seg --config-variant l --gpu 1  # Object-wise evaluation on Waymo"
    echo "  $0 test_gear4_objwise --dataset waymo_seg --config-variant l --gpu 2  # Gear4 object-wise"
    echo "  $0 test_gear5_objwise --dataset waymo_seg --gpu 0  # Gear5 object-wise evaluation"
    echo "  $0 test_gear5_objwise --dataset urbansyn --gpu 1  # Gear5 object-wise evaluation on UrbanSyn"
    echo "  $0 train_gear5 --gpu 0  # Gear5 training with GRU (default)"
    echo "  $0 train_gear5 --mamba --gpu 0  # Gear5 training with Mamba2 for temporal modeling"
    echo "  $0 test_gear5 --gpu 0  # Test Gear5 with GRU"
    echo "  $0 test_gear5 --mamba --gpu 0  # Test Gear5 with Mamba2"
    echo "  $0 train_gear2_ddp --config-variant hybrid --gear-checkpoint train_results/gear2_s/best.pth  # Hybrid training with Gear-S weights"
    echo "  $0 test_original_flashdepth --gpu 0  # Test original FlashDepth (ViT-L)"
    echo "  DATASET=waymo $0 test_original_flashdepth --gpu 1  # Test on Waymo dataset"
    echo "  $0 test_original_flashdepth --config flashdepth-s --gpu 0  # Use ViT-S variant (smaller/faster)"
    echo "  $0 test_original_flashdepth --config flashdepth-s --no-video --gpu 0  # Skip MP4 and .npy saving (faster testing)"
    echo "  CHECKPOINT=/app/configs/flashdepth-s/iter_14001.pth $0 test_original_flashdepth --config flashdepth-s  # ViT-S with matching checkpoint"
    echo "  $0 train --results-dir train_results/results_2  # Custom results directory"
    echo "  $0 shell                              # Interactive development"
}

# Parse command line arguments - optimized for RTX A6000 (2x 48GB)
COMMAND=""
BATCH_SIZE=3
WORKERS=8      # Optimized for 96 CPU cores, prevents I/O bottleneck
TOTAL_ITERS=60001
GPU_ID=0
RESULTS_DIR="train_results/results_1"
FLASHDEPTH_CHECKPOINT="configs/flashdepth-l/iter_10001.pth"
GEAR_CHECKPOINT=""  # Gear-S Phase 1 checkpoint (required for hybrid only)
FRAME_INTERVAL=1
VID_LEN=50
SINGLE_SEQUENCE=""  # Path to single sequence directory (optional)
MEASURE_FPS="true"
CONFIG_VARIANT="l"  # Gear config variant: l (Stage 1 ViT-L), s (Stage 1 ViT-S), hybrid (Stage 2)
NUSCENES="false"  # nuScenes fine-tuning mode (Stage 3)
CONFIG="flashdepth-l"  # FlashDepth config variant (flashdepth, flashdepth-l, flashdepth-s)
INVERSE="false"  # Inverse colormap for depth visualization (original FlashDepth only)
OBJWISE_DATASET=""  # Dataset for evaluation (waymo) - empty means use config default
RESOLUTION="base"  # Resolution mode for testing (base, 2k) - default to base (518x518)
NO_VIDEO="false"  # Skip video (GIF/MP4) generation for faster testing
WHOLE_SEQ_TEST="false"  # Use all sequences in dataset (true) or first 8 sequences only (false, default)
USE_CANONICAL="true"  # Use canonical focal length normalization (default: true)
LOSS_TYPE="log_l1"  # Loss type for Gear5 training: log_l1 (default), importance (importance-weighted)
VISUALIZATION="true"  # Enable visualizations by default (sequence.png, best_frame.png, etc.)
WANDB="true"  # Enable WandB logging by default
WANDB_NAME=""  # WandB experiment name (empty = auto-generated)
MAMBA="false"  # Use Mamba2 for Gear5 TemporalScalePredictor (false=GRU, true=Mamba2)
SEQ=""  # Sequence selection for UnrealStereo4K (test_original_flashdepth)
LIMIT_SCENES=""  # Limit number of scenes for NuScenes dataset (optional, e.g., 50)

# Parse arguments
USER_BATCH_SIZE=""  # Track if user explicitly set batch size
while [[ $# -gt 0 ]]; do
    case $1 in
        build|train|test|train_gear2|train_gear2_ddp|test_gear2|train_gear3|train_gear3_ddp|test_gear3|train_gear4|train_gear4_ddp|test_gear4|train_gear5|train_gear5_ddp|test_gear5|train_gear5_film|train_gear5_film_ddp|test_gear5_film|test_gear2_objwise|test_gear3_objwise|test_gear4_objwise|test_gear5_objwise|test_original_flashdepth|shell|clean|logs)
            COMMAND="$1"
            shift
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            USER_BATCH_SIZE="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --epochs)
            TOTAL_ITERS="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --flashdepth-checkpoint)
            FLASHDEPTH_CHECKPOINT="$2"
            shift 2
            ;;
        --gear-checkpoint)
            GEAR_CHECKPOINT="$2"
            shift 2
            ;;
        --frame-interval)
            FRAME_INTERVAL="$2"
            shift 2
            ;;
        --vid-len)
            VID_LEN="$2"
            shift 2
            ;;
        --single-sequence)
            SINGLE_SEQUENCE="$2"
            shift 2
            ;;
        --measure-fps)
            MEASURE_FPS="$2"
            shift 2
            ;;
        --config-variant)
            CONFIG_VARIANT="$2"
            shift 2
            ;;
        --nuscenes)
            NUSCENES="true"
            shift
            ;;
        --dataset)
            OBJWISE_DATASET="$2"
            shift 2
            ;;
        --objwise)
            OBJWISE_FLAG="true"
            shift
            ;;
        --resolution)
            RESOLUTION="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --inverse)
            INVERSE="$2"
            shift 2
            ;;
        --no-video)
            NO_VIDEO="true"
            shift
            ;;
        --whole-seq-test)
            WHOLE_SEQ_TEST="$2"
            shift 2
            ;;
        --canon)
            USE_CANONICAL="$2"
            shift 2
            ;;
        --loss)
            LOSS_TYPE="$2"
            shift 2
            ;;
        --visualization)
            VISUALIZATION="$2"
            shift 2
            ;;
        --wandb)
            WANDB="$2"
            shift 2
            ;;
        --wandb-name)
            WANDB_NAME="$2"
            shift 2
            ;;
        --mamba)
            MAMBA="true"
            shift
            ;;
        --seq)
            SEQ="$2"
            shift 2
            ;;
        --limit-scenes)
            LIMIT_SCENES="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Set CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES=$GPU_ID

case $COMMAND in
    build)
        echo "Building FlashDepth Docker image..."
        docker compose build
        echo "Build completed!"
        ;;

    train)
        echo "Starting FlashDepth training..."
        echo "Configuration:"
        echo "  - Batch size: $BATCH_SIZE"
        echo "  - Workers: $WORKERS"
        echo "  - Total iterations: $TOTAL_ITERS"
        echo "  - GPU: $GPU_ID"
        echo "  - Results directory: $RESULTS_DIR"
        echo ""

        # Update docker compose command with custom parameters (Stage 1: FlashDepth-L)
        DOCKER_CMD="CUDA_VISIBLE_DEVICES=$GPU_ID docker compose run --rm flashdepth python train_metric_head.py \
            --config-path configs/flashdepth-l \
            dataset.data_root=/data/datasets \
            dataset.train_datasets=[tartanair] \
            dataset.val_datasets=[tartanair] \
            training.batch_size=12 \
            training.workers=4 \
            training.total_iters=$TOTAL_ITERS \
            model.attn_class=Attention \
            +results_dir=$RESULTS_DIR \
            +gpu=$GPU_ID"

        # Add checkpoint loading if specified
        if [ -n "$FLASHDEPTH_CHECKPOINT" ]; then
            DOCKER_CMD="$DOCKER_CMD load=$FLASHDEPTH_CHECKPOINT"
        fi

        eval $DOCKER_CMD
        ;;

    test)
        echo "Starting FlashDepth testing..."
        echo "Configuration:"
        echo "  - GPU: $GPU_ID"
        echo "  - Results directory: $RESULTS_DIR"
        echo "  - Video length: $VID_LEN"
        echo "  - Frame interval: $FRAME_INTERVAL"
        echo ""

        # Build test command with conditional parameters
        TEST_CMD="python test_metric_head.py +results_dir=$RESULTS_DIR +gpu=$GPU_ID"

        if [ -n "$FLASHDEPTH_CHECKPOINT" ]; then
            TEST_CMD="$TEST_CMD +flashdepth_checkpoint=$FLASHDEPTH_CHECKPOINT"
        fi

        # Add frame interval and video length options
        TEST_CMD="$TEST_CMD +frame_interval=$FRAME_INTERVAL +vid_len=$VID_LEN"

        # Run test_metric_head.py with custom parameters
        docker compose run --rm flashdepth $TEST_CMD
        ;;

    train_gear2)
        # Determine canonical focal length based on config variant
        if [ "$CONFIG_VARIANT" = "hybrid" ]; then
            CANONICAL_FX="500.0"
            RES_NAME="2k"
        else
            CANONICAL_FX="500.0"
            RES_NAME="base"
        fi

        echo "Starting Gear2 training (Single GPU) - Ablation Study..."
        echo "Configuration:"
        echo "  - Config variant: $CONFIG_VARIANT"
        echo "  - nuScenes mode: $NUSCENES"
        echo "  - Batch size: $BATCH_SIZE"
        echo "  - Workers: $WORKERS"
        echo "  - Total iterations: $TOTAL_ITERS"
        echo "  - GPU: $GPU_ID"
        echo "  - Results directory: $RESULTS_DIR"
        echo "  - Canonical focal length: $CANONICAL_FX ($RES_NAME resolution)"
        echo ""

        # Build train_gear2 command with config variant
        DOCKER_CMD="CUDA_VISIBLE_DEVICES=$GPU_ID docker compose run --rm flashdepth python train_gear2.py \
            --config-path configs/gear2 \
            --config-name config_$CONFIG_VARIANT \
            dataset.data_root=/data/datasets \
            training.batch_size=$BATCH_SIZE \
            training.workers=$WORKERS \
            training.iterations=$TOTAL_ITERS \
            training.wandb=$WANDB \
            use_canonical_space=$USE_CANONICAL \
            +results_dir=$RESULTS_DIR"

        # Add wandb name if specified
        if [ -n "$WANDB_NAME" ]; then
            DOCKER_CMD="$DOCKER_CMD training.wandb_name=$WANDB_NAME"
        fi

        # Add nuScenes flag if enabled
        if [ "$NUSCENES" = "true" ]; then
            DOCKER_CMD="$DOCKER_CMD +nuscenes=true"
        fi

        # Add checkpoint loading - Gear2 requires FlashDepth or Gear2 pretrained weights
        if [ -n "$FLASHDEPTH_CHECKPOINT" ]; then
            DOCKER_CMD="$DOCKER_CMD load=$FLASHDEPTH_CHECKPOINT"
        fi

        eval $DOCKER_CMD
        ;;

    train_gear2_ddp)
        # Auto-adjust batch size, workers, and iterations for Hybrid (2K resolution) if not explicitly set
        if [ "$CONFIG_VARIANT" = "hybrid" ]; then
            # Hybrid: 2K resolution - reduce both batch size and workers
            ACTUAL_WORKERS=2  # 2 workers per GPU (total 4 across 2 GPUs)
            CANONICAL_FX="500.0"
            RES_NAME="2k"

            # Auto-adjust batch size for 2K resolution if user didn't specify
            if [ -z "$USER_BATCH_SIZE" ]; then
                BATCH_SIZE=1  # Minimal batch size for 2K resolution to avoid OOM
                echo "  NOTE: Auto-adjusted batch size to $BATCH_SIZE for Hybrid (2K resolution)"
            else
                echo "  WARNING: Using user-specified batch size $BATCH_SIZE - may cause OOM on 2K resolution!"
            fi

            # Reduce iterations for Hybrid (same as Stage 1)
            TOTAL_ITERS=60001
            echo "  NOTE: Iterations set to $TOTAL_ITERS for Hybrid training"
        else
            # Stage 1 (L/S): 518x518 resolution - can use more workers and larger batch
            ACTUAL_WORKERS=$WORKERS
            CANONICAL_FX="500.0"
            RES_NAME="base"
        fi

        echo "Starting Gear2 training (Multi-GPU: 0,1) - Ablation Study..."
        echo "Configuration:"
        echo "  - Config variant: $CONFIG_VARIANT"
        echo "  - nuScenes mode: $NUSCENES"
        echo "  - Batch size per GPU: $BATCH_SIZE"
        echo "  - Effective batch size: $((BATCH_SIZE * 2))"
        echo "  - Workers per GPU: $ACTUAL_WORKERS"
        echo "  - Total iterations: $TOTAL_ITERS"
        echo "  - GPUs: 0,1"
        echo "  - Results directory: $RESULTS_DIR"
        echo "  - FPS measurement: $MEASURE_FPS"
        echo "  - Canonical focal length: $CANONICAL_FX ($RES_NAME resolution)"
        echo ""

        DOCKER_CMD="CUDA_VISIBLE_DEVICES=0,1 docker compose run --rm \
            -e GLOO_SOCKET_IFNAME=eth0 \
            -e NCCL_SOCKET_IFNAME=eth0 \
            -e NCCL_P2P_DISABLE=1 \
            -e WANDB_API_KEY=${WANDB_API_KEY:-} \
            flashdepth torchrun \
            --standalone \
            --nproc_per_node=2 \
            train_gear2.py \
            --config-path configs/gear2 \
            --config-name config_$CONFIG_VARIANT \
            dataset.data_root=/data/datasets \
            training.batch_size=$BATCH_SIZE \
            training.workers=$ACTUAL_WORKERS \
            training.iterations=$TOTAL_ITERS \
            training.measure_fps=$MEASURE_FPS \
            training.wandb=$WANDB \
            use_canonical_space=$USE_CANONICAL \
            +results_dir=$RESULTS_DIR"

        # Add wandb name if specified
        if [ -n "$WANDB_NAME" ]; then
            DOCKER_CMD="$DOCKER_CMD training.wandb_name=$WANDB_NAME"
        fi

        # Add nuScenes flag if enabled
        if [ "$NUSCENES" = "true" ]; then
            DOCKER_CMD="$DOCKER_CMD +nuscenes=true"
        fi

        # Add checkpoint loading
        if [ -n "$FLASHDEPTH_CHECKPOINT" ]; then
            DOCKER_CMD="$DOCKER_CMD load=$FLASHDEPTH_CHECKPOINT"
        fi

        # Add gear checkpoint for hybrid
        if [ -n "$GEAR_CHECKPOINT" ]; then
            DOCKER_CMD="$DOCKER_CMD gear_checkpoint=$GEAR_CHECKPOINT"
        fi

        eval $DOCKER_CMD
        ;;

    train_gear3)
        # Determine canonical focal length based on config variant
        if [ "$CONFIG_VARIANT" = "hybrid" ]; then
            CANONICAL_FX="500.0"
            RES_NAME="2k"
        else
            CANONICAL_FX="500.0"
            RES_NAME="base"
        fi

        echo "Starting Gear3 training (Single GPU)..."
        echo "Configuration:"
        echo "  - Config variant: $CONFIG_VARIANT"
        echo "  - nuScenes mode: $NUSCENES"
        echo "  - Batch size: $BATCH_SIZE"
        echo "  - Workers: $WORKERS"
        echo "  - Total iterations: $TOTAL_ITERS"
        echo "  - GPU: $GPU_ID"
        echo "  - Results directory: $RESULTS_DIR"
        echo "  - Canonical focal length: $CANONICAL_FX ($RES_NAME resolution)"
        echo ""

        # Build train_gear3 command with config variant
        DOCKER_CMD="CUDA_VISIBLE_DEVICES=$GPU_ID docker compose run --rm flashdepth python train_gear3.py \
            --config-path configs/gear3 \
            --config-name config_$CONFIG_VARIANT \
            dataset.data_root=/data/datasets \
            training.batch_size=$BATCH_SIZE \
            training.workers=$WORKERS \
            training.iterations=$TOTAL_ITERS \
            training.wandb=$WANDB \
            use_canonical_space=$USE_CANONICAL \
            +results_dir=$RESULTS_DIR"

        # Add wandb name if specified
        if [ -n "$WANDB_NAME" ]; then
            DOCKER_CMD="$DOCKER_CMD training.wandb_name=$WANDB_NAME"
        fi

        # Add nuScenes flag if enabled
        if [ "$NUSCENES" = "true" ]; then
            DOCKER_CMD="$DOCKER_CMD +nuscenes=true"
        fi

        # Add checkpoint loading - Gear3 requires FlashDepth or Gear3 pretrained weights
        if [ -n "$FLASHDEPTH_CHECKPOINT" ]; then
            DOCKER_CMD="$DOCKER_CMD load=$FLASHDEPTH_CHECKPOINT"
        fi

        eval $DOCKER_CMD
        ;;

    train_gear3_ddp)
        # Auto-adjust batch size, workers, and iterations for Hybrid (2K resolution) if not explicitly set
        if [ "$CONFIG_VARIANT" = "hybrid" ]; then
            # Hybrid: 2K resolution - reduce both batch size and workers
            ACTUAL_WORKERS=2  # 2 workers per GPU (total 4 across 2 GPUs)
            CANONICAL_FX="500.0"
            RES_NAME="2k"

            # Auto-adjust batch size for 2K resolution if user didn't specify
            if [ -z "$USER_BATCH_SIZE" ]; then
                BATCH_SIZE=1  # Minimal batch size for 2K resolution to avoid OOM
                echo "  NOTE: Auto-adjusted batch size to $BATCH_SIZE for Hybrid (2K resolution)"
            else
                echo "  WARNING: Using user-specified batch size $BATCH_SIZE - may cause OOM on 2K resolution!"
            fi

            # Iterations for Hybrid (same as Stage 1)
            TOTAL_ITERS=60001
            echo "  NOTE: Iterations set to $TOTAL_ITERS for Hybrid training"
        else
            # Stage 1 (L/S): 518x518 resolution - can use more workers and larger batch
            ACTUAL_WORKERS=$WORKERS
            CANONICAL_FX="500.0"
            RES_NAME="base"
        fi

        echo "Starting Gear3 training (Multi-GPU: 0,1)..."
        echo "Configuration:"
        echo "  - Config variant: $CONFIG_VARIANT"
        echo "  - nuScenes mode: $NUSCENES"
        echo "  - Batch size per GPU: $BATCH_SIZE"
        echo "  - Effective batch size: $((BATCH_SIZE * 2))"
        echo "  - Workers per GPU: $ACTUAL_WORKERS"
        echo "  - Total iterations: $TOTAL_ITERS"
        echo "  - GPUs: 0,1"
        echo "  - Results directory: $RESULTS_DIR"
        echo "  - FPS measurement: $MEASURE_FPS"
        echo "  - Canonical focal length: $CANONICAL_FX ($RES_NAME resolution)"
        echo ""

        DOCKER_CMD="CUDA_VISIBLE_DEVICES=0,1 docker compose run --rm \
            -e GLOO_SOCKET_IFNAME=eth0 \
            -e NCCL_SOCKET_IFNAME=eth0 \
            -e NCCL_P2P_DISABLE=1 \
            -e WANDB_API_KEY=${WANDB_API_KEY:-} \
            flashdepth torchrun \
            --standalone \
            --nproc_per_node=2 \
            train_gear3.py \
            --config-path configs/gear3 \
            --config-name config_$CONFIG_VARIANT \
            dataset.data_root=/data/datasets \
            training.batch_size=$BATCH_SIZE \
            training.workers=$ACTUAL_WORKERS \
            training.iterations=$TOTAL_ITERS \
            training.measure_fps=$MEASURE_FPS \
            training.wandb=$WANDB \
            use_canonical_space=$USE_CANONICAL \
            +results_dir=$RESULTS_DIR"

        # Add wandb name if specified
        if [ -n "$WANDB_NAME" ]; then
            DOCKER_CMD="$DOCKER_CMD training.wandb_name=$WANDB_NAME"
        fi

        # Add nuScenes flag if enabled
        if [ "$NUSCENES" = "true" ]; then
            DOCKER_CMD="$DOCKER_CMD +nuscenes=true"
        fi

        # Add checkpoint loading
        if [ -n "$FLASHDEPTH_CHECKPOINT" ]; then
            DOCKER_CMD="$DOCKER_CMD load=$FLASHDEPTH_CHECKPOINT"
        fi

        # Add gear checkpoint for hybrid
        if [ -n "$GEAR_CHECKPOINT" ]; then
            DOCKER_CMD="$DOCKER_CMD gear_checkpoint=$GEAR_CHECKPOINT"
        fi

        eval $DOCKER_CMD
        ;;

    train_gear4)
        # Auto-select checkpoint based on config variant if user didn't specify
        if [ "$FLASHDEPTH_CHECKPOINT" = "configs/flashdepth-l/iter_10001.pth" ]; then
            # User didn't specify checkpoint, auto-select based on variant
            case "$CONFIG_VARIANT" in
                s)
                    FLASHDEPTH_CHECKPOINT="configs/flashdepth-s/iter_14001.pth"
                    ;;
                l)
                    FLASHDEPTH_CHECKPOINT="configs/flashdepth-l/iter_10001.pth"
                    ;;
                hybrid)
                    FLASHDEPTH_CHECKPOINT="configs/flashdepth/iter_43002.pth"
                    ;;
                *)
                    echo "Unknown config variant: $CONFIG_VARIANT"
                    echo "Valid options: s, l, hybrid"
                    exit 1
                    ;;
            esac
        fi

        # Determine canonical focal length based on config variant
        if [ "$CONFIG_VARIANT" = "hybrid" ]; then
            CANONICAL_FX="500.0"
            RES_NAME="2k"
        else
            CANONICAL_FX="500.0"
            RES_NAME="base"
        fi

        echo "Starting Gear4 training (Single GPU) - Enhanced FG/BG Separation..."
        echo "Configuration:"
        echo "  - Config variant: $CONFIG_VARIANT"
        echo "  - nuScenes mode: $NUSCENES"
        echo "  - Batch size: $BATCH_SIZE"
        echo "  - Workers: $WORKERS"
        echo "  - Total iterations: $TOTAL_ITERS"
        echo "  - GPU: $GPU_ID"
        echo "  - Results directory: $RESULTS_DIR"
        echo "  - Checkpoint: $FLASHDEPTH_CHECKPOINT"
        echo "  - Canonical focal length: $CANONICAL_FX ($RES_NAME resolution)"
        echo ""

        # Build train_gear4 command with config variant
        DOCKER_CMD="CUDA_VISIBLE_DEVICES=$GPU_ID docker compose run --rm flashdepth python train_gear4.py \
            --config-path configs/gear4 \
            --config-name config_$CONFIG_VARIANT \
            dataset.data_root=/data/datasets \
            training.batch_size=$BATCH_SIZE \
            training.workers=$WORKERS \
            training.iterations=$TOTAL_ITERS \
            training.wandb=$WANDB \
            use_canonical_space=$USE_CANONICAL \
            +results_dir=$RESULTS_DIR"

        # Add wandb name if specified
        if [ -n "$WANDB_NAME" ]; then
            DOCKER_CMD="$DOCKER_CMD training.wandb_name=$WANDB_NAME"
        fi

        # Add nuScenes flag if enabled
        if [ "$NUSCENES" = "true" ]; then
            DOCKER_CMD="$DOCKER_CMD +nuscenes=true"
        fi

        # Add checkpoint loading - Gear4 requires FlashDepth or Gear4 pretrained weights
        if [ -n "$FLASHDEPTH_CHECKPOINT" ]; then
            DOCKER_CMD="$DOCKER_CMD load=$FLASHDEPTH_CHECKPOINT"
        fi

        eval $DOCKER_CMD
        ;;

    train_gear4_ddp)
        # Auto-select checkpoint based on config variant if user didn't specify
        if [ "$FLASHDEPTH_CHECKPOINT" = "configs/flashdepth-l/iter_10001.pth" ]; then
            # User didn't specify checkpoint, auto-select based on variant
            case "$CONFIG_VARIANT" in
                s)
                    FLASHDEPTH_CHECKPOINT="configs/flashdepth-s/iter_14001.pth"
                    ;;
                l)
                    FLASHDEPTH_CHECKPOINT="configs/flashdepth-l/iter_10001.pth"
                    ;;
                hybrid)
                    FLASHDEPTH_CHECKPOINT="configs/flashdepth/iter_43002.pth"
                    ;;
                *)
                    echo "Unknown config variant: $CONFIG_VARIANT"
                    echo "Valid options: s, l, hybrid"
                    exit 1
                    ;;
            esac
        fi

        # Auto-adjust batch size, workers, and iterations for Hybrid (2K resolution) if not explicitly set
        if [ "$CONFIG_VARIANT" = "hybrid" ]; then
            # Hybrid: 2K resolution - reduce both batch size and workers
            ACTUAL_WORKERS=2  # 2 workers per GPU (total 4 across 2 GPUs)
            CANONICAL_FX="500.0"
            RES_NAME="2k"

            # Auto-adjust batch size for 2K resolution if user didn't specify
            if [ -z "$USER_BATCH_SIZE" ]; then
                BATCH_SIZE=1  # Minimal batch size for 2K resolution to avoid OOM
                echo "  NOTE: Auto-adjusted batch size to $BATCH_SIZE for Hybrid (2K resolution)"
            else
                echo "  WARNING: Using user-specified batch size $BATCH_SIZE - may cause OOM on 2K resolution!"
            fi

            # Iterations for Hybrid (same as Stage 1)
            TOTAL_ITERS=60001
            echo "  NOTE: Iterations set to $TOTAL_ITERS for Hybrid training"
        else
            # Stage 1 (L/S): 518x518 resolution - can use more workers and larger batch
            ACTUAL_WORKERS=$WORKERS
            CANONICAL_FX="500.0"
            RES_NAME="base"
        fi

        echo "Starting Gear4 training (Multi-GPU: 0,1) - Enhanced FG/BG Separation..."
        echo "Configuration:"
        echo "  - Config variant: $CONFIG_VARIANT"
        echo "  - nuScenes mode: $NUSCENES"
        echo "  - Batch size per GPU: $BATCH_SIZE"
        echo "  - Effective batch size: $((BATCH_SIZE * 2))"
        echo "  - Workers per GPU: $ACTUAL_WORKERS"
        echo "  - Total iterations: $TOTAL_ITERS"
        echo "  - GPUs: 0,1"
        echo "  - Results directory: $RESULTS_DIR"
        echo "  - FPS measurement: $MEASURE_FPS"
        echo "  - Checkpoint: $FLASHDEPTH_CHECKPOINT"
        echo "  - Canonical focal length: $CANONICAL_FX ($RES_NAME resolution)"
        echo ""

        DOCKER_CMD="CUDA_VISIBLE_DEVICES=0,1 docker compose run --rm \
            -e GLOO_SOCKET_IFNAME=eth0 \
            -e NCCL_SOCKET_IFNAME=eth0 \
            -e NCCL_P2P_DISABLE=1 \
            -e WANDB_API_KEY=${WANDB_API_KEY:-} \
            flashdepth torchrun \
            --standalone \
            --nproc_per_node=2 \
            train_gear4.py \
            --config-path configs/gear4 \
            --config-name config_$CONFIG_VARIANT \
            dataset.data_root=/data/datasets \
            training.batch_size=$BATCH_SIZE \
            training.workers=$ACTUAL_WORKERS \
            training.iterations=$TOTAL_ITERS \
            training.measure_fps=$MEASURE_FPS \
            training.wandb=$WANDB \
            use_canonical_space=$USE_CANONICAL \
            +results_dir=$RESULTS_DIR"

        # Add wandb name if specified
        if [ -n "$WANDB_NAME" ]; then
            DOCKER_CMD="$DOCKER_CMD training.wandb_name=$WANDB_NAME"
        fi

        # Add nuScenes flag if enabled
        if [ "$NUSCENES" = "true" ]; then
            DOCKER_CMD="$DOCKER_CMD +nuscenes=true"
        fi

        # Add checkpoint loading
        if [ -n "$FLASHDEPTH_CHECKPOINT" ]; then
            DOCKER_CMD="$DOCKER_CMD load=$FLASHDEPTH_CHECKPOINT"
        fi

        # Add gear checkpoint for hybrid
        if [ -n "$GEAR_CHECKPOINT" ]; then
            DOCKER_CMD="$DOCKER_CMD gear_checkpoint=$GEAR_CHECKPOINT"
        fi

        eval $DOCKER_CMD
        ;;

    test_gear2)
        # Set default checkpoint for gear2 if not explicitly provided
        if [ "$FLASHDEPTH_CHECKPOINT" == "configs/flashdepth-l/iter_10001.pth" ]; then
            FLASHDEPTH_CHECKPOINT="train_results/results_14/gear_2/phase_1/best.pth"
        fi

        echo "Starting Gear2 testing - Ablation Study..."
        echo "Configuration:"
        echo "  - Video length: $VID_LEN"
        echo "  - Frame interval: $FRAME_INTERVAL"
        echo "  - GPU: $GPU_ID"
        echo "  - Results directory: $RESULTS_DIR"
        echo "  - Checkpoint: $FLASHDEPTH_CHECKPOINT"
        echo "  - Config variant: $CONFIG_VARIANT"
        echo "  - Workers: $WORKERS"
        if [ "$OBJWISE_FLAG" == "true" ]; then
            echo "  - Object-wise evaluation: ENABLED"
        fi
        if [ -n "$OBJWISE_DATASET" ]; then
            echo "  - Dataset: $OBJWISE_DATASET"
        fi
        if [ -n "$SINGLE_SEQUENCE" ]; then
            echo "  - Single sequence: $SINGLE_SEQUENCE"
        fi
        echo ""

        # Build test_gear2 command with config variant
        TEST_CMD="python test_gear2.py --config-path configs/gear2 --config-name config_$CONFIG_VARIANT dataset.data_root=/data/datasets training.workers=$WORKERS +results_dir=$RESULTS_DIR +gpu=$GPU_ID"

        # Add --objwise flag if requested
        if [ "$OBJWISE_FLAG" == "true" ]; then
            TEST_CMD="$TEST_CMD --objwise"
        fi

        # Override dataset if specified
        if [ -n "$OBJWISE_DATASET" ]; then
            TEST_CMD="$TEST_CMD eval.test_datasets=[$OBJWISE_DATASET]"
            # Remove _seg suffix for object_wise.dataset config
            OBJWISE_DATASET_BASE="${OBJWISE_DATASET/_seg/}"
            TEST_CMD="$TEST_CMD object_wise.dataset=$OBJWISE_DATASET_BASE"
        fi

        # Add resolution override
        TEST_CMD="$TEST_CMD +resolution=$RESOLUTION"

        # Use --checkpoint option for the unified checkpoint
        if [ -n "$FLASHDEPTH_CHECKPOINT" ]; then
            TEST_CMD="$TEST_CMD load=$FLASHDEPTH_CHECKPOINT"
        fi

        # Add frame interval and video length options
        TEST_CMD="$TEST_CMD +frame_interval=$FRAME_INTERVAL +vid_len=$VID_LEN +whole_seq_test=$WHOLE_SEQ_TEST"

        # Add visualization flag
        TEST_CMD="$TEST_CMD +visualization=$VISUALIZATION"

        # Add single sequence path if specified
        if [ -n "$SINGLE_SEQUENCE" ]; then
            TEST_CMD="$TEST_CMD +single_sequence=$SINGLE_SEQUENCE"
        fi

        # Run test_gear2.py with custom parameters (with GPU selection)
        CUDA_VISIBLE_DEVICES=$GPU_ID docker compose run --rm flashdepth $TEST_CMD
        ;;

    test_gear3)
        # Set default checkpoint for gear3 if not explicitly provided
        if [ "$FLASHDEPTH_CHECKPOINT" == "configs/flashdepth-l/iter_10001.pth" ]; then
            FLASHDEPTH_CHECKPOINT="train_results/results_14/gear_3/phase_1/best.pth"
        fi

        echo "Starting Gear3 testing..."
        echo "Configuration:"
        echo "  - Video length: $VID_LEN"
        echo "  - Frame interval: $FRAME_INTERVAL"
        echo "  - GPU: $GPU_ID"
        echo "  - Results directory: $RESULTS_DIR"
        echo "  - Checkpoint: $FLASHDEPTH_CHECKPOINT"
        echo "  - Config variant: $CONFIG_VARIANT"
        echo "  - Workers: $WORKERS"
        if [ "$OBJWISE_FLAG" == "true" ]; then
            echo "  - Object-wise evaluation: ENABLED"
        fi
        if [ -n "$OBJWISE_DATASET" ]; then
            echo "  - Dataset: $OBJWISE_DATASET"
        fi
        if [ -n "$SINGLE_SEQUENCE" ]; then
            echo "  - Single sequence: $SINGLE_SEQUENCE"
        fi
        echo ""

        # Build test_gear3 command with config variant
        TEST_CMD="python test_gear3.py --config-path configs/gear3 --config-name config_$CONFIG_VARIANT dataset.data_root=/data/datasets training.workers=$WORKERS +results_dir=$RESULTS_DIR +gpu=$GPU_ID"

        # Add --objwise flag if requested
        if [ "$OBJWISE_FLAG" == "true" ]; then
            TEST_CMD="$TEST_CMD --objwise"
        fi

        # Override dataset if specified
        if [ -n "$OBJWISE_DATASET" ]; then
            TEST_CMD="$TEST_CMD eval.test_datasets=[$OBJWISE_DATASET]"
            # Remove _seg suffix for object_wise.dataset config
            OBJWISE_DATASET_BASE="${OBJWISE_DATASET/_seg/}"
            TEST_CMD="$TEST_CMD object_wise.dataset=$OBJWISE_DATASET_BASE"
        fi

        # Add resolution override
        TEST_CMD="$TEST_CMD +resolution=$RESOLUTION"

        # Use --checkpoint option for the unified checkpoint
        if [ -n "$FLASHDEPTH_CHECKPOINT" ]; then
            TEST_CMD="$TEST_CMD load=$FLASHDEPTH_CHECKPOINT"
        fi

        # Add frame interval and video length options
        TEST_CMD="$TEST_CMD +frame_interval=$FRAME_INTERVAL +vid_len=$VID_LEN +whole_seq_test=$WHOLE_SEQ_TEST"

        # Add visualization flag
        TEST_CMD="$TEST_CMD +visualization=$VISUALIZATION"

        # Add single sequence path if specified
        if [ -n "$SINGLE_SEQUENCE" ]; then
            TEST_CMD="$TEST_CMD +single_sequence=$SINGLE_SEQUENCE"
        fi

        # Run test_gear3.py with custom parameters (with GPU selection)
        CUDA_VISIBLE_DEVICES=$GPU_ID docker compose run --rm flashdepth $TEST_CMD
        ;;

    test_gear4)
        # Set default checkpoint for gear4 if not explicitly provided
        if [ "$FLASHDEPTH_CHECKPOINT" == "configs/flashdepth-l/iter_10001.pth" ]; then
            FLASHDEPTH_CHECKPOINT="train_results/results_14/gear_4/phase_1/best.pth"
        fi

        echo "Starting Gear4 testing - Enhanced FG/BG Separation..."
        echo "Configuration:"
        echo "  - Video length: $VID_LEN"
        echo "  - Frame interval: $FRAME_INTERVAL"
        echo "  - GPU: $GPU_ID"
        echo "  - Results directory: $RESULTS_DIR"
        echo "  - Checkpoint: $FLASHDEPTH_CHECKPOINT"
        echo "  - Config variant: $CONFIG_VARIANT"
        echo "  - Workers: $WORKERS"
        if [ "$OBJWISE_FLAG" == "true" ]; then
            echo "  - Object-wise evaluation: ENABLED"
        fi
        if [ -n "$OBJWISE_DATASET" ]; then
            echo "  - Dataset: $OBJWISE_DATASET"
        fi
        if [ -n "$SINGLE_SEQUENCE" ]; then
            echo "  - Single sequence: $SINGLE_SEQUENCE"
        fi
        echo ""

        # Build test_gear4 command with config variant
        TEST_CMD="python test_gear4.py --config-path configs/gear4 --config-name config_$CONFIG_VARIANT dataset.data_root=/data/datasets training.workers=$WORKERS +results_dir=$RESULTS_DIR +gpu=$GPU_ID"

        # Add --objwise flag if requested
        if [ "$OBJWISE_FLAG" == "true" ]; then
            TEST_CMD="$TEST_CMD --objwise"
        fi

        # Override dataset if specified
        if [ -n "$OBJWISE_DATASET" ]; then
            TEST_CMD="$TEST_CMD eval.test_datasets=[$OBJWISE_DATASET]"
            # Remove _seg suffix for object_wise.dataset config
            OBJWISE_DATASET_BASE="${OBJWISE_DATASET/_seg/}"
            TEST_CMD="$TEST_CMD object_wise.dataset=$OBJWISE_DATASET_BASE"
        fi

        # Add limit_scenes for NuScenes if specified
        if [ -n "$LIMIT_SCENES" ]; then
            TEST_CMD="$TEST_CMD +dataset.limit_scenes=$LIMIT_SCENES"
        fi

        # Add resolution override
        TEST_CMD="$TEST_CMD +resolution=$RESOLUTION"

        # Use --checkpoint option for the unified checkpoint
        if [ -n "$FLASHDEPTH_CHECKPOINT" ]; then
            TEST_CMD="$TEST_CMD load=$FLASHDEPTH_CHECKPOINT"
        fi

        # Add frame interval and video length options
        TEST_CMD="$TEST_CMD +frame_interval=$FRAME_INTERVAL +vid_len=$VID_LEN +whole_seq_test=$WHOLE_SEQ_TEST"

        # Add visualization flag
        TEST_CMD="$TEST_CMD +visualization=$VISUALIZATION"

        # Add video generation control
        if [ "$NO_VIDEO" == "true" ]; then
            TEST_CMD="$TEST_CMD eval.out_video=false"
        fi

        # Add single sequence path if specified
        if [ -n "$SINGLE_SEQUENCE" ]; then
            TEST_CMD="$TEST_CMD +single_sequence=$SINGLE_SEQUENCE"
        fi

        # Run test_gear4.py with custom parameters (with GPU selection)
        CUDA_VISIBLE_DEVICES=$GPU_ID docker compose run --rm flashdepth $TEST_CMD
        ;;

    shell)
        echo "Starting interactive shell..."
        # Check if container is already running
        RUNNING_CONTAINER=$(docker ps --filter "name=flashdepth" --format "{{.Names}}" | head -1)
        if [ -n "$RUNNING_CONTAINER" ]; then
            echo "Connecting to running container: $RUNNING_CONTAINER"
            docker exec -it $RUNNING_CONTAINER /bin/bash
        else
            echo "No running container found. Starting new container..."
            docker compose run --rm flashdepth /bin/bash
        fi
        ;;

    test_gear2_objwise)
        # DEPRECATED: Use test_gear2 --objwise instead
        echo "⚠️  WARNING: 'test_gear2_objwise' is deprecated!"
        echo "   Please use: ./run_docker.sh test_gear2 --objwise --dataset $OBJWISE_DATASET"
        echo ""
        echo "Redirecting to new command format..."
        echo ""

        # Redirect to new format
        OBJWISE_FLAG="true"
        OBJWISE_DATASET=${OBJWISE_DATASET:-waymo_seg}

        # Call test_gear2 with objwise flag
        exec "$0" test_gear2 --objwise --dataset "$OBJWISE_DATASET" "${@:2}"
        ;;

    test_gear3_objwise)
        # DEPRECATED: Use test_gear3 --objwise instead
        echo "⚠️  WARNING: 'test_gear3_objwise' is deprecated!"
        echo "   Please use: ./run_docker.sh test_gear3 --objwise --dataset $OBJWISE_DATASET"
        echo ""
        echo "Redirecting to new command format..."
        echo ""

        # Redirect to new format
        OBJWISE_FLAG="true"
        OBJWISE_DATASET=${OBJWISE_DATASET:-waymo_seg}

        # Call test_gear3 with objwise flag
        exec "$0" test_gear3 --objwise --dataset "$OBJWISE_DATASET" "${@:2}"
        ;;

    test_gear4_objwise)
        # DEPRECATED: Use test_gear4 --objwise instead
        echo "⚠️  WARNING: 'test_gear4_objwise' is deprecated!"
        echo "   Please use: ./run_docker.sh test_gear4 --objwise --dataset $OBJWISE_DATASET"
        echo ""
        echo "Redirecting to new command format..."
        echo ""

        # Redirect to new format
        OBJWISE_FLAG="true"
        OBJWISE_DATASET=${OBJWISE_DATASET:-waymo_seg}

        # Call test_gear4 with objwise flag
        exec "$0" test_gear4 --objwise --dataset "$OBJWISE_DATASET" "${@:2}"
        ;;

    train_gear5)
        echo "Starting Gear5 training..."
        echo "Configuration:"
        echo "  - Config variant: $CONFIG_VARIANT"
        echo "  - Config file: configs/gear5/config_$CONFIG_VARIANT.yaml"
        echo "  - Batch size: $BATCH_SIZE"
        echo "  - Workers: $WORKERS"
        echo "  - GPU: $GPU_ID"
        echo "  - Results directory: $RESULTS_DIR"
        echo "  - Loss type: $LOSS_TYPE"
        echo "  - Temporal backend: $([ "$MAMBA" = "true" ] && echo "Mamba2" || echo "GRU")"
        echo ""

        # Build train_gear5 command with config variant
        DOCKER_CMD="CUDA_VISIBLE_DEVICES=$GPU_ID docker compose run --rm -e WANDB_API_KEY=${WANDB_API_KEY:-} flashdepth python train_gear5.py \
            --config-path configs/gear5 \
            --config-name config_$CONFIG_VARIANT \
            dataset.data_root=/data/datasets \
            training.batch_size=$BATCH_SIZE \
            training.workers=$WORKERS \
            training.iterations=$TOTAL_ITERS \
            training.wandb=$WANDB \
            model.use_mamba_temporal=$MAMBA \
            use_canonical_space=$USE_CANONICAL \
            loss_type=$LOSS_TYPE \
            +results_dir=$RESULTS_DIR"

        # Add gear_checkpoint if specified (required for Phase 2 hybrid)
        if [ -n "$GEAR_CHECKPOINT" ]; then
            DOCKER_CMD="$DOCKER_CMD gear_checkpoint=$GEAR_CHECKPOINT"
        fi

        # Add wandb name if specified
        if [ -n "$WANDB_NAME" ]; then
            DOCKER_CMD="$DOCKER_CMD training.wandb_name=$WANDB_NAME"
        fi

        eval $DOCKER_CMD
        ;;

    train_gear5_ddp)
        # Gear5 DDP training with 2 GPUs
        # Phase 2 (Hybrid): Auto-adjust for 2K resolution if config_variant=hybrid
        if [ "$CONFIG_VARIANT" = "hybrid" ]; then
            # Hybrid: 2K resolution - reduce batch size, workers, and video length
            ACTUAL_WORKERS=1  # 1 worker per GPU (total 2 across 2 GPUs) - reduced for memory
            CANONICAL_FX="500.0"
            RES_NAME="2k"
            VIDEO_LENGTH=2  # Reduced from 5→3→2 for extreme memory constraints

            # Auto-adjust batch size for 2K resolution if user didn't specify
            if [ -z "$USER_BATCH_SIZE" ]; then
                BATCH_SIZE=1  # Minimal batch size for 2K resolution to avoid OOM
                echo "  NOTE: Auto-adjusted batch size to $BATCH_SIZE for Hybrid (2K resolution)"
            else
                echo "  WARNING: Using user-specified batch size $BATCH_SIZE - may cause OOM on 2K resolution!"
            fi
        else
            # Phase 1: 518x518 resolution - can use more workers and larger batch
            ACTUAL_WORKERS=$WORKERS
            CANONICAL_FX="500.0"
            RES_NAME="base"
            VIDEO_LENGTH=5  # Default video length
        fi

        echo "Starting Gear5 training (Multi-GPU: 0,1)..."
        echo "Configuration:"
        echo "  - Config variant: $CONFIG_VARIANT"
        echo "  - Config file: configs/gear5/config_$CONFIG_VARIANT.yaml"
        echo "  - Resolution: $RES_NAME"
        echo "  - Batch size per GPU: $BATCH_SIZE"
        echo "  - Effective batch size: $((BATCH_SIZE * 2))"
        echo "  - Workers per GPU: $ACTUAL_WORKERS"
        echo "  - Video length: $VIDEO_LENGTH frames"
        echo "  - Total iterations: $TOTAL_ITERS"
        echo "  - GPUs: 0,1"
        echo "  - Results directory: $RESULTS_DIR"
        echo "  - FPS measurement: $MEASURE_FPS"
        echo "  - Loss type: $LOSS_TYPE"
        echo "  - Temporal backend: $([ "$MAMBA" = "true" ] && echo "Mamba2" || echo "GRU")"
        echo ""

        DOCKER_CMD="CUDA_VISIBLE_DEVICES=0,1 docker compose run --rm \
            -e GLOO_SOCKET_IFNAME=eth0 \
            -e NCCL_SOCKET_IFNAME=eth0 \
            -e NCCL_P2P_DISABLE=1 \
            -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            -e WANDB_API_KEY=${WANDB_API_KEY:-} \
            -e WANDB_API_KEY=\${WANDB_API_KEY:-} \
            flashdepth torchrun \
            --standalone \
            --nproc_per_node=2 \
            train_gear5.py \
            --config-path configs/gear5 \
            --config-name config_$CONFIG_VARIANT \
            dataset.data_root=/data/datasets \
            dataset.video_length=$VIDEO_LENGTH \
            training.batch_size=$BATCH_SIZE \
            training.workers=$ACTUAL_WORKERS \
            training.iterations=$TOTAL_ITERS \
            training.measure_fps=$MEASURE_FPS \
            training.wandb=$WANDB \
            model.use_mamba_temporal=$MAMBA \
            use_canonical_space=$USE_CANONICAL \
            loss_type=$LOSS_TYPE \
            +results_dir=$RESULTS_DIR"

        # Add gear_checkpoint if specified (required for Phase 2 hybrid)
        if [ -n "$GEAR_CHECKPOINT" ]; then
            DOCKER_CMD="$DOCKER_CMD gear_checkpoint=$GEAR_CHECKPOINT"
        fi

        # Add wandb name if specified
        if [ -n "$WANDB_NAME" ]; then
            DOCKER_CMD="$DOCKER_CMD training.wandb_name=$WANDB_NAME"
        fi

        eval $DOCKER_CMD
        ;;

    test_gear5)
        echo "Starting Gear5 testing..."
        echo "Configuration:"
        echo "  - Video length: $VID_LEN"
        echo "  - Frame interval: $FRAME_INTERVAL"
        echo "  - GPU: $GPU_ID"
        echo "  - Results directory: $RESULTS_DIR"
        echo "  - Checkpoint: $FLASHDEPTH_CHECKPOINT"
        echo "  - Config variant: $CONFIG_VARIANT"
        echo "  - Workers: $WORKERS"
        echo "  - Temporal backend: $([ "$MAMBA" = "true" ] && echo "Mamba2" || echo "GRU")"
        if [ "$OBJWISE_FLAG" == "true" ]; then
            echo "  - Object-wise evaluation: ENABLED"
        fi
        if [ -n "$OBJWISE_DATASET" ]; then
            echo "  - Dataset: $OBJWISE_DATASET"
        else
            echo "  - Dataset: Using config defaults (all test datasets)"
        fi
        echo ""

        # Build test_gear5 command with config variant support
        TEST_CMD="python test_gear5.py \
            --config-path configs/gear5 \
            --config-name config_$CONFIG_VARIANT \
            dataset.data_root=/data/datasets \
            model.use_mamba_temporal=$MAMBA \
            training.workers=$WORKERS \
            +results_dir=$RESULTS_DIR \
            +gpu=$GPU_ID \
            +vid_len=$VID_LEN \
            +frame_interval=$FRAME_INTERVAL \
            +visualization=$VISUALIZATION \
            +config_dir=configs/gear5/$CONFIG_VARIANT"

        # Add --objwise flag if requested
        if [ "$OBJWISE_FLAG" == "true" ]; then
            TEST_CMD="$TEST_CMD --objwise"
        fi

        # Add dataset override if specified
        if [ -n "$OBJWISE_DATASET" ]; then
            TEST_CMD="$TEST_CMD eval.test_datasets=[$OBJWISE_DATASET]"
            # Remove _seg suffix for object_wise.dataset config
            OBJWISE_DATASET_BASE="${OBJWISE_DATASET/_seg/}"
            TEST_CMD="$TEST_CMD object_wise.dataset=$OBJWISE_DATASET_BASE"
        fi

        # Add limit_scenes if specified
        if [ -n "$LIMIT_SCENES" ]; then
            TEST_CMD="$TEST_CMD +dataset.limit_scenes=$LIMIT_SCENES"
        fi

        # Add resolution override
        TEST_CMD="$TEST_CMD +resolution=$RESOLUTION"

        # Add checkpoint
        if [ -n "$FLASHDEPTH_CHECKPOINT" ]; then
            TEST_CMD="$TEST_CMD load=$FLASHDEPTH_CHECKPOINT"
        fi

        CUDA_VISIBLE_DEVICES=$GPU_ID docker compose run --rm flashdepth $TEST_CMD
        ;;

    test_original_flashdepth)
        
        # Initialize CHECKPOINT from FLASHDEPTH_CHECKPOINT or environment variable
        if [ -z "$CHECKPOINT" ]; then
            CHECKPOINT="$FLASHDEPTH_CHECKPOINT"
        fi

        # Convert to absolute path if needed
        if [[ "$CHECKPOINT" != /app/* ]] && [[ "$CHECKPOINT" != /* ]]; then
            CHECKPOINT="/app/$CHECKPOINT"
        fi

        # Set default checkpoint based on config variant if using default
        if [ "$CHECKPOINT" = "/app/configs/flashdepth-l/iter_10001.pth" ]; then
            # User didn't specify checkpoint, use default for selected config
            case "$CONFIG" in
                flashdepth-l)
                    CHECKPOINT="/app/configs/flashdepth-l/iter_10001.pth"
                    ;;
                flashdepth-s)
                    CHECKPOINT="/app/configs/flashdepth-s/iter_14001.pth"
                    ;;
                flashdepth)
                    CHECKPOINT="/app/configs/flashdepth/iter_43002.pth"
                    ;;
                *)
                    echo "Unknown config variant: $CONFIG"
                    echo "Valid options: flashdepth, flashdepth-l, flashdepth-s"
                    exit 1
                    ;;
            esac
        fi

        # Use OBJWISE_DATASET if set, otherwise default to nuscenes
        TEST_DATASET="${OBJWISE_DATASET:-nuscenes}"

        # Use custom results dir if provided, otherwise default to test_results/${TEST_DATASET}_original_${CONFIG}
        if [ "$RESULTS_DIR" = "train_results/results_1" ]; then
            # Default value not changed by user, use dataset and config specific default
            OUTFOLDER="/app/test_results/${TEST_DATASET}_original_${CONFIG}"
        else
            # User provided custom results dir
            # If it's a relative path, prefix with /app/ to save to host
            if [[ "$RESULTS_DIR" != /* ]]; then
                OUTFOLDER="/app/$RESULTS_DIR"
            else
                OUTFOLDER="$RESULTS_DIR"
            fi
        fi

        # Extract results directory path for host (remove /app prefix for local path)
        LOCAL_OUTFOLDER="${OUTFOLDER#/app/}"

        # Determine visualization settings based on NO_VIDEO flag
        if [ "$NO_VIDEO" = "true" ]; then
            OUT_VIDEO="false"
            SAVE_DEPTH="false"
            VIS_STATUS="DISABLED (--no-video)"
        else
            OUT_VIDEO="true"
            SAVE_DEPTH="true"
            VIS_STATUS="ENABLED (MP4 + .npy depth files)"
        fi

        echo "Testing Original FlashDepth (inference mode)..."
        echo "Configuration:"
        echo "  - Config variant: $CONFIG"
        echo "  - Dataset: $TEST_DATASET (first sequence only for FPS)"
        echo "  - Resolution: $RESOLUTION"
        echo "  - GPU: $GPU_ID"
        echo "  - Checkpoint: $CHECKPOINT"
        echo "  - Results directory: $OUTFOLDER"
        echo "  - Inverse colormap: $INVERSE"
        echo "  - Visualization: $VIS_STATUS"
        echo "  - DataLoader workers: $WORKERS"
        echo "  - Log file: ${LOCAL_OUTFOLDER}/test.log"
        echo ""

        # Create output directory on host
        mkdir -p "$LOCAL_OUTFOLDER"

        # Build base command
        TEST_CMD="cd /FlashDepth && torchrun --nproc_per_node=1 train.py \
          --config-path configs/$CONFIG \
          inference=true \
          eval.test_datasets=[$TEST_DATASET] \
          eval.metrics=true \
          dataset.data_root=/data/datasets \
          eval.outfolder=$OUTFOLDER \
          load=$CHECKPOINT \
          +eval.inverse=$INVERSE \
          eval.first_seq_only=true \
          eval.compile=false \
          eval.out_video=$OUT_VIDEO \
          eval.save_depth_npy=$SAVE_DEPTH \
          eval.num_workers=$WORKERS \
          eval.test_dataset_resolution=$RESOLUTION"

        # Add limit_scenes if specified
        if [ -n "$LIMIT_SCENES" ]; then
            TEST_CMD="$TEST_CMD \
          +eval.limit_scenes=$LIMIT_SCENES"
        fi

        # Run test and save log
        echo "Running FlashDepth inference on $TEST_DATASET..."
        CUDA_VISIBLE_DEVICES=$GPU_ID docker compose run --rm flashdepth bash -c "$TEST_CMD" 2>&1 | tee "${LOCAL_OUTFOLDER}/test.log"

        # Parse log and save as JSON
        echo ""
        echo "Parsing results to JSON..."
        python3 utils/parse_flashdepth_results.py "${LOCAL_OUTFOLDER}/test.log" "$LOCAL_OUTFOLDER"

        echo ""
        echo "✓ Test complete! Results saved to:"
        echo "  - FPS Summary: ${LOCAL_OUTFOLDER}/fps_results.json"
        echo "  - Per-sequence FPS: ${LOCAL_OUTFOLDER}/per_sequence_fps.json"
        echo "  - Full log: ${LOCAL_OUTFOLDER}/test.log"
        if [ "$NO_VIDEO" != "true" ]; then
            echo "  - MP4 videos: ${LOCAL_OUTFOLDER}/*/*.mp4"
            echo "  - Depth files (.npy): ${LOCAL_OUTFOLDER}/*/*.npy"
        fi
        ;;

    clean)
        echo "Cleaning up Docker containers and images..."
        docker compose down
        docker compose rm -f
        docker image rm flashdepth:latest 2>/dev/null || true
        echo "Cleanup completed!"
        ;;

    logs)
        echo "Showing container logs..."
        docker compose logs -f flashdepth
        ;;

    "")
        echo "No command specified."
        show_usage
        exit 1
        ;;

    *)
        echo "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac