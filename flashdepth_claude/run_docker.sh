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
    echo "  train_gear3 Start Gear3 training (single GPU)"
    echo "  train_gear3_ddp Start Gear3 training with 2 GPUs (GPU 0,1)"
    echo "  test_gear3  Start Gear3 testing with default settings"
    echo "  shell       Start interactive shell in container"
    echo "  clean       Remove containers and images"
    echo "  logs        Show container logs"
    echo ""
    echo "Training Options:"
    echo "  --batch-size SIZE     Set batch size per GPU (default: 22, effective 44 with DDP)"
    echo "  --workers NUM         Set number of DataLoader workers (default: 8)"
    echo "  --epochs NUM          Set number of training iterations (default: 60001)"
    echo "  --gpu ID              Set GPU ID (default: 0)"
    echo "  --results-dir PATH    Set results directory (default: train_results/results_1)"
    echo "  --flashdepth-checkpoint PATH  Set FlashDepth pretrained weights path"
    echo "  --gsp-checkpoint PATH Set GSP module weights path"
    echo "  --frame-interval NUM  Set frame interval for sequence visualization (default: 1)"
    echo "  --vid-len NUM         Set video sequence length for testing (default: 50)"
    echo ""
    echo "Examples:"
    echo "  $0 build                              # Build the image"
    echo "  $0 train                              # Start training with defaults"
    echo "  $0 train --batch-size 4 --gpu 1      # Train with custom settings"
    echo "  $0 test                               # Start testing with defaults"
    echo "  $0 test --vid-len 25 --frame-interval 2  # Test with custom video length and frame interval"
    echo "  $0 train_gear3                        # Start Gear3 training with defaults"
    echo "  $0 train_gear3 --batch-size 8 --gpu 1 # Train Gear3 with custom settings"
    echo "  $0 test_gear3                         # Start Gear3 testing with defaults"
    echo "  $0 test_gear3 --vid-len 25 --gpu 1    # Test Gear3 with custom settings"
    echo "  $0 train --results-dir train_results/results_2  # Custom results directory"
    echo "  $0 shell                              # Interactive development"
}

# Parse command line arguments - optimized for RTX A6000 (2x 48GB)
COMMAND=""
BATCH_SIZE=22  # Per GPU (effective 44 with 2 GPUs in DDP)
WORKERS=8      # Optimized for 96 CPU cores, prevents I/O bottleneck
TOTAL_ITERS=60001
GPU_ID=0
RESULTS_DIR="train_results/results_1"
FLASHDEPTH_CHECKPOINT="configs/flashdepth-l/iter_10001.pth"
GSP_CHECKPOINT="train_results/results_5/best_metric_head_step_21000.pth"
FRAME_INTERVAL=1
VID_LEN=50

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        build|train|test|train_gear3|train_gear3_ddp|test_gear3|shell|clean|logs)
            COMMAND="$1"
            shift
            ;;
        --batch-size)
            BATCH_SIZE="$2"
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
        --gsp-checkpoint)
            GSP_CHECKPOINT="$2"
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

        if [ -n "$GSP_CHECKPOINT" ]; then
            TEST_CMD="$TEST_CMD +gsp_checkpoint=$GSP_CHECKPOINT"
        fi

        # Add frame interval and video length options
        TEST_CMD="$TEST_CMD +frame_interval=$FRAME_INTERVAL +vid_len=$VID_LEN"

        # Run test_metric_head.py with custom parameters
        docker compose run --rm flashdepth $TEST_CMD
        ;;

    train_gear3)
        echo "Starting Gear3 training (Single GPU)..."
        echo "Configuration:"
        echo "  - Batch size: $BATCH_SIZE"
        echo "  - Workers: $WORKERS"
        echo "  - Total iterations: $TOTAL_ITERS"
        echo "  - GPU: $GPU_ID"
        echo "  - Results directory: $RESULTS_DIR"
        echo ""

        # Build train_gear3 command (uses all 5 datasets hardcoded in code)
        DOCKER_CMD="CUDA_VISIBLE_DEVICES=$GPU_ID docker compose run --rm flashdepth python train_gear3.py \
            --config-path configs/gear3 \
            dataset.data_root=/data/datasets \
            training.batch_size=$BATCH_SIZE \
            training.workers=$WORKERS \
            training.iterations=$TOTAL_ITERS \
            +results_dir=$RESULTS_DIR"

        # Add checkpoint loading - Gear3 requires FlashDepth-L pretrained weights
        if [ -n "$FLASHDEPTH_CHECKPOINT" ]; then
            DOCKER_CMD="$DOCKER_CMD load=$FLASHDEPTH_CHECKPOINT"
        fi

        eval $DOCKER_CMD
        ;;

    train_gear3_ddp)
        echo "Starting Gear3 training (Multi-GPU: 0,1)..."
        echo "Configuration:"
        echo "  - Batch size: $BATCH_SIZE (per GPU)"
        echo "  - Effective batch size: $((BATCH_SIZE * 2))"
        echo "  - Workers: $WORKERS"
        echo "  - Total iterations: $TOTAL_ITERS"
        echo "  - GPUs: 0,1"
        echo "  - Results directory: $RESULTS_DIR"
        echo ""

        # Build train_gear3 DDP command using --standalone for single-node multi-GPU
        DOCKER_CMD="CUDA_VISIBLE_DEVICES=0,1 docker compose run --rm \
            -e GLOO_SOCKET_IFNAME=eth0 \
            -e NCCL_SOCKET_IFNAME=eth0 \
            -e NCCL_P2P_DISABLE=1 \
            flashdepth torchrun \
            --standalone \
            --nproc_per_node=2 \
            train_gear3.py \
            --config-path configs/gear3 \
            dataset.data_root=/data/datasets \
            phase=1 \
            training.batch_size=$BATCH_SIZE \
            training.workers=$WORKERS \
            training.iterations=$TOTAL_ITERS \
            +results_dir=$RESULTS_DIR"

        # Add checkpoint loading
        if [ -n "$FLASHDEPTH_CHECKPOINT" ]; then
            DOCKER_CMD="$DOCKER_CMD load=$FLASHDEPTH_CHECKPOINT"
        fi

        eval $DOCKER_CMD
        ;;

    test_gear3)
        echo "Starting Gear3 testing..."
        echo "Configuration:"
        echo "  - GPU: $GPU_ID"
        echo "  - Results directory: $RESULTS_DIR"
        echo "  - Video length: $VID_LEN"
        echo "  - Frame interval: $FRAME_INTERVAL"
        echo ""

        # Build test_gear3 command
        TEST_CMD="python test_gear3.py --config-path configs/gear3 +results_dir=$RESULTS_DIR +gpu=$GPU_ID"

        if [ -n "$FLASHDEPTH_CHECKPOINT" ]; then
            TEST_CMD="$TEST_CMD +flashdepth_checkpoint=$FLASHDEPTH_CHECKPOINT"
        fi

        if [ -n "$GSP_CHECKPOINT" ]; then
            TEST_CMD="$TEST_CMD +gsp_checkpoint=$GSP_CHECKPOINT"
        fi

        # Add frame interval and video length options
        TEST_CMD="$TEST_CMD +frame_interval=$FRAME_INTERVAL +vid_len=$VID_LEN"

        # Run test_gear3.py with custom parameters
        docker compose run --rm flashdepth $TEST_CMD
        ;;

    shell)
        echo "Starting interactive shell..."
        docker compose run --rm flashdepth /bin/bash
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