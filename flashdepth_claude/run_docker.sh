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
    echo "  build     Build the Docker image"
    echo "  train     Start training with default settings"
    echo "  shell     Start interactive shell in container"
    echo "  clean     Remove containers and images"
    echo "  logs      Show container logs"
    echo ""
    echo "Training Options:"
    echo "  --batch-size SIZE     Set batch size (default: 4)"
    echo "  --workers NUM         Set number of workers (default: 10)"
    echo "  --epochs NUM          Set number of training iterations (default: 60001)"
    echo "  --gpu ID              Set GPU ID (default: 0)"
    echo "  --results-dir PATH    Set results directory (default: train_results/results_1)"
    echo ""
    echo "Examples:"
    echo "  $0 build                              # Build the image"
    echo "  $0 train                              # Start training with defaults"
    echo "  $0 train --batch-size 4 --gpu 1      # Train with custom settings"
    echo "  $0 train --results-dir train_results/results_2  # Custom results directory"
    echo "  $0 shell                              # Interactive development"
}

# Parse command line arguments - using original FlashDepth settings
COMMAND=""
BATCH_SIZE=4
WORKERS=10
TOTAL_ITERS=60001
GPU_ID=0
RESULTS_DIR="train_results/results_1"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        build|train|shell|clean|logs)
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
        docker compose run --rm flashdepth python train_metric_head.py \
            --config-path configs/flashdepth-l \
            dataset.data_root=/data/datasets \
            dataset.train_datasets=[tartanair] \
            dataset.val_datasets=[tartanair] \
            training.batch_size=12 \
            training.workers=4 \
            training.total_iters=$TOTAL_ITERS \
            model.attn_class=Attention \
            +results_dir=$RESULTS_DIR
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