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
    echo "  train_gear3_upgrade Start Gear3 Upgrade training (single GPU) - Enhanced FG/BG separation"
    echo "  train_gear3_upgrade_ddp Start Gear3 Upgrade training with 2 GPUs (GPU 0,1)"
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
    echo "  --gsp-checkpoint PATH Set GSP module weights path"
    echo "  --frame-interval NUM  Set frame interval for sequence visualization (default: 1)"
    echo "  --vid-len NUM         Set video sequence length for testing (default: 50)"
    echo "  --single-sequence PATH Test on a single sequence directory (e.g., /path/to/dynamicreplica/seq)"
    echo "  --measure-fps BOOL    Enable/disable FPS measurement (default: true)"
    echo "  --phase NUM           Set training phase (1, 2, or 3, default: 1)"
    echo "  --separation METHOD  Set FG/BG separation method: cls_seg, kmeans, multi_layer (default: cls_seg)"
    echo ""
    echo "Note: Regularization losses are deprecated. Importance map now uses raw DINOv2 attention (frozen)."
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
    echo "  $0 train_gear3_upgrade                # Start Gear3 Upgrade training (cls_seg method)"
    echo "  $0 train_gear3_upgrade --separation kmeans  # Train with K-means clustering"
    echo "  $0 train_gear3_upgrade_ddp --separation multi_layer  # DDP with multi-layer fusion"
    echo "  $0 train --results-dir train_results/results_2  # Custom results directory"
    echo "  $0 shell                              # Interactive development"
}

# Parse command line arguments - optimized for RTX A6000 (2x 48GB)
COMMAND=""
BATCH_SIZE=20  # Per GPU (effective 40 with 2 GPUs in DDP)
WORKERS=8      # Optimized for 96 CPU cores, prevents I/O bottleneck
TOTAL_ITERS=40001
GPU_ID=0
RESULTS_DIR="train_results/results_1"
FLASHDEPTH_CHECKPOINT="configs/flashdepth-l/iter_10001.pth"
GSP_CHECKPOINT="train_results/results_5/best_metric_head_step_21000.pth"  # Only used for old test script
FRAME_INTERVAL=1
VID_LEN=50
SINGLE_SEQUENCE=""  # Path to single sequence directory (optional)
MEASURE_FPS="true"
PHASE=1  # Training phase (1, 2, or 3)
SEPARATION_METHOD="cls_seg"  # FG/BG separation method for Gear3 Upgrade (cls_seg, kmeans, multi_layer)

# Parse arguments
USER_BATCH_SIZE=""  # Track if user explicitly set batch size
while [[ $# -gt 0 ]]; do
    case $1 in
        build|train|test|train_gear2|train_gear2_ddp|test_gear2|train_gear3|train_gear3_ddp|test_gear3|train_gear3_upgrade|train_gear3_upgrade_ddp|shell|clean|logs)
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
        --single-sequence)
            SINGLE_SEQUENCE="$2"
            shift 2
            ;;
        --measure-fps)
            MEASURE_FPS="$2"
            shift 2
            ;;
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --separation)
            SEPARATION_METHOD="$2"
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

    train_gear2)
        echo "Starting Gear2 training (Single GPU) - Ablation Study..."
        echo "Configuration:"
        echo "  - Batch size: $BATCH_SIZE"
        echo "  - Workers: $WORKERS"
        echo "  - Total iterations: $TOTAL_ITERS"
        echo "  - GPU: $GPU_ID"
        echo "  - Results directory: $RESULTS_DIR"
        echo ""

        # Build train_gear2 command (uses all 5 datasets hardcoded in code)
        DOCKER_CMD="CUDA_VISIBLE_DEVICES=$GPU_ID docker compose run --rm flashdepth python train_gear2.py \
            --config-path configs/gear2 \
            dataset.data_root=/data/datasets \
            training.batch_size=$BATCH_SIZE \
            training.workers=$WORKERS \
            training.iterations=$TOTAL_ITERS \
            +results_dir=$RESULTS_DIR"

        # Add checkpoint loading - Gear2 requires FlashDepth-L pretrained weights
        if [ -n "$FLASHDEPTH_CHECKPOINT" ]; then
            DOCKER_CMD="$DOCKER_CMD load=$FLASHDEPTH_CHECKPOINT"
        fi

        eval $DOCKER_CMD
        ;;

    train_gear2_ddp)
        # Auto-adjust batch size, workers, and iterations for Phase 2/3 (2K resolution) if not explicitly set
        if [ "$PHASE" -eq 2 ] || [ "$PHASE" -eq 3 ]; then
            # Phase 2/3: 2K resolution - reduce both batch size and workers
            ACTUAL_WORKERS=2  # 2 workers per GPU (total 4 across 2 GPUs)

            # Auto-adjust batch size for 2K resolution if user didn't specify
            if [ -z "$USER_BATCH_SIZE" ]; then
                BATCH_SIZE=1  # Minimal batch size for 2K resolution to avoid OOM
                echo "  NOTE: Auto-adjusted batch size to $BATCH_SIZE for Phase $PHASE (2K resolution)"
            else
                echo "  WARNING: Using user-specified batch size $BATCH_SIZE - may cause OOM on 2K resolution!"
            fi

            # Reduce iterations for Phase 2/3 (half of Phase 1)
            TOTAL_ITERS=30001
            echo "  NOTE: Auto-adjusted iterations to $TOTAL_ITERS for Phase $PHASE (2K resolution)"
        else
            # Phase 1: 518x518 resolution - can use more workers and larger batch
            ACTUAL_WORKERS=$WORKERS
        fi

        echo "Starting Gear2 training (Multi-GPU: 0,1) - Ablation Study..."
        echo "Configuration:"
        echo "  - Phase: $PHASE"
        echo "  - Batch size per GPU: $BATCH_SIZE"
        echo "  - Effective batch size: $((BATCH_SIZE * 2))"
        echo "  - Workers per GPU: $ACTUAL_WORKERS"
        echo "  - Total iterations: $TOTAL_ITERS"
        echo "  - GPUs: 0,1"
        echo "  - Results directory: $RESULTS_DIR"
        echo "  - FPS measurement: $MEASURE_FPS"
        echo ""

        DOCKER_CMD="CUDA_VISIBLE_DEVICES=0,1 docker compose run --rm \
            -e GLOO_SOCKET_IFNAME=eth0 \
            -e NCCL_SOCKET_IFNAME=eth0 \
            -e NCCL_P2P_DISABLE=1 \
            flashdepth torchrun \
            --standalone \
            --nproc_per_node=2 \
            train_gear2.py \
            --config-path configs/gear2 \
            dataset.data_root=/data/datasets \
            phase=$PHASE \
            training.batch_size=$BATCH_SIZE \
            training.workers=$ACTUAL_WORKERS \
            training.iterations=$TOTAL_ITERS \
            training.measure_fps=$MEASURE_FPS \
            +results_dir=$RESULTS_DIR"

        # Add checkpoint loading
        if [ -n "$FLASHDEPTH_CHECKPOINT" ]; then
            DOCKER_CMD="$DOCKER_CMD load=$FLASHDEPTH_CHECKPOINT"
        fi

        eval $DOCKER_CMD
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
        # Auto-adjust batch size, workers, and iterations for Phase 2/3 (2K resolution) if not explicitly set
        if [ "$PHASE" -eq 2 ] || [ "$PHASE" -eq 3 ]; then
            # Phase 2/3: 2K resolution - reduce both batch size and workers
            ACTUAL_WORKERS=2  # 2 workers per GPU (total 4 across 2 GPUs)

            # Auto-adjust batch size for 2K resolution if user didn't specify
            if [ -z "$USER_BATCH_SIZE" ]; then
                BATCH_SIZE=1  # Minimal batch size for 2K resolution to avoid OOM
                echo "  NOTE: Auto-adjusted batch size to $BATCH_SIZE for Phase $PHASE (2K resolution)"
            else
                echo "  WARNING: Using user-specified batch size $BATCH_SIZE - may cause OOM on 2K resolution!"
            fi

            # Reduce iterations for Phase 2/3 (half of Phase 1)
            TOTAL_ITERS=30001
            echo "  NOTE: Auto-adjusted iterations to $TOTAL_ITERS for Phase $PHASE (2K resolution)"
        else
            # Phase 1: 518x518 resolution - can use more workers and larger batch
            ACTUAL_WORKERS=$WORKERS
        fi

        echo "Starting Gear3 training (Multi-GPU: 0,1)..."
        echo "Configuration:"
        echo "  - Phase: $PHASE"
        echo "  - Batch size per GPU: $BATCH_SIZE"
        echo "  - Effective batch size: $((BATCH_SIZE * 2))"
        echo "  - Workers per GPU: $ACTUAL_WORKERS"
        echo "  - Total iterations: $TOTAL_ITERS"
        echo "  - GPUs: 0,1"
        echo "  - Results directory: $RESULTS_DIR"
        echo "  - FPS measurement: $MEASURE_FPS"
        echo ""

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
            phase=$PHASE \
            training.batch_size=$BATCH_SIZE \
            training.workers=$ACTUAL_WORKERS \
            training.iterations=$TOTAL_ITERS \
            training.measure_fps=$MEASURE_FPS \
            +results_dir=$RESULTS_DIR"

        # Add checkpoint loading
        if [ -n "$FLASHDEPTH_CHECKPOINT" ]; then
            DOCKER_CMD="$DOCKER_CMD load=$FLASHDEPTH_CHECKPOINT"
        fi

        eval $DOCKER_CMD
        ;;

    train_gear3_upgrade)
        echo "Starting Gear3 Upgrade training (Single GPU) - Enhanced FG/BG Separation..."
        echo "Configuration:"
        echo "  - Separation method: $SEPARATION_METHOD"
        echo "  - Batch size: $BATCH_SIZE"
        echo "  - Workers: $WORKERS"
        echo "  - Total iterations: $TOTAL_ITERS"
        echo "  - GPU: $GPU_ID"
        echo "  - Results directory: $RESULTS_DIR"
        echo ""

        # Build train_gear3_upgrade command
        DOCKER_CMD="CUDA_VISIBLE_DEVICES=$GPU_ID docker compose run --rm flashdepth python train_gear3_upgrade.py \
            --config-path configs/gear3_upgrade \
            dataset.data_root=/data/datasets \
            training.batch_size=$BATCH_SIZE \
            training.workers=$WORKERS \
            training.iterations=$TOTAL_ITERS \
            separation_method=$SEPARATION_METHOD \
            +results_dir=$RESULTS_DIR"

        # Add checkpoint loading - Gear3 Upgrade requires FlashDepth-L pretrained weights
        if [ -n "$FLASHDEPTH_CHECKPOINT" ]; then
            DOCKER_CMD="$DOCKER_CMD load=$FLASHDEPTH_CHECKPOINT"
        fi

        eval $DOCKER_CMD
        ;;

    train_gear3_upgrade_ddp)
        # Auto-adjust batch size, workers, and iterations for Phase 2/3 (2K resolution) if not explicitly set
        if [ "$PHASE" -eq 2 ] || [ "$PHASE" -eq 3 ]; then
            # Phase 2/3: 2K resolution - reduce both batch size and workers
            ACTUAL_WORKERS=2  # 2 workers per GPU (total 4 across 2 GPUs)

            # Auto-adjust batch size for 2K resolution if user didn't specify
            if [ -z "$USER_BATCH_SIZE" ]; then
                BATCH_SIZE=1  # Minimal batch size for 2K resolution to avoid OOM
                echo "  NOTE: Auto-adjusted batch size to $BATCH_SIZE for Phase $PHASE (2K resolution)"
            else
                echo "  WARNING: Using user-specified batch size $BATCH_SIZE - may cause OOM on 2K resolution!"
            fi

            # Reduce iterations for Phase 2/3 (half of Phase 1)
            TOTAL_ITERS=30001
            echo "  NOTE: Auto-adjusted iterations to $TOTAL_ITERS for Phase $PHASE (2K resolution)"
        else
            # Phase 1: 518x518 resolution - can use more workers and larger batch
            ACTUAL_WORKERS=$WORKERS
        fi

        echo "Starting Gear3 Upgrade training (Multi-GPU: 0,1) - Enhanced FG/BG Separation..."
        echo "Configuration:"
        echo "  - Separation method: $SEPARATION_METHOD"
        echo "  - Phase: $PHASE"
        echo "  - Batch size per GPU: $BATCH_SIZE"
        echo "  - Effective batch size: $((BATCH_SIZE * 2))"
        echo "  - Workers per GPU: $ACTUAL_WORKERS"
        echo "  - Total iterations: $TOTAL_ITERS"
        echo "  - GPUs: 0,1"
        echo "  - Results directory: $RESULTS_DIR"
        echo "  - FPS measurement: $MEASURE_FPS"
        echo ""

        DOCKER_CMD="CUDA_VISIBLE_DEVICES=0,1 docker compose run --rm \
            -e GLOO_SOCKET_IFNAME=eth0 \
            -e NCCL_SOCKET_IFNAME=eth0 \
            -e NCCL_P2P_DISABLE=1 \
            flashdepth torchrun \
            --standalone \
            --nproc_per_node=2 \
            train_gear3_upgrade.py \
            --config-path configs/gear3_upgrade \
            dataset.data_root=/data/datasets \
            phase=$PHASE \
            training.batch_size=$BATCH_SIZE \
            training.workers=$ACTUAL_WORKERS \
            training.iterations=$TOTAL_ITERS \
            training.measure_fps=$MEASURE_FPS \
            separation_method=$SEPARATION_METHOD \
            +results_dir=$RESULTS_DIR"

        # Add checkpoint loading
        if [ -n "$FLASHDEPTH_CHECKPOINT" ]; then
            DOCKER_CMD="$DOCKER_CMD load=$FLASHDEPTH_CHECKPOINT"
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
        echo "  - GPU: $GPU_ID"
        echo "  - Results directory: $RESULTS_DIR"
        echo "  - Video length: $VID_LEN"
        echo "  - Frame interval: $FRAME_INTERVAL"
        echo "  - Checkpoint: $FLASHDEPTH_CHECKPOINT"
        if [ -n "$SINGLE_SEQUENCE" ]; then
            echo "  - Single sequence: $SINGLE_SEQUENCE"
        fi
        echo ""

        # Build test_gear2 command (uses single checkpoint with all weights)
        TEST_CMD="python test_gear2.py --config-path configs/gear2 dataset.data_root=/data/datasets +results_dir=$RESULTS_DIR +gpu=$GPU_ID"

        # Use --checkpoint option for the unified checkpoint
        if [ -n "$FLASHDEPTH_CHECKPOINT" ]; then
            TEST_CMD="$TEST_CMD load=$FLASHDEPTH_CHECKPOINT"
        fi

        # Add frame interval and video length options
        TEST_CMD="$TEST_CMD +frame_interval=$FRAME_INTERVAL +vid_len=$VID_LEN"

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
        echo "  - GPU: $GPU_ID"
        echo "  - Results directory: $RESULTS_DIR"
        echo "  - Video length: $VID_LEN"
        echo "  - Frame interval: $FRAME_INTERVAL"
        echo "  - Checkpoint: $FLASHDEPTH_CHECKPOINT"
        if [ -n "$SINGLE_SEQUENCE" ]; then
            echo "  - Single sequence: $SINGLE_SEQUENCE"
        fi
        echo ""

        # Build test_gear3 command (uses single checkpoint with all weights)
        TEST_CMD="python test_gear3.py --config-path configs/gear3 dataset.data_root=/data/datasets +results_dir=$RESULTS_DIR +gpu=$GPU_ID"

        # Use --checkpoint option for the unified checkpoint
        if [ -n "$FLASHDEPTH_CHECKPOINT" ]; then
            TEST_CMD="$TEST_CMD load=$FLASHDEPTH_CHECKPOINT"
        fi

        # Add frame interval and video length options
        TEST_CMD="$TEST_CMD +frame_interval=$FRAME_INTERVAL +vid_len=$VID_LEN"

        # Add single sequence path if specified
        if [ -n "$SINGLE_SEQUENCE" ]; then
            TEST_CMD="$TEST_CMD +single_sequence=$SINGLE_SEQUENCE"
        fi

        # Run test_gear3.py with custom parameters (with GPU selection)
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