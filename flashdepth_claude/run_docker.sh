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
    echo "  train_gear5_bankai      Start Gear5 Bankai training - Unified Mamba for temporal + metric"
    echo "  train_gear5_bankai_ddp  Start Gear5 Bankai training with 2 GPUs (GPU 0,1)"
    echo "  test_gear5_bankai       Start Gear5 Bankai testing"
    echo "  train_gear5_film     Start Gear5 FiLM training - Channel-wise FiLM modulation before Mamba"
    echo "  train_gear5_film_ddp Start Gear5 FiLM training with 2 GPUs (GPU 0,1)"
    echo "  test_gear5_film      Start Gear5 FiLM testing"
    echo "  train_onepiece       Start Onepiece training (Unified Global Mamba, single GPU)"
    echo "  train_onepiece_ddp   Start Onepiece training with 2 GPUs"
    echo "  test_onepiece        Start Onepiece testing"
    echo "  infer_avante         Run Gear5 inference on avante_images (custom dataset)"
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
    echo "  --cls-layer LAYERS   Select CLS token extraction layers (1-4). Examples: '4' (single), '2,4' (default), '1,2,3,4' (all)"
    echo "  --tsp-mode MODE      TSP embed_dim mode: auto (default), l (1024-dim), s (384-dim). Use 'l' to force TSP-L in hybrid mode"
    echo "  --bankai-phase PHASE Bankai training phase: 1 (metric head only), 2 (full training), auto (Phase 1→2 at auto-step). Default: auto"
    echo "  --bankai-auto-step N Step at which to transition Phase 1→2 in auto mode (default: 5000)"
    echo "  --tgm-weight WEIGHT  TGM loss weight for Bankai mode (default: 0.3, set 0 to disable)"
    echo "  --seq N              Sequence selection (e.g., --seq 0,4 for sequences 0 and 4)"
    echo "  --limit-scenes N     For NuScenes, limit the number of scenes to process (e.g., 50)"
    echo "  --best-figure        Export best_frame ±4 frames (9 total) as individual images/depth maps"
    echo "  --frame N            Export frame N ±4 frames (9 total) as individual images/depth maps (e.g., --seq 6 --frame 459)"
    echo "  --section START,END  Frame section for infer_avante (e.g., --section 450,480 for frames 450-480)"
    echo "  --no-inverse          Apply scale/shift in depth space instead of inverse depth space"
    echo "  --no-shift            Disable shift (scale-only mode): shift is always 0, only scale is learned/evaluated"
    echo "  --max-depth METERS   Max valid depth threshold (default: 80.0)"
    echo "  --model-type TYPE    Model type for infer_avante: gear5 (default), onepiece"
    echo ""
    echo "Note: Regularization losses are deprecated. Importance map now uses raw DINOv2 attention (frozen)."
    echo "Note: test_original_flashdepth now tests all sequences (use --limit-scenes N to limit)."
    echo "      Supports --seq N (e.g., --seq 0,4) and --frame N for eval visualization."
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
    echo ""
    echo "Bankai Mode Examples:"
    echo "  $0 train_gear5_bankai_ddp --gpu 0  # Auto mode (default): Phase 1 until step 5000, then Phase 2"
    echo "  $0 train_gear5_bankai --bankai-phase 1 --gpu 0  # Phase 1 only: Train metric head only"
    echo "  $0 train_gear5_bankai --bankai-phase 2 --gear-checkpoint train_results/bankai_phase1/best.pth --gpu 0  # Phase 2 only"
    echo "  $0 train_gear5_bankai_ddp --tgm-weight 0.3  # Auto mode with TGM loss"
    echo "  $0 test_gear5_bankai --gear-checkpoint train_results/bankai/best.pth --gpu 0  # Test Bankai model"
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
CLS_LAYERS="2,4"  # CLS token extraction layers (1-4): default is 2,4 (2nd and 4th intermediate layers)
TSP_MODE="auto"  # TSP embed_dim mode: auto (Phase1→model dim, Phase2→Student dim), l (1024), s (384)
SEQ=""  # Sequence selection for UnrealStereo4K (test_original_flashdepth)
LIMIT_SCENES=""  # Limit number of scenes for NuScenes dataset (optional, e.g., 50)
BEST_FIGURE="false"  # Export best_frame ±4 frames (9 total) as individual images/depth maps
FRAME=""  # Specific frame to export ±4 frames
FGWISE_FLAG="false"  # Enable FG-wise (foreground-wise) evaluation using ViT attention masks
SECTION=""  # Frame section for infer_avante (e.g., "450,480" for frames 450-480)
MAX_DEPTH="80.0"  # Max valid depth threshold (meters)
CBAR="false"  # Show colorbar next to depth visualization
TEST_MODE=""  # Test mode: empty (full), tc (temporal consistency only)
BANKAI_PHASE="auto"  # Bankai training phase: 1, 2, or "auto" (auto: Phase 1 until step 5000, then Phase 2)
BANKAI_AUTO_STEP="5000"  # Step at which to transition from Phase 1 to Phase 2 in auto mode
TGM_WEIGHT="0.3"  # TGM loss weight for Bankai mode
USE_LOG_SPACE="true"  # Use log space for depth/TGM loss (--no-log-space to disable)
DDP_GPUS="0,1"  # GPU IDs for DDP training (e.g., "0,1" or "1,2")
NO_INVERSE="false"  # Apply scale/shift in depth space instead of inverse depth (--no-inverse)
NO_SHIFT="false"  # Disable shift, scale-only mode (--no-shift)
MODEL_TYPE="gear5"  # Model type for infer_avante: gear5 (default), onepiece

# Parse arguments
USER_BATCH_SIZE=""  # Track if user explicitly set batch size
while [[ $# -gt 0 ]]; do
    case $1 in
        build|train|test|train_gear2|train_gear2_ddp|test_gear2|train_gear3|train_gear3_ddp|test_gear3|train_gear4|train_gear4_ddp|test_gear4|train_gear5|train_gear5_ddp|test_gear5|train_gear5_bankai|train_gear5_bankai_ddp|test_gear5_bankai|train_gear5_film|train_gear5_film_ddp|test_gear5_film|train_onepiece|train_onepiece_ddp|test_onepiece|infer_avante|test_original_flashdepth|shell|clean|logs)
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
        --fgwise)
            FGWISE_FLAG="true"
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
        --cls-layer)
            CLS_LAYERS="$2"
            shift 2
            ;;
        --tsp-mode)
            TSP_MODE="$2"
            shift 2
            ;;
        --seq)
            SEQ="$2"
            shift 2
            ;;
        --limit-scenes)
            LIMIT_SCENES="$2"
            shift 2
            ;;
        --best-figure)
            BEST_FIGURE="true"
            shift
            ;;
        --frame)
            FRAME="$2"
            shift 2
            ;;
        --section)
            SECTION="$2"
            shift 2
            ;;
        --max-depth)
            MAX_DEPTH="$2"
            shift 2
            ;;
        --cbar)
            CBAR="true"
            shift
            ;;
        --test-mode)
            TEST_MODE="$2"
            shift 2
            ;;
        --bankai-phase)
            BANKAI_PHASE="$2"
            shift 2
            ;;
        --bankai-auto-step)
            BANKAI_AUTO_STEP="$2"
            shift 2
            ;;
        --tgm-weight)
            TGM_WEIGHT="$2"
            shift 2
            ;;
        --no-log-space)
            USE_LOG_SPACE="false"
            shift
            ;;
        --no-inverse)
            NO_INVERSE="true"
            shift
            ;;
        --no-shift)
            NO_SHIFT="true"
            shift
            ;;
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --ddp-gpus)
            DDP_GPUS="$2"
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
        echo "  - CLS layers: $CLS_LAYERS"
        echo "  - TSP mode: $TSP_MODE"
        echo "  - No-inverse: $NO_INVERSE"
        echo "  - No-shift: $NO_SHIFT"
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
            model.tsp_mode=$TSP_MODE \
            use_canonical_space=$USE_CANONICAL \
            loss_type=$LOSS_TYPE \
            cls_layers='[$CLS_LAYERS]' \
            +no_inverse=$NO_INVERSE \
            +no_shift=$NO_SHIFT \
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
        echo "  - CLS layers: $CLS_LAYERS"
        echo "  - TSP mode: $TSP_MODE"
        echo "  - No-inverse: $NO_INVERSE"
        echo "  - No-shift: $NO_SHIFT"
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
            model.tsp_mode=$TSP_MODE \
            use_canonical_space=$USE_CANONICAL \
            loss_type=$LOSS_TYPE \
            cls_layers='[$CLS_LAYERS]' \
            +no_inverse=$NO_INVERSE \
            +no_shift=$NO_SHIFT \
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

    infer_avante)
        # Use --gear-checkpoint if provided, otherwise use default based on model type
        if [ -n "$GEAR_CHECKPOINT" ]; then
            AVANTE_CHECKPOINT="$GEAR_CHECKPOINT"
        else
            if [ "$MODEL_TYPE" = "onepiece" ]; then
                # No default checkpoint for Onepiece - require user to specify
                echo "ERROR: Onepiece model requires --gear-checkpoint <path_to_onepiece_checkpoint>"
                exit 1
            else
                # Default checkpoint for gear5
                AVANTE_CHECKPOINT="train_results/results_20/gear_5/large/best.pth"
            fi
        fi

        echo "Running ${MODEL_TYPE} inference on avante_images..."
        echo "Configuration:"
        echo "  - Model type: $MODEL_TYPE"
        echo "  - Input: /data/datasets/avante_images"
        echo "  - Output: $RESULTS_DIR"
        echo "  - GPU: $GPU_ID"
        echo "  - Checkpoint: $AVANTE_CHECKPOINT"
        echo "  - Config variant: $CONFIG_VARIANT"
        echo "  - Focal length: 900 px (de-canon ratio: 0.5556)"
        echo "  - Max depth: ${MAX_DEPTH}m"
        echo "  - Colormap: 2-98 percentile"
        if [ "$MODEL_TYPE" = "gear5" ]; then
            echo "  - Temporal backend: $([ "$MAMBA" = "true" ] && echo "Mamba2" || echo "GRU")"
        fi
        echo "  - CLS layers: $CLS_LAYERS"
        if [ "$MODEL_TYPE" = "onepiece" ]; then
            echo "  - No-shift: $NO_SHIFT"
        fi
        if [ -n "$SECTION" ]; then
            echo "  - Section: $SECTION (frames)"
        fi
        echo ""

        # Build infer_avante command
        # Note: focal-length 900 is default for avante_images (original size 1600x1100)
        INFER_CMD="python infer_avante.py \
            --input-dir /data/datasets/avante_images \
            --output-dir /app/$RESULTS_DIR \
            --checkpoint $AVANTE_CHECKPOINT \
            --config-variant $CONFIG_VARIANT \
            --gpu 0 \
            --max-depth $MAX_DEPTH \
            --focal-length 900.0 \
            --canonical-fx 500.0 \
            --cls-layers $CLS_LAYERS \
            --model-type $MODEL_TYPE"

        # Add mamba flag if requested (Gear5 only)
        if [ "$MAMBA" = "true" ]; then
            INFER_CMD="$INFER_CMD --mamba"
        fi

        # Add no-shift flag if requested (Onepiece scale-only mode)
        if [ "$NO_SHIFT" = "true" ]; then
            INFER_CMD="$INFER_CMD --no-shift"
        fi

        # Add section filter if specified
        if [ -n "$SECTION" ]; then
            INFER_CMD="$INFER_CMD --section $SECTION"
        fi

        # Add colorbar flag if requested
        if [ "$CBAR" = "true" ]; then
            INFER_CMD="$INFER_CMD --cbar"
        fi

        # Run inference
        CUDA_VISIBLE_DEVICES=$GPU_ID docker compose run --rm flashdepth $INFER_CMD
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
        echo "  - Resolution: $RESOLUTION"
        echo "  - Workers: $WORKERS"
        echo "  - Temporal backend: $([ "$MAMBA" = "true" ] && echo "Mamba2" || echo "GRU")"
        echo "  - CLS layers: $CLS_LAYERS"
        echo "  - TSP mode: $TSP_MODE"
        echo "  - No-inverse: $NO_INVERSE"
        echo "  - No-shift: $NO_SHIFT"
        if [ "$OBJWISE_FLAG" == "true" ]; then
            echo "  - Object-wise evaluation: ENABLED"
        fi
        if [ "$FGWISE_FLAG" == "true" ]; then
            echo "  - FG-wise evaluation: ENABLED"
        fi
        if [ -n "$OBJWISE_DATASET" ]; then
            echo "  - Dataset: $OBJWISE_DATASET"
        else
            echo "  - Dataset: Using config defaults (all test datasets)"
        fi
        if [ -n "$TEST_MODE" ]; then
            echo "  - Test mode: $TEST_MODE"
        fi
        echo ""

        # Build test_gear5 command with config variant support
        TEST_CMD="python test_gear5.py \
            --config-path configs/gear5 \
            --config-name config_$CONFIG_VARIANT \
            dataset.data_root=/data/datasets \
            model.use_mamba_temporal=$MAMBA \
            model.tsp_mode=$TSP_MODE \
            training.workers=$WORKERS \
            +no_inverse=$NO_INVERSE \
            +no_shift=$NO_SHIFT \
            +results_dir=$RESULTS_DIR \
            +gpu=$GPU_ID \
            +vid_len=$VID_LEN \
            +frame_interval=$FRAME_INTERVAL \
            +visualization=$VISUALIZATION \
            cls_layers='[$CLS_LAYERS]' \
            +config_dir=configs/gear5/$CONFIG_VARIANT"


        # Add --fgwise flag if requested (use + prefix to add new config key)
        if [ "$FGWISE_FLAG" == "true" ]; then
            TEST_CMD="$TEST_CMD +fg_wise.enabled=true"
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

        # Add seq_list if specified
        if [ -n "$SEQ" ]; then
            TEST_CMD="$TEST_CMD +seq_list='[$SEQ]'"
        fi

        # Add best-figure export if specified
        if [ "$BEST_FIGURE" == "true" ]; then
            TEST_CMD="$TEST_CMD +best_figure=true"
        fi

        # Add frame export if specified (--frame is stripped from sys.argv before Hydra)
        if [ -n "$FRAME" ]; then
            TEST_CMD="$TEST_CMD --frame $FRAME"
        fi

        # Add resolution override
        TEST_CMD="$TEST_CMD +resolution=$RESOLUTION"

        # Add checkpoint
        if [ -n "$FLASHDEPTH_CHECKPOINT" ]; then
            TEST_CMD="$TEST_CMD load=$FLASHDEPTH_CHECKPOINT"
        fi

        # Disable video (GIF) generation if --no-video flag is set
        if [ "$NO_VIDEO" == "true" ]; then
            TEST_CMD="$TEST_CMD eval.out_video=false"
        fi

        # Add test-mode if specified
        if [ -n "$TEST_MODE" ]; then
            TEST_CMD="$TEST_CMD --test-mode $TEST_MODE"
        fi

        CUDA_VISIBLE_DEVICES=$GPU_ID docker compose run --rm flashdepth $TEST_CMD
        ;;

    train_gear5_bankai)
        echo "Starting Gear5 Bankai training (Single GPU)..."
        echo "Configuration:"
        echo "  - Config variant: $CONFIG_VARIANT"
        echo "  - Bankai phase: $BANKAI_PHASE (auto step: $BANKAI_AUTO_STEP)"
        echo "  - TGM weight: $TGM_WEIGHT"
        echo "  - Log space: $USE_LOG_SPACE"
        echo "  - No-inverse: $NO_INVERSE"
        echo "  - No-shift: $NO_SHIFT"
        echo "  - CLS layers: $CLS_LAYERS"
        echo "  - Batch size: $BATCH_SIZE"
        echo "  - Workers: $WORKERS"
        echo "  - GPU: $GPU_ID"
        echo "  - Results directory: $RESULTS_DIR"
        echo ""

        # Build train_gear5 bankai command
        DOCKER_CMD="CUDA_VISIBLE_DEVICES=$GPU_ID docker compose run --rm -e WANDB_API_KEY=\${WANDB_API_KEY:-} flashdepth python train_gear5.py \
            --config-path configs/gear5 \
            --config-name config_$CONFIG_VARIANT \
            dataset.data_root=/data/datasets \
            training.batch_size=$BATCH_SIZE \
            training.workers=$WORKERS \
            training.iterations=$TOTAL_ITERS \
            training.wandb=$WANDB \
            use_bankai=true \
            bankai_phase=$BANKAI_PHASE \
            +bankai_auto_step=$BANKAI_AUTO_STEP \
            tgm_weight=$TGM_WEIGHT \
            +use_log_space=$USE_LOG_SPACE \
            +no_inverse=$NO_INVERSE \
            +no_shift=$NO_SHIFT \
            use_canonical_space=$USE_CANONICAL \
            cls_layers='[$CLS_LAYERS]' \
            +results_dir=$RESULTS_DIR"

        # Add gear_checkpoint if specified (for Phase 2)
        if [ -n "$GEAR_CHECKPOINT" ]; then
            DOCKER_CMD="$DOCKER_CMD load=$GEAR_CHECKPOINT"
        fi

        # Add wandb name if specified
        if [ -n "$WANDB_NAME" ]; then
            DOCKER_CMD="$DOCKER_CMD training.wandb_name=$WANDB_NAME"
        fi

        eval $DOCKER_CMD
        ;;

    train_gear5_bankai_ddp)
        # Auto-adjust for config variant
        if [ "$CONFIG_VARIANT" = "hybrid" ]; then
            ACTUAL_WORKERS=1
            RES_NAME="2k"
            VIDEO_LENGTH=2
            if [ -z "$USER_BATCH_SIZE" ]; then
                BATCH_SIZE=1
                echo "  NOTE: Auto-adjusted batch size to $BATCH_SIZE for Hybrid (2K resolution)"
            fi
        else
            ACTUAL_WORKERS=$WORKERS
            RES_NAME="base"
            VIDEO_LENGTH=5
        fi

        echo "Starting Gear5 Bankai training (Multi-GPU: $DDP_GPUS)..."
        echo "Configuration:"
        echo "  - Config variant: $CONFIG_VARIANT"
        echo "  - Bankai phase: $BANKAI_PHASE (auto step: $BANKAI_AUTO_STEP)"
        echo "  - TGM weight: $TGM_WEIGHT"
        echo "  - Log space: $USE_LOG_SPACE"
        echo "  - No-inverse: $NO_INVERSE"
        echo "  - No-shift: $NO_SHIFT"
        echo "  - CLS layers: $CLS_LAYERS"
        echo "  - Resolution: $RES_NAME"
        echo "  - Batch size per GPU: $BATCH_SIZE"
        echo "  - Effective batch size: $((BATCH_SIZE * 2))"
        echo "  - Workers per GPU: $ACTUAL_WORKERS"
        echo "  - Video length: $VIDEO_LENGTH frames"
        echo "  - Total iterations: $TOTAL_ITERS"
        echo "  - GPUs: $DDP_GPUS"
        echo "  - Results directory: $RESULTS_DIR"
        echo ""

        DOCKER_CMD="CUDA_VISIBLE_DEVICES=$DDP_GPUS docker compose run --rm \
            -e GLOO_SOCKET_IFNAME=eth0 \
            -e NCCL_SOCKET_IFNAME=eth0 \
            -e NCCL_P2P_DISABLE=1 \
            -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
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
            use_bankai=true \
            bankai_phase=$BANKAI_PHASE \
            +bankai_auto_step=$BANKAI_AUTO_STEP \
            tgm_weight=$TGM_WEIGHT \
            +use_log_space=$USE_LOG_SPACE \
            +no_inverse=$NO_INVERSE \
            +no_shift=$NO_SHIFT \
            use_canonical_space=$USE_CANONICAL \
            cls_layers='[$CLS_LAYERS]' \
            +results_dir=$RESULTS_DIR"

        # Add gear_checkpoint if specified (for Phase 2)
        if [ -n "$GEAR_CHECKPOINT" ]; then
            DOCKER_CMD="$DOCKER_CMD load=$GEAR_CHECKPOINT"
        fi

        # Add wandb name if specified
        if [ -n "$WANDB_NAME" ]; then
            DOCKER_CMD="$DOCKER_CMD training.wandb_name=$WANDB_NAME"
        fi

        eval $DOCKER_CMD
        ;;

    test_gear5_bankai)
        echo "Starting Gear5 Bankai testing..."
        echo "Configuration:"
        echo "  - Video length: $VID_LEN"
        echo "  - Frame interval: $FRAME_INTERVAL"
        echo "  - GPU: $GPU_ID"
        echo "  - Workers: $WORKERS"
        echo "  - Results directory: $RESULTS_DIR"
        echo "  - Checkpoint: $FLASHDEPTH_CHECKPOINT"
        echo "  - Config variant: $CONFIG_VARIANT"
        echo "  - Resolution: $RESOLUTION"
        echo "  - Visualization: $VISUALIZATION"
        echo "  - No-inverse: $NO_INVERSE"
        echo "  - No-shift: $NO_SHIFT"
        if [ -n "$OBJWISE_DATASET" ]; then
            echo "  - Dataset: $OBJWISE_DATASET"
        else
            echo "  - Dataset: Using config defaults (all test datasets)"
        fi
        if [ "$OBJWISE_FLAG" == "true" ]; then
            echo "  - Object-wise evaluation: ENABLED"
        fi
        if [ -n "$TEST_MODE" ]; then
            echo "  - Test mode: $TEST_MODE"
        fi
        echo ""

        # Build test_gear5 bankai command
        TEST_CMD="python test_gear5.py \
            --config-path configs/gear5 \
            --config-name config_$CONFIG_VARIANT \
            dataset.data_root=/data/datasets \
            model.use_mamba_temporal=$MAMBA \
            training.workers=$WORKERS \
            use_bankai=true \
            bankai_phase=$BANKAI_PHASE \
            +bankai_auto_step=$BANKAI_AUTO_STEP \
            +no_inverse=$NO_INVERSE \
            +no_shift=$NO_SHIFT \
            +results_dir=$RESULTS_DIR \
            +gpu=$GPU_ID \
            +vid_len=$VID_LEN \
            +frame_interval=$FRAME_INTERVAL \
            +visualization=$VISUALIZATION \
            cls_layers='[$CLS_LAYERS]' \
            +config_dir=configs/gear5/$CONFIG_VARIANT"


        # Add dataset override if specified
        if [ -n "$OBJWISE_DATASET" ]; then
            TEST_CMD="$TEST_CMD eval.test_datasets=[$OBJWISE_DATASET]"
            OBJWISE_DATASET_BASE="${OBJWISE_DATASET/_seg/}"
            TEST_CMD="$TEST_CMD object_wise.dataset=$OBJWISE_DATASET_BASE"
        fi

        # Add seq_list if specified
        if [ -n "$SEQ" ]; then
            TEST_CMD="$TEST_CMD +seq_list='[$SEQ]'"
        fi

        # Add best-figure export if specified
        if [ "$BEST_FIGURE" == "true" ]; then
            TEST_CMD="$TEST_CMD +best_figure=true"
        fi

        # Add frame export if specified (--frame is stripped from sys.argv before Hydra)
        if [ -n "$FRAME" ]; then
            TEST_CMD="$TEST_CMD --frame $FRAME"
        fi

        # Add resolution override
        TEST_CMD="$TEST_CMD +resolution=$RESOLUTION"

        # Add checkpoint
        if [ -n "$FLASHDEPTH_CHECKPOINT" ]; then
            TEST_CMD="$TEST_CMD load=$FLASHDEPTH_CHECKPOINT"
        fi

        # Disable video (GIF) generation if --no-video flag is set
        if [ "$NO_VIDEO" == "true" ]; then
            TEST_CMD="$TEST_CMD eval.out_video=false"
        fi

        # Add test-mode if specified
        if [ -n "$TEST_MODE" ]; then
            TEST_CMD="$TEST_CMD --test-mode $TEST_MODE"
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
            case "$CONFIG_VARIANT" in
                l)
                    CHECKPOINT="/app/configs/flashdepth-l/iter_10001.pth"
                    ;;
                s)
                    CHECKPOINT="/app/configs/flashdepth-s/iter_14001.pth"
                    ;;
                hybrid)
                    CHECKPOINT="/app/configs/flashdepth/iter_43002.pth"
                    ;;
                *)
                    echo "Unknown config variant: $CONFIG_VARIANT"
                    echo "Valid options: l, s, hybrid"
                    exit 1
                    ;;
            esac
        fi

        # Use OBJWISE_DATASET if set, otherwise default to nuscenes
        TEST_DATASET="${OBJWISE_DATASET:-nuscenes}"

        # Map CONFIG_VARIANT to FlashDepth config path
        case "$CONFIG_VARIANT" in
            l)
                FLASHDEPTH_CONFIG="flashdepth-l"
                ;;
            s)
                FLASHDEPTH_CONFIG="flashdepth-s"
                ;;
            hybrid)
                FLASHDEPTH_CONFIG="flashdepth"
                ;;
        esac

        # Use custom results dir if provided, otherwise default to test_results/${TEST_DATASET}_original_${CONFIG_VARIANT}
        # NOTE: FlashDepth automatically appends /${TEST_DATASET}/ to outfolder, so we need to handle this
        if [ "$RESULTS_DIR" = "train_results/results_1" ]; then
            # Default value not changed by user, use dataset and config specific default
            # FlashDepth will create: test_results/original_${CONFIG_VARIANT}/${TEST_DATASET}/
            OUTFOLDER="/app/test_results/original_${CONFIG_VARIANT}"
            LOCAL_OUTFOLDER="test_results/original_${CONFIG_VARIANT}"
            FINAL_RESULTS_DIR="${LOCAL_OUTFOLDER}/${TEST_DATASET}"
        else
            # User provided custom results dir
            # Remove trailing slash and dataset name if present to avoid duplication
            CLEAN_RESULTS_DIR="${RESULTS_DIR%/}"  # Remove trailing slash
            CLEAN_RESULTS_DIR="${CLEAN_RESULTS_DIR%/$TEST_DATASET}"  # Remove dataset suffix if present

            # If it's a relative path, prefix with /app/ to save to host
            if [[ "$CLEAN_RESULTS_DIR" != /* ]]; then
                OUTFOLDER="/app/$CLEAN_RESULTS_DIR"
            else
                OUTFOLDER="$CLEAN_RESULTS_DIR"
            fi
            LOCAL_OUTFOLDER="${OUTFOLDER#/app/}"
            FINAL_RESULTS_DIR="${LOCAL_OUTFOLDER}/${TEST_DATASET}"
        fi

        # Determine visualization settings based on NO_VIDEO flag
        # --no-video: Disable MP4 generation but still save .npy depth files for eval_aligned
        if [ "$NO_VIDEO" = "true" ]; then
            OUT_VIDEO="false"
            SAVE_DEPTH="true"  # Keep .npy saving for eval_aligned
            VIS_STATUS="LIMITED (no MP4, .npy depth files saved for eval_aligned)"
        else
            OUT_VIDEO="true"
            SAVE_DEPTH="true"
            VIS_STATUS="ENABLED (MP4 + .npy depth files)"
        fi

        echo "Testing Original FlashDepth (inference mode)..."
        echo "Configuration:"
        echo "  - Config variant: $CONFIG_VARIANT ($FLASHDEPTH_CONFIG)"
        echo "  - Dataset: $TEST_DATASET (all sequences)"
        echo "  - Resolution: $RESOLUTION"
        echo "  - GPU: $GPU_ID"
        echo "  - Checkpoint: $CHECKPOINT"
        echo "  - Results directory: $FINAL_RESULTS_DIR"
        echo "  - Inverse colormap: $INVERSE"
        echo "  - Visualization: $VIS_STATUS"
        echo "  - DataLoader workers: $WORKERS"
        if [ -n "$TEST_MODE" ]; then
            echo "  - Test mode: $TEST_MODE"
        fi
        echo "  - Log file: ${FINAL_RESULTS_DIR}/test.log"
        echo ""

        # Create output directory on host
        mkdir -p "$FINAL_RESULTS_DIR"

        # Use test_video_comparison.py for special test modes (e.g., tc)
        if [ -n "$TEST_MODE" ]; then
            echo "Running FlashDepth via test_video_comparison.py (--test-mode $TEST_MODE)..."

            DOCKER_CMD="CUDA_VISIBLE_DEVICES=$GPU_ID docker compose run --rm flashdepth python test_video_comparison.py \
                --method flashdepth \
                --dataset $TEST_DATASET \
                --data-root /data/datasets \
                --checkpoint $CHECKPOINT \
                --results-dir /app/$FINAL_RESULTS_DIR \
                --gpu 0 \
                --video-length $VID_LEN \
                --workers $WORKERS \
                --depth-mode metric \
                --test-mode $TEST_MODE"

            if [ -n "$LIMIT_SCENES" ]; then
                DOCKER_CMD="$DOCKER_CMD --limit-scenes $LIMIT_SCENES"
            fi

            if [ -n "$SEQ" ]; then
                DOCKER_CMD="$DOCKER_CMD --seq $SEQ"
            fi

            eval $DOCKER_CMD 2>&1 | tee "${FINAL_RESULTS_DIR}/test.log"

            echo ""
            echo "✓ Test complete! Results saved to:"
            echo "  - Temporal consistency: ${FINAL_RESULTS_DIR}/temporal_consistency.json"
            echo "  - Full log: ${FINAL_RESULTS_DIR}/test.log"
        else
            # Original train.py inference pipeline
            # Build base command
            TEST_CMD="cd /FlashDepth && torchrun --nproc_per_node=1 train.py \
              --config-path configs/$FLASHDEPTH_CONFIG \
              inference=true \
              eval.test_datasets=[$TEST_DATASET] \
              eval.metrics=true \
              dataset.data_root=/data/datasets \
              eval.outfolder=$OUTFOLDER \
              load=$CHECKPOINT \
              +eval.inverse=$INVERSE \
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
            CUDA_VISIBLE_DEVICES=$GPU_ID docker compose run --rm flashdepth bash -c "$TEST_CMD" 2>&1 | tee "${FINAL_RESULTS_DIR}/test.log"

            # Parse log and save as JSON
            echo ""
            echo "Parsing results to JSON..."
            python3 utils/parse_flashdepth_results.py "${FINAL_RESULTS_DIR}/test.log" "$FINAL_RESULTS_DIR"

            # Run scale/shift alignment evaluation (.npy files are always saved now)
            if [ -d "$FINAL_RESULTS_DIR" ]; then
                echo ""
                echo "Running scale/shift alignment evaluation..."

                # Build eval command with optional --seq and --frame
                EVAL_CMD="python3 scripts/eval_flashdepth_with_alignment.py \
                    --pred-dir \"$FINAL_RESULTS_DIR\" \
                    --dataset \"$TEST_DATASET\" \
                    --data-root /home/cvlab/hsy/Datasets \
                    --max-depth 70.0 \
                    --output-dir \"${FINAL_RESULTS_DIR}/eval_aligned\""

                if [ -n "$SEQ" ]; then
                    EVAL_CMD="$EVAL_CMD --seq \"$SEQ\""
                fi

                if [ -n "$FRAME" ]; then
                    EVAL_CMD="$EVAL_CMD --frame $FRAME"
                fi

                eval "$EVAL_CMD" 2>&1 | tee -a "${FINAL_RESULTS_DIR}/eval_aligned.log" || echo "Warning: Alignment evaluation failed (GT may not be available for this dataset)"
            else
                echo "Warning: Results directory not found: $FINAL_RESULTS_DIR"
            fi

            echo ""
            echo "✓ Test complete! Results saved to:"
            echo "  - FPS Summary: ${FINAL_RESULTS_DIR}/fps_results.json"
            echo "  - Per-sequence FPS: ${FINAL_RESULTS_DIR}/per_sequence_fps.json"
            echo "  - Full log: ${FINAL_RESULTS_DIR}/test.log"
            echo "  - Depth files (.npy): ${FINAL_RESULTS_DIR}/*/*.npy"
            echo "  - Aligned evaluation: ${FINAL_RESULTS_DIR}/eval_aligned/"
            echo "    - eval_results.json (overall metrics with scale/shift)"
            echo "    - per_sequence_scale_shift.json (per-sequence s,t values)"
            if [ "$NO_VIDEO" != "true" ]; then
                echo "  - MP4 videos: ${FINAL_RESULTS_DIR}/*/*.mp4"
            fi
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

    train_onepiece)
        echo "Starting Onepiece V3 training (Single GPU)..."
        echo "Configuration:"
        echo "  - Config variant: $CONFIG_VARIANT (--config-variant, default: l)"
        echo "  - Config file: configs/onepiece/config_$CONFIG_VARIANT.yaml"
        echo "  - Batch size: $BATCH_SIZE (--batch-size, default: 3)"
        echo "  - Workers: $WORKERS (--workers, default: 8)"
        echo "  - GPU: $GPU_ID (--gpu, default: 0)"
        echo "  - Total iterations: $TOTAL_ITERS (--epochs, default: 60001)"
        echo "  - WandB: $WANDB (--wandb, default: true)"
        echo "  - WandB name: ${WANDB_NAME:-auto} (--wandb-name)"
        echo "  - Checkpoint: ${FLASHDEPTH_CHECKPOINT:-config default} (--flashdepth-checkpoint)"
        echo "  - Results directory: $RESULTS_DIR (--results-dir)"
        echo ""

        DOCKER_CMD="CUDA_VISIBLE_DEVICES=$GPU_ID docker compose run --rm -e WANDB_API_KEY=\${WANDB_API_KEY:-} flashdepth python train_onepiece.py \
            --config-path configs/onepiece \
            --config-name config_$CONFIG_VARIANT \
            dataset.data_root=/data/datasets \
            training.batch_size=$BATCH_SIZE \
            training.workers=$WORKERS \
            training.iterations=$TOTAL_ITERS \
            training.wandb=$WANDB \
            +results_dir=$RESULTS_DIR"

        if [ -n "$FLASHDEPTH_CHECKPOINT" ]; then
            DOCKER_CMD="$DOCKER_CMD load=$FLASHDEPTH_CHECKPOINT"
        fi

        if [ -n "$WANDB_NAME" ]; then
            DOCKER_CMD="$DOCKER_CMD training.wandb_name=$WANDB_NAME"
        fi

        eval $DOCKER_CMD
        ;;

    train_onepiece_ddp)
        echo "Starting Onepiece V3 training (Multi-GPU DDP)..."
        echo "Configuration:"
        echo "  - Config variant: $CONFIG_VARIANT (--config-variant, default: l)"
        echo "  - Config file: configs/onepiece/config_$CONFIG_VARIANT.yaml"
        echo "  - Batch size per GPU: $BATCH_SIZE (--batch-size, default: 3)"
        echo "  - Effective batch size: $((BATCH_SIZE * 2))"
        echo "  - Workers: $WORKERS (--workers, default: 8)"
        echo "  - GPUs: $DDP_GPUS (--ddp-gpus, default: 0,1)"
        echo "  - Total iterations: $TOTAL_ITERS (--epochs, default: 60001)"
        echo "  - WandB: $WANDB (--wandb, default: true)"
        echo "  - WandB name: ${WANDB_NAME:-auto} (--wandb-name)"
        echo "  - Checkpoint: ${FLASHDEPTH_CHECKPOINT:-config default} (--flashdepth-checkpoint)"
        echo "  - Results directory: $RESULTS_DIR (--results-dir)"
        echo ""

        DOCKER_CMD="CUDA_VISIBLE_DEVICES=$DDP_GPUS docker compose run --rm \
            -e GLOO_SOCKET_IFNAME=eth0 \
            -e NCCL_SOCKET_IFNAME=eth0 \
            -e NCCL_P2P_DISABLE=1 \
            -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            -e WANDB_API_KEY=\${WANDB_API_KEY:-} \
            flashdepth torchrun \
            --standalone \
            --nproc_per_node=2 \
            train_onepiece.py \
            --config-path configs/onepiece \
            --config-name config_$CONFIG_VARIANT \
            dataset.data_root=/data/datasets \
            training.batch_size=$BATCH_SIZE \
            training.workers=$WORKERS \
            training.iterations=$TOTAL_ITERS \
            training.wandb=$WANDB \
            +results_dir=$RESULTS_DIR"

        if [ -n "$FLASHDEPTH_CHECKPOINT" ]; then
            DOCKER_CMD="$DOCKER_CMD load=$FLASHDEPTH_CHECKPOINT"
        fi

        if [ -n "$WANDB_NAME" ]; then
            DOCKER_CMD="$DOCKER_CMD training.wandb_name=$WANDB_NAME"
        fi

        eval $DOCKER_CMD
        ;;

    test_onepiece)
        # Determine test dataset (default: all datasets)
        ONEPIECE_TEST_DATASET="${OBJWISE_DATASET:-}"

        # === Dataset vid-len defaults for onepiece ===
        _onepiece_vid_len() {
            case "$1" in
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

        _onepiece_workers() {
            case "$1" in
                eth3d|unreal4k) echo 1 ;;
                waymo_seg|waymo) echo 2 ;;
                sintel|vkitti)  echo 4 ;;
                *)              echo 4 ;;
            esac
        }

        # === Handle --dataset all ===
        if [ "$ONEPIECE_TEST_DATASET" = "all" ]; then
            ALL_ONEPIECE_DATASETS="eth3d sintel waymo_seg vkitti unreal4k"

            # Check if user explicitly set --vid-len
            USER_VID_LEN_SET=false
            for arg in "${BASH_ARGV[@]}"; do
                if [ "$arg" = "--vid-len" ]; then
                    USER_VID_LEN_SET=true
                    break
                fi
            done

            # Max-depth suffix
            if [ "$MAX_DEPTH" != "80.0" ] && [ -n "$MAX_DEPTH" ]; then
                MD_SUFFIX="_${MAX_DEPTH}"
            else
                MD_SUFFIX=""
            fi

            # Base results dir from user (strip trailing dataset name if present)
            BASE_RESULTS_DIR="$RESULTS_DIR"

            echo "========================================"
            echo "Onepiece Batch Testing: ALL datasets"
            echo "========================================"
            echo "Datasets: $ALL_ONEPIECE_DATASETS"
            echo "GPU: $GPU_ID"
            echo "Base results dir: $BASE_RESULTS_DIR"
            echo "========================================"
            echo "Press Ctrl+C to abort all runs"
            echo ""

            BATCH_ABORT=false
            trap 'echo ""; echo "⚠️  Ctrl+C received — aborting batch..."; BATCH_ABORT=true' INT

            TOTAL_RUNS=0
            COMPLETED_RUNS=0
            FAILED_RUNS=0
            for ds in $ALL_ONEPIECE_DATASETS; do
                TOTAL_RUNS=$((TOTAL_RUNS + 1))
            done

            for ds in $ALL_ONEPIECE_DATASETS; do
                if [ "$BATCH_ABORT" = true ]; then
                    break
                fi

                COMPLETED_RUNS=$((COMPLETED_RUNS + 1))

                if [ "$USER_VID_LEN_SET" = false ]; then
                    CUR_VID_LEN=$(_onepiece_vid_len "$ds")
                else
                    CUR_VID_LEN=$VID_LEN
                fi
                CUR_WORKERS=$(_onepiece_workers "$ds")

                # Results dir: base_dir/../dataset_name (replace last path component)
                CUR_RESULTS_DIR="${BASE_RESULTS_DIR%/*}/${ds}${MD_SUFFIX}"

                echo ""
                echo "========================================"
                echo "[$COMPLETED_RUNS/$TOTAL_RUNS] Onepiece on $ds (vid-len=$CUR_VID_LEN, workers=$CUR_WORKERS)"
                echo "  Results: $CUR_RESULTS_DIR"
                echo "========================================"

                DOCKER_CMD="CUDA_VISIBLE_DEVICES=$GPU_ID docker compose run --rm flashdepth python test_onepiece.py \
                    --config-path configs/onepiece \
                    --config-name config_$CONFIG_VARIANT \
                    dataset.data_root=/data/datasets \
                    dataset.video_length=$CUR_VID_LEN \
                    training.workers=$CUR_WORKERS \
                    +frame_interval=$FRAME_INTERVAL \
                    +results_dir=$CUR_RESULTS_DIR \
                    eval.test_datasets=[$ds]"

                if [ -n "$GEAR_CHECKPOINT" ]; then
                    DOCKER_CMD="$DOCKER_CMD load=$GEAR_CHECKPOINT"
                fi
                if [ "$NO_VIDEO" == "true" ]; then
                    DOCKER_CMD="$DOCKER_CMD eval.out_video=false"
                fi
                if [ -n "$SEQ" ]; then
                    DOCKER_CMD="$DOCKER_CMD +seq_list='[$SEQ]'"
                fi
                if [ "$BEST_FIGURE" == "true" ]; then
                    DOCKER_CMD="$DOCKER_CMD +best_figure=true"
                fi
                if [ -n "$FRAME" ]; then
                    DOCKER_CMD="$DOCKER_CMD --frame $FRAME"
                fi
                if [ -n "$TEST_MODE" ]; then
                    DOCKER_CMD="$DOCKER_CMD --test-mode $TEST_MODE"
                fi
                if [ -n "$MAX_DEPTH" ]; then
                    DOCKER_CMD="$DOCKER_CMD --max-depth $MAX_DEPTH"
                fi

                if eval $DOCKER_CMD; then
                    echo "✅ [$COMPLETED_RUNS/$TOTAL_RUNS] Onepiece on $ds completed"
                else
                    echo "❌ [$COMPLETED_RUNS/$TOTAL_RUNS] Onepiece on $ds FAILED"
                    FAILED_RUNS=$((FAILED_RUNS + 1))
                fi
            done

            trap - INT

            echo ""
            echo "========================================"
            if [ "$BATCH_ABORT" = true ]; then
                echo "Batch ABORTED by user (Ctrl+C)"
            fi
            echo "Onepiece Batch Testing Summary"
            echo "  Total: $TOTAL_RUNS, Completed: $((TOTAL_RUNS - FAILED_RUNS)), Failed: $FAILED_RUNS"
            echo "========================================"
        else
            # === Single dataset mode (original behavior) ===
            echo "Starting Onepiece testing..."
            echo "Configuration:"
            echo "  - Config variant: $CONFIG_VARIANT (--config-variant, default: l)"
            echo "  - Config file: configs/onepiece/config_$CONFIG_VARIANT.yaml"
            echo "  - GPU: $GPU_ID (--gpu, default: 0)"
            echo "  - Video length: $VID_LEN (--vid-len, default: 50)"
            echo "  - Frame interval: $FRAME_INTERVAL (--frame-interval, default: 1)"
            echo "  - Checkpoint: ${GEAR_CHECKPOINT:-config default} (--gear-checkpoint)"
            echo "  - Results directory: $RESULTS_DIR (--results-dir)"
            echo "  - Dataset: ${ONEPIECE_TEST_DATASET:-all (default)}"
            echo "  - No-video: $NO_VIDEO (--no-video, default: false)"
            if [ -n "$TEST_MODE" ]; then
                echo "  - Test mode: $TEST_MODE"
            fi
            echo ""

            DOCKER_CMD="CUDA_VISIBLE_DEVICES=$GPU_ID docker compose run --rm flashdepth python test_onepiece.py \
                --config-path configs/onepiece \
                --config-name config_$CONFIG_VARIANT \
                dataset.data_root=/data/datasets \
                dataset.video_length=$VID_LEN \
                +frame_interval=$FRAME_INTERVAL \
                +results_dir=$RESULTS_DIR"

            if [ -n "$GEAR_CHECKPOINT" ]; then
                DOCKER_CMD="$DOCKER_CMD load=$GEAR_CHECKPOINT"
            fi

            # Filter to specific dataset if --dataset is provided
            if [ -n "$ONEPIECE_TEST_DATASET" ]; then
                DOCKER_CMD="$DOCKER_CMD eval.test_datasets=[$ONEPIECE_TEST_DATASET]"
            fi

            # Disable video (GIF) generation if --no-video flag is set
            if [ "$NO_VIDEO" == "true" ]; then
                DOCKER_CMD="$DOCKER_CMD eval.out_video=false"
            fi

            # Add seq_list if specified (e.g., --seq 0,4)
            if [ -n "$SEQ" ]; then
                DOCKER_CMD="$DOCKER_CMD +seq_list='[$SEQ]'"
            fi

            # Add best_figure export if specified (--best-figure)
            if [ "$BEST_FIGURE" == "true" ]; then
                DOCKER_CMD="$DOCKER_CMD +best_figure=true"
            fi

            # Add frame export if specified (--frame N or --frame N,M)
            if [ -n "$FRAME" ]; then
                DOCKER_CMD="$DOCKER_CMD --frame $FRAME"
            fi

            # Add test-mode if specified
            if [ -n "$TEST_MODE" ]; then
                DOCKER_CMD="$DOCKER_CMD --test-mode $TEST_MODE"
            fi

            # Add max-depth if specified
            if [ -n "$MAX_DEPTH" ]; then
                DOCKER_CMD="$DOCKER_CMD --max-depth $MAX_DEPTH"
            fi

            eval $DOCKER_CMD
        fi
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