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
    echo "  train_gear5     Start Gear5 training - Two-stage Global + FG modulation"
    echo "  train_gear5_ddp Start Gear5 training with 2 GPUs (GPU 0,1)"
    echo "  test_gear5      Start Gear5 testing"
    echo "  train_onepiece       Start Onepiece training (Unified Global Mamba, single GPU)"
    echo "  train_onepiece_ddp   Start Onepiece training with 2 GPUs (DDP)"
    echo "  train_onepiece_fsdp  Start Onepiece FSDP2 training with 2 GPUs (2K hybrid)"
    echo "  test_onepiece        Start Onepiece testing"
    echo "  infer_avante         Run inference on avante_images (custom dataset)"
    echo "  test_original_flashdepth  Test original FlashDepth (without Gear modules) for comparison"
    echo "  shell       Start interactive shell in container"
    echo "  clean       Remove containers and images"
    echo "  logs        Show container logs"
    echo ""
    echo "Training Options:"
    echo "  --batch-size SIZE     Set batch size per GPU (default: 3)"
    echo "  --workers NUM         Set number of DataLoader workers (default: 8)"
    echo "  --epochs NUM          Set number of training iterations (default: 60001)"
    echo "  --gpu ID              Set GPU ID (default: 0)"
    echo "  --results-dir PATH    Set results directory (default: train_results/results_1)"
    echo "  --flashdepth-checkpoint PATH  Set FlashDepth pretrained weights path"
    echo "  --gear-checkpoint PATH  Set checkpoint path (required for hybrid/onepiece)"
    echo "  --frame-interval NUM  Set frame interval for sequence visualization (default: 1)"
    echo "  --vid-len NUM         Set video sequence length for testing (default: 50)"
    echo "  --config-variant VARIANT  Set config variant: l, s, hybrid (default: l)"
    echo "  --dataset DATASET     Set evaluation dataset (e.g., waymo, all)"
    echo "  --resolution MODE    Set resolution mode: base (518x518), 2k (1918x1078) (default: base)"
    echo "  --config VARIANT     Set FlashDepth config: flashdepth, flashdepth-l, flashdepth-s (default: flashdepth-l)"
    echo "  --inverse BOOL       Inverse colormap for depth (original FlashDepth only, default: false)"
    echo "  --no-video           Skip video (GIF/MP4) generation for faster testing"
    echo "  --whole-seq-test BOOL    Use all sequences in dataset (default: false)"
    echo "  --canon BOOL         Use canonical focal length normalization (default: true)"
    echo "  --loss TYPE          Set loss type: log_l1 (default), importance"
    echo "  --visualization BOOL Enable/disable visualizations (default: true)"
    echo "  --wandb BOOL         Enable/disable WandB logging (default: true)"
    echo "  --wandb-name NAME    Set WandB experiment name (default: auto-generated)"
    echo "  --mamba              Use Mamba2 instead of GRU for TemporalScalePredictor (default: false/GRU)"
    echo "  --cls-layer LAYERS   Select CLS token extraction layers (default: '2,4')"
    echo "  --tsp-mode MODE      TSP embed_dim mode: auto (default), l (1024-dim), s (384-dim)"
    echo "  --seq N              Sequence selection (e.g., --seq 0,4)"
    echo "  --limit-scenes N     Limit the number of scenes to process"
    echo "  --best-figure        Export best_frame ±4 frames as individual images/depth maps"
    echo "  --frame N            Export frame N ±4 frames as individual images/depth maps"
    echo "  --section START,END  Frame section for infer_avante (e.g., --section 450,480)"
    echo "  --no-inverse          Apply scale/shift in depth space instead of inverse depth space"
    echo "  --no-shift            Disable shift (scale-only mode)"
    echo "  --max-depth METERS   Max valid depth threshold (default: 80.0)"
    echo "  --model-type TYPE    Model type for infer_avante: gear5 (default), onepiece"
    echo "  --cbar               Show colorbar next to depth visualization"
    echo "  --test-mode MODE     Test mode: empty (full), tc (temporal consistency only)"
    echo "  --ddp-gpus IDS       GPU IDs for DDP/FSDP training (default DDP: 0,1 / default FSDP: 2,3)
  --teacher-checkpoint PATH  Teacher model checkpoint for FSDP hybrid training (Onepiece-L or FlashDepth-L)"
    echo "  --save-depth-maps    Save depth maps as .npy files"
    echo "  --fgwise             Enable FG-wise evaluation using ViT attention masks"
    echo "  --student-cls        Use student (ViT-S) CLS for Hybrid Onepiece instead of teacher (ViT-L) CLS"
    echo "  --measure-fps BOOL   Enable/disable FPS measurement (default: true)"
    echo ""
    echo "Examples:"
    echo "  $0 build                              # Build the image"
    echo "  $0 train_gear5 --gpu 0                # Gear5 training with GRU (default)"
    echo "  $0 train_gear5 --mamba --gpu 0         # Gear5 training with Mamba2"
    echo "  $0 test_gear5 --gpu 0                  # Test Gear5"
    echo "  $0 train_onepiece --gpu 0              # Onepiece training (single GPU)"
    echo "  $0 train_onepiece_ddp --ddp-gpus 0,1   # Onepiece DDP training
  $0 train_onepiece_fsdp                           # Onepiece FSDP2 hybrid 2K training (GPUs 2,3)
  $0 train_onepiece_fsdp --ddp-gpus 0,2           # Use GPU 0 and 2
  $0 train_onepiece_fsdp --flashdepth-checkpoint train_results/results_s/best.pth --teacher-checkpoint train_results/results_l/best.pth  # Load from Onepiece-S+L"
    echo "  $0 test_onepiece --dataset all --gpu 0 # Test Onepiece on all datasets"
    echo "  $0 infer_avante --model-type onepiece --gear-checkpoint path/to/ckpt  # Avante inference"
    echo "  $0 test_original_flashdepth --gpu 0    # Test original FlashDepth (ViT-L)"
    echo "  $0 test_original_flashdepth --dataset all --max-depth 80 --gpu 0  # Batch test all datasets"
    echo "  $0 shell                               # Interactive development"
}

# Parse command line arguments - optimized for RTX A6000 (2x 48GB)
COMMAND=""
BATCH_SIZE=3
WORKERS=8      # Optimized for 96 CPU cores, prevents I/O bottleneck
TOTAL_ITERS=60001
GPU_ID=0
RESULTS_DIR="train_results/results_1"
FLASHDEPTH_CHECKPOINT=""
GEAR_CHECKPOINT=""  # Checkpoint path (required for hybrid/onepiece)
FRAME_INTERVAL=1
VID_LEN=50
SINGLE_SEQUENCE=""  # Path to single sequence directory (optional)
MEASURE_FPS="true"
CONFIG_VARIANT="l"  # Config variant: l, s, hybrid
USER_CONFIG_VARIANT=""  # Track if user explicitly set --config-variant
CONFIG="flashdepth-l"  # FlashDepth config variant (flashdepth, flashdepth-l, flashdepth-s)
INVERSE="false"  # Inverse colormap for depth visualization (original FlashDepth only)
OBJWISE_DATASET=""  # Dataset for evaluation - empty means use config default
RESOLUTION="base"  # Resolution mode for testing (base, 2k)
NO_VIDEO="false"  # Skip video (GIF/MP4) generation for faster testing
WHOLE_SEQ_TEST="false"  # Use all sequences in dataset
USE_CANONICAL="true"  # Use canonical focal length normalization
LOSS_TYPE="log_l1"  # Loss type: log_l1 (default), importance
VISUALIZATION="true"  # Enable visualizations by default
WANDB="true"  # Enable WandB logging by default
WANDB_NAME=""  # WandB experiment name (empty = auto-generated)
MAMBA="false"  # Use Mamba2 for TemporalScalePredictor (false=GRU, true=Mamba2)
CLS_LAYERS="2,4"  # CLS token extraction layers
TSP_MODE="auto"  # TSP embed_dim mode: auto, l (1024), s (384)
SEQ=""  # Sequence selection
LIMIT_SCENES=""  # Limit number of scenes (optional)
BEST_FIGURE="false"  # Export best_frame ±4 frames as individual images/depth maps
FRAME=""  # Specific frame to export ±4 frames
FGWISE_FLAG="false"  # Enable FG-wise evaluation using ViT attention masks
SECTION=""  # Frame section for infer_avante (e.g., "450,480")
MAX_DEPTH="80.0"  # Max valid depth threshold (meters)
CBAR="false"  # Show colorbar next to depth visualization
TEST_MODE=""  # Test mode: empty (full), tc (temporal consistency only)
DDP_GPUS="0,1"  # GPU IDs for DDP training
USER_DDP_GPUS=""  # Track if user explicitly set --ddp-gpus (for FSDP default override)
NO_INVERSE="false"  # Apply scale/shift in depth space instead of inverse depth
NO_SHIFT="false"  # Disable shift, scale-only mode
MODEL_TYPE="gear5"  # Model type for infer_avante: gear5, onepiece
SAVE_DEPTH_MAPS="false"  # Save depth maps as .npy files
TEACHER_CHECKPOINT=""  # Teacher model checkpoint for FSDP hybrid training
USE_TEACHER_CLS="true"  # Use teacher (ViT-L) CLS for Hybrid Onepiece (false = student ViT-S CLS)

# Parse arguments
USER_BATCH_SIZE=""  # Track if user explicitly set batch size
while [[ $# -gt 0 ]]; do
    case $1 in
        build|train_gear5|train_gear5_ddp|test_gear5|train_onepiece|train_onepiece_ddp|train_onepiece_fsdp|test_onepiece|infer_avante|test_original_flashdepth|shell|clean|logs)
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
            USER_CONFIG_VARIANT="$2"
            shift 2
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
        --no-inverse)
            NO_INVERSE="true"
            shift
            ;;
        --no-shift)
            NO_SHIFT="true"
            shift
            ;;
        --save-depth-maps)
            SAVE_DEPTH_MAPS="true"
            shift
            ;;
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --ddp-gpus)
            DDP_GPUS="$2"
            USER_DDP_GPUS="$2"
            shift 2
            ;;
        --teacher-checkpoint)
            TEACHER_CHECKPOINT="$2"
            shift 2
            ;;
        --student-cls)
            USE_TEACHER_CLS="false"
            shift
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
            ACTUAL_WORKERS=1
            CANONICAL_FX="500.0"
            RES_NAME="2k"
            VIDEO_LENGTH=2

            if [ -z "$USER_BATCH_SIZE" ]; then
                BATCH_SIZE=1
                echo "  NOTE: Auto-adjusted batch size to $BATCH_SIZE for Hybrid (2K resolution)"
            else
                echo "  WARNING: Using user-specified batch size $BATCH_SIZE - may cause OOM on 2K resolution!"
            fi
        else
            ACTUAL_WORKERS=$WORKERS
            CANONICAL_FX="500.0"
            RES_NAME="base"
            VIDEO_LENGTH=5
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
                echo "ERROR: Onepiece model requires --gear-checkpoint <path_to_onepiece_checkpoint>"
                exit 1
            else
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

        if [ "$MAMBA" = "true" ]; then
            INFER_CMD="$INFER_CMD --mamba"
        fi

        if [ "$NO_SHIFT" = "true" ]; then
            INFER_CMD="$INFER_CMD --no-shift"
        fi

        if [ -n "$SECTION" ]; then
            INFER_CMD="$INFER_CMD --section $SECTION"
        fi

        if [ "$CBAR" = "true" ]; then
            INFER_CMD="$INFER_CMD --cbar"
        fi

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

        if [ "$FGWISE_FLAG" == "true" ]; then
            TEST_CMD="$TEST_CMD +fg_wise.enabled=true"
        fi

        if [ -n "$OBJWISE_DATASET" ]; then
            TEST_CMD="$TEST_CMD eval.test_datasets=[$OBJWISE_DATASET]"
            OBJWISE_DATASET_BASE="${OBJWISE_DATASET/_seg/}"
            TEST_CMD="$TEST_CMD object_wise.dataset=$OBJWISE_DATASET_BASE"
        fi

        if [ -n "$LIMIT_SCENES" ]; then
            TEST_CMD="$TEST_CMD +dataset.limit_scenes=$LIMIT_SCENES"
        fi

        if [ -n "$SEQ" ]; then
            TEST_CMD="$TEST_CMD +seq_list='[$SEQ]'"
        fi

        if [ "$BEST_FIGURE" == "true" ]; then
            TEST_CMD="$TEST_CMD +best_figure=true"
        fi

        if [ -n "$FRAME" ]; then
            TEST_CMD="$TEST_CMD --frame $FRAME"
        fi

        TEST_CMD="$TEST_CMD +resolution=$RESOLUTION"

        if [ -n "$FLASHDEPTH_CHECKPOINT" ]; then
            TEST_CMD="$TEST_CMD load=$FLASHDEPTH_CHECKPOINT"
        fi

        if [ "$NO_VIDEO" == "true" ]; then
            TEST_CMD="$TEST_CMD eval.out_video=false"
        fi

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

        # Determine max-depth for eval_aligned (default: 80.0)
        EVAL_MAX_DEPTH="${MAX_DEPTH:-80.0}"

        # Map CONFIG_VARIANT to human-readable name for results dir
        case "$CONFIG_VARIANT" in
            l)      CONFIG_DIR_NAME="large" ;;
            s)      CONFIG_DIR_NAME="small" ;;
            hybrid) CONFIG_DIR_NAME="hybrid" ;;
        esac

        # Max-depth suffix for results dir (remove trailing .0 for clean paths)
        MD_DISPLAY=$(echo "$EVAL_MAX_DEPTH" | sed 's/\.0$//')
        MD_SUFFIX="_${MD_DISPLAY}"

        # Determine visualization settings based on NO_VIDEO flag
        if [ "$NO_VIDEO" = "true" ]; then
            OUT_VIDEO="false"
            OUT_MP4="false"
            SAVE_DEPTH="true"
            VIS_STATUS="LIMITED (no MP4, .npy depth files saved for eval_aligned)"
        else
            OUT_VIDEO="true"
            OUT_MP4="true"
            SAVE_DEPTH="true"
            VIS_STATUS="ENABLED (MP4 + .npy depth files)"
        fi

        # === Dataset-specific vid-len and workers defaults ===
        _original_vid_len() {
            case "$1" in
                eth3d)       echo 30 ;;
                sintel)      echo 50 ;;
                waymo_seg|waymo) echo 200 ;;
                vkitti)      echo 200 ;;
                unreal4k)    echo 500 ;;
                *)           echo 50 ;;
            esac
        }

        _original_workers() {
            case "$1" in
                eth3d|unreal4k) echo 1 ;;
                waymo_seg|waymo) echo 2 ;;
                sintel|vkitti)  echo 4 ;;
                *)              echo 4 ;;
            esac
        }

        # === Helper function: run single dataset for original FlashDepth ===
        _run_original_flashdepth_single() {
            local CUR_DATASET="$1"
            local CUR_VID_LEN="$2"
            local CUR_WORKERS="$3"

            local CUR_DIR_NAME="${CUR_DATASET}${MD_SUFFIX}"
            if [ "$RESULTS_DIR" = "train_results/results_1" ]; then
                local CUR_OUTFOLDER="/app/test_results/original/${CONFIG_DIR_NAME}/${CUR_DIR_NAME}"
                local CUR_LOCAL_OUTFOLDER="test_results/original/${CONFIG_DIR_NAME}/${CUR_DIR_NAME}"
            else
                local CLEAN_DIR="${RESULTS_DIR%/}"
                if [[ "$CLEAN_DIR" != /* ]]; then
                    local CUR_OUTFOLDER="/app/$CLEAN_DIR"
                else
                    local CUR_OUTFOLDER="$CLEAN_DIR"
                fi
                local CUR_LOCAL_OUTFOLDER="${CUR_OUTFOLDER#/app/}"
            fi
            local CUR_FINAL_DIR="$CUR_LOCAL_OUTFOLDER"

            mkdir -p "$CUR_FINAL_DIR"

            echo "  - Dataset: $CUR_DATASET"
            echo "  - Video length: $CUR_VID_LEN"
            echo "  - Workers: $CUR_WORKERS"
            echo "  - Results: $CUR_FINAL_DIR"

            # Use test_video_comparison.py for special test modes (e.g., tc)
            if [ -n "$TEST_MODE" ]; then
                echo "Running FlashDepth via test_video_comparison.py (--test-mode $TEST_MODE)..."

                DOCKER_CMD="CUDA_VISIBLE_DEVICES=$GPU_ID docker compose run --rm flashdepth python test_video_comparison.py \
                    --method flashdepth \
                    --dataset $CUR_DATASET \
                    --data-root /data/datasets \
                    --checkpoint $CHECKPOINT \
                    --results-dir /app/$CUR_FINAL_DIR \
                    --gpu 0 \
                    --video-length $CUR_VID_LEN \
                    --workers $CUR_WORKERS \
                    --depth-mode relative \
                    --max-depth $EVAL_MAX_DEPTH \
                    --test-mode $TEST_MODE"

                if [ -n "$LIMIT_SCENES" ]; then
                    DOCKER_CMD="$DOCKER_CMD --limit-scenes $LIMIT_SCENES"
                fi

                if [ -n "$SEQ" ]; then
                    DOCKER_CMD="$DOCKER_CMD --seq $SEQ"
                fi

                eval $DOCKER_CMD 2>&1 | tee "${CUR_FINAL_DIR}/test.log"

                echo ""
                echo "✓ Test complete! Results saved to:"
                echo "  - Temporal consistency: ${CUR_FINAL_DIR}/temporal_consistency.json"
                echo "  - TC summary: ${CUR_FINAL_DIR}/tc_summary.json"
                echo "  - Full log: ${CUR_FINAL_DIR}/test.log"
            else
                # Original train.py inference pipeline
                TEST_CMD="cd /FlashDepth && torchrun --nproc_per_node=1 train.py \
                  --config-path configs/$FLASHDEPTH_CONFIG \
                  inference=true \
                  eval.test_datasets=[$CUR_DATASET] \
                  eval.metrics=true \
                  dataset.data_root=/data/datasets \
                  dataset.video_length=$CUR_VID_LEN \
                  eval.outfolder=$CUR_OUTFOLDER \
                  load=$CHECKPOINT \
                  +eval.inverse=$INVERSE \
                  eval.compile=false \
                  eval.out_video=$OUT_VIDEO \
                  eval.out_mp4=$OUT_MP4 \
                  eval.save_depth_npy=$SAVE_DEPTH \
                  ++eval.num_workers=$CUR_WORKERS \
                  eval.test_dataset_resolution=$RESOLUTION"

                if [ -n "$LIMIT_SCENES" ]; then
                    TEST_CMD="$TEST_CMD \
                  +eval.limit_scenes=$LIMIT_SCENES"
                fi

                echo "Running FlashDepth inference on $CUR_DATASET..."
                CUDA_VISIBLE_DEVICES=$GPU_ID docker compose run --rm flashdepth bash -c "$TEST_CMD" 2>&1 | tee "${CUR_FINAL_DIR}/test.log"

                # train.py auto-appends /{dataset}/ to outfolder — flatten it
                local NESTED_DIR="${CUR_FINAL_DIR}/${CUR_DATASET}"
                if [ -d "$NESTED_DIR" ]; then
                    mv "$NESTED_DIR"/* "$CUR_FINAL_DIR"/ 2>/dev/null || true
                    rmdir "$NESTED_DIR" 2>/dev/null || true
                fi

                echo ""
                echo "Parsing results to JSON..."
                python3 utils/parse_flashdepth_results.py "${CUR_FINAL_DIR}/test.log" "$CUR_FINAL_DIR"

                # Run scale/shift alignment evaluation
                if [ -d "$CUR_FINAL_DIR" ]; then
                    echo ""
                    echo "Running scale/shift alignment evaluation (max-depth=${EVAL_MAX_DEPTH})..."

                    EVAL_CMD="python3 scripts/eval_flashdepth_with_alignment.py \
                        --pred-dir \"$CUR_FINAL_DIR\" \
                        --dataset \"$CUR_DATASET\" \
                        --data-root /data/datasets \
                        --max-depth $EVAL_MAX_DEPTH \
                        --output-dir \"${CUR_FINAL_DIR}/eval_aligned\""

                    if [ -n "$SEQ" ]; then
                        EVAL_CMD="$EVAL_CMD --seq \"$SEQ\""
                    fi

                    if [ -n "$FRAME" ]; then
                        EVAL_CMD="$EVAL_CMD --frame $FRAME"
                    fi

                    CUDA_VISIBLE_DEVICES=$GPU_ID docker compose run --rm flashdepth bash -c "$EVAL_CMD" 2>&1 | tee -a "${CUR_FINAL_DIR}/eval_aligned.log" || echo "Warning: Alignment evaluation failed (GT may not be available for this dataset)"
                else
                    echo "Warning: Results directory not found: $CUR_FINAL_DIR"
                fi

                echo ""
                echo "✓ Test on $CUR_DATASET complete! Results saved to:"
                echo "  - FPS Summary: ${CUR_FINAL_DIR}/fps_results.json"
                echo "  - Per-sequence FPS: ${CUR_FINAL_DIR}/per_sequence_fps.json"
                echo "  - Full log: ${CUR_FINAL_DIR}/test.log"
                echo "  - Depth files (.npy): ${CUR_FINAL_DIR}/*/*.npy"
                echo "  - Aligned evaluation: ${CUR_FINAL_DIR}/eval_aligned/"
                echo "    - eval_results.json (overall metrics with scale/shift)"
                echo "    - per_sequence_scale_shift.json (per-sequence s,t values)"
                if [ "$NO_VIDEO" != "true" ]; then
                    echo "  - MP4 videos: ${CUR_FINAL_DIR}/*/*.mp4"
                fi
            fi
        }

        # Check if user explicitly set --vid-len
        USER_VID_LEN_SET=false
        for arg in "${BASH_ARGV[@]}"; do
            if [ "$arg" = "--vid-len" ]; then
                USER_VID_LEN_SET=true
                break
            fi
        done

        # Check if user explicitly set --workers
        USER_WORKERS_SET=false
        for arg in "${BASH_ARGV[@]}"; do
            if [ "$arg" = "--workers" ]; then
                USER_WORKERS_SET=true
                break
            fi
        done

        # === Handle --dataset all ===
        if [ "$TEST_DATASET" = "all" ]; then
            ALL_FLASHDEPTH_DATASETS="eth3d sintel waymo_seg vkitti unreal4k"

            echo "========================================"
            echo "Original FlashDepth Batch Testing: ALL datasets"
            echo "========================================"
            echo "Datasets: $ALL_FLASHDEPTH_DATASETS"
            echo "Config variant: $CONFIG_VARIANT ($FLASHDEPTH_CONFIG)"
            echo "GPU: $GPU_ID"
            echo "Checkpoint: $CHECKPOINT"
            echo "Max depth: $EVAL_MAX_DEPTH"
            echo "Resolution: $RESOLUTION"
            echo "Visualization: $VIS_STATUS"
            echo "Results base: test_results/original/${CONFIG_DIR_NAME}/"
            echo "========================================"
            echo "Press Ctrl+C to abort all runs"
            echo ""

            BATCH_ABORT=false
            trap 'echo ""; echo "⚠️  Ctrl+C received — aborting..."; exit 130' INT

            TOTAL_RUNS=0
            COMPLETED_RUNS=0
            FAILED_RUNS=0
            for ds in $ALL_FLASHDEPTH_DATASETS; do
                TOTAL_RUNS=$((TOTAL_RUNS + 1))
            done

            for ds in $ALL_FLASHDEPTH_DATASETS; do
                if [ "$BATCH_ABORT" = true ]; then
                    break
                fi

                COMPLETED_RUNS=$((COMPLETED_RUNS + 1))

                if [ "$USER_VID_LEN_SET" = false ]; then
                    CUR_VID_LEN=$(_original_vid_len "$ds")
                else
                    CUR_VID_LEN=$VID_LEN
                fi
                if [ "$USER_WORKERS_SET" = false ]; then
                    CUR_WORKERS=$(_original_workers "$ds")
                else
                    CUR_WORKERS=$WORKERS
                fi

                echo ""
                echo "========================================"
                echo "[$COMPLETED_RUNS/$TOTAL_RUNS] Original FlashDepth on $ds (vid-len=$CUR_VID_LEN, workers=$CUR_WORKERS)"
                echo "========================================"

                if _run_original_flashdepth_single "$ds" "$CUR_VID_LEN" "$CUR_WORKERS"; then
                    echo "✅ [$COMPLETED_RUNS/$TOTAL_RUNS] Original FlashDepth on $ds completed"
                else
                    echo "❌ [$COMPLETED_RUNS/$TOTAL_RUNS] Original FlashDepth on $ds FAILED"
                    FAILED_RUNS=$((FAILED_RUNS + 1))
                fi
            done

            trap - INT

            echo ""
            echo "========================================"
            if [ "$BATCH_ABORT" = true ]; then
                echo "Batch ABORTED by user (Ctrl+C)"
            fi
            echo "Original FlashDepth Batch Testing Summary"
            echo "  Total: $TOTAL_RUNS, Completed: $((TOTAL_RUNS - FAILED_RUNS)), Failed: $FAILED_RUNS"
            echo "========================================"
        else
            # === Single dataset mode ===
            if [ "$USER_VID_LEN_SET" = false ]; then
                CUR_VID_LEN=$(_original_vid_len "$TEST_DATASET")
            else
                CUR_VID_LEN=$VID_LEN
            fi
            if [ "$USER_WORKERS_SET" = false ]; then
                CUR_WORKERS=$(_original_workers "$TEST_DATASET")
            else
                CUR_WORKERS=$WORKERS
            fi

            echo "Testing Original FlashDepth (inference mode)..."
            echo "Configuration:"
            echo "  - Config variant: $CONFIG_VARIANT ($FLASHDEPTH_CONFIG)"
            echo "  - Resolution: $RESOLUTION"
            echo "  - GPU: $GPU_ID"
            echo "  - Checkpoint: $CHECKPOINT"
            echo "  - Max depth: $EVAL_MAX_DEPTH"
            echo "  - Inverse colormap: $INVERSE"
            echo "  - Visualization: $VIS_STATUS"
            if [ -n "$TEST_MODE" ]; then
                echo "  - Test mode: $TEST_MODE"
            fi
            echo ""

            _run_original_flashdepth_single "$TEST_DATASET" "$CUR_VID_LEN" "$CUR_WORKERS"
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
            model.use_teacher_cls=$USE_TEACHER_CLS \
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
            model.use_teacher_cls=$USE_TEACHER_CLS \
            +results_dir=$RESULTS_DIR"

        if [ -n "$FLASHDEPTH_CHECKPOINT" ]; then
            DOCKER_CMD="$DOCKER_CMD load=$FLASHDEPTH_CHECKPOINT"
        fi

        if [ -n "$WANDB_NAME" ]; then
            DOCKER_CMD="$DOCKER_CMD training.wandb_name=$WANDB_NAME"
        fi

        eval $DOCKER_CMD
        ;;

    train_onepiece_fsdp)
        # FSDP2 Onepiece training — uses 2 GPUs with torchrun
        # Default: config_hybrid_fsdp (2K hybrid).
        # Smoke test (non-hybrid 518): pass --config-variant l explicitly.
        if [ -n "$USER_CONFIG_VARIANT" ] && [ "$CONFIG_VARIANT" = "l" ]; then
            FSDP_CONFIG="config_fsdp"          # explicit smoke-test: non-hybrid 518
        else
            FSDP_CONFIG="config_hybrid_fsdp"   # default: 2K hybrid
        fi

        # GPU selection: use --ddp-gpus if specified, otherwise default to 2,3
        FSDP_GPUS="${USER_DDP_GPUS:-2,3}"

        # Auto-adjust batch size for 2K if not explicitly set
        if [ -z "$USER_BATCH_SIZE" ] && [ "$FSDP_CONFIG" = "config_hybrid_fsdp" ]; then
            BATCH_SIZE=1
            echo "  NOTE: Auto-set batch_size=1 for 2K hybrid FSDP (use --batch-size 2 to override)"
        fi

        echo "Starting Onepiece V3 FSDP2 training (2 GPUs, torchrun)..."
        echo "Configuration:"
        echo "  - Config file: configs/onepiece/$FSDP_CONFIG.yaml"
        echo "  - Batch size per GPU: $BATCH_SIZE"
        echo "  - Workers: $WORKERS"
        echo "  - GPUs: $FSDP_GPUS (--ddp-gpus, default: 2,3)"
        echo "  - Total iterations: $TOTAL_ITERS"
        echo "  - WandB: $WANDB (--wandb, default: true)"
        echo "  - WandB name: ${WANDB_NAME:-auto} (--wandb-name)"
        echo "  - Student checkpoint: ${FLASHDEPTH_CHECKPOINT:-config default} (--flashdepth-checkpoint)"
        echo "  - Teacher checkpoint: ${TEACHER_CHECKPOINT:-none} (--teacher-checkpoint)"
        echo "  - Results directory: $RESULTS_DIR (--results-dir)"
        echo ""

        DOCKER_CMD="CUDA_VISIBLE_DEVICES=$FSDP_GPUS docker compose run --rm \
            -e GLOO_SOCKET_IFNAME=eth0 \
            -e NCCL_SOCKET_IFNAME=eth0 \
            -e NCCL_P2P_DISABLE=1 \
            -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            -e WANDB_API_KEY=\${WANDB_API_KEY:-} \
            flashdepth torchrun \
            --standalone \
            --nproc_per_node=2 \
            train_onepiece_fsdp.py \
            --config-path configs/onepiece \
            --config-name $FSDP_CONFIG \
            dataset.data_root=/data/datasets \
            training.batch_size=$BATCH_SIZE \
            training.workers=$WORKERS \
            training.iterations=$TOTAL_ITERS \
            training.wandb=$WANDB \
            model.use_teacher_cls=$USE_TEACHER_CLS \
            +results_dir=$RESULTS_DIR"

        if [ -n "$FLASHDEPTH_CHECKPOINT" ]; then
            DOCKER_CMD="$DOCKER_CMD load=$FLASHDEPTH_CHECKPOINT"
        fi

        if [ -n "$TEACHER_CHECKPOINT" ]; then
            DOCKER_CMD="$DOCKER_CMD load_teacher=$TEACHER_CHECKPOINT"
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

            # Base results dir from user
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
            trap 'echo ""; echo "⚠️  Ctrl+C received — aborting..."; exit 130' INT

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
                    +resolution=$RESOLUTION \
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
                if [ "$SAVE_DEPTH_MAPS" = "true" ]; then
                    DOCKER_CMD="$DOCKER_CMD --save-depth-maps"
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
                training.workers=$WORKERS \
                +frame_interval=$FRAME_INTERVAL \
                +resolution=$RESOLUTION \
                +results_dir=$RESULTS_DIR"

            if [ -n "$GEAR_CHECKPOINT" ]; then
                DOCKER_CMD="$DOCKER_CMD load=$GEAR_CHECKPOINT"
            fi

            if [ -n "$ONEPIECE_TEST_DATASET" ]; then
                DOCKER_CMD="$DOCKER_CMD eval.test_datasets=[$ONEPIECE_TEST_DATASET]"
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

            if [ "$SAVE_DEPTH_MAPS" = "true" ]; then
                DOCKER_CMD="$DOCKER_CMD --save-depth-maps"
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
