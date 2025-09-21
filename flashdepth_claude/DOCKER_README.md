# FlashDepth Docker Setup

This guide helps you run FlashDepth training in a Docker environment, specifically configured for `train_metric_head.py` with TartanAir dataset.

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA Docker runtime (for GPU support)
- TartanAir dataset available at `/home/cvlab/hsy/Datasets/tartanair/`

## Quick Start

### 1. Build the Docker Image

```bash
./run_docker.sh build
```

### 2. Start Training

```bash
# Default training (batch_size=2, workers=4)
./run_docker.sh train

# Custom training parameters
./run_docker.sh train --batch-size 4 --workers 8 --epochs 30000
```

### 3. Development Shell

```bash
./run_docker.sh shell
```

## Directory Structure

The Docker setup creates the following volume mappings:

```
Host System                    → Container
/home/cvlab/hsy/Datasets      → /data/datasets          (read-only)
./train_results               → /app/train_results      (read-write)
./checkpoints                 → /app/checkpoints        (read-write)
./configs                     → /app/configs            (read-write)
./flashdepth                  → /app/flashdepth         (read-write)
./dataloaders                 → /app/dataloaders        (read-write)
./utils                       → /app/utils              (read-write)
```

## Result Access

Training results are automatically saved and accessible both inside and outside the container:

- **Training logs**: Console output and container logs
- **Checkpoints**: `./checkpoints/` directory
  - `latest_metric_head.pth` - Latest model checkpoint
  - `best_metric_head.pth` - Best performing model
  - `metric_head_step_X.pth` - Periodic checkpoints
- **Training results**: `./train_results/` directory (if configured in training script)

## Configuration

### Dataset Configuration

The setup is pre-configured for TartanAir dataset:
- Dataset path: `/data/datasets/tartanair/`
- Training datasets: `[tartanair]`
- Validation datasets: `[tartanair]`

### GPU Configuration

Default GPU setup:
- Uses GPU 0 by default
- To use different GPU: `./run_docker.sh train --gpu 1`
- Set `CUDA_VISIBLE_DEVICES` environment variable

### Training Parameters

Key parameters you can modify:

```bash
./run_docker.sh train \
    --batch-size 4 \        # Batch size (default: 2)
    --workers 8 \           # Data loader workers (default: 4)
    --epochs 30000 \        # Training iterations (default: 60001)
    --gpu 1                 # GPU ID (default: 0)
```

## Manual Docker Commands

If you prefer manual control:

### Build
```bash
docker compose build
```

### Run Training
```bash
docker compose run --rm flashdepth python train_metric_head.py \
    --config-path configs/flashdepth \
    dataset.data_root=/data/datasets \
    dataset.train_datasets=[tartanair] \
    dataset.val_datasets=[tartanair] \
    training.batch_size=2
```

### Interactive Shell
```bash
docker compose run --rm flashdepth /bin/bash
```

### Check Logs
```bash
docker compose logs -f flashdepth
```

## Monitoring Training

### Real-time Logs
```bash
./run_docker.sh logs
```

### Checkpoint Monitoring
Check the `./checkpoints/` directory for saved models:
```bash
ls -la checkpoints/
```

### Training Progress
The training script outputs:
- Loss values per batch
- Scale and shift metrics
- Validation metrics (if validation frequency is set)

## Troubleshooting

### Common Issues

1. **GPU not detected**
   - Ensure NVIDIA Docker runtime is installed
   - Check: `docker info | grep nvidia`

2. **Permission issues**
   - Ensure Docker has access to dataset directory
   - Check directory permissions

3. **Out of memory**
   - Reduce batch size: `--batch-size 1`
   - Reduce workers: `--workers 2`

4. **Dataset not found**
   - Verify TartanAir dataset exists at `/home/cvlab/hsy/Datasets/tartanair/`
   - Check dataset structure matches expected format

### Debug Commands

```bash
# Check container status
docker compose ps

# Check container resources
docker stats

# Inspect volumes
docker compose run --rm flashdepth ls -la /data/datasets

# Test GPU access
docker compose run --rm flashdepth nvidia-smi
```

## Customization

### Using Different Datasets

Modify the command in `docker-compose.yml` or use manual commands:

```bash
docker compose run --rm flashdepth python train_metric_head.py \
    --config-path configs/flashdepth \
    dataset.data_root=/data/datasets \
    dataset.train_datasets=[spring,mvs-synth] \
    dataset.val_datasets=[sintel,waymo]
```

### Custom Configurations

Mount additional config files:
```yaml
volumes:
  - ./my_custom_config.yaml:/app/configs/custom/config.yaml
```

Then use:
```bash
python train_metric_head.py --config-path configs/custom
```

## Architecture Notes

- **Base Image**: NVIDIA CUDA 11.8 with Ubuntu 22.04
- **Python**: 3.11
- **PyTorch**: 2.4.0 (required for Mamba2 compatibility)
- **Key Dependencies**: Mamba2, Flash-Attention, Xformers
- **Compute Capability**: Supports modern NVIDIA GPUs

## Performance Tips

1. **Batch Size**: Start with 2, increase if GPU memory allows
2. **Workers**: Set to 2x number of CPU cores available to container
3. **Mixed Precision**: Enabled by default in training script
4. **Gradient Checkpointing**: Enabled for memory efficiency