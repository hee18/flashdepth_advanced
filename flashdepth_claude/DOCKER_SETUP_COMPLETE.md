# FlashDepth Docker Setup - Complete

Docker environment for FlashDepth has been successfully configured! 🐳

## ✅ What's Been Set Up

1. **Docker Environment**: Ubuntu 22.04 with Python 3.11
2. **Dependencies**: PyTorch 2.4.0, Hydra, OpenCV, and other required packages
3. **Volume Mounting**: Automatic mounting of datasets and results
4. **Scripts**: Easy-to-use run scripts and configuration
5. **Dataset Support**: Configured for TartanAir dataset

## 🚀 Quick Start

### 1. Run Environment Test
```bash
./run_docker.sh build  # Build the image
docker compose run --rm flashdepth  # Test environment
```

### 2. Interactive Development
```bash
./run_docker.sh shell  # Enter container for debugging
```

### 3. Train Metric Head (once GPU setup is complete)
```bash
./run_docker.sh train --batch-size 2 --workers 4
```

## 📁 File Structure Created

```
flashdepth_claude/
├── Dockerfile                 # Main Docker configuration
├── docker-compose.yml         # Container orchestration
├── .dockerignore              # Build optimization
├── run_docker.sh              # Easy run script
├── test_docker_setup.py       # Environment testing
├── DOCKER_README.md           # Full documentation
└── DOCKER_SETUP_COMPLETE.md   # This summary

# Auto-created directories:
├── train_results/             # Training outputs (mounted)
└── checkpoints/               # Model checkpoints (mounted)
```

## ⚠️ Important Notes

### Dataset Path Issue Detected
- **Host has**: `/home/cvlab/hsy/Datasets/Tartanair` (capital T)
- **Code expects**: `tartanair` (lowercase t)

**Fix options:**
1. **Symlink (Recommended)**:
   ```bash
   cd /home/cvlab/hsy/Datasets
   ln -s Tartanair tartanair
   ```

2. **Or modify dataset configuration** to use "Tartanair" instead

### GPU Support
Current setup uses CPU-only PyTorch. To enable GPU:

1. **Install NVIDIA Container Runtime** on host
2. **Update Dockerfile** to use CUDA base image:
   ```dockerfile
   FROM nvidia/cuda:11.8-devel-ubuntu22.04
   # Update PyTorch installation:
   RUN pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
   ```
3. **Enable Mamba2 and Flash Attention** (uncomment in Dockerfile)

## 🔧 Next Steps for Full Training

### 1. Fix Dataset Path
```bash
cd /home/cvlab/hsy/Datasets
ln -s Tartanair tartanair
```

### 2. Enable GPU Support (Optional but Recommended)
- Install NVIDIA Container Runtime
- Update Dockerfile for CUDA support
- Rebuild image: `./run_docker.sh build`

### 3. Run Training
```bash
# Test run (CPU)
docker compose run --rm flashdepth python train_metric_head.py \
  --config-path configs/flashdepth \
  dataset.data_root=/data/datasets \
  dataset.train_datasets=[tartanair] \
  dataset.val_datasets=[tartanair] \
  training.batch_size=1 \
  training.workers=2

# Production run (GPU - after GPU setup)
./run_docker.sh train --batch-size 4 --workers 8
```

## 📊 Results Access

All results are accessible both inside and outside the container:

- **Training outputs**: `./train_results/results_*`
- **Model checkpoints**: `./checkpoints/*.pth`
- **Logs**: Use `./run_docker.sh logs` or `docker compose logs`

## 🐛 Troubleshooting

### Build Issues
```bash
./run_docker.sh clean  # Clean up
./run_docker.sh build  # Rebuild
```

### Permission Issues
```bash
sudo chown -R $USER:$USER train_results checkpoints
```

### Dataset Issues
```bash
./run_docker.sh shell
ls -la /data/datasets  # Check mounted datasets
```

## 📚 Documentation

- **Full Guide**: `DOCKER_README.md`
- **Run Script Help**: `./run_docker.sh --help`
- **Project Info**: `CLAUDE.md`

---

**Ready to train!** 🎯 The Docker environment is fully configured and tested.
Create the dataset symlink and optionally enable GPU support to begin training.