FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and setup files first for better caching
COPY setup_env.sh /app/
COPY mamba/ /app/mamba/

# PyTorch is already installed in the base image (2.4.0 with CUDA 12.4)
# Install torchvision to match
RUN pip install torchvision==0.19.0

# Install other dependencies
RUN pip install ipdb tqdm wandb
RUN pip install matplotlib einops scipy h5py OpenEXR seaborn pandas
RUN pip install hydra-core
RUN pip install opencv-python pillow pyarrow
RUN pip install packaging wheel

# Install ninja for faster compilation
RUN pip install ninja

# Install xformers from source for CUDA 12.4 compatibility
# Ensure CUDA is available during build
RUN nvcc --version && \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" \
    FORCE_CUDA=1 \
    MAX_JOBS=4 \
    pip install -v -U git+https://github.com/facebookresearch/xformers.git@v0.0.27 --no-build-isolation

# Install flash-attention
RUN pip install flash-attn --no-build-isolation

# Install causal-conv1d (required for Mamba2)
# Pin to specific version for compatibility with mamba-ssm
RUN pip install causal-conv1d==1.4.0 --no-build-isolation

# Install local Mamba package
WORKDIR /app/mamba
ENV MAMBA_FORCE_BUILD=TRUE
RUN python -m pip install --no-build-isolation .

# Go back to app directory
WORKDIR /app

# Copy the entire project
COPY . /app/

# Create directories for data mounting and results
RUN mkdir -p /data/datasets /app/train_results

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0

# Disable IPv6 to prevent socket connection issues with DDP
ENV GLOO_SOCKET_IFNAME=eth0
ENV NCCL_SOCKET_IFNAME=eth0

# Default command to run train_metric_head.py (Stage 1: FlashDepth-L)
CMD ["python", "train_metric_head.py", "--config-path", "configs/flashdepth-l", "dataset.data_root=/data/datasets"]