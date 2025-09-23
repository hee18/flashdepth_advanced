FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel

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

# PyTorch is already installed in the base image (2.4.0 with CUDA 11.8)
# Install torchvision to match
RUN pip install torchvision==0.19.0

# Install other dependencies
RUN pip install xformers==0.0.27.post2+cu118 --index-url https://download.pytorch.org/whl/cu118
RUN pip install ipdb tqdm wandb
RUN pip install matplotlib einops scipy h5py OpenEXR seaborn pandas
RUN pip install hydra-core
RUN pip install opencv-python pillow
RUN pip install packaging wheel

# Install flash-attention
RUN pip install flash-attn --no-build-isolation

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

# Default command to run train_metric_head.py (Stage 1: FlashDepth-L)
CMD ["python", "train_metric_head.py", "--config-path", "configs/flashdepth-l", "dataset.data_root=/data/datasets"]