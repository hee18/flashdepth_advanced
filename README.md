# Metric-FlashDepth

**Real-time metric depth estimation for streaming video, built on top of [FlashDepth](https://github.com/Eyeline-Labs/FlashDepth) (ICCV 2025 Highlight)**

Repository: [`hee18/flashdepth_advanced`](https://github.com/hee18/flashdepth_advanced) — branch [`onepiece_final`](https://github.com/hee18/flashdepth_advanced/tree/onepiece_final)

---

## Overview

FlashDepth predicts **relative** depth from monocular video in real time. Metric-FlashDepth (internally called **Onepiece**) extends it with a lightweight, additional prediction head that recovers **absolute (metric) depth** — while keeping the original model's speed almost untouched and improving temporal stability across frames.

### Why this works

Most relative-depth methods (including the MiDaS family) are trained with a **Scale- and Shift-Invariant Loss (SSIL)**: before computing the error against ground truth, the optimal scale and shift that minimize least-squares error between the prediction and GT are applied to the prediction. This implies a simple but powerful idea:

```
Metric Depth = Relative Depth × Scale + Shift
```

If a model can predict just two extra numbers — **scale** and **shift** — a relative depth map can be converted directly into a metric depth map, without redesigning the whole depth backbone.

### Key contributions

- **Scale & Shift prediction head + Dual-CSTM (Canonical Space Transformation Module)**
  A dedicated head predicts scale/shift from a DINOv2 CLS token processed through a Mamba block. Dual-CSTM normalizes training targets into a focal-length-canonical space and de-canonicalizes predictions back to actual metric depth at inference, giving competitive absolute-depth accuracy while preserving the baseline model's inference speed.
- **Temporal consistency losses (TGM, OFC) + Scene-Cut Detection (SCD)**
  - **TGM (Temporal Gradient Matching) loss** and **OFC (Optical-Flow based feature Consistency) loss** enforce consistency between consecutive frames, removing the *flickering* artifacts common in per-frame depth estimation.
  - **SCD (Scene-Cut Detector)** detects hard cuts between frames (via DINOv2 patch/CLS feature distance) and resets the temporal (Mamba) hidden state at scene boundaries, so temporal smoothing is never applied across unrelated scenes.

### Architecture (high level)

```
Input Video ──▶ DINOv2 Encoder (frozen) ──▶ DPT Decoder ──▶ Relative Depth
                        │
                        ├── Fused CLS token ──▶ Scene-Cut Detector (inference only)
                        │
                        └── CLS + DPT features ──▶ Spatial Mamba (temporal)
                                                        │
                                       ┌────────────────┴────────────────┐
                                       ▼                                 ▼
                             Relative stream (DPT)             Metric stream (CLS)
                             → final head → relative depth     → Metric Head → scale, shift
                                       │                                 │
                                       └───────────────┬─────────────────┘
                                                        ▼
                                     Metric Depth = Scale × Relative Depth + Shift
```

Only the small scale/shift head (and, during full fine-tuning, the Mamba block) is trained — the DINOv2 backbone and DPT decoder can remain frozen, keeping the number of trainable parameters small (~260K) relative to the ~300M-parameter backbone.

### Results

Evaluated qualitatively on the **Unreal4K** dataset over sequences of consecutive frames:

- Depth predictions stay closest to ground truth among compared methods.
- No visible flickering across consecutive frames, confirming that the temporal losses (TGM/OFC) and Scene-Cut Detection are effective.

Quantitative comparisons (MAE, RMSE, AbsRel, δ1/δ2/δ3, and Temporal Alignment Error) are produced by `test_onepiece.py` on TartanAir, Sintel, ETH3D, VKITTI, Waymo, and Unreal4K sequences (see [Testing](#testing) below).

---

## Repository Structure

```
flashdepth_claude/
├── flashdepth/            # Core model (DINOv2 backbone, DPT head, Mamba, Onepiece modules)
├── configs/
│   ├── flashdepth/        # Original FlashDepth (Full)
│   ├── flashdepth-l/      # FlashDepth-L
│   ├── flashdepth-s/      # FlashDepth-S
│   └── onepiece/          # Metric-FlashDepth configs (config_l / config_s / config_hybrid, FSDP variants)
├── dataloaders/           # MVS-Synth, Spring, TartanAir, PointOdyssey, DynamicReplica, combined dataset
├── mamba/                 # Local Mamba2 package (required for install)
├── train.py                       # Original FlashDepth training/inference entry point
├── train_onepiece.py               # Metric-FlashDepth (Onepiece) training entry point
├── train_onepiece_fsdp.py          # Onepiece training with FSDP2 (multi-GPU, high-res hybrid)
├── test_onepiece.py                # Metric-FlashDepth evaluation across benchmark datasets
├── test_scd_ablation.py            # Scene-Cut Detection ablation study
├── run_docker.sh                   # Docker-based build/train/test/inference commands
└── Onepiece.md / changelog.md      # Detailed architecture notes and change history
```

---

## Installation

Requires **Python 3.11** and **torch ≤ 2.4** (Mamba2 does not compile against torch 2.5+ as of this writing).

```bash
conda create -n flashdepth python=3.11 --yes
conda activate flashdepth
bash setup_env.sh
```

`setup_env.sh` installs Mamba2 from the local `mamba/` folder, so the versions stay compatible.

A Docker-based workflow is also provided (`Dockerfile`, `docker-compose.yml`, `run_docker.sh`) for reproducible builds and GPU-based training/testing.

---

## Downloading Pretrained Models

The base FlashDepth checkpoints (relative depth) are hosted on Hugging Face:
[FlashDepth (Full)](https://huggingface.co/Eyeline-Labs/FlashDepth/tree/main/flashdepth) ·
[FlashDepth-L](https://huggingface.co/Eyeline-Labs/FlashDepth/tree/main/flashdepth-l) ·
[FlashDepth-S](https://huggingface.co/Eyeline-Labs/FlashDepth/tree/main/flashdepth-s)

Save them to:

```
configs/flashdepth/iter_43002.pth
configs/flashdepth-l/iter_10001.pth
configs/flashdepth-s/iter_14001.pth
```

These serve as the frozen backbone that the Onepiece scale/shift head is trained on top of.

---

## Training

### 1. Base FlashDepth (relative depth)

Two-stage training as in the original paper — first at 518×518, then at higher resolution:

```bash
# Stage 1: FlashDepth-L / FlashDepth-S at 518x518
torchrun --nproc_per_node=8 train.py --config-path configs/flashdepth-l/ \
  load=checkpoints/depth_anything_v2_vitl.pth dataset.data_root=<path_to_data>
torchrun --nproc_per_node=8 train.py --config-path configs/flashdepth-s/ \
  load=checkpoints/depth_anything_v2_vits.pth dataset.data_root=<path_to_data>

# Stage 2: FlashDepth (Full) at higher resolution, distilled from stage-1 checkpoints
torchrun --nproc_per_node=8 train.py --config-path configs/flashdepth \
  load=configs/flashdepth-s/<latest_checkpoint>.pth \
  hybrid_configs.teacher_model_path=configs/flashdepth-l/<latest_checkpoint>.pth \
  dataset.data_root=<path_to_data>
```

See `dataloaders/README.md` for the expected data directory layout per dataset.

### 2. Metric-FlashDepth / Onepiece head

```bash
python train_onepiece.py \
  --config-path configs/onepiece --config-name config_l \
  dataset.data_root=<path_to_data> \
  training.batch_size=3 \
  training.workers=8 \
  training.iterations=60001 \
  load=<path_to_flashdepth_checkpoint>
```

Or with the Docker helper script:

```bash
./run_docker.sh train_onepiece --gpu 0 --config-variant l
./run_docker.sh train_onepiece_ddp --ddp-gpus 0,1        # 2-GPU DDP
./run_docker.sh train_onepiece_fsdp                       # FSDP2, high-res hybrid
```

Training runs in two phases:
- **Phase 1** — only the metric (scale/shift) head is trained; the backbone, DPT decoder, and Spatial Mamba stay frozen.
- **Phase 2** — Spatial Mamba, the metric head, DPT, and the output head are jointly fine-tuned with a warm-up for newly unfrozen parameters.

To ablate Dual-CSTM (train directly on actual depth, no canonical-space normalization), pass `use_dual_cstm=false` (or `--no-cstm` via `run_docker.sh`).

---

## Testing

```bash
python test_onepiece.py \
  --config-path configs/onepiece --config-name config_l \
  --checkpoint <path_to_onepiece_checkpoint> \
  --dataset all
```

Or:

```bash
./run_docker.sh test_onepiece --dataset all --gpu 0
```

Reports MAE, RMSE, AbsRel, δ1/δ2/δ3, and Temporal Alignment Error (TAE) across same-domain and cross-domain sequences (TartanAir, Sintel, ETH3D, VKITTI, Waymo, Unreal4K). Scene-Cut Detection ablations are available via `test_scd_ablation.py`.

---

## Inference on a Video

```bash
torchrun train.py --config-path configs/flashdepth inference=true \
  eval.random_input=<path_to_video> eval.outfolder=output
```

Outputs (`.npy` depth maps and `.mp4` visualizations) are saved to `configs/flashdepth/output/`.

> If you hit `TypeError: Invalid NaN comparison`, add `eval.compile=false`.

For FPS/latency measurement:

```bash
torchrun train.py --config-path configs/flashdepth inference=true eval.dummy_timing=true
```

---

## Notes

- Torch must stay at **2.4** for Mamba2 compatibility.
- `xformers`'s memory-efficient attention is not exportable to ONNX; export uses the standard `scaled_dot_product_attention` fallback instead, without code changes (runtime flag only).

---

## Acknowledgements

This project builds on [FlashDepth](https://github.com/Eyeline-Labs/FlashDepth) (Chou et al., ICCV 2025) and [Depth Anything V2](https://depth-anything-v2.github.io/), and uses [Mamba2](https://github.com/state-spaces/mamba) for temporal modeling.

```bibtex
@inproceedings{chou2025flashdepth,
  title     = {FlashDepth: Real-time Streaming Video Depth Estimation at 2K Resolution},
  author    = {Chou, Gene and Xian, Wenqi and Yang, Guandao and Abdelfattah, Mohamed and Hariharan, Bharath and Snavely, Noah and Yu, Ning and Debevec, Paul},
  journal   = {The IEEE International Conference on Computer Vision (ICCV)},
  year      = {2025},
}
```

---

**Author**: Heesang Yoon (윤희상) — Computer Vision Engineer
E-mail: heesang545@gmail.com
