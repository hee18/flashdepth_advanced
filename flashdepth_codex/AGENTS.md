# Repository Guidelines

## Project Structure & Module Organization
- `train.py`: Main entry for training and inference (Hydra + DDP-ready).
- `configs/`: Model-specific folders (e.g., `flashdepth`, `flashdepth-l`, `flashdepth-s`). Each contains `config.yaml`, outputs, and expected checkpoints.
- `flashdepth/`: Core model and layers (`model.py`, DPT/Mamba/xLSTM modules).
- `dataloaders/`: Dataset loaders and README for data layout.
- `utils/`: Initialization, logging, metrics, and checkpoint utilities.
- `examples/`: Sample videos for quick inference tests.
- `setup_env.sh`: Dependency installation (Python 3.11, PyTorch 2.4).

## Build, Test, and Development Commands
- Create env: `conda create -n flashdepth python=3.11 --yes && conda activate flashdepth && bash setup_env.sh`
- Inference (single video):
  `torchrun train.py --config-path configs/flashdepth inference=true eval.random_input=examples/video1.mp4 eval.outfolder=output`
- Timing smoke test:
  `torchrun train.py --config-path configs/flashdepth inference=true eval.dummy_timing=true`
- Training (stage 1 example, 8 GPUs):
  `torchrun --nproc_per_node=8 train.py --config-path configs/flashdepth-l/ load=checkpoints/depth_anything_v2_vitl.pth dataset.data_root=<path>`
- Note: If you see `TypeError: Invalid NaN comparison`, add `eval.compile=false`.

## Coding Style & Naming Conventions
- Python, PEP 8, 4-space indentation; prefer type hints.
- Names: files/functions `snake_case`; classes `CamelCase`.
- Config-first: do not hardcode paths; read from `cfg` (OmegaConf/DictConfig).
- Logging: use `utils/logging_config.py`; keep prints minimal.
- Keep modules small and colocate helpers in the closest package.

## Testing Guidelines
- Preferred: add lightweight `pytest` tests for new modules (if added).
- Minimum checks before PR:
  - Inference produces depth/MP4 under `configs/<model>/output/`.
  - `eval.dummy_timing=true` runs without errors.
  - Dataloaders resolve paths per `dataloaders/README.md`.

## Commit & Pull Request Guidelines
- Commits: imperative, concise (e.g., "add dataloader for TartanAir"). Conventional Commits optional.
- PRs must include: summary, commands used, sample logs/screenshots, and linked issue (if any).
- Update README/configs when changing flags, paths, or defaults.
- Keep PRs focused; avoid unrelated refactors.

## Security & Configuration Tips
- Do not commit datasets, checkpoints, or wandb artifacts. Store checkpoints under `configs/<model>/*.pth` as documented.
- Set `WANDB_MODE=disabled` locally if logging is not desired.
- Respect `.gitignore`; prefer small, reviewable diffs.

## Agent-Specific Instructions
- Follow this file’s guidance for code structure and style.
- Make minimal, targeted patches; do not reformat untouched files.
- Prefer repository-native patterns (Hydra configs, `torchrun`) for examples and scripts.
