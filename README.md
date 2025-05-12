# FlashDepth: Real-time Streaming Video Depth Estimation at 2K Resolution

[**Arxiv**](https://arxiv.org/abs/2504.07093) | [**Project Page**](https://eyeline-research.github.io/flashdepth/) <br>


This repository contains the official implementation of **FlashDepth: Real-time Streaming Video Depth Estimation at 2K Resolution**. 



## Installation
We recommend creating a [conda](https://www.anaconda.com/) environment then installing the required packages using the following commands. Note that the mamba package should be installed from our local folder.

```
conda create -n flashdepth python=3.11 --yes
conda activate flashdepth
bash setup_env.sh
```

## Downloading Pretrained Models 
We provide three checkpoints on huggingface. They correspond to FlashDepth (Full), FlashDepth-L, and FlashDepth-S, as referenced in the paper. Generally, FlashDepth-L is most accurate and FlashDepth (Full) is fastest. Save the checkpoints to `configs/flashdepth/flashdepth.pth`, `configs/flashdepth-l/flashdepth-l.pth`, and `configs/flashdepth-s/flashdepth-s.pth`, respectively. 

## Inference 
To run inference on a video:
```
torchrun train.py --config-path configs/flashdepth inference=true eval.random_input=<path to video> eval.outfolder=inference
```
The output depth maps (as npy files) and mp4s will be saved to `configs/flashdepth/inference/`. Change the configs path to use another model.

## Evaluation
To run evaluation metrics, using waymo and eth3d as examples:
```
torchrun train.py --config-path configs/flashdepth inference=true eval.test_datasets=[eth3d,waymo] eval.outfolder=test_set &&\
cd evaluation/testdata/ &&\
python metrics.py \
    --src_base evaluation/testdata/waymo/scenes  \
    --output_path evaluation/testdata/waymo/metrics.json \
    --paths configs/flashdepth/test_set/waymo/ && \
python metrics.py \
    --src_base evaluation/testdata/eth3d/scenes  \
    --output_path evaluation/testdata/eth3d/metrics.json \
    --paths configs/flashdepth/test_set/eth3d/
```
See the evaluation folder for our default format of the test data. 

## Training
As reported in the paper, training is split into two stages. We first train FlashDepth-L and FlashDepth-S at resolution 518x518. Then, we train FlashDepth (Full) at higher resolution. 
To train the first stage, download the [Depth Anything V2](https://depth-anything-v2.github.io/) checkpoints and save them to `checkpoints`.
```
# first stage 
torchrun --nproc_per_node=8 train.py --config-path configs/flashdepth-l/ load=checkpoints/depth_anything_v2_vitl.pth 
torchrun --nproc_per_node=8 train.py --config-path configs/flashdepth-s/ load=checkpoints/depth_anything_v2_vits.pth 

# second stage 
torchrun --nproc_per_node=8 train.py --config-path configs/flashdepth load=configs/flashdepth-s/<latest flashdepth-s checkpoint> hybrid_configs.teacher_model_path=configs/flashdepth-l/<latest flashdepth-l checkpoint> hybrid_configs.teacher_resolution=518
``` 

Check the `config.yaml` files in the `configs` folders for hyperparameters and logging.


If you find our code or paper useful, please consider citing
```bibtex
@misc{chou2025flashdepth,
  author    = {Chou, Gene and Xian, Wenqi and Yang, Guandao and Abdelfattah, Mohamed and Hariharan, Bharath and Snavely, Noah and Yu, Ning and Debevec, Paul},
  title     = {FlashDepth: Real-time Streaming Video Depth Estimation at 2K Resolution},
  booktitle = {arXiv preprint arXiv:2504.07093},
  url       = {https://arxiv.org/abs/2504.07093},
  year      = {2025},
}
```

## References
Our code was modified and heavily borrowed from the following projects: <br>
[Depth Anything V2](https://depth-anything-v2.github.io/) <br>
[Mamba 2](https://github.com/state-spaces/mamba)