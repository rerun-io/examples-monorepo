# Monopriors
A library to easily get monocular priors such as scale-invariant depths, metric depths, or surface normals. Using Rerun viewer, Pixi and Gradio for easy use
<p align="center">
  <img src="media/depth-compare.gif" alt="example output" width="480" />
</p>


## Installation
Easily installable via [Pixi](https://pixi.sh/latest/).
```bash
git clone https://github.com/pablovela5620/monoprior.git
cd monoprior
pixi run -e monoprior depth-compare-app
```

## Demo
Hosted Demos can be found on huggingface spaces

<a href='https://huggingface.co/spaces/pablovela5620/depth-compare'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>

### CLI demos (auto-download example data from HuggingFace)
```bash
pixi run -e monoprior relative-depth          # Single-image relative depth
pixi run -e monoprior multiview-depth          # VGGT multi-view depth
pixi run -e monoprior multiview-calibration    # Multi-view calibration
pixi run -e monoprior video-depth              # Video depth estimation
pixi run -e monoprior polycam-inference        # Polycam dataset inference
pixi run -e monoprior promptda-polycam         # PromptDA depth completion
pixi run -e monoprior compare-normals          # Compare surface normal models
```

### Gradio web UIs
```bash
pixi run -e monoprior depth-compare-app        # Depth comparison UI
pixi run -e monoprior video-depth-app          # Video depth UI
pixi run -e monoprior calibration-app          # Multi-view calibration UI
```

To see all available tasks
```bash
pixi task list -e monoprior
```

## Features

### Multi-view Depth Estimation
Use VGGT model for multi-view consistent depth estimation and camera pose recovery:

```bash
pixi run -e monoprior multiview-depth
```

## Removed Models

### Metric3D
Metric3D was removed because it depends on mmcv/mmengine (OpenMMLab), which is not conda-installable with CUDA support and therefore incompatible with Pixi. The last mmcv release (April 2024) has no wheels for recent PyTorch/CUDA versions, and the model is loaded via `torch.hub.load` which clones the entire 290 MB repo unpinned from `main`.

To re-add Metric3D in the future, vendor the model code directly into `monopriors/third_party/metric3d/` instead of using `torch.hub`. The actual model architecture (ViT backbone, RAFT heads) is pure PyTorch — mmcv is only used for config parsing and can be replaced. The relevant source files are in the `mono/` directory of [YvanYin/Metric3D](https://github.com/YvanYin/Metric3D).

## Acknowledgements
Thanks to the following great works!

[DepthAnything](https://github.com/LiheYoung/Depth-Anything)
```bibtex
@inproceedings{depthanything,
      title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
      author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
      booktitle={CVPR},
      year={2024}
}
```

[Unidepth](https://github.com/lpiccinelli-eth/UniDepth)
```bibtex
@inproceedings{piccinelli2024unidepth,
    title     = {{U}ni{D}epth: Universal Monocular Metric Depth Estimation},
    author    = {Piccinelli, Luigi and Yang, Yung-Hsu and Sakaridis, Christos and Segu, Mattia and Li, Siyuan and Van Gool, Luc and Yu, Fisher},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2024}
}
```

[VGGT](https://github.com/facebookresearch/vggt)
```bibtex
@inproceedings{wang2025vggt,
  title={VGGT: Visual Geometry Grounded Transformer},
  author={Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```
