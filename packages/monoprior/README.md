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
pixi run app
```

## Demo
Hosted Demos can be found on huggingface spaces

<a href='https://huggingface.co/spaces/pablovela5620/depth-compare'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>


To run the gradio frontend
```bash
pixi run app
```

To see all available tasks
```bash
pixi task list
```

## Features

### Multi-view Depth Estimation
Use VGGT model for multi-view consistent depth estimation and camera pose recovery:

```bash
pixi run multiview_inference --image-dir /path/to/images
```

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

[Metric3D V2](https://github.com/YvanYin/Metric3D)
```bibtex
@article{hu2024metric3dv2,
  title={Metric3D v2: A Versatile Monocular Geometric Foundation Model for Zero-shot Metric Depth and Surface Normal Estimation},
  author={Hu, Mu and Yin, Wei and Zhang, Chi and Cai, Zhipeng and Long, Xiaoxiao and Chen, Hao and Wang, Kaixuan and Yu, Gang and Shen, Chunhua and Shen, Shaojie},
  journal={arXiv preprint arXiv:2404.15506},
  year={2024}
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