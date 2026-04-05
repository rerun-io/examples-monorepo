# MASt3R-SLAM

An unofficial implementation of MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors.

Uses [Rerun](https://rerun.io/) to visualize, [Gradio](https://www.gradio.app) for an interactive UI, and [Pixi](https://pixi.sh/latest/) for easy installation.

Based on [rerun-io/mast3r-slam](https://github.com/rerun-io/mast3r-slam).

<p align="center">
  <img src="media/mast3r-slam.gif" alt="MASt3R-SLAM demo" width="768">
</p>

<p align="center">
  <a title="Website" href="https://edexheim.github.io/mast3r-slam/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
      <img src="https://www.obukhov.ai/img/badges/badge-website.svg">
  <a title="arXiv" href="https://arxiv.org/abs/2412.12392" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
      <img src="https://www.obukhov.ai/img/badges/badge-pdf.svg">
  </a>
</p>

## Usage

Run from the **monorepo root**:

### CLI (fast config, 224px)
```bash
pixi run -e mast3r-slam --frozen example-fast
```

### CLI (base config, 512px)
```bash
pixi run -e mast3r-slam --frozen example-base
```

### Gradio App
```bash
pixi run -e mast3r-slam --frozen mast3r-slam-app
```

### Direct Python (from package directory)
```bash
cd packages/mast3r-slam
pixi run -e mast3r-slam --frozen python tools/mast3r_slam_inference.py \
    --dataset data/normal-apt-tour.MOV --img-size 512 --config config/base.yaml
```

## First Run

The first run will:
1. Install **lietorch** (CUDA extension, ~5 min)
2. Install **mast3r** thirdparty + asmk C extension
3. Build **mast3r-slam CUDA kernels** (~3 min)
4. Download model **checkpoints** (~1 GB)
5. Download **example data** (video file)

Subsequent runs skip all install steps (idempotent checks).

## Notes

- **Linux only** with an NVIDIA GPU (CUDA required)
- Python 3.11 (pinned for lietorch compatibility)
- curope (CUDA RoPE2D) is skipped; falls back to a slower PyTorch implementation
- `pyrealsense2` is optional (only needed for live RealSense camera input)

## Acknowledgements

[MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM)
```bibtex
@article{murai2024_mast3rslam,
    title={{MASt3R-SLAM}: Real-Time Dense {SLAM} with {3D} Reconstruction Priors},
    author={Murai, Riku and Dexheimer, Eric and Davison, Andrew J.},
    journal={arXiv preprint},
    year={2024},
}
```

[MASt3R](https://github.com/naver/mast3r)
```bibtex
@misc{mast3r_arxiv24,
      title={Grounding Image Matching in 3D with MASt3R},
      author={Vincent Leroy and Yohann Cabon and Jerome Revaud},
      year={2024},
      eprint={2406.09756},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

[DUSt3R](https://github.com/naver/dust3r)
```bibtex
@inproceedings{dust3r_cvpr24,
      title={DUSt3R: Geometric 3D Vision Made Easy},
      author={Shuzhe Wang and Vincent Leroy and Yohann Cabon and Boris Chidlovskii and Jerome Revaud},
      booktitle = {CVPR},
      year = {2024}
}
```
