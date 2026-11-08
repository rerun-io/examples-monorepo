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
pixi run -e mast3r-slam --frozen mast3r-slam-example-fast
```

### CLI (base config, 512px)
```bash
pixi run -e mast3r-slam --frozen mast3r-slam-example-base
```

### Gradio App
```bash
pixi run -e mast3r-slam --frozen mast3r-slam-app
```

### Direct Python (from package directory)
```bash
cd packages/mast3r-slam
pixi run -e mast3r-slam --frozen python tools/mast3r_slam_inference.py \
    base --dataset data/normal-apt-tour.mp4 --img-size 512
```

The first argument names a SLAM preset (`base`, `fast`, `test`, `calib`, `eval_calib`, `eval_no_calib`, `eth3d`;
`base` when omitted; they live in `mast3r_slam/config.py`). Override any field with `--slam.<group>.<field>`, e.g.
`fast --slam.tracking.max-iters 30`; `<preset> --help` lists every field with that preset's value. A recording
carries the resolved config as its `slam_config` property.

## First Run

The first run will:
1. Build **lietorch-mojo** (Mojo extension)
2. Install **mast3r** thirdparty + asmk C extension
3. Build **mast3r-slam Mojo kernels**
4. Download model **checkpoints** (~1 GB)
5. Download **example data** (video file)

Subsequent runs skip all install steps (idempotent checks).

## Notes

- **Linux only** with an NVIDIA GPU (CUDA required)
- curope (CUDA RoPE2D) is skipped; falls back to a slower PyTorch implementation.
  **TODO:** Add curope as a pixi-build workspace package (`packages/curope/`) to enable
  the faster CUDA kernel from `naver/croco/models/curope/`.

## Known limitations

- **TODO (relocalisation, not being worked on):** once tracking is lost (fewer than 5 % valid matches, e.g. a blank
  wall or glass filling the view), mast3r-slam freezes the pose and only recovers if image retrieval finds an
  already-mapped keyframe that matches. When the camera keeps moving into new areas, every attempt fails and the
  rest of the clip is lost. dpvo's `iphone/pool.mp4` shows this (lost at 78.5 s with every frame, at 8.3 s with
  every 5th frame), so it is left out of the example clips and the download task. Possible fixes, smallest first:
  record relocalisation attempts in the saved recording (today they only reach a connected viewer); try
  `reloc.strict = False`; start a new map after repeated failures and join the maps on loop closure (as ORB-SLAM3
  does). Related: in single-thread mode the relocalisation wait spins forever if the backend process dies.

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
