# pysfm
COLMAP Structure-from-Motion reconstruction through [pycolmap](https://github.com/colmap/pycolmap) with [Rerun](https://rerun.io) visualization. Supports monocular video, multi-camera rigs, and standard image collections. Uses **tyro** for the CLI and **Pixi** for one-command setup.

<p align="center">
  <a title="Rerun" href="https://rerun.io" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Rerun-0.30%2B-0b82f9" alt="Rerun badge">
  </a>
  <a title="Pixi" href="https://pixi.sh/latest/" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Install%20with-Pixi-16A34A" alt="Pixi badge">
  </a>
  <a title="CUDA" href="https://developer.nvidia.com/cuda-toolkit" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/CUDA-12.x-76b900" alt="CUDA badge">
  </a>
</p>

<p align="center">
  <img src="media/pycolmap-github.gif" alt="pysfm demo" width="720" />
</p>

## Installation

Make sure you have the [Pixi](https://pixi.sh/latest/#installation) package manager installed.

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

This is part of the [examples-monorepo](https://github.com/rerun-io/examples-monorepo). Clone and run:

```bash
git clone https://github.com/rerun-io/examples-monorepo.git
cd examples-monorepo
pixi run -e pysfm sfm-reconstruction-demo
```

On the first run, example datasets are automatically downloaded. Subsequent runs skip the download.

## Usage

### CLI demos

```bash
pixi run -e pysfm sfm-reconstruction-demo    # COLMAP SfM on Fountain dataset (saves .rrd)
pixi run -e pysfm pysfm-vid-recon-demo       # Monocular video reconstruction
pixi run -e pysfm pysfm-vid-to-img-demo      # Video frame extraction (saves .rrd)
```

### Gradio web UIs

```bash
pixi run -e pysfm sfm-reconstruction-app     # SfM reconstruction app with Rerun viewer
pixi run -e pysfm pysfm-vid-to-img-node      # Video-to-image extraction app
pixi run -e pysfm pysfm-vid-to-img-node-dev  # Same, with Gradio hot reload
```

To see all available tasks:
```bash
pixi task list -e pysfm
```

## Project structure

```
packages/pysfm/
├── pyproject.toml                          # Package metadata
├── pysfm/                                  # Main module (editable install)
│   ├── __init__.py                         # Beartype activation (dev env only)
│   ├── streamed_pipeline.py                # Per-image pipeline stage implementations
│   ├── apis/
│   │   ├── pycolmap_rig_recon.py           # Multi-camera rig reconstruction
│   │   ├── pycolmap_vid_recon.py           # Monocular video reconstruction
│   │   ├── sfm_reconstruction.py           # General SfM reconstruction
│   │   └── video_to_image.py               # Video frame extraction
│   └── gradio_ui/
│       └── nodes/                          # Gradio UI components
│           ├── video_to_image_ui.py
│           └── sfm_reconstruction_ui.py
├── tools/
│   ├── demos/                              # CLI entry points (tyro)
│   │   ├── sfm_reconstruction.py
│   │   ├── pycolmap_vid_recon.py
│   │   ├── pycolmap_rig_recon.py
│   │   └── video_to_image.py
│   ├── nodes/                              # Gradio app launchers
│   │   ├── sfm_reconstruction_app.py
│   │   └── video_to_image_node.py
│   └── brush/
│       └── train_colmap_brush.py           # Brush/3DGS training from COLMAP output
└── tests/
    └── test_streamed_extraction.py
```

## Pipelines

### SfM Reconstruction
Standard COLMAP SfM from an image directory: feature extraction, matching, and incremental reconstruction. Logs cameras, sparse points, and reconstruction progress to Rerun.

```bash
python tools/demos/sfm_reconstruction.py --image-dir path/to/images --rr-config.save output.rrd
```

### Monocular Video Reconstruction
End-to-end pipeline from a single video: frame extraction, sequential feature matching, and 3D reconstruction.

```bash
python tools/demos/pycolmap_vid_recon.py --config.video-path path/to/video.mp4
```

### Multi-Camera Rig Reconstruction
Reconstruct scenes from calibrated or uncalibrated multi-camera rigs. Supports both known extrinsics and automatic rig calibration.

```bash
python tools/demos/pycolmap_rig_recon.py --help
```

### Video to Image
Extract evenly-spaced frames from video for downstream SfM processing.

```bash
python tools/demos/video_to_image.py --video-path path/to/video.mp4 --output-dir /tmp/frames --rr-config.save /tmp/frames/output.rrd
```

## Acknowledgements

- [COLMAP](https://colmap.github.io/) / [pycolmap](https://github.com/colmap/pycolmap) — Structure-from-Motion and Multi-View Stereo
- [Rerun](https://rerun.io) — 3D visualization
- [Brush](https://github.com/ArthurBrussworthy/brush) — 3D Gaussian Splatting

```bibtex
@inproceedings{schoenberger2016sfm,
    author={Sch\"{o}nberger, Johannes Lutz and Frahm, Jan-Michael},
    title={Structure-from-Motion Revisited},
    booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2016},
}
```
