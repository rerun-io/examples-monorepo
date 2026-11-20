# colsfm

An open rebuild of NVIDIA cuSFM ([paper](https://arxiv.org/abs/2510.15271), [code](https://github.com/nvidia-isaac/PyCuSFM))
in Python on pycolmap: rig-aware structure from motion that **refines an initial trajectory** through cuSFM's eight stages
(keyframe selection, feature extraction, pair selection, matching, loop closure, pose graph, triangulation + bundle
adjustment with rig-extrinsic refinement, export). It reads a rig, its video and a starting trajectory from a Rerun catalog
segment, and writes the run back as a `/world/runs/colsfm` layer of that segment.

The stage specs, the reverse-engineering record and the full measurement history live in the source fork,
[pablovela5620/pyCuSFM](https://github.com/pablovela5620/pyCuSFM/tree/feat/pycolmap-open-pipeline) (branch
`feat/pycolmap-open-pipeline`: `NOTES.md`, `docs/spec/`, `docs/full-pipeline-results.md`). The cuSFM binaries are not part
of this package.

## Run it

```bash
pixi run -e colsfm colsfm-raco-export        # once: RaCo-ALIKED + LightGlue+ ONNX graphs -> packages/colsfm/data/models
pixi run -e colsfm colsfm-download-flexglue  # once: FlexGlue's ONNX graphs (sha256-pinned) -> packages/colsfm/data/models/flexglue
pixi run -e colsfm colsfm-flexglue-fp8       # once: FlexGlue's FP8 matcher, calibrated on the Galileo frames (TensorRT backend)
pixi run -e colsfm colsfm-galileo            # r2b Galileo sample (downloaded by colsfm-download-galileo), default pipeline
pixi run -e colsfm colsfm-galileo-ablation   # COLMAP's own ALIKED + LightGlue, Ceres, no refinement, no loops
pixi run -e colsfm colsfm-robocap            # RoboCap fisheye segment s00000021 from the Spark catalog (slam_rs start)
pixi run -e colsfm colsfm-kitti06            # KITTI 06 from the Spark catalog (cuvslam layer start)
pixi run -e colsfm colsfm-walkthrough        # Galileo stage by stage, as a Rerun recording
```

`python tools/run.py catalog --help` lists the catalog source options (dataset, segment, cameras, stereo pairs, the entity
whose temporal `Transform3D` is the initial trajectory, stride, trim) and every pipeline knob.

## The default pipeline and its ablations

| Knob | Default | Ablation |
|---|---|---|
| `--features-backend` | `raco` (RaCo-ALIKED, TensorRT) | `pycolmap` (COLMAP's ALIKED through ONNX Runtime; no engine) |
| `--matching-backend` | `flexglue` (FlexGlue on TensorRT, covisibility gate) | `raco` (LightGlue+, TensorRT, real SSC); `pycolmap` (COLMAP's LightGlue); `flexglue_ort` (COLMAP's ONNX Runtime FlexGlue, parity reference, image folders only) |
| `--ba-backend` | `caspar` (COLMAP's GPU bundle adjuster; reduced camera system, no closing Ceres solve) | `ceres`; `--caspar-ceres-polish` adds one Ceres global bundle adjustment after CASPAR; `--caspar-option reduced_camera_system=false` runs CASPAR's full-system PCG |
| `--optimize-extrinsics` | on (regularised, cuSFM's priors) | `--no-optimize-extrinsics`, `--no-regularised-extrinsics` |
| `--loop-closure` | on | `--no-loop-closure` |

The RaCo graphs are exported, not shipped: a run that cannot find them **stops** and names `colsfm-raco-export` (a silent
change of extractor would be a different experiment under the same name). TensorRT 11 builds strongly typed engines; both
graphs run in fp32/TF32 (the `fp16-trunk` rewrite fails to parse on RaCo's mixed Float/Half Add, a follow-up). Engines are
cached per machine by trtkit.

## What it is built on

| Part | From |
|---|---|
| pycolmap with CASPAR + `OPENCV_FISHEYE` and fixed-calibration `FULL_OPENCV` Caspar adapters | `recipes/colmap-src` (COLMAP 4.2.0, CUDA 13, sm_120 / sm_121) |
| RaCo-ALIKED / LightGlue+ torch modules (export only) | `recipes/lightglue-onnx-src` (fabio-sim/LightGlue-ONNX @ d12b4ba) |
| TensorRT engine builds | `packages/trtkit` |
| Catalog rig, calibration and video readers, Umeyama | `packages/simplecv` |
| Pose graph and extrinsic refinement | pyceres 2.6 |

Where it sits: `pysfm` is COLMAP's own incremental/global SfM from video (no initial trajectory, pycolmap 4.0 cpu);
colsfm is cuSFM-style mapping that needs a starting trajectory — `slam_rs` (RoboCap), cuVSLAM (KITTI) or any
`/world/runs/<source>/rig` layer.

## Tests

```bash
pixi run -e colsfm-dev --frozen tests              # unit lane (synthetic scenes, no GPU or data)
pixi run -e colsfm-dev --frozen tests-integration  # GPU, Galileo sample, RaCo engines, catalog
pixi run -e colsfm-dev --frozen tests-golden       # Galileo + KITTI 06 parity bands
pixi run -e colsfm-dev --frozen gate
```

## TODO

TODO: one package, two modes. pysfm (plain pycolmap: no prior trajectory, intrinsics estimated, rig discovered) and
colsfm (cuSFM-style refinement: needs rig, calibration and a starting trajectory) are complementary. Pablo leans towards
one package with two modes, for example pysfm bootstrapping the start trajectory that colsfm refines, sharing one
rig-layer contract (rrd in, rrd out). Not decided: first define what each mode does and which features we want.

The other half: [pysfm](../pysfm/README.md).
