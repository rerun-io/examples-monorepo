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
pixi run -e colsfm colsfm-raco-export        # once: RaCo-ALIKED + LightGlue+ ONNX graphs -> the model directory
pixi run -e colsfm colsfm-download-flexglue  # once: FlexGlue's ONNX graphs (sha256-pinned) -> <model directory>/flexglue
pixi run -e colsfm colsfm-flexglue-fp8       # once: FlexGlue's FP8 matcher, calibrated on the Galileo frames (TensorRT backend)
pixi run -e colsfm colsfm-galileo            # r2b Galileo sample (downloaded by colsfm-download-galileo), default pipeline
pixi run -e colsfm colsfm-galileo-ablation   # COLMAP's own ALIKED + LightGlue, Ceres, no refinement, no loops
pixi run -e colsfm colsfm-robocap            # RoboCap fisheye segment s00000021 from the Spark catalog (slam_rs start)
pixi run -e colsfm colsfm-kitti06            # KITTI 06 from the Spark catalog (cuvslam layer start)
pixi run -e colsfm colsfm-walkthrough        # Galileo stage by stage, as a Rerun recording
```

The model directory is `$COLSFM_MODEL_DIR`, else this checkout's `packages/colsfm/data/models` when it already holds
graphs, else `~/.cache/colsfm/models`, shared by every worktree on the host (`colsfm.resolve_model_dir`).

`python tools/run.py catalog --help` lists the catalog source options (dataset, segment, cameras, stereo pairs, the entity
whose temporal `Transform3D` is the initial trajectory, stride, trim) and every pipeline knob.
`--source.frame-stride` defaults to 2 (every second frame of the reference camera; on an Insta360 cube layer, whose
framesets are every `cube_stride`-th video frame, every second frameset): half the run, Sim(3) ATE within 5 % of stride 1
(MSD MGO05 -1.3 %, EuRoC V1_01 -0.9 %, KITTI 06 +1.7 %, TartanAir P004 cube +3.6 %). Pass `--source.frame-stride 1` for every frame.

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

## Two modes

colsfm has two modes on one camera contract (`/world/rig_NN/cam_MM` camera nodes, or a calibration layer):

- **`run catalog`** refines a starting trajectory with cuSFM's eight stages. It needs the rig's calibration and a
  starting trajectory: `slam_rs` (RoboCap), cuVSLAM (KITTI) or any `/world/runs/<source>/rig` layer.
- **`calibrate`** makes the calibration when there is none: each camera's intrinsics and its pose in the rig, from a
  window of the segment, as a calibration layer under `/world/runs/colsfm_calibrate/rig`. It does not refine a
  trajectory.

### calibrate

```bash
pixi run -e colsfm colsfm-calibrate-robocap    # RoboCap Cap A s00000021, its factory calibration ignored
python tools/run.py calibrate --help           # segment, cameras, trims, --options.camera-model, --options.num-rig-frames
```

300 rig frames (default) on the reference camera's packets, colsfm's RaCo + FlexGlue on TensorRT, every camera of a
rig frame matched against every camera of it and of the next 5; COLMAP's global mapper without a rig from seed
intrinsics, `apply_rig_config`, and the global mapper again on the rig with the extrinsics and the intrinsics free. About
8 minutes for 4 fisheyes on the 5090.

What it does not solve: the rig frame is the first camera, and the translations are in the reconstruction's own scale
(the layer's root says `metric_scale=False`). `run catalog --source.calibration-entity` refuses such a calibration,
because with a metric trajectory in another rig frame (slam_rs: the IMU frame) every camera would land in the wrong
place. slam-rs always needs a calibration and stops without one.

Lessons carried over from pysfm (deleted; its `docs/workflow.md`, sections 7 and 9):

- Without cross-camera pairs of the same and nearby rig frames, the mapping without a rig splits into one model per
  camera: on a 4 x 50-frame rig, 601 vs 2601 cross-camera pairs and 0 vs 196 of 200 images registered.
- COLMAP's mappers discard a rig whose `sensor_from_rig` is unknown, so the first mapping runs on single images and the
  rig enters only after it (`apply_rig_config` with the model).
- Judge the first model by the rig it seeds, not by how it looks; compare extrinsics up to scale.
- A fisheye without a focal prior gives degenerate pairs in COLMAP's verification (a fundamental matrix cannot recover
  its focal length), so the fisheye seed is a prior.

Checked against the factory Kalibr calibrations of two RoboCaps (`tests/test_calibrate_golden.py`, 300 rig frames, 4
fisheyes each): every image registers, and per camera the estimate reprojects at 1.5-1.9 px against 1.9-9.2 px with the
factory calibration held on the same observations (Cap A s00000021, Cap B s00000028).

## Tests

```bash
pixi run -e colsfm-dev --frozen tests              # unit lane (synthetic scenes, no GPU or data)
pixi run -e colsfm-dev --frozen tests-integration  # GPU, Galileo sample, RaCo engines, catalog
pixi run -e colsfm-dev --frozen tests-golden       # Galileo + KITTI 06 parity bands
pixi run -e colsfm-dev --frozen gate
```
