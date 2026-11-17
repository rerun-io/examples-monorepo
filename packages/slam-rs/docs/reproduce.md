# Reproduce the benchmark

The README quickstart runs three short clips. This page covers the rest of the
dataset, the release gate, and where the recordings came from.

## The whole dataset

```bash
pixi run -e slam-rs slam-rs-download-all       # all 64 recordings, about 15.5 GB
pixi run -e slam-rs slam-rs-register           # idempotent: only new files are added
```

The downloads land in `packages/slam-rs/data/msd-rrd/`, one directory per
layer (`base`, `gt`, `sensor_metadata`); registration writes the device blueprints
into `blueprints/` beside them. A download is skipped when every layer of its
recordings is already there, so it is safe to re-run and it picks up clips added
to a tier.

## EuRoC

EuRoC MAV (RootSLAM's EMH/EV1/EV2 groups, 11 sequences) is the catalog dataset
`vslam-lab-euroc`: dataforge's VSLAM-LAB conversion (`dataforge-vslam-lab-rrd`). Its `video_time` is the benchmark's own nanosecond clock, which
`slam.toml` states (`video_time_is_absolute`), and its cameras are radtan with no
valid radius (Basalt's `rpmax` 0, no bound).

The `sensor_metadata` layer carries part of Basalt's EuRoC calibration
(`crates/slam-rs/tests/fixtures/euroc_ds_calib.json`, the file RootSLAM ran EuRoC
with), in place of VSLAM-LAB's OKVIS sigmas:

- carried: the four noise sigmas (`accel_noise_std` 0.016, `gyro_noise_std`
  0.000282, `accel_bias_std` 0.001, `gyro_bias_std` 0.0001) and the update rate
  (200 Hz); the applied time shift is 0;
- not carried: the static biases `calib_accel_bias` (y 0.12 m/s²) and
  `calib_gyro_bias` (z 0.077 rad/s), which Basalt subtracts from every sample.
  The catalog path starts from zero static bias and feeds the raw IMU;
- not carried: the double-sphere cameras and their extrinsics. The cameras
  are the conversion's radtan intrinsics and extrinsics.

The registration root holds `base` and `gt` as links to the conversion's files
and that `sensor_metadata`; it is registered once, skipping duplicates (from the
repository root; the root must be an absolute path, the two tasks run in
different directories):

```bash
export DATAFORGE_OUTPUT_ROOT=<absolute registration root>
pixi run -e dataforge dataforge-register vslam-lab --benchmark euroc     # base + gt
pixi run -e slam-rs slam-rs-import-imu-calibration --dataset vslam-lab-euroc \
  --catalog rerun+http://127.0.0.1:51235 --applied-time-shift-ns 0 --no-register \
  --calibration crates/slam-rs/tests/fixtures/euroc_ds_calib.json --output-dir $DATAFORGE_OUTPUT_ROOT/sensor_metadata
pixi run -e dataforge dataforge-register vslam-lab --benchmark euroc     # adds sensor_metadata
```

Known data issue: the conversion keeps cam1 by frame count, not by stamp. The
raw cameras share every stamp, but the converted MH_04_difficult loses one
stereo pair and V2_03_difficult its last 295 (1626 of 1921 are fed; RootSLAM's
scorer reports completion 0.83). Re-pairing needs a new conversion and a layer
replacement on the shared catalog, so it is deferred.

The eleven segments are the `euroc` tier of `benchmarks.toml`, with no baseline
yet; score them with RootSLAM's `evaluate.py` against
`mav0/state_groundtruth_estimate0/data.csv`.

## TUM-VI

TUM-VI's 512x512 16-bit release (RootSLAM's TC/TM/TR/TS groups: corridor 1-5,
magistrale 1-6, room 1-6, slides 1-3) is the catalog dataset `tum-vi`, written
by dataforge's `tum-vi` converter from the raw ASL tree. Its `video_time` is the
dataset's own nanosecond clock (`video_time_is_absolute`). Only frames that both
cameras took at the same stamp are kept, so framesets match exactly (magistrale4
drops one cam0 frame).

What reaches slam-rs:

- cameras: TUM-VI's own Kalibr calibration (`dso/camchain.yaml`), equidistant =
  KB4, with `T_cam_imu` inverted into the IMU (rig) frame. RootSLAM ran the
  double-sphere calibration `tumvi_512_ds_calib.json` instead. slam-rs parses a
  DS calibration but does not project it yet (the fisheye-360 stack adds that),
  so the two runs use different camera models and extrinsics: a caveat on every
  comparison;
- IMU noise: `dso/imu_config.yaml` (Kalibr's inflated values: accel 0.0028,
  gyro 0.00016, accel walk 0.00086, gyro walk 0.000022), on the
  `sensor_metadata` layer with `applied_time_shift_ns` 0. RootSLAM ran Basalt's
  values from `tumvi_512_ds_calib.json` (0.016, 0.000282, 0.001, 0.0001);
- not carried: the ds calibration's static biases (`calib_accel_bias`,
  `calib_gyro_bias`) and its vignette; the 16-bit frames reach the feed as their
  upper 8 bits (AV1, 8-bit).

`configs/tumvi_config.json` is RootSLAM's `tumvi_512_config.json`, key by key,
for the keys slam-rs reads: it is `msdmg_config.json` with a 40 px detection
grid and a 320 px fisheye mask (the recall patch norms are the MSD values
already). Not carried: `vio_use_lm` false and `vio_lm_lambda_initial` 1e-7
(slam-rs parses `vio_use_lm` but always runs the damped solve, so it keeps
msdmg's 1e-4), and every key slam-rs does not read (`map_*`, `mapper_*`,
`loop_closing_*`, `vio_filter_iteration`, `vio_outlier_threshold`, the LM damping
variants, `optical_flow_predict_with_imu`).

Ground truth is `mav0/mocap0/data.csv` (world_T_imu0, every sample). The
motion-capture system covers one room, so the corridor, magistrale and slides
runs have ground truth at their start and end only; the `gt` layer's properties
list those spans (`property:gt:span_start_ns`, `span_stop_ns`). On TM, ATE
measures whether the start-to-end loop closed.

Convert and register once (from the repository root; the output root must be an
absolute path on local disk):

```bash
export DATAFORGE_OUTPUT_ROOT=<absolute rrd root>
pixi run -e dataforge-dev dataforge-convert --jobs 2 tum-vi --root <dir holding dataset-*_512_16>
pixi run -e dataforge-dev dataforge-register tum-vi --root <same dir>   # base + gt + sensor_metadata
```

The twenty segments are the `tumvi` tier of `benchmarks.toml`, with no baseline
there; score them with RootSLAM's `evaluate.py` against `mav0/mocap0/data.csv`.

## The gate

```bash
pixi run -e slam-rs slam-rs-gate --tier smoke              # two short clips
pixi run -e slam-rs slam-rs-gate --tier release            # three longer clips (MIO07, MIO14, MGO07)
pixi run -e slam-rs slam-rs-gate --tier listed             # the other five gated clips
pixi run -e slam-rs slam-rs-gate --tier euroc              # EuRoC's eleven (needs vslam-lab-euroc; no baseline yet)
pixi run -e slam-rs slam-rs-gate --tier tumvi              # TUM-VI's twenty (needs tum-vi; no baseline yet)
pixi run -e slam-rs slam-rs-gate --tier release --backend cpu   # any tier on the CPU lane
```

The gate builds the core first and runs on the GPU lane when the host has a
usable adapter (`--backend auto`, the default); it prints the lane it chose and
records it in the report, and each lane is scored against its own baselines.
Decoding stays on the CPU's single-threaded dav1d path, on which the baselines
were measured.

The acceptance rules (tracking, association, the 10 % RMSE band, speed gated on
the baseline host only, the no-baseline case) are in one place:
[docs/design-notes.md, The gate](design-notes.md#the-gate). The per-recording
table in [benchmarks.md](benchmarks.md) is the same measurement run over every
segment on the catalog rather than the ten gated ones.

## Replaying and scoring one recording

```bash
cd packages/slam-rs
pixi run -e slam-rs slam-rs-replay --stage vio --segment msd-g2__MGO_others__MGO07_mapping_easy
pixi run -e slam-rs slam-rs-replay --stage vio --backend cpu --profile reference --segment msd-g2__MGO_others__MGO07_mapping_easy
```

`--stage input` (the default) logs only what the estimator is fed; `--stage
frontend` runs the optical flow and draws its keypoints; `--stage vio` is the
whole pipeline. Long segments want `--max-framesets`. Without a display, add
`--rr-config.headless --rr-config.save out.rrd`.

## Another catalog

Every tool takes `--catalog <url>`; the default in `slam.toml` is the local
server the quickstart starts. A catalog that already holds the datasets
`msd-index`, `msd-g2`, `msd-odyssey` (and `vslam-lab-euroc`, `tum-vi`) under the same
recording ids works unchanged.

## Where the recordings came from

The HuggingFace dataset [pablovela5620/msd-rrd](https://huggingface.co/datasets/pablovela5620/msd-rrd)
is the upstream [Monado SLAM Dataset](https://huggingface.co/datasets/collabora/monado-slam-datasets)
converted once by the `dataforge` package in this repository: PNG frames encoded
to AV1 upright per the factory calibration, IMU and ground truth at native rate,
the ground-truth world-up axis measured. The converter needs a Linux host with an
NVIDIA AV1 encoder and pulls the 350 GB upstream one sequence at a time; see
[packages/dataforge/docs/msd.md](../../dataforge/docs/msd.md). Nobody needs to
run it to use slam-rs.
