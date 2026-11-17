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
valid radius (Basalt's `rpmax` 0, no bound). The IMU model slam-rs reads is
Basalt's, the one RootSLAM ran EuRoC with, imported as the `sensor_metadata`
layer rather than VSLAM-LAB's OKVIS sigmas. The registration root holds `base`
and `gt` as links to the conversion's files and that `sensor_metadata`; it is
registered once, skipping duplicates:

```bash
export DATAFORGE_OUTPUT_ROOT=<registration root>
python ../dataforge/tools/apps/register.py vslam-lab --benchmark euroc        # dataforge env: base + gt
pixi run -e slam-rs slam-rs-import-imu-calibration --dataset vslam-lab-euroc \
  --catalog rerun+http://127.0.0.1:51235 --applied-time-shift-ns 0 --no-register \
  --calibration <rootslam>/data/euroc/euroc_ds_calib.json --output-dir $DATAFORGE_OUTPUT_ROOT/sensor_metadata
python ../dataforge/tools/apps/register.py vslam-lab --benchmark euroc        # adds sensor_metadata
```

The eleven segments are listed in `benchmarks.toml` with no baseline yet; score
them with RootSLAM's `evaluate.py` against `mav0/state_groundtruth_estimate0/data.csv`.

## The gate

```bash
pixi run -e slam-rs slam-rs-gate --tier smoke              # two short clips
pixi run -e slam-rs slam-rs-gate --tier release            # three longer clips (MIO07, MIO14, MGO07)
pixi run -e slam-rs slam-rs-gate --tier listed             # the other five MSD clips and EuRoC's eleven (no baseline yet)
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
`msd-index`, `msd-g2`, `msd-odyssey` (and `vslam-lab-euroc`) under the same
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
