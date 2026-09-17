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
into `blueprints/` beside them. Each download leaves a
marker file, so pixi skips one that is already there.

## The gate

```bash
pixi run -e slam-rs slam-rs-gate --tier smoke              # two short clips
pixi run -e slam-rs slam-rs-gate --tier release            # three longer clips (MIO07, MIO14, MGO07), CPU lane
pixi run -e slam-rs slam-rs-gate --tier listed             # the other five gated clips
pixi run -e slam-rs slam-rs-gate --tier release --gpu      # any tier on the GPU lane
```

Every clip must track every frameset, associate enough poses with ground truth,
and stay within 10 % of its baseline's RMSE. `benchmarks.toml` holds the
baselines by lane, profile and host; the speed clause (10 % over the median
tracker call) is only gated on the host that recorded the baseline, and
reported everywhere else. The README's per-recording table is the same
measurement run over every segment on the catalog rather than the ten gated
ones. A tier is exactly its own clips: `smoke` is two, `release` three, `listed` five.

## Replaying and scoring one recording

```bash
cd packages/slam-rs
pixi run -e slam-rs python tools/apps/replay.py --stage vio --segment msd-g2__MGO_others__MGO07_mapping_easy
pixi run -e slam-rs python tools/apps/replay.py --stage vio --gpu --profile reference --segment msd-g2__MGO_others__MGO07_mapping_easy
```

`--stage input` (the default) logs only what the estimator is fed; `--stage
frontend` runs the optical flow and draws its keypoints; `--stage vio` is the
whole pipeline. Long segments want `--max-framesets`. Without a display, add
`--rr-config.headless --rr-config.save out.rrd`.

## Another catalog

Every tool takes `--catalog <url>`; the default in `slam.toml` is the local
server the quickstart starts. A catalog that already holds the datasets
`msd-index`, `msd-g2` and `msd-odyssey` under the same recording ids works
unchanged.

## Where the recordings came from

The HuggingFace dataset [pablovela5620/msd-rrd](https://huggingface.co/datasets/pablovela5620/msd-rrd)
is the upstream [Monado SLAM Dataset](https://huggingface.co/datasets/collabora/monado-slam-datasets)
converted once by the `dataforge` package in this repository: PNG frames encoded
to AV1 upright per the factory calibration, IMU and ground truth at native rate,
the ground-truth world-up axis measured. The converter needs a Linux host with an
NVIDIA AV1 encoder and pulls the 350 GB upstream one sequence at a time; see
[packages/dataforge/docs/msd.md](../../dataforge/docs/msd.md). Nobody needs to
run it to use slam-rs.
