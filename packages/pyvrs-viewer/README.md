# pyvrs-viewer

Python port of [rerun-io/cpp-example-vrs](https://github.com/rerun-io/cpp-example-vrs) using [pyvrs](https://github.com/facebookresearch/pyvrs). Converts [VRS](https://github.com/facebookresearch/vrs) sensor recordings to Rerun `.rrd` files with AV1 video encoding for **13-42x compression**.

[![Rerun](https://img.shields.io/badge/rerun-0.30+-blue)](https://rerun.io)

## Features

- **Camera streams**: JPEG/RAW/video codec images logged as `rr.VideoStream` (AV1) or `rr.EncodedImage` (JPEG passthrough)
- **IMU data**: Accelerometer, gyroscope, magnetometer logged via `rr.send_columns()` (batch)
- **AV1 NVENC encoding**: Hardware-accelerated on NVIDIA GPUs (5000+ fps encode), with libsvtav1 CPU fallback
- **Parallel pipeline**: turbojpeg YUV decode (8 threads) overlapped with NVENC encode
- **Dynamic blueprint**: Auto-arranges camera views, IMU plots, and metadata panels

## Quick Start

```bash
pixi run -e pyvrs-viewer vrs-to-rrd-quest
```

On the first run, example VRS files are automatically downloaded from the [Hot3D dataset](https://www.projectaria.com/datasets/hot3d/). Subsequent runs skip the download.

## Usage

### Demo tasks (Hot3D example data)

```bash
pixi run -e pyvrs-viewer vrs-to-rrd-quest    # Quest: 2 mono SLAM cameras (~2.7 GB download)
pixi run -e pyvrs-viewer vrs-to-rrd-aria     # Aria: 3 cameras + 2 IMUs (~1.7 GB download)
```

### Custom VRS files (via task)

```bash
# Save to .rrd (AV1 encoded)
pixi run -e pyvrs-viewer vrs-to-rrd -- --vrs-path /path/to/file.vrs --rr-config.save output.rrd

# View live in Rerun viewer
pixi run -e pyvrs-viewer vrs-to-rrd -- --vrs-path /path/to/file.vrs

# JPEG passthrough (no encoding, larger files)
pixi run -e pyvrs-viewer vrs-to-rrd -- --vrs-path /path/to/file.vrs --no-encode-video

# H265 instead of AV1
pixi run -e pyvrs-viewer vrs-to-rrd -- --vrs-path /path/to/file.vrs --video-codec H265
```

### Custom VRS files (via python directly)

You can also run the CLI script directly in the pixi environment without a task:

```bash
pixi run -e pyvrs-viewer python packages/pyvrs-viewer/tools/demos/vrs_to_rrd.py \
  --vrs-path /path/to/file.vrs \
  --rr-config.save output.rrd
```

### All available tasks

```bash
pixi task list -e pyvrs-viewer
```

## CLI Options

```
--vrs-path PATH           Path to the input .vrs file (required)
--rr-config.save PATH     Save .rrd to file (default: opens viewer)
--rr-config.connect       Connect to existing Rerun viewer
--rr-config.headless      Run without viewer
--encode-video / --no-encode-video
                          AV1 video encoding (default: on)
--video-codec {H265,AV1}  Video codec (default: AV1)
--decode-threads N        Parallel JPEG decode threads (default: 8)
```

## Performance

Tested on RTX 5090 with Hot3D VRS files:

| Device | VRS Size | AV1 Time | AV1 RRD | Compression | JPEG Time | JPEG RRD |
|--------|----------|----------|---------|-------------|-----------|----------|
| Quest (2 cams) | 0.8-2.7 GB | 2-5s | 31-66 MB | 21-41x | 0.3-0.9s | 0.8-2.7 GB |
| Aria (3 cams + IMU) | 0.8-1.8 GB | 4-9s | 50-110 MB | 15-16x | 5-11s | 0.8-1.8 GB |

AV1 encoding is **faster than JPEG passthrough** on Aria files because `send_columns()` batch IMU logging eliminates the per-record overhead.

## Running the Benchmark

The benchmark script tests 5 Quest + 5 Aria VRS files in both AV1 and JPEG modes.

### 1. Get the download URLs

Download the Hot3D download URL JSON files from [projectaria.com/datasets/hot3d](https://www.projectaria.com/datasets/hot3d/) (requires accepting the license agreement). Place them in the benchmark directory:

```
packages/pyvrs-viewer/tools/bench/
  hot3dquest_download_urls.json   # Hot3DQuest_download_urls.json
  hot3daria_download_urls.json    # Hot3DAria_download_urls.json
```

### 2. Run the benchmark

```bash
pixi run -e pyvrs-viewer benchmark
```

This will:
- Download the first 5 VRS files from each JSON (~18 GB total, cached for re-runs)
- Run both AV1 encode and JPEG passthrough on each file
- Print a results table and save it to `data/benchmark/results.md`

## Building the C++ Reference

The original C++ VRS viewer is included as a submodule for comparison:

```bash
cd packages/pyvrs-viewer/thirdparty/cpp-example-vrs

# Install C++ dependencies and build
pixi install
pixi run build

# Run (opens Rerun viewer)
pixi run example /path/to/file.vrs
```

Note: The C++ version decodes every JPEG frame and logs as `rr.Image` (no video encoding). It takes ~19s for a Quest VRS file vs ~5s for the Python AV1 pipeline.

## Architecture

```
src/pyvrs_viewer/
  vrs_to_rerun.py      # Pipeline orchestration: parallel decode + streaming encode
  frame_player.py      # Camera handler: VideoStream (AV1/H265) or EncodedImage (JPEG)
  imu_player.py        # IMU handler: send_columns batch or row-by-row
  video_encoder.py     # AV1/H265 encoder: NVENC hardware → CPU fallback
  blueprint.py         # Dynamic Rerun blueprint generation
```

### Pipeline (encode_video=True)

```
Phase 1: Read VRS records
  ├── Image records → collect JPEG bytes
  ├── IMU records → accumulate for batch logging
  └── Config/state → log immediately

Phase 2+3: Parallel decode overlapped with encode
  ├── ThreadPoolExecutor (8 threads) → turbojpeg YUV decode
  └── Main thread → NVENC AV1 encode + rr.log(VideoStream)

Phase 4: Batch IMU logging via rr.send_columns()
```

## Development

```bash
# Install dev environment (adds ruff, pytest, beartype, pyrefly)
pixi install -e pyvrs-viewer-dev

# Lint
pixi run -e pyvrs-viewer-dev lint

# Test
pixi run -e pyvrs-viewer-dev tests

# Typecheck
pixi run -e pyvrs-viewer-dev typecheck
```
