# RoboCap direct recording components

Status: experimental direct camera/sensor capture and isolated live SLAM,
tested on Cap A and Cap B. Control Center integration and a successful real
ten-minute rollover remain unfinished. Cap B pose accuracy is unresolved with
the borrowed Cap A calibration.

The library accepts encoded H.264 access units and original sensor samples
directly. Its production dependencies contain no MP4 or SQLite input adapter.
The separate `validate_cap_fixture` example reads existing recordings for host
validation; SQLite is a development dependency for that example.

## DataForge display in live files

Prepare the same calibrated camera transforms, distortion-aware pinholes, cap
mesh, overview/follow views, six camera panes and IMU plots as DataForge:

```sh
pixi run -e dataforge-dev --frozen robocap-live-display \
  --output /tmp/dataforge-live-display.rrd
```

Stage that small static asset beside the ARM executable and set
`ROBOCAP_DISPLAY_RRD` to its device path when launching `robocap-direct` under
the supervised capture handoff. The recorder embeds it in every new part; no
post-recording conversion or separate blueprint download is needed.
The asset's source device must match the recording's calibration source.

Live poses animate `/world/rig_00`, so the cameras and mesh follow together.
Connected edges at `/world/runs/slam_rs/{trajectory,trail}` accumulate in the
overview and show a ten-second trail in the follow view. Edges are logged once,
with bounded writer state; no line bridges a reported tracking gap.

On 2026-09-13 a fresh 30-second Cap A capture was opened directly in a fresh
viewer and visually checked with this embedded display. It retained all six
camera and sensor streams and produced 258 live updates. That establishes
capture/display integration, not trajectory accuracy or the 15 Hz target.

## Live SLAM performance diagnostics

The cap runs a Rust recorder and an isolated Rust SLAM worker. A native V4L2
adapter supplies NV12 frames; SLAM takes downsampled luma before H.264 encoding.
Python prepares display assets off the cap. There is no MP4 decoding or SQLite
conversion in this live path.

SLAM accepts matched four-camera sets at approximately 15 Hz from the 30 Hz
source. Keep that input limit: the performance target is 33–50 ms of compute per
update, leaving headroom within the 66.7 ms arrival interval. All six recordings
retain their full source rate. Both verified cap hostnames pin the SLAM child
and its four frontend threads to CPUs 4–7; capture keeps normal scheduling.

Set `ROBOCAP_SLAM_PROFILE=1` for per-update JSON `slam_profile` lines on stderr.
They expose the core's existing nanosecond timers, input age, and keyframe flag.
`measure_ns` includes optimization and marginalization; `optimize_ns` includes
its detailed solver stages. Do not sum these nested totals. Recorder checkpoint
wall time is reported separately as `capture_checkpoint_profile`. Save stderr
on the cap during measurements so an SSH log consumer cannot stall the process.

Use the `robocap-direct-build` task below for deployment comparisons. The
development profile is optimized at level 2 but
retains debug assertions; release uses level 3 and thin LTO when the cross-build
script's explicit target rustflags override the older ARM level-2 workaround.
Retain the exact build command with performance evidence. Compare identical
inputs with `compare_live_adapter` before attributing scene-dependent speed
changes to a build. Its direct-estimator comparison also checks pose parity.

On 2026-09-13, Cap A release trials produced 899 updates per 60-second capture
(15 Hz), with mean compute 35.8–36.9 ms. One run had 884 visually supported
updates and retained about 1,798 frames per camera. Slow updates still exceeded
66.7 ms. A fixed-input replay isolated gains from A76 affinity and release
settings, with identical pose CSVs across all five runs. Synchronous storage
checkpoints also delayed report publication by about 200 ms each second while
the isolated estimator continued. This is short-run throughput evidence;
long-session clock safety, thermal headroom, and pose accuracy remain separate
acceptance gates. The measurement logs live with the project notes, outside this
repository.

## Implemented boundaries

- `DirectWriter`: six canonical camera paths, three independent gyro/accel
  pairs, and one magnetometer. Preserves int32 counts, int64 source timestamps,
  and source sequences. Raw plots use a `/raw` child; physical-unit Scalars
  require an explicitly supplied positive scale.
- `DurableRrdSink`: creates an exclusive `.rrd.partial`, propagates storage
  errors through checkpoints and completion, syncs contents, and publishes
  without replacing an existing recording. Parent-directory synchronization
  follows publication. A checkpoint can be read before a footer is written.
- `SegmentedWriter`: one session clock across parts. A camera enters the next
  part at an IDR containing SPS/PPS. Sensors cross the same time boundary without
  interpolation. At most two writers remain open during the transition; the
  old part closes after all 13 streams cross. Estimator state belongs outside
  this file-rotation owner.
- `SamplePipeline`, behind `gstreamer-capture`: GStreamer appsink transport with
  a bounded queue, complete owned buffers, PTS, source offsets, EOS/error
  reporting, and resource release. PTS is still a pipeline timestamp; this
  component does not claim that it is an original device-clock timestamp.
- `IioScanLayout`: reads and validates enabled ICM42688/MMC5983MA scan metadata
  without changing the device. Decodes signed counts, optional temperature and
  aligned integer timestamps. Retains the declared clock. It does not configure
  buffers, read live samples, convert clocks, or establish physical MAG units.

The coordinator must call `DirectWriter::checkpoint` at least once per second.
The timeout bounds SDK dispatch waiting; it cannot cancel an OS storage syscall.
No physical power-loss guarantee has been established.

## Run the host checks

From the monorepo root:

```sh
pixi run -e robocap-recorder --frozen robocap-recorder-test
pixi run -e robocap-recorder --frozen robocap-recorder-clippy
pixi run -e robocap-recorder --frozen cargo fmt --manifest-path packages/slam-rs/Cargo.toml -p robocap-recorder --check
```

The failure tests use an isolated child process with a file-size limit. The
ignored child helper is explicitly invoked by its parent test.

The other ignored test, `cap_b_encodes_six_generated_streams_without_losing_frames`,
is a device integration check. Run it explicitly only on an idle Cap B with its
Rockchip MPP plugin. It generates six 1080p/30 streams for ten seconds and checks
every output count, timestamp interval and periodic keyframe. It acquires no
camera, trigger or IIO device. A host run leaving this test ignored is not device
validation.

Cap B validation on 2026-09-13 passed:

- Nine writer/completion/rotation tests on both tmpfs and the recording exFAT
  partition, including incomplete files and simulated storage failure.
- Raw GStreamer transport and two IIO decoder tests using synthetic samples.
- Six concurrent hardware H.264 encoders: 300 frames and ten keyframes per
  stream in 10.094 seconds. `h264parse config-interval=-1` and byte-stream/AU
  caps are required after `mpph264enc`: the first trial without parsing failed
  the first-keyframe assertion. This test does not establish camera capture,
  SLAM throughput, or long-session resource stability.

## ARM build

`robocap-cross` contains the local ARM cross-compiler and Rust target. The
release recorder build is part of the repo:

```sh
pixi run -e robocap-cross --frozen robocap-direct-build /path/to/cap-runtime-sysroot
```

This prepares the workspace's patched dependencies and builds
`packages/slam-rs/target/aarch64-unknown-linux-gnu/release/robocap-direct`.
Copy that executable and the generated display asset to the cap. The build
task does not connect to a device, transfer files, or take capture ownership.
The verified idle-only handoff scripts are still trial evidence under `/tmp`;
there is no production deployment/start service yet.

The writer-only checks can be built with:

```sh
pixi run -e robocap-cross --frozen cargo test \
  --manifest-path packages/slam-rs/Cargo.toml --locked \
  --target aarch64-unknown-linux-gnu -p robocap-recorder --tests --no-run
```

The optional GStreamer build also needs Cap B's runtime libraries in a local
linker sysroot, with pkg-config metadata for GStreamer 1.24.11 and GLib 2.76.1.
Set `PKG_CONFIG_ALLOW_CROSS`, `PKG_CONFIG_LIBDIR`, `PKG_CONFIG_SYSROOT_DIR` and
target linker search paths to that sysroot. Use matching libc compatibility
libraries and the glibc 2.34 startup objects in the cross environment; mixing old
startup objects with the cap's glibc 2.41 fails to link. Device libraries were
copied for linking only; none were replaced on the cap. The sysroot is a
build-host artifact outside this repository; `scripts/build-arm.sh` (the
`robocap-direct-build` task) takes its absolute path as its one argument and
checks the directories it needs before building.

To validate original Cap B samples against this writer:

```sh
pixi run -e robocap-recorder --frozen cargo run \
  --manifest-path packages/slam-rs/Cargo.toml --locked \
  -p robocap-recorder --features gstreamer-capture \
  --example validate_cap_fixture -- SESSION_DIRECTORY OUTPUT.rrd DURATION_SECONDS
```

The directory must contain exactly one segment's six video files and four
sensor databases. The example uses `ffprobe` to read each camera's own epoch.
It retains raw sensor counts without borrowing scales or calibration from
another device. Existing output files are refused.

## Remaining device work

Direct capture, isolated four-camera VIO, and DataForge display embedding have
run on both caps. Trial scripts temporarily take camera/trigger/IIO ownership
from the idle vendor recorder and restore it afterward. No firmware or boot
configuration was changed.

Still needed: production start/stop ownership and the Control Center selector,
DataForge part discovery, two complete consecutive ten-minute device parts,
boundary/fault checks, and sustained thermal/latency validation. Storage
checkpoints currently delay report publication. IIO clock guards have also
stopped device trials; selecting `monotonic` alone does not close that issue.

Cap B's factory calibration remains missing. The user-authorized Cap A
placeholder is marked in recordings and does not establish Cap B pose accuracy.
Physical power-cut tests are excluded by the user.
