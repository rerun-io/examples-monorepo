# Monado SLAM Datasets: what was measured, and what was decided

The [README](../README.md#monado-slam-datasets-msd) says how to run the converter
and what comes out. This note is the other half: the claims the converter makes
about a corpus that does not document itself, the evidence behind each, and the
format decisions that are not obvious from the code. Nothing here is repeated in
the README or in `datasets/msd.py` — that module's docstring covers the raw
layout and the invariants, and the constants it defines carry the one-line
version of each answer.

Upstream is [`collabora/monado-slam-datasets`](https://huggingface.co/datasets/collabora/monado-slam-datasets)
(CC-BY 4.0): three VR headsets, ~350 GB of per-sequence zips holding PNG frames
plus EuRoC-style csv sensor logs.

## The world up axis is measured, not documented

MSD states no world axes at all. The Index's ground truth comes from SteamVR
Lighthouse; the G2's and the Odyssey+'s from a MoCap rig with no documented
convention. Gravity settles it: an accelerometer at rest measures the *reaction*
to gravity, so its reading points **up**, and rotating the first samples into the
world with the ground truth's own orientation (`world_R_rig @ a_rig`) and
averaging gives a vector along the world's up axis. `measured_world_up` does
exactly that over `MEASURED_UP_WINDOW_NS` (2 s), because a headset is typically
still at the start of a capture and the mean gets noisier the longer the window.

Measured on each device's `*09_short_1_updown` sequence:

| device | axis | fraction of \|g\| | ground truth from |
| --- | --- | --- | --- |
| `index` | `+y` | 0.96 | SteamVR Lighthouse |
| `g2` | `+y` | 0.98 | undocumented MoCap |
| `odyssey` | `+y` | 0.93 | undocumented MoCap |

All three therefore state `RIGHT_HAND_Y_UP` (`RUB`) at the root of their `gt`
layer. The measurement is **not** a restatement of the raw samples: over the
Index's window the accelerometer's own mean points along the headset's `-x`, and
only the gt rotation (123–142° from identity there) turns it into world `+y`.

The answer is fixed per device in `MSD_DEVICES` rather than applied per sequence,
because every rrd of a device must carry the same root `ViewCoordinates` — and
because `register` builds a device's default blueprint from the registry alone,
with no sequence on disk. Every `convert` re-measures its own sequence, records
`measured_up` and `measured_up_fraction` in the `gt` properties, and **warns** on
a disagreement instead of silently reorienting one rrd out of step with the rest.

## The follow frame comes from the baseline, not from image-up

`follow_frame` derives a headset's forward and up from its front stereo pair:
forward is the mean optical axis (camera `+z` in RDF, averaged to cancel each
camera's slight outward yaw), and up is `right × forward` with `right`
Gram-Schmidt'd off the pair's **baseline**.

The baseline and not image-up, because the G2 mounts all four of its cameras
rolled a quarter turn: its image-up is rig `+x` while its up is rig `-y`. Up from
image-up matches the baseline within about a degree on the Index and the
Odyssey+, and is 90° out on the G2. Two independent checks agree with the
baseline answer on every device — the raw accelerometer mean points the same way,
and it puts the G2's front pair 15.6° *below* the horizon and the Odyssey+'s
21.3° below, which is where tracking cameras are aimed.

Derived on the real `calibration.json` files (2026-09-06), and then written down
as each device's `follow`, for the same registry reason as the up axis. The
checked-in fixtures under `tests/fixtures/msd/` are verbatim copies of those
files, so `test_msd` re-derives the pair and holds the constants to 0.05°;
`convert` re-derives it per sequence and warns past `FOLLOW_FRAME_TOLERANCE_DEG`
(5°).

## Format decisions

**One clock, no resampling.** Every csv timestamp is nanoseconds on one monotonic
device clock (values around 1e13, not a Unix epoch). `video_time` is that clock
minus `t0`, and `t0` is the earliest sample of **any** stream *including* `gt` —
the two layers must share an origin, and the gt file is usually the earliest
stream. `duration_ns` is deliberately **not** bounded by gt: it describes the
sensor layer. Nothing anywhere is resampled.

**`rig_T_cam` needs no inversion.** basalt's `T_imu_cam` is the camera's pose in
the IMU frame, and MSD's rig frame *is* the IMU frame, so the rig node states
`reference = "imu_00"` and the calibration's pose goes in as-is. (simplecv's
RoboCap loader is one inversion away: Kalibr writes the inverse `T_cam_imu`.)

**Both camera models are kept as upstream ships them** — `kb4` on the Index, the
`pinhole-radtan8` rational model on the G2 and Odyssey+ — so each camera node
states its `camera_model` and a radtan8 camera also carries `rpmax` as
`distortion_valid_radius`. `rpmax` is validated as **present**, never defaulted:
the Odyssey+ really writes `"rpmax": 0.0`, which is a claim, whereas a missing
key is a truncated file, and a zero default would make the two the same thing.

**A repaired quaternion is not a dropout.** A `gt` row whose quaternion is not
unit-norm (upstream writes `0 0 0 0`) breaks the rotation chain from that row on,
so every child frustum of the rig stops tracking. The repair is per-row: identity
rotation, translation untouched, and the count reported as `num_sanitized` rather
than hidden — the same fix slam-evals settled on for ROVER-T265. This is a
**source-side** repair and is distinct from an emitted NaN dropout (exoego:v2 §2),
which hides the rig for a frame; MSD emits none.

**Only `data.csv` is read.** Where a stream also ships `data.raw.csv` /
`data.extra.csv` siblings, those are ignored.

**Three sequences are Info-ZIP multi-volume sets** (`.z01 … .zip`) — the
`*_long_session` archives of all three headsets — which Python's `zipfile` cannot
read at all: it opens the closing volume, whose central directory is intact, then
fails on the first member spanning a boundary. Those go through the `7zz` CLI (the
conda-forge `7zip` package), one directory extracted at a time, so peak scratch
is one camera's PNGs rather than the whole sequence.

**One dataset per device.** A catalog dataset holds one default blueprint, hence
one camera layout, and the three headsets have two, four and two cameras — so
`--device` picks the corpus *and* the catalog dataset.
