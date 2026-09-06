# dataforge

One package for dataset work: **download → convert → register → view**.
Train and eval are the designed-for second half.

The full design — decisions, evidence from a 9-system study, and the ranked
work plan — is in **[docs/dataforge-design-report.html](docs/dataforge-design-report.html)**
(single self-contained file; open it in a browser).

## Run it

From the repo root, with `<dataset>` one of `robocap`, `selfcap`, `wildcap`, `msd`, `lamaria`:

```bash
# 1. Catalog server, in a tmux session so it outlives your shell. Registrations
#    live in its memory only: if it restarts, repeat step 4 for each dataset.
tmux new -d -s dataforge-catalog 'pixi run -e dataforge --frozen dataforge-serve'   # rerun server on :51235

# 2. Fetch the raw corpus, or verify a local one (--root <dir> if it is not at the default)
pixi run -e dataforge --frozen dataforge-download <dataset>

# 3. One rrd per sequence per layer; existing rrds are skipped
#    --sequence belongs to the verb, so it goes BEFORE the dataset subcommand
pixi run -e dataforge --frozen dataforge-convert <dataset>               # --sequence <key> | --force

# 4. Register the rrds and the dataset's two blueprints
pixi run -e dataforge --frozen dataforge-register <dataset>              # --catalog-url <url>

# 5. Open one recording straight from disk
pixi run -e dataforge --frozen dataforge-view <dataset> --rr-config.headless   # --sequence <key>
```

Open the catalog with `rerun rerun+http://127.0.0.1:51235`, or a web viewer
with `?url=rerun+http://<host>:51235`. Reload the viewer after re-registering;
it caches catalog entries. The `register` and `view` tasks set
`RERUN_INSECURE_SKIP_HOST_CHECK=1`, as every catalog task in this workspace
does: the local catalog and the rrds it is handed are served over plain
`rerun+http` on the loopback or the tailnet, with no certificate to check.

`view` opens **every** layer of the chosen recording — `base/` and, when msd has
written one, `gt/`. They share a recording id, so the viewer merges them.

### Your own footage (wildcap)

wildcap takes bare mp4s — no calibration, no metadata, no sync. Because one
catalog dataset holds one camera layout (see *Blueprints*), each corpus gets a
`--corpus` name:

```
data/raw/wildcap-<corpus>/<capture>/exo/*.mp4      one file per static camera
data/raw/wildcap-<corpus>/<capture>/ego/*.mp4      optional: one head-mounted device's streams
```

```bash
pixi run -e dataforge --frozen dataforge-convert  wildcap --corpus <corpus> --root data/raw/wildcap-<corpus>
pixi run -e dataforge --frozen dataforge-register wildcap --corpus <corpus> --root data/raw/wildcap-<corpus>
```

That registers dataset `wildcap-<corpus>`. `.mov` is skipped with a warning
(remux: `ffmpeg -i in.mov -c copy out.mp4`).

### Monado SLAM Datasets (msd)

[`collabora/monado-slam-datasets`](https://huggingface.co/datasets/collabora/monado-slam-datasets)
(CC-BY 4.0) ships each VR-headset sequence as a zip of PNG frames plus csv
sensor logs — about 350 GB in total. dataforge never keeps that: `download`
fetches only the device calibration and prints the plan, and `convert` fetches
**one** sequence, streams its PNGs straight into the AV1 encoder, writes the
rrds, and deletes the archive again.

One sequence is **two** rrds under one recording id, so the catalog stacks them
as layers of one segment: `base/` holds the video, the IMU and the
magnetometer, and `gt/` holds the ground truth — the temporal `world_T_rig` on
the rig node at the archive's full ~1 kHz rate, the whole path as a static
`LineStrips3D` at `/world/runs/gt/trajectory`, a per-pose `Points3D` at
`/world/runs/gt/trail` that the default blueprint shows through a −10 s
cursor-relative window, and the root `ViewCoordinates`. Both layers come out of
one archive fetch, so `convert` skips a sequence only when both exist and
rebuilds both when either is missing.

One headset is one catalog dataset, because a catalog dataset holds one default
blueprint and the three headsets have different camera counts:

| `--device` | catalog dataset | cameras | magnetometer |
| --- | --- | --- | --- |
| `index` | `msd-index` | 2 × 960×960 @ ~54 fps, `kb4` fisheye | no |
| `g2` | `msd-g2` | 4 × 640×480 @ ~30 fps, `pinhole-radtan8` | yes |
| `odyssey` | `msd-odyssey` | 2 × 640×480, `pinhole-radtan8` | yes |

Every device keeps upstream's default calibration — `kb4` on the Index, the
`pinhole-radtan8` rational model on the G2 and Odyssey+ — so each camera node
states its `camera_model`, and a radtan8 camera also carries that model's
validity radius as `distortion_valid_radius`.

```bash
export DATAFORGE_OUTPUT_ROOT=/mnt/nas/datasets/msd-rrd     # rrds go to the NAS
pixi run -e dataforge --frozen dataforge-download msd --device index
pixi run -e dataforge --frozen dataforge-convert --sequence MIO09_short_1_updown msd --device index
pixi run -e dataforge --frozen dataforge-convert msd --device index   # every sequence, one at a time
pixi run -e dataforge --frozen dataforge-register msd --device index
```

`--root` is scratch, not storage: point it at local NVMe (it defaults to
`packages/dataforge/data/raw/msd`). `--raw-budget-gb` (default 50) caps what
that directory may hold — this sequence's archives **plus** everything already
in it, minus HuggingFace's own `.cache/` bookkeeping — so a batch run that fails
halfway cannot fill the disk. Two things ride on top of the cap and are not
counted: one camera's extracted PNGs (split archives only) and the temp mp4s,
both of which live in `work/` for the length of one sequence. A sequence whose
archives alone exceed the cap — the Index and G2 long sessions, at 66 and 55 GB
— warns and is converted anyway, but leftovers that breach the cap on their own
are still a hard error naming the files to delete. `--keep-raw` keeps the
archive and the encoded mp4s for debugging.

Three sequences ship as Info-ZIP multi-volume sets (`.z01 … .zip`) — the
`*_long_session` archives of all three headsets — which Python's `zipfile`
cannot read; those go through the `7zz` CLI (the conda-forge `7zip` package),
one camera directory extracted at a time.

`video_time` is the device clock minus `t0`, and `t0` is the earliest sample of
**any** stream including `gt` — the two layers must share the origin, and the gt
file is usually the earliest stream. Camera extrinsics come from the device's
basalt `calibration.json`, whose `T_imu_cam` is the camera pose in the IMU
frame; the rig frame *is* that frame, so the rig node states
`reference = "imu_00"`.

MSD documents no world axes: the Index's ground truth comes from SteamVR
Lighthouse and the G2's and Odyssey+'s from an undocumented MoCap rig. The up
axis is therefore **measured** — `measured_world_up` rotates the first two
seconds of accelerometer samples into the world with the ground truth's own
orientation and averages, since an accelerometer at rest reads +g pointing up —
and then fixed per device in `MSD_DEVICES`, so every rrd of a device carries the
same root `ViewCoordinates`. Every convert re-measures its own sequence, records
the result in the gt properties (`measured_up`, `measured_up_fraction`), and
warns when it disagrees with the declared axis instead of reorienting one rrd on
its own. All three headsets measure **+y** on their `*09_short_1_updown`
sequence — 0.96, 0.98 and 0.93 of |g| for the Index, G2 and Odyssey+ — so all
three state `RIGHT_HAND_Y_UP` (`RUB`) at the root. A `gt` row whose quaternion is
not unit-norm is a tracking dropout: its
rotation becomes identity so the chain keeps working for later frames, and the
count lands in the properties as `num_sanitized`.

The Follow view's eye is derived rather than hand-placed: `follow_frame` reads
the front camera pair out of the same `calibration.json` — their mean optical
axis is the headset's forward, and their stereo baseline crossed with it is the
wearer's up — and each device's answer is fixed in `MSD_DEVICES` as its `follow`
frame, so the chase camera sits behind and above every headset with a level
horizon. Up comes from the baseline and not from image-up because the G2 mounts
all four of its cameras rolled a quarter turn. Every convert re-derives the frame
and warns past 5°, as it does for the world up axis.
### LaMAria (lamaria)

[LaMAria](https://github.com/cvg/lamaria) (ETH CVG) is an Aria Gen1 egocentric
SLAM benchmark served from a plain Apache-indexed archive at
[cvg-data.inf.ethz.ch/lamaria](https://cvg-data.inf.ethz.ch/lamaria/): one VRS
per sequence (897 MB to 10 GB), a published body-frame calibration, and — for
the training split — dense pseudo ground truth plus surveyed control points.

dataforge never keeps the VRS. `download` reads the archive's index pages,
resolves each selected sequence's split and ground truth, writes
`<root>/manifest.json`, and fetches only the small files; `convert` then fetches
**one** VRS (resuming, and retrying a stalled transfer up to four times),
encodes its three camera streams to AV1, writes the rrd, and deletes the VRS and
the temp mp4s again. `--keep-raw` keeps them. The raw tree is the *official*
layout, so the upstream evaluation tools run on it unchanged:

```
<root>/<split>/<seq>/raw_data/<seq>.vrs
<root>/<split>/<seq>/aria_calibrations/<seq>.json
<root>/<split>/<seq>/ground_truth/pGT/<seq>.txt
<root>/<split>/<seq>/ground_truth/control_points/<seq>.json
```

```bash
export DATAFORGE_OUTPUT_ROOT=/mnt/nas/datasets/lamaria-rrd    # rrds go to the NAS
export DATAFORGE_FFMPEG=/home/pablo/.pixi/bin/ffmpeg          # the ffmpeg with av1_nvenc
pixi run -e dataforge --frozen dataforge-download lamaria                 # --sequences A B C
# --sequence belongs to the verb, so it goes before the dataset subcommand
pixi run -e dataforge --frozen dataforge-convert --sequence R_01_easy lamaria
pixi run -e dataforge --frozen dataforge-convert lamaria                  # every sequence, one at a time
pixi run -e dataforge --frozen dataforge-register lamaria                 # --catalog-url <url> goes first
```

Unset, `--sequences` means the five training sequences worth 18.2 GB of VRS
(`R_01_easy`, `R_04_medium`, `R_11_5cp`, `sequence_1_19`, `sequence_4_11`) at
`download` — both upstream collections, all three difficulty tiers, one
low-light capture, and every ground-truth shape — and every sequence the
manifest holds at `convert`, so a narrower `download --sequences` needs no flag
at all afterwards. A name the archive does not list is a hard error at
`download`; a sequence whose ground-truth files are not on disk is announced and
skipped at `convert`. `--root` is scratch, not storage: point it at local NVMe.

One sequence is one recording, `lamaria__<seq>`. The rig frame is **imu-right**,
which is what the published calibration uses as its body frame, so the rig node
states `reference = "imu_00"` and every logged `rig_T_sensor` is directly
comparable with that file's `T_b_s`. `cam_00` is camera-slam-left (640×480 gray,
20 fps), `cam_01` camera-slam-right, `cam_02` camera-rgb (1408×1408, 10 fps);
`imu_00` is imu-right (identity `rig_T_imu`, since it *is* the rig) and `imu_01`
imu-left, 129 mm away and rotated. Frames are logged in their **native**
(sideways) orientation, because that is what the calibration describes.
Everything comes out of the VRS device calibration via projectaria-tools: the
published JSON has no RGB camera and no imu-left, and is used to cross-check the
transform chain rather than to build it.

`video_time` is Aria's raw DEVICE clock in nanoseconds with **no shift**, so a
pseudo-GT row's own timestamp lands on its frame 1:1. The base layer therefore
logs no transform on the rig node and no root `ViewCoordinates`: the `gt` layer
establishes the world frame and owns both. The ground-truth files are downloaded
and kept for it.

#### The gt layer

One sequence is **two** rrds under one recording id, so the catalog stacks them
as layers of one segment: `base/` holds the video and the two IMUs, and `gt/`
holds everything the published ground truth establishes.

```
/                                      ViewCoordinates = RIGHT_HAND_Z_UP (static)
/world/rig_00                          Transform3D = world_T_rig, one row per pGT stamp
/world/runs/gt/trajectory              LineStrips3D, the whole path (static)
/world/runs/gt/trail                   Points3D, one per pose — the Follow view's −10 s window
/world/gt/control_points               Points3D, labelled, static (surveyed sequences only)
/world/rig_00/cam_MM/pinhole/cp_uv     Points2D at the detection stamps (SLAM pair only)
```

The pGT poses camera-slam-left, and the schema animates the rig, so every pose
is composed: `world_T_rig = world_T_cam0 @ inv(rig_T_cam0)`, with `rig_T_cam0`
straight out of the VRS device calibration (the published `cam0.T_b_s` agrees to
5e-16). The transform is logged child-to-parent, i.e. the stored value *is*
`world_T_rig`, so every camera frustum rides it.

Two world frames exist and both are Z-up: `R_01`…`R_10` are posed in MPS's own
gravity-aligned frame, and everything from `R_11` onwards — the control-point
sequences and the whole additional set — is surveyed in Switzerland's LV95/LN02
grid, translated by `CUSTOM_ORIGIN = (2683594.412, 1247727.747, 417.307)` exactly
as the official tooling does. The gt properties record which (`gt_world`). The up
axis is published rather than guessed, but `convert` still **measures** it on
every sequence — `measured_world_up` rotates the first two seconds of imu-right
samples into the world with the ground truth's own orientation and averages,
since an accelerometer at rest reads +g pointing up — and warns instead of
reorienting one rrd on its own. All five default sequences measure **+z**, at
0.92 to 1.01 of |g|.

A surveyed control point is drawn as a labelled sphere at its translated
position, with a radius that only grows past 0.1 m for a genuinely uncertain
point. A point the survey never levelled has no height: its `z` is the origin's
own, so it gets a distinct colour and an `OB1881 (no height)` label, and its
`NaN` height uncertainty never reaches Rerun. Every **levelled** point has to lie
within 50 m of the trajectory — its tag was photographed by these cameras — and
`convert` prints each point's closest approach and refuses the sequence
otherwise, because distance is what catches a wrong world frame or a missing
origin translation. On the three surveyed sequences every levelled point comes
within 0.7 m.

Both layers come out of one VRS fetch, so `convert` skips a sequence only when
both exist and rebuilds both when either is missing. A sequence the archive
publishes no ground truth for (the whole test split) writes no `gt` rrd at all
and is done once its base rrd exists. `register` then registers both layers
under one dataset:

```bash
# --catalog-url belongs to the verb, so it precedes the dataset subcommand
RERUN_INSECURE_SKIP_HOST_CHECK=1 DATAFORGE_OUTPUT_ROOT=/mnt/nas/datasets/lamaria-rrd \
  pixi run -e dataforge --frozen dataforge-register --catalog-url rerun+http://127.0.0.1:9988 lamaria
```

### Environment variables

| variable | default | purpose |
| --- | --- | --- |
| `DATAFORGE_OUTPUT_ROOT` | `packages/dataforge/data/dataforge/rrd` | where rrds and blueprints go; set it for convert **and** register |
| `DATAFORGE_RAW_ROOT` | `packages/dataforge/data/raw` | where raw corpora are fetched to |
| `DATAFORGE_FFMPEG` | the env's ffmpeg | an ffmpeg with hardware encoding, used both to re-encode B-frame sources (most phone HEVC) and to encode image sequences. Without `av1_nvenc` the encoder refuses to start rather than falling back to a software encode that looks like a hang. Check yours with `ffmpeg -hide_banner -encoders \| grep av1_nvenc` |

Paths in `--root`/`--sequence` and the defaults above are relative to
`packages/dataforge/` (the tasks run there).

## What exists

**Contracts** (`dataforge/`): `identity.py` (one `SequenceIdentity` derives the
sequence key, the Rerun recording id, and the rrd filename), `paths.py` (the
layer-major output tree: `base/` and its sibling `gt/`), `schema.py` (the
`exoego:v2` entity paths and the single `video_time` timeline, as code),
`writing.py` (atomic publication and the capture/convert recording properties),
`logging_toolkit.py` (the shared rig-node, video-stream, camera, IMU,
magnetometer and pose-track writers, plus the AV1 encoder), `blueprints.py` (the
single-rig viewer layout every dataset builds from), `archives.py` (reading
members out of a plain zip or an Info-ZIP volume set), `basalt.py` (basalt's
`calibration.json`: camera models, extrinsics, and the follow frame),
`transports.py` (`local_verify`, `hf_fetch`, and the resuming `http_fetch` plus
its Apache index parser), and `aria.py` (the Aria Gen1 VRS and LaMAria
ground-truth readers).

Two of those carry their weight for datasets that do not ship video.
`encode_frames_to_mp4` pipes PNG or raw frames straight into ffmpeg's stdin, so
an image-sequence dataset is encoded (AV1, NVENC, no B-frames) without an
intermediate frame tree on disk; `log_video_stream(..., times_ns=...)` then
stamps each sample with its real capture time, because the container's own PTS
is a nominal-rate fiction for such a file.

**Datasets** (`dataforge/datasets/`), one config + one dataset class each:

| dataset | one sequence | rrd contents |
| --- | --- | --- |
| `robocap` | one `(device, session)`; file-roll segments merge at convert | 6 fisheye video streams, the dev0 IMU, the cap mesh |
| `selfcap` | one cut episode | 4 phone exo rigs + the OAK ego rig + the Quest (9 cameras), the OAK IMU, the Quest head-pose track |
| `wildcap` | one capture directory | the videos only: no `Pinhole`, no `ViewCoordinates`, no transforms — calibration, sync and localization are later layers |
| `msd` | one Monado SLAM sequence, fetched on demand | **two** rrds: `base` with 2 or 4 grayscale video streams (AV1-encoded from the archive's PNGs), the IMU and on the G2/Odyssey+ the magnetometer; `gt` with the ~1 kHz `world_T_rig`, its path and trail, and the root `ViewCoordinates` |
| `lamaria` | one Aria Gen1 sequence, its VRS fetched on demand | **two** rrds: `base` with 3 AV1 video streams (2 gray SLAM + 1 RGB, native sideways orientation) and both raw IMUs (`imu_01` carrying its real `rig_T_imu`), on Aria's unshifted device clock; `gt` with the published `world_T_rig` (20 Hz on the controlled set, ~3 Hz on the surveyed ones), its path and trail, the surveyed control points and their 2D detections, and the root `ViewCoordinates` |

A config's `command` is its CLI subcommand; its `name` is the catalog dataset
and the prefix of every recording id. They are equal for robocap, selfcap and
lamaria; wildcap derives `wildcap-<corpus>` and msd derives `msd-<device>`.

**Verbs**, each a thin `tools/apps/*.py` shim over `dataforge/apis/`.
`convert` writes `<DATAFORGE_OUTPUT_ROOT>/base/<recording_id>.rrd` through a
temp file that atomically replaces the target, so "exists = done" and a re-run
never truncates a file the catalog server holds open. Derived layers (slam,
pose, labels) stack onto the same entities as sibling layers; msd and lamaria
already write one, their `gt/` rrd, in the same convert. `register` walks
every layer directory it knows — `base`, which is required, then `gt` — and
registers each under its own layer name, so a corpus with no ground-truth pass
registers exactly as before.

## Blueprints

Every dataset provides two (abstract on `DataforgeDataset`; missing one fails
at `setup()`):

- **default** — applied when a segment is opened from the catalog. The catalog
  holds exactly one per dataset, so one dataset = one camera layout.
- **segment table** — the preview card the table renders for every visible row
  at once, so it decodes exactly one video stream.

`register` adds them once and never replaces them (each registration is a new
entry in the viewer's blueprint list). To refresh: delete the dataset, then
re-register — files on disk are untouched.

```bash
pixi run -e dataforge --frozen python -c "from rerun.catalog import CatalogClient; CatalogClient('rerun+http://127.0.0.1:51235').get_dataset(name='<name>').delete()"
```

Conventions this package follows — beartype under `PIXI_DEV_MODE`, thin `tools/`
shims, jaxtyping annotations, `pixi run -e dataforge-dev {lint,typecheck,deadcode,tests}` —
are the monorepo ones in the root `AGENTS.md`. The logging schema is
`packages/simplecv/docs/exoego_schema.md`.
