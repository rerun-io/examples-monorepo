# dataforge

One package for dataset work: **download → convert → register → view**.
Train and eval are the designed-for second half.

The full design — decisions, evidence from a 9-system study, and the ranked
work plan — is in **[docs/dataforge-design-report.html](docs/dataforge-design-report.html)**
(single self-contained file; open it in a browser).

## Run it

From the repo root, with `<dataset>` one of `robocap`, `selfcap`, `wildcap`:

```bash
# 1. Catalog server, in a tmux session so it outlives your shell. Registrations
#    live in its memory only: if it restarts, repeat step 4 for each dataset.
tmux new -d -s dataforge-catalog 'pixi run -e dataforge --frozen dataforge-serve'   # rerun server on :51235

# 2. Fetch the raw corpus, or verify a local one (--root <dir> if it is not at the default)
pixi run -e dataforge --frozen dataforge-download <dataset>

# 3. One base-layer rrd per sequence; existing rrds are skipped
pixi run -e dataforge --frozen dataforge-convert <dataset>               # --sequence <key> | --force

# 4. Register the rrds and the dataset's two blueprints
pixi run -e dataforge --frozen dataforge-register <dataset>              # --catalog-url <url>

# 5. Open one recording straight from disk
pixi run -e dataforge --frozen dataforge-view <dataset> --rr-config.headless   # --sequence <key>
```

Open the catalog with `rerun rerun+http://127.0.0.1:51235`, or a web viewer
with `?url=rerun+http://<host>:51235`. Reload the viewer after re-registering;
it caches catalog entries.

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
`logging_toolkit.py` (the shared rig-node, video-stream, IMU, and magnetometer
writers, plus the AV1 encoder), and `transports.py` (`local_verify` and
`hf_fetch`).

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

A config's `command` is its CLI subcommand; its `name` is the catalog dataset
and the prefix of every recording id. They are equal for robocap and selfcap;
wildcap derives `wildcap-<corpus>`.

**Verbs**, each a thin `tools/apps/*.py` shim over `dataforge/apis/`.
`convert` writes `<DATAFORGE_OUTPUT_ROOT>/base/<recording_id>.rrd` through a
temp file that atomically replaces the target, so "exists = done" and a re-run
never truncates a file the catalog server holds open. Derived layers (slam,
pose, labels) stack onto the same entities as sibling layers. `register` walks
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
