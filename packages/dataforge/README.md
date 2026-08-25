# dataforge

One package for dataset work: **download → convert → register → view**.
Train and eval are the designed-for second half.

The full design — decisions, evidence from a 9-system study, and the ranked
work plan — is in **[docs/dataforge-design-report.html](docs/dataforge-design-report.html)**
(single self-contained file; open it in a browser).

## What exists

**Contracts** (`dataforge/`): `identity.py` (one `SequenceIdentity` derives the
sequence key, the Rerun recording id, and the rrd filename), `paths.py` (the
layer-major output tree), `schema.py` (the `exoego:v2` entity paths and the
single `video_time` timeline, as code), `writing.py` (atomic publication and the
capture/convert recording properties), `logging_toolkit.py` (the shared rig-node,
video-stream, and IMU writers), and `transports.py` (`local_verify`; the fetchers
land with the next dataset).

**Datasets** (`dataforge/datasets/`), one config + one dataset class each:

| dataset | raw corpus | one sequence | rrd contents |
| --- | --- | --- | --- |
| `robocap` | `/mnt/nas/datasets/robocap` | one `(device, session)`; file-roll segments merge at convert | 6 fisheye video streams, the dev0 IMU, the cap mesh |
| `selfcap` | `/mnt/nas/datasets/exoego-self-collected/main` | one cut episode | 4 iPhone exo rigs + the OAK ego rig + the Quest (9 cameras), the OAK IMU, the Quest head-pose track |

**Verbs**, each a thin `tools/apps/*.py` shim over `dataforge/apis/`:

```bash
pixi run -e dataforge --frozen dataforge-download selfcap        # fetch, or verify a local corpus
pixi run -e dataforge --frozen dataforge-convert robocap --sequence f408193e6447b3b0/s1
pixi run -e dataforge --frozen dataforge-register selfcap        # into a local `rerun server` catalog
pixi run -e dataforge --frozen dataforge-view robocap --rr-config.headless
```

`convert` writes **one base-layer rrd per sequence**, layer-major, at
`<DATAFORGE_OUTPUT_ROOT>/base/<recording_id>.rrd` — a temp file that atomically
replaces the target, so "exists = done" and a re-run never truncates a file a
catalog server holds open. Derived layers (slam, pose, labels) stack onto the
same entities as sibling layers; v1 emits `base` only. Each rrd embeds a
per-recording blueprint; `register` additionally sets the dataset-wide default
blueprint on the catalog dataset.

Conventions this package follows — beartype under `PIXI_DEV_MODE`, thin `tools/`
shims, jaxtyping annotations, `pixi run -e dataforge-dev {lint,typecheck,deadcode,tests}` —
are the monorepo ones in the root `AGENTS.md`. The logging schema is
`packages/simplecv/docs/exoego_schema.md`.
