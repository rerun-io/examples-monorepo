# Overnight SOMA batch runner

Current mesh output: [full-rate mesh and exact timestamps](full-rate-mesh.md).

Current body temporal defaults and validation: [body smoothing](body-smoothing.md).
The performance table below describes the original v1 run.

Run from the `selfcap-curate/soma-fit` worktree on the 5090 host. The prod
`selfcap-soma` environment is already installed. No Sapiens inference is run.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 pixi run -e selfcap-soma --frozen python \
  packages/selfcap-soma/tools/batch.py \
  --dataset selfcap-post-ep1-v2 \
  --segment selfcap__2025-12-03__d31bf1da__ep001 \
  --spark-dir /home/pablo/dataforge-dev/selfcap-post-ep1-v2/soma \
  --exo-source rtmw
```

First review this one example before launching the corpus. For multiple
segments, pass several values after `--segment`, or `--segments-file /path/ids.txt`
(one ID per line, optional blank lines and `#` comments). A process handles its
list sequentially. Split IDs across at most four processes; do not give the
same ID to simultaneous processes sharing an output root. Start with three if
other jobs need substantial CPU/RAM. No pushes or blueprint writes are made.

Required inputs: existing dataset/segment; confident RTMW 3D plus four exo raw
2D tracks; calibrated exo and OAK cameras; time-sync; Quest hand/head tracks;
raw 26-joint hand-orientation CSVs; SOMA assets. Missing inputs exit nonzero and
name the segment. Raw orientations are read directly because the catalog does
not carry them. `--raw-root` defaults to the original ingest tree
`/mnt/nas/datasets/exoego-self-collected/main/cut`; `--assets` defaults to
`/tmp/soma-fit/assets`. No source layer is rewritten.

Each segment gets a fresh shape stage with legs included, using well-observed
frames from the first 25 s. Identity, scale, symmetric limb ratios and one
headset-to-skull rotation are then frozen for the full-rate pose stage. Ears
supervise head position; no Quest head translation is used. Pose-stage knees,
ankles and feet have zero weight and frozen rotations. The rest-pose waist cut
and residual-scaled per-view evidence overlays are retained.

## Resuming

Default checkpoint root: `/tmp/soma-batch/<dataset>/<segment>/`. Keep it on a
persistent disk if `/tmp` is cleared overnight; override with `--output`.
`checkpoint.json` records completed stages and timings. Input/calibration/raw
orientation/settings changes invalidate the cache. Interrupted stages rerun;
completed shape/fit/export stages are reused. Completed layers are SHA-256
checked. A matching completed local rerun took 0.96 s with `--no-publish`.
With publishing enabled, resume re-ships and re-registers the same layer so it
also works after a remote file loss or catalog restart. `--force` refits.
A local advisory lock rejects duplicate simultaneous work on the same segment.

Only `soma` is registered, with REPLACE, at
`file://<spark-dir>/<segment>.rrd`. Upload uses a temporary file and atomic
rename. `--no-publish` keeps all artifacts local. Protected datasets
`selfcap-post-ep1` and `dataforge-selfcap` are rejected.

## Measured performance

Full 182 s d31bf1da segment, 4,367 frames, RTMW, warm assets:

| Stage | Seconds |
|---|---:|
| Catalog read and provenance | 0.98 |
| Shape | 11.42 |
| Pose and mesh evaluation | 10.41 |
| Fit RRD write | 3.13 |
| Mesh normals, overlays, layer write | 9.12 |
| Total before publish | 35.16 |
| Ship / register (same-size Sapiens layer test) | 0.57 / 0.02 |

Four independent full jobs completed concurrently with other GPU jobs active:
96.8–97.7 s/job, about 2.4x throughput versus serial. Each peaked at 427 MiB
PyTorch allocation, about 810 MiB driver allocation while fitting, and 5.4 GiB
host RSS. Whole-GPU usage at a sample during this test was 17.5 GiB of 32.6 GiB,
including unrelated jobs. Four workers fit on this host in that tested state;
it is not a guarantee if other jobs grow. Expect about 4.7 h for 690 segments at
that measured throughput, excluding outliers, transfers, retries and cold reads.
No automatic worker spawning is built into the CLI.

## RTMW versus Sapiens2 on the refined calibration

Each fit gets its own fresh shape and head attachment. Median / p90:

| Metric | Sapiens2 | RTMW |
|---|---:|---:|
| Body mm | 20.15 / 43.90 | 22.35 / 51.81 |
| Shoulders mm | 17.25 / 29.86 | 13.84 / 37.20 |
| Ears mm | 4.12 / 8.85 | 2.42 / 5.90 |
| Quest hands mm | 6.10 / 11.12 | 6.48 / 11.94 |
| OAK fingers px | 11.69 / 26.63 | 11.82 / 26.89 |
| Pelvis-height std mm | 6.63 | 10.69 |

Own-source residuals are not independent accuracy scores. Median exo pixels,
rig_00/01/02/03, evaluated against the same detector tracks:

- RTMW image tracks: Sapiens-fit 12.60/8.50/10.02/10.80;
  RTMW-fit 12.41/8.28/9.48/10.55.
- Sapiens2 image tracks: Sapiens-fit 8.51/6.87/7.67/7.53;
  RTMW-fit 9.28/6.97/8.13/7.98.

Current ear source disagreement is 24.2 mm left / 12.1 mm right in median 3D
distance, rather than the earlier ~6 cm estimate. Both fits calibrate their
own head rotation; do not reuse one source's attachment or body shape for the
other. RTMW's median reported head pitch is -54.7 deg versus -59.2 deg for
Sapiens2; both render a downward-looking head, and ears leave absolute pitch
weakly constrained. This is a remaining head-pose limitation, not a reason to
pin position to the headset origin.

## Fresh-shape correction now published in dev

`selfcap-post-dev/soma` contains the Sapiens2 fresh-shape result. Previous stale
shape → fresh shape: body 21.66→20.15 mm, shoulders 27.10→17.25 mm, ears
26.24→4.12 mm, Quest hands 6.09→6.10 mm, OAK fingers 11.56→11.69 px,
pelvis-height std 36.69→6.63 mm. Shape 7.87 s; pose/mesh 9.96 s. The shared
head-rotation attachment was refit too. Low ear residuals do not resolve the
remaining pitch ambiguity.

Artifacts: `/tmp/soma-fit/fresh-shape/`, `/tmp/soma-fit/rtmw-preview/`, and
`/tmp/soma-fit/batch-validation/selfcap-post-dev/<segment>/`. Front/side/back
waist and head renders were inspected. `report.json`, `metrics.json` and
`checkpoint.json` hold residuals, image checks and stage timings. The RTMW
validation was local only; dev remains on the Sapiens2 result.

Validation: eight targeted unit tests passed; Ruff and targeted Pyrefly passed.
The real RTMW pipeline, resume, missing-segment error, four concurrent pipelines,
and live shipping/REPLACE (using the identical Sapiens2 layer already published)
were exercised. Pixel evidence was inspected for the fresh Sapiens2 publish and
local RTMW preview at 39.862 s; both show upper-body overlays, with grey links
where that source's evidence does not pass the visibility rule. No full gate.

## Opt-in local cleanup

Add `--cleanup-after-publish` to release heavy SOMA files for a completed
segment. The flag is off by default. It deletes only `fit.rrd`,
`<segment>.rrd`, and `<segment>.pending.rrd` directly in that segment's output
directory. JSON reports, calibration, checkpoint, logs, screenshots and any
unrecognized files remain. It does not remove source/corpus files or recurse
into other output directories, and it refuses symlink artifacts.

Cleanup runs only after successful ship, REPLACE registration and readback:
the remote file SHA-256, registered soma URI, complete mesh timestamp array,
and first/last mesh vertex hashes must agree. Failure leaves the local files
intact and exits nonzero. The checkpoint records verification and cleanup
intent before unlinking, so an interrupted cleanup can finish safely on resume.

A cleaned published checkpoint resumes by checking the remote file and catalog
again; it does not require or regenerate deleted local RRDs. Missing/replaced
remote output fails explicitly; use `--force` to refit if recovery is needed.
`--no-publish` never cleans local artifacts. Changing the cleanup flag does not
invalidate an existing fit. Cleanup does not change the fitting algorithm.

Validation: temporary-file tests cover the allowlist, preserved reports/logs,
failed/unverified publication, read-only calls and symlinks. Live read-only
verification of d31 succeeded, then a synthetic cleaned checkpoint with no
local RRD resumed successfully in 4.94 s. No d31 evidence was deleted and no
corpus job was launched for these checks.
