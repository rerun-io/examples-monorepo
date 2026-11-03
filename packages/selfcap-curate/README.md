# SelfCap alignment and calibration

Read SelfCap episodes from a catalog, align their clocks, calibrate the cameras,
and write keypoint layers. `selfcap-run` drives the full production path.

## Run it yourself

The raw captures are private Hugging Face datasets. You need read access and `hf auth login`.
Run from the repository root on a machine with an NVIDIA GPU.

**1. Start a catalog** that can read your run directory. A local one is enough:

```bash
pixi run -e selfcap-curate --frozen rerun server --host 127.0.0.1 -p 51300 &
CATALOG=rerun+http://127.0.0.1:51300
```

A catalog on another machine also works (for example `rerun+http://dgx-spark:51235`) if it reads
the run directory at the same path, such as the NAS SSD share `/mnt/nas/ssd-8tb`.

**2a. Smoke run: 3 segments (about 5 minutes).**

```bash
RAW=/mnt/nas/ssd-8tb/selfcap/raw/easy26-smoke
pixi exec --spec huggingface_hub hf download pablovela5620/selfcap-easy26-raw --repo-type dataset \
  --revision 12de5d4fa29aa443298e8f6ac560ad5a72c9815a --local-dir $RAW/selfcap \
  --include 'cut/2025-11-25/426c5f35-*/**' --include 'cut/2025-12-23/4444205e-*/**' --include 'cut/2025-12-23/d3dc7bbf-*/**'
pixi run -e selfcap-curate --frozen selfcap-run --raw-root $RAW \
  --run-root /mnt/nas/ssd-8tb/selfcap/runs/easy26-smoke-$(date +%Y%m%d-%H%M) \
  --catalog-url $CATALOG --dataset selfcap-easy26-smoke-$(date +%Y%m%d-%H%M)
```

**2b. Full easy set: 26 segments (5.5 GB download, about 30 minutes).**

```bash
RAW=/mnt/nas/ssd-8tb/selfcap/raw/easy26
pixi exec --spec huggingface_hub hf download pablovela5620/selfcap-easy26-raw --repo-type dataset \
  --revision 12de5d4fa29aa443298e8f6ac560ad5a72c9815a --local-dir $RAW/selfcap
pixi run -e selfcap-curate --frozen selfcap-run --raw-root $RAW \
  --run-root /mnt/nas/ssd-8tb/selfcap/runs/easy26-$(date +%Y%m%d-%H%M) \
  --catalog-url $CATALOG --dataset selfcap-easy26-$(date +%Y%m%d-%H%M)
```

The run directory and the dataset name must be new. `<run-root>/progress.log` shows each step;
the last lines give the summary and `PIPELINE_DONE`. Open the result with
`pixi run -e selfcap-curate --frozen rerun "$CATALOG"`. The model files (EgoHand, the hand
detector, SOMA-X) download on first use at pinned revisions. The hard set is
`pablovela5620/selfcap-hard26-raw` and runs the same way.

## Pipeline

| Step | Task (environment) | At a time | Layers swapped in after it |
|---|---|---|---|
| ingest (raw only) | `dataforge-convert` (dataforge) | one | |
| register | `dataforge-register` (dataforge) | one | base, gt: `rrd/` |
| align | `selfcap-align` (selfcap-curate) | `--jobs` | base, gt, sapiens2 |
| calibrate | `selfcap-calibrate` (selfcap-curate) | `--jobs` | egohand |
| keypoints | `selfcap-keypoints` (selfcap-curate) | one | calib |
| publish | `dataforge-publish-blueprints` (dataforge) | one | device assets and blueprints, for viewing while SOMA runs |
| SOMA | `soma-batch` (selfcap-soma) | `--soma-jobs` | soma |
| soma_xlo | `soma-xlo` (selfcap-soma) | one process, every segment | soma_xlo |
| preview | `selfcap-preview` (selfcap-curate) | one process, `--preview-workers` segments | preview |
| blueprints | `dataforge-publish-blueprints` (dataforge) | one | |

A step receives only the segments that passed the step before it. A segment passes when its output file exists; in a one-process step a segment with no file fails even if the others passed. A failed segment is logged with the last meaningful line of its log, and skipped. These stop the run with a nonzero exit: a step that leaves no segment, and a failed register, publish, swap or blueprints step, because the later steps read what these write. The run also exits nonzero at the end if a segment failed. A segment that fails stays in the dataset with the layers it had, and the default blueprints always include the SOMA mesh, and show the preview card table when any segment has a preview layer.

To save disk, the run deletes two kinds of its own files once the catalog shows the new file as the layer, and logs each one in `progress.log`: the as-cut base under `rrd/base` after the base swap (in a `--base-dir` run only the link, never the as-cut file), and `calib/<segment>/<segment>.rrd` after the calib swap. Nothing outside the run root is deleted.

`<run-root>/progress.log` has one `[HH:MM:SS] <dataset>: ...` line per segment and step, `STEP<number>_<STEP>_WALL` lines, and at the end one `SUMMARY <step>: ok N, failed N, wall N s` line per step and a `SOMA_QUALITY` line with pass and flag counts and medians from the `property:soma_quality:*` columns of the catalog segment table. `<run-root>/logs` holds each process's log.

## Clocks

`capture_time` preserves presented video PTS, including the container start,
and raw CSV timestamps for IMU, Quest pose and hands. `video_time` aligns them
to OAK left. Offsets are milliseconds to **add** to capture time. RGB uses
`a_ms + b_ms_per_s * (t_start_session_s + capture_seconds)`; the session origin
is the ingested OAK left frame counter divided by 24. Phone alignment uses
`tau - 48 ms + quest_pose`. Missing measurements remain null with flags.

## Optical refinement

`selfcap-exo-label` records optical landmarks. `selfcap-exo-refine` evaluates
candidate camera poses with held-out gates. The `exo_refine*`, `exo_landmarks`,
`calibration_objective`, and `device_vp` modules hold this optional stack.
`selfcap-calibrate` exposes `--exo-refine`, `--exo-focal-source`,
`--exo-focal-file`, and `--exo-reference-dataset` for it.
