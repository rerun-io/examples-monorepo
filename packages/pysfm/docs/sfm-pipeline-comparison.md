# SfM Pipeline Comparison: pysfm vs hloc vs COLMAP

## Context

These three systems operate at **different abstraction layers** on top of the same core engine (COLMAP/pycolmap). They are NOT doing the same thing — they solve different problems but share the same SfM primitives.

---

## TL;DR

| System | Goal | Owns pose estimation? | Uses triangulation standalone? |
|---|---|---|---|
| **COLMAP `incremental_mapping`** | The engine. Estimates poses + 3D points from scratch | Yes (it IS the pose estimator) | Internally, per-image after registration |
| **hloc** | Reconstruction with swappable features/matchers | Delegates to `pycolmap.incremental_mapping` for full SfM; uses `pycolmap.triangulate_points` separately for the "known poses" case | Yes — `triangulation.py` is for adding 3D points to an existing model |
| **pysfm (`pycolmap_recon.py`)** | Multi-camera rig reconstruction from video | Delegates to `pycolmap.incremental_mapping` (bootstrap) + `pycolmap.global_mapping` (rig-aware) | No — relies entirely on COLMAP's built-in pipelines |

---

## State Machine 1: COLMAP `incremental_mapping` (the engine)

This is what happens **inside** `pycolmap.incremental_mapping()`. Both hloc and pysfm call this as a black box.

```
┌─────────────────────────────────────────────────────────────────┐
│                   COLMAP INCREMENTAL MAPPING                     │
│                                                                  │
│  ┌──────────────┐                                                │
│  │  FIND INITIAL │──failed──► relax constraints, retry           │
│  │  IMAGE PAIR   │                                               │
│  └──────┬───────┘                                                │
│         │ found pair                                             │
│  ┌──────▼───────────────┐                                        │
│  │ REGISTER INITIAL PAIR │  (estimate relative pose,             │
│  │ + TRIANGULATE         │   triangulate initial points)         │
│  └──────┬───────────────┘                                        │
│         │                                                        │
│  ┌──────▼───────────┐                                            │
│  │ GLOBAL BUNDLE ADJ │  (optimize all poses + points)            │
│  │ + FILTER          │  (remove high-error points/frames)        │
│  └──────┬───────────┘                                            │
│         │                                                        │
│  ┌──────▼──────────────────────────────────────────────────┐     │
│  │                    MAIN LOOP                             │     │
│  │                                                          │     │
│  │  ┌─────────────────┐                                     │     │
│  │  │ FIND NEXT IMAGE  │──no more──► FINAL GLOBAL BA ──► DONE│   │
│  │  │ (most visible    │                                     │    │
│  │  │  3D points)      │                                     │    │
│  │  └──────┬──────────┘                                     │    │
│  │         │                                                │     │
│  │  ┌──────▼──────────┐                                     │     │
│  │  │ REGISTER IMAGE   │  (PnP: estimate pose from          │     │
│  │  │                  │   2D-3D correspondences)            │     │
│  │  └──────┬──────────┘                                     │     │
│  │         │                                                │     │
│  │  ┌──────▼──────────┐                                     │     │
│  │  │ TRIANGULATE      │  (add NEW 3D points visible        │     │
│  │  │ IMAGE            │   from this image + existing ones)  │    │
│  │  └──────┬──────────┘                                     │     │
│  │         │                                                │     │
│  │  ┌──────▼──────────┐                                     │     │
│  │  │ LOCAL BUNDLE ADJ │  (optimize nearby poses + points)   │    │
│  │  └──────┬──────────┘                                     │     │
│  │         │                                                │     │
│  │  ┌──────▼──────────┐                                     │     │
│  │  │ GLOBAL BA?       │──no──► back to FIND NEXT IMAGE     │     │
│  │  │ (if model grew   │                                     │    │
│  │  │  enough)         │                                     │    │
│  │  └──────┬──────────┘                                     │     │
│  │         │ yes                                            │     │
│  │  ┌──────▼──────────┐                                     │     │
│  │  │ GLOBAL BA        │                                     │    │
│  │  │ + FILTER         │                                     │    │
│  │  │ + MERGE TRACKS   │                                     │    │
│  │  └──────┬──────────┘                                     │     │
│  │         │                                                │     │
│  │         └──── back to FIND NEXT IMAGE                    │     │
│  └──────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

**Key insight**: Triangulation is NOT a separate step here — it's woven into every iteration. Each time an image is registered, its new 2D-3D correspondences produce new 3D points via triangulation. Bundle adjustment then refines everything.

---

## State Machine 2: hloc — TWO distinct flows

hloc has **two separate pipelines** and this is the source of the confusion:

### Flow A: Full Reconstruction (`hloc/reconstruction.py`)
When you have NO prior poses — just images, features, matches.

```
┌──────────────────────────────────────────────────────┐
│               hloc FULL RECONSTRUCTION                │
│                                                       │
│  ┌───────────────────┐                                │
│  │ CREATE EMPTY DB    │  pycolmap.Database.open()      │
│  └──────┬────────────┘                                │
│         │                                             │
│  ┌──────▼────────────┐                                │
│  │ IMPORT IMAGES      │  pycolmap.import_images()      │
│  └──────┬────────────┘                                │
│         │                                             │
│  ┌──────▼────────────┐                                │
│  │ IMPORT FEATURES    │  db.write_keypoints()          │
│  │ (from HDF5)        │  (hloc's own feature files)    │
│  └──────┬────────────┘                                │
│         │                                             │
│  ┌──────▼────────────┐                                │
│  │ IMPORT MATCHES     │  db.write_matches()            │
│  │ (from HDF5)        │  (hloc's own match files)      │
│  └──────┬────────────┘                                │
│         │                                             │
│  ┌──────▼────────────┐                                │
│  │ GEOMETRIC VERIFY   │  pycolmap.verify_matches()     │
│  └──────┬────────────┘                                │
│         │                                             │
│  ┌──────▼────────────┐                                │
│  │ INCREMENTAL        │  pycolmap.incremental_mapping()│
│  │ MAPPING            │  ◄── same black box as above   │
│  └──────┬────────────┘                                │
│         │                                             │
│  ┌──────▼────────────┐                                │
│  │ SELECT LARGEST     │                                │
│  │ MODEL              │                                │
│  └──────────────────┘                                │
└──────────────────────────────────────────────────────┘
```

### Flow B: Triangulation with Known Poses (`hloc/triangulation.py`)
When you ALREADY HAVE camera poses (e.g., from a prior SfM run, GPS, or SLAM).

```
┌──────────────────────────────────────────────────────┐
│           hloc TRIANGULATION (known poses)             │
│                                                       │
│  ┌───────────────────┐                                │
│  │ LOAD REFERENCE     │  pycolmap.Reconstruction()     │
│  │ MODEL (has poses)  │  (existing model with poses)   │
│  └──────┬────────────┘                                │
│         │                                             │
│  ┌──────▼────────────┐                                │
│  │ CREATE DB +        │  same import steps as above    │
│  │ IMPORT FEATURES    │  but with NEW features/matches │
│  │ + MATCHES          │                                │
│  └──────┬────────────┘                                │
│         │                                             │
│  ┌──────▼────────────┐                                │
│  │ GEOMETRIC VERIFY   │  pycolmap.verify_matches()     │
│  └──────┬────────────┘                                │
│         │                                             │
│  ┌──────▼────────────┐                                │
│  │ TRIANGULATE POINTS │  pycolmap.triangulate_points() │
│  │                    │  Poses are FIXED, only adds    │
│  │                    │  new 3D points                 │
│  └──────────────────┘                                │
└──────────────────────────────────────────────────────┘
```

**Why does Flow B exist?** The Aachen Day-Night benchmark is the canonical example: you have a SfM model built from SIFT features, but you want to re-triangulate with SuperPoint features (better for localization). The poses are already known — you just need better 3D points for matching queries against.

---

## State Machine 3: pysfm (`pycolmap_recon.py`) — Rig-aware two-pass

This pipeline does something neither hloc nor vanilla COLMAP does: **multi-camera rig reconstruction**.

```
┌────────────────────────────────────────────────────────────────┐
│               pysfm RIG RECONSTRUCTION                          │
│                                                                 │
│  ┌─────────────────┐                                            │
│  │ DISCOVER VIDEOS  │  find cam1.mp4, cam2.mp4, ...             │
│  └──────┬──────────┘                                            │
│         │                                                       │
│  ┌──────▼──────────┐                                            │
│  │ EXTRACT SYNCED   │  evenly-spaced frames from each video     │
│  │ FRAMES           │  same indices → synchronized timestamps   │
│  └──────┬──────────┘                                            │
│         │                                                       │
│  ┌──────▼──────────┐                                            │
│  │ GENERATE RIG     │  rig_config.json (which cams, ref sensor) │
│  │ CONFIG           │  NO poses yet — just grouping info        │
│  └──────┬──────────┘                                            │
│         │                                                       │
│  ┌──────▼──────────┐                                            │
│  │ FEATURE          │  pycolmap.extract_features()               │
│  │ EXTRACTION       │  ALIKED_N16ROT, GPU                       │
│  └──────┬──────────┘                                            │
│         │                                                       │
│  ┌──────▼──────────────────┐                                    │
│  │ SEQUENTIAL MATCHING     │  pycolmap.match_sequential()        │
│  │ (no rig awareness)      │  only temporal neighbors            │
│  └──────┬──────────────────┘                                    │
│         │                                                       │
│  ┌──────▼──────────────────┐                                    │
│  │ INCREMENTAL MAPPING     │  pycolmap.incremental_mapping()     │
│  │ (NO-RIG BOOTSTRAP)      │  ◄── same black box as COLMAP      │
│  │                         │  treats each cam independently      │
│  └──────┬──────────────────┘                                    │
│         │ produces rough poses for all cameras                  │
│         │                                                       │
│  ┌──────▼──────────────────┐                                    │
│  │ APPLY RIG CONFIG        │  pycolmap.apply_rig_config()        │
│  │                         │  derives cam_from_rig transforms    │
│  │                         │  from the bootstrap poses           │
│  └──────┬──────────────────┘                                    │
│         │                                                       │
│  ┌──────▼──────────────────┐                                    │
│  │ RIG-AWARE MATCHING      │  pycolmap.match_sequential()        │
│  │ (expand_rig_images)     │  cross-camera pairs now included    │
│  │                         │  cam1/img5 ↔ cam2/img4, etc.       │
│  └──────┬──────────────────┘                                    │
│         │                                                       │
│  ┌──────▼──────────────────┐                                    │
│  │ GLOBAL MAPPING          │  pycolmap.global_mapping()          │
│  │ (refine_sensor_from_rig)│  rig constraints enforced           │
│  │                         │  jointly optimizes all cameras      │
│  └──────────────────────────┘                                   │
└────────────────────────────────────────────────────────────────┘
```

---

## How They Relate — The Layer Cake

```
┌─────────────────────────────────────────────────┐
│  YOUR CODE (pysfm)                               │
│  Problem: multi-camera rig from video            │
│  Unique: rig config, 2-pass (bootstrap → rig),   │
│          global_mapping with rig constraints      │
├─────────────────────────────────────────────────┤
│  hloc                                            │
│  Problem: SfM with swappable features/matchers   │
│  Unique: HDF5 feature/match storage, triangulate │
│          with known poses for localization        │
├─────────────────────────────────────────────────┤
│  pycolmap (Python bindings)                      │
│  incremental_mapping(), triangulate_points(),    │
│  global_mapping(), extract_features(), etc.      │
├─────────────────────────────────────────────────┤
│  COLMAP C++ (the engine)                         │
│  IncrementalMapper, BundleAdjustment,            │
│  Triangulator, PoseEstimation                    │
└─────────────────────────────────────────────────┘
```

---

## The Triangulation Confusion — Resolved

There are **three different uses of the word "triangulation"**:

1. **Inside `incremental_mapping`**: After each image is registered (pose estimated via PnP), COLMAP triangulates new 3D points from 2D matches. This is automatic and internal — you never call it.

2. **`pycolmap.triangulate_points()` (standalone)**: Takes an existing model WITH known poses, and creates new 3D points from feature matches. hloc uses this in `triangulation.py` for the "re-triangulate with better features" use case. pysfm does NOT use this.

3. **The geometric concept**: Given 2+ rays from known camera positions, find their 3D intersection. This is what both #1 and #2 do internally, just triggered differently.

---

## What About Bundle Adjustment?

- **Inside `incremental_mapping`**: BA runs automatically — local BA after every image, global BA periodically. You don't call it.
- **Inside `global_mapping`** (pysfm step 9): Also runs BA internally, but with rig constraints (`refine_sensor_from_rig`).
- **Standalone `pycolmap.bundle_adjustment()`**: Exists but hloc and pysfm don't call it directly — they rely on the mapping pipelines to handle it.
- **The COLMAP Python example** (custom pipeline on DeepWiki): Shows calling BA explicitly as a teaching example of "build your own pipeline from primitives" — but in practice, `incremental_mapping()` does all of this for you.

---

## Are They All Doing the Same Thing?

**No.** They solve different problems using the same engine:

| | COLMAP engine | hloc | pysfm |
|---|---|---|---|
| **Input** | DB with features + matches | Images + any extractor/matcher | Videos from multi-cam rig |
| **Feature extraction** | Built-in (SIFT, ALIKED, etc.) | External (SuperPoint, DISK, etc.) stored in HDF5 | Built-in via pycolmap (ALIKED) |
| **Matching** | Built-in | External (SuperGlue, LightGlue, etc.) stored in HDF5 | Built-in via pycolmap (LightGlue) |
| **Pose estimation** | `incremental_mapping` | Delegates to `incremental_mapping` OR skips it (known poses) | `incremental_mapping` (bootstrap) + `global_mapping` (rig) |
| **Rig support** | Yes (apply_rig_config + global_mapping) | No | Yes — this is the whole point |
| **Triangulation standalone** | Available as primitive | Used for known-pose re-triangulation | Not used |
| **Primary use case** | General SfM | Visual localization (build map → localize queries) | Calibrate unknown camera rigs |
