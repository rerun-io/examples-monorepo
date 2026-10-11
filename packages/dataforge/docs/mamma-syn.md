# MammaSyn (MAMMA's synthetic training set)

## Source

- MammaSyn is MAMMA's synthetic training data (https://mamma.is.tue.mpg.de), the gated Hugging Face dataset
  `Intelligent-Systems/MammaSyn`: 6.63 TB of WebDataset shards, 8,610 files, downloaded 2026-07-27..28 to
  `/mnt/nas/datasets/mamma/MammaSyn` at commit `01ecdc88` and read in place. The MPI capture collections are in
  [mamma.md](mamma.md).
- Two families of the `mamma` command, one catalog dataset each:
  - `--family syn` (`dataforge-mamma-syn`): MAMMA's own renders, 14 dataset directories in three groups (Hands: Interhand;
    Interactions: Harmony4D, Inter-X, InteractionCouple, LatinDance10; Singles: BEDLAM bodies, MoYo), 2,144 sequence directories,
    each one scene seen by 2 to 8 fixed cameras of MAMMA's 32-camera IOI rig. Identity `mamma-syn__<dataset>__<sequence>`.
  - `--family syn-bedlam` (`dataforge-mamma-syn-bedlam`): `MammaSyn-Singles/BEDLAM_MASKS_WD`, the BEDLAM release repacked: 30
    batch directories of short single-camera clips (about 10,000). Identity `mamma-syn-bedlam__<batch>__<clip>`.
- A sample is one camera view of one frame, three shard members under one key: `<key>.png` (2056x1504 or 1504x2056 RGBA render;
  BEDLAM 1280x720), `<key>.mask.png` (8-bit instance mask: person index + 1, 0 = background) and `<key>.data.pyd` (a pickled dict
  of the people that view labels). The pickles refer to numpy arrays and scalars only; they are read with an unpickler that
  refuses anything else.
- A render key is `.../<sequence>/<camera>/<frame>`; within a camera the frames ascend, and a camera may straddle shards. A
  BEDLAM key is `.../png/<clip>/<clip>_<frame>`, the clips' samples unsorted in their batch's shards. Within a render shard each
  camera is one run, a sample's members in the order labels, render, mask.
- Reading over NFS: the NAS gives about 340 MB/s to 4 sequential streams (180 to one) and about 250 random reads per second, and a
  tar header walk is one random read per member (50 to 120 ms each under load). So while one render sequence converts, the next
  one's shards are copied to local disk in order, 4 spans of 256 MB at a time (`work/mamma-syn-stage/<pid>/<recording>`, about
  340 MB/s; each span must copy its full length); its header walk (0.1 s) and member reads come from that copy, which is removed
  once the cameras are read. Each run stages under its own process id, and a run removes the directories of runs that died. The page
  cache cannot stand in for the copy: on dl-server kswapd reclaimed a read-ahead within seconds while the encoders ran, with
  70 GB free. The index (each sample's member offsets) is cached as JSON under `--index-root` (local disk), keyed by each shard's
  size and mtime, which the copies keep. BEDLAM's discovery walks every batch's headers over NFS (4 batches at once) on its
  first run; while one clip converts, the next clip's members are read into memory in shard order, members less than 1 MB apart
  in one read (a clip lies in a few runs of its batch's shards).
- Body models: each person is posed with the neutral model whose forward gives back the shipped `joints3d` at its first labelled
  row. In a sample of one sequence per dataset directory, Interhand and MoYo were rendered with the locked-head neutral
  (`model` = `body_models/smplx_lockedhead/SMPLX_NEUTRAL.npz`); Harmony4D, Inter-X, InteractionCouple, LatinDance10 and the BEDLAM
  bodies (`b1_all`, `BEDLAM_MASKS_WD`) with SMPL-X v1.1 neutral (`model` = `SMPLX_NEUTRAL.npz` of the v1.1 mirror). The matching
  model reproduces the shipped joints, 3D vertices and 2D vertices exactly (under 0.001 mm, 0.05 px); the other one misses by
  0.6 mm or more. Flat hand mean, 16 betas (BEDLAM: 11; its gender labels are not the model).
  Every conversion checks every person at every frame a view labels them against the shipped joints (finite, within 0.1 mm;
  a failure names the person and frame) and records the largest gap as `property:episode:joints3d_max_error_mm`.

## Get the raw data

```bash
pixi run -e dataforge dataforge-download mamma --family syn --root /mnt/nas/datasets/mamma               # models + verify in place
pixi run -e dataforge dataforge-download mamma --family syn --root /mnt/nas/datasets/mamma \
    --sequences interhand_2-6_NC_200_00/be_NB-NZj2zRkeQ_seq_000002 --fetch-into /local/scratch                 # Hub fetch proof
pixi run -e dataforge dataforge-convert mamma --family syn-bedlam --root /mnt/nas/datasets/mamma \
    --sequences 20221024_10_100_batch01handhair_zoom_suburb_d/seq_000000
```

- `download` never downloads the release again. It fetches the SMPL-X models and lists the pinned Hub revision: every shard and
  `metadata.json` of the family must be on disk at its listed size (2026-10-11: renders 6,945 files, BEDLAM 1,633, all present at
  their size). With `--sequences --fetch-into` it fetches the shards holding those sequences (a BEDLAM clip lives in one or two
  shards of its batch) and compares them byte for byte; `interhand.../be_NB-NZj2zRkeQ_seq_000002` (2 shards, 1.95 GB) and the
  shard of BEDLAM clip `20221024_10_100_batch01handhair_zoom_suburb_d/seq_000000` came back identical.
- Every sequence's samples must number what its `metadata.json` declares, or it fails; a BEDLAM batch that does not fails the
  family's discovery, naming the batch. One render directory,
  `interactions_couple_1_C_200_00_contact/be_TyJqNcw9QoeT_seq_000161`, has shards and no `metadata.json` (on the Hub too): it is
  skipped and reported.

## Raw inventory

| Shipped member or field | Rate / clock | Layer | Entity or not ingested: reason |
| --- | --- | --- | --- |
| `<key>.png` | per sample | base | `/world/rig_NN/cam_00/pinhole/video`, AV1 4:2:0 from the RGBA render, alpha dropped (99.8% of pixels are opaque) |
| `cam_int`, `cam_ext` (every person of every view) | static (renders), per frame (BEDLAM) | base | `Pinhole` and the rig pose (renders: fixed per camera, checked); BEDLAM: K per frame (zoom clips), the camera at the origin |
| `pose_world`, `shape`, `trans_world`, `gender` | per frame | body_pose | `/world/gt/people/person_NN/body/smplx` (`pose`, `trans` per frame, NaN where absent; `betas` static); BEDLAM adds `cam_T_body` |
| `joints3d` | per frame | — | checked against the forward (`joints3d_max_error_mm`), not stored: the forward reproduces it |
| `vertices3d`, `vertices2d`, `joints2d`, `camera_normals` | per frame | — | not ingested: the forward of the stored parameters and calibration (golden test) |
| `<key>.mask.png` | per sample | masks | `/world/rig_NN/cam_00/pinhole/masks/person_NN`, a palette PNG per person at a quarter size; everyone a mask shows |
| `vertex_visibility`, `person_visitiblity_rate`, `floor_contact_mask` | per view, per frame | — | not ingested: training aids derived from the meshes and cameras |
| `center`, `scale`, `imgname` | per view | — | not ingested: the upstream crop box and file name |
| `motion_info`, `sub`, `race` (BEDLAM: AMASS motion, texture, skin) | per person | — | not ingested; still in the release |
| `metadata.json` `num_samples` | per directory | — | checked against the shards' samples (`property:episode:shard_samples`) |

## Clocks

The shards carry no timestamps; the member names carry the source frame numbers. MAMMA renders MammaSyn at 6 fps (MAMMA paper,
`docs/DATASETS.md` upstream): `video_time` is frame / 6 s. BEDLAM numbers its frames at its 30 fps render rate and ships every
5th: `video_time` is frame / 30 s, samples 1/6 s apart (`property:capture:fps` is the stream's 6; `clock_source` names the 30 fps
numbering). `frame_index` is the source frame number. A frame the shards do not hold
is a gap on the clock, not a renumbering (the sampled Hands sequence lacks frame 2 in every camera); a camera may lack a frame
another camera has.

## Layers and entities

- **base**: the cameras and videos, capture and episode properties (`group`, `collection` = the dataset directory, `sequence`,
  `subjects`, `shard_samples`, `joints3d_max_error_mm`; BEDLAM: `batch`, `genders`). The renders' world is y down, as their
  labels are; a BEDLAM clip's world is its camera frame (x right, y down, z ahead), the only frame its bodies share: each BEDLAM
  body comes in a world of its own, and on orbit clips those worlds do not agree.
- **body_pose**, per person `NN` = the release's person index (its mask value - 1): the shipped parameters, the `source` tag
  `synthetic_gt`, the `model` file, and COCO-133 keypoints from the posed model; NaN rows where no view labels the person (a view
  leaves out people it does not see).
- **body_mesh**: each person's posed SMPL-X mesh at every frame (6 Hz), an empty mesh where the person is absent. BEDLAM bodies are
  posed in their own world and moved by their per-frame `cam_ext`.
- **projections**: COCO-133 through each camera's pinhole (no lens), per frame K on BEDLAM.
- **masks**: per camera, one palette PNG per person and frame at a quarter of the video's size, as the eval masks (mamma.md), for
  everyone the labels name and anyone else a mask shows; an empty one where the person is not in that view.
- Default blueprint: the scene above a pane per camera (renders: all of their 2 to 8; BEDLAM: one); on more than four cameras
  the videos are hidden in 3D. Table card: the scene beside camera 1.

## Timing

TODO

## Known gaps

- The per-view training aids (vertex visibility, floor contact, crop boxes) and BEDLAM's motion and texture names are not ingested.
- A rendered person that no view labels at a frame has no body there, even when a mask shows them.
