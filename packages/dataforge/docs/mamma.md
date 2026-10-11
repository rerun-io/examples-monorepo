# MAMMA (MPI collections)

## Source

- MAMMA (MPI for Intelligent Systems, https://mamma.is.tue.mpg.de): markerless multi-person capture with 16- and
  32-camera IOI rigs and four iPhones. Access needs an MPI account. The six MPI collections (~221 GB, every video HEVC Rext
  4:4:4 CRF 16 at 30 fps) were downloaded 2026-07-27..29 to `/mnt/nas/datasets/mamma` and verified complete. MammaSyn, the
  synthetic WebDataset part, is a second PR.
- One command, one catalog dataset per family (`--family`): `dataforge-mamma-eval` (`mamma_eval_singles`, `_extra`, `_dance`:
  MoSh++ ground truth from Vicon markers), `dataforge-mamma-markerless-dance`, `dataforge-mamma-markerless-multiple-people` and
  `dataforge-mamma-markerless-iphones` (the MAMMA model's own predictions). Identity `mamma-<family>__<path>`: the sequence
  directory under its collection, after the eval set's short name for eval (`mamma-eval__singles__230929_WhiteRabbit_CatchBall_50048_1`,
  `mamma-markerless-iphones__indoors__crossing_arms`).
- Rig `NN` is the sequence's `NN`-th camera in name order (`IOI_01` is rig 1; the iPhones `A001`–`D001` are rigs 1–4), each one
  camera (`cam_00`).
- `property:episode:{collection,sequence,body_source,fit_method,num_people,subjects}`; `property:capture:{fps,clock_source,
  source_num_frames,source_resolution,video_source}`.
- Camera-source metadata: `source_width/height` are the release's native sizes (portrait cameras ship upright). Videos above
  3.2 MP are stored at half size and carry `stored_width/height`: eval Singles 4112×3008 → 2056×1504, the iPhones 3840×2160 →
  1920×1080. **The originals stay in the MAMMA release**; re-download them from MPI if a full-resolution frame is ever needed.
  Every video is re-encoded (`cq` 36, `gop` 60). IOI cameras also carry the lens as `vicon_radial_2` (below).
- Body models (`dataforge-download mamma` fetches them into `--model-root`): the release's reference is SMPL-X with the head bun
  removed ("locked head", 2023). Neutral fits use the locked-head neutral (`pablovela5620/mamma-streaming-data` at `0a961eed`,
  private, sha256 `43d8f3a1…`). The gendered fits (eval Singles) use SMPL-X v1.1 male and female from the public mirror
  `camenduru/SMPLer-X` at `9e5548b7`: they carry a subject template and zero betas, which replace the model's template and leave
  its shape basis unused, the only parts in which the two models differ (`test_a_template_fit_does_not_depend_on_the_model_variant`).
  v1.1 for the fits with betas would be off by up to 11 mm on Extra and 3.5 mm on the predictions (measured 2026-10-10).

## Get the raw data

```bash
pixi run -e dataforge dataforge-download mamma --family eval --root /mnt/nas/datasets/mamma      # models + verify in place
pixi run -e dataforge dataforge-download mamma --family markerless-iphones --root /mnt/nas/datasets/mamma \
    --sequences indoors/crossing_arms --fetch-into /local/scratch                                 # MPI fetch proof
pixi run -e dataforge dataforge-convert mamma --family eval --root /mnt/nas/datasets/mamma --sequences dance/250225_WestCoastSwing_Basic_Whip_03675_03676_1
```

- `download` never re-downloads the release. It fetches the three model files (pinned, sha256-verified) and verifies the
  family in place: every file the converter reads opens and is not empty or an MPI error page, every rig has a video per
  calibrated camera, every markerless sequence has the prediction files its name implies (a couple dance names both dancers'
  ids, a multiple-people capture its head count, the 13 two-person iPhone scenes are listed; a missing `params_00` would
  otherwise renumber everyone after it), and each collection has its manifest's sequence count (22, 12, 18, 123, 34, 42).
  Nothing is written.
- `--fetch-into` with `--sequences` downloads those sequences from the MPI server (`MAMMA_USERNAME`, `MAMMA_PASSWORD` in the
  environment; the account goes in a POST body) and compares every file with the release copy byte for byte. It refuses to run
  without `--sequences`. 2026-10-10: `indoors/crossing_arms` (387 MB, 72 s) and eval Dance `Basic_Whip` (1.5 GB, 370 s) came back
  byte-identical. The server sent Python clients HTML in July 2026, which is why `mamma-download-dataset` uses wget; it no
  longer does.
- The release is read in place and never modified (`/mnt/nas` is a protected input root); output goes to `DATAFORGE_OUTPUT_ROOT`.

## Raw inventory

| Shipped file or stream | Rate / clock | Layer | Entity or not ingested: reason |
| --- | --- | --- | --- |
| `videos_crf16/IOI_NN.mp4`, `videos/<iphone>.mp4` | 30 Hz, frame `i` | base | `/world/rig_NN/cam_00/pinhole/video`, AV1 4:2:0 (NVDEC decodes the 4:4:4 source, `scale_cuda` converts it) |
| `gt/IOI_NN.npz`, `meta/<cam>.npz`: `cam_int`, `cam_ext`, `cam_img_w/h`, `cam_portrait` | static | base | `Pinhole` (K scaled to the stored size) and `world_T_cam`; translations over 200 units would be read as millimetres (none are) |
| same, `vicon_radial_2` (IOI only) | static | base | `/world/rig_NN/cam_00` `vicon_radial_2`; used by projections |
| same, eval `is_body_in_img` (frames × people), Dance `visibility` (frames × people × 10,475, float64) | per frame | — | not ingested: derived from the GT meshes; the MPI raw stays |
| same, iPhone `img_abs_path`, `img_rel_path`, empty `bbox` | — | — | not ingested: the upstream JPEG names |
| `gt/global.npz` (eval): `pose_world`, `pose_trans_world`, `shape`, `gender`, `v_template`, `flat_hand_mean`, `subject_name`, `method_used` | 30 Hz | body_pose | `/world/gt/people/person_NN/body/smplx` (`pose`, `trans` per frame, float64 as shipped; `betas`, `v_template` static) |
| `meta/global.npz` (markerless): `fps`, `frames_len`, `seq_name`, `method_used` (iPhones: `frame_start/end`, `ioi_dir`, `c3d_path`) | — | base | the clock and the episode properties; upstream paths not ingested |
| `pred/params_XX.npz`: `poses`, `trans`, `betas` (16), `gender`, `flat_hand_mean`, `mocap_frame_rate` | 30 Hz | body_pose | `/world/gt/people/person_NN/body/smplx`, one person per file in name order; `mocap_frame_rate` must equal `fps` |
| `masks/IOI_NN_masks.tar` (eval): `mask_<frame>_<person>.png`, 0/255 at the native size | 30 Hz | masks | `/world/rig_NN/cam_00/pinhole/masks/person_NN`, a two-colour palette PNG at a quarter of the stored size |
| `markers/labels_m37.npy`, `vicon_m37.npy`, `baseline_m37.npy` (eval Extra) | 30 Hz | markers | `/world/gt/markers/vicon` and `/world/gt/markers/mosh_baseline`, 37 labelled points, world metres |
| `mamma_markerless_dance/bachata_improv_2p_32cam` | — | — | not ingested: a variant tree with only `videos_crf24`, `proxies_720p` and `meta` (no CRF 16 clips, no predictions) |

## Clocks

The release has no capture timestamps. `video_time` is `frame × 10⁹ / fps` ns from the sequence's own `global.npz` `fps` (30 in
every MPI collection; MammaSyn renders at 6), rounded to the nearest nanosecond, origin at frame 0; `frame_index` is the
frame. Every camera's video, every
label array and every mask tar holds exactly `frames_len` frames; base refuses a video with another count.

## Layers and entities

- **base**: the cameras and videos, capture and episode properties. The pinhole is K at the stored size; the Vicon lens is not
  part of a Rerun pinhole, so 3D content drawn in a camera's image (frustum images) is off by the lens' few pixels near the edges.
- **body_pose**, per person `NN` (subject order, which is not the sequence name's): `/world/gt/people/person_NN` with `subject`,
  `gender`, `source` (`mosh_gt` or `mamma_pred`, the same tag as `property:episode:body_source`), `fit_method`, `model`; the
  shipped parameters at `.../body/smplx`; `.../coco133_xyz` from the posed SMPL-X (144 joints with the face contour, all 133
  slots mapped by name; no shipped confidence, so 1.0).
- **body_mesh**: each person's posed SMPL-X mesh every third frame (10 Hz display layer).
- **projections**: each person's `coco133_xyz` through each camera's lens, in stored pixels, at
  `<pinhole>/coco133_uv_projected/person_NN`. The IOI lens is the release's `vicon_radial_2`: `[cx, cy, k1, k2, k3]` in native
  pixels, mapping a distorted pixel to its undistorted one, `u = c + d (1 + k1 r² + k2 r⁴ + k3 r⁶)` with `d` the distorted offset
  from `c`. Projection is the pinhole, then that map inverted by fixed-point iteration (non-converging points are NaN). Checked
  against the eval Dance masks: on the 12 most distorted unoccluded person views the GT silhouettes reach IoU 0.920 with the
  lens and 0.894 without (`test_the_vicon_lens_fits_the_eval_masks_better_than_a_pinhole`). The upstream `capture/undistort.py`
  applies the polynomial in the other direction. The iPhones ship no lens: pinhole. No measured 2D.
- **masks** (eval): one palette PNG per person, camera and frame: index 1 is the person (a colour per person), index 0 the
  transparent background, so the ids stay exact. Stored at a quarter of the video's width and height (Pablo's call, 2026-10-11:
  an overlay needs no more) and laid over it by a static `Transform3D` scale; area downscale with a majority threshold. The
  image is drawn at `opacity` 0.5, set explicitly (at the default 1.0 Rerun 0.38 can draw a translucent PNG's transparent pixels
  black), and each person has its own draw order. Mask `_NN+1` is `person_NN`: the release numbers its masks in subject order
  (checked by silhouette overlap on Dance). An eval sequence missing a mask tar fails its masks layer rather than skipping it.
- **markers** (eval Extra): the 37 Vicon markers MoSh++ held out, and the MoSh++ baseline's prediction of them. They sit at a
  median 15.1 mm from the GT mesh (`test_the_held_out_vicon_markers_sit_on_the_extra_mesh`).
- A layer a sequence has no source for (masks outside eval, markers outside Extra) is not written; register skips absent layers.
- Default blueprint: the scene (z up, orbital eye; projections and masks stay in the panes) above a strip of camera panes: all
  cameras of a rig of up to 16, eight evenly spaced ones of a 32-camera rig (every 4th). Each base rrd embeds its own sequence's
  layout; the catalog default is the family's (eval: panes for the first 16 cameras, which every eval rig has, with the scene's
  rules covering all 32 of Dance). On rigs of more than four cameras the
  videos are in the 3D view but hidden: toggle them in the blueprint panel. A hidden 3D video is not decoded (Rerun 0.38.1, Dance
  sample: viewer CPU 0.5 s over 30 time steps hidden, 128.5 s visible). Table card: the scene without video beside camera 1.

## Differences from simplecv

simplecv's `exoego/mamma.py` reads the same release for training views. dataforge differs where the release says otherwise:

- Every person gets COCO-133 keypoints (simplecv: person 0 only).
- Each person keeps its own gender, hand mean and template (simplecv uses person 0's settings for everyone).
- Beta-only GT (`mosh_betas`, eval Extra) has no `v_template`; simplecv indexes it unconditionally and fails.
- Projections apply the Vicon lens (simplecv: pinhole).

## Parity with simplecv

Not run: simplecv's adapter forwards SMPL-X through the same `smplx` package; parity is held against the release instead (the
template-variant, marker and mask tests above).

## Timing

Full run (2026-10-11, pablo-dl-server, RTX 5090, prod env, `d46e20e3`, the release read in place from the NAS, rrds written to
local disk and shipped afterwards): all 251 sequences, 0 failed, 108 min of capture in 154 min of conversion; 220 GB of release
files became 77 GB of rrds (2.9×). Each sequence's total is its in-process time in `timing/convert.jsonl`.

| Family | Sequences | Capture | Convert | × real time | Release read | rrds | Ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| eval | 52 | 23 min | 75 min | 3.21 | 79.6 GB | 9.2 GB | 8.7× |
| markerless-dance | 123 | 67 min | 64 min | 0.96 | 107.8 GB | 54.3 GB | 2.0× |
| markerless-multiple-people | 34 | 8 min | 12 min | 1.43 | 15.5 GB | 11.2 GB | 1.4× |
| markerless-iphones | 42 | 10 min | 3 min | 0.33 | 17.5 GB | 2.2 GB | 8.1× |

- Stages: the transcode (NVDEC 4:4:4 → NVENC AV1) dominates every markerless family (Dance 53.8 of 64 min); eval spends 53.0 of
  75 min on masks (the shipped PNG decode, 16 threads), 19.9 on the transcode. SMPL-X and projections take under 10% everywhere.
- Output by layer (GB): eval base 5.3, body_mesh 2.1, projections 1.1, masks 0.5, body_pose 0.2, markers < 0.1; Dance base 35.7,
  body_mesh 10.0, projections 8.0, body_pose 0.6; multiple people base 5.8, body_mesh 2.9, projections 2.3, body_pose 0.2;
  iPhones body_mesh 1.1, base 0.9, projections 0.1, body_pose 0.1. The meshes and projections (one 133-point row per person,
  camera and frame) are a third to a half of each family's output.

## Known gaps

- The 3D view draws each camera's video through its pinhole without the Vicon lens.
- Eval `is_body_in_img` and Dance per-vertex `visibility` are not ingested (both follow from the GT meshes and the calibration).
- simplecv parity was not run.
