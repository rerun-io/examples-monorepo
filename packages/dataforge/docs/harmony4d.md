# Harmony4D

## Source

- Harmony4D (Khirodkar, Song, Cao, Luo, Kitani; NeurIPS 2024 Datasets and Benchmarks): close two-person interactions
  (hugging, grappling, sword, ballroom, karate, MMA) seen by ~20 static GoPros and, in most takes, the two people's Aria
  glasses. Release: Hugging Face `Jyun-Ting/Harmony4D` at `3fedb23f` (2024-11-09), MIT, not gated: 22 zips (7 test, 15 train),
  351.6 GB, one per scene and split, several sequences each. Code: `jyuntins/harmony4d`.
- Catalog dataset `dataforge-harmony4d`; identity `harmony4d__<split>__<scene>__<sequence>`
  (`harmony4d__test__01_hugging__002_hugging`).
- `property:episode:{split,scene,sequence,activity,num_people,people,num_exo_cameras,glasses}`, plus `glasses_left_out` on the
  29 takes whose glasses are not placed (see Glasses below);
  `property:capture:{fps,source_revision,clock_source,world_frame,source_num_frames,source_resolution,video_source}`.
- Camera-source metadata: GoPros `source_width/height` 3840×2160, `stored_width/height` 1920×1080 (videos above 3.2 MP are
  stored at half size, decoded straight to it by libjpeg's DCT scaling); Aria cameras native (rgb 1408×1408, slam 640×480 in
  device orientation) with `stream_id` 214-1 / 1201-1 / 1201-2. Every video is re-encoded (`cq` 36, `gop` 60) from the
  release's JPEG frames.

## Get the raw data

```bash
pixi run -e dataforge dataforge-download harmony4d --list-remote                     # the 22 zips and their sizes
pixi run -e dataforge dataforge-download harmony4d --sequences test/01_hugging       # one zip (1.7 GB) + the SMPL model
pixi run -e dataforge dataforge-convert harmony4d --sequences test/01_hugging/002_hugging
```

- `--sequences` takes archive keys (`train/04_sword_part1`) or sequence keys (`test/01_hugging/002_hugging`); a zip is the unit
  to fetch and prune (`remote_sequences` lists one entry per zip). Each zip is checked against its LFS sha256; complete files are
  not fetched again (huggingface_hub does not resume a broken transfer, it restarts it).
- Conversion reads the members straight from the zip; nothing is extracted. A zip may be deleted once its sequences are
  converted; the full run (2026-10-13) did exactly that, one zip at a time, keeping the local footprint under 55 GB.
- `body_mesh` needs the official neutral SMPL (only for its triangle list): `--smpl-model-root`, by default `<root>/body_models`,
  which `download` fills from the private `pablovela5620/mamma-streaming-data` (EPFL's pinned copy); anyone else places
  `smpl/SMPL_NEUTRAL.pkl` there.

## Raw inventory

| Shipped file or stream | Rate / clock | Layer | Entity or not ingested: reason |
| --- | --- | --- | --- |
| `exo/camNN/images/NNNNN.jpg` (3840×2160) | 20 fps, frame number | base | `/world/rig_NN/cam_00/pinhole/video` (AV1, 1920×1080) |
| `colmap/workplace/cameras.txt` (OPENCV_FISHEYE) | static | base | `Pinhole` + Kannala-Brandt k1..k4 at the stored size, `camera_model` |
| `colmap/workplace/images.txt` | static | base | `world_T_cam` of each GoPro (its earliest registration), scale divided out |
| `colmap/workplace/colmap_from_aria_transforms.pkl` / `scale.npy` | static | base | the world mapping; `property:capture:world_frame` names which |
| `colmap/workplace/{points3D.txt,temp.db,project.ini,aria_from_colmap_transforms.pkl}` | — | — | not ingested: the reconstruction's sparse points and database (the inverse transform is redundant) |
| `ego/ariaNN/images/{rgb,left,right}/NNNNN.jpg` (upright) | 20 fps | base | `/world/rig_9N/cam_0{0,1,2}/pinhole/video` (AV1, native size) |
| `ego/ariaNN/calib/NNNNN.txt` (FISHEYE624 + world→camera) | per frame | base, projections | `world_T_rig` per frame on `/world/rig_9N` (rgb camera); first frame's lens and slam offsets static; every frame's lens in projections |
| `processed_data/poses3d/NNNNN.npy` ({person: 17×(x,y,z,c)}) | 20 fps | body_pose | `/world/gt/people/person_NN/coco133_xyz` (COCO-17 in slots 0–16, shipped confidence) |
| `processed_data/smpl/NNNNN.npy` (global_orient, body_pose, betas, transl, epoch_loss, vertices, joints) | 20 fps | body_pose, body_mesh | `.../body/smpl` parameters, SMPL-45 `joints`, `fit_loss`; `.../body/mesh` shipped vertices at 10 Hz |
| `processed_data/bbox/<camera>/<mode>/NNNNN.npy` | 20 fps | projections | `<pinhole>/boxes/person_NN`, as shipped (scaled to the stored GoPro size) |
| `processed_data/poses2d/<camera>/<mode>/NNNNN.npy` (SMPL-45 2D) | 20 fps | — | not ingested (Pablo, 2026-10-13): the exact projection of the shipped SMPL joints (golden test: ≤ 2.5 px) |
| `exo/cam01/images/{rgb,output}.mp4` | — | — | not ingested: upstream preview videos (mpeg4) |

## Clocks

The release ships frame numbers, no timestamps: `video_time` = (frame − first frame) / 20 fps, from 0 at each sequence's first
frame; `frame_index` is the release frame number (from 1, as the file names). `property:capture:clock_source` = "release frame
number / 20 fps (the release ships frames, no capture timestamps)". Every stream and label is on that clock; the GoPros and
glasses were synchronised by the authors.

## Layers and entities

- **World.** aria01's Aria SLAM world: gravity-aligned, z up, metres; its origin is aria01's first head pose (~1.5 m above the
  floor), so the default 3D grid is off. Exo poses come from COLMAP through `colmap_from_aria[aria01]` (a similarity, scale ~1.8:
  divided out of every pose); each Aria's per-frame extrinsic through `inv(colmap_from_aria[ariaNN]) @ colmap_from_aria[aria01]`,
  as upstream's loaders do. A take that ships `scale.npy` lives in that world instead (COLMAP made metric and gravity-aligned),
  as upstream's loader decides: its labels are there, glasses or not (`train/12_mma` wears glasses and ships `scale.npy`; its
  labels sit metres from the Aria world). `property:capture:world_frame` names the world of each take.
- **Glasses only in the Aria world.** 29 takes ship glasses (`ego/ariaNN`, `colmap_from_aria_transforms.pkl`) *and*
  `scale.npy`: 5 in `train/11_karate3`, 6 in `train/12_mma`, 17 in `train/13_mma2`, 1 in `train/14_mma3`. dataforge logs no
  glasses for them and names the shipped pairs in `property:episode:glasses_left_out`. Why:
  - Upstream's loader (`lib/datasets/ego_exo_scene.py` in jyuntins/harmony4d) sets `sequence_mode = "exo"` whenever
    `colmap/workplace/scale.npy` exists, and then `ego_camera_names_with_mode = []`: it never uses glasses in such a take.
  - When dataforge did place them (through the release's Aria transforms, as for Aria-world takes), 31 of those 58 pairs sat
    more than 25 cm from every head, e.g. `11_karate3` aria02 0.96-1.46 m away, where a well-placed pair sits 6-10 cm from its
    wearer's head. The transforms do not reach the `scale.npy` world.
  - Everything else of those takes stays: GoPros, people, bodies, keypoints and GoPro boxes. Their Aria images and boxes stay
    in the release zip; the Aria panes of the default layout are empty there.
- **Rigs.** GoPro `camNN` = `rig_NN` (one camera). Glasses `ariaNN` = `rig_(90+NN)`: `cam_00` rgb, `cam_01` slam-left, `cam_02`
  slam-right; the rig frame is the rgb camera's device-orientation frame and moves per frame. The Aria images are upright: each
  Aria camera node logs the lens turned a quarter turn clockwise (`image_rotation_cw_deg` 90).
- **People.** `person_(NN−1)` after the key `ariaNN` (the key stays in takes without glasses): `body_pose` holds the SMPL
  parameters (72 axis-angle, 10 betas, transl), the SMPL-45 joints and the fit loss at 20 fps, NaN where a person has no fit;
  COCO-17 keypoints with the shipped confidence in COCO-133 slots 0–16 (the rest NaN, confidence 0). `body_mesh` holds the
  shipped posed vertices at 10 Hz (the dataforge mesh display rate) with SMPL's triangles.
- **2D.** No measured COCO 2D is shipped. `projections` holds `coco133_xyz` projected through each full lens model per person,
  `<pinhole>/coco133_uv_projected/person_NN`, with `property:projections:{derived_from,camera_model,calibration_source}`; plus
  the shipped person boxes. The GoPros project through the very Kannala-Brandt lens `base` logs (simplecv `project_fisheye62`:
  rays behind the camera, past the lens fold or outside the image are NaN); the Aria cameras through each frame's FISHEYE624
  in the upright image, inside projectaria-tools' Gen1 cone (1.0 rad rgb, 1.4 rad slam) and valid radius (scaled to 1408 px).
- **Fisheye rule.** Both lens models are fisheye (a pinhole is off by up to ~500 px at a GoPro's corners), so camera panes show
  the video, projected keypoints and boxes, and the meshes stay in 3D.
- **Blueprint.** The scene (videos hidden in 3D, grid off, orbital eye fitted per take) with aria01/aria02 rgb on the right and
  eight GoPros (cam01, 04, 06, 09, 12, 15, 17, 20: every take ships cam01-cam20) along the bottom; the card shows the scene
  beside cam01; table fields split, scene, people, frames. 119 of the 208 takes show no glasses, so their aria01/aria02 panes
  stay empty (one dataset blueprint serves every take): 90 whose zip has no `ego/` directory (e.g. every `05_sword2` take), and
  the 29 `scale.npy` takes above.

## Differences from simplecv

None: simplecv never read Harmony4D. Where we and the release differ:

- **Aria pixel convention.** The release turns Aria pixels with `x' = W − y`; we use `W − 1 − y` (projectaria-tools and the
  JPEG pixels), so our Aria projections sit ~1 px from the shipped 2D.
- **Exo registration.** The release poses a GoPro per frame with the nearest of its few COLMAP registrations (they agree within
  ~2 mm); we log one static pose, the earliest.
- **Aria boxes.** The release turns each box corner on its own, so many Aria boxes have x2 < x1, and boxes of people behind or
  beside the camera wrap to huge values. A label fault: logged as shipped.

## Parity with simplecv

No simplecv reader to compare with. Golden checks against the release instead (`tests/test_harmony4d_real.py`, 2026-10-13,
pablo-dl-server): the shipped SMPL-45 joints through our lenses land on the shipped `poses2d` within 2.5 px on the 22 GoPros
(median ≤ 0.5 px) and within 2 px on the six Aria cameras; the official neutral SMPL on the shipped parameters reproduces the
shipped vertices within 0.01 mm (measured 0.0007 mm). Pixel evidence: https://pablos-4800gt.ilish-ruler.ts.net:8768/dataforge-harmony-egohumans/

## Timing

`test/01_hugging/002_hugging` (301 frames, 15.05 s, 22 GoPros + 2 glasses), prod env: 44.7 s total, of which 38.8 s transcode
of 28 streams, four at a time (JPEG decode, ffmpeg's range conversion and NVENC per camera; which one limits is not yet
profiled on an idle host), 4.0 s projections; 183 MB of rrds from 1.7 GB of JPEG frames (~9×). Full-run figures: see the PR body.

## Known gaps

- `poses2d` (SMPL-45 2D) is not stored (decision); the release's `colmap` points and database are not stored.
- The world grid is off: the floor height varies per take (the world origin is a head pose); a ground-plane fit is a follow-up.
- Each GoPro's pose is the earliest COLMAP registration (≤ 2 px from the release's per-frame choice).
- Every layer is built from the zip. Once a converted zip is deleted, rebuilding one derived layer (projections after a lens
  fix, say) means fetching that zip again: no other layer keeps each frame's Aria calibration or the shipped boxes.
- `coco133_uv_projected` confidence is 1 wherever the lens places a joint, as in MAMMA and Ego-Exo4D; the shipped 3D confidence
  stays on `coco133_xyz` only.
- **Release fault, three takes:** `train/12_mma/{001,002,009}_mma` ship glasses and the Aria transforms but no `scale.npy`.
  Their labels are not in the Aria world: our projection, by upstream's own loader rule, lands 200-380 px (median) from the
  release's `poses2d`. That `poses2d` sits inside the release's boxes, so the release placed these labels in a world it does
  not ship. None of the Aria world, the raw COLMAP frame or a sibling take's `scale.npy` fits (205-586 px). Left as converted
  (a label fault, not corrected). Corpus check (projected COCO keypoints inside the shipped GoPro boxes, enlarged 25 %):
  1.000 on the other 205 takes, 0.10-0.18 on these three.
- **Release fault, Aria placement:** 42 of the 178 logged glasses (89 takes, all in the Aria world) sit more than 25 cm from
  every head (median over the take; a well-placed pair sits 6-10 cm from its wearer's): aria02 in all 35 `train/02_grappling`
  takes (0.4-1.6 m) and in `train/11_karate3/001`, and both pairs of the three `12_mma` takes above. Where the release ships
  Aria `poses2d` (`11_karate3/001`, `02_grappling/002`), our projection lands within 0.5 px of it: the release's own
  `colmap_from_aria_transforms.pkl` puts these glasses there. Left as converted. (The 29 `scale.npy` takes log no glasses at
  all; see Glasses only in the Aria world.)
- **Release fault, person keys:** in `train/02_grappling/{034,035,038,040}_grappling` aria01 sits 8-10 cm from the other
  person's head and 0.5-0.7 m from its wearer's: the labels' `aria01`/`aria02` keys look swapped there. Left as converted.
- Aria joints just past projectaria-tools' Gen1 cone or valid radius are NaN, where the release's own 2D still places them (12
  of 453 sampled on `002_hugging`, at 57-64° off-axis for rgb and 80-88° for slam).
