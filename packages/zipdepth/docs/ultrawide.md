# Ultrawide training lane for ZipDepth-PromptDA

## Goal

Teach the prompted student to produce metric depth for the **rectified ultrawide
camera** while it is prompted with the **wide** camera's ARKit LiDAR map. The
ultrawide sees about 2.1x the wide field of view on both axes, so most of an
ultrawide frame has no LiDAR support at all. The model must therefore keep its
metric scale from the prompted centre and extend it outward from image evidence.

The lane is off by default: `TrainCatalogConfig.cameras` is `"wide"`, and
`packages/zipdepth/tests/test_catalog_ultrawide.py` proves the wide sample is
bit-identical with and without an ultrawide policy attached to the builder.

## Data

New catalog layer `ultrawide_depth`, registered on **2,024 of the 2,025**
PromptDA segments (`43649417` lacks it; the lane logs one warning and skips it).
Under `world/rig_00/cam_01/pinhole_rect`:

| Entity | Component | Content |
| --- | --- | --- |
| `pinhole_rect` | `Pinhole:image_from_camera`, `Pinhole:resolution` | rectified intrinsics |
| `pinhole_rect/rgb` | `EncodedImage:blob` | JPEG quality 90, 640x480 landscape or 480x640 portrait |
| `pinhole_rect/depth` | `EncodedDepthImage:blob` | uint16 millimetre PNG, raycast from the PromptDA TSDF mesh; 0 is a miss |

Rows exist only at the ultrawide chosen frames (10 Hz, 480 to 1,840 per segment).
Those timestamps are **disjoint** from the wide chosen frames — on `47115416`
only 1 of 651 ultrawide timestamps coincides with a wide one — so the wide and
ultrawide passes are genuinely different views of a capture, not duplicates.

Portrait captures store the rectified frame as 480x640 — 813 of the 2,024
segments — because the layer writer (`gauss_surf/apis/ultrawide_depth_batch.py`,
PR #214) reads `image_wh` straight from the calibration and has no orientation
guard of its own. The lane rotates to landscape with the same
`property:capture:orientation_quarter_turns_ccw` the wide path already uses, then
resizes to the lane's 768x1024. Because nothing upstream enforces this, the
builders assert that an ultrawide frame **is** landscape after the turns and fail
loudly otherwise, rather than training on a rotated image.

Rows can also be missing a payload. The ultrawide track has its own
`drop_leading` and its own 10 Hz selection, so an ultrawide chosen frame can
precede the first ARKit lowres depth row; latest-at cannot fill a column that has
not been logged yet, so that row's prompt is null. The lane drops any chosen row
whose RGB, depth, prompt, or confidence cell is null and counts it in
`skipped_missing_payload_frames`. A preflight over all 2,024 segments found
**zero** such rows today (see *Preflight* below), but the guard is what keeps a
future re-registration from killing a producer mid-run.

## Why the prompt is resized and padded, not reprojected

The prompt could be reprojected into ultrawide pixels, leaving the outer ring
empty. That is strictly more work and buys nothing, because of where the prompt
is consumed:

* ZipDepth-PromptDA bilinearly resamples the 192x256 prompt to each decoder
  feature size. The finest is H/4, which at the 768x1024 training input is
  192x256 — exactly the canvas size. A prompt canvas that covered the full
  ultrawide frame would be downsampled to the same 122x91 block of information
  before the decoder ever reads it.
* The ARKit map is itself an upsampled product: a few hundred LiDAR returns
  smeared over 256x192. Reprojecting it to a finer ultrawide grid would invent
  detail that the sensor never measured.

So the lane resamples the oriented wide prompt to `round(256 * s) x round(192 * s)
= 122 x 91` with nearest-exact resampling — every output value stays a real LiDAR
reading — and writes it into a **zeroed** 192x256 canvas at `top=50, left=67`.
Zero means "no prompt": the model's own `[0.1, 4.0] m` range gate already treats
0 as invalid, so no extra signalling is needed and `prompt_valid` follows the
same rule.

Mirroring is applied to the **block**, before padding, so a horizontal flip stays
exact for any canvas margin (at the default scale the horizontal margin is 67 on
both sides, but the vertical one is 50/51).

## The 2.1 constant

`ULTRAWIDE_FOV_RATIO = 2.1`, `prompt_scale = 1 / 2.1`.

Measured from the catalog calibration over 24 segments spanning both stored
orientations, as the per-axis ratio of `resolution / focal_length`:

```
fov ratio x: mean 2.1087  sd 0.0152  min 2.0688  max 2.1383
fov ratio y: mean 2.1087  sd 0.0152  min 2.0688  max 2.1383
principal-point offset from centre: <= 0.66% of width, <= 0.69% of height
```

x and y agree exactly because both cameras are square-pixel and share a 4:3
aspect. Example (`47115416`): wide `f = 1590.66` at 1920x1440, ultrawide
`f = 252.14` at 640x480, giving `(640/252.14) / (1920/1590.66) = 2.103`.

The principal point sits within 0.7% of the image centre, which is why a
**centred** block is the right model; a per-segment offset block would move the
prompt by at most 1.3 canvas pixels and is not worth the complexity.

## Prompt timing

The ARKit lowres depth (`cam_00/pinhole_lowres/depth`) runs at 60 Hz, and the
ultrawide rows are at 10 Hz on their own grid. The query uses
`reader(index="video_time", fill_latest_at=True)` filtered on the ultrawide depth
blob being non-null, so each ultrawide frame takes the **nearest LiDAR map at or
before** its timestamp. Measured staleness on `47115416` and `42444511`:
`max 16.1 ms, mean 7.0 ms`, i.e. bounded by one 17 ms ARKit interval, with no
frame preceding the first ARKit sample.

## Mask policy

The raycast target has holes wherever the TSDF mesh has none: glazing, outdoor
views through windows, and thin structure.

* **Erosion `ultrawide_valid_erosion_px = 1`.** Raycast silhouettes bleed about
  one output pixel past the mesh, so the boundary ring is the least trustworthy
  supervision in the frame. Implemented as a max pool over the inverted mask, so
  the CPU and CUDA builders run the same kernel and agree exactly. Zero padding
  leaves the image border valid: genuine border holes are already invalid and
  erode inward on their own.
* **Drop `ultrawide_min_valid_fraction = 0.7`,** measured on the resized target
  *before* erosion so the threshold does not move when the erosion radius does.
  Sampled valid fractions over 41 frames per segment: `47115416` mean 0.914,
  p10 0.866, min 0.669, 95% of frames at or above 0.7; `42444511` mean 0.958,
  p10 0.901, min 0.612, 98% above 0.7. The gate therefore costs a few percent of
  frames and removes the ones that teach the least per decoded frame.

The existing flat-frame filter (`min_depth_span`, p95/p5 of valid depth) applies
unchanged.

Evaluation defaults `ultrawide_min_valid_fraction` to **0.0**: dropping sparse
frames is a training data-efficiency choice, not a property of the benchmark.
Erosion still applies, so eval and training score the same pixels per frame.

## The 50/50 sampler

With `cameras="both"`, each producer prepares a segment once and yields:

* every wide chosen frame, exactly as today; and
* a **seeded random subsample of the ultrawide chosen frames of the same size**,

round-robin interleaved, after which the existing shuffle buffer mixes them into
roughly balanced batches. The subsample seed is the segment seed, which already
mixes run seed, epoch, rank, and segment index, so every pass draws a fresh
subset and the whole ultrawide set is seen over several epochs. The ultrawide
augmentation seed stream is offset from the wide one so the two cameras of a
segment never share an augmentation draw.

In practice the two counts are close already (`47115416`: 651 wide, 651
ultrawide; `42444511`: 528 and 528), so the subsample mostly reorders rather
than discards.

Two edge cases:

* A segment lacking the `ultrawide_depth` layer (1 of 2,025) **falls back to
  wide-only** under `both`; under `ultrawide` it yields nothing. Either way the
  lane warns once for the whole run.
* A segment with no wide chosen frames is skipped entirely under `both`. There is
  nothing to balance the ultrawide track against, and streaming its whole 10 Hz
  selection (up to 1,840 frames) would silently skew the mix.

**Known limitation.** The subsample is seeded by the pass index, so a run resumed
from a checkpoint restarts at pass 0 and re-draws the subsets the original run
already used. Over 140k steps that costs some ultrawide coverage after a resume.
It is not a correctness problem and is left for a follow-up that carries the pass
index in the checkpoint.

## Measured cost

`tools/catalog_throughput.py`, 8 segments, 8 producers, batch 8, 768x1024 out,
CUDA builder, RTX 5090 (one full pass per configuration):

| cameras | frames/s | segment_query | video_decode | jpeg_decode | blob_decode | augment |
| --- | --- | --- | --- | --- | --- | --- |
| wide | 173.9 | 11.92 | 5.53 | 0.00 | 4.00 | 0.52 |
| both | 344.2 | 5.00 | 3.64 | 0.39 | 3.29 | 1.14 |
| ultrawide | 681.3 | 1.73 | 0.00 | 0.75 | 2.50 | 2.24 |

(stage columns in ms/frame.)

The ultrawide lane is far cheaper than the wide one: no NVDEC, a 640x480 JPEG
instead of a 1920x1440 video frame, and a 640x480 depth PNG instead of a
1920x1440 one. Mixing the cameras therefore **doubles** loader throughput rather
than costing anything, and the lane stays comfortably ahead of the ~150 frames/s
the trainer consumes. JPEG decode is the new stage and is not the bottleneck.

The RGB JPEG is decoded on the **CPU** with `torchvision.io.decode_jpeg`, not on
the GPU with nvjpeg. Measured on this host: nvjpeg 0.171 ms/frame batched versus
0.566 ms/frame on the CPU — but nvjpeg's output differs from libjpeg-turbo's by
up to 27/255 per channel, which would break the CPU/CUDA builder parity the
probe and evaluation paths rely on, and torchvision's nvjpeg handle is shared
across the producer threads. Buying 0.4 ms/frame is not worth either risk while
the loader already runs at twice the trainer's rate.

## The widened output range

The sibling PR (`zipdepth/uw-1-range`) adds `range_margin_m` to `ZipDepthPrompt`,
with a matching `TrainCatalogConfig.range_margin_m` and checkpoint key. The
margin is an **absolute distance in metres**, not a fraction of the prompt span:

```
min_out = clamp(prompt_min - range_margin_m, min=0.1)
max_out = clamp(prompt_max + range_margin_m, max=4.0)
```

The prompted model otherwise gates its output to the prompt's own depth span. On
an ultrawide frame the prompt covers only the central 1/2.1, so the true depth
**outside** the footprint routinely falls outside the prompted span — a corridor
continuing past the prompted wall, or a ceiling entering only at the frame edge.

A span-relative margin does not fix this. Half of a close-up prompt's span turns
`0.8-1.2 m` into `0.6-1.4 m`, and a far wall in the periphery stays unreachable
exactly where the ultrawide most needs the room. The uw-v1 fine-tune therefore
uses `range_margin_m = 3.9`, which opens the head to the full `[0.1, 4.0] m`
window for every frame.

That does not throw away the prompt: scale still enters through the unchanged
prompt-normalization path, which is what anchors the prediction metrically. The
head's range only controls where its sigmoid can land. The cost of the wider
sigmoid is quantization precision — spreading the same output resolution over
3.9 m instead of a 0.4 m span — and that works out to a few millimetres, far
below the lane's ~70 mm edge-MAE gate.

## Training recipe

Fine-tune from `zdpda-v4` (`data/checkpoints/zdpda-v4/final_model.pth`):

```
--cameras both --preset fast --max-lr 1e-4 --batch-size 8
--height 768 --width 1024 --total-steps 140000 --freeze-bn --target-mode metric
--range-margin-m 3.9
```

(`--range-margin-m` lands with the sibling PR; this lane does not read it.)

Resize-only augmentation (the lane's `build_train_transform`: flip plus mild
colour, no crop zoom), BatchNorm pinned to the released running statistics, and
the same AdamW/OneCycle machinery as the wide lane. 140k steps is about one pass
over the mixed corpus at batch 8.

## Evaluation

20-segment seed-0 holdout, `frame_stride=10`, `--cameras both`.

Ultrawide metrics are reported **split by the prompt footprint**, the centred
`[200:564, 268:756]` box of the 768x1024 output:

* `ultrawide_whole_*` — AbsRel, delta1, MAE over every valid pixel;
* `ultrawide_inside_*` — inside the footprint, against
  `ultrawide_inside_prompt_upsample_*`, the zero-parameter bilinear
  prompt-upsample floor the student must beat there;
* `ultrawide_outside_*` — **the headline**: pixels with no prompt at all.

The wide gates are unchanged and must still hold: AbsRel <= 0.0140, edge
MAE <= 73 mm, hard-20 macro edge MAE <= 128.5 mm.

## Deployment

One static 768x1024 batch-8 TensorRT engine serves both cameras. Ultrawide
callers upscale their 640x480 rectified frame to 768x1024 on the way in and
downscale the prediction on the way out; the prompt is padded exactly as in
training. No second engine shape, no second set of weights.

## Fallback if inside-footprint numbers regress

If the outside-footprint metrics are good but inside-footprint quality drops
below the wide lane's, the cause is prompt resolution: the 122x91 block is all
the decoder's finest H/4 level ever sees. The fallback is to run the ultrawide at
a **higher input resolution** so H/4 grows (e.g. 1536x2048 in, H/4 = 384x512,
prompt block 244x182). That costs roughly 4x the compute and forces a second
engine shape, which is why it is the fallback and not the design.

## Naming

Checkpoints from this lane are `zdpda-uw-vN`, kept separate from the wide-only
`zdpda-vN` series so a wide-lane regression can never be blamed on the mixed
recipe by accident.
