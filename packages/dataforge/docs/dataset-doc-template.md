# <Dataset>

Every dataset doc uses these sections, in this order. Keep each one short; move
long reports to a linked artifact. The shared rules (confidence, timelines,
measured vs projected 2D, timing records) are in the README's "Conventions for
exoego ports"; state only how this dataset applies them.

## Source

- Upstream release, mirror and pinned revision; license or access gate.
- Catalog name, sample catalog name and recording identity.
- Capture and episode properties, including `source_revision` and `clock_source`.
- Camera-source metadata, written only through `logging_toolkit.log_camera_source`. Every
  camera node carries `source_width`, `source_height` (the grid the calibration and 2D labels
  describe) and `video_codec`. `cq`/`gop` appear when dataforge re-encoded the video and are
  absent when the source bitstream is remuxed unchanged. `stored_width`/`stored_height` appear
  only when the source ships the video rescaled, `stream_id` only for a source with stream ids,
  and `source_num_frames` only when cameras differ in frame count.
  `property:capture:source_resolution` lists the same sizes as `"<id>:<W>x<H>"`, one per camera
  (id = `stream_id`, else the camera name). Other calibration facts (distortion, provenance)
  are logged beside the node, outside this contract.

## Get the raw data

- `dataforge-download` commands: everything, `--sequences`, `--list-remote`
  (or "verify only" when the source can no longer be fetched).
- Sizes, what is fetched and what is skipped, shared files that must stay.
- Resume, verification and mismatch policy; what, if anything, is deleted.
- Default raw root, overrides, and the output-root rules.

## Raw inventory

Account for every shipped file and stream, including unused ones.

| Shipped file or stream | Rate / clock | Layer | Entity or not ingested: reason |
| --- | --- | --- | --- |
| `<path>` | `<native rate and clock>` | `<layer>` | `<entity>` |
| `<path>` | `<native rate and clock>` | — | not ingested: `<reason>` |

## Clocks

What `video_time` is (source, unit, origin, rebased or not), what `frame_index`
counts, the native rate of each stream, how labels join the camera clock, and the
exact `property:capture:clock_source` rule.

## Layers and entities

List each layer, its entities, components, units, and confidence source. State
which keypoint slots are derived (the Assembly-Hands thumb-base midpoint and wrist
copies) or missing, and the default blueprint and table card.

State which 2D the dataset has:

- **Measured:** `<pinhole>/coco133_uv` holds only the 2D the source ships.
- **Projected:** `<pinhole>/coco133_uv_projected`, in the `projections` layer,
  holds `coco133_xyz` projected through the camera's lens model, with
  `property:projections:derived_from` and `property:projections:camera_model`.

Name the lens model and say which panes hide 3D content because the viewer's
pinhole cannot apply it.

## Differences from simplecv

For each difference, name the raw-source evidence and the audit defect it fixes.
An unexplained difference blocks the port.

## Parity with simplecv

Date, host, both commits and environments, the parity sequences and how rows are
matched. Give joint counts, maximum errors (gate: 1e-4 m), the link to the full
report, and pixel evidence URLs.

## Timing

Read `<output_root>/timing/convert.jsonl` with
`dataforge.timing.load_records(path, ConvertRecord)`; read registration times
in `timing/register.jsonl` with `load_records(path, RegisterRecord)`. Report host, converter version, selected sequences,
output bytes, capture seconds, and elapsed seconds per capture-minute. Separate
skipped runs. Compare with simplecv preprocessing plus conversion in the prod
environment on the same sequences. Stages can overlap; total is elapsed wall time.
Record the soft gate before a full-corpus run.

## Known gaps

Deferred streams and layers, open checks, missed gates and their levers.
