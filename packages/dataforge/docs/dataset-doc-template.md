# <Dataset>

## Source

- Source URL, revision, license, raw root, and converted root.
- Source resolution and revision recording properties.
- Clock origin, native stream rates, timestamp units, and `property:capture:clock_source`.

## Raw inventory

Account for every shipped file and stream, including unused ones.

| Shipped file or stream | Rate / clock | Layer | Entity or not ingested: reason |
| --- | --- | --- | --- |
| `<path>` | `<native rate and clock>` | `<layer>` | `<entity>` |
| `<path>` | `<native rate and clock>` | — | not ingested: `<reason>` |

## Layers and entities

List each layer, its entities, components, units, and confidence source. State
which keypoint slots are derived or missing. Use `video_time` as the true timeline
and `frame_index` as the second timeline. Only write shipped 2D points.

Present joints retain shipped confidence; use 1.0 only when none is shipped.
Missing joints and uncovered COCO slots have NaN positions and confidence 0.0.
Document the Assembly-Hands thumb-base midpoint and wrist-copy slots where used.

## Differences from simplecv

For each difference, name the raw-source evidence and audit defect it fixes.
Include parity sequences, matched timestamps, errors (gate: 1e-4 m), and pixel
evidence. An unexplained difference blocks the port.

## Timing

Read `<output_root>/timing/convert.jsonl` with
`dataforge.timing.load_convert_records`; registration times are in
`timing/register.jsonl`. Report host, converter version, selected sequences,
output bytes, capture seconds, and elapsed seconds per capture-minute. Separate
skipped runs. Compare with simplecv preprocessing plus conversion in the prod
environment on the same sequences. Stages can overlap; total is elapsed wall time.
Link the benchmark report and record the soft gate before a full-corpus run.
