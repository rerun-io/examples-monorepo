# cuSFM protobuf schema

`cusfm_protos.fdset` is a `FileDescriptorSet` holding the 31 `.proto` files that
cuSFM's prebuilt binaries embed in their `protodesc_cold` ELF section. It is the
contract that makes reading and writing cuSFM protobufs from Python legitimate
rather than guesswork: `colsfm/schema.py` and `tools/raco_extract.py` build real
message classes from it, so the keyframes, configs and metadata they touch are
serialised by protobuf itself.

The set covers all 21 binaries, not just the extractor: on top of the original
19 files it now carries `visual/cusfm/pose_graph.proto` (`PoseGraph`,
`PoseGraphConfig`, `ConstraintPair`), `visual/cusfm/vision_mapping_config.proto`,
`visual/geometry/bundle_adjustment_config.proto`, `common/ransac/ransac_config.proto`,
`common/image/pnp_config.proto`, `visual/cusfm/association_params.proto`,
`visual/loop_closing/*` and `visual/cuvgl/*`.

Verified by round trip: parsing every keyframe that cuSFM's own
`feature_extractor_main` produced under `data/cusfm_runs/galileo/**/keyframes/`
and re-serialising it reproduces the file **byte for byte** — 516 of 516,
e.g. `cusfm/keyframes/back_stereo_camera_left/1707938736136244532.pb`
(1,080,911 bytes, sha256 prefix `b1489decbbe0d920`).

It is vendored because it is 30 KB of schema and the alternative -- extracting it
from the binaries on every machine -- made the extractor and its contract tests
fail anywhere but the machine that first produced it.

Regenerate with

    pixi run -e colsfm python tools/extract_cusfm_schema.py

which reads the descriptors back out of `pycusfm/x86_cuda13/bin/`. Upstream is
frozen, so this should only ever be needed after a `pycusfm` version bump -- and
if the regenerated set differs, the message layout changed and
`colsfm/schema.py` plus `tools/raco_extract.py` need review.
