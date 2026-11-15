"""LightGlue+ matching on RaCo-ALIKED features, through TensorRT.

The `raco` half of `colsfm.matching`. LightGlue+ is the matcher fabio-sim
publishes *trained against RaCo-ALIKED* — upstream's
`Extractor.raco_aliked.lightglue_config` — and it is the other half of the
"RaCo-ALIKED-LightGlue+" pair the blog post benchmarks. The graph is
`packages/colsfm/data/models/raco/aliked_lightglue/lightglue_aliked.onnx`, exported by
`tools/export_cusfm_raco.py` in the `raco` Pixi environment; the engine is built
and cached in the `colsfm` environment's TensorRT 10.13 by
`colsfm.tensorrt_runtime.resolve_engine`, under its
`<stem>_fp16_<digest>_<trt-version>_sm_<arch>.engine` name.

**Everything except the input frame is `colsfm.matching_trt`.** Same four
bindings, same `[n, 2]` int32 `matches0` and `[n]` float32 `mscores0` with a
zero-score sentinel in the unused rows, same score gate at
`MatchingOptions.tensorrt_min_score` (0.3, the blob's `match_threshold`), same
SSC spatial NMS on the LightGlue score before verification, same
`pycolmap.verify_matches` afterwards. `match_pairs_tensorrt` therefore does the
work and this module supplies three constants and one function.

**The one difference: the frame `kpts0`/`kpts1` are in.** The blob's graph takes
the standard LightGlue normalisation, `(p - size/2) / (max(size)/2)`, and does
its own scaling internally. Upstream's adapter instead takes cuSFM's
`align_corners=False` normalisation over the *network* input — the extractor's
own `keypoints` output, unchanged — and applies `[1, H_proc / W]` inside the
graph, which composes to exactly the same LightGlue frame. So the round trip is
`normalized_to_pixels` on the way into the database and
`normalize_keypoints_raco` on the way back out, and it is exact: both use
`(size - 1)` and they are inverses.

**Pairs run one at a time.** The exported graph's batch axis is fixed at one
pair, and widening it means re-exporting in the `raco` environment and rebuilding
an engine. Measured on Galileo the matching stage is already the small half of
the budget, so the batching that matters is the extractor's
(`colsfm.features_raco`), and this note is why it stops there.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
from jaxtyping import Float32
from numpy import ndarray

from colsfm.database import ImagePair, KeypointsXY
from colsfm.matching import MatchingOptions, MatchReport
from colsfm.matching_trt import LIGHTGLUE_PROFILE, match_pairs_tensorrt
from colsfm.model_assets import RACO_LIGHTGLUE_ONNX_PATH
from colsfm.tensorrt_runtime import ShapeProfile

RACO_LIGHTGLUE_PROFILE: Final[dict[str, ShapeProfile]] = LIGHTGLUE_PROFILE
"""The same keypoint-axis profile as the blob's graph: minimum 1, optimal 3200, maximum 5500.

Both graphs are fed 2048 keypoints per image, so reusing the blob's numbers keeps
the two engines comparable rather than tuning one of them for the measurement."""


def normalize_keypoints_raco(
    keypoints_xy: KeypointsXY, image_width: int, image_height: int
) -> Float32[ndarray, "1 num_keypoints 2"]:
    """Put pixel keypoints back into the frame the RaCo extractor emitted them in.

    The exact inverse of `colsfm.features_trt.normalized_to_pixels`,
    `2 * p / (size - 1) - 1`, which is what the exported LightGlue+ adapter takes
    before it applies its own `[1, H_proc / W]` scaling. Passing the blob's
    `(p - size/2) / (max(size)/2)` here would scale x correctly by luck and y by
    `H / W`, which silently degrades matching rather than failing.

    Args:
        keypoints_xy: Float32 `[num_keypoints, 2]` xy pixel coordinates.
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        Float32 `[1, num_keypoints, 2]`, the engine's binding shape.
    """
    scale_xy: Float32[ndarray, " 2"] = np.asarray([(image_width - 1) / 2.0, (image_height - 1) / 2.0], dtype=np.float32)
    normalized: Float32[ndarray, "num_keypoints 2"] = np.asarray(keypoints_xy, dtype=np.float32) / scale_xy - np.float32(1.0)
    return np.ascontiguousarray(normalized[None])


def match_pairs_raco(database_path: Path, pairs: Sequence[ImagePair], options: MatchingOptions) -> MatchReport:
    """Match and verify a pair list with the RaCo-trained LightGlue+ engine.

    Args:
        database_path: An existing COLMAP database whose images carry RaCo
            keypoints and float descriptors.
        pairs: Image-id pairs, normalised and deduplicated by `colsfm.pairs`.
        options: Matching and verification settings.

    Returns:
        Per-pair raw and inlier counts, the device used and the wall time.

    Raises:
        FileNotFoundError: When the database is missing.
        KeyError: When a pair names an image with no features.
    """
    return match_pairs_tensorrt(
        database_path,
        pairs,
        options,
        onnx_path=RACO_LIGHTGLUE_ONNX_PATH,
        profile=RACO_LIGHTGLUE_PROFILE,
        normalize=normalize_keypoints_raco,
        label="raco",
    )
