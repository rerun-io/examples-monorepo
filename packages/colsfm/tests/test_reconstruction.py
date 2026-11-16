"""`build_reconstruction`, checked against the metadata the two shipped datasets carry.

The seam is `colsfm.reconstruction.build_reconstruction(frames_meta)` and the
`pycolmap.Reconstruction` it returns: poses read back through COLMAP's own
accessors, never through the module's internals.
"""

from __future__ import annotations

import numpy as np
import pycolmap
from conftest import pose_delta

from colsfm.frames_meta import FramesMeta, KeyframeMeta
from colsfm.reconstruction import (
    build_reconstruction,
)

POSE_TOLERANCE: float = 1e-9
"""Round-trip tolerance in metres and in the matrix norm; this is pure algebra."""


def test_a_rig_inexact_input_snaps_to_the_rig(galileo_input: FramesMeta) -> None:
    """The raw cuVSLAM metadata is not rig-exact, and the rig model rounds it off.

    `data/r2b_galileo/frames_meta.json` carries an independent `camera_to_world`
    per image, and the vehicle poses those imply disagree across the eight
    cameras of one `synced_sample_id` by 13.7 um and 4.5e-4 deg — measured
    straight off the file, with no reconstruction involved. A single rig pose per
    frame cannot reproduce all eight, so the round trip is exact only for the
    rig-exact files cuSFM's own stages write (`pose_graph/`, `kpmap/keyframes/`).
    The residual is three orders of magnitude below the 5 mm the mapper moves
    poses by, so nothing downstream notices.
    """
    frames_meta: FramesMeta = galileo_input
    keyframe_by_id: dict[int, KeyframeMeta] = frames_meta.keyframe_by_id()
    worst_spread_m: float = max(
        float(
            np.linalg.norm(
                np.asarray(frames_meta.world_T_vehicle(keyframe_by_id[keyframe_id]).translation)
                - np.asarray(frames_meta.world_T_vehicle(keyframe_by_id[rig_frame.reference_keyframe_id]).translation)
            )
        )
        for rig_frame in frames_meta.rig_frames()
        for keyframe_id in rig_frame.keyframe_ids
    )
    assert 1e-6 < worst_spread_m < 1e-4

    reconstruction: pycolmap.Reconstruction = build_reconstruction(frames_meta).reconstruction
    worst_translation_m: float = max(
        pose_delta(reconstruction.image(keyframe.keyframe_id).cam_from_world().inverse(), keyframe.world_T_cam)[0]
        for keyframe in frames_meta.keyframes
    )
    print(f"[colsfm] raw Galileo input: rig spread {worst_spread_m * 1e6:.2f} um, snap {worst_translation_m * 1e6:.2f} um")
    # The snap is the spread carried through each camera's lever arm, so it is
    # the same size but not the same number.
    assert worst_spread_m < worst_translation_m < 1e-4


REFERENCE_CAMERA_PARAMS_ID: int = 0
"""The camera the extrinsic-refinement tests put the rig origin on."""


