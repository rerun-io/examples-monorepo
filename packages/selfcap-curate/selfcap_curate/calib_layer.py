"""One calibration-only layer per episode; no hands, predictions, or catalog writes."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rerun as rr
from dataforge import schema
from dataforge.writing import atomic_recording
from jaxtyping import Float64
from numpy import ndarray
from serde import serde, to_dict
from serde.json import to_json

from selfcap_curate.exo_calibration import SessionCalibration
from selfcap_curate.exo_refine import Refinement
from selfcap_curate.factory import QuestBlock
from selfcap_curate.records import OakFit, SessionEpisode, oak_quest_offset_ms

METHOD: str = "selfcap-calib v1 (2026-10-16)"
WORLD: str = "Quest world: head_pose.csv frame, metres, Y-up; cameras OpenCV RDF"


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class CalibSession:
    session: str  # Date__sid8 session id.
    exo: SessionCalibration  # Refined exo cameras and episode overrides.
    oak: OakFit  # Session RGB mount in Quest rig coordinates.
    quest: QuestBlock  # Factory pinholes and resolved camera-50 borrowing.
    stereo: dict[str, Float64[ndarray, "4 4"]]  # Factory rgb_T_left and rgb_T_right in metres.
    version: str = "selfcap-curate-calib-v1"  # Implementation provenance, distinct from the legacy method label.
    exo_refinements: dict[str, Refinement] = field(default_factory=dict)
    """Opt-in episode-specific optical calibration and its complete provenance."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Provenance:
    session: str  # Source session.
    method: str  # Calibration method.
    episode: str  # Episode id.


def write_calib_layer(path: Path, session: CalibSession, motion: SessionEpisode) -> None:
    """Write RDF transforms/pinholes and one calib property group atomically."""
    layout = motion.layout
    sync, oak_rig, quest_rig = motion.sync, layout.oak_rig, layout.quest_rig
    refinement = session.exo_refinements.get(sync.episode_key.segment)
    if layout.phones != tuple(sorted(sync.exo_fit)):
        raise ValueError(f"{sync.key}: base phone layout differs from sync exo_fit")
    with atomic_recording(path, recording_id=sync.episode_key.segment, send_properties=False) as recording:

        def tf(entity: str, matrix: Float64[ndarray, "4 4"], **extra: str | float | None) -> None:
            recording.log(entity, rr.Transform3D(translation=matrix[:3, 3], mat3x3=matrix[:3, :3]), static=True)
            recording.log(entity, rr.AnyValues(drop_untyped_nones=True, **{k: v for k, v in extra.items() if v is not None}), static=True)

        def pinhole(entity: str, matrix: Float64[ndarray, "3 3"], width: int, height: int) -> None:
            recording.log(
                entity,
                rr.Pinhole(
                    image_from_camera=matrix.astype(np.float32),
                    width=width,
                    height=height,
                    camera_xyz=rr.ViewCoordinates.RDF,
                    image_plane_distance=0.3,
                ),
                rr.AnyValues(lens_source="fitted"),  # replaces the base layer's "assumed" lens
                static=True,
            )

        count, flagged = 0, 0
        for rig, phone in enumerate(sorted(sync.exo_fit)):
            camera = session.exo.phones.get(phone)
            if camera is None or camera.camera is None:
                continue
            episode = camera.episodes.get(sync.episode)
            refined = refinement.camera_for_rig(rig) if refinement is not None else None
            matrix = (refined.world_T_cam if refined is not None else
                      episode.world_T_cam if episode is not None and episode.world_T_cam is not None else camera.camera.world_T_cam)
            status = "flagged" if episode is not None and not episode.ok else "ok"
            count += 1
            flagged += status == "flagged"
            entity = schema.cam_path(rig, 0)
            tf(
                entity,
                matrix,
                calib_source=refinement.method if refinement is not None and refined is not None else METHOD,
                calib_scope=episode.scope if episode else "session",
                calib_status=status,
                calib_phone=phone,
                calib_hand_median_px=(episode.median_px if episode is not None else camera.median_px),
            )
            p = (refined if refined is not None else camera.camera).parameters
            pinhole(schema.pinhole_path(rig, 0), np.array([[p[6], 0.0, p[7]], [0.0, p[6], p[8]], [0.0, 0.0, 1.0]]), 1920, 1080)
        # The OAK stereo and healthy-Quest camera offsets are factory facts written by the base layer.
        qp = oak_quest_offset_ms(sync)
        mount = session.oak.questrig_T_rgb if session.oak.ok else None
        posed = qp is not None and mount is not None
        if posed:
            # Same static components as the base layer's nominal prior, so this fit replaces it. The aligned base puts the
            # Quest track on the OAK's clock, so the static mount needs no time shift.
            recording.log(
                schema.rig_path(oak_rig),
                rr.Transform3D(translation=mount[:3, 3], mat3x3=mount[:3, :3], parent_frame=f"tf#{schema.rig_path(quest_rig)}"),
                rr.AnyValues(oak_mount_source="fitted", calib_source=METHOD, calib_oak_source=session.oak.observation_source,
                             calib_oak_median_px=session.oak.median_px),
                static=True,
            )
        if session.quest.mode != "healthy":  # a borrowed camera-50 transform is an estimate, so it lives here
            for slot, matrix in session.quest.rig_T_cam.items():
                tf(schema.cam_path(quest_rig, int(slot)), matrix, calib_source=f"quest calibration.json ({session.quest.mode})")
        recording.send_property(
            "calib",
            rr.AnyValues(
                drop_untyped_nones=True,
                method=METHOD,
                version=session.version,
                world=WORLD,
                session=session.session,
                exo_cameras=count,
                exo_flagged=flagged,
                exo_bundle_body_median_px=session.exo.body_median_px,
                oak_posed=posed,
                oak_unposed_reason=None if posed else ("missing Quest/OAK clock" if qp is None else "OAK mount not fitted"),
                oak_median_px=session.oak.median_px,
                oak_rotation_source=("gyro" if session.oak.rotation_source.startswith("gyro") else
                                     "unposed" if session.oak.rotation_source.startswith("unposed") else "free"),
                oak_rotation_reason=session.oak.rotation_reason,
                quest_mode=session.quest.mode,
                record_json=json.dumps(to_dict(Provenance(session.session, METHOD, sync.episode))),
                exo_refinement_json=to_json(refinement) if refinement is not None else None,
                exo_label_layer_hash=refinement.label_layer_hash if refinement is not None else None,
                exo_focal_source=refinement.focal_source if refinement is not None else None,
                exo_refinement_version=refinement.version if refinement is not None else None,
            ),
        )
