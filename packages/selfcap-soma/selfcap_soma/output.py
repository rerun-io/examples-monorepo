"""Save fitted geometry and numerical diagnostics without catalog registration."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import trimesh
from jaxtyping import Float64
from numpy import ndarray
from serde import serde
from serde.json import to_json

from selfcap_soma.catalog import Observations
from selfcap_soma.fitting import FitResult
from selfcap_soma.observations import coco_joint_names


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Residual:
    median_mm: float | None
    p90_mm: float | None
    count: int


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Report:
    frames: int
    elapsed_s: float
    ms_per_frame: float
    peak_allocated_mib: float
    shared_scale: float
    quest_offset_ms: float
    requested_time_s: float
    sampled_time_s: float
    exo_body: Residual
    exo_hands: Residual
    quest_hands: Residual
    tracked_pose_to_anatomical_eye: Residual
    attachment_position: Residual
    ears: Residual
    head_anchor: str
    head_attachment_rotation: list[list[float]]
    head_offset_m: list[float]
    head_rotation_median_deg: float
    head_rotation_p90_deg: float
    target_fit_vs_quest: Residual
    target_exo_vs_quest: Residual
    target_fit_vs_exo: Residual
    limitations: tuple[str, ...]


def residual(a: Float64[ndarray, "... 3"], b: Float64[ndarray, "... 3"]) -> Residual:
    """Unweighted Euclidean error on jointly finite points only."""
    distances = np.linalg.norm(a - b, axis=-1)
    valid = distances[np.isfinite(distances)] * 1000.0
    return Residual(float(np.median(valid)) if valid.size else None, float(np.percentile(valid, 90)) if valid.size else None, int(valid.size))


def save(output: Path, data: Observations, result: FitResult, target_s: float, mesh_stride: int = 1) -> Report:
    """Persist a local comparison recording and a strict JSON report."""
    output.mkdir(parents=True, exist_ok=True)
    mapped = np.full_like(data.exo, np.nan)
    for coco, name in coco_joint_names().items():
        mapped[:, coco] = result.joints[:, result.names.index(name)]
    mapped[:, 3:5] = result.ear_landmarks
    frame = int(np.searchsorted(data.times, target_s, side="right") - 1)
    frame = max(0, min(frame, len(data.times) - 1))
    report = Report(
        len(data.times),
        result.seconds,
        1000.0 * result.seconds / len(data.times),
        result.peak_bytes / 2**20,
        result.scale,
        data.offset_ms,
        target_s,
        float(data.times[frame]),
        residual(mapped[:, 5:17], data.exo[:, 5:17]),
        residual(mapped[:, 91:], data.exo[:, 91:]),
        residual(mapped[:, 91:], data.hands[:, 91:]),
        residual(result.joints[:, result.names.index("LeftEye")].astype(np.float64), data.head_positions),
        residual(
            result.joints[:, result.names.index("Head" if result.head_anchor == "head" else "LeftEye")].astype(np.float64),
            data.head_positions
            + np.einsum(
                "nij,j->ni", data.head_rotations @ result.head_attachment if result.head_anchor == "head" else data.head_rotations, result.head_offset
            ),
        ),
        residual(result.ear_landmarks.astype(np.float64), data.exo[:, 3:5]),
        result.head_anchor,
        result.head_attachment.astype(float).tolist(),
        result.head_offset.astype(float).tolist(),
        float(np.median(result.head_angle_deg)),
        float(np.percentile(result.head_angle_deg, 90)),
        residual(mapped[frame, 91:], data.hands[frame, 91:]),
        residual(data.exo[frame, 91:], data.hands[frame, 91:]),
        residual(mapped[frame, 91:], data.exo[frame, 91:]),
        (
            "Training residuals, not independent accuracy measurements.",
            "Headset attachment is estimated from this segment; ear landmarks are hand-selected low-LOD surface vertices.",
            "Shared native identity, global scale and optional body bone ratios; no hand PCA or learned pose prior.",
            "No foot surface terms, contact model, image evidence, twist supervision or uncertainty estimate.",
            "Low-LOD mesh with procedural twists and pose correctives disabled.",
            "Temporal prior is timestamp-scaled velocity change in translation and axis-angle parameters.",
        ),
    )
    (output / "report.json").write_text(to_json(report))
    recording = rr.RecordingStream("selfcap-soma-prototype")
    recording.save(output / "fit.rrd")
    recording.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
    edges = [(int(parent) - 1, child - 1) for child, parent in enumerate(result.parents) if child > 0 and parent > 0]
    recording.log(
        "/world/fit/joints",
        rr.AnnotationContext(rr.ClassDescription(info=rr.AnnotationInfo(id=0, color=[60, 220, 170]), keypoint_connections=edges)),
        static=True,
    )
    recording.log(
        "/world/fit/joints", rr.Points3D.from_fields(keypoint_ids=np.arange(77), class_ids=0, colors=[60, 220, 170], radii=0.006), static=True
    )
    recording.log("/world/inputs/exo", rr.Points3D.from_fields(colors=[255, 150, 35], radii=0.009), static=True)
    recording.log("/world/inputs/quest", rr.Points3D.from_fields(colors=[220, 75, 230], radii=0.008), static=True)
    recording.log("/world/fit/mesh", rr.Mesh3D.from_fields(triangle_indices=result.faces, albedo_factor=[0.25, 0.65, 0.8, 0.35]), static=True)
    recording.log(
        "/description",
        rr.TextDocument("SOMA fit (cyan); triangulated points (orange); Quest hands (magenta).\n" + "\n".join(report.limitations)),
        static=True,
    )
    index = [rr.TimeColumn("video_time", duration=data.times)]
    recording.send_columns(
        "/world/fit/joints", indexes=index, columns=rr.Points3D.columns(positions=result.joints.reshape(-1, 3)).partition([77] * len(data.times))
    )
    recording.log("/world/fit/ears", rr.Points3D.from_fields(colors=[0, 255, 255], radii=0.008), static=True)
    recording.send_columns(
        "/world/fit/ears", indexes=index, columns=rr.Points3D.columns(positions=result.ear_landmarks.reshape(-1, 3)).partition([2] * len(data.times))
    )
    recording.send_columns(
        "/world/inputs/exo", indexes=index, columns=rr.Points3D.columns(positions=data.exo.reshape(-1, 3)).partition([133] * len(data.times))
    )
    recording.send_columns(
        "/world/inputs/quest", indexes=index, columns=rr.Points3D.columns(positions=data.hands.reshape(-1, 3)).partition([133] * len(data.times))
    )
    for name, points, color in (
        ("fit", mapped[:, 91:], [60, 220, 170]),
        ("exo", data.exo[:, 91:], [255, 150, 35]),
        ("quest", data.hands[:, 91:], [220, 75, 230]),
    ):
        # Each digit starts at the wrist, then follows its own four joints.
        hand_edges = [(base, base + 1 + finger * 4) for base in (0, 21) for finger in range(5)] + [
            (base + 1 + finger * 4 + joint, base + 2 + finger * 4 + joint) for base in (0, 21) for finger in range(5) for joint in range(3)
        ]
        path = f"/world/detail/{name}"
        recording.log(
            path, rr.AnnotationContext(rr.ClassDescription(info=rr.AnnotationInfo(id=0, color=color), keypoint_connections=hand_edges)), static=True
        )
        recording.log(path, rr.Points3D.from_fields(keypoint_ids=np.arange(42), class_ids=0, colors=color, radii=0.003), static=True)
        recording.send_columns(path, indexes=index, columns=rr.Points3D.columns(positions=points.reshape(-1, 3)).partition([42] * len(data.times)))
    for stamp, pose in zip(data.times, result.poses, strict=True):
        recording.set_time("video_time", duration=float(stamp))
        recording.log("/parameters/poses", rr.Tensor(pose))
    recording.send_columns("/parameters/root", indexes=index, columns=rr.Points3D.columns(positions=result.translations))
    recording.log("/parameters/identity", rr.Tensor(result.identity), static=True)
    recording.log("/parameters/head_attachment", rr.Tensor(result.head_attachment), static=True)
    recording.log("/parameters/head_anchor", rr.TextDocument(result.head_anchor), static=True)
    recording.log("/parameters/head_offset", rr.Tensor(result.head_offset), static=True)
    recording.log("/parameters/bone_scales", rr.Tensor(result.bone_scales), static=True)
    recording.log("/parameters/joint_names", rr.TextDocument("\n".join(result.names)), static=True)
    recording.log("/parameters/scale", rr.Scalars(result.scale), static=True)
    mesh_rows = np.unique(np.r_[np.arange(0, len(data.times), mesh_stride), len(data.times) - 1])
    mesh_vertices = result.vertices[mesh_rows]
    normals = np.stack([trimesh.Trimesh(v, result.faces, process=False).vertex_normals for v in mesh_vertices]).astype(np.float32)
    recording.send_columns(
        "/world/fit/mesh",
        indexes=[rr.TimeColumn("video_time", duration=data.times[mesh_rows])],
        columns=rr.Mesh3D.columns(vertex_positions=mesh_vertices.reshape(-1, 3), vertex_normals=normals.reshape(-1, 3)).partition(
            [result.vertices.shape[1]] * len(mesh_rows)
        ),
    )
    recording.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin="/world",
                    name="Body and observations",
                    contents=["/world/fit/**", "/world/inputs/**"],
                    eye_controls=rrb.EyeControls3D(position=[1.3, 1.7, 2.2], look_target=[-0.35, 1.15, 0.4], eye_up=[0.0, 1.0, 0.0]),
                ),
                rrb.Spatial3DView(origin="/world/detail", name="Hands: cyan fit / orange exo / magenta Quest"),
            ),
            collapse_panels=True,
        )
    )
    recording.flush()
    recording.disconnect()
    return report
