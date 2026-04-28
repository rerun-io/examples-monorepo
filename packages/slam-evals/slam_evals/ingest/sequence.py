"""Orchestrate ingestion of one VSLAM-LAB sequence into per-stream layer ``.rrd`` files.

Each sequence becomes a directory ``data/slam-evals/rrd/<dataset>/<sequence>/``
with one ``.rrd`` file per source data stream. All layer files share
``recording_id`` + ``application_id`` so the catalog server composes them
into a single segment at query/view time. See ``docs/schema.md`` for the
full schema description, the entity tree, and the per-layer property bags.

Layer count by modality: 3 (mono) → 7 (stereo-rgbd-vi). Driven off
``Sequence.has_camera/has_depth/has_imu`` (disk presence checks) so adding
a new sensor type later is just a new helper + a new ``write_*_layer``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal, get_args

from slam_evals.data.parse import GroundTruth, ImuSamples, parse_calibration, parse_groundtruth, parse_imu, parse_rgb_csv
from slam_evals.data.types import Sequence
from slam_evals.ingest.compact import compact_rrd
from slam_evals.ingest.layer_calibration import write_calibration_layer
from slam_evals.ingest.layer_depth import write_depth_layer
from slam_evals.ingest.layer_groundtruth import write_groundtruth_layer
from slam_evals.ingest.layer_imu import write_imu_layer
from slam_evals.ingest.layer_video import write_video_layer
from slam_evals.ingest.layer_view_coordinates import write_view_coordinates_layer

# Layer names follow source-stream names (matches docs/schema.md). Order
# determines write order — calibration first so any layer-registration
# smoke test sees property:info:* on the first file processed. The
# ``LayerName`` Literal is the source of truth so tyro CLIs can show
# the choices in --help; ``_ALL_LAYERS`` derives from it for runtime
# iteration.
LayerName = Literal[
    "calibration",
    "groundtruth",
    "view_coordinates",
    "video_0",
    "video_1",
    "depth_0",
    "depth_1",
    "imu_0",
]
_ALL_LAYERS: tuple[LayerName, ...] = get_args(LayerName)


def applicable_layers(sequence: Sequence) -> tuple[str, ...]:
    """Layer names that have on-disk source data for ``sequence``.

    A subset of ``_ALL_LAYERS``. ``calibration`` and ``groundtruth`` are
    unconditional (calibration falls back to a no-camera no-property file
    when absent; groundtruth falls back to identity for empty CSVs).
    ``view_coordinates`` is gated on the dataset having a published
    convention — see ``slam_evals.data.datasets``.
    """
    layers: list[str] = ["calibration", "groundtruth", "video_0"]
    spec = sequence.dataset_spec
    if spec is not None and spec.world_view_coordinates is not None:
        layers.append("view_coordinates")
    if sequence.has_camera(1):
        layers.append("video_1")
    if sequence.has_depth(0):
        layers.append("depth_0")
    if sequence.has_depth(1):
        layers.append("depth_1")
    if sequence.has_imu(0):
        layers.append("imu_0")
    return tuple(layers)


def ingest_sequence(
    sequence: Sequence,
    out_dir: Path,
    *,
    layers: set[str] | None = None,
    application_id: str = "slam-evals",
) -> tuple[Path, ...]:
    """Ingest ``sequence`` into a directory of layer ``.rrd`` files.

    Parameters
    ----------
    sequence:
        The discovered sequence.
    out_dir:
        Output directory. Layer files land at
        ``<out_dir>/<dataset>/<name>/<layer_name>.rrd``.
    layers:
        Optional subset of layer names to (re-)emit. Defaults to all
        applicable layers for the sequence's modality. Use this for
        selective re-ingestion (e.g. ``{"video_0"}`` to only re-encode
        camera 0 video after an NVENC flake).
    application_id:
        Application id stamped onto every layer's RecordingStream — must
        match the catalog mount's expected app id, default ``"slam-evals"``.

    Returns
    -------
    tuple of paths to the layer files actually written this call.
    """
    seq_dir = out_dir / sequence.dataset / sequence.name
    seq_dir.mkdir(parents=True, exist_ok=True)

    applicable = set(applicable_layers(sequence))
    selected = applicable if layers is None else applicable & set(layers)
    if not selected:
        return ()

    # Parse only what's needed for the selected layers. rgb.csv carries
    # timestamps + paths for video_<i> and depth_<i> (the source dir
    # naming is upstream VSLAM-LAB's ``rgb_<i>``; the layer name on our
    # side is ``video_<i>``), so it's needed by most layers; the t0
    # epoch comes from rgb.csv too.
    rgb_csv = parse_rgb_csv(sequence.root / "rgb.csv")
    t0_ns = int(rgb_csv.ts_rgb_0_ns[0])

    calibration = parse_calibration(sequence.root / "calibration.yaml") if sequence.has_calibration else None

    # Lazy parse: only read GT / IMU when those layers are selected.
    def _gt_loader() -> GroundTruth:
        return parse_groundtruth(sequence.root / "groundtruth.csv")

    def _imu_loader() -> ImuSamples:
        return parse_imu(sequence.root / "imu_0.csv")

    written: list[Path] = []
    todo: list[tuple[str, Callable[[], Path]]] = [
        ("calibration", lambda: write_calibration_layer(
            sequence,
            calibration,
            out_path=seq_dir / "calibration.rrd",
            application_id=application_id,
        )),
        ("groundtruth", lambda: write_groundtruth_layer(
            sequence,
            groundtruth=_gt_loader(),
            t0_ns=t0_ns,
            out_path=seq_dir / "groundtruth.rrd",
            application_id=application_id,
        )),
        ("view_coordinates", lambda: write_view_coordinates_layer(
            sequence,
            out_path=seq_dir / "view_coordinates.rrd",
            application_id=application_id,
        )),
        ("video_0", lambda: write_video_layer(
            sequence,
            cam_idx=0,
            rgb_csv=rgb_csv,
            out_path=seq_dir / "video_0.rrd",
            application_id=application_id,
        )),
        ("video_1", lambda: write_video_layer(
            sequence,
            cam_idx=1,
            rgb_csv=rgb_csv,
            out_path=seq_dir / "video_1.rrd",
            application_id=application_id,
        )),
        ("depth_0", lambda: write_depth_layer(
            sequence,
            cam_idx=0,
            rgb_csv=rgb_csv,
            calibration=calibration,
            t0_ns=t0_ns,
            out_path=seq_dir / "depth_0.rrd",
            application_id=application_id,
        )),
        ("depth_1", lambda: write_depth_layer(
            sequence,
            cam_idx=1,
            rgb_csv=rgb_csv,
            calibration=calibration,
            t0_ns=t0_ns,
            out_path=seq_dir / "depth_1.rrd",
            application_id=application_id,
        )),
        ("imu_0", lambda: write_imu_layer(
            sequence,
            imu_idx=0,
            imu=_imu_loader(),
            calibration=calibration,
            t0_ns=t0_ns,
            out_path=seq_dir / "imu_0.rrd",
            application_id=application_id,
        )),
    ]

    for name, fn in todo:
        if name in selected:
            path = fn()
            # Compact each layer file in place after the writer's
            # ``rec.save(...)`` returns. The catalog's mount-time parse
            # cost scales with chunk count, and a freshly-written .rrd
            # carries one chunk per send_columns / log call. Compacting
            # merges them so subsequent reads are faster — see
            # ``slam_evals.ingest.compact``.
            compact_rrd(path)
            written.append(path)

    return tuple(written)
