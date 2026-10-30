"""Resumable per-segment SOMA fitting and soma-only catalog publication."""

import fcntl
import hashlib
import re
import shlex
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
import pyarrow as pa
from datafusion import col, lit
from jaxtyping import Float64, Int64
from numpy import ndarray
from rerun.catalog import CatalogClient, OnDuplicateSegmentLayer
from rerun.chunk import RrdReader
from serde import serde
from serde.json import from_json, to_json

from selfcap_soma.catalog import ExoSource, read_window
from selfcap_soma.fitting import Calibration, FitOptions, fit
from selfcap_soma.hand_pose import read_orientations
from selfcap_soma.output import save
from selfcap_soma.overlay import Config as OverlayConfig
from selfcap_soma.overlay import main as export
from selfcap_soma.shape_frames import select_shape_frames
from selfcap_soma.shoulders import subset
from selfcap_soma.temporal import group_jitter

VERSION = "soma-batch-v5-exact-time-full-mesh"
SEGMENT = re.compile(r"selfcap__(\d{4}-\d{2}-\d{2})__([a-f0-9]{8})__ep(\d{3})")


@dataclass(frozen=True, slots=True)
class Config:
    dataset: str
    """Explicit catalog dataset. Only its soma layer is written."""
    spark_dir: Path
    """Absolute destination directory on Spark for the soma RRDs."""
    segment: list[str] = field(default_factory=list)
    """One or more SelfCap segment IDs."""
    segments_file: Path | None = None
    """Optional newline-separated IDs; blank/comment lines are ignored."""
    exo_source: ExoSource = "rtmw"
    """Triangulation and per-view evidence source."""
    catalog_url: str = "rerun+http://dgx-spark:51235"
    """Catalog holding the source layers and destination dataset."""
    spark_host: str = "spark"
    """SSH host or configured alias for the catalog's file host."""
    output: Path = Path("/tmp/soma-batch")
    """Persistent local checkpoints, one directory per dataset/segment."""
    raw_root: Path = Path("/mnt/nas/datasets/exoego-self-collected/main/cut")
    """Raw SelfCap cut tree, needed only for missing catalog hand rotations."""
    assets: Path = Path("/tmp/soma-fit/assets")
    """Downloaded SOMA model files."""
    iterations: int = 600
    """Full-rate pose iterations after freezing shape."""
    body_acceleration_weight: float = 1.0
    """Body-chain acceleration weight; excludes elbows, wrists and fingers."""
    torso_temporal_weight: float = 1.0
    """Root/chest rotation-velocity-change weight."""
    final_learning_rate: float = 0.001
    """Adam learning rate at the end of the pose pass."""
    shape_iterations: int = 300
    """Shape-update iterations within the 600-step sparse calibration pass."""
    publish: bool = True
    """Ship and register soma; disable for local validation runs."""
    cleanup_after_publish: bool = False
    """After verified catalog readback, delete only this segment's known heavy local RRDs."""
    force: bool = False
    """Ignore matching local checkpoints and refit."""


@serde(deny_unknown_fields=True)
@dataclass(slots=True)
class Timings:
    catalog_s: float = 0.0
    """Catalog metadata and observation materialization."""
    shape_s: float = 0.0
    """Sparse shape fitting and its model/mesh work."""
    fit_s: float = 0.0
    """Full-rate pose fitting, including mesh evaluation."""
    write_s: float = 0.0
    """Fit RRD and report serialization."""
    mesh_overlay_s: float = 0.0
    """Final mesh normals, projections and layer writing."""
    ship_s: float = 0.0
    """SSH/SCP file transfer."""
    register_s: float = 0.0
    """Catalog soma-layer registration."""
    total_s: float = 0.0
    """Wall time for the most recent invocation, including resumed work."""
    peak_gpu_mib: float = 0.0
    """Maximum allocated CUDA memory across shape and pose stages."""


@serde(deny_unknown_fields=True)
@dataclass(slots=True)
class Checkpoint:
    signature: str
    """Digest of algorithm/settings, input streams, calibration and raw rotations."""
    phase: Literal["new", "shape", "fit", "export", "published"]
    """Last completely written stage; incomplete stages are rerun."""
    timings: Timings
    """Per-stage measurements."""
    layer_sha256: str = ""
    """Integrity check for a completed local layer."""
    layer_bytes: int = 0
    """Completed local layer size."""
    mesh_sample_sha256: str = ""
    """First/last mesh samples retained as a readback integrity check."""
    readback_verified: bool = False
    """Catalog URI, remote digest, mesh times and endpoint geometry were checked."""
    cleaned: bool = False
    """Heavy local artifacts may be absent; resume must verify the published layer."""


def selected_segments(values: list[str], path: Path | None) -> list[str]:
    """Validate explicit IDs before constructing local or remote paths."""
    if path is not None:
        values = [*values, *(line.strip() for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#"))]
    result = list(dict.fromkeys(values))
    if not result or any(SEGMENT.fullmatch(value) is None for value in result):
        raise ValueError("provide at least one valid selfcap__DATE__HASH__epNNN segment")
    return result


def resolve_quest_directory(root: Path, segment: str) -> Path:
    """Resolve the unique raw session; rotations are not carried by the catalog."""
    match = SEGMENT.fullmatch(segment)
    if match is None:
        raise ValueError(f"invalid segment: {segment}")
    date, prefix, episode = match.groups()
    matches = list((root / date).glob(f"{prefix}*/episodes/episode-{episode}/quest"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{segment}: expected one raw Quest directory under {root}, found {len(matches)}")
    for side in ("left", "right"):
        path = matches[0] / f"{side}_hand_poses.csv"
        if not path.is_file():
            raise FileNotFoundError(f"{segment}: missing orientation input {path}")
    return matches[0]


def write_checkpoint(path: Path, state: Checkpoint) -> None:
    """Atomically advance a checkpoint after its artifact is complete."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(to_json(state))
    temporary.replace(path)


def run_segment(config: Config, segment: str) -> None:
    """Fit one segment under an exclusive local lock; raise on missing inputs."""
    if config.dataset in ("selfcap-post-ep1", "dataforge-selfcap"):
        raise ValueError(f"publication to protected dataset {config.dataset!r} is disabled")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", config.dataset) or not config.spark_dir.is_absolute():
        raise ValueError("dataset must be a simple name and spark-dir must be absolute")
    directory = config.output / config.dataset / segment
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _run_locked(config, segment, directory)


def _run_locked(config: Config, segment: str, directory: Path) -> None:
    started = perf_counter()
    dataset = CatalogClient(config.catalog_url).get_dataset(config.dataset)
    metadata = dataset.filter_segments(segment).segment_table().to_arrow_table()
    if len(metadata) != 1:
        raise ValueError(f"{segment}: segment is absent or ambiguous in {config.dataset}")
    end = metadata["video_time:end"][0].as_py().total_seconds()
    data = read_window(config.catalog_url, config.dataset, segment, 0.0, float(end + 0.1), 1, config.exo_source)
    raw = resolve_quest_directory(config.raw_root, segment)
    digest = hashlib.sha256()
    digest.update(
        f"{VERSION}:{config.exo_source}:{config.iterations}:{config.shape_iterations}:{config.body_acceleration_weight}:{config.torso_temporal_weight}:{config.final_learning_rate}".encode()
    )
    for array in (data.times, data.exo, data.confidence, data.hands, data.head_rotations):
        digest.update(array.tobytes())
    if data.support is not None:
        digest.update(data.support.tobytes())
    # Camera/calibration changes invalidate overlays even if 3D inputs stay unchanged.
    calib = [name for name in metadata.column_names if name.startswith("property:calib:")]
    digest.update(metadata.select(calib).combine_chunks().to_batches()[0].serialize().to_pybytes())
    for side in ("left", "right"):
        digest.update((raw / f"{side}_hand_poses.csv").read_bytes())
    signature = digest.hexdigest()
    state_path = directory / "checkpoint.json"
    state = from_json(Checkpoint, state_path.read_text()) if state_path.exists() else Checkpoint(signature, "new", Timings())
    if config.force or state.signature != signature:
        state = Checkpoint(signature, "new", Timings())
    state.timings.catalog_s = perf_counter() - started
    layer = directory / f"{segment}.rrd"
    if state.phase == "published" and state.cleaned:
        verify_publication(config, segment, state, data.times)
        if config.cleanup_after_publish and config.publish:
            cleanup_published(config, directory, segment, state)
        state.timings.total_s = perf_counter() - started
        write_checkpoint(state_path, state)
        print(segment, to_json(state), flush=True)
        return
    if state.phase in ("export", "published") and (not layer.is_file() or hashlib.sha256(layer.read_bytes()).hexdigest() != state.layer_sha256):
        raise ValueError(f"{segment}: completed layer is missing/corrupt; rerun with --force")
    options = FitOptions(
        head_mode="ears-rotation",
        head_rotation_weight=0.3,
        learn_head_rotation=True,
        shape_components=128,
        identity_prior=0.05,
        scale_prior=2.0,
        learn_bone_scales=True,
        bone_length_weight=2.0,
        limb_symmetry_weight=10.0,
        display_twists=True,
    )
    if state.phase == "new":
        began = perf_counter()
        selection = select_shape_frames(data.times, data.exo, data.confidence, data.support)
        (directory / "shape-frames.json").write_text(to_json(selection))
        print(f"shape frames: {len(selection.indices)}, whole-segment fallback: {selection.used_fallback}", flush=True)
        mask = np.zeros(len(data.times), dtype=bool)
        mask[selection.indices] = True
        shape = fit(subset(data, mask), config.assets, 600, config.shape_iterations, options)
        calibration = Calibration(shape.identity, shape.scale, shape.head_offset, shape.bone_scales, shape.head_attachment, shape.head_anchor)
        (directory / "calibration.json").write_text(to_json(calibration))
        state.timings.shape_s = perf_counter() - began
        state.timings.peak_gpu_mib = shape.peak_bytes / 2**20
        state.phase = "shape"
        write_checkpoint(state_path, state)
    if state.phase == "shape":
        began = perf_counter()
        calibration = from_json(Calibration, (directory / "calibration.json").read_text())
        orientations = read_orientations(raw, data.times, data.offset_ms)
        result = fit(
            data,
            config.assets,
            config.iterations,
            1,
            replace(
                options,
                learn_head_rotation=False,
                freeze_legs=True,
                leg_support_weight=0.0,
                hip_landmark_weight=0.15,
                torso_temporal_weight=config.torso_temporal_weight,
                body_acceleration_weight=config.body_acceleration_weight,
                final_learning_rate=config.final_learning_rate,
                orientation_weight=30.0,
            ),
            fixed=calibration,
            orientations=orientations,
        )
        state.timings.fit_s = perf_counter() - began
        state.timings.peak_gpu_mib = max(state.timings.peak_gpu_mib, result.peak_bytes / 2**20)
        began = perf_counter()
        save(directory, data, result, 39.862079571, 1, upper_body_only=True)
        (directory / "jitter.json").write_text(to_json(group_jitter(result.joints, data.times, result.names)))
        state.timings.write_s = perf_counter() - began
        state.phase = "fit"
        write_checkpoint(state_path, state)
    if state.phase == "fit":
        began = perf_counter()
        temporary = layer.with_suffix(".pending.rrd")
        export(
            OverlayConfig(
                source=directory / "fit.rrd",
                output=temporary,
                catalog_url=config.catalog_url,
                dataset=config.dataset,
                segment=segment,
                exo_source=config.exo_source,
            )
        )
        temporary.replace(layer)
        state.timings.mesh_overlay_s = perf_counter() - began
        state.layer_sha256 = hashlib.sha256(layer.read_bytes()).hexdigest()
        state.layer_bytes = layer.stat().st_size
        state.phase = "export"
        write_checkpoint(state_path, state)
    if config.publish:
        state.timings.ship_s, state.timings.register_s = publish_layer(config, layer)
        state.phase = "published"
        if config.cleanup_after_publish:
            endpoints = np.rint(data.times[[0, -1]] * 1e9).astype(np.int64)
            state.mesh_sample_sha256 = local_mesh_digest(layer, endpoints)
            verify_publication(config, segment, state, data.times)
            cleanup_published(config, directory, segment, state)
    state.timings.total_s = perf_counter() - started
    write_checkpoint(state_path, state)
    print(segment, to_json(state), flush=True)


def publish_layer(config: Config, layer: Path) -> tuple[float, float]:
    """Atomically ship one RRD and register only the soma layer with REPLACE."""
    if config.dataset in ("selfcap-post-ep1", "dataforge-selfcap"):
        raise ValueError(f"publication to protected dataset {config.dataset!r} is disabled")
    remote = config.spark_dir / layer.name
    began = perf_counter()
    subprocess.run(["ssh", config.spark_host, f"mkdir -p -- {shlex.quote(str(config.spark_dir))}"], check=True)
    subprocess.run(["scp", str(layer), f"{config.spark_host}:{shlex.quote(str(remote) + '.upload')}"], check=True)
    subprocess.run(["ssh", config.spark_host, f"mv -- {shlex.quote(str(remote) + '.upload')} {shlex.quote(str(remote))}"], check=True)
    ship_s = perf_counter() - began
    began = perf_counter()
    client = CatalogClient(config.catalog_url)
    client.get_dataset(config.dataset).register([remote.as_uri()], layer_name="soma", on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
    return ship_s, perf_counter() - began


def local_mesh_digest(path: Path, endpoints: Int64[ndarray, "2"]) -> str:
    """Hash the two endpoint mesh samples before local cleanup."""
    rows = []
    component = "Mesh3D:vertex_positions"
    for chunk in RrdReader(path).stream().filter(content="/world/pred/soma_x/mesh").to_chunks():
        batch = chunk.to_record_batch()
        if "video_time" not in batch.schema.names or component not in batch.schema.names:
            continue
        stamps = batch.column("video_time").cast("int64").to_numpy()
        for index in np.flatnonzero(np.isin(stamps, endpoints)):
            rows.append((int(stamps[index]), np.asarray(batch.column(component)[int(index)].as_py(), dtype=np.float32)))
    if len(rows) != 2:
        raise ValueError("local layer must contain both endpoint mesh samples before cleanup")
    digest = hashlib.sha256()
    for stamp, points in sorted(rows, key=lambda row: row[0]):
        digest.update(np.asarray([stamp], dtype="<i8").tobytes())
        digest.update(points.astype("<f4").tobytes())
    return digest.hexdigest()


def verify_publication(config: Config, segment: str, state: Checkpoint, times: Float64[ndarray, "n"]) -> None:
    """Verify registration, remote file integrity and actual catalog mesh readback."""
    state.readback_verified = False
    remote = config.spark_dir / f"{segment}.rrd"
    result = subprocess.run(["ssh", config.spark_host, f"sha256sum -- {shlex.quote(str(remote))}"], check=True, capture_output=True, text=True)
    if not result.stdout.split() or result.stdout.split()[0] != state.layer_sha256:
        raise ValueError(f"{segment}: published remote layer digest mismatch; cleanup refused")
    client = CatalogClient(config.catalog_url)
    dataset = client.get_dataset(config.dataset)
    view = dataset.filter_segments(segment)
    metadata = view.segment_table().select("rerun_layer_names", "rerun_storage_urls").to_arrow_table()
    if len(metadata) != 1:
        raise ValueError(f"{segment}: missing catalog registration; cleanup refused")
    registered = dict(zip(metadata["rerun_layer_names"][0].as_py(), metadata["rerun_storage_urls"][0].as_py(), strict=True))
    if registered.get("soma") != remote.as_uri():
        raise ValueError(f"{segment}: catalog soma URI mismatch; cleanup refused")
    component = "/world/pred/soma_x/mesh:Mesh3D:vertex_positions"
    reader = view.filter_contents(["/world/pred/soma_x/mesh"]).reader(index="video_time")
    stamps = reader.filter(col(component).is_not_null()).select("video_time").to_arrow_table()["video_time"].cast(pa.int64()).to_numpy()
    expected = np.rint(times * 1e9).astype(np.int64)
    if not np.array_equal(np.sort(stamps), expected):
        raise ValueError(f"{segment}: catalog mesh timestamps mismatch; cleanup refused")
    samples = (
        reader.filter((col("video_time").cast(pa.int64()) == lit(int(expected[0]))) | (col("video_time").cast(pa.int64()) == lit(int(expected[-1]))))
        .select("video_time", component)
        .to_arrow_table()
        .sort_by("video_time")
    )
    digest = hashlib.sha256()
    if len(samples) != 2:
        raise ValueError(f"{segment}: catalog endpoint mesh samples missing; cleanup refused")
    for index in range(2):
        stamp = int(samples["video_time"][index].cast(pa.int64()).as_py())
        points = np.asarray(samples[component][index].as_py(), dtype=np.float32)
        if not np.isfinite(points).all():
            raise ValueError(f"{segment}: catalog mesh has nonfinite vertices; cleanup refused")
        digest.update(np.asarray([stamp], dtype="<i8").tobytes())
        digest.update(points.astype("<f4").tobytes())
    if not state.mesh_sample_sha256 or digest.hexdigest() != state.mesh_sample_sha256:
        raise ValueError(f"{segment}: catalog mesh sample digest mismatch; cleanup refused")
    state.readback_verified = True


def cleanup_published(config: Config, directory: Path, segment: str, state: Checkpoint) -> None:
    """Delete only known local RRDs after successful publication and readback."""
    if not config.cleanup_after_publish or not config.publish or state.phase != "published" or not state.readback_verified:
        raise ValueError("cleanup requires opt-in and a verified successful publication")
    if directory.resolve().parent != (config.output / config.dataset).resolve() or directory.name != segment:
        raise ValueError("cleanup directory is outside the segment output directory")
    paths = [directory / name for name in ("fit.rrd", f"{segment}.rrd", f"{segment}.pending.rrd")]
    if any(path.is_symlink() for path in paths):
        raise ValueError("cleanup refuses symlink artifacts")
    # Persist first: a crash between unlinks resumes through catalog verification.
    state.cleaned = True
    write_checkpoint(directory / "checkpoint.json", state)
    for path in paths:
        path.unlink(missing_ok=True)


def main(config: Config) -> None:
    """Process explicit segments serially; external workers may run different IDs."""
    for segment in selected_segments(config.segment, config.segments_file):
        try:
            run_segment(config, segment)
        except (ValueError, FileNotFoundError, KeyError, StopIteration, IndexError, subprocess.CalledProcessError, BlockingIOError) as error:
            raise RuntimeError(f"SOMA batch failed for {config.dataset}/{segment}: {error}") from error
