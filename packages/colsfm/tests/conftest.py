"""Shared fixtures and helpers for the colsfm tests.

Everything here is used by more than one test module: the Galileo data paths and
their parsed metadata, the two geometry helpers every synthetic-scene test needs
(`pose_delta`, `project_and_mask`), the reader for the blob's own `Keyframe`
protos, and the synthetic cuSFM-layout run builder that `test_benchmark` and
`test_rerun_log` both score against.

It also owns the `perf` marker: tests whose assertion is a *timing* bound, which
measure the machine as much as the code and have no business failing an ordinary
run. They are collected always and skipped unless somebody asks for them, by
`-m perf` or `COLSFM_PERF_TESTS=1`.

`tests/colsfm` has no `__init__.py` and is therefore on `sys.path`, so the plain
functions below are imported as `from conftest import ...`; only the fixtures go
through pytest's own injection.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pycolmap
import pytest
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray

# colsfm is imported inside the fixtures and helpers, never here: pytest-beartype installs its import hook after this
# file loads, and a module imported first would stay unchecked (ci/tests/test_manifest.py enforces it).
if TYPE_CHECKING:
    from mapping_helpers import SyntheticRig

    from colsfm.config import CusfmConfig
    from colsfm.frames_meta import CameraParams, FramesMeta, KeyframeMeta

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
"""`packages/colsfm` (colsfm's `PACKAGE_DIR`), so data paths resolve regardless of the working directory."""


# ─────────────────────────────────────────────────────────────────────────────
# The `perf` marker
# ─────────────────────────────────────────────────────────────────────────────


PERF_MARKER: Final[str] = "perf"
"""Marker for a test whose assertion is a wall-clock or throughput bound."""

PERF_ENVIRONMENT_VARIABLE: Final[str] = "COLSFM_PERF_TESTS"
"""Set to `1` to run the `perf` tests without naming the marker on the command line."""


def pytest_configure(config: pytest.Config) -> None:
    """Register the `perf` marker so `--strict-markers` and `--markers` know it.

    Args:
        config: pytest's configuration.
    """
    config.addinivalue_line(
        "markers",
        "perf: asserts a timing or throughput bound. Skipped unless selected with "
        f"`-m {PERF_MARKER}` or `{PERF_ENVIRONMENT_VARIABLE}=1`, because the bound "
        "measures the machine as much as the code.",
    )


INTEGRATION_FIXTURES: Final[frozenset[str]] = frozenset({"galileo_input_dir", "galileo_input_meta", "galileo_input", "caspar_enabled"})
"""Fixtures that read the downloaded Galileo sample or run a CASPAR (GPU) solve."""

INTEGRATION_MODULES: Final[frozenset[str]] = frozenset(
    {"test_raco_backend.py", "test_gpu_preprocess.py", "test_mapping_ba_backend.py", "test_gpu.py"}
)
"""Modules that build TensorRT engines, solve on CASPAR, or probe the CUDA device throughout."""


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark the tests that need the GPU or a downloaded asset `integration`, and skip `perf` ones unless asked for.

    The integration marker follows from what a test uses, so a new test lands in the right lane by itself: any
    test that takes a Galileo-sample fixture (`colsfm-download-galileo`) or the CASPAR probe, and every test in the
    modules that drive TensorRT engines, CASPAR or the CUDA device directly.

    Deliberately a skip rather than a deselect: the default run still reports
    that the performance tests exist and were not run, which a `-m "not perf"`
    default in the task definition would hide. It is also what lets the exclusion
    live here instead of in `pixi.toml`.

    Args:
        config: pytest's configuration; its `-m` expression is what "asked for
            them" means.
        items: The collected tests, marked in place.
    """
    for item in items:
        if INTEGRATION_FIXTURES & set(getattr(item, "fixturenames", ())) or item.path.name in INTEGRATION_MODULES:
            item.add_marker(pytest.mark.integration)
    if os.environ.get(PERF_ENVIRONMENT_VARIABLE) == "1":
        return
    if PERF_MARKER in str(config.getoption("-m") or ""):
        return
    skip: pytest.MarkDecorator = pytest.mark.skip(
        reason=f"timing bound; select it with `-m {PERF_MARKER}` or {PERF_ENVIRONMENT_VARIABLE}=1"
    )
    for item in items:
        if item.get_closest_marker(PERF_MARKER) is not None:
            item.add_marker(skip)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def galileo_input_dir() -> Path:
    """The r2b_galileo input directory: `frames_meta.json` plus `ground_truth.txt`."""
    return REPO_ROOT / "data" / "r2b_galileo"


@pytest.fixture(scope="session")
def galileo_input_meta(galileo_input_dir: Path) -> Path:
    """The 226-keyframe input metadata shipped with r2b_galileo."""
    from colsfm.frames_meta import FRAMES_META_NAME
    return galileo_input_dir / FRAMES_META_NAME


@pytest.fixture(scope="session")
def galileo_input(galileo_input_meta: Path) -> FramesMeta:
    """The parsed 226-keyframe Galileo input collection."""
    from colsfm.frames_meta import read_frames_meta
    return read_frames_meta(galileo_input_meta)


@pytest.fixture(scope="session")
def three_samples(galileo_input: FramesMeta) -> FramesMeta:
    """The first three synchronised samples of the Galileo input, ~24 keyframes."""
    keep: list[int] = [keyframe_id for rig_frame in galileo_input.rig_frames()[:3] for keyframe_id in rig_frame.keyframe_ids]
    return galileo_input.filtered(keep)


@pytest.fixture(autouse=True, scope="session")
def _quiet_glog() -> None:
    """Silence COLMAP's own logging so the tests' own numbers stay readable.

    Session scope, and here rather than in each mapping module: `minloglevel` is a
    process-wide setting, so five module-scoped copies were five spellings of one
    effect that the first mapping module already had for the rest of the run.
    """
    pycolmap.logging.minloglevel = 2


@pytest.fixture(scope="module")
def synthetic_rig() -> SyntheticRig:
    """A 20-frame, two-camera rig with 500 points and 0.3 px observation noise.

    `mapping_helpers` is imported inside the body, not at the top of the file:
    it imports `project_and_mask` from here, so a module-level import would be a
    cycle. Module scope, as the four copies this replaces had — building the rig
    costs 8 ms and its database 39 ms, so each module still gets its own.
    """
    from mapping_helpers import build_synthetic_rig

    return build_synthetic_rig()


@pytest.fixture(scope="module")
def synthetic_database(synthetic_rig: SyntheticRig, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The synthetic rig's observations as a COLMAP database."""
    from mapping_helpers import synthetic_matches, write_database

    database_path: Path = tmp_path_factory.mktemp("synthetic") / "database.db"
    write_database(
        database_path, synthetic_rig.frames_meta, synthetic_rig.keypoints_px, synthetic_matches(synthetic_rig)
    )
    return database_path


@pytest.fixture(scope="module")
def isaac_config(repo_root: Path) -> CusfmConfig:
    """The isaac profile: 5 BA rounds, a 25 -> 5 px gate, CAUCHY at sigma 4.

    Here rather than in one test module because all five mapping modules read it,
    and it is the profile the shipped cuSFM runs were produced with.
    """
    from colsfm import RESOURCES_DIR
    from colsfm.config import read_config_directory
    return read_config_directory(RESOURCES_DIR / "configs" / "isaac")


@pytest.fixture(scope="session")
def caspar_enabled() -> bool:
    """Whether this environment's pycolmap can build a CASPAR bundle adjuster at all.

    The `BundleAdjustmentBackend.CASPAR` enum is bound unconditionally, so it proves
    nothing; a pycolmap without `CASPAR_ENABLED` refuses only when the adjuster is
    created. An empty problem is enough to ask, and cheap.
    """
    options: pycolmap.BundleAdjustmentOptions = pycolmap.BundleAdjustmentOptions()
    options.backend = pycolmap.BundleAdjustmentBackend.CASPAR
    try:
        pycolmap.create_default_bundle_adjuster(
            options, pycolmap.BundleAdjustmentConfig(), pycolmap.Reconstruction()
        )
    except ValueError:
        return False
    return True


@pytest.fixture(scope="session")
def pixi_environment_name() -> str:
    """Which pixi environment this run is in, e.g. `colsfm-dev`.

    The three CASPAR environments differ only in the pycolmap they carry, and the
    thing that tells them apart — which camera models CASPAR projects — is exactly
    what the probe under test measures. Asking pixi instead keeps the expectation
    independent of the code that produces it. Empty outside a pixi shell.
    """
    return os.environ.get("PIXI_ENVIRONMENT_NAME", "")


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────


def pose_delta(actual: pycolmap.Rigid3d, expected: pycolmap.Rigid3d) -> tuple[float, float]:
    """Translation and rotation difference between two poses.

    Args:
        actual: Pose under test.
        expected: Reference pose.

    Returns:
        Translation difference in metres and rotation difference in degrees.
    """
    from colsfm.geometry import relative_rotation_degrees
    translation_m: float = float(np.linalg.norm(np.asarray(actual.translation) - np.asarray(expected.translation)))
    return translation_m, relative_rotation_degrees(actual, expected)


def project_and_mask(
    camera: pycolmap.Camera, cam_T_world: pycolmap.Rigid3d, points_xyz: Float64[ndarray, "n_points 3"]
) -> tuple[Float64[ndarray, "n_points 2"], Bool[ndarray, " n_points"]]:
    """Project world points into one image and mark the ones that landed inside it.

    Args:
        camera: The camera's intrinsics.
        cam_T_world: That image's `cam_from_world` pose.
        points_xyz: Float64 world points with shape `[n_points, 3]`, in metres.

    Returns:
        The pixel projections, and a mask that is True where the point is in
        front of the camera, projects finitely and lands inside the image.
    """
    points_in_cam: Float64[ndarray, "n_points 3"] = (
        points_xyz @ np.asarray(cam_T_world.rotation.matrix(), dtype=np.float64).T
        + np.asarray(cam_T_world.translation, dtype=np.float64)
    )
    projected: Float64[ndarray, "n_points 2"] = np.asarray(camera.img_from_cam(points_in_cam), dtype=np.float64)
    visible: Bool[ndarray, " n_points"] = (
        np.isfinite(projected).all(axis=1)
        & (points_in_cam[:, 2] > 0.0)
        & (projected[:, 0] >= 0.0)
        & (projected[:, 0] < camera.width)
        & (projected[:, 1] >= 0.0)
        & (projected[:, 1] < camera.height)
    )
    return projected, visible


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic cuSFM-layout runs
# ─────────────────────────────────────────────────────────────────────────────


SYNTHETIC_ERROR_PX: float = 0.75
"""Pixel offset baked into every synthetic observation, so the recomputed mean is known."""

BLOB_ERROR_PLACEHOLDER: float = 2.0
"""`kpmap_to_colmap` hardcodes this into `points3D.txt`; the synthetic runs mimic it."""

POINT_DEPTH_METERS: float = 4.0
"""Distance in front of a camera the synthetic points are placed at."""

POINTS_PER_CAMERA: int = 12
"""Points generated in front of each camera; enough for tracks of length > 1."""

DEMO_REPROJECTION_PX: float = 1.55
"""What `demo_rerun.py` reported for the Galileo blob reference run."""

DEMO_REGISTERED_IMAGES: int = 224
"""Registered images the plan's "Blob reference runs" table records for Galileo."""

RIGID_TOLERANCE_MM: float = 0.02
"""Rigidity noise floor: axis-angle-in-degrees storage costs ~10 um per pose."""


def transform_rig_trajectory(frames_meta: FramesMeta, *, rig_transform: pycolmap.Rigid3d, scale: float = 1.0) -> FramesMeta:
    """Move every keyframe as if the whole rig trajectory had been transformed.

    Rewrites each `camera_to_world` as
    `new_world_T_vehicle * vehicle_T_cam`, where `new_world_T_vehicle` applies
    `rig_transform` and then scales the position about the world origin. Keeping
    the rig extrinsics untouched means the collection stays perfectly rigid, so
    the rigidity spread is unaffected and only the trajectory metrics move.

    Args:
        frames_meta: Collection to transform.
        rig_transform: Rigid transform applied to every rig pose.
        scale: Multiplier applied to the rig position after the rigid transform.

    Returns:
        A new collection with the transformed poses.
    """
    poses: dict[int, pycolmap.Rigid3d] = {}
    for keyframe in frames_meta.keyframes:
        camera: CameraParams = frames_meta.cameras[keyframe.camera_params_id]
        world_T_vehicle: pycolmap.Rigid3d = frames_meta.world_T_vehicle(keyframe)
        moved: pycolmap.Rigid3d = rig_transform * world_T_vehicle
        scaled: pycolmap.Rigid3d = pycolmap.Rigid3d(moved.rotation, np.asarray(moved.translation, dtype=np.float64) * scale)
        poses[keyframe.keyframe_id] = scaled * camera.vehicle_T_cam
    return frames_meta.with_camera_to_world(poses)


def build_reconstruction(frames_meta: FramesMeta, error_px: float) -> pycolmap.Reconstruction:
    """Build a COLMAP model consistent with a collection, off by a known pixel error.

    One PINHOLE camera per `camera_params_id`, one rig whose reference sensor is
    the lowest camera id, one frame per `synced_sample_id`, and one image per
    keyframe. Points are placed in front of each camera and observed wherever they
    project inside an image; every observation is displaced by exactly `error_px`
    along +x, so the recomputed mean reprojection error is `error_px`.

    Args:
        frames_meta: Poses and calibration to mirror.
        error_px: Displacement applied to every 2D observation.

    Returns:
        A reconstruction with points, tracks and `error` set to the blob's placeholder.
    """
    reconstruction: pycolmap.Reconstruction = pycolmap.Reconstruction()
    camera_ids: list[int] = sorted(frames_meta.cameras)
    for camera_params_id in camera_ids:
        camera_params: CameraParams = frames_meta.cameras[camera_params_id]
        k_matrix: Float64[ndarray, "3 3"] = np.asarray(camera_params.projection_matrix, dtype=np.float64)[:3, :3]
        camera: pycolmap.Camera = pycolmap.Camera.create_from_model_id(
            camera_params_id + 1,
            pycolmap.CameraModelId.PINHOLE,
            float(k_matrix[0, 0]),
            camera_params.image_width,
            camera_params.image_height,
        )
        camera.params = [float(k_matrix[0, 0]), float(k_matrix[1, 1]), float(k_matrix[0, 2]), float(k_matrix[1, 2])]
        reconstruction.add_camera(camera)

    rig: pycolmap.Rig = pycolmap.Rig()
    rig.rig_id = 1
    rig.add_ref_sensor(pycolmap.sensor_t(pycolmap.SensorType.CAMERA, camera_ids[0] + 1))
    reference_vehicle_T_cam: pycolmap.Rigid3d = frames_meta.cameras[camera_ids[0]].vehicle_T_cam
    for camera_params_id in camera_ids[1:]:
        cam_T_reference: pycolmap.Rigid3d = frames_meta.cameras[camera_params_id].vehicle_T_cam.inverse() * reference_vehicle_T_cam
        rig.add_sensor(pycolmap.sensor_t(pycolmap.SensorType.CAMERA, camera_params_id + 1), cam_T_reference)
    reconstruction.add_rig(rig)

    by_id: dict[int, KeyframeMeta] = frames_meta.keyframe_by_id()
    image_ids_by_keyframe: dict[int, int] = {}
    for frame_index, rig_frame in enumerate(frames_meta.rig_frames()):
        frame: pycolmap.Frame = pycolmap.Frame()
        frame.frame_id = frame_index + 1
        frame.rig_id = 1
        # The rig frame is the reference *camera*, not the vehicle, so the rig
        # pose carries the reference camera's extrinsic.
        frame.rig_from_world = (rig_frame.world_T_vehicle * reference_vehicle_T_cam).inverse()
        pending: list[tuple[int, KeyframeMeta]] = []
        for keyframe_id in rig_frame.keyframe_ids:
            keyframe: KeyframeMeta = by_id[keyframe_id]
            image_id: int = len(image_ids_by_keyframe) + 1
            image_ids_by_keyframe[keyframe_id] = image_id
            frame.add_data_id(pycolmap.data_t(pycolmap.sensor_t(pycolmap.SensorType.CAMERA, keyframe.camera_params_id + 1), image_id))
            pending.append((image_id, keyframe))
        reconstruction.add_frame(frame)
        reconstruction.register_frame(frame.frame_id)
        for image_id, keyframe in pending:
            image: pycolmap.Image = pycolmap.Image(name=keyframe.image_name, camera_id=keyframe.camera_params_id + 1, image_id=image_id)
            image.frame_id = frame.frame_id
            reconstruction.add_image(image)

    points_xyz: Float64[ndarray, "n_points 3"] = _points_in_front_of_cameras(frames_meta)
    _observe(reconstruction, points_xyz, error_px)
    for point in reconstruction.points3D.values():
        # kpmap_to_colmap hardcodes ERROR = 2.0; the harness must recompute it.
        point.error = BLOB_ERROR_PLACEHOLDER
    return reconstruction


def _points_in_front_of_cameras(frames_meta: FramesMeta) -> Float64[ndarray, "n_points 3"]:
    """Scatter points a few metres in front of each camera at its first keyframe."""
    generator: np.random.Generator = np.random.default_rng(7)
    seen: set[int] = set()
    points: list[Float64[ndarray, "3"]] = []
    for keyframe in frames_meta.keyframes:
        if keyframe.camera_params_id in seen:
            continue
        seen.add(keyframe.camera_params_id)
        world_T_cam: pycolmap.Rigid3d = keyframe.world_T_cam
        local: Float64[ndarray, "k 3"] = np.column_stack(
            [
                generator.uniform(-1.0, 1.0, POINTS_PER_CAMERA),
                generator.uniform(-0.8, 0.8, POINTS_PER_CAMERA),
                np.full(POINTS_PER_CAMERA, POINT_DEPTH_METERS),
            ]
        )
        points.extend(local @ np.asarray(world_T_cam.rotation.matrix()).T + np.asarray(world_T_cam.translation))
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)


def _observe(reconstruction: pycolmap.Reconstruction, points_xyz: Float64[ndarray, "n_points 3"], error_px: float) -> None:
    """Add every visible projection as an observation, displaced by `error_px` in +x."""
    observations: dict[int, list[tuple[int, int]]] = {index: [] for index in range(len(points_xyz))}
    for image_id in sorted(reconstruction.images):
        image: pycolmap.Image = reconstruction.image(image_id)
        camera: pycolmap.Camera = reconstruction.camera(image.camera_id)
        projected, visible = project_and_mask(camera, image.cam_from_world(), points_xyz)
        indices: Int64[ndarray, "k"] = np.flatnonzero(visible).astype(np.int64)
        image.points2D = pycolmap.Point2DList(
            [pycolmap.Point2D(projected[index] + np.array([error_px, 0.0])) for index in indices]
        )
        for slot, index in enumerate(indices):
            observations[int(index)].append((image_id, slot))

    for index, track_entries in observations.items():
        if len(track_entries) < 2:
            continue
        track: pycolmap.Track = pycolmap.Track()
        for image_id, slot in track_entries:
            track.add_element(image_id, slot)
        point_id: int = reconstruction.add_point3D(points_xyz[index], track, np.array([180, 190, 200], dtype=np.uint8))
        for image_id, slot in track_entries:
            reconstruction.image(image_id).set_point3D_for_point2D(slot, point_id)


def write_synthetic_run(run_dir: Path, frames_meta: FramesMeta, *, error_px: float, runtimes: dict[str, float]) -> Path:
    """Write a complete cuSFM-layout run to disk through the production writers.

    Args:
        run_dir: Destination workspace, the equivalent of `.../cusfm`.
        frames_meta: Poses and calibration for the run.
        error_px: Pixel error to bake into the observations.
        runtimes: `runtime.csv` rows as `command -> seconds`, in insertion order.

    Returns:
        `run_dir`, now holding `sparse/`, `kpmap/keyframes/frames_meta.json` and `runtime.csv`.
    """
    from colsfm.export import RuntimeRecord, append_runtime_record, write_colmap_model, write_optimised_frames_meta
    write_colmap_model(run_dir, build_reconstruction(frames_meta, error_px))
    write_optimised_frames_meta(run_dir, frames_meta, {})
    for command, seconds in runtimes.items():
        append_runtime_record(run_dir, RuntimeRecord(command=command, runtime_seconds=seconds))
    return run_dir


BLOB_RUNTIMES: dict[str, float] = {
    "feature_extractor_main --input_image_directory in": 8.0,
    "generate_bow_vocabulary_main --keyframe_directory kf": 6.0,
    "generate_bow_index_main --keyframe_directory kf": 4.0,
    "generate_association_main --keyframe_dir kf": 3.0,
    "pose_graph_main --keyframe_directory kf": 3.5,
    "feature_matcher_task_builder_main --current_keyframe_directory kf": 2.0,
    "feature_matcher_main --current_keyframe_directory kf": 2.5,
    "keypoints_mapper_main --keyframe_dir kf": 6.0,
    "kpmap_to_colmap --map_dir kpmap": 2.0,
    "extract_pose_from_map_main --output_pose_dir poses": 0.5,
}
"""A blob run's `runtime.csv`, one row per binary; totals 37.5 s."""

COLSFM_RUNTIMES: dict[str, float] = {
    "colsfm.features --input in": 16.0,
    "colsfm.retrieval --keyframes kf": 5.0,
    "colsfm.loop_closure --keyframes kf": 6.0,
    "colsfm.pose_graph --keyframes kf": 7.0,
    "colsfm.pairs --keyframes kf": 4.0,
    "colsfm.matching --keyframes kf": 5.0,
    "colsfm.mapping --keyframes kf": 30.0,
    "colsfm.export --output out": 2.0,
}
"""A colsfm run's `runtime.csv`, one row per stage name; totals 75.0 s (2.0x the blob)."""

COLSFM_PIPELINE_RUNTIMES: dict[str, float] = {
    "keyframe_selection --input in": 1.0,
    "feature_extraction --input in": 15.0,
    "pair_selection --keyframes kf": 4.0,
    "matching --keyframes kf": 5.0,
    "loop_closure --keyframes kf": 6.0,
    "pose_graph --keyframes kf": 7.0,
    "reconstruction --keyframes kf": 30.0,
    "export --output out": 2.0,
}
"""The names and the order `colsfm.pipeline` actually writes — matching precedes
loop closure, so pipeline-order run detection would wrongly split this."""
