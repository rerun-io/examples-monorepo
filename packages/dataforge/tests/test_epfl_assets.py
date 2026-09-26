"""Driver-run EPFL asset and reference gates; raw trees are read-only."""

import csv
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import rerun.chunk as rrc
from conftest import read_chunks
from jaxtyping import Bool, Float32, Int64

from dataforge.datasets.epfl import EpflConfig
from dataforge.datasets.epfl_source import FITS, PoseRow, array_cell, parse_pose_row, read_timestamps

KEY = "train/YH2007/2023_10_30_10_05_27"
POSE_ROOT: Path = Path(os.environ.get("DATAFORGE_EPFL_POSE_ROOT", "/mnt/nas/datasets/epfl-smart-kitchen"))
VIDEO_ROOT: Path = Path(os.environ.get("DATAFORGE_EPFL_VIDEO_ROOT", "/home/pablo/exoego-data/epfl/raw"))
PARITY_ROOT: Path = Path(os.environ.get("DATAFORGE_EPFL_PARITY_ROOT", "/home/pablo/exoego-data/epfl/simplecv_root"))
SMPL_ROOT: Path = Path(
    os.environ.get("DATAFORGE_SMPL_MODEL_ROOT", "/home/pablo/0Dev/work/rerun-projects/examples-monorepo/packages/lamp/data/body_models")
)


def require(path: Path) -> Path:
    """Skip with the exact missing asset, never silently replace it."""
    if not path.exists():
        pytest.skip(f"EPFL asset absent: {path}")
    return path


@pytest.mark.integration
def test_real_session_all_layers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = EpflConfig(root=POSE_ROOT, video_root=VIDEO_ROOT, sequences=(KEY,), frame_limit=60, smpl_model_root=SMPL_ROOT)
    pose = POSE_ROOT / "Public_release_pose" / KEY
    video = VIDEO_ROOT / "Public_release_videos" / KEY
    for model in ("mano", "smpl"):
        require(pose / "pose_3d" / f"pose3d_{model}.csv")
    from dataforge.datasets.epfl_source import CAMERA_NAMES

    for name in CAMERA_NAMES:
        require(video / "videos" / f"{name}.mp4")
    for name in ("camera_matrix.json", "timestamps.txt", "holo_data_wpose.csv"):
        require(video / "meta_data" / name)
    for name in ("actions_annotations.xlsx", "activity_annotations.json"):
        require(pose / "annotations" / name)
    require(config.smpl_model_root / "smpl/SMPL_NEUTRAL.pkl")
    import simplecv

    for side in ("LEFT", "RIGHT"):
        require(Path(simplecv.__file__).parent / "data" / f"MANO_{side}.pkl")
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path))
    dataset = config.setup()
    identity, source = dataset.discover()[0]
    dataset.convert(identity, source, force=True)
    targets = dataset.targets(identity)
    assert set(targets) == {"base", "hand_pose", "body_pose", "hand_mesh", "body_mesh", "actions", "projections"}
    for path in targets.values():
        assert path.stat().st_size > 0
    base = read_chunks(targets["base"])
    samples = [chunk for chunk in base if "VideoStream:sample" in chunk.to_record_batch().schema.names]
    assert sum(chunk.num_rows for chunk in samples) == 600
    joints = [chunk for chunk in read_chunks(targets["hand_pose"]) if str(chunk.entity_path) == "/world/gt/coco133_xyz" and not chunk.is_static]
    assert sum(chunk.num_rows for chunk in joints) == 60
    assert not any(str(chunk.entity_path) == "/world/gt/coco133_xyz" for chunk in read_chunks(targets["body_pose"]))
    # body_mesh is 10 Hz (BODY_MESH_STRIDE = 3): 20 of the 60 frames; hand_mesh stays at full rate.
    body = [chunk for chunk in read_chunks(targets["body_mesh"]) if str(chunk.entity_path) == "/world/gt/body/mesh" and not chunk.is_static]
    assert sum(chunk.num_rows for chunk in body) == 20
    left = [chunk for chunk in read_chunks(targets["hand_mesh"]) if str(chunk.entity_path) == "/world/gt/hands/left/mesh" and not chunk.is_static]
    assert sum(chunk.num_rows for chunk in left) == 60


@pytest.mark.golden
def test_simplecv_parity_on_matching_source_rows() -> None:
    reference_root: Path = Path(os.environ.get("DATAFORGE_EPFL_REFERENCE_ROOT", "/mnt/nas/datasets/exoego-forge-catalog-rig"))
    reference: Path = reference_root / "epfl-smart-kitchen" / f"{KEY}.rrd"
    try:
        with reference.open("rb") as handle:
            handle.read(1)
    except FileNotFoundError:
        pytest.skip(f"EPFL simplecv reference asset absent: {reference}")
    except OSError as error:
        pytest.skip(f"EPFL simplecv reference asset unreadable: {reference}: {error}")
    root: Path = POSE_ROOT / "Public_release_pose" / KEY / "pose_3d"
    mano_path: Path = require(root / "pose3d_mano.csv")
    smpl_path: Path = require(root / "pose3d_smpl.csv")
    times: Int64[np.ndarray, "n"] = read_timestamps(require(VIDEO_ROOT / "Public_release_videos" / KEY / "meta_data/timestamps.txt"))
    matched: int = 0
    static_rows: int = 0
    common_joints: int = 0
    empty_residual_joints: int = 0
    zero_confidence_joints: int = 0
    max_position_error: float = 0.0
    max_confidence_error: float = 0.0
    first_frame: int | None = None
    # Keep only the current Arrow chunk and one decoded CSV pair in memory.
    with mano_path.open(newline="") as mano_handle, smpl_path.open(newline="") as smpl_handle:
        mano_rows = csv.DictReader(mano_handle)
        smpl_rows = csv.DictReader(smpl_handle)
        for chunk in rrc.RrdReader(reference).stream():
            if str(chunk.entity_path) != "/world/gt/coco133_xyz":
                continue
            if chunk.is_static:
                static_rows += chunk.num_rows
                continue
            batch: pa.RecordBatch = chunk.to_record_batch()
            stamps: Int64[np.ndarray, "n"] = batch.column("video_time").cast(pa.int64()).to_numpy()
            for index, stamp in enumerate(stamps):
                assert matched < len(times), "extra temporal reference row"
                assert int(stamp) + int(times[0]) == int(times[matched]), f"reference clock mismatch at frame {matched}"
                mano: dict[str, str] = next(mano_rows)
                smpl: dict[str, str] = next(smpl_rows)
                actual: PoseRow = parse_pose_row(mano, smpl)
                if first_frame is None:
                    first_frame = actual.rgb_frameid
                assert actual.rgb_frameid == first_frame + matched
                expected: Float32[np.ndarray, "133 3"] = np.asarray(batch.column("Points3D:positions")[index].as_py(), dtype=np.float32)
                confidence: Float32[np.ndarray, "133"] = np.asarray(
                    batch.column("simplecv.KeypointConfidence3D:confidences")[index].as_py(), dtype=np.float32
                )
                assert expected.shape == (133, 3) and confidence.shape == (133,)
                present: Bool[np.ndarray, "133"] = np.isfinite(actual.positions).all(axis=1)
                reference_present: Bool[np.ndarray, "133"] = np.isfinite(expected).all(axis=1)
                common: Bool[np.ndarray, "133"] = present & reference_present
                np.testing.assert_allclose(actual.positions[common], expected[common], atol=1e-4, rtol=0)
                np.testing.assert_allclose(actual.confidence[common], confidence[common], atol=1e-6, rtol=0)
                common_joints += int(common.sum())
                if common.any():
                    max_position_error = max(max_position_error, float(np.abs(actual.positions[common] - expected[common]).max()))
                    max_confidence_error = max(max_confidence_error, float(np.abs(actual.confidence[common] - confidence[common]).max()))
                empty_residual: Bool[np.ndarray, "133"] = np.zeros(133, dtype=np.bool_)
                shipped_zero: Bool[np.ndarray, "133"] = np.zeros(133, dtype=np.bool_)
                sources = {"pose3d_mano.csv": mano, "pose3d_smpl.csv": smpl}
                for spec in FITS:
                    raw = sources[spec.file]
                    empty_residual[spec.coco] = not raw[spec.residual].strip()
                    shipped_zero[spec.coco] = np.asarray(array_cell(raw["kp3ds_conf"]), dtype=np.float32)[spec.source] == 0.0
                reference_only: Bool[np.ndarray, "133"] = reference_present & ~present
                ours_only: Bool[np.ndarray, "133"] = present & ~reference_present
                assert not np.any(reference_only & ~empty_residual), f"unexplained reference-only joints at frame {matched}"
                assert not np.any(ours_only & ~shipped_zero), f"unexplained dataforge-only joints at frame {matched}"
                np.testing.assert_array_equal(actual.confidence[reference_only], 0.0)
                np.testing.assert_array_equal(confidence[ours_only], 0.0)
                empty_residual_joints += int(reference_only.sum())
                zero_confidence_joints += int(ours_only.sum())
                matched += 1
        assert next(mano_rows, None) is None and next(smpl_rows, None) is None, "unmatched source rows"
    assert matched == len(times) == 67_890
    assert static_rows == 1  # The extra reference row is static keypoint metadata, not a pose.
    assert common_joints > 0
    print(
        f"{KEY}: matched={matched}, static reference rows={static_rows}, common finite joints={common_joints}, "
        f"empty-residual differences={empty_residual_joints}, shipped-zero-confidence differences={zero_confidence_joints}, "
        f"unexplained=0, max position error={max_position_error:.9g} m, max confidence error={max_confidence_error:.9g}"
    )


@pytest.mark.golden
def test_mano_fk_matches_shipped_joints_band() -> None:
    import simplecv

    from dataforge.datasets.epfl_mesh import MeshWriter, corrected_translation
    from dataforge.datasets.epfl_source import pose_batches

    root = PARITY_ROOT / "Public_release_pose" / KEY / "pose_3d"
    for model in ("mano", "smpl"):
        require(root / f"pose3d_{model}.csv")
    for side in ("LEFT", "RIGHT"):
        require(Path(simplecv.__file__).parent / "data" / f"MANO_{side}.pkl")
    require(SMPL_ROOT / "smpl/SMPL_NEUTRAL.pkl")
    rows = next(pose_batches(root, 60, total=67_890))
    writer = MeshWriter(SMPL_ROOT)
    distances = []
    for spec in FITS:
        if spec.name == "body":
            continue
        for row in rows:
            fit = getattr(row, spec.name)
            if not fit.accepted:
                continue
            model = writer.hand_model(spec.name, fit.parameters.shapes)
            pose = fit.parameters.poses[None].copy()
            pose[:, :3] = fit.parameters.Rh
            joints = model(pose, corrected_translation([fit], model.root_trans.reshape(3)))[1][0]
            distances.extend(np.linalg.norm(joints - row.positions[spec.coco], axis=1).tolist())
    assert distances
    assert np.median(distances) <= 0.003
    assert max(distances) <= 0.01
