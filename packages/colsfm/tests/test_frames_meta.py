"""`frames_meta.json` parsing, round tripping and the derived rig poses."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pycolmap
from google.protobuf import json_format

from colsfm.frames_meta import CameraParams, FramesMeta, KeyframeMeta, parse_message, read_frames_meta, write_frames_meta
from colsfm.geometry import parse_tum_line, rigid3d_from_axis_angle_degrees


def test_reads_every_keyframe_and_camera(galileo_input: FramesMeta) -> None:
    """Galileo is 226 keyframes over 8 PINHOLE cameras and 4 stereo pairs (NOTES.md)."""
    assert len(galileo_input.keyframes) == 226
    assert sorted(galileo_input.cameras) == list(range(8))
    assert len(galileo_input.stereo_pairs) == 4
    assert galileo_input.initial_pose_type == "EGO_MOTION"
    assert {camera.projection_model for camera in galileo_input.cameras.values()} == {"PINHOLE"}
    assert {(camera.image_width, camera.image_height) for camera in galileo_input.cameras.values()} == {(1920, 1200)}


def test_missing_camera_params_id_reads_as_zero(galileo_input: FramesMeta) -> None:
    """NOTES.md gotcha 8: 29 Galileo keyframes omit `camera_params_id`, all camera 0."""
    counts: Counter[int] = Counter(keyframe.camera_params_id for keyframe in galileo_input.keyframes)
    assert counts[0] == 29
    assert galileo_input.cameras[0].sensor_name == "back_stereo_camera_left"
    zero_camera_names: set[str] = {
        keyframe.image_name.split("/")[0] for keyframe in galileo_input.keyframes if keyframe.camera_params_id == 0
    }
    assert zero_camera_names == {"back_stereo_camera_left"}


def test_missing_stereo_pair_left_id_reads_as_zero(galileo_input: FramesMeta) -> None:
    """The same default suppression hits `stereo_pair.left_camera_param_id` (NOTES.md gotcha 8)."""
    left_ids: set[int] = {pair.left_camera_params_id for pair in galileo_input.stereo_pairs}
    assert 0 in left_ids
    assert all(pair.baseline_meters > 0.1 for pair in galileo_input.stereo_pairs)


def test_round_trip_preserves_the_whole_collection(galileo_input_meta: Path, tmp_path: Path) -> None:
    """Reading and re-writing `frames_meta.json` loses nothing.

    Compared as protobuf messages, which is the semantic comparison: it sees
    through JSON map ordering and through proto3 default omission, both of which
    differ textually between cuSFM's writer and protobuf's.
    """
    original: FramesMeta = read_frames_meta(galileo_input_meta)
    destination: Path = tmp_path / "frames_meta.json"
    write_frames_meta(destination, original)
    reloaded: FramesMeta = read_frames_meta(destination)
    assert reloaded.message == original.message
    assert reloaded.to_json() == original.to_json()


def test_round_trip_output_never_contradicts_the_blob_file(galileo_input_meta: Path) -> None:
    """Every value we write matches the blob's own file; we only ever omit defaults.

    Byte equality is unreachable and not wanted: protobuf maps carry no order, so
    cuSFM's hash order and protobuf-python's sorted order both round trip, and
    proto3 drops `camera_projection_model_type: "PINHOLE"` because `PINHOLE` is
    the zero value while cuSFM's writer prints it. Nothing else in the 140 KB
    file moves, which is what this checks.
    """
    original_json: object = json.loads(galileo_input_meta.read_text())
    round_tripped_json: object = json.loads(read_frames_meta(galileo_input_meta).to_json())
    assert_json_subset(round_tripped_json, original_json)


def assert_json_subset(subset: object, superset: object, path: str = "") -> None:
    """Assert every leaf of `subset` appears in `superset` with an equal value.

    Args:
        subset: The value that must be contained.
        superset: The value that must contain it.
        path: JSON pointer built up for the failure message.
    """
    if isinstance(subset, dict) and isinstance(superset, dict):
        for key, value in subset.items():
            assert key in superset, f"{path}/{key} is not in the reference file"
            assert_json_subset(value, superset[key], f"{path}/{key}")
    elif isinstance(subset, list) and isinstance(superset, list):
        assert len(subset) == len(superset), f"{path} has {len(subset)} entries, reference has {len(superset)}"
        for index, (value, reference) in enumerate(zip(subset, superset, strict=True)):
            assert_json_subset(value, reference, f"{path}/{index}")
    else:
        assert subset == superset, f"{path}: {subset!r} != {superset!r}"


def test_vehicle_trajectory_matches_the_shipped_ground_truth(galileo_input: FramesMeta, repo_root: Path) -> None:
    """NOTES.md gotcha 10: the derived vehicle track follows `ground_truth.txt`.

    The ground truth lives in its own world frame, so the comparison is a
    scale-free Kabsch fit of the 226 derived vehicle positions onto the nearest
    ground-truth sample. NOTES.md quotes 3.9 mm ATE for cuSFM's own output on
    this sequence; the input trajectory lands at 4.29 mm RMSE / 8.57 mm max.
    A wrong composition (multiplying by `vehicle_T_cam` instead of its inverse)
    misses by tens of centimetres, so the bound is diagnostic.
    """
    ground_truth_path: Path = repo_root / "data" / "r2b_galileo" / "ground_truth.txt"
    ground_truth_lines: list[str] = ground_truth_path.read_text().splitlines()
    ground_truth_microseconds: np.ndarray = np.array([parse_tum_line(line).timestamp_microseconds for line in ground_truth_lines])
    ground_truth_xyz: np.ndarray = np.array([parse_tum_line(line).world_T_body.translation for line in ground_truth_lines])

    derived_xyz: np.ndarray = np.array([galileo_input.world_T_vehicle(keyframe).translation for keyframe in galileo_input.keyframes])
    nearest: np.ndarray = np.array(
        [int(np.argmin(np.abs(ground_truth_microseconds - keyframe.timestamp_microseconds))) for keyframe in galileo_input.keyframes]
    )
    matched_xyz: np.ndarray = ground_truth_xyz[nearest]

    source_centroid: np.ndarray = derived_xyz.mean(axis=0)
    target_centroid: np.ndarray = matched_xyz.mean(axis=0)
    covariance: np.ndarray = (derived_xyz - source_centroid).T @ (matched_xyz - target_centroid)
    left, _, right_transposed = np.linalg.svd(covariance)
    reflection: np.ndarray = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(right_transposed.T @ left.T)))])
    rotation: np.ndarray = right_transposed.T @ reflection @ left.T
    translation: np.ndarray = target_centroid - rotation @ source_centroid
    residuals_m: np.ndarray = np.linalg.norm(derived_xyz @ rotation.T + translation - matched_xyz, axis=1)

    assert float(np.sqrt((residuals_m**2).mean())) < 0.005
    assert float(residuals_m.max()) < 0.010


def test_filtered_keeps_a_subset_and_renumbers_rig_ids(galileo_input: FramesMeta) -> None:
    """Selection output is a subset with rewritten `synced_sample_id`s."""
    kept_ids: list[int] = [keyframe.keyframe_id for keyframe in galileo_input.keyframes if keyframe.synced_sample_id in {318, 320}]
    renumbered: dict[int, int] = dict.fromkeys(kept_ids, 1)
    subset: FramesMeta = galileo_input.filtered(kept_ids, renumbered)
    assert [keyframe.keyframe_id for keyframe in subset.keyframes] == kept_ids
    assert {keyframe.synced_sample_id for keyframe in subset.keyframes} == {1}
    assert sorted(subset.cameras) == sorted(galileo_input.cameras)


def test_an_edited_camera_survives_serialisation_filtering_and_both_updates(three_samples: FramesMeta) -> None:
    """`dataclasses.replace` on a camera is what every consumer then sees.

    The typed fields are the authority: a 640x480 crop declared through `replace`
    (the pattern `test_raco_backend` already uses) must come back out of
    `to_json`, out of `filtered`, and out of both update methods. Before the fix
    each of those rebuilt the collection from the untouched source message and
    silently restored 1920x1200.
    """
    camera_params_id: int = three_samples.keyframes[0].camera_params_id
    camera: CameraParams = three_samples.cameras[camera_params_id]
    cropped: FramesMeta = replace(
        three_samples,
        cameras={**three_samples.cameras, camera_params_id: replace(camera, image_width=640, image_height=480)},
    )

    reparsed: FramesMeta = parse_message(json_format.Parse(cropped.to_json(), type(cropped.message)()))
    assert (reparsed.cameras[camera_params_id].image_width, reparsed.cameras[camera_params_id].image_height) == (640, 480)

    kept: list[int] = [keyframe.keyframe_id for keyframe in cropped.keyframes[:4]]
    assert cropped.filtered(kept).cameras[camera_params_id].image_width == 640

    moved: pycolmap.Rigid3d = rigid3d_from_axis_angle_degrees(
        axis_xyz=np.array([0.0, 0.0, 1.0]), angle_degrees=3.0, translation_xyz=np.array([1.0, 2.0, 3.0])
    )
    assert cropped.with_camera_to_world({kept[0]: moved}).cameras[camera_params_id].image_width == 640
    assert cropped.with_extrinsics({camera_params_id: moved}).cameras[camera_params_id].image_width == 640

    other_id: int = next(other for other in three_samples.cameras if other != camera_params_id)
    assert cropped.filtered(kept).cameras[other_id].image_width == three_samples.cameras[other_id].image_width


def test_an_edited_keyframe_pose_survives_serialisation_and_filtering(three_samples: FramesMeta) -> None:
    """The same authority holds for a keyframe field edited through `replace`."""
    keyframe: KeyframeMeta = three_samples.keyframes[0]
    moved: pycolmap.Rigid3d = rigid3d_from_axis_angle_degrees(
        axis_xyz=np.array([0.0, 1.0, 0.0]), angle_degrees=11.0, translation_xyz=np.array([4.0, 5.0, 6.0])
    )
    edited: FramesMeta = replace(
        three_samples, keyframes=(replace(keyframe, world_T_cam=moved), *three_samples.keyframes[1:])
    )

    reparsed: FramesMeta = parse_message(json_format.Parse(edited.to_json(), type(edited.message)()))
    assert np.allclose(reparsed.keyframe_by_id()[keyframe.keyframe_id].world_T_cam.translation, [4.0, 5.0, 6.0], atol=1e-9)

    subset: FramesMeta = edited.filtered([keyframe.keyframe_id])
    assert np.allclose(subset.keyframes[0].world_T_cam.translation, [4.0, 5.0, 6.0], atol=1e-9)


def test_an_untouched_collection_still_serialises_byte_for_byte(galileo_input: FramesMeta) -> None:
    """Overlaying the typed fields writes nothing when nothing changed.

    The guard on the fix: unmodelled protobuf fields and cuSFM's default omission
    only survive if an unedited collection is serialised from its source message
    unchanged, byte for byte.
    """
    assert galileo_input.to_json() == json_format.MessageToJson(
        galileo_input.message, indent=2, preserving_proto_field_name=True
    )
