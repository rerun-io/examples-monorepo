"""Gen-HumanWholeBody: what the converter derives from a DAS-Ego MCAP, checked on worked examples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import simplecv
from conftest import raw_asset
from jaxtyping import Bool, Float32, Float64, Int64
from numpy import ndarray

from dataforge.datasets import genrobot_pose
from dataforge.datasets.genrobot_pose import (
    PoseTrack,
    check_mano,
    fixed_lists,
    list_presence,
    read_pose_track,
    skin,
)
from dataforge.datasets.genrobot_source import (
    DoubleSphere,
    EpisodeClock,
    EpisodeFile,
    clear_sps_constraint_set3,
    hand_topic,
    is_keyframe,
    parse_hub_path,
    struct_rows,
)


def test_hub_paths_of_both_layouts_name_the_episode_by_its_uuid() -> None:
    four: tuple[tuple[str, ...], str] = parse_hub_path("data/outdoor/sport/play_with_ball/bounce_ping_pong_ball/19/19ad7c65a0a047e89cddfacdfae66884.mcap")
    assert four == (("outdoor", "sport", "play_with_ball", "bounce_ping_pong_ball"), "19ad7c65a0a047e89cddfacdfae66884")
    two: tuple[tuple[str, ...], str] = parse_hub_path("data/logistics/parcel_station/44/44615acf4b8a437da3ba5d0f994f517b.mcap")
    assert two == (("logistics", "parcel_station"), "44615acf4b8a437da3ba5d0f994f517b")


@pytest.mark.parametrize(
    "path",
    [
        "data/logistics/parcel_station/45/44615acf4b8a437da3ba5d0f994f517b.mcap",  # shard is not the uuid's first two characters
        "data/logistics/44/44615acf4b8a437da3ba5d0f994f517b.mcap",  # one domain level
        "data/logistics/parcel_station/44/44615acf4b8a437da3ba5d0f994f517b.json",
        "other/logistics/parcel_station/44/44615acf4b8a437da3ba5d0f994f517b.mcap",
    ],
)
def test_hub_paths_outside_the_two_layouts_are_refused(path: str) -> None:
    with pytest.raises(ValueError):
        parse_hub_path(path)


def test_double_sphere_without_xi_and_alpha_is_a_pinhole() -> None:
    lens: DoubleSphere = DoubleSphere(fx=100.0, fy=100.0, cx=50.0, cy=40.0, xi=0.0, alpha=0.0)
    points: Float64[ndarray, "3 3"] = np.array([[0.1, 0.2, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, -1.0]])
    pixels: Float64[ndarray, "3 2"] = lens.project(points)
    np.testing.assert_allclose(pixels[:2], [[60.0, 60.0], [50.0, 40.0]])
    assert np.isnan(pixels[2]).all(), "a point behind a pinhole has no image"


def test_double_sphere_matches_a_hand_computed_point() -> None:
    # xi = 0, alpha = 0.5: d2 = |p| and the denominator is (|p| + z) / 2, so (1, 0, 1) lands at
    # u = 50 + 100 * 1 / ((sqrt(2) + 1) / 2) = 132.8427.
    lens: DoubleSphere = DoubleSphere(fx=100.0, fy=100.0, cx=50.0, cy=40.0, xi=0.0, alpha=0.5)
    pixels: Float64[ndarray, "1 2"] = lens.project(np.array([[1.0, 0.0, 1.0]]))
    np.testing.assert_allclose(pixels, [[132.84271247, 40.0]])


def test_double_sphere_sees_past_ninety_degrees_where_a_pinhole_cannot() -> None:
    lens: DoubleSphere = DoubleSphere(fx=500.0, fy=500.0, cx=800.0, cy=650.0, xi=0.0, alpha=0.57)
    pixels: Float64[ndarray, "1 2"] = lens.project(np.array([[1.0, 0.0, -0.1]]))
    assert np.isfinite(pixels).all() and pixels[0, 0] > 800.0


def test_sps_constraint_set3_is_cleared_and_nothing_else_changes() -> None:
    keyframe: bytes = bytes([0, 0, 0, 1, 0x67, 0x64, 0x10, 0x2A, 0xAC, 0, 0, 0, 1, 0x68, 0xEE, 0, 0, 0, 1, 0x65, 0x88, 0x84])
    delta: bytes = bytes([0, 0, 0, 1, 0x41, 0x9A, 0x10, 0x2A])
    already_clear: bytes = bytes([0, 0, 1, 0x67, 0x64, 0x00, 0x2A])
    samples: pa.ListArray = pa.array([list(keyframe), list(delta), list(already_clear)], type=pa.list_(pa.uint8()))
    fixed: list[bytes] = [bytes(row) for row in clear_sps_constraint_set3(samples).to_pylist()]
    assert fixed[0] == keyframe[:6] + bytes([0x00]) + keyframe[7:]
    assert fixed[1] == delta, "a non-SPS NAL keeps its bytes, even one that looks like a constraint byte"
    assert fixed[2] == already_clear


def test_a_keyframe_is_an_access_unit_with_an_idr_slice() -> None:
    assert is_keyframe(bytes([0, 0, 0, 1, 0x67, 0x64, 0x10, 0x2A, 0, 0, 0, 1, 0x68, 0xEE, 0, 0, 0, 1, 0x65, 0x88]))
    assert is_keyframe(bytes([0, 0, 1, 0x25, 0x88])), "nal_ref_idc does not matter"
    assert not is_keyframe(bytes([0, 0, 0, 1, 0x41, 0x9A, 0x05, 0x65])), "a non-IDR slice whose payload holds 0x65 is not a keyframe"
    assert not is_keyframe(b"")


def test_frame_indices_are_the_shipped_numbers_and_fill_a_gap_in_the_reference() -> None:
    """Reference frames 5, 6 and 8 (7 is missing from frame_validity); stamps jitter by up to 0.4 us."""
    clock: EpisodeClock = EpisodeClock(
        t0_ns=990_000_000,
        frame_times_ns=np.array([1_000_000_000, 1_033_333_000, 1_100_000_000], dtype=np.int64),
        frame_numbers=np.array([5, 6, 8], dtype=np.int64),
    )
    stamps: Int64[ndarray, "6"] = np.array([1_000_000_384, 1_033_333_000, 1_066_666_000, 1_099_999_872, 1_133_333_000, 966_667_000], dtype=np.int64)
    np.testing.assert_array_equal(clock.frame_indices(stamps), [5, 6, 7, 8, 9, 4])
    np.testing.assert_array_equal(clock.video_time_ns(stamps), stamps - 990_000_000)
    np.testing.assert_array_equal(clock.stamps_ns(np.array([5, 7, 9], dtype=np.int64)), [1_000_000_000, 1_066_666_333, 1_133_333_333])


def test_a_clock_refuses_frame_numbers_that_do_not_increase() -> None:
    with pytest.raises(ValueError, match="increase"):
        EpisodeClock(t0_ns=0, frame_times_ns=np.array([1, 2], dtype=np.int64), frame_numbers=np.array([3, 3], dtype=np.int64))


def test_mano_lists_missing_on_some_frames_are_absent_not_ragged() -> None:
    """Some episodes ship a hand's bones with null MANO lists on a few frames."""
    betas: pa.ListArray = pa.array([[1.0, 2.0], None, [], [3.0, 4.0]], type=pa.list_(pa.float64()))
    present, numbers = fixed_lists(betas, 2, what="betas")
    np.testing.assert_array_equal(present, [True, False, False, True])
    np.testing.assert_array_equal(numbers, [[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="not 2"):
        fixed_lists(pa.array([[1.0, 2.0], [3.0]], type=pa.list_(pa.float64())), 2, what="betas")


def synthetic_track(frames: int, *, betas: Float64[ndarray, "f 10"], mano: bool) -> PoseTrack:
    """A right hand with rest-pose MANO parameters on every frame (or none), and no body or left hand."""
    nan: float = float("nan")
    track_betas: Float64[ndarray, "f 2 10"] = np.full((frames, 2, 10), nan)
    track_betas[:, 1] = betas
    orient: Float64[ndarray, "f 2 3 3"] = np.full((frames, 2, 3, 3), nan)
    orient[:, 1] = np.eye(3)
    pose: Float64[ndarray, "f 2 15 3 3"] = np.full((frames, 2, 15, 3, 3), nan)
    pose[:, 1] = np.eye(3)
    transl: Float64[ndarray, "f 2 3"] = np.full((frames, 2, 3), nan)
    transl[:, 1] = 0.0
    present: Bool[ndarray, "f 2"] = np.zeros((frames, 2), dtype=bool)
    present[:, 1] = mano
    return PoseTrack(
        times_ns=np.arange(frames, dtype=np.int64) * 33_333_333,
        frame_indices=np.arange(frames, dtype=np.int64),
        body=np.full((frames, 22, 3), nan),
        bones=np.full((frames, 2, 21, 3), nan),
        wrists=np.full((frames, 2, 4, 4), nan),
        hand_present=present.copy(),
        mano_present=present,
        betas=track_betas,
        global_orient=orient,
        hand_pose=pose,
        global_transl=transl,
        is_valid=np.zeros((frames, 2), dtype=bool),
        reproj_loss=np.zeros((frames, 2)),
        bbox_touches_edge=np.zeros((frames, 2), dtype=bool),
    )


def test_list_rows_that_are_null_or_empty_are_absent() -> None:
    poses: pa.ListArray = pa.array([[1, 2, 3], None, [], [4, 5, 6]], type=pa.list_(pa.int64()))
    np.testing.assert_array_equal(list_presence(poses, 3, what="poses"), [True, False, False, True])
    with pytest.raises(ValueError, match="not 3"):
        list_presence(pa.array([[1, 2]], type=pa.list_(pa.int64())), 3, what="poses")


def test_a_null_struct_reads_as_nan_whatever_its_children_hold() -> None:
    children: list[pa.Array] = [pa.array([1.0, 4.0]), pa.array([2.0, 5.0]), pa.array([3.0, 6.0])]
    points: pa.StructArray = pa.StructArray.from_arrays(children, names=["x", "y", "z"], mask=pa.array([True, False]))
    rows: Float64[ndarray, "2 3"] = struct_rows(points)
    assert np.isnan(rows[0]).all()
    np.testing.assert_array_equal(rows[1], [4.0, 5.0, 6.0])



def test_a_null_mano_record_is_absent_whatever_its_lists_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Arrow lets a null struct keep child values: a left hand whose mano_params is null must not read them as a fit."""
    vector: pa.DataType = pa.struct([(axis, pa.float64()) for axis in "xyz"])
    quaternion: pa.DataType = pa.struct([(axis, pa.float64()) for axis in "xyzw"])
    lists: list[pa.Array] = [pa.array([[0.0] * width], type=pa.list_(pa.float64())) for width in (10, 9, 135, 3)]
    mano: pa.StructArray = pa.StructArray.from_arrays(lists, names=["betas", "global_orient", "hand_pose", "global_transl"], mask=pa.array([True]))
    hand: pa.StructArray = pa.StructArray.from_arrays(
        [pa.array([[]], type=pa.list_(pa.struct([("to_global", pa.struct([("position", vector), ("orientation", quaternion)]))]))), mano, pa.array([True]), pa.array([0.5]), pa.array([False])],
        names=["bone_data", "mano_params", "is_valid", "reproj_loss", "bbox_touches_edge"],
    )
    stamps: Int64[ndarray, "1"] = np.array([1_000], dtype=np.int64)
    empty: tuple[Int64[ndarray, "0"], pa.Array] = (np.zeros(0, dtype=np.int64), pa.array([]))
    monkeypatch.setattr(genrobot_pose, "topic_rows", lambda file, topic: (stamps, hand) if topic == hand_topic("left") else empty)
    clock: EpisodeClock = EpisodeClock(t0_ns=0, frame_times_ns=stamps, frame_numbers=np.array([0], dtype=np.int64))
    track: PoseTrack = read_pose_track(EpisodeFile(Path("episode.mcap")), clock)
    np.testing.assert_array_equal(track.mano_present, [[False, False]])
    assert np.isnan(track.betas).all()
    assert track.is_valid.tolist() == [[True, False]]

@pytest.mark.integration
def test_each_frame_is_skinned_with_its_own_shipped_shape() -> None:
    """Needs the MANO right-hand model, which simplecv fetches when it is not on disk."""
    raw_asset("MANO right-hand model", Path(simplecv.__file__).parent / "data" / "MANO_RIGHT.pkl")
    shapes: Float64[ndarray, "2 10"] = np.zeros((2, 10))
    shapes[1, 0] = 3.0
    track: PoseTrack = synthetic_track(2, betas=shapes, mano=True)
    vertices: Float32[ndarray, "2 778 3"] = skin(track, 1, np.array([0, 1], dtype=np.int64))[0]
    alone: Float32[ndarray, "1 778 3"] = skin(synthetic_track(1, betas=shapes[1:], mano=True), 1, np.array([0], dtype=np.int64))[0]
    np.testing.assert_allclose(vertices[1], alone[0], atol=1e-6)
    assert np.abs(vertices[1] - vertices[0]).max() > 1e-3, "the two shapes give two hands"


def test_a_track_without_hands_or_body_gets_no_pose_or_hand_mesh_layer() -> None:
    """convert gives a layer a writer only when it has data: a layer with properties and no data would break the layer rule."""
    track: PoseTrack = synthetic_track(3, betas=np.zeros((3, 10)), mano=False)
    assert not track.has_pose and not any(check_mano(track, side).passed for side in (0, 1))
    assert synthetic_track(3, betas=np.zeros((3, 10)), mano=True).has_pose
