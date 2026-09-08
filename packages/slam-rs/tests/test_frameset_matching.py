"""The frameset matcher, the two inertial clocks and the camera subset.

Every case here is :mod:`slam_rs.catalog_feed`, on synthetic tables and without
touching the NAS: which frames of which cameras are one frameset when the rig is
not hardware-synced, how an accelerometer on its own clock reaches the
gyroscope's, and which cameras of a six-camera rig are fed and in what order.
RoboCap is why the code exists, so its numbers are the fixtures; what is under
test is the feed. The rig's own calibration arithmetic and the two real-recording
tests are in ``test_robocap_probe``.
"""

from dataclasses import replace

import numpy as np
import pyarrow as pa
import pytest
from beartype.roar import BeartypeException
from jaxtyping import Float64, Int64
from numpy import ndarray

from slam_rs.catalog_feed import (
    MSD_RIG,
    RigProfile,
    _frame_nearest_anchor,
    match_framesets,
    pair_accel_onto_gyro,
    select_cameras,
)
from slam_rs.reference import ReferenceManifest


def camera_name_statics(names: list[str]) -> pa.Table:
    """A statics table carrying just the ``name`` component of each camera node."""
    return pa.table({f"/world/rig_00/cam_{position:02d}:name": [[name]] for position, name in enumerate(names)})

# --- the accelerometer onto the gyroscope's clock ----------------------------


def test_the_accelerometer_lands_on_the_gyroscopes_timestamps() -> None:
    """Two clocks in, one out: every gyroscope time kept, the acceleration interpolated onto it."""
    gyro_t_ns: Int64[ndarray, " 3"] = np.array([100, 200, 300], dtype=np.int64)
    accel_t_ns: Int64[ndarray, " 3"] = np.array([50, 250, 450], dtype=np.int64)
    accel: Float64[ndarray, "3 3"] = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 8.0], [4.0, 8.0, 16.0]])
    paired = pair_accel_onto_gyro(gyro_t_ns, np.ones((3, 3)), accel_t_ns, accel)

    assert paired.t_ns.tolist() == [100, 200, 300]
    # 100 sits a quarter of the way from 50 to 250, 200 three quarters, and 300 a
    # quarter of the way from 250 to 450.
    assert paired.accel_m_s2[:, 0].tolist() == pytest.approx([0.5, 1.5, 2.5])
    assert paired.accel_m_s2[:, 2].tolist() == pytest.approx([2.0, 6.0, 10.0])
    assert paired.gyro_rad_s.shape == (3, 3)


def test_a_gyroscope_sample_the_accelerometer_does_not_cover_is_dropped() -> None:
    """`numpy.interp` clamps; a clamped endpoint is a measurement nobody took."""
    gyro_t_ns: Int64[ndarray, " 5"] = np.array([10, 60, 110, 160, 210], dtype=np.int64)
    accel_t_ns: Int64[ndarray, " 2"] = np.array([50, 150], dtype=np.int64)
    paired = pair_accel_onto_gyro(gyro_t_ns, np.ones((5, 3)), accel_t_ns, np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]]))

    # The result is a subset of the gyroscope's own timestamps, so 10 and 210 are
    # dropped and 150 — an accelerometer time — was never a candidate.
    assert paired.t_ns.tolist() == [60, 110]
    assert len(paired) == len(paired.gyro_rad_s) == len(paired.accel_m_s2)


def test_the_accelerometers_span_is_closed_at_both_ends() -> None:
    """basalt's endpoints, one nanosecond either side (`dataset_io_robocap.cpp:476-483`).

    A gyroscope sample **on** the first accelerometer timestamp interpolates with
    alpha 0 and one on the last with alpha 1, so both are measurements and both
    are kept; one nanosecond outside has no accelerometer sample on both sides of
    it and is dropped rather than clamped.
    """
    gyro_t_ns: Int64[ndarray, " 4"] = np.array([49, 50, 150, 151], dtype=np.int64)
    accel: Float64[ndarray, "2 3"] = np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
    paired = pair_accel_onto_gyro(gyro_t_ns, np.ones((4, 3)), np.array([50, 150], dtype=np.int64), accel)

    assert paired.t_ns.tolist() == [50, 150]
    assert paired.accel_m_s2[:, 0].tolist() == pytest.approx([1.0, 3.0])


def test_two_accelerometer_samples_on_one_timestamp_take_the_first() -> None:
    """basalt deduplicates the raw channel before pairing (`dataset_io_robocap.cpp:472`).

    Its `interval == 0` guard reads alpha 0 — the sample *before* — and the
    deduplication is what makes that the only possible answer.
    ``numpy.interp`` takes the second of the pair instead, and then interpolates
    the following gyroscope sample from the wrong end of the gap.
    """
    paired = pair_accel_onto_gyro(
        np.array([20, 30], dtype=np.int64),
        np.ones((2, 3)),
        np.array([20, 20, 40], dtype=np.int64),
        np.array([[1.0, 1.0, 1.0], [9.0, 9.0, 9.0], [5.0, 5.0, 5.0]]),
    )

    assert paired.t_ns.tolist() == [20, 30]
    assert paired.accel_m_s2[:, 0].tolist() == pytest.approx([1.0, 3.0])


def test_two_channels_that_do_not_overlap_are_refused() -> None:
    """basalt errors rather than deliver an empty stream (`dataset_io_robocap.cpp:496`).

    The two channels have their own clocks, so a segment whose accelerometer
    stops before its gyroscope starts pairs to nothing. Returning that silently
    fed the estimator a rig with no inertial data at all.
    """
    with pytest.raises(ValueError, match=r"gyroscope spans 1000\.\.2000 ns and the accelerometer 10\.\.20 ns"):
        pair_accel_onto_gyro(
            np.array([1000, 2000], dtype=np.int64),
            np.ones((2, 3)),
            np.array([10, 20], dtype=np.int64),
            np.ones((2, 3)),
        )


def test_one_accelerometer_sample_is_not_enough_to_interpolate() -> None:
    """`raw_accel.size() < 2` is basalt's own bar (`dataset_io_robocap.cpp:473`)."""
    with pytest.raises(ValueError, match="two accelerometer samples"):
        pair_accel_onto_gyro(np.array([10], dtype=np.int64), np.ones((1, 3)), np.array([10], dtype=np.int64), np.ones((1, 3)))


def test_pairing_an_empty_channel_says_which_one() -> None:
    with pytest.raises(ValueError, match="0 gyro"):
        pair_accel_onto_gyro(np.array([], dtype=np.int64), np.zeros((0, 3)), np.array([1], dtype=np.int64), np.ones((1, 3)))
    with pytest.raises(ValueError, match="0 accel"):
        pair_accel_onto_gyro(np.array([1], dtype=np.int64), np.ones((1, 3)), np.array([], dtype=np.int64), np.zeros((0, 3)))
def test_the_pairing_boundary_is_typed() -> None:
    """float32 acceleration is a different array; beartype refuses it rather than upcasting."""
    with pytest.raises(BeartypeException):
        pair_accel_onto_gyro(
            np.array([1, 2], dtype=np.int64),
            np.ones((2, 3)),
            np.array([0, 3], dtype=np.int64),
            np.ones((2, 3), dtype=np.float32),  # pyrefly: ignore[bad-argument-type]
        )

# --- which cameras of the rig are fed ---------------------------------------


def test_the_named_cameras_come_back_in_the_callers_order() -> None:
    """RoboCap's rig order is not the C++'s: cam_04, cam_00, cam_01, cam_05."""
    statics: pa.Table = camera_name_statics(["left_front", "right_front", "left_eye", "right_eye", "left", "right"])
    assert select_cameras(statics, 6, ("left", "left_front", "right_front", "right")) == (4, 0, 1, 5)
    assert select_cameras(statics, 6, None) == (0, 1, 2, 3, 4, 5)


def test_a_hyphenated_name_matches_the_underscored_one() -> None:
    """The rig writes ``left-front`` where basalt's driver spells it ``left_front``.

    Both sides are normalised, so a manifest that spells a camera the way the
    recording itself does selects it rather than being refused.
    """
    statics: pa.Table = camera_name_statics(["left-front", "right-front"])
    assert select_cameras(statics, 2, ("left_front",)) == (0,)
    assert select_cameras(statics, 2, ("left-front",)) == (0,)
    assert select_cameras(camera_name_statics(["left_front", "right_front"]), 2, ("left-front",)) == (0,)


def test_a_camera_that_is_not_there_names_the_ones_that_are() -> None:
    statics: pa.Table = camera_name_statics(["left_front", "right_front"])
    with pytest.raises(ValueError, match=r"no camera named \['upside_down'\].*left_front.*right_front"):
        select_cameras(statics, 2, ("left_front", "upside_down"))


def test_two_cameras_answering_to_one_name_is_refused() -> None:
    with pytest.raises(ValueError, match="cam_00 and cam_01 are both named 'left'"):
        select_cameras(camera_name_statics(["left", "left"]), 2, ("left",))

# --- the frameset matcher ----------------------------------------------------


def test_the_matcher_reproduces_basalts_median_on_robocaps_first_frameset() -> None:
    """Session 15's four cameras start 59 us apart and basalt calls that 70258640500 ns.

    An even camera count takes the lower middle plus half the gap to the upper
    one, which is neither the mean nor either middle value, and it is what the
    NAS `slam` layer's own first row carries.
    """
    left: Int64[ndarray, " 2"] = np.array([70258648000, 70291970222], dtype=np.int64)
    left_front: Int64[ndarray, " 2"] = np.array([70258633000, 70291955222], dtype=np.int64)
    right_front: Int64[ndarray, " 2"] = np.array([70258662000, 70291984222], dtype=np.int64)
    right: Int64[ndarray, " 2"] = np.array([70258603000, 70291936333], dtype=np.int64)
    t_ns, frame_index = match_framesets([left, left_front, right_front, right], 1_000_000)

    assert int(t_ns[0]) == 70258640500
    assert frame_index[0].tolist() == [0, 0, 0, 0]
    # Not the mean (70258636500), and not either middle value.
    assert int(np.mean([70258603000, 70258633000, 70258648000, 70258662000])) == 70258636500


def test_an_odd_camera_count_takes_the_middle_frame() -> None:
    times = [np.array([100, 200], dtype=np.int64), np.array([120, 220], dtype=np.int64), np.array([140, 240], dtype=np.int64)]
    t_ns, _ = match_framesets(times, 1_000)
    assert t_ns.tolist() == [120, 220]


def test_a_camera_that_misses_the_anchor_drops_the_frameset() -> None:
    """The anchor's middle frame has no partner inside the tolerance, so it is not a frameset.

    This is the mechanism that turns RoboCap's 1,594 / 1,588 / 1,596 / 1,592
    frames into 1,588 framesets: a frameset needs all four.
    """
    anchor: Int64[ndarray, " 3"] = np.array([100, 200, 300], dtype=np.int64)
    partner: Int64[ndarray, " 2"] = np.array([105, 305], dtype=np.int64)
    t_ns, frame_index = match_framesets([anchor, partner], 50)

    assert t_ns.tolist() == [102, 302]
    assert frame_index.tolist() == [[0, 0], [2, 1]]


def test_a_frame_belongs_to_one_frameset() -> None:
    """basalt consumes the frame it took, so the next anchor cannot have it again.

    Cursors move to ``selected + 1`` once every camera is inside the tolerance
    (`dataset_io_robocap.cpp:439`). Leaving them on the selected frame fed the
    estimator the same image twice under two frameset timestamps, which is a
    measurement the rig never made.
    """
    anchors: Int64[ndarray, " 2"] = np.array([100, 180], dtype=np.int64)
    partner: Int64[ndarray, " 1"] = np.array([140], dtype=np.int64)
    t_ns, frame_index = match_framesets([anchors, partner], 50)

    assert t_ns.tolist() == [120]
    assert frame_index.tolist() == [[0, 0]]


def test_a_camera_selected_for_an_incomplete_frameset_keeps_its_frame() -> None:
    """A provisional selection is thrown away with the frameset it was for.

    The C++ moves the cursors only after ``complete``
    (`dataset_io_robocap.cpp:439`), so a camera picked for a frameset that then
    fell keeps its frame. Anchor 100 takes camera 1's 150 (|150 - 100| = 50, the
    inclusive edge) but camera 2's only frame is 200, which is 100 away, so the
    frameset falls. Anchor 200 then takes camera 1's 150 (index 0, again the
    inclusive edge) and camera 2's 200 (index 0): one frameset, whose three-camera
    median of (200, 150, 200) is 200 — the matcher adds no offset of its own, the
    caller applies ``cam_time_offset_ns``. Committing camera 1's selection after
    anchor 100 would exhaust that camera and leave the rig with no frameset.
    """
    anchors: Int64[ndarray, " 2"] = np.array([100, 200], dtype=np.int64)
    cameras: list[Int64[ndarray, " 1"]] = [np.array([150], dtype=np.int64), np.array([200], dtype=np.int64)]
    t_ns, frame_index = match_framesets([anchors, *cameras], 50)

    assert t_ns.tolist() == [200]
    assert frame_index.tolist() == [[1, 0, 0]]


def test_a_frame_the_anchor_is_too_early_for_waits_for_the_next_anchor() -> None:
    """A camera ahead of the anchor keeps its frame; one behind it is consumed.

    The C++ advances the cursor past the nearest frame only when that frame is
    *earlier* than the anchor (`dataset_io_robocap.cpp:428`), because a late
    camera's frame is still the right partner for the anchor after this one.
    """
    anchors: Int64[ndarray, " 2"] = np.array([100, 200], dtype=np.int64)
    partner: Int64[ndarray, " 1"] = np.array([190], dtype=np.int64)
    t_ns, frame_index = match_framesets([anchors, partner], 50)

    assert t_ns.tolist() == [195]
    assert frame_index.tolist() == [[1, 0]]


def test_a_frame_no_later_anchor_can_reach_is_consumed_on_the_spot() -> None:
    """The one matcher rule whose effect never reaches the output, tested where it lives.

    `dataset_io_robocap.cpp:428` moves a camera's cursor past its nearest frame
    when that frame is out of tolerance *and* earlier than the anchor. Anchors
    only increase, so such a frame is farther from every later anchor still and
    no frameset can ever take it. That also makes the rule invisible in
    `match_framesets`'s output: leaving the cursor on the stale frame costs only
    the nearest-frame walk, which re-reaches the same frame at the next anchor.
    So it is pinned on the per-camera step instead, hand-computed:

    * frames (0, 1000), cursor 0, anchor 100, tolerance 20 — the walk stays on 0
      (|1000 - 100| = 900 is no closer than |0 - 100| = 100), 100 > 20 drops the
      frameset, and 0 < 100, so the cursor moves to 1 and 0 is gone.
    * frame (200) alone, same anchor — 100 > 20 drops it too, but 200 > 100, so
      the cursor stays on 0 and 200 waits for the next anchor.
    * frames (90, 110), same anchor — the walk ties onto 110, |110 - 100| = 10 is
      inside the tolerance, and the cursor stays where it was: the caller commits
      ``selected + 1`` only once the whole frameset stands.
    * cursor 1 on a one-frame camera — exhausted, and it stays exhausted.
    """
    assert _frame_nearest_anchor(np.array([0, 1000], dtype=np.int64), 0, 100, 20) == (None, 1)
    assert _frame_nearest_anchor(np.array([200], dtype=np.int64), 0, 100, 20) == (None, 0)
    assert _frame_nearest_anchor(np.array([90, 110], dtype=np.int64), 0, 100, 20) == (1, 0)
    assert _frame_nearest_anchor(np.array([0], dtype=np.int64), 1, 100, 20) == (None, 1)


def test_the_tolerance_is_inclusive() -> None:
    """`> kFramesetToleranceNs` drops it, so the tolerance itself still joins."""
    anchor: Int64[ndarray, " 1"] = np.array([0], dtype=np.int64)
    t_ns, _ = match_framesets([anchor, np.array([1_000_000], dtype=np.int64)], 1_000_000)
    assert t_ns.tolist() == [500_000]
    with pytest.raises(ValueError, match="no frameset has all 2 cameras"):
        match_framesets([anchor, np.array([1_000_001], dtype=np.int64)], 1_000_000)


def test_a_tie_takes_the_later_frame() -> None:
    """The advance condition is ``<=``, so two frames equally close pick the later one."""
    t_ns, frame_index = match_framesets([np.array([100], dtype=np.int64), np.array([90, 110], dtype=np.int64)], 50)
    assert frame_index.tolist() == [[0, 1]]
    assert t_ns.tolist() == [105]


def test_interior_drops_are_allowed_one_in_a_thousand() -> None:
    """A run that drops more interior framesets than basalt tolerates is not a run.

    `dataset_io_robocap.cpp:458` allows ``max(1, ceil(interior * 0.001))``
    incomplete framesets whose anchor lies inside every camera's own span; more
    than that is a rig whose cameras are not the same recording.
    """
    anchors: Int64[ndarray, " 4"] = np.array([1000, 1100, 1200, 1300], dtype=np.int64)
    # Four interior anchors allow one drop: this partner misses the third anchor.
    t_ns, _ = match_framesets([anchors, np.array([990, 1110, 1310], dtype=np.int64)], 50)
    assert t_ns.tolist() == [995, 1105, 1305]

    # The same rig missing two of them is refused, and the error carries both counts.
    with pytest.raises(ValueError, match="2 of 4 interior framesets are incomplete, more than the 1"):
        match_framesets([anchors, np.array([990, 1310], dtype=np.int64)], 50)


def test_the_drop_allowance_rounds_up_past_a_thousand_anchors() -> None:
    """Past a thousand interior anchors the ceil, not the floor of one, sets the bar.

    The allowance is ``max(1, ceil(interior * 0.001))``
    (`dataset_io_robocap.cpp:458`), so 1,001 interior anchors allow
    ceil(1.001) = 2 drops. The anchors here are 0, 100, ... 100,000 and the
    partner is the same list minus two of its interior frames, which keeps its
    first and last frame and therefore keeps all 1,001 anchors interior: 999
    framesets stand, two interior anchors drop, and the run is accepted. Removing
    a third makes 3 > 2 and the rig is refused. Truncation instead of a ceil, or
    ``max(1, ...)`` alone, would allow one and refuse the accepted case.
    """
    anchors: Int64[ndarray, " 1001"] = np.arange(1001, dtype=np.int64) * 100
    two_missing: Int64[ndarray, " 999"] = np.delete(anchors, [300, 600])
    t_ns, _ = match_framesets([anchors, two_missing], 50)
    assert t_ns.tolist() == two_missing.tolist()

    three_missing: Int64[ndarray, " 998"] = np.delete(anchors, [300, 600, 900])
    with pytest.raises(ValueError, match="3 of 1001 interior framesets are incomplete, more than the 2"):
        match_framesets([anchors, three_missing], 50)


def test_only_a_drop_inside_every_cameras_span_counts_against_the_run() -> None:
    """An anchor no camera could partner is not the rig's fault, and is not counted.

    Both counters are gated on ``overlap_start <= anchor <= overlap_end``
    (`dataset_io_robocap.cpp:410` for the anchors, `:436` for the drops). The
    anchors here are 0, 10, 20, 30, 40 and the one partner has 20 and 30, so the
    overlap is [20, 30]: two interior anchors, both complete, and the misses at
    0, 10 and 40 are exterior. Counting those would be 3 drops of 5 anchors
    against an allowance of max(1, ceil(0.005)) = 1, and this rig — a partner
    camera that simply started late and stopped early — would be refused.
    """
    anchors: Int64[ndarray, " 5"] = np.array([0, 10, 20, 30, 40], dtype=np.int64)
    t_ns, frame_index = match_framesets([anchors, np.array([20, 30], dtype=np.int64)], 1)

    assert t_ns.tolist() == [20, 30]
    assert frame_index.tolist() == [[2, 0], [3, 1]]


def test_frameset_timestamps_must_strictly_increase() -> None:
    """Two anchors on one timestamp would file two framesets under one time."""
    with pytest.raises(ValueError, match="frameset timestamps are not strictly increasing: 100 follows 100"):
        match_framesets([np.array([100, 100], dtype=np.int64)], 1_000)


def test_a_camera_with_no_frames_is_named() -> None:
    """basalt refuses the rig rather than the frameset (`dataset_io_robocap.cpp:380`)."""
    with pytest.raises(ValueError, match="camera 1 has no frames"):
        match_framesets([np.array([100, 200], dtype=np.int64), np.array([], dtype=np.int64)], 50)


def test_a_rig_no_frameset_survives_says_so() -> None:
    with pytest.raises(ValueError, match="no frameset has all 2 cameras within 10 ns"):
        match_framesets([np.array([0, 100], dtype=np.int64), np.array([500, 600], dtype=np.int64)], 10)


def test_the_matcher_needs_a_camera() -> None:
    with pytest.raises(ValueError, match="at least one camera"):
        match_framesets([], 1_000)


def test_the_profile_comes_from_the_manifest_not_the_code(manifest: ReferenceManifest) -> None:
    """One place says what the C++ ran, and the profile only reads it.

    All four fields, each against the manifest's own value: the tolerance and the
    pairing rule were constants in the tool, which left the claim half true.
    """
    profile = RigProfile.from_robocap(manifest.robocap)
    assert profile.camera_names == manifest.robocap.camera_names == ("left", "left_front", "right_front", "right")
    assert profile.downscale == manifest.robocap.downscale == 3
    assert profile.interpolate_accel_onto_gyro is manifest.robocap.interpolate_accel_onto_gyro is True
    assert profile.frameset_tolerance_ns == manifest.robocap.frameset_tolerance_ns == 1_000_000
    assert profile.video_time_is_absolute is manifest.robocap.video_time_is_absolute is True
    # What MSD is, and what every default in the feed means: the other state of
    # each of the five, so the profile is a statement and not a shape.
    assert RigProfile(camera_names=None, downscale=1, interpolate_accel_onto_gyro=False, frameset_tolerance_ns=0, video_time_is_absolute=False) == MSD_RIG


def test_a_profile_with_no_frames_left_is_refused_on_construction(manifest: ReferenceManifest) -> None:
    """The downscale is checked where it is stated, before a byte is read.

    `_build_feed` reads the whole video index off the recording before it builds
    the first `CameraCalib`, which is where the downscale used to be validated.
    """
    with pytest.raises(ValueError, match="downscale must be at least 1; got 0"):
        RigProfile(downscale=0)
    with pytest.raises(ValueError, match="downscale must be at least 1; got -3"):
        replace(RigProfile.from_robocap(manifest.robocap), downscale=-3)
