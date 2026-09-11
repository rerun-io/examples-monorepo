"""The D17 hold, which the replay tool and the V2 gate both drive.

The rule under test is that arrival order never reaches the trajectory: a
frameset refused for want of inertial samples is held and tracked again once
they arrive, before the frameset that brought them, and produces the pose a run
that had the samples all along would have produced. ``_core`` proves that for
one frameset (``test_import.py``); what is proved here is that
:class:`slam_rs.tracking.Lockstep` — the one loop both drivers use — really does
hold it, that the hold cannot grow, and that a run whose framesets were never
covered still says so.

The rig and the pipeline are :mod:`conftest` fixtures, which pytest injects; the
factory aliases below are declared here rather than imported from another test
module, because ``tests`` is not on the typechecker's search path and every
module in this directory therefore stands alone.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rerun as rr
from fixture_types import FRAME_PERIOD_NS, IMU_PERIOD_NS, CameraFactory, PipelineFactory, TextureFactory, gravity_batch
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray

from slam_rs import _core, tracking
from slam_rs.catalog_feed import Frameset, ImuStream
from slam_rs.reference import ReferenceManifest
from slam_rs.tracking import MAX_HELD_FRAMESETS, Lockstep, robocap_estimator_files
from slam_rs.trajectory import empty_trajectory
from slam_rs.vio_log import VioLogger, VioStage


def frameset(step: int, texture: TextureFactory, sample_t_ns: Int64[ndarray, " n_samples"]) -> Frameset:
    """One synthetic frameset of the two-camera rig, carrying the given samples.

    Args:
        step: Frameset index; the scene is shifted by it, and it sets the timestamp.
        texture: The scene each frameset is a shifted copy of.
        sample_t_ns: Inertial sample times to attach, which may cover the frame or not.

    Returns:
        The frameset, with gravity along +z on every sample and no ground truth.
    """
    images: list[UInt8[ndarray, "h w"]] = [texture(step, 0), texture(step + 1, 0)]
    gyro_rad_s, accel_m_s2 = gravity_batch(sample_t_ns)
    return Frameset(
        t_ns=step * FRAME_PERIOD_NS,
        images=images,
        imu=ImuStream(t_ns=sample_t_ns, gyro_rad_s=gyro_rad_s, accel_m_s2=accel_m_s2),
        ground_truth=None,
    )


def test_a_held_frameset_tracks_before_the_one_that_unblocked_it(pipeline: PipelineFactory, texture: TextureFactory) -> None:
    """Time order is the trajectory, so the retry comes first and the poses are exact.

    One lockstep gets each frameset's samples with the frameset. The other gets
    only the samples up to its own frame time, which do not cover it, so the
    frameset is held; the next frameset's batch runs past both frame times and
    the held one tracks then. Both trajectories must be identical, pose by pose.
    """
    batches: list[Int64[ndarray, " n_samples"]] = [
        np.arange(step * FRAME_PERIOD_NS, (step + 1) * FRAME_PERIOD_NS, IMU_PERIOD_NS, dtype=np.int64) for step in range(4)
    ]
    # The same samples in the same order, only split differently: the first
    # frameset arrives with one sample at its own frame time, which does not
    # cover it, and the rest of them come with the second frameset.
    held_batches: list[Int64[ndarray, " n_samples"]] = [
        batches[0][:1],
        np.concatenate([batches[0][1:], batches[1]]),
        batches[2],
        batches[3],
    ]
    covered: Lockstep = Lockstep(vio=pipeline(2))
    held: Lockstep = Lockstep(vio=pipeline(2))
    covered_poses: list[Float64[ndarray, " 7"]] = []
    held_poses: list[Float64[ndarray, " 7"]] = []
    for step, (whole, split) in enumerate(zip(batches, held_batches, strict=True)):
        covered_poses.extend(result.world_from_rig for _tracked, result in covered.push(frameset(step, texture, whole)))
        held_poses.extend(result.world_from_rig for _tracked, result in held.push(frameset(step, texture, split)))

    assert len(covered.elapsed_ms) == len(held.elapsed_ms) == 4
    assert not covered.pending and not held.pending
    assert covered.retries == 0
    # One refusal, and the frameset it refused tracked again on the next push,
    # before the frameset whose samples unblocked it.
    assert held.retries == 1
    for expected, actual in zip(covered_poses, held_poses, strict=True):
        np.testing.assert_array_equal(expected, actual)


def test_the_hold_never_grows_past_one_refused_frameset(pipeline: PipelineFactory, texture: TextureFactory) -> None:
    """A feed that stops supplying samples fails at the frameset that broke the rule.

    Without the bound the hold retains every remaining frameset — two decoded
    960x960 frames each on a reference segment — and the error names a count
    instead of the frameset.
    """
    lockstep: Lockstep = Lockstep(vio=pipeline(2))
    nothing: Int64[ndarray, " 0"] = np.zeros(0, dtype=np.int64)
    for step in range(MAX_HELD_FRAMESETS):
        assert list(lockstep.push(frameset(step, texture, nothing))) == []
    with pytest.raises(ValueError, match=f"frameset {MAX_HELD_FRAMESETS * FRAME_PERIOD_NS} takes the hold to 3 framesets"):
        list(lockstep.push(frameset(MAX_HELD_FRAMESETS, texture, nothing)))


def test_a_run_that_never_tracked_still_reports_what_it_held(
    pipeline: PipelineFactory, camera: CameraFactory, texture: TextureFactory, tmp_path: Path
) -> None:
    """The run that most needs the diagnostic is the one that used to suppress it.

    A replay whose framesets are never covered has no timings, so a summary
    guarded on them printed nothing at all — and the held framesets are what the
    reader needed to see.
    """
    rr.init("slam-rs-tracking-test", recording_id="never-tracked")
    rr.save(tmp_path / "never-tracked.rrd")
    stage: VioStage = VioStage(
        lockstep=Lockstep(vio=pipeline(2)),
        logger=VioLogger(
            cameras=(camera(0, 0.0), camera(1, 0.1)),
            ground_truth=empty_trajectory(),
            frame_t_ns=np.zeros(1, dtype=np.int64),
        ),
    )
    stage.run(frameset(0, texture, np.zeros(0, dtype=np.int64)))
    rr.disconnect()

    assert not stage.elapsed_ms
    assert "nothing tracked" in stage.summary()
    assert "NEVER COVERED BY THE IMU at [0]" in stage.summary()




def test_the_estimator_is_configured_from_basalts_own_two_files(manifest: ReferenceManifest) -> None:
    """The RoboCap estimator uses the two configuration files named by its manifest.
    Both files are checked into the package, so configuration needs no external bundle.
    """
    calibration: _core.Calibration
    flow: _core.VioConfig
    config_text: str
    calibration, flow, config_text = robocap_estimator_files(manifest)
    assert list(calibration.resolution) == [(640, 360)] * 4
    assert flow.optical_flow_image_safe_radius > 0.0
    # The text handed back is the one the config came from: a digest over it names what the estimator read.
    assert _core.VioConfig.from_json(config_text).to_json() == flow.to_json()


def test_robocap_estimator_files_hand_back_the_very_string_they_parsed(manifest: ReferenceManifest, monkeypatch: pytest.MonkeyPatch) -> None:
    """The RoboCap config text is the one string ``VioConfig.from_json`` received, and the profile reaches it."""
    parsed: list[str] = []

    class RecordingVioConfig:
        @staticmethod
        def from_json(text: str) -> _core.VioConfig:
            parsed.append(text)
            return _core.VioConfig.from_json(text)

    monkeypatch.setattr(tracking, "_core", SimpleNamespace(VioConfig=RecordingVioConfig, Calibration=_core.Calibration))
    config_text: str
    _calibration, _flow, config_text = robocap_estimator_files(manifest, profile="fast")
    assert parsed == [config_text]
    assert '"port.redetect_survivor_ratio"' in config_text
