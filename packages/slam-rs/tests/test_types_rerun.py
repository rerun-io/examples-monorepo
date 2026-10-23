"""``RigPose.from_layer`` reads back what ``to_rerun`` wrote, ``CatalogPoses`` the whole table (integration: an in-process catalog)."""

from pathlib import Path

import numpy as np
import pytest
import rerun as rr
from dagkit.sources import Column, RecordingSource
from dagkit.testing import rerun_roundtrip

from slam_rs.types import CatalogPoses, RigPose

pytestmark = pytest.mark.integration


def test_a_rig_pose_reads_back_at_its_time_with_its_quaternion_order_in_float32() -> None:
    quaternion = np.array([0.9, -0.1, 0.3, 0.2])  # distinct components: a w-first / w-last mix-up cannot pass
    pose = RigPose(np.array([1.25, -3.5, 0.125]), quaternion / np.linalg.norm(quaternion))
    ((t_ns, back),) = rerun_roundtrip(pose, "/world/rig_00", "video_time", 1_654_613_237_343_500)
    assert t_ns == 1_654_613_237_343_500 and isinstance(back, RigPose)
    assert back.position_m.dtype == back.quaternion_wxyz.dtype == np.float64
    np.testing.assert_allclose(back.position_m, pose.position_m, atol=1e-6)  # the layer stores Transform3D in float32
    np.testing.assert_allclose(back.quaternion_wxyz, pose.quaternion_wxyz, atol=1e-6)


def test_the_pose_table_reads_a_recording_the_way_slam_rs_writes_it(tmp_path: Path) -> None:
    path: Path = tmp_path / "slam_rs.rrd"
    recording = rr.RecordingStream("slam-rs", recording_id="segment", send_properties=False)
    recording.save(path)
    for i, t_ns in enumerate([3_000_000, 1_000_000, 2_000_000]):
        recording.set_time("video_time", duration=np.timedelta64(t_ns, "ns"))
        recording.log("/world/rig_00", rr.Transform3D(translation=[float(i), 0.0, 0.0], quaternion=rr.Quaternion(xyzw=[0.0, 0.0, 0.0, 1.0])))
    recording.flush()
    recording.disconnect()
    (event,) = list(RecordingSource(path, {"poses": Column(CatalogPoses, "/world/rig_00")}).events())  # one STATIC table, sorted by time
    assert isinstance(event.value, CatalogPoses) and event.value.t_ns.tolist() == [1_000_000, 2_000_000, 3_000_000]
    assert event.value.world_from_rig[:, 0, 3].tolist() == [1.0, 2.0, 0.0]
