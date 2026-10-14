"""``RigPose.from_layer`` reads back what ``to_rerun`` wrote (integration: ``rerun_roundtrip`` starts an in-process catalog)."""

import numpy as np
import pytest
from dagkit.testing import rerun_roundtrip

from slam_rs.types import RigPose

pytestmark = pytest.mark.integration


def test_a_rig_pose_reads_back_at_its_time_with_its_quaternion_order_in_float32() -> None:
    quaternion = np.array([0.9, -0.1, 0.3, 0.2])  # distinct components: a w-first / w-last mix-up cannot pass
    pose = RigPose(np.array([1.25, -3.5, 0.125]), quaternion / np.linalg.norm(quaternion))
    ((t_ns, back),) = rerun_roundtrip(pose, "/world/rig_00", "video_time", 1_654_613_237_343_500)
    assert t_ns == 1_654_613_237_343_500 and isinstance(back, RigPose)
    assert back.position_m.dtype == back.quaternion_wxyz.dtype == np.float64
    np.testing.assert_allclose(back.position_m, pose.position_m, atol=1e-6)  # the layer stores Transform3D in float32
    np.testing.assert_allclose(back.quaternion_wxyz, pose.quaternion_wxyz, atol=1e-6)
