"""``from_layer`` reads back exactly what ``to_rerun`` wrote (integration: ``rerun_roundtrip`` starts an in-process catalog)."""

import numpy as np
import pytest
from conftest import FakeScene
from dagkit.time import STATIC
from dagkit.types import rerun_roundtrip
from test_nodes import kp2d_records

from exo_calib.types import MultiViewKeypoints

pytestmark = pytest.mark.integration


def test_rig_roundtrip_is_static_and_exact() -> None:
    rig = FakeScene().rig
    ((t_ns, back),) = rerun_roundtrip(rig, "/world/exocalib/init", "video_time", STATIC)
    assert t_ns == STATIC and back.names == rig.names
    np.testing.assert_allclose(back.cam_T_world, rig.cam_T_world, atol=1e-6)  # Rerun stores float32 matrices
    np.testing.assert_allclose(back.intrinsics, rig.intrinsics, atol=1e-4)
    np.testing.assert_array_equal(back.resolution_wh, rig.resolution_wh)


def test_kp2d_record_roundtrip_keeps_every_field() -> None:
    t_ns, record = kp2d_records(FakeScene())[3]
    ((back_t, back),) = rerun_roundtrip(record, "/world/exocalib/kp2d", "video_time", t_ns)
    assert isinstance(back, MultiViewKeypoints) and back_t == t_ns
    assert (back.frame_index, back.views, back.box_source, back.crop_input_wh, back.video_wh) == (
        record.frame_index, record.views, record.box_source, record.crop_input_wh, record.video_wh,
    )  # fmt: skip
    for name in ("cam_t_ns", "kp_xy", "conf", "bbox_xyxy", "crop_origin_xy", "crop_size_wh"):
        np.testing.assert_array_equal(getattr(back, name), getattr(record, name))
        assert getattr(back, name).dtype == getattr(record, name).dtype
