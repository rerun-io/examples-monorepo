import pytest

pytest.importorskip("rerun.experimental.dataloader", reason="mvs depth stream imports need the Rerun dataloader")
pytest.importorskip("torchcodec", reason="mvs depth stream imports need TorchCodec")

import torch  # noqa: E402

from mvs.depth_stream import DepthCollate, DepthRow  # noqa: E402
from mvs.pose_stream import KeyframeBuffer  # noqa: E402

START_NS: int = 5_000
NS_PER_SAMPLE: int = 16_666_667
STEP_M: float = 0.2


def _row(frame_index: int) -> DepthRow:
    """A grid slot whose camera sits STEP_M * frame_index along x, so every slot moves far enough to be a keyframe."""
    K_33 = torch.tensor([[500.0, 0.0, 32.0], [0.0, 500.0, 24.0], [0.0, 0.0, 1.0]])
    return DepthRow(
        video=torch.zeros(3, 48, 64, dtype=torch.uint8),
        rig_quaternion=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        rig_translation=torch.tensor([STEP_M * frame_index, 0.0, 0.0]),
        cam_quaternion=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        cam_translation=torch.zeros(3),
        image_from_camera=K_33.T.reshape(9),
        resolution=torch.tensor([64.0, 48.0]),
    )


def test_source_times_follow_the_order_of_the_stacked_source_poses() -> None:
    """source_times_ns[i] is the grid time of the keyframe whose pose is src_cam_T_world[0, i]: the source views
    reference video frames by these times, so a reorder would show the wrong picture in a slot."""
    # __new__: the real constructor needs a catalog dataset only for the grid timing set here.
    collate = DepthCollate.__new__(DepthCollate)
    collate._index_start_ns = START_NS
    collate._ns_per_sample = NS_PER_SAMPLE
    collate._device = torch.device("cpu")
    collate._keyframes = KeyframeBuffer()
    collate._grid_index = -1
    frame_index: int
    for frame_index in range(40):
        sample = collate([_row(frame_index)])
        assert sample is not None
        if sample.inputs is not None:
            break
    assert sample.inputs is not None
    source_frames: list[int] = [round(-float(sample.inputs.src_cam_T_world_bm44[0, i, 0, 3]) / STEP_M) for i in range(7)]
    assert len(set(source_frames)) == 7 and max(source_frames) < sample.frame_index
    assert list(sample.source_times_ns) == [START_NS + index * NS_PER_SAMPLE for index in source_frames]
