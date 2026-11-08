"""FramePrefetcher hands frames over in order, on the caller's CUDA stream, and re-raises the worker's errors."""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="MASt3R-SLAM requires PyTorch")

pytestmark = [pytest.mark.integration, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")]


def _fake_dataset(fail_at: int | None = None):
    """A dataset whose read_resized(i) is an image filled with i, and which raises at ``fail_at``."""
    from mast3r_slam.dataloader import MonocularDataset

    class FakeDataset(MonocularDataset):
        def read_resized(self, idx: int):
            return _read_resized(idx, fail_at)

    return FakeDataset(img_size=224)


def _read_resized(i: int, fail_at: int | None):
    from mast3r_slam.image_utils import ResizedImage

    if i == fail_at:
        raise ValueError(f"frame {i} is broken")
    # A slow GPU op on the worker's stream, so a missing stream handover would show stale values.
    rgb = torch.full((1, 3, 224, 224), float(i), device="cuda")
    for _ in range(20):
        rgb = rgb @ torch.eye(224, device="cuda")
    return ResizedImage(rgb_tensor=rgb, true_shape=np.array([[224, 224]], dtype=np.int32), rgb_uint8=np.full((224, 224, 3), i, dtype=np.uint8))


def test_frames_arrive_in_order_and_ready() -> None:
    from mast3r_slam.frame_prefetch import FramePrefetcher

    with FramePrefetcher(_fake_dataset(), 6, img_downsample=1) as frames:
        for i in range(6):
            frame = frames.get()
            assert frame.frame_id == i
            assert float(frame.rgb_tensor.mean()) == float(i)
            assert int(frame.rgb[0, 0, 0] * 255 + 0.5) == i


def test_worker_errors_reach_the_caller() -> None:
    from mast3r_slam.frame_prefetch import FramePrefetcher

    with FramePrefetcher(_fake_dataset(fail_at=2), 6, img_downsample=1) as frames:
        frames.get()
        frames.get()
        with pytest.raises(ValueError, match="frame 2 is broken"):
            frames.get()


def test_exit_does_not_wait_for_unread_frames() -> None:
    from mast3r_slam.frame_prefetch import FramePrefetcher

    with FramePrefetcher(_fake_dataset(), 1000, img_downsample=1) as frames:
        frames.get()
