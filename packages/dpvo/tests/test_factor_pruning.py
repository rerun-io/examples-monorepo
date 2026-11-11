"""Factor pruning keeps edge state aligned and compacts each mask once."""

import pytest
from jaxtyping import Bool, Int64

torch = pytest.importorskip("torch", reason="DPVO requires PyTorch")
pytest.importorskip("lietorch_mojo", reason="DPVO requires lietorch_mojo")

from torch import Tensor  # noqa: E402

from dpvo.config import DPVOConfig  # noqa: E402
from dpvo.dpvo import DPVO  # noqa: E402
from dpvo.net import VONet  # noqa: E402


@pytest.mark.parametrize("store", [False, True])
def test_pruning_keeps_each_edges_state_and_order_with_one_compaction(store: bool) -> None:
    tracker: DPVO = DPVO(DPVOConfig(buffer_size=16, patches_per_frame=2), VONet(), ht=32, wd=32, device=torch.device("cpu"))
    tracker.pg.index_[:4] = torch.arange(4)[:, None]
    patches: Int64[Tensor, "4"] = torch.tensor([5, 2, 6, 1])
    targets: Int64[Tensor, "4"] = torch.tensor([0, 3, 1, 2])
    tracker.append_factors(patches, targets)
    tracker.pg.weight = torch.tensor([[[0.0, 0.5], [1.0, 1.5], [2.0, 2.5], [3.0, 3.5]]])
    tracker.pg.target = tracker.pg.weight + 10.0
    tracker.pg.net[:] = torch.arange(4)[None, :, None]
    remove: Bool[Tensor, "4"] = torch.tensor([False, True, False, True])
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        tracker.remove_factors(remove, store=store)
    assert tracker.pg.ii.tolist() == [2, 3]
    assert tracker.pg.jj.tolist() == [0, 1]
    assert tracker.pg.kk.tolist() == [5, 6]
    assert tracker.pg.weight.tolist() == [[[0.0, 0.5], [2.0, 2.5]]]
    assert tracker.pg.target.tolist() == [[[10.0, 10.5], [12.0, 12.5]]]
    assert tracker.pg.net[0, :, 0].tolist() == [0.0, 2.0]
    if store:
        assert tracker.pg.ii_inac.tolist() == [1, 0]
        assert tracker.pg.jj_inac.tolist() == [3, 2]
        assert tracker.pg.kk_inac.tolist() == [2, 1]
        assert tracker.pg.weight_inac.tolist() == [[[1.0, 1.5], [3.0, 3.5]]]
        assert tracker.pg.target_inac.tolist() == [[[11.0, 11.5], [13.0, 13.5]]]
    else:
        assert tracker.pg.ii_inac.numel() == 0
    nonzero_calls: int = sum(event.count for event in profile.key_averages() if event.key == "aten::nonzero")
    assert nonzero_calls <= 1 + int(store)


def test_motion_magnitude_selects_a_frame_pair_with_one_compaction() -> None:
    tracker: DPVO = DPVO(DPVOConfig(buffer_size=16, patches_per_frame=2), VONet(), ht=32, wd=32, device=torch.device("cpu"))
    tracker.pg.index_[:2] = torch.arange(2)[:, None]
    tracker.pg.poses_[1, 0] = 0.1
    tracker.pg.intrinsics_[:2] = torch.tensor([40.0, 40.0, 16.0, 16.0])
    tracker.pg.patches_[:2, :, 0:2] = 16.0
    tracker.pg.patches_[:2, :, 2] = 0.5
    tracker.append_factors(torch.tensor([0, 2, 1]), torch.tensor([1, 0, 1]))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        magnitude: float = tracker.motionmag(0, 1)
    # A 0.1 m translation at 2 m depth with fx=40 moves each point by 2 pixels.
    assert magnitude == pytest.approx(2.0)
    nonzero_calls: int = sum(event.count for event in profile.key_averages() if event.key == "aten::nonzero")
    assert nonzero_calls <= 1
