"""Classical (DBoW) loop closure end to end on a KITTI 00 slice: retrieval finds the loop, the pose graph is solved
(`dpvo.fastba.solve_system`, in loop closure's worker process) and the result is applied to the tracker.

Raw KITTI 00 frames 0 to 2,660 at stride 2 hold the sequence's first revisit (keyframes near frames 400 and 2,580);
a slice that starts at frame 300 finds no closure. Before this test no gate called solve_system; the run also covers the
retrieval queues, which deadlocked after the first PGO when their backlog exceeded both queues (this slice hangs on them).
"""

import faulthandler
from pathlib import Path

import numpy as np
import pytest
import rerun as rr

torch = pytest.importorskip("torch", reason="DPVO requires PyTorch")
pytest.importorskip("dpretrieval", reason="classic loop closure needs dpretrieval (DBoW2)")

from dpvo.api.inference import DPVOPipelineHandle, run_dpvo_pipeline  # noqa: E402
from dpvo.config import DPVOConfig  # noqa: E402
from dpvo.loop_closure.long_term import LongTermLoopClosure  # noqa: E402

pytestmark = pytest.mark.integration

PACKAGE_DIR: Path = Path(__file__).resolve().parents[1]
KITTI_00: Path = Path("/mnt/nas/ssd-8tb/metrictrail/data/KITTI/rgb_raw/00")
CHECKPOINT: Path = PACKAGE_DIR / "checkpoints" / "dpvo.pth"
VOCABULARY: Path = PACKAGE_DIR / "ORBvoc.txt"


def test_kitti_00_slice_closes_its_loop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device for the DPVO kernels")
    for asset in (KITTI_00, CHECKPOINT, VOCABULARY):
        if not asset.exists():
            pytest.skip(f"missing {asset} (KITTI 00 raw frames on the NAS; pixi tasks _download-dpvo-model, _dpvo-download-orb-vocab)")
    monkeypatch.chdir(PACKAGE_DIR)  # RetrievalDBOW reads ORBvoc.txt from the working directory
    closed: list[bool] = []
    applied: list[int] = []
    close_loop = LongTermLoopClosure.close_loop
    rescale = LongTermLoopClosure._rescale_deltas

    def recording_close_loop(self: LongTermLoopClosure, i: int, j: int, n: int) -> bool:
        closed.append(close_loop(self, i, j, n))
        return closed[-1]

    def recording_rescale(self: LongTermLoopClosure, s: torch.Tensor) -> None:
        applied.append(int(s.shape[0]))  # runs only when a PGO result is applied to the tracker
        rescale(self, s)

    monkeypatch.setattr(LongTermLoopClosure, "close_loop", recording_close_loop)
    monkeypatch.setattr(LongTermLoopClosure, "_rescale_deltas", recording_rescale)
    rr.init("dpvo-classic-loop-closure-test")  # the pipeline logs to the global stream
    rr.save(tmp_path / "run.rrd")
    torch.manual_seed(1234)
    faulthandler.dump_traceback_later(600, exit=True)  # the deadlock this guards hangs instead of failing (~105 s on a 5090)
    handle = DPVOPipelineHandle()
    for _ in run_dpvo_pipeline(dpvo_config=DPVOConfig.slam_classic_fast(), network_path=str(CHECKPOINT), imagedir=str(KITTI_00),
                               stride=2, max_frames=1330, handle=handle, device=torch.device("cuda")):  # fmt: skip
        pass
    faulthandler.cancel_dump_traceback_later()
    assert any(closed), f"no classical loop closure accepted ({len(closed)} attempts)"
    assert applied, "a loop closure was accepted but its pose-graph result never reached the tracker"
    assert handle.prediction is not None and np.isfinite(handle.prediction.final_poses).all()
