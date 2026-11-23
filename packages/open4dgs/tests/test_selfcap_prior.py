"""The SelfCap ``--stage prior`` layer carries the clip's camera grid, so the trainer and ``--stage init`` accept it.

MoGe-2 is replaced by a constant predictor and the catalog read by a synthetic ROI clip; the stage itself (compute,
write) runs as in production on the GPU, with registration off.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("gsplat", reason="the prior modules import gsplat (open4dgs envs)")
pytestmark = pytest.mark.integration

NAME: str = "rig_00/cam_00"


def _roi_clip(pad: float):
    from open4dgs.catalog_clip import ClipRoi, MultiViewClip
    from open4dgs.roi_cameras import fit_virtual_camera

    source_K = np.array([[500.0, 0.0, 319.5], [0.0, 500.0, 239.5], [0.0, 0.0, 1.0]])
    points = np.array([[0.3, 0.1, 4.0], [0.6, 0.5, 4.2], [0.4, -0.2, 3.9]], dtype=np.float32)
    camera = fit_virtual_camera(points, source_K, np.eye(4), (640, 480), pad=pad)
    width, height = camera.size
    return MultiViewClip(
        camera_names=(NAME,),
        frames=(torch.zeros(2, height, width, 3, dtype=torch.uint8),),
        valid=(torch.ones(height, width, dtype=torch.bool),),
        K=torch.from_numpy(camera.K).float()[None],
        world_T_cam=torch.eye(4)[None],
        fps=30.0,
        frame_indices=torch.tensor([10, 11]),
        people=None,
        stored_sizes=((640, 480),),
        lenses=(None,),
        roi=ClipRoi(pad=pad, max_side=None, cameras=(camera,)),
    )


class _ConstantMoGe:
    """Stands in for both MoGe-2 predictors: depth 2 m, normals facing the camera, everything valid."""

    def __init__(self, **_kwargs: object) -> None:
        pass

    def serves(self, input_hw: tuple[int, int]) -> bool:
        del input_hw
        return True

    def predict_geometry(self, batch: torch.Tensor, fov_x: float) -> types.SimpleNamespace:
        del fov_x
        n, h, w = batch.shape[0], batch.shape[1], batch.shape[2]
        normals = torch.zeros(n, h, w, 3, device=batch.device)
        normals[..., 2] = -1.0
        ones = torch.ones(n, h, w, device=batch.device)
        return types.SimpleNamespace(depth_bhw=2.0 * ones, mask_bhw=ones, normals_bhw3=normals)


def test_selfcap_roi_prior_layer_records_the_grid_the_reader_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("the prior stage runs MoGe-2's batches on cuda")
    from rerun.chunk import RrdReader

    from open4dgs.apis import selfcap
    from open4dgs.prior_layer import PRIOR_LAYER, check_roi
    from open4dgs.properties import parse_roi_cameras, rrd_cells
    from open4dgs.selfcap_clip import TimedClip

    clip = _roi_clip(0.15)
    timed = TimedClip(clip=clip, times_s=np.array([[10 / 30.0, 11 / 30.0]]), joints=None)
    monkeypatch.setattr(selfcap, "read_timed_clip", lambda *_args, **_kwargs: timed)
    fake = types.ModuleType("monopriors.models.moge_v2.predictor")
    fake.MoGeV2TrtPredictor = fake.MoGeV2TorchPredictor = _ConstantMoGe  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "monopriors.models.moge_v2.predictor", fake)

    selfcap.main(selfcap.Config(stage="prior", roi=True, register=False, output_root=tmp_path, png_dir=None, num_frames=2))

    written = sorted((tmp_path / "prior" / selfcap.COMPANION_DATASET).glob(f"*.{PRIOR_LAYER}*.rrd"))
    assert len(written) == 1
    cells = rrd_cells(RrdReader(written[0]), PRIOR_LAYER)
    assert "roi_cameras" in cells, "the layer records no roi_cameras: read_clip_priors would treat it as real cameras"
    stored = parse_roi_cameras(str(cells["roi_cameras"][0]))
    check_roi(clip, (NAME,), stored, "selfcap roi layer")  # --stage init --roi and the trainer read it back
    with pytest.raises(ValueError, match="grid"):
        check_roi(_roi_clip(0.3), (NAME,), stored, "another pad")
