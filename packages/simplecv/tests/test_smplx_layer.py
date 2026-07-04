from pathlib import Path

import numpy as np
import pytest
import torch
from jaxtyping import Float32
from numpy import ndarray

from simplecv.ops.smplx.smplx_torch import SmplxLayerTorch

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SMPLX_MODEL_ROOT: Path = PROJECT_ROOT / "simplecv" / "data" / "body_models"


def _have_smplx_model() -> bool:
    return (SMPLX_MODEL_ROOT / "smplx" / "SMPLX_NEUTRAL.npz").exists()


@pytest.mark.skipif(not _have_smplx_model(), reason="SMPL-X model file not available under simplecv/data/body_models/")
def test_smplx_layer_forward_shapes() -> None:
    betas: Float32[ndarray, "16"] = np.zeros(16, dtype=np.float32)
    layer: SmplxLayerTorch = SmplxLayerTorch(betas=betas, model_root_dir=SMPLX_MODEL_ROOT).eval()
    with torch.no_grad():
        verts, joints = layer(torch.zeros((2, 165), dtype=torch.float32), torch.zeros((2, 3), dtype=torch.float32))
    assert verts.shape == (2, 10475, 3)
    assert joints.shape[0] == 2 and joints.shape[2] == 3
    assert layer.faces.shape == (20908, 3)


@pytest.mark.skipif(not _have_smplx_model(), reason="SMPL-X model file not available under simplecv/data/body_models/")
def test_smplx_layer_rest_root_joint_matches_zero_pose_forward() -> None:
    betas: Float32[ndarray, "16"] = np.zeros(16, dtype=np.float32)
    layer: SmplxLayerTorch = SmplxLayerTorch(betas=betas, model_root_dir=SMPLX_MODEL_ROOT).eval()
    root_joint: Float32[ndarray, "3"] = layer.rest_root_joint()
    with torch.no_grad():
        joints = layer(torch.zeros((1, 165), dtype=torch.float32), torch.zeros((1, 3), dtype=torch.float32))[1]
    np.testing.assert_allclose(root_joint, joints[0, 0].numpy(), atol=1e-6)


def test_smplx_layer_missing_smpl_model_raises(tmp_path: Path) -> None:
    betas: Float32[ndarray, "10"] = np.zeros(10, dtype=np.float32)
    with pytest.raises(RuntimeError, match="No SMPL model file"):
        SmplxLayerTorch(betas=betas, model_type="smpl", model_root_dir=tmp_path)


def test_coco133_mapping_is_complete_and_semantically_consistent() -> None:
    from smplx.joint_names import JOINT_NAMES

    from simplecv.data.skeleton.coco_133 import COCO_133_ID2NAME
    from simplecv.ops.smplx.smplx_coco133 import _coco133_to_smplx_indices

    coco_ids, smplx_ids = _coco133_to_smplx_indices()
    assert len(coco_ids) == 133
    assert smplx_ids.max() < len(JOINT_NAMES)
    # Left/right sides must not cross: any coco name containing left/right maps
    # to a smplx name on the same side (face-N points are positional, skip them).
    for coco_id, smplx_id in zip(coco_ids, smplx_ids, strict=True):
        coco_name: str = COCO_133_ID2NAME[int(coco_id)]
        smplx_name: str = JOINT_NAMES[int(smplx_id)]
        if coco_name.startswith("face-"):
            continue
        for side, other in (("left", "right"), ("right", "left")):
            if coco_name.startswith(side):
                assert side in smplx_name and other not in smplx_name, f"{coco_name} -> {smplx_name}"
    # Midline chin: face-8 -> contour_middle.
    chin_pos = list(coco_ids).index(31)
    assert JOINT_NAMES[int(smplx_ids[chin_pos])] == "contour_middle"


def test_smplx_joints_to_coco133_without_contour_leaves_jawline_nan() -> None:
    from simplecv.ops.smplx.smplx_coco133 import smplx_joints_to_coco133_xyzc

    joints: Float32[ndarray, "n_frames n_joints 3"] = np.ones((2, 127, 3), dtype=np.float32)
    xyzc: Float32[ndarray, "n_frames 133 4"] = smplx_joints_to_coco133_xyzc(joints)
    assert xyzc.shape == (2, 133, 4)
    # face-0..face-16 (coco ids 23..39) need the contour joints (127..143).
    assert np.all(np.isnan(xyzc[:, 23:40, :3])) and np.all(xyzc[:, 23:40, 3] == 0.0)
    assert np.all(np.isfinite(xyzc[:, :23, :3])) and np.all(xyzc[:, 40:, 3] == 1.0)
