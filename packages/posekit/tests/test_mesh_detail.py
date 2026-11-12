"""The viewer's hand mesh at a chosen detail: EgoHand sends one of SOMA's coarser levels (low, xlo), other models their own mesh."""

import pytest
import torch

pytest.importorskip("warp", reason="EgoHand needs the posekit-hands Pixi environment (posekit-egohand feature)")
pytest.importorskip("soma", reason="EgoHand needs the SOMA model package in the posekit-hands Pixi environment")

from pathlib import Path

from posekit.models import egohand
from posekit.models.egohand import SOMA_REVISION, EgoHandReconstruction, SomaMeshLod, soma_assets_path, soma_mesh_lod
from posekit.predictions import Keypoints3d, MeshDetail
from posekit.skeletons import SOMA_HAND


def _hands() -> EgoHandReconstruction:
    """A left and a right hand with 6-vertex meshes whose vertex v sits at x = v (+100 on the right hand)."""
    n, k = 2, SOMA_HAND.num_keypoints
    vertices = torch.zeros((n, 6, 3))
    vertices[:, :, 0] = torch.arange(6.0) + torch.tensor([[0.0], [100.0]])
    return EgoHandReconstruction(
        keypoints=Keypoints3d(torch.zeros((n, k, 2)), torch.zeros((n, k)), torch.ones((n, k)), torch.tensor([0, 0]), SOMA_HAND, (0,), torch.zeros((n, k, 3))),
        sides=("left", "right"),
        vertices=vertices,
        faces={"left": torch.tensor([[0, 1, 2], [3, 4, 5]]), "right": torch.tensor([[0, 2, 1], [3, 5, 4]])},
        rotations=torch.zeros((n, k, 3, 3)), identity=torch.zeros((n, 5)), scale=torch.ones(n), translation=torch.zeros((n, 3)),
        scale_parameters=None, soma=Path("/unused"),
    )


def test_a_coarser_level_maps_each_side_with_its_own_rule_and_takes_its_triangles(monkeypatch: pytest.MonkeyPatch) -> None:
    levels = {
        "low": {
            "left": SomaMeshLod(torch.tensor([[0, 1, 2]]), from_mid=lambda mid: mid[:, :3]),
            "right": SomaMeshLod(torch.tensor([[0, 2, 1]]), from_mid=lambda mid: mid[:, 3:]),
        },
        # an xlo-like interpolation: 3 vertices from 6
        "xlo": {side: SomaMeshLod(torch.tensor([[0, 1, 2]]), from_mid=lambda mid: (mid[:, 0::2] + mid[:, 1::2]) / 2) for side in ("left", "right")},
    }
    monkeypatch.setattr(egohand, "soma_mesh_lod", lambda soma, detail: levels[detail])
    hands = _hands()
    low = hands.at_mesh_detail("low")
    assert low.vertices is not None and low.faces is not None
    assert low.vertices[:, :, 0].tolist() == [[0.0, 1.0, 2.0], [103.0, 104.0, 105.0]]
    assert low.faces["right"].tolist() == [[0, 2, 1]]
    assert low.keypoints is hands.keypoints  # the joints are untouched
    xlo = hands.at_mesh_detail("xlo")
    assert xlo.vertices is not None and xlo.vertices[:, :, 0].tolist() == [[0.5, 2.5, 4.5], [100.5, 102.5, 104.5]]
    assert hands.at_mesh_detail("full") is hands


@pytest.mark.integration
@pytest.mark.parametrize(("detail", "vertices", "triangles"), [("low", 718, 1422), ("xlo", 134, 258)])
def test_soma_levels_of_detail_have_their_vertex_and_triangle_counts(detail: MeshDetail, vertices: int, triangles: int) -> None:
    try:
        soma = soma_assets_path(None, "nvidia/SOMA-X", SOMA_REVISION)
    except OSError as error:
        pytest.skip(f"SOMA hand assets (nvidia/SOMA-X) not available: {error}")
    if not torch.cuda.is_available():
        pytest.skip("SOMA levels of detail are kept on CUDA")
    mid = torch.rand((2, 2859, 3), device="cuda")
    for side, mesh in soma_mesh_lod(soma, detail).items():
        assert tuple(mesh.from_mid(mid).shape) == (2, vertices, 3), side
        assert tuple(mesh.faces.shape) == (triangles, 3) and int(mesh.faces.max()) == vertices - 1, side
