"""The viewer's hand mesh at a chosen detail: EgoHand sends SOMA's low level of detail, other models their own mesh."""

import pytest
import torch

pytest.importorskip("warp", reason="EgoHand needs the posekit-hands Pixi environment (posekit-egohand feature)")
pytest.importorskip("soma", reason="EgoHand needs the SOMA model package in the posekit-hands Pixi environment")

from posekit.models.egohand import SOMA_REVISION, EgoHandReconstruction, SomaLowMesh, soma_assets_path, soma_low_mesh
from posekit.predictions import Keypoints3d
from posekit.skeletons import SOMA_HAND


def _hands(low_mesh: dict | None) -> EgoHandReconstruction:
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
        scale_parameters=None, low_mesh=low_mesh,
    )


def test_low_detail_keeps_the_low_vertices_of_each_side_and_its_triangles() -> None:
    low = {
        "left": SomaLowMesh(vertex_ids=torch.tensor([0, 1, 2]), faces=torch.tensor([[0, 1, 2]])),
        "right": SomaLowMesh(vertex_ids=torch.tensor([3, 4, 5]), faces=torch.tensor([[0, 2, 1]])),
    }
    hands = _hands(low)
    coarse = hands.at_mesh_detail("low")
    assert coarse.vertices is not None and coarse.faces is not None
    assert coarse.vertices[:, :, 0].tolist() == [[0.0, 1.0, 2.0], [103.0, 104.0, 105.0]]
    assert coarse.faces["right"].tolist() == [[0, 2, 1]]
    assert coarse.keypoints is hands.keypoints  # the joints are untouched
    assert hands.at_mesh_detail("full") is hands
    unchanged = _hands(None).at_mesh_detail("low")  # no low mesh: the model's own
    assert unchanged.vertices is not None and unchanged.vertices.shape == (2, 6, 3)


@pytest.mark.integration
def test_soma_low_mesh_is_the_first_718_vertices_with_1422_triangles_per_side() -> None:
    try:
        soma = soma_assets_path(None, "nvidia/SOMA-X", SOMA_REVISION)
    except OSError as error:
        pytest.skip(f"SOMA hand assets (nvidia/SOMA-X) not available: {error}")
    for side, mesh in soma_low_mesh(soma).items():
        assert mesh.vertex_ids.tolist() == list(range(718)), side
        assert tuple(mesh.faces.shape) == (1422, 3) and int(mesh.faces.max()) == 717, side
