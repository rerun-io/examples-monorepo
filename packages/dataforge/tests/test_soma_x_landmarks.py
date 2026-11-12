"""SOMA-X hands fitted to UmeTrack landmarks: the written layer's rows meet the hand bar and its parameters regenerate its mesh."""

import os
from pathlib import Path

import numpy as np
import pytest
import rerun.chunk as rrc

pytest.importorskip("soma", reason="SOMA-X is installed only in the dataforge-soma-x environments")
torch = pytest.importorskip("torch")

from dataforge import soma_x  # noqa: E402
from dataforge.apis import soma_x as api  # noqa: E402

pytestmark = pytest.mark.integration

TREE: Path = Path(os.environ.get("DATAFORGE_UMETRACK_TREE", "/mnt/nas/datasets/dataforge-umetrack-rrd"))
SEGMENT: str = "umetrack__real__hand_hand__testing__user_05__recording_00"


def test_umetrack_landmark_fit_writes_trusted_hands_whose_parameters_regenerate_the_mesh(tmp_path: Path) -> None:
    if not (TREE / "hand_pose" / f"{SEGMENT}.rrd").is_file():
        pytest.skip(f"requires the UmeTrack tree's hand_pose layer for {SEGMENT} (DATAFORGE_UMETRACK_TREE)")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA on the host")
    config = api.Config(dataset="umetrack", tree=TREE, output_dir=tmp_path, max_seconds=3.0)
    fitters = api.Fitters(hand_bar_mm=3.0, body_bar_mm=8.0)
    errors = api.write_segment(config, fitters, SEGMENT)
    assert set(errors) == {"/world/gt/hands/left", "/world/gt/hands/right"}
    for root, error in errors.items():
        assert len(error) > 50, root
        assert float(np.median(error)) < 2.5, root
        assert float(error.max()) <= 3.0, root

    store = rrc.ChunkStore.from_chunks(list(rrc.RrdReader(tmp_path / "soma_x" / f"{SEGMENT}.rrd").stream()))
    root = "/world/gt/hands/right"
    params = store.reader(index="video_time", contents=f"{soma_x.params_path(root)}/**").to_arrow_table()
    mesh = store.reader(index="video_time", contents=f"{soma_x.mesh_path(root)}/**").to_arrow_table()
    row = 10
    rotations = np.asarray(params.column(f"{soma_x.params_path(root)}:rotations")[row].as_py(), dtype=np.float32).reshape(1, 25, 3)
    translation = np.asarray(params.column(f"{soma_x.params_path(root)}:root_translation")[row].as_py(), dtype=np.float32).reshape(1, 3)
    identity = np.asarray(params.column(f"{soma_x.params_path(root)}:identity_coeffs")[row].as_py(), dtype=np.float32).reshape(1, 10)
    logged = np.asarray(mesh.column(f"{soma_x.mesh_path(root)}:Mesh3D:vertex_positions")[row].as_py(), dtype=np.float32)
    regenerated = fitters.hand("right").pose(identity, rotations, translation)[0]
    np.testing.assert_allclose(regenerated, logged, atol=1e-5)
