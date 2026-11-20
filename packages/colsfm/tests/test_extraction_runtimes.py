"""Extraction holds one engine runtime at a time: a rig with several image sizes must not keep every context alive."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from mapping_helpers import FixedFrames
from torch import Tensor

from colsfm.catalog import CatalogCameraModel, frames_meta_document
from colsfm.database import create_database, read_keypoints
from colsfm.frames_meta import FRAMES_META_NAME, read_frames_meta

pytest.importorskip("tensorrt", reason="colsfm.features_native imports tensorrt at module scope")
pytest.importorskip("cuda.bindings.runtime", reason="colsfm.features_native allocates through cuda-python")

from colsfm.features_native import ExtractionGroup, ImageTask, image_tasks, run_native_extraction  # noqa: E402
from colsfm.features_raco import RACO_MODEL  # noqa: E402

NUM_KEYPOINTS: int = 3


class _CountingRuntime:
    """A fake engine runtime that records how many instances are open at once; only ``close()`` releases one."""

    alive: int = 0
    most_alive: int = 0
    opened: list[Path] = []

    def __init__(self, path: Path, width: int, height: int) -> None:
        type(self).alive += 1
        type(self).most_alive = max(type(self).most_alive, type(self).alive)
        type(self).opened.append(path)
        self.max_input_shapes: dict[str, tuple[int, ...]] = {"image": (8, 3, height, width)}

    def close(self) -> None:
        type(self).alive -= 1

    def __enter__(self) -> "_CountingRuntime":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __call__(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        batch: int = int(inputs["image"].shape[0])
        return {
            "keypoints": torch.zeros((batch, NUM_KEYPOINTS, 2)),
            "descriptors": torch.zeros((batch, NUM_KEYPOINTS, 128)),
            "scores": torch.ones((batch, NUM_KEYPOINTS)),
        }


def test_each_size_group_runs_on_its_own_runtime_and_releases_it_before_the_next(tmp_path: Path) -> None:
    k_matrix = np.asarray([[32.0, 0.0, 31.5], [0.0, 32.0, 23.5], [0.0, 0.0, 1.0]])
    shapes: list[tuple[str, int, int]] = [("wide", 64, 32), ("tall", 32, 64), ("square", 64, 64)]
    cameras: list[CatalogCameraModel] = [
        CatalogCameraModel(
            entity=f"/world/rig_00/cam_0{index}",
            name=name,
            rig_T_cam=np.eye(4),
            k_matrix=k_matrix,
            width=width,
            height=height,
            projection="PINHOLE",
            distortion=(),
        )
        for index, (name, width, height) in enumerate(shapes)
    ]
    document = frames_meta_document(cameras, [{name: (1_000, np.eye(4)) for name, _, _ in shapes}], ())
    (tmp_path / FRAMES_META_NAME).write_text(json.dumps(document))
    database_path: Path = tmp_path / "groups.db"
    create_database(database_path, read_frames_meta(tmp_path / FRAMES_META_NAME))
    tasks: list[ImageTask] = image_tasks(database_path, [f"{name}/1000" for name, _, _ in shapes])
    groups: list[ExtractionGroup] = [
        ExtractionGroup(
            engine_path=tmp_path / f"{task.name.split('/')[0]}.engine", tasks=[task], network_height=task.image_height, network_width=task.image_width
        )
        for task in tasks
    ]
    sizes: dict[str, tuple[int, int]] = {task.name: (task.image_width, task.image_height) for task in tasks}
    _CountingRuntime.alive, _CountingRuntime.most_alive, _CountingRuntime.opened = 0, 0, []

    run_native_extraction(
        database_path,
        groups,
        RACO_MODEL,
        FixedFrames(sizes, random=False),
        min_score=0.0,
        max_num_features=0,
        batch_size=8,
        open_runtime=lambda path: _CountingRuntime(path, *sizes[f"{path.stem}/1000"]),
    )

    assert _CountingRuntime.opened == [group.engine_path for group in groups]
    assert _CountingRuntime.most_alive == 1
    assert _CountingRuntime.alive == 0
    assert all(read_keypoints(database_path, task.image_id).shape == (NUM_KEYPOINTS, 2) for task in tasks)


def test_a_lower_pixel_budget_lowers_each_cube_face_engine_batch_and_the_default_is_unchanged() -> None:
    from colsfm.features_raco import ALIKED_IMAGE_BINDING, raco_dynamic_profile, select_raco_engine

    assert [select_raco_engine(size, "dynamic").maximum_batch_size for size in ((800, 800), (480, 800))] == [7, 8]
    assert [select_raco_engine(size, "dynamic", 1_920_000).maximum_batch_size for size in ((800, 800), (480, 800), (800, 480))] == [3, 5, 5]
    engine = select_raco_engine((800, 800), "dynamic", 1_920_000)
    assert raco_dynamic_profile((800, 800), engine.maximum_batch_size)[ALIKED_IMAGE_BINDING][2] == (3, 3, 800, 800)
    assert raco_dynamic_profile((800, 800)) == raco_dynamic_profile((800, 800), 7)
