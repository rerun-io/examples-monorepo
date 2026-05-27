import pickle
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from jaxtyping import Float32
from numpy import ndarray
from rerun.experimental import RrdReader

from egoexo_forge.api.sam3d_body_mesh import _load_mesh_faces
from egoexo_forge.api.sam3d_body_parquet_to_rrd import (
    Sam3dBodyParquetToRrdConfig,
    convert_sam3d_body_parquet_to_rrd,
)

_TINY_PNG: bytes = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de0000000c49444154789c6360f8cf00000301010018dd8db00000000049454e44ae426082"
)
_PNG_HEADER: bytes = bytes.fromhex("89504e470d0a1a0a")


class _FakeMhrModel(torch.nn.Module):
    def forward(self, shape_params: torch.Tensor, model_params: torch.Tensor, expr_params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size: int = int(shape_params.shape[0])
        base_verts_cm: torch.Tensor = torch.tensor(
            [[[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 100.0, 0.0]]],
            dtype=torch.float32,
        ).repeat(batch_size, 1, 1)
        translation_cm: torch.Tensor = model_params[:, :3].reshape(batch_size, 1, 3) * 10.0
        verts: torch.Tensor = base_verts_cm + translation_cm
        skeleton_state: torch.Tensor = torch.zeros((batch_size, 127, 8), dtype=torch.float32)
        return verts, skeleton_state


def _write_fake_mhr_assets(mhr_model_path: Path, checkpoint_path: Path) -> None:
    traced_model: Any = torch.jit.trace(
        _FakeMhrModel(),
        (
            torch.zeros((1, 45), dtype=torch.float32),
            torch.zeros((1, 204), dtype=torch.float32),
            torch.zeros((1, 72), dtype=torch.float32),
        ),
    )
    traced_model.save(str(mhr_model_path))
    torch.save({"state_dict": {"head_pose.faces": torch.tensor([[0, 1, 2]], dtype=torch.int64)}}, checkpoint_path)


def _write_bbox_parquet(
    path: Path,
    *,
    dataset: str = "coco",
    bbox_format: str = "xywh",
    image_name: str = "COCO_train2014_000000000001.jpg",
) -> None:
    keypoints_2d: Float32[ndarray, "kpts channels"] = np.zeros((70, 3), dtype=np.float32)
    keypoints_2d[:, 0] = np.linspace(20.0, 180.0, 70, dtype=np.float32)
    keypoints_2d[:, 1] = np.linspace(30.0, 210.0, 70, dtype=np.float32)
    keypoints_2d[:, 2] = 0.9

    keypoints_3d: Float32[ndarray, "kpts channels"] = np.zeros((70, 4), dtype=np.float32)
    keypoints_3d[:, 0] = np.linspace(-0.4, 0.4, 70, dtype=np.float32)
    keypoints_3d[:, 1] = np.linspace(-0.2, 0.2, 70, dtype=np.float32)
    keypoints_3d[:, 2] = np.linspace(1.0, 1.8, 70, dtype=np.float32)
    keypoints_3d[:, 3] = 0.8

    cam_int: Float32[ndarray, "3 3"] = np.array(
        [
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    rows: list[dict[str, object]] = [
        {
            "dataset": dataset,
            "image": image_name,
            "bbox": [10.0, 20.0, 120.0, 180.0],
            "bbox_format": bbox_format,
            "person_id": 7,
            "cam_int": cam_int.tolist(),
            "keypoints_2d": keypoints_2d.tolist(),
            "keypoints_3d": keypoints_3d.tolist(),
            "model_params": np.concatenate(
                [np.array([10.0, 20.0, 30.0], dtype=np.float32), np.linspace(-0.1, 0.1, 201, dtype=np.float32)]
            ).tolist(),
            "shape_params": np.linspace(-0.01, 0.01, 45, dtype=np.float32).tolist(),
        },
        {
            "dataset": "coco",
            "image": "COCO_train2014_000000000002.jpg",
            "bbox": [0.0, 0.0, 2.0, 4.0],
            "bbox_format": bbox_format,
            "person_id": 8,
            "cam_int": cam_int.tolist(),
            "keypoints_2d": keypoints_2d.tolist(),
            "keypoints_3d": keypoints_3d.tolist(),
            "model_params": np.concatenate(
                [np.array([1.0, 2.0, 3.0], dtype=np.float32), np.linspace(-0.2, 0.2, 201, dtype=np.float32)]
            ).tolist(),
            "shape_params": np.linspace(-0.02, 0.02, 45, dtype=np.float32).tolist(),
        },
    ]
    table: pa.Table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def _component_values(rrd_path: Path, entity_path: str, component_name: str) -> list[object]:
    store = RrdReader(rrd_path).store()
    for chunk in store.stream().to_chunks():
        if str(chunk.entity_path) == entity_path:
            values: list[object] | None = chunk.to_record_batch().to_pydict().get(component_name)
            if values is not None:
                return values
    raise AssertionError(f"Missing {component_name} at {entity_path}")


def test_convert_sam3d_body_parquet_to_rrd_derives_boxes2d_from_xywh_chunks(tmp_path: Path) -> None:
    parquet_path: Path = tmp_path / "000000.parquet"
    rrd_path: Path = tmp_path / "sam3d-body.rrd"
    _write_bbox_parquet(parquet_path)

    result = convert_sam3d_body_parquet_to_rrd(
        Sam3dBodyParquetToRrdConfig(
            parquet_path=parquet_path,
            rrd_path=rrd_path,
            parquet_reader_summary=False,
        )
    )

    assert result.row_count == 2
    assert result.rrd_path == rrd_path

    summary: str = RrdReader(rrd_path).store().summary()
    assert "/world/cam/pinhole/image/pred/bbox rows=2 static=False timelines=['row_index']" in summary
    assert "Boxes2D:centers" in summary
    assert "Boxes2D:half_sizes" in summary
    assert "/world/cam/pinhole rows=2 static=False timelines=['row_index']" in summary
    assert "Pinhole:image_from_camera" in summary
    assert "Pinhole:resolution" in summary
    assert "Pinhole:camera_xyz" in summary
    assert "/world/cam/pinhole/image rows=2 static=False timelines=['row_index']" in summary
    assert "EncodedImage:blob" in summary
    assert "/world/cam/pinhole/image/pred/mhr70_uv rows=2 static=False timelines=['row_index']" in summary
    assert "Points2D:positions" in summary
    assert "/world/gt/mhr70_xyz rows=2 static=False timelines=['row_index']" in summary
    assert "Points3D:positions" in summary
    assert "/source/sam3d_body/parquet/bbox" not in summary

    assert "/world/cam/pinhole/pred/bbox" not in summary
    assert "/world/cam/pinhole/pred/mhr70_uv" not in summary

    centers: list[object] = _component_values(rrd_path, "/world/cam/pinhole/image/pred/bbox", "Boxes2D:centers")
    half_sizes: list[object] = _component_values(rrd_path, "/world/cam/pinhole/image/pred/bbox", "Boxes2D:half_sizes")
    centers_array: Float32[ndarray, "rows boxes xy"] = np.asarray(centers, dtype=np.float32)
    half_sizes_array: Float32[ndarray, "rows boxes xy"] = np.asarray(half_sizes, dtype=np.float32)

    np.testing.assert_allclose(centers_array[:, 0, :], np.asarray([[70.0, 110.0], [1.0, 2.0]], dtype=np.float32))
    np.testing.assert_allclose(half_sizes_array[:, 0, :], np.asarray([[60.0, 90.0], [1.0, 2.0]], dtype=np.float32))
    keypoints_2d: list[object] = _component_values(rrd_path, "/world/cam/pinhole/image/pred/mhr70_uv", "Points2D:positions")
    keypoints_3d: list[object] = _component_values(rrd_path, "/world/gt/mhr70_xyz", "Points3D:positions")
    first_uv_points: list[object] = cast(list[object], keypoints_2d[0])
    first_xyz_points: list[object] = cast(list[object], keypoints_3d[0])
    assert len(first_uv_points) == 70
    assert len(first_xyz_points) == 70
    first_uv: Float32[ndarray, "xy"] = np.asarray(first_uv_points[0], dtype=np.float32)
    first_xyz: Float32[ndarray, "xyz"] = np.asarray(first_xyz_points[0], dtype=np.float32)
    np.testing.assert_allclose(first_uv, np.asarray([20.0, 30.0], dtype=np.float32))
    np.testing.assert_allclose(first_xyz, np.asarray([-0.4, -0.2, 1.0], dtype=np.float32))
    resolution_values: list[object] = _component_values(rrd_path, "/world/cam/pinhole", "Pinhole:resolution")
    first_resolution: Float32[ndarray, "xy"] = np.asarray(resolution_values[0], dtype=np.float32).reshape(-1)
    np.testing.assert_allclose(first_resolution, np.asarray([640.0, 480.0], dtype=np.float32))
    camera_xyz_values: list[object] = _component_values(rrd_path, "/world/cam/pinhole", "Pinhole:camera_xyz")
    first_camera_xyz: np.ndarray = np.asarray(camera_xyz_values[0], dtype=np.uint8).reshape(-1)
    np.testing.assert_array_equal(first_camera_xyz, np.asarray([3, 2, 5], dtype=np.uint8))
    image_blobs: list[object] = _component_values(rrd_path, "/world/cam/pinhole/image", "EncodedImage:blob")
    first_image_blob_batches: list[object] = cast(list[object], image_blobs[0])
    first_image_blob: list[int] = cast(list[int], first_image_blob_batches[0])
    assert len(bytes(first_image_blob)) > 0


def test_convert_sam3d_body_parquet_to_rrd_can_preserve_source_columns(tmp_path: Path) -> None:
    parquet_path: Path = tmp_path / "000000.parquet"
    rrd_path: Path = tmp_path / "sam3d-body-with-source.rrd"
    _write_bbox_parquet(parquet_path)

    convert_sam3d_body_parquet_to_rrd(
        Sam3dBodyParquetToRrdConfig(
            parquet_path=parquet_path,
            rrd_path=rrd_path,
            include_source_columns=True,
            parquet_reader_summary=False,
        )
    )

    summary: str = RrdReader(rrd_path).store().summary()
    assert "/source/sam3d_body/parquet/bbox rows=2 static=False timelines=['row_index']" in summary
    assert "/source/sam3d_body/parquet/person_id rows=2 static=False timelines=['row_index']" in summary
    assert "/world/cam/pinhole/image/pred/bbox rows=2 static=False timelines=['row_index']" in summary


def test_convert_sam3d_body_parquet_to_rrd_reads_images_colocated_with_parquet_by_default(tmp_path: Path) -> None:
    parquet_path: Path = tmp_path / "000000.parquet"
    rrd_path: Path = tmp_path / "sam3d-body-with-image.rrd"
    image_name: str = "COCO_train2014_000000000001.png"
    image_path: Path = tmp_path / "train2014" / image_name
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(_TINY_PNG)
    _write_bbox_parquet(parquet_path, image_name=image_name)

    convert_sam3d_body_parquet_to_rrd(
        Sam3dBodyParquetToRrdConfig(
            parquet_path=parquet_path,
            rrd_path=rrd_path,
            parquet_reader_summary=False,
        )
    )

    image_blobs: list[object] = _component_values(rrd_path, "/world/cam/pinhole/image", "EncodedImage:blob")
    first_image_blob_batches: list[object] = cast(list[object], image_blobs[0])
    first_image_blob: list[int] = cast(list[int], first_image_blob_batches[0])
    assert bytes(first_image_blob) == _TINY_PNG


def test_convert_sam3d_body_parquet_to_rrd_reads_uppercase_coco_dataset_images(tmp_path: Path) -> None:
    parquet_path: Path = tmp_path / "000000.parquet"
    rrd_path: Path = tmp_path / "sam3d-body-uppercase-coco.rrd"
    image_name: str = "COCO_train2014_000000000001.png"
    image_path: Path = tmp_path / "train2014" / image_name
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(_TINY_PNG)
    _write_bbox_parquet(parquet_path, dataset="COCO", image_name=image_name)

    convert_sam3d_body_parquet_to_rrd(
        Sam3dBodyParquetToRrdConfig(
            parquet_path=parquet_path,
            rrd_path=rrd_path,
            parquet_reader_summary=False,
        )
    )

    image_blobs: list[object] = _component_values(rrd_path, "/world/cam/pinhole/image", "EncodedImage:blob")
    first_image_blob_batches: list[object] = cast(list[object], image_blobs[0])
    first_image_blob: list[int] = cast(list[int], first_image_blob_batches[0])
    assert bytes(first_image_blob) == _TINY_PNG


def test_convert_sam3d_body_parquet_to_rrd_reads_images_next_to_coco_split_dir_by_default(tmp_path: Path) -> None:
    parquet_path: Path = tmp_path / "coco_train" / "000000.parquet"
    rrd_path: Path = tmp_path / "sam3d-body-with-coco-image.rrd"
    image_name: str = "COCO_train2014_000000000001.png"
    image_path: Path = tmp_path / "train2014" / image_name
    parquet_path.parent.mkdir(parents=True)
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(_TINY_PNG)
    _write_bbox_parquet(parquet_path, image_name=image_name)

    convert_sam3d_body_parquet_to_rrd(
        Sam3dBodyParquetToRrdConfig(
            parquet_path=parquet_path,
            rrd_path=rrd_path,
            parquet_reader_summary=False,
        )
    )

    image_blobs: list[object] = _component_values(rrd_path, "/world/cam/pinhole/image", "EncodedImage:blob")
    first_image_blob_batches: list[object] = cast(list[object], image_blobs[0])
    first_image_blob: list[int] = cast(list[int], first_image_blob_batches[0])
    assert bytes(first_image_blob) == _TINY_PNG


def test_convert_sam3d_body_parquet_to_rrd_ignores_invalid_image_files(tmp_path: Path) -> None:
    parquet_path: Path = tmp_path / "coco_train" / "000000.parquet"
    rrd_path: Path = tmp_path / "sam3d-body-with-invalid-image.rrd"
    image_name: str = "COCO_train2014_000000000001.jpg"
    image_path: Path = tmp_path / "train2014" / image_name
    parquet_path.parent.mkdir(parents=True)
    image_path.parent.mkdir(parents=True)
    image_path.write_text("<Error>NoSuchKey</Error>")
    _write_bbox_parquet(parquet_path, image_name=image_name)

    convert_sam3d_body_parquet_to_rrd(
        Sam3dBodyParquetToRrdConfig(
            parquet_path=parquet_path,
            rrd_path=rrd_path,
            parquet_reader_summary=False,
        )
    )

    image_blobs: list[object] = _component_values(rrd_path, "/world/cam/pinhole/image", "EncodedImage:blob")
    media_types: list[object] = _component_values(rrd_path, "/world/cam/pinhole/image", "EncodedImage:media_type")
    first_image_blob_batches: list[object] = cast(list[object], image_blobs[0])
    first_image_blob: list[int] = cast(list[int], first_image_blob_batches[0])
    first_media_type_batches: list[object] = cast(list[object], media_types[0])
    assert bytes(first_image_blob).startswith(_PNG_HEADER)
    assert first_media_type_batches[0] == "image/png"


def test_convert_sam3d_body_parquet_to_rrd_derives_boxes2d_from_xyxy_chunks(tmp_path: Path) -> None:
    parquet_path: Path = tmp_path / "000000.parquet"
    rrd_path: Path = tmp_path / "sam3d-body.rrd"
    _write_bbox_parquet(parquet_path, bbox_format="xyxy")

    convert_sam3d_body_parquet_to_rrd(
        Sam3dBodyParquetToRrdConfig(
            parquet_path=parquet_path,
            rrd_path=rrd_path,
            parquet_reader_summary=False,
        )
    )

    centers: list[object] = _component_values(rrd_path, "/world/cam/pinhole/image/pred/bbox", "Boxes2D:centers")
    half_sizes: list[object] = _component_values(rrd_path, "/world/cam/pinhole/image/pred/bbox", "Boxes2D:half_sizes")
    centers_array: Float32[ndarray, "rows boxes xy"] = np.asarray(centers, dtype=np.float32)
    half_sizes_array: Float32[ndarray, "rows boxes xy"] = np.asarray(half_sizes, dtype=np.float32)

    np.testing.assert_allclose(centers_array[:, 0, :], np.asarray([[65.0, 100.0], [1.0, 2.0]], dtype=np.float32))
    np.testing.assert_allclose(half_sizes_array[:, 0, :], np.asarray([[55.0, 80.0], [1.0, 2.0]], dtype=np.float32))


def test_convert_sam3d_body_parquet_to_rrd_rejects_unsupported_bbox_format(tmp_path: Path) -> None:
    parquet_path: Path = tmp_path / "000000.parquet"
    rrd_path: Path = tmp_path / "sam3d-body.rrd"
    _write_bbox_parquet(parquet_path, bbox_format="unsupported")

    with pytest.raises(ValueError, match="Only bbox_format values"):
        convert_sam3d_body_parquet_to_rrd(
            Sam3dBodyParquetToRrdConfig(
                parquet_path=parquet_path,
                rrd_path=rrd_path,
                parquet_reader_summary=False,
            )
        )


def test_convert_sam3d_body_parquet_to_rrd_rejects_missing_image_column_with_clear_error(tmp_path: Path) -> None:
    parquet_path: Path = tmp_path / "000000.parquet"
    rrd_path: Path = tmp_path / "sam3d-body.rrd"
    _write_bbox_parquet(parquet_path)
    table_without_image: pa.Table = pq.read_table(parquet_path).drop_columns(["image"])
    pq.write_table(table_without_image, parquet_path)

    with pytest.raises(ValueError, match=r"SAM-3D-Body parquet is missing required columns: \['image'\]"):
        convert_sam3d_body_parquet_to_rrd(
            Sam3dBodyParquetToRrdConfig(
                parquet_path=parquet_path,
                rrd_path=rrd_path,
                parquet_reader_summary=False,
            )
        )


def test_convert_sam3d_body_parquet_to_rrd_rejects_non_coco_rows(tmp_path: Path) -> None:
    parquet_path: Path = tmp_path / "000000.parquet"
    rrd_path: Path = tmp_path / "sam3d-body.rrd"
    _write_bbox_parquet(parquet_path, dataset="mpii", image_name="000001.jpg")

    with pytest.raises(ValueError, match="Only dataset='coco'"):
        convert_sam3d_body_parquet_to_rrd(
            Sam3dBodyParquetToRrdConfig(
                parquet_path=parquet_path,
                rrd_path=rrd_path,
                parquet_reader_summary=False,
            )
        )


def test_convert_sam3d_body_parquet_to_rrd_logs_mhr_mesh_when_assets_are_available(tmp_path: Path) -> None:
    parquet_path: Path = tmp_path / "000000.parquet"
    rrd_path: Path = tmp_path / "sam3d-body-mesh.rrd"
    mhr_model_path: Path = tmp_path / "mhr_model.pt"
    checkpoint_path: Path = tmp_path / "model.ckpt"
    _write_bbox_parquet(parquet_path)
    _write_fake_mhr_assets(mhr_model_path=mhr_model_path, checkpoint_path=checkpoint_path)

    convert_sam3d_body_parquet_to_rrd(
        Sam3dBodyParquetToRrdConfig(
            parquet_path=parquet_path,
            rrd_path=rrd_path,
            mhr_model_path=mhr_model_path,
            sam3d_body_checkpoint_path=checkpoint_path,
            require_mesh=True,
            parquet_reader_summary=False,
        )
    )

    summary: str = RrdReader(rrd_path).store().summary()
    assert "/world/gt/mhr_mesh rows=2 static=False timelines=['row_index']" in summary
    assert "Mesh3D:vertex_positions" in summary
    vertices: list[object] = _component_values(rrd_path, "/world/gt/mhr_mesh", "Mesh3D:vertex_positions")
    first_vertices: list[object] = cast(list[object], vertices[0])
    first_vertex: Float32[ndarray, "xyz"] = np.asarray(first_vertices[0], dtype=np.float32)
    np.testing.assert_allclose(first_vertex, np.asarray([1.0, 2.0, 3.0], dtype=np.float32), atol=1e-6)


def test_convert_sam3d_body_parquet_to_rrd_skips_mesh_when_metadata_columns_are_missing(tmp_path: Path) -> None:
    parquet_path: Path = tmp_path / "000000.parquet"
    rrd_path: Path = tmp_path / "sam3d-body-no-mesh-columns.rrd"
    mhr_model_path: Path = tmp_path / "mhr_model.pt"
    checkpoint_path: Path = tmp_path / "model.ckpt"
    _write_bbox_parquet(parquet_path)
    table_without_mesh: pa.Table = pq.read_table(parquet_path).drop_columns(["person_id", "model_params", "shape_params"])
    pq.write_table(table_without_mesh, parquet_path)
    _write_fake_mhr_assets(mhr_model_path=mhr_model_path, checkpoint_path=checkpoint_path)

    convert_sam3d_body_parquet_to_rrd(
        Sam3dBodyParquetToRrdConfig(
            parquet_path=parquet_path,
            rrd_path=rrd_path,
            mhr_model_path=mhr_model_path,
            sam3d_body_checkpoint_path=checkpoint_path,
            parquet_reader_summary=False,
        )
    )

    summary: str = RrdReader(rrd_path).store().summary()
    assert "/world/gt/mhr_mesh" not in summary


def test_convert_sam3d_body_parquet_to_rrd_requires_mesh_metadata_columns_when_mesh_is_required(tmp_path: Path) -> None:
    parquet_path: Path = tmp_path / "000000.parquet"
    rrd_path: Path = tmp_path / "sam3d-body-required-mesh-columns.rrd"
    mhr_model_path: Path = tmp_path / "mhr_model.pt"
    checkpoint_path: Path = tmp_path / "model.ckpt"
    _write_bbox_parquet(parquet_path)
    table_without_mesh: pa.Table = pq.read_table(parquet_path).drop_columns(["person_id", "model_params", "shape_params"])
    pq.write_table(table_without_mesh, parquet_path)
    _write_fake_mhr_assets(mhr_model_path=mhr_model_path, checkpoint_path=checkpoint_path)

    with pytest.raises(ValueError, match="MHR mesh logging requires parquet columns"):
        convert_sam3d_body_parquet_to_rrd(
            Sam3dBodyParquetToRrdConfig(
                parquet_path=parquet_path,
                rrd_path=rrd_path,
                mhr_model_path=mhr_model_path,
                sam3d_body_checkpoint_path=checkpoint_path,
                require_mesh=True,
                parquet_reader_summary=False,
            )
        )


def test_load_mesh_faces_uses_weights_only_checkpoint_loading_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint_path: Path = tmp_path / "model.ckpt"
    checkpoint_path.touch()
    weights_only_values: list[bool | None] = []

    def fake_torch_load(path: str, *, map_location: str, weights_only: bool | None = None) -> dict[str, dict[str, torch.Tensor]]:
        assert path == str(checkpoint_path)
        assert map_location == "cpu"
        weights_only_values.append(weights_only)
        return {"state_dict": {"head_pose.faces": torch.tensor([[0, 1, 2]], dtype=torch.int64)}}

    monkeypatch.setattr(torch, "load", fake_torch_load)

    faces: np.ndarray = _load_mesh_faces(checkpoint_path)

    np.testing.assert_array_equal(faces, np.asarray([[0, 1, 2]], dtype=np.int32))
    assert weights_only_values == [True]


def test_load_mesh_faces_falls_back_for_legacy_checkpoint_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint_path: Path = tmp_path / "model.ckpt"
    checkpoint_path.touch()
    weights_only_values: list[bool | None] = []

    def fake_torch_load(path: str, *, map_location: str, weights_only: bool | None = None) -> dict[str, dict[str, torch.Tensor]]:
        assert path == str(checkpoint_path)
        assert map_location == "cpu"
        weights_only_values.append(weights_only)
        if weights_only is True:
            raise pickle.UnpicklingError("legacy checkpoint")
        return {"state_dict": {"head_pose.faces": torch.tensor([[0, 1, 2]], dtype=torch.int64)}}

    monkeypatch.setattr(torch, "load", fake_torch_load)

    faces: np.ndarray = _load_mesh_faces(checkpoint_path)

    np.testing.assert_array_equal(faces, np.asarray([[0, 1, 2]], dtype=np.int32))
    assert weights_only_values == [True, False]
