from __future__ import annotations

import struct
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rerun as rr
from jaxtyping import Float32
from numpy import ndarray
from rerun.experimental import Chunk, DeriveLens, LazyChunkStream, OptimizationProfile, ParquetReader, Selector

from egoexo_forge.api.sam3d_body_mesh import Sam3dBodyMeshReconstructor, create_mesh_reconstructor, mesh_chunks

__all__: list[str] = [
    "BBOX_ENTITY_PATH",
    "Sam3dBodyParquetToRrdConfig",
    "Sam3dBodyRrdResult",
    "convert_sam3d_body_parquet_to_rrd",
]

SOURCE_ENTITY_PREFIX: str = "/source/sam3d_body/parquet"
ROW_TIMELINE: str = "row_index"
CAMERA_ENTITY_PATH: str = "/world/cam/pinhole"
DEFAULT_PLACEHOLDER_RESOLUTION: tuple[int, int] = (640, 480)
IMAGE_ENTITY_PATH: str = f"{CAMERA_ENTITY_PATH}/image"
BBOX_ENTITY_PATH: str = f"{IMAGE_ENTITY_PATH}/pred/bbox"
KEYPOINTS_2D_ENTITY_PATH: str = f"{IMAGE_ENTITY_PATH}/pred/mhr70_uv"
KEYPOINTS_3D_ENTITY_PATH: str = "/world/gt/mhr70_xyz"
MESH_ENTITY_PATH: str = "/world/gt/mhr_mesh"
SAM3D_BODY_REQUIRED_COLUMNS: frozenset[str] = frozenset(("dataset", "image", "bbox", "bbox_format", "cam_int", "keypoints_2d", "keypoints_3d"))
SAM3D_BODY_MESH_COLUMNS: frozenset[str] = frozenset(("person_id", "model_params", "shape_params"))
SUPPORTED_BBOX_FORMATS: dict[str, rr.Box2DFormat] = {
    "xywh": rr.Box2DFormat.XYWH,
    "xyxy": rr.Box2DFormat.XYXY,
}


@dataclass(frozen=True, slots=True)
class Sam3dBodyParquetToRrdConfig:
    """Configuration for converting one SAM-3D-Body parquet shard to RRD chunks."""

    parquet_path: Path
    """Input SAM-3D-Body parquet shard."""
    rrd_path: Path
    """Output RRD path."""
    application_id: str = "sam3d_body_parquet_to_rrd"
    """Rerun application id written into the RRD."""
    recording_id: str | None = None
    """Rerun recording id. Defaults to ``sam3d_body_<parquet stem>``."""
    camera_id: str = "sam3d_image"
    """Logical ego camera id used for duplicate ego-camera semantic streams."""
    image_root: Path | None = None
    """Optional root containing dataset images referenced by the parquet ``dataset`` and ``image`` columns."""
    placeholder_resolution: tuple[int, int] = DEFAULT_PLACEHOLDER_RESOLUTION
    """Fallback image resolution used for Pinhole resolution and generated placeholder images."""
    include_source_columns: bool = False
    """Whether to also preserve raw parquet columns under ``/source/sam3d_body/parquet``."""
    parquet_reader_summary: bool = True
    """Whether to materialize a Rerun ParquetReader summary in the returned result."""
    optimize_for_object_store: bool = True
    """Whether to optimize the materialized chunks for object-store/catalog-style access before writing."""
    mhr_model_path: Path | None = None
    """TorchScript MHR model path used for optional mesh reconstruction."""
    sam3d_body_checkpoint_path: Path | None = None
    """SAM-3D-Body checkpoint path containing mesh faces."""
    require_mesh: bool = False
    """Whether missing or failed mesh reconstruction should raise an error."""
    auto_discover_mesh_assets: bool = True
    """Whether to look for SAM-3D-Body mesh assets in the local Hugging Face cache."""


@dataclass(frozen=True, slots=True)
class Sam3dBodyRrdResult:
    """Summary of a SAM-3D-Body parquet-to-RRD conversion."""

    rrd_path: Path
    """Output RRD path."""
    row_count: int
    """Number of parquet rows converted."""
    parquet_reader_summary: str
    """Optional summary of the raw parquet chunks produced by ``ParquetReader``."""


def convert_sam3d_body_parquet_to_rrd(config: Sam3dBodyParquetToRrdConfig) -> Sam3dBodyRrdResult:
    """Convert a SAM-3D-Body parquet shard into an RRD using Rerun chunk processing."""
    schema: pa.Schema = pq.read_schema(config.parquet_path)
    missing_columns: list[str] = sorted(SAM3D_BODY_REQUIRED_COLUMNS - set(schema.names))
    if missing_columns:
        raise ValueError(f"SAM-3D-Body parquet is missing required columns: {missing_columns}")

    parquet_metadata: pq.FileMetaData | None = pq.ParquetFile(config.parquet_path).metadata
    row_count: int = 0 if parquet_metadata is None else int(parquet_metadata.num_rows)
    if row_count == 0:
        raise ValueError(f"No SAM-3D-Body rows were found in {config.parquet_path}")
    _validate_coco_rows(config.parquet_path)
    bbox_format: rr.Box2DFormat = _bbox_format(config.parquet_path)

    ego_pinhole_entity_path: str = f"/world/ego/{config.camera_id}/pinhole"
    ego_image_entity_path: str = f"{ego_pinhole_entity_path}/image"
    semantic_streams: list[LazyChunkStream] = [
        _boxes2d_stream(config.parquet_path, BBOX_ENTITY_PATH, bbox_format=bbox_format),
        _boxes2d_stream(config.parquet_path, f"{ego_image_entity_path}/bbox", bbox_format=bbox_format),
        _pinhole_stream(config.parquet_path, CAMERA_ENTITY_PATH, resolution=config.placeholder_resolution),
        _pinhole_stream(config.parquet_path, ego_pinhole_entity_path, resolution=config.placeholder_resolution),
        _image_stream(config, IMAGE_ENTITY_PATH, row_count=row_count),
        _image_stream(config, ego_image_entity_path, row_count=row_count),
        _points2d_stream(config.parquet_path, KEYPOINTS_2D_ENTITY_PATH),
        _points2d_stream(config.parquet_path, f"{ego_image_entity_path}/mhr70_uv"),
        _points3d_stream(config.parquet_path, KEYPOINTS_3D_ENTITY_PATH),
    ]
    mesh_stream: LazyChunkStream | None = _mesh_stream(config)
    if mesh_stream is not None:
        semantic_streams.append(mesh_stream)

    output_stream: LazyChunkStream = LazyChunkStream.merge(*semantic_streams)
    if config.include_source_columns:
        source_stream: LazyChunkStream = ParquetReader(
            config.parquet_path,
            entity_path_prefix=SOURCE_ENTITY_PREFIX,
            column_grouping="individual",
        ).stream()
        output_stream = LazyChunkStream.merge(source_stream, output_stream)

    config.rrd_path.parent.mkdir(parents=True, exist_ok=True)
    recording_id: str = config.recording_id or f"sam3d_body_{config.parquet_path.stem}"
    if config.optimize_for_object_store:
        output_stream.collect(optimize=OptimizationProfile.OBJECT_STORE).write_rrd(
            config.rrd_path,
            application_id=config.application_id,
            recording_id=recording_id,
        )
    else:
        output_stream.write_rrd(
            config.rrd_path,
            application_id=config.application_id,
            recording_id=recording_id,
        )

    parquet_summary: str = ""
    if config.parquet_reader_summary:
        parquet_summary = (
            ParquetReader(
                config.parquet_path,
                entity_path_prefix=SOURCE_ENTITY_PREFIX,
                column_grouping="individual",
            )
            .stream()
            .collect()
            .summary()
        )
    return Sam3dBodyRrdResult(config.rrd_path, row_count, parquet_summary)


@dataclass(frozen=True, slots=True)
class _MeshSubject:
    """Minimal subject record needed by the mesh reconstructor."""

    person_id: int
    """Person id used for mesh coloring."""
    model_params: Float32[ndarray, "mhr_params"] | None
    """MHR model parameters."""
    shape_params: Float32[ndarray, "shape_params"] | None
    """MHR shape parameters."""


def _boxes2d_stream(parquet_path: Path, entity_path: str, *, bbox_format: rr.Box2DFormat) -> LazyChunkStream:
    source_stream: LazyChunkStream = ParquetReader(
        parquet_path,
        entity_path_prefix=SOURCE_ENTITY_PREFIX,
        column_grouping="individual",
    ).stream()
    bbox_column_stream: LazyChunkStream = source_stream.filter(content=f"{SOURCE_ENTITY_PREFIX}/bbox")
    return bbox_column_stream.lenses(
        [
            DeriveLens("bbox", output_entity=entity_path).to_component(
                rr.Boxes2D.descriptor_centers(),
                Selector(".").pipe(_centers_from_bbox_format(bbox_format)),
            ),
            DeriveLens("bbox", output_entity=entity_path).to_component(
                rr.Boxes2D.descriptor_half_sizes(),
                Selector(".").pipe(_half_sizes_from_bbox_format(bbox_format)),
            ),
        ],
        output_mode="drop_unmatched",
    )


def _pinhole_stream(parquet_path: Path, entity_path: str, *, resolution: tuple[int, int]) -> LazyChunkStream:
    source_stream: LazyChunkStream = ParquetReader(
        parquet_path,
        entity_path_prefix=SOURCE_ENTITY_PREFIX,
        column_grouping="individual",
    ).stream()
    cam_int_stream: LazyChunkStream = source_stream.filter(content=f"{SOURCE_ENTITY_PREFIX}/cam_int")
    resolution_selector: Selector = Selector(".").pipe(_resolution_from_cam_int(resolution))
    return cam_int_stream.lenses(
        (
            DeriveLens("cam_int", output_entity=entity_path)
            .to_component(
                rr.Pinhole.descriptor_image_from_camera(),
                Selector(".").pipe(_pinhole_projection_from_cam_int),
            )
            .to_component(
                rr.Pinhole.descriptor_resolution(),
                resolution_selector,
            )
            .to_component(
                rr.Pinhole.descriptor_camera_xyz(),
                Selector(".").pipe(_camera_xyz_from_cam_int),
            )
        ),
        output_mode="drop_unmatched",
    )


def _points2d_stream(parquet_path: Path, entity_path: str) -> LazyChunkStream:
    position_rows: list[Float32[ndarray, "keypoints 2"]] = _keypoint_position_rows(parquet_path, column_name="keypoints_2d", channels=2)
    row_indices: np.ndarray = np.arange(len(position_rows), dtype=np.int64)
    chunk: Chunk = Chunk.from_columns(
        entity_path,
        indexes=[rr.TimeColumn(ROW_TIMELINE, sequence=row_indices)],
        columns=rr.Points2D.columns(positions=position_rows),
    )
    return LazyChunkStream.from_iter([chunk])


def _points3d_stream(parquet_path: Path, entity_path: str) -> LazyChunkStream:
    position_rows: list[Float32[ndarray, "keypoints 3"]] = _keypoint_position_rows(parquet_path, column_name="keypoints_3d", channels=3)
    row_indices: np.ndarray = np.arange(len(position_rows), dtype=np.int64)
    chunk: Chunk = Chunk.from_columns(
        entity_path,
        indexes=[rr.TimeColumn(ROW_TIMELINE, sequence=row_indices)],
        columns=rr.Points3D.columns(positions=position_rows),
    )
    return LazyChunkStream.from_iter([chunk])


def _centers_from_bbox_format(bbox_format: rr.Box2DFormat) -> Callable[[pa.Array], pa.Array]:
    def centers_from_bbox(boxes: pa.Array) -> pa.Array:
        boxes2d: rr.Boxes2D = rr.Boxes2D(array=_bbox_array(boxes), array_format=bbox_format)
        if boxes2d.centers is None:
            raise ValueError(f"Rerun did not derive Boxes2D centers from {bbox_format.value} boxes.")
        return boxes2d.centers.as_arrow_array()

    return centers_from_bbox


def _half_sizes_from_bbox_format(bbox_format: rr.Box2DFormat) -> Callable[[pa.Array], pa.Array]:
    def half_sizes_from_bbox(boxes: pa.Array) -> pa.Array:
        boxes2d: rr.Boxes2D = rr.Boxes2D(array=_bbox_array(boxes), array_format=bbox_format)
        if boxes2d.half_sizes is None:
            raise ValueError(f"Rerun did not derive Boxes2D half-sizes from {bbox_format.value} boxes.")
        return boxes2d.half_sizes.as_arrow_array()

    return half_sizes_from_bbox


def _bbox_array(boxes: pa.Array) -> Float32[ndarray, "n_boxes 4"]:
    if isinstance(boxes, pa.ListArray | pa.FixedSizeListArray):
        flattened_values: Float32[ndarray, "flat_values"] = np.asarray(boxes.values.to_numpy(zero_copy_only=False), dtype=np.float32)
    else:
        raise ValueError(f"Expected bbox column chunks to be Arrow list values, got {type(boxes).__name__}")
    if flattened_values.size % 4 != 0:
        raise ValueError(f"Expected flattened box values to be divisible by 4, got {flattened_values.size}")
    box_values: Float32[ndarray, "n_boxes 4"] = flattened_values.reshape((-1, 4)).astype(np.float32, copy=False)
    return box_values


def _pinhole_projection_from_cam_int(cam_int: pa.Array) -> pa.Array:
    matrices: Float32[ndarray, "n_rows 3 3"] = _nested_matrix_array(cam_int, rows=3, cols=3)
    pinhole: rr.Pinhole = rr.Pinhole(image_from_camera=matrices)
    if pinhole.image_from_camera is None:
        raise ValueError("Rerun did not derive Pinhole image_from_camera from cam_int.")
    return pinhole.image_from_camera.as_arrow_array()


def _resolution_from_cam_int(resolution: tuple[int, int]):
    def resolution_from_cam_int(cam_int: pa.Array) -> pa.Array:
        resolution_xy: Float32[ndarray, "n_rows 2"] = np.tile(np.asarray(resolution, dtype=np.float32), (len(cam_int), 1))
        identity_intrinsics: Float32[ndarray, "n_rows 3 3"] = np.tile(np.eye(3, dtype=np.float32), (len(cam_int), 1, 1))
        pinhole: rr.Pinhole = rr.Pinhole(image_from_camera=identity_intrinsics, resolution=resolution_xy)
        if pinhole.resolution is None:
            raise ValueError("Rerun did not derive Pinhole resolution.")
        return pinhole.resolution.as_arrow_array()

    return resolution_from_cam_int


def _camera_xyz_from_cam_int(cam_int: pa.Array) -> pa.Array:
    identity_intrinsics: Float32[ndarray, "n_rows 3 3"] = np.tile(np.eye(3, dtype=np.float32), (len(cam_int), 1, 1))
    pinhole: rr.Pinhole = rr.Pinhole(
        image_from_camera=identity_intrinsics,
        camera_xyz=[rr.ViewCoordinates.RDF] * len(cam_int),
    )
    if pinhole.camera_xyz is None:
        raise ValueError("Rerun did not derive Pinhole camera_xyz.")
    return pinhole.camera_xyz.as_arrow_array()


def _nested_matrix_array(values: pa.Array, *, rows: int, cols: int) -> Float32[ndarray, "n_rows rows cols"]:
    if isinstance(values, pa.ListArray | pa.FixedSizeListArray) and isinstance(values.values, pa.ListArray | pa.FixedSizeListArray):
        flattened_values: Float32[ndarray, "flat_values"] = np.asarray(values.values.values.to_numpy(zero_copy_only=False), dtype=np.float32)
    else:
        raise ValueError(f"Expected nested Arrow list matrix values, got {type(values).__name__}")
    expected_stride: int = rows * cols
    if flattened_values.size % expected_stride != 0:
        raise ValueError(f"Expected flattened matrix values to be divisible by {expected_stride}, got {flattened_values.size}")
    matrices: Float32[ndarray, "n_rows rows cols"] = flattened_values.reshape((-1, rows, cols)).astype(np.float32, copy=False)
    return matrices


def _keypoint_position_rows(parquet_path: Path, *, column_name: str, channels: int) -> list[Float32[ndarray, "keypoints channels"]]:
    table: pa.Table = pq.read_table(parquet_path, columns=[column_name])
    rows: list[object] = table.column(column_name).to_pylist()
    position_rows: list[Float32[ndarray, "keypoints channels"]] = []
    for row_index, row in enumerate(rows):
        if row is None:
            position_rows.append(np.zeros((0, channels), dtype=np.float32))
            continue
        keypoints: Float32[ndarray, "keypoints source_channels"] = np.asarray(row, dtype=np.float32)
        if keypoints.ndim != 2 or keypoints.shape[1] < channels:
            raise ValueError(
                f"Expected {column_name} row {row_index} to have shape (keypoints, >= {channels}), got {keypoints.shape}"
            )
        positions: Float32[ndarray, "keypoints channels"] = keypoints[:, :channels].astype(np.float32, copy=True)
        position_rows.append(positions)
    return position_rows


def _mesh_stream(config: Sam3dBodyParquetToRrdConfig) -> LazyChunkStream | None:
    schema: pa.Schema = pq.read_schema(config.parquet_path)
    missing_mesh_columns: list[str] = sorted(SAM3D_BODY_MESH_COLUMNS - set(schema.names))
    if missing_mesh_columns:
        if config.require_mesh:
            raise ValueError(f"MHR mesh logging requires parquet columns: {missing_mesh_columns}")
        return None

    mesh_reconstructor: Sam3dBodyMeshReconstructor | None = create_mesh_reconstructor(
        mhr_model_path=config.mhr_model_path,
        sam3d_body_checkpoint_path=config.sam3d_body_checkpoint_path,
        require_mesh=config.require_mesh,
        auto_discover_mesh_assets=config.auto_discover_mesh_assets,
    )
    if mesh_reconstructor is None:
        return None

    table: pa.Table = pq.read_table(config.parquet_path, columns=["person_id", "model_params", "shape_params"])
    person_ids: list[object] = table.column("person_id").to_pylist()
    model_params_values: list[object] = table.column("model_params").to_pylist()
    shape_params_values: list[object] = table.column("shape_params").to_pylist()
    records: list[tuple[float, _MeshSubject]] = []
    for row_index, (person_id, model_params, shape_params) in enumerate(zip(person_ids, model_params_values, shape_params_values, strict=True)):
        if not isinstance(person_id, int):
            raise ValueError(f"Expected person_id to be an int, got {type(person_id).__name__}")
        subject: _MeshSubject = _MeshSubject(
            person_id=person_id,
            model_params=None if model_params is None else np.asarray(model_params, dtype=np.float32).reshape(-1),
            shape_params=None if shape_params is None else np.asarray(shape_params, dtype=np.float32).reshape(-1),
        )
        records.append((float(row_index), subject))

    chunks = mesh_chunks(
        MESH_ENTITY_PATH,
        records=records,
        mesh_reconstructor=mesh_reconstructor,
        require_mesh=config.require_mesh,
        timeline_name=ROW_TIMELINE,
        timeline_kind="sequence",
    )
    return LazyChunkStream.from_iter(chunks)


def _image_stream(config: Sam3dBodyParquetToRrdConfig, entity_path: str, *, row_count: int) -> LazyChunkStream:
    image_bytes: list[bytes] = []
    media_types: list[str] = []
    image_records: list[tuple[str, str]] = _image_records(config.parquet_path)
    image_root: Path = config.image_root or _default_coco_image_root(config.parquet_path)
    placeholder_png: bytes = _placeholder_png(config.placeholder_resolution)
    for row_index in range(row_count):
        image_blob: bytes | None = None
        media_type: str = "image/png"
        if row_index < len(image_records):
            dataset: str
            image_name: str
            dataset, image_name = image_records[row_index]
            image_path: Path = image_root / _dataset_image_relative_path(dataset=dataset, image_name=image_name)
            if image_path.exists():
                image_file_blob: bytes = image_path.read_bytes()
                image_file_media_type: str | None = _image_media_type(image_file_blob)
                if image_file_media_type is not None:
                    image_blob = image_file_blob
                    media_type = image_file_media_type
        image_bytes.append(placeholder_png if image_blob is None else image_blob)
        media_types.append(media_type)

    row_indices: np.ndarray = np.arange(row_count, dtype=np.int64)
    image_chunk: Chunk = Chunk.from_columns(
        entity_path,
        indexes=[rr.TimeColumn(ROW_TIMELINE, sequence=row_indices)],
        columns=rr.EncodedImage.columns(blob=image_bytes, media_type=media_types),
    )
    return LazyChunkStream.from_iter([image_chunk])


def _image_records(parquet_path: Path) -> list[tuple[str, str]]:
    schema: pa.Schema = pq.read_schema(parquet_path)
    if "dataset" not in schema.names or "image" not in schema.names:
        return []
    table: pa.Table = pq.read_table(parquet_path, columns=["dataset", "image"])
    datasets: list[object] = table.column("dataset").to_pylist()
    images: list[object] = table.column("image").to_pylist()
    return [(str(dataset), str(image)) for dataset, image in zip(datasets, images, strict=True)]


def _default_coco_image_root(parquet_path: Path) -> Path:
    if parquet_path.parent.name.startswith("coco_"):
        return parquet_path.parent.parent
    return parquet_path.parent


def _dataset_image_relative_path(*, dataset: str, image_name: str) -> Path:
    if dataset.lower() != "coco":
        raise ValueError(f"Only dataset='coco' is supported for image lookup, got {dataset!r}.")
    image_parts: list[str] = image_name.split("_")
    if len(image_parts) < 3 or image_parts[0] != "COCO":
        raise ValueError(f"Expected COCO image names like COCO_train2014_000000000001.jpg, got {image_name!r}.")
    return Path(image_parts[1]) / image_name


def _image_media_type(image_blob: bytes) -> str | None:
    if image_blob.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_blob.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    return None


def _placeholder_png(resolution: tuple[int, int]) -> bytes:
    width: int
    height: int
    width, height = resolution
    if width <= 0 or height <= 0:
        raise ValueError(f"Expected positive placeholder resolution, got {resolution}")
    raw_rows_parts: list[bytes] = []
    for y in range(height):
        row_parts: list[bytes] = []
        for x in range(width):
            tile_is_light: bool = ((x // 32) + (y // 32)) % 2 == 0
            pixel: bytes = b"\xe8\xe8\xe8" if tile_is_light else b"\x58\x58\x58"
            row_parts.append(pixel)
        raw_rows_parts.append(b"\x00" + b"".join(row_parts))
    raw_rows: bytes = b"".join(raw_rows_parts)

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc: int = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    png_signature: bytes = b"\x89PNG\r\n\x1a\n"
    ihdr: bytes = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return png_signature + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw_rows)) + chunk(b"IEND", b"")


def _bbox_format(parquet_path: Path) -> rr.Box2DFormat:
    table: pa.Table = pq.read_table(parquet_path, columns=["bbox_format"])
    bbox_format_values: list[str] = sorted({str(bbox_format).lower() for bbox_format in table.column("bbox_format").to_pylist()})
    if len(bbox_format_values) != 1:
        raise ValueError(f"Expected one bbox_format per parquet shard, got {bbox_format_values}.")
    bbox_format_value: str = bbox_format_values[0]
    if bbox_format_value not in SUPPORTED_BBOX_FORMATS:
        raise ValueError(f"Only bbox_format values {sorted(SUPPORTED_BBOX_FORMATS)} are supported, got {bbox_format_value!r}.")
    return SUPPORTED_BBOX_FORMATS[bbox_format_value]


def _validate_coco_rows(parquet_path: Path) -> None:
    table: pa.Table = pq.read_table(parquet_path, columns=["dataset", "image"])
    datasets: list[object] = table.column("dataset").to_pylist()
    images: list[object] = table.column("image").to_pylist()
    invalid_datasets: list[object] = [dataset for dataset in datasets if str(dataset).lower() != "coco"]
    if invalid_datasets:
        raise ValueError("Only dataset='coco' rows are supported by this parquet-to-RRD converter.")
    for dataset, image in zip(datasets, images, strict=True):
        _dataset_image_relative_path(dataset=str(dataset), image_name=str(image))
