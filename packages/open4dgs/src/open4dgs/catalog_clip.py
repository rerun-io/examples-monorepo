"""Read a catalog segment into a multi-view training clip: undistorted frames, pinhole cameras, SMPL-X meshes.

The segment follows the dataforge ``exoego:v2`` rig layout (``simplecv.catalog_rig_layout``): one camera per
``/world/rig_NN/cam_MM`` node, its static ``Transform3D`` (camera-from-world) on the node, its ``Pinhole`` on
``.../pinhole`` and its AV1 ``VideoStream`` on ``.../pinhole/video``, on the ``frame_index`` sequence timeline.
The calibration comes from ``simplecv.catalog_calibration.camera_models_from_static_table`` and the packets from
``simplecv.catalog_video.catalog_video_from_table``; both read one table this module queries once per segment.

MAMMA's IOI cameras carry the Vicon lens as ``vicon_radial_2`` on the camera node (dataforge
``mamma_layers.log_cameras``): ``[cx, cy, k1, k2, k3]`` in the camera's NATIVE pixel grid (``source_width`` x
``source_height``), a polynomial that maps a distorted pixel ``d`` to its undistorted pinhole pixel
``u = c + (d - c) (1 + k1 r^2 + k2 r^4 + k3 r^6)``, ``r = |d - c|``. The logged ``Pinhole`` is the undistorted pinhole at
the stored size. The lens is ``simplecv.sensors.camera.vicon_radial.ViconRadialLens`` (read by
``simplecv.catalog_calibration.read_lens``; its torch path keeps the remap on the GPU).

Pixel convention: pixel centres sit at integer coordinates and a ``w``-wide image spans ``[-0.5, w - 0.5]``, so a
resize by ``s`` maps ``x -> (x + 0.5) s - 0.5`` (``scale_intrinsics``), the convention of ``grid_sample`` and
``interpolate`` with ``align_corners=False``.
"""

import re
from collections import OrderedDict
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pyarrow as pa
import torch
import torch.nn.functional as F
from datafusion import col
from jaxtyping import Bool, Float32, Float64, Int64, UInt8, UInt32
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry
from simplecv.catalog_calibration import CatalogCameraCalibration, camera_models_from_static_table, read_lens
from simplecv.catalog_rig_layout import CatalogCamera, CatalogRigLayout, catalog_components, parse_rig_layout
from simplecv.catalog_video import CatalogVideo, catalog_video_from_table
from simplecv.catalog_video_codec import wrap_mp4
from simplecv.catalog_video_reader import StreamIndex, read_video_index, select_decode_device
from simplecv.rrd_query_utils import first_valid_value
from simplecv.sensors.camera.vicon_radial import ViconRadialLens
from simplecv.video_io import bind_cuda_device
from torch import Tensor

from open4dgs.roi_cameras import VirtualCamera, fit_virtual_camera, virtual_to_source_pixels

LOCAL_CATALOG: str = "rerun+http://pablo-dl-server:51235"
"""The local catalog: open4dgs's companion datasets, layers and runs are registered here."""
SPARK_CATALOG: str = "rerun+http://dgx-spark:51235"
"""The Spark catalog (read only): the dataforge source datasets."""
FRAME_TIMELINE: str = "frame_index"
"""The dataforge sequence timeline every layer is indexed on (also ``video_time``, a duration)."""
RESAMPLE_CHUNK: int = 16
"""Frames undistorted and resized per call (16 x 2056x1504 is 0.6 GB as float32 per buffer)."""
GRID_CACHE_BYTES: int = 2 << 30
"""Bytes of sampling grids kept for reuse (a 2056x1504 grid is 25 MB: a 32-camera MAMMA rig is 0.8 GB)."""
_GRID_CACHE: OrderedDict[Hashable, Tensor] = OrderedDict()
_MESH_PATH: re.Pattern[str] = re.compile(r"^/world/(?P<source>.+)/people/(?P<person>person_\d+)/body/mesh$")


def companion_location(catalog_url: str, dataset: str, local_url: str = LOCAL_CATALOG) -> tuple[str, str]:
    """Where layers computed for ``dataset`` on ``catalog_url`` are registered: there when it is ``local_url``, else on
    ``local_url`` in the companion dataset ``open4dgs-mamma`` (the read-only Spark, whose only open4dgs source is MAMMA)."""
    return (catalog_url, dataset) if catalog_url == local_url else (local_url, "open4dgs-mamma")


@dataclass(frozen=True, slots=True)
class ClipPeople:
    """Every person's posed SMPL-X mesh at the clip's frames, from the segment's ``body_mesh`` layer."""

    source: str
    """The body source under ``/world``: ``pred/mamma`` (markerless MAMMA) or ``gt`` (MAMMA eval)."""
    names: tuple[str, ...]
    """Person entity ids, e.g. ``person_00``, in index order."""
    vertices: Float32[Tensor, "frames people verts 3"]
    """World-frame vertices, metres; NaN where ``present`` is False."""
    present: Bool[Tensor, "frames people"]
    """Whether the layer holds a mesh for the person at the frame. dataforge stores meshes every
    ``BODY_MESH_STRIDE`` (3) frames, so off-stride frames are absent (full-rate bodies need the ``body_pose`` layer)."""
    faces: UInt32[Tensor, "faces 3"]
    """Triangle list shared by every person (SMPL-X: 20,908 faces over 10,475 vertices)."""


@dataclass(frozen=True, slots=True)
class ClipRoi:
    """How a clip's person-centred virtual cameras were made (``read_catalog_clip(..., roi="people")``: every person's
    mesh vertices over the whole window stay in view)."""

    pad: float
    """Fraction of the points' box added on every side."""
    max_side: int | None
    """Cap on the longer side of the virtual images."""
    cameras: tuple[VirtualCamera, ...]
    """Per clip camera, the virtual camera its frames, ``valid``, ``K`` and ``world_T_cam`` belong to."""


@dataclass(frozen=True, slots=True)
class MultiViewClip:
    """A training clip: ``frames[c][t]`` is camera ``c`` at source frame ``frame_indices[t]``, undistorted to pinhole ``K[c]``.

    Cameras keep their own size: a MAMMA rig mixes landscape and portrait cameras (2056x1504, 2048x1504, 1504x2056,
    1504x2048), so frames are one tensor per camera, each resized so its longer side is ``max_side``.
    """

    camera_names: tuple[str, ...]
    """Camera names under ``/world``, e.g. ``rig_01/cam_00``, in rig order."""
    frames: tuple[UInt8[Tensor, "frames h w 3"], ...]
    """Per camera, RGB frames undistorted and resized, on the CPU."""
    valid: tuple[Bool[Tensor, "h w"], ...]
    """Per camera, pixels whose undistorted ray lands inside the source image; outside pixels are black in ``frames``."""
    K: Float32[Tensor, "cams 3 3"]
    """Undistorted pinhole intrinsics at each camera's returned size, pixel centres at integers."""
    world_T_cam: Float32[Tensor, "cams 4 4"]
    """Camera-to-world, OpenCV RDF camera axes; world is the segment's ``/world`` frame (dataforge: right-handed,
    Z up, metres)."""
    fps: float
    """The segment's nominal frame rate (``property:capture:fps``)."""
    frame_indices: Int64[Tensor, " frames"]
    """Source ``frame_index`` values of the clip's frames."""
    people: ClipPeople | None
    """SMPL-X meshes; None when the segment has no ``body/mesh`` entity."""
    stored_sizes: tuple[tuple[int, int], ...] = ()
    """Each camera's stored video ``(width, height)`` (the grid catalog layers such as ``people_matte`` are aligned with)."""
    lenses: tuple[ViconRadialLens | None, ...] = ()
    """Each camera's lens, None for a pinhole camera or when the clip was read with ``undistort=False``."""
    roi: ClipRoi | None = None
    """None: the real cameras. Otherwise frames, ``valid``, ``K`` and ``world_T_cam`` are those of these virtual cameras."""

    @property
    def sizes(self) -> tuple[tuple[int, int], ...]:
        """Each camera's returned ``(width, height)``."""
        return tuple((frames.shape[2], frames.shape[1]) for frames in self.frames)

    @property
    def nbytes(self) -> int:
        """Bytes held by the frames."""
        return sum(frames.numel() for frames in self.frames)


def select_frame_indices(frame_start: int, num_frames: int, frame_step: int, available: Int64[ndarray, " n"]) -> Int64[ndarray, " t"]:
    """``frame_start, frame_start + step, ...`` (``num_frames`` of them), checked against the frames a segment holds.

    Raises:
        ValueError: If the arguments are not positive, or a requested frame is not in ``available``.
    """
    if frame_start < 0 or num_frames < 1 or frame_step < 1:
        raise ValueError(f"need frame_start >= 0, num_frames >= 1, frame_step >= 1; got {frame_start}, {num_frames}, {frame_step}")
    frames: Int64[ndarray, " t"] = frame_start + frame_step * np.arange(num_frames, dtype=np.int64)
    missing: Int64[ndarray, " m"] = frames[~np.isin(frames, available)]
    if len(missing):
        span: str = f"{int(available.min())}..{int(available.max())}" if len(available) else "none"
        raise ValueError(f"frames {missing.tolist()[:5]} are not in the segment (it holds frames {span})")
    return frames


def fit_max_side(width: int, height: int, max_side: int | None) -> tuple[int, int]:
    """The size that keeps the aspect ratio with the longer side at most ``max_side`` (unchanged when None or larger)."""
    if max_side is None or max(width, height) <= max_side:
        return width, height
    scale: float = max_side / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def scale_intrinsics(K: Float64[ndarray, "3 3"], source_wh: tuple[int, int], target_wh: tuple[int, int]) -> Float64[ndarray, "3 3"]:
    """K of an image resized from ``source_wh`` to ``target_wh`` (pixel centres at integers: ``x -> (x + 0.5) s - 0.5``)."""
    sx: float = target_wh[0] / source_wh[0]
    sy: float = target_wh[1] / source_wh[1]
    scaled: Float64[ndarray, "3 3"] = K.astype(np.float64).copy()
    scaled[0, 0] *= sx
    scaled[0, 1] *= sx
    scaled[1, 1] *= sy
    scaled[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    scaled[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return scaled


def _device_key(device: torch.device) -> str:
    """``cuda`` and ``cuda:<current>`` are one device for the grid cache."""
    if device.type == "cuda" and device.index is None:
        return f"cuda:{torch.cuda.current_device()}"
    return str(device)


def _lens_key(lens: ViconRadialLens | None) -> Hashable:
    return None if lens is None else (lens.centre.tobytes(), lens.radial.tobytes(), lens.native_wh)


def _cached_grid(key: Hashable, build: Callable[[], Tensor]) -> Float32[Tensor, "1 h w 2"]:
    """The grid stored under ``key``, built by ``build`` on a miss; least recently used grids go past ``GRID_CACHE_BYTES``."""
    grid: Tensor | None = _GRID_CACHE.get(key)
    if grid is not None:
        _GRID_CACHE.move_to_end(key)
        return grid
    grid = build()
    _GRID_CACHE[key] = grid
    while len(_GRID_CACHE) > 1 and sum(cached.nbytes for cached in _GRID_CACHE.values()) > GRID_CACHE_BYTES:
        _GRID_CACHE.popitem(last=False)
    return grid


def clear_grid_cache() -> None:
    """Drop every cached sampling grid (frees their device memory)."""
    _GRID_CACHE.clear()


def undistortion_grid(lens: ViconRadialLens, stored_wh: tuple[int, int], device: torch.device) -> Float32[Tensor, "1 h w 2"]:
    """``grid_sample`` grid that resamples a stored distorted frame onto its undistorted pinhole (same size, same K).

    Each output pixel is an undistorted stored pixel; it is mapped to the native grid, through ``lens.distort``, and
    back to the stored grid. Pixels whose inversion does not converge get a grid value outside ``[-1, 1]``.
    The grid is cached per (lens, size, device): frames and mattes of one clip share it; do not modify it in place.
    """
    return _cached_grid(("undistortion", _lens_key(lens), stored_wh, _device_key(device)), lambda: _undistortion_grid(lens, stored_wh, device))


def _undistortion_grid(lens: ViconRadialLens, stored_wh: tuple[int, int], device: torch.device) -> Float32[Tensor, "1 h w 2"]:
    width, height = stored_wh
    to_native: tuple[float, float] = (lens.native_wh[0] / width, lens.native_wh[1] / height)
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float64, device=device), torch.arange(width, dtype=torch.float64, device=device), indexing="ij"
    )
    undistorted_stored: Float64[Tensor, "n 2"] = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)
    scale_to_native: Float64[Tensor, "2"] = torch.tensor(to_native, dtype=torch.float64, device=device)
    distorted_native: Float64[Tensor, "n 2"] = lens.distort((undistorted_stored + 0.5) * scale_to_native - 0.5)
    distorted_stored: Float64[Tensor, "n 2"] = (distorted_native + 0.5) / scale_to_native - 0.5
    size: Float64[Tensor, "2"] = torch.tensor([width, height], dtype=torch.float64, device=device)
    grid: Float64[Tensor, "n 2"] = (distorted_stored + 0.5) / size * 2.0 - 1.0
    grid = torch.nan_to_num(grid, nan=-2.0)
    return grid.reshape(1, height, width, 2).to(torch.float32)


def roi_grid(lens: ViconRadialLens | None, camera: VirtualCamera, device: torch.device) -> Float32[Tensor, "1 h w 2"]:
    """``grid_sample`` grid from a stored (distorted) frame to ``camera``'s sampling grid: rotation, zoom and undistortion in one.

    Each virtual pixel goes to the undistorted stored pixel that sees its ray (``virtual_to_source_pixels``), then through
    the lens to the stored grid (as ``undistortion_grid``). Pixels with no source get a grid value outside ``[-1, 1]``.
    Cached per (lens, virtual camera, device) like ``undistortion_grid``; do not modify it in place.
    """
    camera_key: Hashable = (
        camera.source_K.tobytes(),
        camera.source_size,
        camera.cam_R_virtual.tobytes(),
        camera.sample_K.tobytes(),
        camera.sample_size,
    )
    return _cached_grid(("roi", _lens_key(lens), camera_key, _device_key(device)), lambda: _roi_grid(lens, camera, device))


def _roi_grid(lens: ViconRadialLens | None, camera: VirtualCamera, device: torch.device) -> Float32[Tensor, "1 h w 2"]:
    width, height = camera.source_size
    undistorted_stored: Float64[Tensor, "n 2"] = virtual_to_source_pixels(camera, device).reshape(-1, 2)
    distorted_stored: Float64[Tensor, "n 2"] = undistorted_stored
    if lens is not None:
        scale_to_native: Float64[Tensor, "2"] = torch.tensor(
            (lens.native_wh[0] / width, lens.native_wh[1] / height), dtype=torch.float64, device=device
        )
        distorted_native: Float64[Tensor, "n 2"] = lens.distort((undistorted_stored + 0.5) * scale_to_native - 0.5)
        distorted_stored = (distorted_native + 0.5) / scale_to_native - 0.5
    size: Float64[Tensor, "2"] = torch.tensor([width, height], dtype=torch.float64, device=device)
    grid: Float64[Tensor, "n 2"] = torch.nan_to_num((distorted_stored + 0.5) / size * 2.0 - 1.0, nan=-2.0)
    return grid.reshape(1, camera.sample_size[1], camera.sample_size[0], 2).to(torch.float32)


def resample_images(
    images: Float32[Tensor, "n c h w"], grid: Float32[Tensor, "1 h2 w2 2"] | None, target_wh: tuple[int, int]
) -> Float32[Tensor, "n c h3 w3"]:
    """Undistort (when ``grid`` is given) at the stored size, then resize to ``target_wh`` with antialiasing.

    Frames and catalog layers aligned with the stored video (``people_matte``) go through this one function, so their pixels agree.
    """
    if grid is not None:
        images = F.grid_sample(images, grid.expand(len(images), -1, -1, -1), mode="bilinear", padding_mode="zeros", align_corners=False)
    if (images.shape[3], images.shape[2]) != target_wh:
        images = F.interpolate(images, size=(target_wh[1], target_wh[0]), mode="bilinear", antialias=True, align_corners=False)
    return images


def resample_frames(
    frames_nchw: UInt8[Tensor, "n 3 h w"], grid: Float32[Tensor, "1 h2 w2 2"] | None, target_wh: tuple[int, int]
) -> UInt8[Tensor, "n 3 h3 w3"]:
    """``resample_images`` for uint8 RGB frames."""
    return resample_images(frames_nchw.float(), grid, target_wh).round().clamp(0, 255).to(torch.uint8)


def valid_mask(grid: Float32[Tensor, "1 h w 2"] | None, stored_wh: tuple[int, int], target_wh: tuple[int, int]) -> Bool[Tensor, "h2 w2"]:
    """Output pixels whose source lies inside the stored frame (all of them without a lens)."""
    if grid is None:
        return torch.ones((target_wh[1], target_wh[0]), dtype=torch.bool)
    inside: Float32[Tensor, "1 1 h w"] = (grid.abs() <= 1.0).all(dim=-1).float()[:, None]
    if stored_wh != target_wh:
        inside = F.interpolate(inside, size=(target_wh[1], target_wh[0]), mode="area")
    return (inside[0, 0] > 0.999).cpu()


def static_cell(table: pa.Table, column_name: str) -> object | None:
    """The first valid value of a static table's column; None when the column is absent or empty."""
    if column_name not in table.column_names:
        return None
    return first_valid_value(table.column(column_name), allow_none=True, component_name=column_name)


def _mesh_entities(dataset: DatasetEntry, segment_id: str, body_source: str | None) -> tuple[str, list[str]] | None:
    """The body source and its person mesh entities; with no ``body_source``, ``gt`` wins over predictions."""
    by_source: dict[str, list[str]] = {}
    for component in catalog_components(dataset.filter_segments(segment_id).schema()):
        match: re.Match[str] | None = _MESH_PATH.fullmatch(component.entity_path)
        if match is not None and component.component == "Mesh3D:vertex_positions":
            by_source.setdefault(match.group("source"), []).append(component.entity_path)
    if not by_source:
        return None
    source: str = body_source if body_source is not None else ("gt" if "gt" in by_source else sorted(by_source)[0])
    if source not in by_source:
        raise ValueError(f"segment {segment_id} has no body meshes under /world/{source}; it has {sorted(by_source)}")
    return source, sorted(by_source[source])


def mesh_vertices(column: pa.ChunkedArray, rows: Sequence[int | None], num_vertices: int) -> tuple[Float32[ndarray, "t v 3"], Bool[ndarray, " t"]]:
    """Per slot, the mesh at table row ``rows[slot]`` of a ``Mesh3D:vertex_positions`` column, NaN and absent where the row
    is None, null, or not ``num_vertices`` long.

    The column is ``list<fixed_size_list<f32, 3>>``: rows are slices of the flat xyz buffer (no Python lists).
    """
    cells: pa.ListArray | pa.LargeListArray = column.combine_chunks()
    offsets: Int64[ndarray, " r1"] = cells.offsets.to_numpy().astype(np.int64)
    xyz: Float32[ndarray, "n 3"] = np.asarray(cells.values.flatten().to_numpy(zero_copy_only=False), dtype=np.float32).reshape(-1, 3)
    is_valid: Bool[ndarray, " r"] = cells.is_valid().to_numpy(zero_copy_only=False)
    vertices: Float32[ndarray, "t v 3"] = np.full((len(rows), num_vertices, 3), np.nan, dtype=np.float32)
    present: Bool[ndarray, " t"] = np.zeros(len(rows), dtype=bool)
    for slot, row in enumerate(rows):
        if row is not None and is_valid[row] and offsets[row + 1] - offsets[row] == num_vertices:
            vertices[slot] = xyz[offsets[row] : offsets[row + 1]]
            present[slot] = True
    return vertices, present


def _read_people(dataset: DatasetEntry, segment_id: str, frames: Int64[ndarray, " t"], body_source: str | None) -> ClipPeople | None:
    found: tuple[str, list[str]] | None = _mesh_entities(dataset, segment_id, body_source)
    if found is None:
        return None
    source, entities = found
    view = dataset.filter_segments(segment_id).filter_contents(entities)
    vertex_columns: list[str] = [f"{entity}:Mesh3D:vertex_positions" for entity in entities]
    table: pa.Table = (
        view.reader(index=FRAME_TIMELINE)
        .filter((col(FRAME_TIMELINE) >= int(frames.min())) & (col(FRAME_TIMELINE) <= int(frames.max())))
        .select(FRAME_TIMELINE, *vertex_columns)
        .to_arrow_table()
    )
    # dataforge writes the triangle list once, on the first trusted row of the first batch, which may precede the window.
    faces_table: pa.Table = view.reader(index=FRAME_TIMELINE).select(*[f"{entity}:Mesh3D:triangle_indices" for entity in entities]).to_arrow_table()
    faces: UInt32[ndarray, "f 3"] | None = None
    for entity in entities:
        value: object | None = first_valid_value(faces_table.column(f"{entity}:Mesh3D:triangle_indices"), allow_none=True)
        if value is not None:
            faces = np.asarray(value, dtype=np.uint32).reshape(-1, 3)
            break
    if faces is None:
        raise ValueError(f"segment {segment_id}: no triangle list under /world/{source}/people")
    row_of: dict[int, int] = {int(frame): row for row, frame in enumerate(table.column(FRAME_TIMELINE).to_numpy())}
    num_vertices: int = int(faces.max()) + 1
    vertices: Float32[ndarray, "t p v 3"] = np.full((len(frames), len(entities), num_vertices, 3), np.nan, dtype=np.float32)
    present: Bool[ndarray, "t p"] = np.zeros((len(frames), len(entities)), dtype=bool)
    rows: list[int | None] = [row_of.get(frame) for frame in frames.tolist()]
    for person, column_name in enumerate(vertex_columns):
        vertices[:, person], present[:, person] = mesh_vertices(table.column(column_name), rows, num_vertices)
    return ClipPeople(
        source=source,
        names=tuple(entity.split("/")[-3] for entity in entities),
        vertices=torch.from_numpy(vertices),
        present=torch.from_numpy(present),
        faces=torch.from_numpy(faces),
    )


def read_catalog_clip(
    catalog_url: str,
    dataset: str,
    segment_id: str,
    frame_start: int,
    num_frames: int,
    frame_step: int = 1,
    cameras: Sequence[str] | None = None,
    max_side: int | None = None,
    *,
    body_source: str | None = None,
    decode_device: torch.device | None = None,
    undistort: bool = True,
    roi: Literal["people"] | None = None,
    roi_pad: float = 0.15,
) -> MultiViewClip:
    """Read ``num_frames`` frames (every ``frame_step``-th from ``frame_start``) of a segment's cameras into a clip.

    Args:
        catalog_url: Rerun catalog URL, e.g. ``rerun+http://dgx-spark:51235``.
        dataset: Dataset name.
        segment_id: Segment id.
        frame_start: First source ``frame_index``.
        num_frames: Frames in the clip.
        frame_step: Stride between source frames.
        cameras: Camera names (``rig_01/cam_00``) to read, in that order; None reads every video camera in rig order.
        max_side: Longer side of the returned frames (after undistortion); None keeps the stored size.
        body_source: Body source under ``/world`` (``gt``, ``pred/mamma``); None prefers ``gt``.
        decode_device: Where NVDEC decodes and the undistortion runs (default ``cuda``).
        undistort: False returns the stored (distorted) frames, e.g. to compute a layer aligned with the stored video;
            ``K`` is then only the pinhole approximation of a lensed camera.
        roi: None returns the real cameras. ``people`` returns per camera a person-centred virtual camera
            (``open4dgs.roi_cameras``): same optical centre, rotated to the people and sized so every person's mesh
            over the window, padded by ``roi_pad``, fits at the source's pixel density; ``max_side`` caps it. Frames,
            ``valid``, ``K`` (centred principal point), ``world_T_cam`` and sizes are then the virtual camera's, and
            ``clip.roi`` records how they were made.
        roi_pad: Fraction of the people's box added on every side (hair, clothes, props, off-stride frames).

    Raises:
        ValueError: If a camera is missing or uncalibrated, a frame is outside the segment,
            or a camera's lens model has no undistortion here; with ``roi``, if the segment has no meshes
            or ``undistort`` is False.
    """
    if roi is not None and not undistort:
        raise ValueError("roi needs undistort=True (the virtual cameras are pinholes)")
    from torchcodec.decoders import VideoDecoder

    decode_on: torch.device = decode_device if decode_device is not None else torch.device(select_decode_device())
    entry: DatasetEntry = CatalogClient(catalog_url).get_dataset(dataset)
    layout: CatalogRigLayout = parse_rig_layout(catalog_components(entry.filter_segments(segment_id).schema()))
    by_name: dict[str, CatalogCamera] = {camera.name: camera for camera in layout.cameras}
    chosen: list[CatalogCamera] = [by_name[name] for name in cameras if name in by_name] if cameras is not None else list(layout.cameras)
    if cameras is not None and len(chosen) != len(cameras):
        raise ValueError(f"cameras {sorted(set(cameras) - set(by_name))} are not video cameras of {segment_id}")
    uncalibrated: list[str] = [camera.name for camera in chosen if not camera.has_calibration]
    if not chosen or uncalibrated:
        raise ValueError(f"{segment_id}: no cameras chosen, or cameras without a static Transform3D + Pinhole: {uncalibrated}")
    nodes: list[str] = [camera.transform_entity for camera in chosen]

    # One static query: calibration, lens, and the capture fps.
    static: pa.Table = (
        entry.filter_segments(segment_id)
        .filter_contents([f"{node}/**" for node in nodes] + ["/__properties/capture"])
        .reader(index=None)
        .to_arrow_table()
    )
    models: dict[str, CatalogCameraCalibration] = camera_models_from_static_table(static, nodes, f"segment {segment_id}", with_distortion=False)
    fps_cell: object | None = static_cell(static, "property:capture:fps")
    if fps_cell is None:
        raise ValueError(f"segment {segment_id} has no property:capture:fps")
    fps: float = float(np.asarray(fps_cell).reshape(-1)[0])

    # The cameras' keyframe flags (no samples) fix each camera's decode window.
    videos: list[str] = [camera.video_entity for camera in chosen]
    streams: tuple[StreamIndex, ...] = read_video_index(entry, segment_id, videos, FRAME_TIMELINE)
    timeline: Int64[ndarray, " n"] = np.unique(np.concatenate([stream.t_ns for stream in streams]))
    frames: Int64[ndarray, " t"] = select_frame_indices(frame_start, num_frames, frame_step, timeline)
    window_start: int = int(frames.min())
    for video, indexed in zip(videos, streams, strict=True):
        keyframe_times: Int64[ndarray, " k"] = indexed.t_ns[indexed.keyframes]
        before: Int64[ndarray, " k"] = keyframe_times[keyframe_times <= frames.min()]
        if len(before) == 0:
            raise ValueError(f"{video}: no keyframe at or before frame {int(frames.min())}")
        window_start = min(window_start, int(before.max()))
    packets: pa.Table = (
        entry.filter_segments(segment_id)
        .filter_contents(videos)
        .reader(index=FRAME_TIMELINE)
        .filter((col(FRAME_TIMELINE) >= window_start) & (col(FRAME_TIMELINE) <= int(frames.max())))
        .select(FRAME_TIMELINE, *[f"{video}:VideoStream:{field}" for video in videos for field in ("sample", "is_keyframe", "codec")])
        .to_arrow_table()
    )

    out_frames: list[UInt8[Tensor, "t h w 3"]] = []
    valid: list[Bool[Tensor, "h w"]] = []
    stored_sizes: list[tuple[int, int]] = []
    lenses: list[ViconRadialLens | None] = []
    virtual: list[VirtualCamera] = []
    roi_points: Float32[ndarray, "n 3"] | None = None
    people: ClipPeople | None = _read_people(entry, segment_id, frames, body_source)
    if roi is not None:
        if people is None:
            raise ValueError(f"roi={roi!r}: segment {segment_id} has no body meshes")
        roi_points = people.vertices.reshape(-1, 3).numpy()
    K: Float32[Tensor, "c 3 3"] = torch.empty((len(chosen), 3, 3), dtype=torch.float32)
    world_T_cam: Float32[Tensor, "c 4 4"] = torch.empty((len(chosen), 4, 4), dtype=torch.float32)
    for index, (node, video) in enumerate(zip(nodes, videos, strict=True)):
        stored_wh: tuple[int, int] = (models[node].pinhole.intrinsics.width, models[node].pinhole.intrinsics.height)
        target_wh: tuple[int, int] = fit_max_side(*stored_wh, max_side)
        stream: CatalogVideo = catalog_video_from_table(packets, video, FRAME_TIMELINE, f"{segment_id} {video}")
        times: Int64[ndarray, " n"] = stream.times.astype(np.int64)
        first: int = int(np.flatnonzero(np.asarray(stream.keyframes) & (times <= frames.min()))[-1])
        positions: list[int] = [int(np.searchsorted(times[first:], frame)) for frame in frames.tolist()]
        if any(times[first + p] != f for p, f in zip(positions, frames.tolist(), strict=True)):
            raise ValueError(f"{video}: some clip frames have no packet")
        bind_cuda_device(str(decode_on))
        # The clip's window of packets, not the stream's: the table above holds only the clip.
        mp4: bytes = wrap_mp4(stream.samples[first:], stream.keyframes[first:], max(1, round(fps)), codec=stream.codec)
        decoder = VideoDecoder(mp4, device=str(decode_on), seek_mode="exact", num_ffmpeg_threads=0)
        decoded: UInt8[Tensor, "t 3 h w"] = decoder.get_frames_at(positions).data
        if (decoded.shape[3], decoded.shape[2]) != stored_wh:
            raise ValueError(f"{video}: decoded {decoded.shape[3]}x{decoded.shape[2]}, the pinhole says {stored_wh}")
        lens: ViconRadialLens | None = read_lens(static, node) if undistort else None
        stored_sizes.append(stored_wh)
        lenses.append(lens)
        k_matrix: Float64[ndarray, "3 3"] | None = models[node].pinhole.intrinsics.k_matrix
        assert k_matrix is not None  # Intrinsics always derives it
        pose: Float64[ndarray, "4 4"] = np.asarray(models[node].pinhole.extrinsics.world_T_cam, dtype=np.float64)
        grid: Float32[Tensor, "1 h w 2"] | None
        grid_wh: tuple[int, int] = stored_wh
        if roi_points is not None:
            camera: VirtualCamera = fit_virtual_camera(roi_points, k_matrix, pose, stored_wh, roi_pad, max_side)
            virtual.append(camera)
            grid, grid_wh, target_wh = roi_grid(lens, camera, decode_on), camera.sample_size, camera.size
        else:
            grid = None if lens is None else undistortion_grid(lens, stored_wh, decode_on)
        # Float resampling costs 24 bytes per stored pixel and frame: chunk it to bound the GPU peak.
        out_frames.append(
            torch.cat(
                [
                    resample_frames(decoded[start : start + RESAMPLE_CHUNK], grid, target_wh).permute(0, 2, 3, 1).cpu()
                    for start in range(0, len(decoded), RESAMPLE_CHUNK)
                ]
            )
        )
        valid.append(valid_mask(grid, grid_wh, target_wh))
        if roi_points is not None:
            K[index] = torch.from_numpy(virtual[-1].K.astype(np.float32))
            world_T_cam[index] = torch.from_numpy(virtual[-1].world_T_virtual(pose).astype(np.float32))
        else:
            K[index] = torch.from_numpy(scale_intrinsics(k_matrix, stored_wh, target_wh).astype(np.float32))
            world_T_cam[index] = torch.from_numpy(pose.astype(np.float32))
        del decoder, decoded, grid
    return MultiViewClip(
        camera_names=tuple(camera.name for camera in chosen),
        frames=tuple(out_frames),
        valid=tuple(valid),
        K=K,
        world_T_cam=world_T_cam,
        fps=fps,
        frame_indices=torch.from_numpy(frames),
        people=people,
        stored_sizes=tuple(stored_sizes),
        lenses=tuple(lenses),
        roi=None if roi is None else ClipRoi(pad=roi_pad, max_side=max_side, cameras=tuple(virtual)),
    )
