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
the stored size. This module ports dataforge's ``ViconLens`` (``dataforge/datasets/mamma_source.py``: the same map and
the same fixed-point inversion, iteration count and tolerances) to torch so the remap runs on the GPU;
``tests/test_catalog_clip.py`` checks the port against ``ViconLens`` when dataforge is importable. dataforge itself is
not imported here: ``dataforge.datasets`` imports every dataset module (3 s).

Pixel convention: pixel centres sit at integer coordinates and a ``w``-wide image spans ``[-0.5, w - 0.5]``, so a
resize by ``s`` maps ``x -> (x + 0.5) s - 0.5`` (``scale_intrinsics``), the convention of ``grid_sample`` and
``interpolate`` with ``align_corners=False``.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import torch
import torch.nn.functional as F
from datafusion import col
from jaxtyping import Bool, Float32, Float64, Int64, UInt8, UInt32
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry
from simplecv.catalog_calibration import CatalogCameraCalibration, camera_models_from_static_table
from simplecv.catalog_rig_layout import CatalogCamera, CatalogRigLayout, catalog_components, parse_rig_layout
from simplecv.catalog_video import CatalogVideo, catalog_keyframes, catalog_video_from_table
from simplecv.catalog_video_codec import wrap_mp4
from simplecv.rrd_query_utils import first_valid_value
from torch import Tensor
from torchcodec.decoders import VideoDecoder

FRAME_TIMELINE: str = "frame_index"
"""The dataforge sequence timeline every layer is indexed on (also ``video_time``, a duration)."""
LENS_ITERATIONS: int = 50
"""Fixed-point steps of the lens inversion (dataforge ``mamma_source.LENS_ITERATIONS``)."""
LENS_TOLERANCE_PX: float = 1e-9
"""Largest step, in pixels, of a converged lens inversion (dataforge ``LENS_TOLERANCE_PX``)."""
LENS_RESIDUAL_PX: float = 1e-6
"""Largest error, in pixels, of an inverted point mapped back (dataforge ``LENS_RESIDUAL_PX``)."""
RESAMPLE_CHUNK: int = 16
"""Frames undistorted and resized per call (16 x 2056x1504 is 0.6 GB as float32 per buffer)."""
_MESH_PATH: re.Pattern[str] = re.compile(r"^/world/(?P<source>.+)/people/(?P<person>person_\d+)/body/mesh$")


@dataclass(frozen=True, slots=True)
class ViconRadialLens:
    """MAMMA's ``vicon_radial_2`` lens in the camera's native pixel grid (see the module docstring)."""

    centre: Float64[ndarray, "2"]
    """Distortion centre, native pixels."""
    radial: Float64[ndarray, "3"]
    """k1, k2, k3 per pixel^2, pixel^4, pixel^6."""
    native_width: int
    """Width of the grid the lens is stated in (the camera node's ``source_width``)."""
    native_height: int
    """Height of that grid (``source_height``)."""

    def _scale(self, offset: Float64[Tensor, "n 2"]) -> Float64[Tensor, "n 1"]:
        radial: Float64[Tensor, "3"] = torch.as_tensor(self.radial, dtype=torch.float64, device=offset.device)
        r2: Float64[Tensor, "n 1"] = (offset**2).sum(dim=-1, keepdim=True)
        return 1.0 + radial[0] * r2 + radial[1] * r2**2 + radial[2] * r2**3

    def undistort(self, distorted: Float64[Tensor, "n 2"]) -> Float64[Tensor, "n 2"]:
        """Distorted (image) pixels to undistorted (pinhole) pixels, native grid: the release's map."""
        centre: Float64[Tensor, "2"] = torch.as_tensor(self.centre, dtype=torch.float64, device=distorted.device)
        offset: Float64[Tensor, "n 2"] = distorted - centre
        return centre + offset * self._scale(offset)

    def distort(self, undistorted: Float64[Tensor, "n 2"]) -> Float64[Tensor, "n 2"]:
        """Undistorted pixels to image pixels by fixed-point inversion, native grid; a point that does not converge is NaN."""
        centre: Float64[Tensor, "2"] = torch.as_tensor(self.centre, dtype=torch.float64, device=undistorted.device)
        target: Float64[Tensor, "n 2"] = undistorted - centre
        offset: Float64[Tensor, "n 2"] = target.clone()
        for _ in range(LENS_ITERATIONS):
            step: Float64[Tensor, "n 2"] = target / self._scale(offset) - offset
            offset = offset + step
            finite: Float64[Tensor, " k"] = step[torch.isfinite(step)].abs()
            if finite.numel() == 0 or not bool(finite.max() > LENS_TOLERANCE_PX):
                break
        distorted: Float64[Tensor, "n 2"] = centre + offset
        converged: Bool[Tensor, " n"] = (self.undistort(distorted) - undistorted).abs().amax(dim=-1) <= LENS_RESIDUAL_PX
        return torch.where(converged[:, None], distorted, torch.full_like(distorted, float("nan")))


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
class MultiViewClip:
    """A training clip: ``frames[c][t]`` is camera ``c`` at source frame ``frame_indices[t]``, undistorted to pinhole ``K[c]``.

    Cameras keep their own size: a MAMMA rig mixes landscape and portrait cameras (2056x1504, 2048x1504, 1504x2056,
    1504x2048), so frames are one tensor per camera, each resized so its longer side is ``max_side``.
    """

    camera_names: tuple[str, ...]
    """Camera names under ``/world``, e.g. ``rig_01/cam_00``, in rig order."""
    frames: tuple[UInt8[Tensor, "frames h w 3"], ...]
    """Per camera, RGB frames undistorted and resized; on the ``storage_device`` the reader was given (CPU by default)."""
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


def undistortion_grid(lens: ViconRadialLens, stored_wh: tuple[int, int], device: torch.device) -> Float32[Tensor, "1 h w 2"]:
    """``grid_sample`` grid that resamples a stored distorted frame onto its undistorted pinhole (same size, same K).

    Each output pixel is an undistorted stored pixel; it is mapped to the native grid, through ``lens.distort``, and
    back to the stored grid. Pixels whose inversion does not converge get a grid value outside ``[-1, 1]``.
    """
    width, height = stored_wh
    to_native: tuple[float, float] = (lens.native_width / width, lens.native_height / height)
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


def resample_frames(
    frames_nchw: UInt8[Tensor, "n 3 h w"], grid: Float32[Tensor, "1 h2 w2 2"] | None, target_wh: tuple[int, int]
) -> UInt8[Tensor, "n 3 h3 w3"]:
    """Undistort (when ``grid`` is given) at the stored size, then resize to ``target_wh`` with antialiasing."""
    images: Float32[Tensor, "n 3 h w"] = frames_nchw.float()
    if grid is not None:
        images = F.grid_sample(images, grid.expand(len(images), -1, -1, -1), mode="bilinear", padding_mode="zeros", align_corners=False)
    if (images.shape[3], images.shape[2]) != target_wh:
        images = F.interpolate(images, size=(target_wh[1], target_wh[0]), mode="bilinear", antialias=True, align_corners=False)
    return images.round().clamp(0, 255).to(torch.uint8)


def valid_mask(grid: Float32[Tensor, "1 h w 2"] | None, stored_wh: tuple[int, int], target_wh: tuple[int, int]) -> Bool[Tensor, "h2 w2"]:
    """Output pixels whose source lies inside the stored frame (all of them without a lens)."""
    if grid is None:
        return torch.ones((target_wh[1], target_wh[0]), dtype=torch.bool)
    inside: Float32[Tensor, "1 1 h w"] = (grid.abs() <= 1.0).all(dim=-1).float()[:, None]
    if stored_wh != target_wh:
        inside = F.interpolate(inside, size=(target_wh[1], target_wh[0]), mode="area")
    return (inside[0, 0] > 0.999).cpu()


def _static_cell(table: pa.Table, column_name: str) -> object | None:
    if column_name not in table.column_names:
        return None
    return first_valid_value(table.column(column_name), allow_none=True, component_name=column_name)


def _camera_lens(table: pa.Table, entity: str) -> ViconRadialLens | None:
    """The camera node's lens: None for a pinhole camera; raises for a model this reader cannot undistort yet."""
    model: object | None = _static_cell(table, f"{entity}:camera_model")
    if isinstance(model, list):
        model = model[0] if model else None
    if model is None or model == "pinhole":
        return None
    if model != "vicon_radial_2":
        raise ValueError(f"{entity}: camera_model {model!r} has no undistortion here yet (vicon_radial_2 and pinhole only)")
    values: Float64[ndarray, " n"] = np.asarray(_static_cell(table, f"{entity}:vicon_radial_2"), dtype=np.float64).reshape(-1)
    width: object | None = _static_cell(table, f"{entity}:source_width")
    height: object | None = _static_cell(table, f"{entity}:source_height")
    if len(values) != 5 or width is None or height is None:
        raise ValueError(f"{entity}: vicon_radial_2 needs [cx, cy, k1, k2, k3] and source_width/height")
    return ViconRadialLens(
        centre=values[:2], radial=values[2:], native_width=int(np.asarray(width).reshape(-1)[0]), native_height=int(np.asarray(height).reshape(-1)[0])
    )


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
    for person, column_name in enumerate(vertex_columns):
        cells: list[object | None] = table.column(column_name).to_pylist()
        for slot, frame in enumerate(frames.tolist()):
            row: int | None = row_of.get(frame)
            cell: object | None = None if row is None else cells[row]
            if isinstance(cell, list) and len(cell) == num_vertices:
                vertices[slot, person] = np.asarray(cell, dtype=np.float32)
                present[slot, person] = True
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
    storage_device: torch.device | None = None,
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
        storage_device: Where ``frames`` live (default CPU: 32 cams x 120 frames at 1028x752 is 8.9 GB).

    Raises:
        ValueError: If a camera is missing or uncalibrated, a frame is outside the segment,
            or a camera's lens model has no undistortion here.
    """
    decode_on: torch.device = decode_device if decode_device is not None else torch.device("cuda")
    store_on: torch.device = storage_device if storage_device is not None else torch.device("cpu")
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
    fps_cell: object | None = _static_cell(static, "property:capture:fps")
    if fps_cell is None:
        raise ValueError(f"segment {segment_id} has no property:capture:fps")
    fps: float = float(np.asarray(fps_cell).reshape(-1)[0])

    # Keyframe flags of the whole segment (no samples) fix each camera's decode window.
    videos: list[str] = [camera.video_entity for camera in chosen]
    flag_table: pa.Table = (
        entry.filter_segments(segment_id)
        .filter_contents(videos)
        .reader(index=FRAME_TIMELINE)
        .select(FRAME_TIMELINE, *[f"{video}:VideoStream:is_keyframe" for video in videos])
        .to_arrow_table()
    )
    timeline: Int64[ndarray, " n"] = flag_table.column(FRAME_TIMELINE).to_numpy().astype(np.int64)
    frames: Int64[ndarray, " t"] = select_frame_indices(frame_start, num_frames, frame_step, timeline)
    window_start: int = int(frames.min())
    for video in videos:
        keyframe_times: Int64[ndarray, " k"] = timeline[
            np.asarray(catalog_keyframes(flag_table.column(f"{video}:VideoStream:is_keyframe")), dtype=bool)
        ]
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
        decoder: VideoDecoder = VideoDecoder(
            wrap_mp4(stream.samples[first:], stream.keyframes[first:], max(1, round(fps)), codec=stream.codec),
            device=decode_on,
            seek_mode="exact",
            num_ffmpeg_threads=0,
        )
        decoded: UInt8[Tensor, "t 3 h w"] = decoder.get_frames_at(positions).data
        if (decoded.shape[3], decoded.shape[2]) != stored_wh:
            raise ValueError(f"{video}: decoded {decoded.shape[3]}x{decoded.shape[2]}, the pinhole says {stored_wh}")
        lens: ViconRadialLens | None = _camera_lens(static, node)
        grid: Float32[Tensor, "1 h w 2"] | None = None if lens is None else undistortion_grid(lens, stored_wh, decode_on)
        # Float resampling costs 24 bytes per stored pixel and frame: chunk it to bound the GPU peak.
        out_frames.append(
            torch.cat(
                [
                    resample_frames(decoded[start : start + RESAMPLE_CHUNK], grid, target_wh).permute(0, 2, 3, 1).to(store_on)
                    for start in range(0, len(decoded), RESAMPLE_CHUNK)
                ]
            )
        )
        valid.append(valid_mask(grid, stored_wh, target_wh))
        k_matrix: Float64[ndarray, "3 3"] | None = models[node].pinhole.intrinsics.k_matrix
        assert k_matrix is not None  # Intrinsics always derives it
        K[index] = torch.from_numpy(scale_intrinsics(k_matrix, stored_wh, target_wh).astype(np.float32))
        world_T_cam[index] = torch.from_numpy(np.asarray(models[node].pinhole.extrinsics.world_T_cam, dtype=np.float32))
        del decoder, decoded, grid
    return MultiViewClip(
        camera_names=tuple(camera.name for camera in chosen),
        frames=tuple(out_frames),
        valid=tuple(valid),
        K=K,
        world_T_cam=world_T_cam,
        fps=fps,
        frame_indices=torch.from_numpy(frames),
        people=_read_people(entry, segment_id, frames, body_source),
    )
