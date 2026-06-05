from __future__ import annotations

import atexit
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import rerun.experimental as rre
import torch
from jaxtyping import Float32, UInt8
from numpy import ndarray

from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.data.exo.base_exo import BaseExoSequence, ExoData
from simplecv.rerun_log_utils import (
    extract_asset_video_blob_fast,
    mux_h264_to_mp4,
    read_h264_samples_from_rrd,
)
from simplecv.rrd_query_utils import RRDQuerySession, first_valid_value
from simplecv.video_io import TorchCodecMultiVideoReader, rgb_chw_tensor_to_bgr_hwc

if TYPE_CHECKING:
    from simplecv.data.exoego.rrd_exoego import RRDExoEgoConfig
else:  # pragma: no cover - runtime alias to avoid circular import
    from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig as RRDExoEgoConfig


@dataclass(slots=True)
class _RRDCameraStream:
    """Metadata describing an exo camera stream discovered in an RRD file."""

    name: str
    video_entity: str
    pinhole_entity: str
    transform_entity: str
    data_kind: Literal["video_stream", "asset_video"]


class RRDExoSequence(BaseExoSequence[RRDExoEgoConfig]):
    """RRD-backed exo sequence using TorchCodec for fast in-memory video decoding.

    For asset_video data, extracts video bytes directly from the RRD recording
    and decodes in-memory using TorchCodec (~30x faster than disk-based remuxing).
    """

    _recording: rre.LazyStore | None = None
    _video_blobs: dict[str, bytes] | None = None

    def __init__(
        self,
        cfg: RRDExoEgoConfig,
        recording: rre.LazyStore | None = None,
        query_session: RRDQuerySession | None = None,
    ) -> None:
        self._recording = recording
        self._video_blobs = {}
        self._query_session = query_session or RRDQuerySession(cfg.rrd_path)
        # Call base class __init__ but we'll override the video reader setup
        super().__init__(cfg)

    def __getitem__(self, idx: int) -> ExoData:
        reader: TorchCodecMultiVideoReader = cast(TorchCodecMultiVideoReader, self.exo_video_readers)
        rgb_list: list[UInt8[torch.Tensor, "3 h w"]] = reader[idx]
        bgr_list: list[UInt8[ndarray, "H W 3"]] = [rgb_chw_tensor_to_bgr_hwc(rgb_chw) for rgb_chw in rgb_list]
        return ExoData(
            cam_params_list=cast(list[PinholeParameters], self.exo_cam_list),
            bgr_list=bgr_list,
            xyz=None,
            uv_dict=None,
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.exo_video_readers)

    @property
    def exo_video_names(self) -> list[str]:  # type: ignore[override]
        camera_streams: list[_RRDCameraStream] | None = getattr(self, "_camera_streams", None)
        if camera_streams:
            stream_names: list[str] = [stream.name for stream in camera_streams]
            return stream_names
        return super().exo_video_names

    def load_video_paths(self) -> list[Path]:
        """Load video blobs into memory and return placeholder paths.

        For asset_video data, this extracts bytes directly using fast pyarrow buffer
        access (~680x faster than as_py()) and stores them in self._video_blobs.
        The video reader is then created using TorchCodec with in-memory bytes.

        For video_stream data (H.264), falls back to disk-based remuxing.
        """
        rrd_path: Path = self.config.rrd_path
        assert rrd_path.exists(), f"RRD path {rrd_path} does not exist"

        assert self._recording is not None, "Recording must be provided by caller"
        schema = self._recording.schema()
        self._video_timeline: str = self._select_timeline(schema)
        self._camera_streams: list[_RRDCameraStream] = self._discover_camera_streams(schema)
        assert self._camera_streams, "No exo camera streams found in recording"
        video_blobs = self._video_blobs
        assert video_blobs is not None

        video_sources: list[Path | bytes] = []
        video_paths: list[Path] = []  # For compatibility with base class

        # Check if all streams are asset_video (can use fast in-memory path)
        all_asset_video: bool = all(
            stream.data_kind == "asset_video" for stream in self._camera_streams
        )

        if all_asset_video:
            # FAST PATH: Extract blobs directly without writing to disk
            for camera_stream in self._camera_streams:
                video_bytes: bytes = extract_asset_video_blob_fast(
                    video_entity=camera_stream.video_entity,
                    timeline=self._video_timeline,
                    query_session=self._query_session,
                )
                video_blobs[camera_stream.name] = video_bytes
                video_sources.append(video_bytes)
                # Create placeholder path for compatibility
                video_paths.append(Path(f"<rrd:{camera_stream.name}>"))

            # Create TorchCodec reader with in-memory bytes
            self.exo_video_readers = TorchCodecMultiVideoReader(video_sources)
        else:
            # SLOW PATH: Some streams need H.264 remuxing to disk
            self._remux_tmpdir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory(
                prefix="rrd_exo_remux_"
            )
            atexit.register(self._remux_tmpdir.cleanup)

            for camera_stream in self._camera_streams:
                mp4_path: Path = Path(self._remux_tmpdir.name) / f"{camera_stream.name}.mp4"
                match camera_stream.data_kind:
                    case "video_stream":
                        times, samples = read_h264_samples_from_rrd(
                            str(rrd_path),
                            camera_stream.video_entity,
                            self._video_timeline,
                        )
                        mux_h264_to_mp4(times, samples, str(mp4_path))
                    case "asset_video":
                        # Still use fast extraction, but write to disk for mixed mode
                        video_bytes = extract_asset_video_blob_fast(
                            video_entity=camera_stream.video_entity,
                            timeline=self._video_timeline,
                            query_session=self._query_session,
                        )
                        mp4_path.write_bytes(video_bytes)
                    case _:
                        raise ValueError(
                            f"Unsupported data kind for RRD camera stream: {camera_stream.data_kind}"
                        )

                assert mp4_path.exists(), f"Expected remuxed video at {mp4_path}"
                video_paths.append(mp4_path)
                video_sources.append(mp4_path)

            # Create TorchCodec reader with file paths
            self.exo_video_readers = TorchCodecMultiVideoReader(video_sources)

        return video_paths

    def load_exo_cams(self) -> list[PinholeParameters | None]:
        """Load camera parameters for each stream, returning None for uncalibrated cameras.

        Returns a list aligned with `exo_video_names` - cameras without valid
        intrinsics/extrinsics get `None` so the list stays synchronized with video readers.
        """
        assert self._recording is not None, "Recording must be provided by caller"
        recording: rre.LazyStore = self._recording
        schema = recording.schema()
        timeline: str = getattr(self, "_video_timeline", self._select_timeline(schema))
        camera_streams: list[_RRDCameraStream] = getattr(
            self,
            "_camera_streams",
            self._discover_camera_streams(schema),
        )
        assert camera_streams, "No exo camera streams found in recording"

        exo_cams: list[PinholeParameters | None] = []
        for camera_stream in camera_streams:
            try:
                intrinsics = self._load_intrinsics(camera_stream.pinhole_entity, timeline)
                extrinsics = self._load_extrinsics(camera_stream.transform_entity, timeline)
                exo_cams.append(PinholeParameters(name=camera_stream.name, intrinsics=intrinsics, extrinsics=extrinsics))
            except ValueError as exc:
                warnings.warn(
                    f"Camera '{camera_stream.name}' has no calibration data: {exc}",
                    stacklevel=2,
                )
                exo_cams.append(None)  # Keep slot for sync with video readers
        return exo_cams

    def _discover_camera_streams(self, schema: Any) -> list[_RRDCameraStream]:
        component_columns = schema.component_columns()
        descriptors: list[Any] = (
            list(component_columns.keys()) if isinstance(component_columns, dict) else list(component_columns)
        )

        stream_map: dict[str, _RRDCameraStream] = {}
        for descriptor in descriptors:
            entity_path = getattr(descriptor, "entity_path", None)
            component_name = getattr(descriptor, "component", None)
            if not isinstance(component_name, str) or entity_path is None:
                continue

            entity_str = str(entity_path).lstrip("/")
            if not entity_str.startswith("world/exo"):
                continue

            data_kind: Literal["video_stream", "asset_video"] | None = None
            if component_name.endswith("VideoStream:sample"):
                data_kind = "video_stream"
            elif component_name.endswith("AssetVideo:blob"):
                data_kind = "asset_video"

            if data_kind is None:
                continue

            video_entity = entity_str
            pinhole_entity = str(PurePosixPath(video_entity).parent)
            transform_entity = str(PurePosixPath(pinhole_entity).parent)
            camera_name = PurePosixPath(transform_entity).name

            stream = stream_map.get(video_entity)
            if stream is None:
                stream_map[video_entity] = _RRDCameraStream(
                    name=camera_name,
                    video_entity=video_entity,
                    pinhole_entity=pinhole_entity,
                    transform_entity=transform_entity,
                    data_kind=data_kind,
                )
            else:
                # Prefer video streams if both exist; otherwise keep detected kind.
                if stream.data_kind != data_kind and data_kind == "video_stream":
                    stream_map[video_entity] = _RRDCameraStream(
                        name=camera_name,
                        video_entity=video_entity,
                        pinhole_entity=pinhole_entity,
                        transform_entity=transform_entity,
                        data_kind=data_kind,
                    )

        camera_streams: list[_RRDCameraStream] = sorted(stream_map.values(), key=lambda stream: stream.name)
        return camera_streams

    def _select_timeline(self, schema: Any) -> str:
        timeline_names: list[str] = []
        try:
            for index_col in schema.index_columns():
                timeline_name = getattr(index_col, "name", None)
                if timeline_name is None:
                    timeline_name = str(index_col)
                timeline_names.append(str(timeline_name))
        except Exception:
            pass

        preferred_order: tuple[str, ...] = ("video_time", "time", "timestamp", "frame_time")
        for candidate in preferred_order:
            if candidate in timeline_names:
                return candidate
        if timeline_names:
            return timeline_names[0]
        raise AssertionError("No timeline columns found in recording schema")

    def _load_intrinsics(self, pinhole_entity: str, timeline: str) -> Intrinsics:
        selectors = [
            f"{pinhole_entity}:Pinhole:image_from_camera",
            f"{pinhole_entity}:Pinhole:camera_xyz",
            f"{pinhole_entity}:Pinhole:resolution",
        ]
        table = self._query_session.read_arrow(
            contents=pinhole_entity,
            selectors=selectors,
            index=None,
        )
        value_offset = 0

        k_value: Any | None = None
        camera_xyz_value: Any | None = None
        resolution_value: Any | None = None

        if table.num_rows > 0:
            k_value = first_valid_value(
                table.column(value_offset),
                component_name=f"{pinhole_entity}:Pinhole:image_from_camera",
            )
            camera_xyz_value = first_valid_value(
                table.column(value_offset + 1),
                allow_none=True,
                component_name=f"{pinhole_entity}:Pinhole:camera_xyz",
            )
            resolution_value = first_valid_value(
                table.column(value_offset + 2),
                allow_none=True,
                component_name=f"{pinhole_entity}:Pinhole:resolution",
            )

        if k_value is None:
            table = self._query_session.read_arrow(
                contents=pinhole_entity,
                selectors=selectors,
                index=timeline,
            )
            value_offset = 1
            if table.num_rows > 0:
                k_value = first_valid_value(
                    table.column(value_offset),
                    component_name=f"{pinhole_entity}:Pinhole:image_from_camera",
                )
                camera_xyz_value = first_valid_value(
                    table.column(value_offset + 1),
                    allow_none=True,
                    component_name=f"{pinhole_entity}:Pinhole:camera_xyz",
                )
                resolution_value = first_valid_value(
                    table.column(value_offset + 2),
                    allow_none=True,
                    component_name=f"{pinhole_entity}:Pinhole:resolution",
                )

        if k_value is None:
            raise ValueError(f"Missing image_from_camera for {pinhole_entity}")
        k_matrix: Float32[ndarray, "3 3"] = np.asarray(k_value, dtype=np.float32).reshape(3, 3, order="F")

        camera_conventions = "RDF"
        camera_xyz: ndarray | None = None
        if camera_xyz_value is not None:
            camera_xyz = np.asarray(camera_xyz_value, dtype=np.int32).reshape(-1)
        if camera_xyz is not None and camera_xyz.size == 3:
            axis_tuple = tuple(int(v) for v in camera_xyz)
            if axis_tuple == (3, 5, 2):
                camera_conventions = "RUB"

        width: int | None = None
        height: int | None = None
        resolution: ndarray | None = None
        if resolution_value is not None:
            resolution = np.asarray(resolution_value, dtype=np.float32).reshape(-1)
        if resolution is not None and resolution.size >= 2:
            width = int(round(float(resolution[0])))
            height = int(round(float(resolution[1])))

        if width is None:
            width = int(round(2 * float(k_matrix[0, 2])))
        if height is None:
            height = int(round(2 * float(k_matrix[1, 2])))

        return Intrinsics(
            camera_conventions=camera_conventions,
            fl_x=float(k_matrix[0, 0]),
            fl_y=float(k_matrix[1, 1]),
            cx=float(k_matrix[0, 2]),
            cy=float(k_matrix[1, 2]),
            width=width,
            height=height,
        )

    def _load_extrinsics(self, entity: str, timeline: str) -> Extrinsics:
        translation_value: list[float] | None = None
        rotation_value: list[float] | None = None

        table = self._query_session.read_arrow(
            contents=entity,
            selectors=[
                f"{entity}:Transform3D:translation",
                f"{entity}:Transform3D:mat3x3",
            ],
            index=None,
        )

        if table.num_rows > 0:
            translation_value = first_valid_value(
                table.column(0),
                component_name=f"{entity}:Transform3D:translation",
            )
            rotation_value = first_valid_value(
                table.column(1),
                component_name=f"{entity}:Transform3D:mat3x3",
            )

        if translation_value is None or rotation_value is None:
            table = self._query_session.read_arrow(
                contents=entity,
                selectors=[
                    f"{entity}:Transform3D:translation",
                    f"{entity}:Transform3D:mat3x3",
                ],
                index=timeline,
            )
            translation_value = translation_value or first_valid_value(
                table.column(1),
                component_name=f"{entity}:Transform3D:translation",
            )
            rotation_value = rotation_value or first_valid_value(
                table.column(2),
                component_name=f"{entity}:Transform3D:mat3x3",
            )

        translation_arr = np.array(translation_value, dtype=np.float32)
        if translation_arr.ndim > 1:
            translation_arr = translation_arr.reshape(-1)
        translation: Float32[ndarray, "3"] = translation_arr.astype(np.float32)

        rotation_arr = np.array(rotation_value, dtype=np.float32)
        if rotation_arr.ndim > 1:
            rotation_arr = rotation_arr.reshape(-1)
        rotation: Float32[ndarray, "3 3"] = rotation_arr.reshape(3, 3, order="F")
        return Extrinsics(cam_R_world=rotation, cam_t_world=translation)

    @property
    def image_plane_distance(self) -> int | float:
        return 0.1
