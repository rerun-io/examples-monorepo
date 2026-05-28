from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import rerun.experimental as rre
from jaxtyping import Float32, UInt8
from numpy import ndarray
from rerun.catalog import ComponentColumnDescriptor, IndexColumnDescriptor, Schema

from simplecv.camera_parameters import BrownConradyDistortion, Extrinsics, Intrinsics, PinholeParameters
from simplecv.data.ego.base_ego import BaseEgoSequence, CameraParam, CamNameType, EgoData
from simplecv.rerun_log_utils import extract_asset_video_blob_fast
from simplecv.rrd_query_utils import RRDQuerySession, first_valid_value
from simplecv.video_io import TorchCodecMultiVideoReader, TorchCodecVideoReader

if TYPE_CHECKING:
    from simplecv.data.exoego.rrd_exoego import RRDExoEgoConfig
else:  # pragma: no cover - runtime alias to avoid circular import
    from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig as RRDExoEgoConfig


AXIS_CODES: dict[int, str] = {1: "U", 2: "D", 3: "R", 4: "L", 5: "F", 6: "B"}
_DISTORTION_MODEL_COMPONENT = "simplecv.components.DistortionModel"
_DISTORTION_COEFF_COMPONENT = "simplecv.components.DistortionCoefficients"


@dataclass(slots=True)
class _RRDEgoCameraStream:
    """Metadata describing an ego camera stream discovered in an RRD file."""

    name: str
    video_entity: str
    pinhole_entity: str
    transform_entity: str
    data_kind: Literal["video_stream", "asset_video"]


class RRDEgoSequence(BaseEgoSequence[RRDExoEgoConfig]):
    """RRD-backed ego sequence using TorchCodec for fast in-memory video decoding.

    Extracts video bytes directly from the RRD recording and decodes in-memory
    using TorchCodec (~30x faster than disk-based remuxing).
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
        super().__init__(cfg)

    def load_video_paths(self) -> list[Path]:
        """Load video blobs into memory and return placeholder paths.

        Extracts bytes directly using fast pyarrow buffer access (~680x faster
        than as_py()) and stores them in self._video_blobs.
        """
        assert self._recording is not None, "Recording must be provided by caller"
        recording: rre.LazyStore = self._recording
        schema: Schema = recording.schema()
        timelines: list[IndexColumnDescriptor] = list(schema.index_columns())
        # make sure the timeline exists
        timeline_name = "video_time"
        has_timeline: bool = any(timeline.name == timeline_name for timeline in timelines)
        assert has_timeline, f"RRD recording is missing expected timeline: {timeline_name}"

        rrd_path: Path = self.config.rrd_path
        assert rrd_path.exists(), f"RRD path {rrd_path} does not exist"

        # Extract video blobs directly into memory (FAST PATH)
        video_sources: list[bytes] = []
        video_path_map: dict[str, Path] = {}

        for cam_name in self.cam_names:
            video_entity: str = f"world/ego/{cam_name}/pinhole/video"
            video_bytes: bytes = extract_asset_video_blob_fast(
                video_entity=video_entity,
                timeline=timeline_name,
                query_session=self._query_session,
            )
            self._video_blobs[cam_name] = video_bytes
            video_sources.append(video_bytes)
            # Create placeholder path for compatibility
            video_path_map[cam_name] = Path(f"<rrd:{cam_name}>")

        self._video_path_map: dict[str, Path] = video_path_map

        # Create TorchCodec reader with in-memory bytes
        # Note: We set this directly here since base class will try to create
        # MultiVideoReader with the returned paths
        self._video_sources: list[bytes] = video_sources

        ordered_paths: list[Path] = [video_path_map[cam_name] for cam_name in self.cam_names]
        return ordered_paths

    def load_ego_cams(self) -> dict[CamNameType, list[CameraParam]]:
        assert self._recording is not None, "Recording must be provided by caller"
        recording: rre.LazyStore = self._recording
        schema: Schema = recording.schema()
        timelines: list[IndexColumnDescriptor] = list(schema.index_columns())
        # Component Columns
        components: list[ComponentColumnDescriptor] = list(schema.component_columns())

        # make sure the timeline exsits
        timeline_name = "video_time"
        has_timeline: bool = any(timeline.name == timeline_name for timeline in timelines)
        assert has_timeline, f"RRD recording is missing expected timeline: {timeline_name}"

        # make sure that /world/ego exists
        entity_paths: list[str] = sorted({component.entity_path for component in components})
        ego_entity_path = Path("/world/ego")
        has_ego = any(p == ego_entity_path or p.startswith(f"{ego_entity_path}/") for p in entity_paths)
        assert has_ego, "RRD recording is missing expected /world/ego entity path"

        ego_video_paths: set[Path] = set()
        # get all the intrinsics camera components
        for component in components:
            if component.archetype == "rerun.archetypes.AssetVideo" and (
                component.entity_path == ego_entity_path or component.entity_path.startswith(f"{ego_entity_path}/")
            ):
                ego_video_paths.add(Path(component.entity_path))

        print(f"Ego video paths: {ego_video_paths}")
        self.cam_names: set[str] = {p.parent.parent.name for p in ego_video_paths}
        print(f"Ego camera names: {self.cam_names}")

        ego_cam_dict: dict[str, list[CameraParam]] = {}
        for cam_name in self.cam_names:
            pinhole_entity: Path = ego_entity_path / cam_name / "pinhole"
            transform_entity: Path = ego_entity_path / cam_name
            try:
                intrinsics: Intrinsics = self._load_intrinsics(pinhole_entity, timeline_name)
                distortion: BrownConradyDistortion | None = self._load_distortion(pinhole_entity, timeline_name)
            except ValueError as exc:
                warnings.warn(
                    (
                        f"\033[33mSkipping ego camera '{cam_name}' due to missing metadata: {exc}. "
                        "Video frames are still remuxed but remain unlogged until intrinsics are available.\033[0m"
                    ),
                    stacklevel=2,
                )
                continue

            # this was cam_R_world ect before, for hocap it changed. We need to revalidate
            world_R_cam_batch, world_t_cam_batch = self._load_extrinsics_series(str(transform_entity), timeline_name)
            min_len: int = min(len(world_R_cam_batch), len(world_t_cam_batch))
            if min_len == 0:
                translation_default: Float32[ndarray, "3"] = np.zeros(3, dtype=np.float32)
                rotation_default: Float32[ndarray, "3 3"] = np.eye(3, dtype=np.float32)
                extrinsics_default = Extrinsics(cam_R_world=rotation_default, cam_t_world=translation_default)
                ego_cam_dict[cam_name] = [
                    PinholeParameters(
                        name=cam_name,
                        intrinsics=intrinsics,
                        extrinsics=extrinsics_default,
                        distortion=distortion,
                    )
                ]
                continue

            cam_params: list[CameraParam] = []
            for idx in range(min_len):
                rotation_mat: Float32[ndarray, "3 3"] = world_R_cam_batch[idx]
                translation_vec: Float32[ndarray, "3"] = world_t_cam_batch[idx]
                # NOTE: This legacy ego RRD path assumes the logged transform is world_T_cam
                # (parent-from-child / world_from_cam), which matches the HOCAP recording we tested.
                # If another RRD encodes cam_T_world instead (child-from-parent / cam_from_world),
                # this is the constructor to flip to Extrinsics(cam_R_world=..., cam_t_world=...).
                extrinsics = Extrinsics(world_R_cam=rotation_mat, world_t_cam=translation_vec)
                cam_params.append(
                    PinholeParameters(
                        name=cam_name,
                        intrinsics=intrinsics,
                        extrinsics=extrinsics,
                        distortion=distortion,
                    )
                )
            if cam_params:
                ego_cam_dict[cam_name] = cam_params

        return cast(dict[CamNameType, list[CameraParam]], ego_cam_dict)

    def align_cams_and_videos(
        self,
        video_path_list: list[Path],
        ego_cam_dict: dict[CamNameType, list[CameraParam]],
    ) -> tuple[dict[CamNameType, list[CameraParam]], dict[CamNameType, Path]]:
        """Align camera params with video streams and create the TorchCodec reader."""
        # Extract camera names from paths (handling both placeholder and real paths)
        video_by_name: dict[str, Path] = {}
        for path in video_path_list:
            stem: str = path.stem
            # Handle placeholder paths like "<rrd:cam_name>"
            if stem.startswith("<rrd:") and stem.endswith(">"):
                cam_name_extracted: str = stem[5:-1]  # Strip "<rrd:" and ">"
                video_by_name[cam_name_extracted] = path
            else:
                video_by_name[stem] = path
        assert video_by_name, "No ego videos were produced"

        aligned_cam_dict: dict[str, list[CameraParam]] = {}
        aligned_video_map: dict[str, Path] = {}
        aligned_sources: list[bytes] = []

        for cam_name, cam_params in ego_cam_dict.items():
            video_path = video_by_name.get(cam_name)
            if video_path is None:
                continue

            # Get video length from blob or file
            video_blob: bytes | None = self._video_blobs.get(cam_name) if self._video_blobs else None
            if video_blob is not None:
                reader = TorchCodecVideoReader(video_blob)
                aligned_sources.append(video_blob)
            else:
                reader = TorchCodecVideoReader(video_path)
                aligned_sources.append(video_path.read_bytes())

            video_len: int = len(reader)
            if not cam_params:
                continue

            if len(cam_params) < video_len:
                last_param: CameraParam = cam_params[-1]
                cam_params = cam_params + [last_param] * (video_len - len(cam_params))
            elif len(cam_params) > video_len:
                cam_params = cam_params[:video_len]

            aligned_cam_dict[cam_name] = cam_params
            aligned_video_map[cam_name] = video_path

        assert aligned_cam_dict, "No ego cameras aligned with the recorded videos"

        ordered_names: list[str] = sorted(aligned_video_map.keys())
        ordered_cam_dict: dict[str, list[CameraParam]] = {name: aligned_cam_dict[name] for name in ordered_names}
        ordered_video_map: dict[str, Path] = {name: aligned_video_map[name] for name in ordered_names}

        # Create TorchCodec reader with aligned sources
        ordered_sources: list[bytes] = [self._video_blobs[name] for name in ordered_names if self._video_blobs and name in self._video_blobs]
        if ordered_sources:
            self.ego_video_readers = TorchCodecMultiVideoReader(ordered_sources)

        return (
            cast(dict[CamNameType, list[CameraParam]], ordered_cam_dict),
            cast(dict[CamNameType, Path], ordered_video_map),
        )

    def __getitem__(self, idx: int) -> EgoData:
        cam_params_list: list[CameraParam] = [cam_list[idx] for cam_list in self._ego_cam_dict.values()]
        return EgoData(
            cam_params_list=cam_params_list,
            bgr_list=self.ego_video_readers[idx],
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.ego_video_readers)

    @property
    def ego_video_names(self) -> list[str]:  # type: ignore[override]
        return sorted(self.cam_names)

    @property
    def image_plane_distance(self) -> int | float:
        return 0.02

    def _load_intrinsics(self, pinhole_entity: Path, timeline: str) -> Intrinsics:
        selectors = [
            f"{pinhole_entity}:Pinhole:image_from_camera",
            f"{pinhole_entity}:Pinhole:camera_xyz",
            f"{pinhole_entity}:Pinhole:resolution",
        ]
        table = self._query_session.read_arrow(
            contents=str(pinhole_entity),
            selectors=selectors,
            index=None,
        )
        value_offset = 0
        if table.num_rows == 0:
            table = self._query_session.read_arrow(
                contents=str(pinhole_entity),
                selectors=selectors,
                index=timeline,
            )
            value_offset = 1

        if table.num_rows == 0:
            raise ValueError(f"No intrinsics found for {pinhole_entity}")

        k_list = first_valid_value(
            table.column(value_offset),
            component_name=f"{pinhole_entity}:Pinhole:image_from_camera",
        )
        xyz_list = first_valid_value(
            table.column(value_offset + 1),
            component_name=f"{pinhole_entity}:Pinhole:camera_xyz",
        )
        res_list = first_valid_value(
            table.column(value_offset + 2),
            component_name=f"{pinhole_entity}:Pinhole:resolution",
        )

        image_from_camera: Float32[ndarray, "3 3"] = np.asarray(k_list, dtype=np.float32).reshape(3, 3, order="F")
        camera_xyz: UInt8[ndarray, "3"] = np.asarray(xyz_list, dtype=np.uint8).reshape(3)
        resolution: Float32[ndarray, "2"] = np.asarray(res_list, dtype=np.float32).reshape(2)

        axes_label: str = "".join(AXIS_CODES[int(v)] for v in camera_xyz)
        assert axes_label in {"RDF"}, f"Unexpected camera axes: {axes_label}"

        return Intrinsics.from_k_matrix(
            k_matrix=image_from_camera,
            camera_conventions=axes_label,
            height=int(resolution[1]),
            width=int(resolution[0]),
        )

    def _load_distortion(
        self,
        pinhole_entity: Path,
        timeline: str,
    ) -> BrownConradyDistortion | None:
        """Load optional Brown–Conrady distortion components if present."""
        model_path: str = f"{pinhole_entity}:{_DISTORTION_MODEL_COMPONENT}"
        coeff_path: str = f"{pinhole_entity}:{_DISTORTION_COEFF_COMPONENT}"

        table = self._query_session.read_arrow(
            contents=str(pinhole_entity),
            selectors=[model_path, coeff_path],
            index=None,
            allow_missing=True,
        )
        value_offset = 0
        if table.num_rows == 0:
            table = self._query_session.read_arrow(
                contents=str(pinhole_entity),
                selectors=[model_path, coeff_path],
                index=timeline,
                allow_missing=True,
            )
            value_offset = 1

        if table.num_rows == 0:
            return None

        model_raw: Literal["brown_conrady"] | None = first_valid_value(
            table.column(value_offset),
            allow_none=True,
            component_name=model_path,
        )
        coeffs_raw: list[float] | None = first_valid_value(
            table.column(value_offset + 1),
            allow_none=True,
            component_name=coeff_path,
        )

        if model_raw is None and coeffs_raw is None:
            return None
        else:
            assert model_raw is not None, "Distortion model is missing though coefficients are present"
            assert coeffs_raw is not None, "Distortion coefficients are missing though model is present"
            coeffs_arr: Float32[ndarray, "14"] = np.asarray(coeffs_raw, dtype=np.float32).flatten()
            assert model_raw == "brown_conrady", f"Unsupported distortion model: {model_raw}"

            # Helper to guard missing trailing coefficients.
            def _safe(idx: int) -> float:
                return float(coeffs_arr[idx]) if idx < len(coeffs_arr) else 0.0

            return BrownConradyDistortion(
                k1=_safe(0),
                k2=_safe(1),
                p1=_safe(2),
                p2=_safe(3),
                k3=_safe(4),
                k4=_safe(5),
                k5=_safe(6),
                k6=_safe(7),
                s1=_safe(8),
                s2=_safe(9),
                s3=_safe(10),
                s4=_safe(11),
                tau_x=_safe(12),
                tau_y=_safe(13),
            )

    def _load_extrinsics_series(
        self,
        entity: str,
        timeline: str,
    ) -> tuple[Float32[ndarray, "n 3 3"], Float32[ndarray, "n 3"]]:
        table = self._query_session.read_arrow(
            contents=entity,
            selectors=[
                f"{entity}:Transform3D:mat3x3",
                f"{entity}:Transform3D:translation",
            ],
            index=timeline,
        )

        cam_R_world_list: list[Float32[ndarray, "3 3"]] = [
            np.asarray(mat, np.float32).reshape(3, 3, order="F")
            for mat in table.column(1).to_pylist()
            if mat is not None
        ]
        cam_t_world_list: list[Float32[ndarray, "3"]] = [
            np.asarray(translation, np.float32).reshape(-1)[:3]
            for translation in table.column(2).to_pylist()
            if translation is not None
        ]

        if not cam_R_world_list or not cam_t_world_list:
            default_R: Float32[ndarray, "1 3 3"] = np.eye(3, dtype=np.float32)[np.newaxis, ...]
            default_t: Float32[ndarray, "1 3"] = np.zeros((1, 3), dtype=np.float32)
            return default_R, default_t

        cam_R_world_batch: Float32[ndarray, "n 3 3"] = np.stack(cam_R_world_list, axis=0)
        cam_t_world_batch: Float32[ndarray, "n 3"] = np.stack(cam_t_world_list, axis=0)
        return cam_R_world_batch, cam_t_world_batch
