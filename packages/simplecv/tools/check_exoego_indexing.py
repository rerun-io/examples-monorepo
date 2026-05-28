"""Smoke-test canonical timeline indexing for ExoEgo datasets.

Runs with the dev environment (beartype enabled), logs video/frame outputs to
Rerun, and fetches a few sampled frames for quick inspection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import tyro
from jaxtyping import Float32
from rerun import ColorModel

from simplecv.configs.exoego_dataset_configs import AnnotatedExoEgoDatasetUnion
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoSample
from simplecv.data.skeleton.coco_133 import COCO_133_ID2NAME, COCO_133_IDS, COCO_133_LINKS
from simplecv.rerun_custom_types import Points3DWithConfidence, confidence_scores_to_rgb
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole, log_video


@dataclass
class CheckIndexingConfig:
    """CLI options for the indexing smoke test."""

    dataset: AnnotatedExoEgoDatasetUnion
    """Dataset config to instantiate."""

    rr: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Rerun configuration; uses the shared Tyro-friendly config from rerun_log_utils."""

    max_frames: int | None = None
    """Limit the number of canonical frames to log; None logs all."""


def _make_side_by_side_tabs(
    *,
    ego_video_log_paths: list[Path] | None,
    exo_video_log_paths: list[Path] | None,
) -> rrb.ContainerLike:
    """Port of create_container from view_exoego.py with side-by-side video + sampled frame per camera."""

    main_view: rrb.ContainerLike = rrb.Spatial3DView(
        origin="/",
        name="3D View",
        spatial_information=rrb.SpatialInformation(show_axes=True),
    )

    if ego_video_log_paths:
        ego_tabs: list[rrb.Container | rrb.View] = []
        for video_log_path in ego_video_log_paths:
            # video_log_path ends with /world/.../pinhole/video; pinhole entity is parent
            pinhole_path = video_log_path.parent
            name: str = pinhole_path.parent.name
            ego_tabs.append(
                rrb.Horizontal(
                    contents=[
                        rrb.Spatial2DView(origin=str(pinhole_path), name=f"ego/{name} • video"),
                        rrb.Spatial2DView(origin=f"/sample/index/ego/{name}", name=f"ego/{name} • index sample"),
                    ],
                    name=f"ego/{name}",
                )
            )
        ego_view = rrb.Vertical(
            contents=[rrb.Tabs(contents=ego_tabs)],
        )
        main_view = rrb.Horizontal(contents=[main_view, ego_view], column_shares=[4, 1])

    if exo_video_log_paths:
        exo_tabs: list[rrb.Container | rrb.View] = []
        for video_log_path in exo_video_log_paths:
            pinhole_path = video_log_path.parent
            name: str = pinhole_path.parent.name
            exo_tabs.append(
                rrb.Horizontal(
                    contents=[
                        rrb.Spatial2DView(origin=str(pinhole_path), name=f"exo/{name} • video"),
                        rrb.Spatial2DView(origin=f"/sample/index/exo/{name}", name=f"exo/{name} • index sample"),
                    ],
                    name=f"exo/{name}",
                )
            )
        exo_view = rrb.Horizontal(contents=[rrb.Tabs(contents=exo_tabs)])
        main_view = rrb.Vertical(contents=[main_view, exo_view], row_shares=[4, 1])

    return main_view


def _set_annotation_context() -> None:
    """Register minimal COCO-133 metadata for label rendering."""
    rr.log(
        "/",
        rr.AnnotationContext(
            [
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=0, label="Coco Wholebody", color=(0, 0, 255)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=id, label=name) for id, name in COCO_133_ID2NAME.items()
                    ],
                    keypoint_connections=COCO_133_LINKS,
                ),
            ]
        ),
        static=True,
    )


def _log_cameras(exoego_sequence: BaseExoEgoSequence) -> None:
    """Log camera intrinsics/extrinsics following the schema."""
    # Exo: static
    if exoego_sequence.exo_sequence is not None:
        for cam in exoego_sequence.exo_sequence.exo_cam_list:
            cam_log_path = Path("/world") / "exo" / cam.name
            log_pinhole(
                cam,
                cam_log_path=cam_log_path,
                image_plane_distance=exoego_sequence.exo_sequence.image_plane_distance,
                static=True,
            )

    # Ego: dynamic transforms (log per frame; intrinsics repeated)
    if exoego_sequence.ego_sequence is not None:
        for cam_name, cam_list in exoego_sequence.ego_sequence.ego_cam_dict.items():
            if not cam_list:
                continue
            cam_log_path = Path("/world") / "ego" / cam_name
            n = min(len(cam_list), len(exoego_sequence.canonical_timestamps_ns))
            if n == 0:
                continue
            print(
                f"[ego {cam_name}] first trans {cam_list[0].extrinsics.cam_t_world} "
                f"last {cam_list[n - 1].extrinsics.cam_t_world}"
            )
            # log per-frame pinhole (intrinsics + transform)
            n = min(len(cam_list), len(exoego_sequence.canonical_timestamps_ns))
            if n == 0:
                continue
            for cam_dyn, ts_ns in zip(cam_list[:n], exoego_sequence.canonical_timestamps_ns[:n], strict=True):
                rr.set_time(timeline="video_time", duration=1e-9 * int(ts_ns))
                log_pinhole(
                    cam_dyn,
                    cam_log_path=cam_log_path,
                    image_plane_distance=exoego_sequence.ego_sequence.image_plane_distance,
                    static=False,
                )


def main(cfg: CheckIndexingConfig) -> None:
    exoego_sequence: BaseExoEgoSequence = cfg.dataset.setup()
    total_full: int = len(exoego_sequence)
    total: int = total_full if cfg.max_frames is None else min(total_full, cfg.max_frames)
    print(f"[info] canonical timeline length: {total_full}")
    print(f"[info] canonical end (ns): {exoego_sequence.canonical_end_ns}")
    if cfg.max_frames is not None and cfg.max_frames < total_full:
        print(f"[info] truncating to {total} frames via --max-frames")

    ego_entity_paths: list[Path] = []
    exo_entity_paths: list[Path] = []
    # Log video assets once so we can compare asset timeline vs sampled frames
    if exoego_sequence.ego_sequence is not None:
        for name, path in zip(
            exoego_sequence.ego_sequence.ego_video_names,
            exoego_sequence.ego_sequence.ego_video_paths,
            strict=True,
        ):
            entity_path = Path("/world") / "ego" / name / "pinhole" / "video"
            ego_entity_paths.append(entity_path)
            log_video(video_path=path, video_log_path=entity_path, timeline="video_time")

    if exoego_sequence.exo_sequence is not None:
        for name, path in zip(
            exoego_sequence.exo_sequence.exo_video_names,
            exoego_sequence.exo_sequence.exo_video_paths,
            strict=True,
        ):
            entity_path = Path("/world") / "exo" / name / "pinhole" / "video"
            exo_entity_paths.append(entity_path)
            log_video(video_path=path, video_log_path=entity_path, timeline="video_time")

    ego_paths: list[Path] | None = ego_entity_paths if ego_entity_paths else None
    exo_paths: list[Path] | None = exo_entity_paths if exo_entity_paths else None
    container: rrb.ContainerLike = _make_side_by_side_tabs(
        ego_video_log_paths=ego_paths,
        exo_video_log_paths=exo_paths,
    )
    blueprint = rrb.Blueprint(container, collapse_panels=True)
    rr.send_blueprint(blueprint)
    rr.log("/", exoego_sequence.world_coordinate_system, static=True)
    _set_annotation_context()
    _log_cameras(exoego_sequence)

    for idx, sample in enumerate(exoego_sequence):
        if idx >= total:
            break
        rr.set_time(timeline="video_time", duration=1e-9 * sample.canonical_timestamp_ns)
        if sample.ego_bgr_list is not None and exoego_sequence.ego_sequence is not None:
            for bgr, cam_name in zip(sample.ego_bgr_list, exoego_sequence.ego_sequence.ego_video_names, strict=True):
                rr.log(
                    f"/sample/index/ego/{cam_name}",
                    rr.Image(bgr, color_model=ColorModel.BGR).compress(jpeg_quality=80),
                )
        if sample.exo_bgr_list is not None and exoego_sequence.exo_sequence is not None:
            for bgr, cam_params in zip(sample.exo_bgr_list, exoego_sequence.exo_sequence.exo_cam_list, strict=True):
                rr.log(
                    f"/sample/index/exo/{cam_params.name}",
                    rr.Image(bgr, color_model=ColorModel.BGR).compress(jpeg_quality=80),
                )
        if sample.labels is not None:
            pts: Float32[np.ndarray, "133 3"] = sample.labels.xyzc_stack[0, :, :3].astype(np.float32, copy=False)
            conf: Float32[np.ndarray, "133"] = sample.labels.xyzc_stack[0, :, 3].astype(np.float32, copy=False)
            conf_rgb: np.ndarray = confidence_scores_to_rgb(conf[np.newaxis, :, np.newaxis])[0]
            rr.log(
                "/sample/index/labels/coco133",
                Points3DWithConfidence(
                    positions=pts,
                    confidences=conf,
                    class_ids=0,
                    keypoint_ids=COCO_133_IDS,
                    colors=conf_rgb,
                    radii=0.01,
                ),
            )

    # Log every canonical frame for full side-by-side playback
    print("[info] logging all canonical samples")
    for idx, ts_ns in enumerate(exoego_sequence.canonical_timestamps_ns):
        if idx >= total:
            break
        sample_all: ExoEgoSample = exoego_sequence[idx]
        rr.set_time(timeline="video_time", duration=1e-9 * int(ts_ns))
        if sample_all.ego_bgr_list is not None and exoego_sequence.ego_sequence is not None:
            for bgr, cam_name in zip(
                sample_all.ego_bgr_list, exoego_sequence.ego_sequence.ego_video_names, strict=True
            ):
                rr.log(
                    f"/sample/index/ego/{cam_name}",
                    rr.Image(bgr, color_model=ColorModel.BGR).compress(jpeg_quality=70),
                )
        if sample_all.exo_bgr_list is not None and exoego_sequence.exo_sequence is not None:
            for bgr, cam_params in zip(sample_all.exo_bgr_list, exoego_sequence.exo_sequence.exo_cam_list, strict=True):
                rr.log(
                    f"/sample/index/exo/{cam_params.name}",
                    rr.Image(bgr, color_model=ColorModel.BGR).compress(jpeg_quality=70),
                )
        if sample_all.labels is not None:
            pts_all: Float32[np.ndarray, "133 3"] = sample_all.labels.xyzc_stack[0, :, :3].astype(
                np.float32, copy=False
            )
            conf_all: Float32[np.ndarray, "133"] = sample_all.labels.xyzc_stack[0, :, 3].astype(
                np.float32, copy=False
            )
            colors_all: np.ndarray = confidence_scores_to_rgb(conf_all[np.newaxis, :, np.newaxis])[0]
            rr.log(
                "/sample/index/labels/coco133",
                Points3DWithConfidence(
                    positions=pts_all,
                    confidences=conf_all,
                    class_ids=0,
                    keypoint_ids=COCO_133_IDS,
                    colors=colors_all,
                    radii=0.01,
                ),
            )


if __name__ == "__main__":
    tyro.cli(main, description=__doc__)
