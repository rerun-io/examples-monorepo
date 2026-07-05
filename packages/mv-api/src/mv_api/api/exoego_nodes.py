from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rerun as rr
from jaxtyping import Int
from numpy import ndarray
from simplecv.apis.view_exoego import SceneSetupResult, setup_scene
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig
from simplecv.rerun_rig_logger import log_rig_static

from mv_api.api import full_exoego_pipeline
from mv_api.api.full_exoego_pipeline import RRDPipelineConfig


@dataclass(slots=True)
class ScenePreparationConfig:
    """Configuration for preparing and logging the exo/ego scene shell."""

    log_ego: bool = True
    """Whether ego video streams should be logged during scene setup."""
    log_exo: bool = True
    """Whether exo video streams should be logged during scene setup."""
    verbose: bool = False
    """Whether the node should log additional intermediate diagnostics."""


@dataclass(slots=True)
class ScenePreparationResult:
    """Dataset-derived scene inputs required by downstream exo/ego stages."""

    exoego_sequence: BaseExoEgoSequence[Any]
    """Dataset sequence returned by the selected exo/ego dataset config."""
    scene_setup_result: SceneSetupResult
    """Scene setup result containing Rerun log paths and synchronized timestamps."""
    parent_log_path: Path
    """Root Rerun entity path used for the exo/ego scene."""
    timeline: str
    """Timeline name used for frame-aligned exo/ego logs."""
    shortest_timestamp: Int[ndarray, "n_frames"]
    """Common timestamp sequence produced by scene setup."""
    exo_cam_paths: dict[str, Path]
    """Rig-layout camera node path per exo stream name (exoego:v2 layout)."""
    ego_cam_paths: dict[str, Path]
    """Rig-layout camera node path per ego stream name (exoego:v2 layout)."""


class ScenePreparationNode:
    """Prepare the dataset scene and log the static Rerun scene shell."""

    def __init__(
        self,
        *,
        config: ScenePreparationConfig,
        parent_log_path: Path,
        timeline: str,
    ) -> None:
        self.config: ScenePreparationConfig = config
        self.parent_log_path: Path = parent_log_path
        self.timeline: str = timeline

    def __call__(
        self,
        *,
        dataset: BaseExoEgoDatasetConfig,
        recording: rr.RecordingStream | None,
    ) -> ScenePreparationResult:
        exoego_sequence: BaseExoEgoSequence[Any] = dataset.setup()
        rr.log("/", exoego_sequence.world_coordinate_system, static=True, recording=recording)
        full_exoego_pipeline.set_annotation_context(recording=recording)

        # COLMAP-style rig layout (exoego:v2): assign cameras to rig nodes and
        # log the static skeletons before ingesting the per-frame videos.
        rig_layout = exoego_sequence.build_rig_layout(
            world_path=self.parent_log_path, log_exo=self.config.log_exo, log_ego=self.config.log_ego
        )
        for rig in rig_layout.rigs:
            log_rig_static(rig, world_path=str(self.parent_log_path), recording=recording)

        scene_setup_result: SceneSetupResult = setup_scene(
            exoego_sequence,
            rig_layout=rig_layout,
            parent_log_path=self.parent_log_path,
            timeline=self.timeline,
            log_ego=self.config.log_ego,
            log_exo=self.config.log_exo,
            recording=recording,
        )
        full_exoego_pipeline.send_scene_blueprint(log_paths=scene_setup_result.log_paths, recording=recording)

        return ScenePreparationResult(
            exoego_sequence=exoego_sequence,
            scene_setup_result=scene_setup_result,
            parent_log_path=self.parent_log_path,
            timeline=self.timeline,
            shortest_timestamp=scene_setup_result.shortest_timestamp,
            exo_cam_paths=dict(rig_layout.exo_cam_paths),
            ego_cam_paths=dict(rig_layout.ego_cam_paths),
        )


@dataclass(slots=True)
class ModelBackedPipelineConfig:
    """Configuration for the model-backed full exo/ego stage."""

    verbose: bool = False
    """Whether the node should log additional intermediate diagnostics."""


@dataclass(slots=True)
class ModelBackedPipelineResult:
    """Summary of the model-backed exo/ego stage execution."""

    processed_timestamps: Int[ndarray, "n_frames"]
    """Common timestamps that were available to the model-backed stage."""


class ModelBackedPipelineNode:
    """Run the model-backed stage behind a node-shaped interface."""

    def __init__(
        self,
        *,
        config: ModelBackedPipelineConfig,
        parent_log_path: Path,
        timeline: str,
    ) -> None:
        self.config: ModelBackedPipelineConfig = config
        self.parent_log_path: Path = parent_log_path
        self.timeline: str = timeline

    def __call__(
        self,
        *,
        pipeline_config: RRDPipelineConfig,
        scene: ScenePreparationResult,
        recording: rr.RecordingStream | None,
    ) -> ModelBackedPipelineResult:
        full_exoego_pipeline.run_model_backed_pipeline(
            config=pipeline_config,
            exoego_sequence=scene.exoego_sequence,
            scene_setup_result=scene.scene_setup_result,
            parent_log_path=self.parent_log_path,
            timeline=self.timeline,
            exo_cam_paths=scene.exo_cam_paths,
            ego_cam_paths=scene.ego_cam_paths,
            recording=recording,
        )
        return ModelBackedPipelineResult(processed_timestamps=scene.shortest_timestamp)
