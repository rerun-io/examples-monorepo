from dataclasses import dataclass, field
from pathlib import Path

import rerun as rr

from mv_api.api.exoego_nodes import (
    ModelBackedPipelineConfig,
    ModelBackedPipelineNode,
    ModelBackedPipelineResult,
    ScenePreparationConfig,
    ScenePreparationNode,
    ScenePreparationResult,
)
from mv_api.api.full_exoego_pipeline import RRDPipelineConfig


@dataclass(slots=True)
class FullExoEgoRunResult:
    """Summary returned by the node-based full exo/ego app."""

    rrd_path: Path | None
    """Path to the saved RRD when the run was configured with ``RerunTyroConfig.save``."""
    parent_log_path: Path
    """Root Rerun entity path used for the exo/ego scene."""
    timeline: str
    """Timeline name used for frame-aligned exo/ego logs."""
    scene: ScenePreparationResult
    """Scene preparation output passed into model-backed stages."""
    model_backed: ModelBackedPipelineResult
    """Model-backed stage summary."""


@dataclass(slots=True)
class FullExoEgoApp:
    """Composite app that wires node-shaped exo/ego stages together."""

    scene_config: ScenePreparationConfig = field(default_factory=ScenePreparationConfig)
    """Configuration for the scene preparation node."""
    model_backed_config: ModelBackedPipelineConfig = field(default_factory=ModelBackedPipelineConfig)
    """Configuration for the model-backed stage node."""
    parent_log_path: Path = Path("world")
    """Root Rerun entity path used for the exo/ego scene."""
    timeline: str = "video_time"
    """Timeline name used for frame-aligned exo/ego logs."""

    def __call__(self, *, config: RRDPipelineConfig) -> FullExoEgoRunResult:
        recording: rr.RecordingStream = config.rr_config.rec_stream
        scene_node: ScenePreparationNode = ScenePreparationNode(
            config=self.scene_config,
            parent_log_path=self.parent_log_path,
            timeline=self.timeline,
        )
        model_backed_node: ModelBackedPipelineNode = ModelBackedPipelineNode(
            config=self.model_backed_config,
            parent_log_path=self.parent_log_path,
            timeline=self.timeline,
        )

        scene: ScenePreparationResult = scene_node(dataset=config.dataset, recording=recording)
        model_backed: ModelBackedPipelineResult = model_backed_node(
            pipeline_config=config,
            scene=scene,
            recording=recording,
        )
        recording.flush(timeout_sec=30.0)

        return FullExoEgoRunResult(
            rrd_path=config.rr_config.save,
            parent_log_path=self.parent_log_path,
            timeline=self.timeline,
            scene=scene,
            model_backed=model_backed,
        )


def run_full_exoego_app(config: RRDPipelineConfig) -> FullExoEgoRunResult:
    """Run the node-based full exo/ego composite app."""
    app: FullExoEgoApp = FullExoEgoApp()
    return app(config=config)


def main(config: RRDPipelineConfig) -> None:
    """Tyro-compatible entrypoint for the node-based full exo/ego app."""
    run_full_exoego_app(config=config)
