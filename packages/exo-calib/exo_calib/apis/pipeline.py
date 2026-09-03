"""The exo-calib pipeline: Stage A once, two passes of Stage B + refinement, then the report.

The second Stage B pass tracks person boxes through the refined rig instead of
the Stage A estimate, which tightens the 2D observations the second refinement
consumes. Both passes replace the same ``exocalib_kp2d`` / ``exocalib_refined``
layers, so the catalog ends up holding the final pass only. The individual
stage tools remain available for diagnosis.
"""

from dataclasses import dataclass, field

import tyro

from exo_calib.apis import calibrate_init, keypoints2d, refine, report
from exo_calib.catalog_io import StageConfig
from exo_calib.layer_io import CALIBRATION_VARIANTS


@dataclass
class PipelineConfig:
    """Config for the full calibration pipeline: the shared stage fields plus each stage's own knob."""

    stage: tyro.conf.OmitArgPrefixes[StageConfig] = field(default_factory=StageConfig)
    """Shared stage flags (catalog, dataset, segment, rigs, output, register); the CLI shows them unprefixed."""
    frame_index: int = 0
    """Sample index of the synchronized frame Stage A calibrates from."""
    batch_size: int = 32
    """Largest inference batch the Stage B TensorRT engines are built for."""


def main(config: PipelineConfig) -> None:
    """Run every stage in order, registering each layer into the catalog."""
    print("=== Stage A: initial calibration ===")
    calibrate_init.main(calibrate_init.InitCalibrationConfig(stage=config.stage, frame_index=config.frame_index))
    for camera_variant in CALIBRATION_VARIANTS:
        print(f"\n=== Stage B: 2D keypoints, boxes tracked through the {camera_variant} rig ===")
        keypoints2d.main(keypoints2d.Keypoints2dConfig(stage=config.stage, batch_size=config.batch_size), camera_variant=camera_variant)
        print("\n=== Stage C+D: refinement ===")
        refine.main(config.stage)
    print("\n=== Stage E: report ===")
    report.main(config.stage)
