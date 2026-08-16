"""Shared catalog-layer contracts for gauss-surf."""

from typing import Final, Literal, TypeAlias

from arkitscenes_download.ingest.paths import CAM_ULTRAWIDE, DEPTH_PROMPTDA, FRAME_SELECTION_ULTRAWIDE, FRAME_SELECTION_WIDE, PINHOLE_WIDE, RIG

PROMPTDA_LAYER: Final[str] = "promptda"
FRAME_SELECTION_LAYER: Final[str] = "frame_selection"
MOGE_NORMALS_LAYER: Final[str] = "moge_normals"
ULTRAWIDE_DEPTH_LAYER: Final[str] = "ultrawide_depth"
ULTRAWIDE_NORMALS_LAYER: Final[str] = "ultrawide_normals"
SPLAT_LAYER: Final[str] = "splat"
SPLAT_DEPTH_LAYER: Final[str] = "splat_depth"
SPLAT_TRIAGE_LAYER: Final[str] = "splat_triage"

LAYERS: Final[dict[str, str]] = {
    PROMPTDA_LAYER: "data/promptda/{video_id}.rrd",
    FRAME_SELECTION_LAYER: "data/frame_selection/{video_id}.rrd",
    MOGE_NORMALS_LAYER: "data/moge_normals/{video_id}.rrd",
    ULTRAWIDE_DEPTH_LAYER: "data/ultrawide_signals/{video_id}/ultrawide_depth.rrd",
    ULTRAWIDE_NORMALS_LAYER: "data/ultrawide_signals/{video_id}/ultrawide_normals.rrd",
    SPLAT_LAYER: "data/splat/{video_id}.rrd",
    SPLAT_DEPTH_LAYER: "data/splat_depth/{video_id}.rrd",
    SPLAT_TRIAGE_LAYER: "data/splat_triage/{video_id}.rrd",
}
"""Derived layer name to local per-segment recovery path."""

WIDE_FPS: Final[float] = 60.0
"""Nominal wide-camera packet rate."""
ULTRAWIDE_FPS: Final[float] = 10.0
"""Nominal ultrawide-camera packet rate."""
MOGE_INFERENCE_BATCH_SIZE: Final[int] = 8
"""Fixed TensorRT batch size for both MoGe normal stages."""
RGB_JPEG_QUALITY: Final[int] = 90
"""JPEG quality used for fitted and rectified RGB products."""
RENDER_BACKGROUND: Final[tuple[float, float, float]] = (0.1490, 0.1647, 0.2157)
"""Fixed RGB background shared by holdout evaluation and publication."""

WIDE_CHOSEN_SHARPNESS_COLUMN: Final[str] = f"/{FRAME_SELECTION_WIDE}:sharpness"
"""Sparse chosen-wide-frame component."""
ULTRAWIDE_CHOSEN_SHARPNESS_COLUMN: Final[str] = f"/{FRAME_SELECTION_ULTRAWIDE}:sharpness"
"""Sparse chosen-ultrawide-frame component."""
WIDE_INTRINSICS_COLUMN: Final[str] = f"/{PINHOLE_WIDE}:Pinhole:image_from_camera"
"""Temporal native wide-camera intrinsic matrix component."""
WIDE_RESOLUTION_COLUMN: Final[str] = f"/{PINHOLE_WIDE}:Pinhole:resolution"
"""Temporal native wide-camera resolution component."""
RIG_TRANSLATION_COLUMN: Final[str] = f"/{RIG}:Transform3D:translation"
"""Temporal rig translation component."""
RIG_QUATERNION_COLUMN: Final[str] = f"/{RIG}:Transform3D:quaternion"
"""Temporal rig quaternion component."""
ULTRAWIDE_TRANSLATION_COLUMN: Final[str] = f"/{CAM_ULTRAWIDE}:Transform3D:translation"
"""Static ultrawide-to-wide translation component."""
ULTRAWIDE_QUATERNION_COLUMN: Final[str] = f"/{CAM_ULTRAWIDE}:Transform3D:quaternion"
"""Static ultrawide-to-wide quaternion component."""
PROMPTDA_DEPTH_BLOB_COLUMN: Final[str] = f"/{DEPTH_PROMPTDA}:EncodedDepthImage:blob"
"""PromptDA encoded depth blob component."""

CameraTag: TypeAlias = Literal["wide", "uw"]
