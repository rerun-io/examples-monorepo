"""Public schema contracts shared by gauss-surf stages."""

from gauss_surf.contracts import (
    FRAME_SELECTION_LAYER,
    LAYERS,
    SPLAT_DEPTH_LAYER,
    SPLAT_LAYER,
    SPLAT_TRIAGE_LAYER,
    ULTRAWIDE_FPS,
    WIDE_FPS,
)


def test_layers_are_the_single_recovery_path_spec() -> None:
    """Every derived catalog layer has one stable name and local recovery path."""
    assert LAYERS == {
        "promptda": "data/promptda/{video_id}.rrd",
        "frame_selection": "data/frame_selection/{video_id}.rrd",
        "moge_normals": "data/moge_normals/{video_id}.rrd",
        "ultrawide_depth": "data/ultrawide_signals/{video_id}/ultrawide_depth.rrd",
        "ultrawide_normals": "data/ultrawide_signals/{video_id}/ultrawide_normals.rrd",
        "splat": "data/splat/{video_id}.rrd",
        "splat_depth": "data/splat_depth/{video_id}.rrd",
        "splat_triage": "data/splat_triage/{video_id}.rrd",
    }
    assert FRAME_SELECTION_LAYER in LAYERS
    assert SPLAT_LAYER in LAYERS
    assert SPLAT_DEPTH_LAYER in LAYERS
    assert SPLAT_TRIAGE_LAYER in LAYERS


def test_shared_camera_rates_match_the_source_streams() -> None:
    """Selection and exact-decode stages use one nominal rate per camera."""
    assert WIDE_FPS == 60.0
    assert ULTRAWIDE_FPS == 10.0
