"""View titles of the side-by-side recording carry the model, its size and the viewer FPS measured for it."""

from gsplat_rust_renderer.apis.hexels_compare import display_name, view_name
from gsplat_rust_renderer.hexels_metrics import MetricRow


def test_display_names_follow_the_run_layout_methods() -> None:
    assert display_name("brush-7k") == "Brush 3DGS 7k"
    assert display_name("hexels-v1-7k") == "Hexels V1 7k"
    assert display_name("hexels-v3-30k-pub-wgsl") == "Hexels V3 30k published"


def test_view_name_shows_the_probe_fps_only_when_measured() -> None:
    measured: MetricRow = MetricRow("Brush 3DGS 7k", "ours", primitives=138_766, viewer_ms_median=3.587)
    assert view_name(measured, "splats") == "Brush 3DGS 7k · 138,766 splats · 279 FPS viewer"
    unmeasured: MetricRow = MetricRow("Hexels V1 7k", "ours", primitives=31_205)
    assert view_name(unmeasured, "hexels") == "Hexels V1 7k · 31,205 hexels"
