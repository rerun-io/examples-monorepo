"""View titles of the side-by-side recording carry the model, its size and the viewer FPS measured for it."""

from gsplat_rust_renderer.apis.hexels_compare import display_name, hardware_note, view_name
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


def test_the_hardware_note_claims_only_what_the_records_name() -> None:
    brush: MetricRow = MetricRow("Brush", "ours", fps=500.0, fps_gpu="GPU A", train_min=1.0, train_gpu="GPU A", viewer_ms_median=3.0, viewer_gpu="GPU B")
    hexels: MetricRow = MetricRow("Hexels", "ours", fps=900.0, viewer_ms_median=2.0, viewer_gpu="GPU B")
    assert hardware_note([brush, hexels], "RTX A5000") == "viewer timing on GPU B; training on GPU A; published on RTX A5000", (
        "the Hexels FPS has no recorded GPU, so the note makes no FPS claim"
    )
    assert hardware_note([brush], None) == "viewer timing on GPU B; test-view FPS and training on GPU A"
    everything: MetricRow = MetricRow("Brush", "ours", fps=1.0, fps_gpu="G", train_min=1.0, train_gpu="G", viewer_ms_median=1.0, viewer_gpu="G")
    assert hardware_note([everything], None) == "viewer timing, test-view FPS and training on G"
    assert hardware_note([MetricRow("Hexels", "ours", psnr=24.0)], None) == ""
