"""Self-contained HTML report for the exo-calib evaluation.

Reads ``eval.json`` (Stage E) and renders the init-vs-refined comparison as a
single HTML file with inline SVG charts, hover tooltips, table views, and
embedded evidence screenshots. Regenerate after every evaluation run.
"""

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path

from exo_calib.catalog_io import DEFAULT_CATALOG_URL, DEFAULT_DATASET_NAME, connect_dataset, only_segment_id

SERIES_LIGHT: tuple[str, str] = ("#2a78d6", "#eb6834")
SERIES_DARK: tuple[str, str] = ("#3987e5", "#d95926")


@dataclass
class HtmlReportConfig:
    """Config for the HTML report generator."""

    catalog_url: str = DEFAULT_CATALOG_URL
    """Rerun catalog server URL (used only to resolve the segment id)."""
    dataset_name: str = DEFAULT_DATASET_NAME
    """Catalog dataset holding the registered segment."""
    segment_id: str | None = None
    """Segment scored in ``eval.json``; ``None`` uses the dataset's single segment."""
    output_dir: Path = Path("data/outputs")
    """Directory holding ``<segment>/eval.json``."""
    html_path: Path = Path("/tmp/fleet-artifacts/exo-calib/assembly101-exo-calib-report.html")
    """Destination of the generated single-file report."""
    screenshots: tuple[Path, ...] = field(default_factory=tuple)
    """Evidence PNGs embedded (base64) at the bottom of the report."""
    window_s: float = 5.0
    """Processed window length noted in the header."""


def _grouped_bar_svg(labels: list[str], init_v: list[float], refined_v: list[float], unit: str) -> str:
    """Render a two-series grouped bar chart as inline SVG with hover targets."""
    n: int = len(labels)
    width, height, margin_left, margin_bottom, margin_top = 760, 300, 46, 40, 14
    plot_w, plot_h = width - margin_left - 12, height - margin_bottom - margin_top
    peak: float = max(max(init_v), max(refined_v)) * 1.15 or 1.0
    group_w: float = plot_w / n
    bar_w: float = min(26.0, (group_w - 14.0) / 2.0)
    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" role="img" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto">'
    ]
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y: float = margin_top + plot_h * (1 - frac)
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - 12}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{margin_left - 6}" y="{y + 4:.1f}" text-anchor="end" class="tick">{peak * frac:.1f}</text>')
    for i, label in enumerate(labels):
        cx: float = margin_left + group_w * (i + 0.5)
        for k, (value, cls) in enumerate(((init_v[i], "s1"), (refined_v[i], "s2"))):
            bar_h: float = plot_h * value / peak
            x: float = cx - bar_w - 1.0 + k * (bar_w + 2.0)
            y = margin_top + plot_h - bar_h
            parts.append(
                f'<rect class="bar {cls}" x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{max(bar_h, 1.0):.1f}" rx="4"'
                f' data-tip="{label} · {"init" if k == 0 else "refined"}: {value:.2f} {unit}"/>'
            )
        parts.append(f'<text x="{cx:.1f}" y="{height - 22}" text-anchor="middle" class="tick">{label}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _metric_block(title: str, unit: str, labels: list[str], init_v: list[float], refined_v: list[float]) -> str:
    """One chart + its table view."""
    rows: str = "".join(
        f"<tr><td>{label}</td><td>{i:.2f}</td><td>{r:.2f}</td></tr>" for label, i, r in zip(labels, init_v, refined_v, strict=True)
    )
    return f"""
<section>
  <h2>{title}</h2>
  <div class="legend"><span class="chip s1"></span>init (G3T + MoGe-2)<span class="chip s2"></span>refined (+ Kineo BA)</div>
  {_grouped_bar_svg(labels, init_v, refined_v, unit)}
  <details><summary>table view</summary>
    <table><thead><tr><th>camera</th><th>init ({unit})</th><th>refined ({unit})</th></tr></thead><tbody>{rows}</tbody></table>
  </details>
</section>"""


def render_report(report: dict, config: HtmlReportConfig, segment_id: str) -> str:
    """Render the full HTML document from the Stage E report dict."""
    cameras: list[str] = [f"rig_{i:02d}" for i in range(8)]
    blocks: list[str] = []
    tiles: list[str] = []
    if "init" in report and "refined" in report:
        for metric, unit, key in (("translation", "cm", "translation_cm"), ("rotation", "deg", "rotation_deg")):
            init_m: dict = report["init"]["se3"][key]
            refined_m: dict = report["refined"]["se3"][key]
            blocks.append(_metric_block(f"Per-camera {metric} error — SE(3) aligned", unit, cameras, init_m["per_camera"], refined_m["per_camera"]))
            tiles.append(
                f'<div class="tile"><div class="tile-label">{metric} (SE3 mean)</div>'
                f'<div class="tile-value">{init_m["mean"]:.2f} <span class="tile-unit">{unit}</span>'
                f' → {refined_m["mean"]:.2f} <span class="tile-unit">{unit}</span></div></div>'
            )
    summary_rows: list[str] = []
    for variant in ("init", "refined"):
        if variant not in report:
            continue
        for mode in ("se3", "sim3"):
            m: dict = report[variant][mode]
            summary_rows.append(
                f"<tr><td>{variant}</td><td>{mode.upper()}</td>"
                f"<td>{m['translation_cm']['mean']:.2f} / {m['translation_cm']['median']:.2f} / {m['translation_cm']['max']:.2f}</td>"
                f"<td>{m['rotation_deg']['mean']:.2f} / {m['rotation_deg']['median']:.2f} / {m['rotation_deg']['max']:.2f}</td>"
                f"<td>{m['scale']:.4f}</td></tr>"
            )
    focal: dict | None = report.get("init", {}).get("focal_error_pct")
    focal_html: str = ""
    if focal is not None:
        cells: str = "".join(f"<td>{v:+.1f}</td>" for v in focal["per_camera"])
        focal_html = f"""
<section><h2>Estimated focal length error vs GT (%)</h2>
<table><thead><tr>{"".join(f"<th>{c}</th>" for c in cameras)}</tr></thead><tbody><tr>{cells}</tr></tbody></table>
<p class="note">mean |err| {focal['mean_abs']:.2f}% · max |err| {focal['max_abs']:.2f}% — intrinsics are estimated (never read from GT); the pipeline is fully self-contained.</p></section>"""

    shots: list[str] = []
    for png in config.screenshots:
        if Path(png).exists():
            encoded: str = base64.b64encode(Path(png).read_bytes()).decode()
            shots.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="{Path(png).stem}"/><figcaption>{Path(png).stem}</figcaption></figure>')
    shots_html: str = f"<section><h2>Viewer evidence</h2>{''.join(shots)}</section>" if shots else ""

    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Assembly101 exo-camera calibration — init vs refined</title>
<style>
:root {{ color-scheme: light dark;
  --surface:#fcfcfb; --ink:#1a1a19; --ink-2:#585856; --grid:#e5e4df;
  --s1:{SERIES_LIGHT[0]}; --s2:{SERIES_LIGHT[1]}; }}
@media (prefers-color-scheme: dark) {{ :root {{
  --surface:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7; --grid:#33332f;
  --s1:{SERIES_DARK[0]}; --s2:{SERIES_DARK[1]}; }} }}
body {{ margin:0 auto; max-width:860px; padding:24px 20px 60px; background:var(--surface); color:var(--ink);
  font:15px/1.5 system-ui,-apple-system,sans-serif; }}
h1 {{ font-size:1.4rem; margin:0 0 4px; }} h2 {{ font-size:1.05rem; margin:28px 0 8px; }}
.sub {{ color:var(--ink-2); font-size:.85rem; }}
.tiles {{ display:flex; gap:12px; flex-wrap:wrap; margin:18px 0; }}
.tile {{ border:1px solid var(--grid); border-radius:10px; padding:10px 14px; }}
.tile-label {{ font-size:.75rem; color:var(--ink-2); text-transform:uppercase; letter-spacing:.04em; }}
.tile-value {{ font-size:1.25rem; font-variant-numeric:tabular-nums; }} .tile-unit {{ font-size:.8rem; color:var(--ink-2); }}
.grid {{ stroke:var(--grid); stroke-width:1; }} .tick {{ fill:var(--ink-2); font-size:11px; }}
.bar.s1 {{ fill:var(--s1); }} .bar.s2 {{ fill:var(--s2); }} .bar:hover {{ opacity:.85; }}
.legend {{ display:flex; gap:8px; align-items:center; font-size:.85rem; color:var(--ink-2); margin-bottom:4px; }}
.chip {{ width:12px; height:12px; border-radius:3px; display:inline-block; margin-left:12px; }}
.chip.s1 {{ background:var(--s1); margin-left:0; }} .chip.s2 {{ background:var(--s2); }}
table {{ border-collapse:collapse; font-variant-numeric:tabular-nums; font-size:.85rem; margin-top:6px; }}
th,td {{ border:1px solid var(--grid); padding:4px 10px; text-align:right; }} th:first-child,td:first-child {{ text-align:left; }}
details summary {{ cursor:pointer; color:var(--ink-2); font-size:.85rem; margin-top:4px; }}
figure {{ margin:12px 0; }} img {{ max-width:100%; border:1px solid var(--grid); border-radius:8px; }}
figcaption {{ font-size:.8rem; color:var(--ink-2); }}
.note {{ color:var(--ink-2); font-size:.85rem; }}
#tip {{ position:fixed; pointer-events:none; background:var(--ink); color:var(--surface); padding:4px 8px;
  border-radius:6px; font-size:.8rem; opacity:0; transition:opacity .1s; z-index:9; }}
</style>
<body>
<h1>Assembly101 exo-camera calibration — init vs refined</h1>
<p class="sub">segment <code>{segment_id}</code> · first {config.window_s:.0f} s · 8 exo cameras · pipeline: G3T + MoGe-2 metric init →
YOLOX + Sapiens2-1B (TensorRT) COCO-133 keypoints → AssemblyHands-X confidence post-processing → Kineo spatio-temporal
correspondences → confidence-weighted DLT → kornia-rs Schur bundle adjustment with soft center priors.
Ground truth touches evaluation only.</p>
<div class="tiles">{"".join(tiles)}</div>
<section><h2>Summary — errors vs ground truth (mean / median / max)</h2>
<table><thead><tr><th>variant</th><th>align</th><th>translation cm</th><th>rotation deg</th><th>scale</th></tr></thead>
<tbody>{"".join(summary_rows)}</tbody></table>
<p class="note">SE(3) alignment is the primary metric (tests the metric-scale claim); Sim(3) isolates residual scale error.</p></section>
{"".join(blocks)}
{focal_html}
{shots_html}
<div id="tip"></div>
<script>
const tip = document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el => {{
  el.addEventListener('mousemove', e => {{ tip.textContent = el.dataset.tip; tip.style.opacity = 1;
    tip.style.left = (e.clientX + 12) + 'px'; tip.style.top = (e.clientY - 30) + 'px'; }});
  el.addEventListener('mouseleave', () => tip.style.opacity = 0);
}});
</script>
</body></html>"""


def main(config: HtmlReportConfig) -> None:
    """Generate the report HTML from ``eval.json``."""
    dataset = connect_dataset(config.catalog_url, config.dataset_name)
    segment_id: str = config.segment_id or only_segment_id(dataset)
    report: dict = json.loads((config.output_dir / segment_id / "eval.json").read_text())
    html: str = render_report(report, config, segment_id)
    config.html_path.parent.mkdir(parents=True, exist_ok=True)
    config.html_path.write_text(html)
    print(f"wrote {config.html_path}")
