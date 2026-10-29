"""Rerun recordings of the 2D benchmark (specs/twod.md 4): one Figure 2 row per image and a summary.

- `runs/imagegs/<stem>/figure2.rrd`: Image-GS | Hexels | ground truth | Image-GS error | Hexels error,
  full image above, the 256 x 256 crop with the largest Image-GS-minus-Hexels error below. Error maps
  are the per-pixel RGB mean absolute error (f32, value = MAE), magma over 0-0.10, saturating.
- `runs/imagegs/summary.rrd` + `runs/imagegs/TABLE2D.json`: every method and condition per image and
  over all images, beside the HF catalog numbers of the same images and the paper's Table 1.

Torch-free (reads metrics.json and 16-bit PNG renders). Needs `rerun-sdk`: run it with an env that has
it (gsplat-rust-renderer-dev) and `PYTHONPATH=packages/hexels` until the hexels env carries rerun-sdk.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float32, UInt8, UInt16
from numpy import ndarray
from serde.json import to_json

from hexels_bench.layout import Roots
from hexels_bench.png16 import read_png16
from hexels_bench.results2d import (
    ERROR_RANGE,
    Condition,
    ImageRun,
    TableRow,
    image_category,
    load_image_run,
    mae_map,
    table_rows,
    worst_window,
)

CROP: int = 256
PAPER_NUMBERS: Path = Path.home() / "hexels-work" / "specs" / "paper-numbers.json"


@dataclass
class Config:
    """Write the Figure 2 recordings and the summary recording of the 2D benchmark."""

    stems: list[str] = field(default_factory=list)
    """Images to draw; empty means every image whose two columns are decoded."""
    imagegs_method: str = "imagegs-2d"
    """Run directory of the Image-GS column."""
    hexels_method: str = "hexels-2d-pub-wgsl"
    """Run directory of the Hexels column (the published rich fit decoded by WGSL)."""
    summary: bool = True
    """Also write summary.rrd and TABLE2D.json."""
    roots: Roots = field(default_factory=Roots)
    """Data and run roots."""


def encode_png(rgb: UInt8[ndarray, "h w 3"]) -> bytes:
    ok, data = cv2.imencode(".png", np.ascontiguousarray(rgb[:, :, ::-1]))
    if not ok:
        raise OSError("PNG encode failed")
    return data.tobytes()


def view_name(label: str, run: ImageRun) -> str:
    return f"{label} · {run.scores().psnr:.2f} dB · LPIPS {run.scores().lpips_alex:.3f}"


def figure_blueprint(names: tuple[str, str], crop_xy: tuple[int, int], size: tuple[int, int]) -> rrb.Blueprint:
    """Figure 2 row: five full-image views above the same five views of one 256 x 256 crop."""
    entities: tuple[str, ...] = ("/image/imagegs", "/image/hexels", "/image/gt", "/image/imagegs_error", "/image/hexels_error")
    full_names: tuple[str, ...] = (*names, "Ground truth", f"Image-GS error (MAE 0-{ERROR_RANGE:.2f}+)", f"Hexels error (MAE 0-{ERROR_RANGE:.2f}+)")
    crop_names: tuple[str, ...] = (
        "Image-GS crop",
        "Hexels crop",
        f"GT crop at {crop_xy[0]},{crop_xy[1]}",
        "Image-GS error crop",
        "Hexels error crop",
    )
    full = rrb.VisualBounds2D(x_range=[0.0, float(size[0])], y_range=[0.0, float(size[1])])
    crop = rrb.VisualBounds2D(
        x_range=[float(crop_xy[0]), float(crop_xy[0] + CROP)],
        y_range=[float(crop_xy[1]), float(crop_xy[1] + CROP)],
    )

    def row(bounds: rrb.VisualBounds2D, view_names: tuple[str, ...]) -> rrb.Horizontal:
        return rrb.Horizontal(
            *(
                rrb.Spatial2DView(name=name, origin="/image", contents=entity, visual_bounds=bounds)
                for name, entity in zip(view_names, entities, strict=True)
            )
        )

    return rrb.Blueprint(
        rrb.Vertical(row(full, full_names), row(crop, crop_names), row_shares=[3, 2]),
        rrb.TimePanel(state="collapsed"),
        collapse_panels=True,
    )


def write_figure(roots: Roots, stem: str, imagegs_method: str, hexels_method: str) -> Path:
    image_dir: Path = roots.runs / "imagegs" / stem
    gt_rgb: UInt16[ndarray, "h w 3"] = read_png16(image_dir / "gt" / "test" / f"{stem}.png")
    target: Float32[ndarray, "h w 3"] = gt_rgb.astype(np.float32) / np.float32(65535.0)
    runs: dict[str, ImageRun] = {method: load_image_run(image_dir / method / "metrics.json") for method in (imagegs_method, hexels_method)}
    errors: dict[str, Float32[ndarray, "h w"]] = {}
    out: Path = image_dir / "figure2.rrd"
    recording = rr.RecordingStream("hexels-2d-figure2", recording_id=stem)
    recording.save(out)
    recording.log(
        "/image/gt", rr.EncodedImage(contents=(roots.data / "imagegs" / "images" / f"{stem}.png").read_bytes(), media_type="image/png"), static=True
    )
    for key, method in (("imagegs", imagegs_method), ("hexels", hexels_method)):
        render: UInt16[ndarray, "h w 3"] = read_png16(image_dir / method / runs[method].scores().render)
        prediction: Float32[ndarray, "h w 3"] = render.astype(np.float32) / np.float32(65535.0)
        errors[key] = mae_map(prediction, target)
        rgb8: UInt8[ndarray, "h w 3"] = np.round(prediction * np.float32(255.0)).astype(np.uint8)
        recording.log(f"/image/{key}", rr.EncodedImage(contents=encode_png(rgb8), media_type="image/png"), static=True)
        recording.log(
            f"/image/{key}_error",
            rr.DepthImage(errors[key], meter=1.0, colormap="magma", depth_range=[0.0, ERROR_RANGE]),
            static=True,
        )
        scores = runs[method].scores()
        recording.log(
            f"/metrics/2d/{method}",
            rr.AnyValues(
                psnr=scores.psnr, ssim=scores.ssim, lpips_alex=scores.lpips_alex, fps=scores.fps or 0.0, bytes=runs[method].bytes_packed or 0
            ),
            static=True,
        )
    crop_xy: tuple[int, int] = worst_window(errors["imagegs"] - errors["hexels"], size=CROP, stride=16)
    names: tuple[str, str] = (view_name("Image-GS", runs[imagegs_method]), view_name("Hexels", runs[hexels_method]))
    recording.send_blueprint(figure_blueprint(names, crop_xy, (gt_rgb.shape[1], gt_rgb.shape[0])))
    recording.flush()
    return out


BAR_SERIES: tuple[tuple[str, Condition, tuple[int, int, int]], ...] = (
    ("imagegs-2d", "uint16", (230, 140, 50)),
    ("hexels-2d-flat-pub-wgsl", "decoded_hxm", (120, 170, 230)),
    ("hexels-2d-pub-wgsl", "decoded_hxm", (40, 90, 200)),
    ("hexels-2d", "decoded_hxm", (60, 190, 120)),
)
"""Bar chart series: (method, condition, colour), drawn as grouped bars per image."""


def write_summary(roots: Roots, stems: list[str]) -> Path:
    """TABLE2D.json and summary.rrd: one table row per method, condition and image (and their mean)."""
    paper: dict = json.loads(PAPER_NUMBERS.read_text())
    rows: list[TableRow] = table_rows(roots.runs / "imagegs", stems, paper)
    (roots.runs / "imagegs" / "TABLE2D.json").write_text(to_json(rows) + "\n")
    out: Path = roots.runs / "imagegs" / "summary.rrd"
    recording = rr.RecordingStream("hexels-2d-summary", recording_id="summary")
    recording.save(out)
    groups: list[str] = [*stems, "all"]
    for index, row in enumerate(rows):  # one table row per time step: the DataframeView lists them all
        recording.set_time("row", sequence=index)
        values = rr.AnyValues(  # unscored fields are None and dropped
            image=row.group, method=row.method, condition=row.condition, images=float(row.images),
            psnr=row.psnr, ssim=row.ssim, lpips_alex=row.lpips_alex, fps=row.fps,
            train_s=row.train_seconds, vram_gib=row.peak_vram_gib, train_gpu=row.train_gpu,
            published_psnr=row.published_psnr, published_ssim=row.published_ssim,
            published_lpips_alex=row.published_lpips_alex, published_fps_a5000=row.published_fps_a5000,
            table1_psnr=row.table1_psnr, table1_ssim=row.table1_ssim, table1_lpips_alex=row.table1_lpips_alex,
        )  # fmt: skip
        recording.log("/table", values)
    width: float = 0.8 / len(BAR_SERIES)
    for index, (method, condition, color) in enumerate(BAR_SERIES):
        psnr: list[float] = [next((r.psnr for r in rows if (r.group, r.method, r.condition) == (g, method, condition)), 0.0) for g in groups]
        positions = np.arange(len(groups), dtype=np.float64) + (index - (len(BAR_SERIES) - 1) / 2) * width
        chart = rr.BarChart(np.array(psnr, dtype=np.float64), abscissa=positions, widths=[width] * len(groups), color=color)
        recording.log(f"/psnr/{method}/{condition}", chart, static=True)
    image_names: str = ", ".join(f"{index} {group.removesuffix('_2k')}" for index, group in enumerate(groups))
    recording.send_blueprint(
        rrb.Blueprint(
            rrb.Vertical(
                rrb.DataframeView(
                    origin="/table",
                    name="2D benchmark: every method and condition per image and over all images, ours beside published",
                    query=rrb.archetypes.DataframeQuery(timeline="row"),
                ),
                rrb.BarChartView(origin="/psnr", name=f"PSNR dB per image: {image_names}"),
                row_shares=[3, 2],
            ),
            rrb.TimePanel(timeline="row", state="hidden"),
            collapse_panels=True,
        )
    )
    recording.flush()
    return out


def main(config: Config) -> None:
    image_root: Path = config.roots.runs / "imagegs"
    stems: list[str] = config.stems or sorted(
        path.name
        for path in image_root.iterdir()
        if (path / config.imagegs_method / "metrics.json").is_file() and (path / config.hexels_method / "metrics.json").is_file()
    )
    for stem in stems:
        image_category(stem)
        print(f"wrote {write_figure(config.roots, stem, config.imagegs_method, config.hexels_method)}", flush=True)
    if config.summary:
        print(f"wrote {write_summary(config.roots, stems)}", flush=True)
