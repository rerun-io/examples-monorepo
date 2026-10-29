"""Build `runs/TABLE.json` and `runs/table.rrd` from the benchmark run directories."""

from dataclasses import dataclass
from pathlib import Path

from hexels_bench.bench_table import BenchTable, MeanRow, build_table, write_table_json, write_table_rrd
from hexels_bench.layout import read_json
from hexels_bench.paper import PAPER_NUMBERS, PaperNumbers


@dataclass(frozen=True, slots=True)
class Config:
    """Turn runs/ into the paper-style table (JSON) and a Rerun metrics recording."""

    runs: Path = Path.home() / "hexels-work" / "runs"
    """Runs root (<dataset>/<scene>/<method>/)."""


def fmt(value: float | None, digits: int) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def main(config: Config) -> None:
    table: BenchTable = build_table(config.runs, read_json(PaperNumbers, PAPER_NUMBERS))
    write_table_json(table, config.runs / "TABLE.json")
    write_table_rrd(table, config.runs / "table.rrd")
    print(
        f"{'dataset':<11} {'method':<24} {'n':>5} {'train min':>9} {'GiB':>5} {'FPS':>7} {'PSNR':>6} {'paper':>6} {'SSIM':>6} {'LPIPS':>6} {'MB':>7} {'K':>6}"
    )
    for mean in table.means:
        row: MeanRow = mean
        paper_psnr: str = fmt(None if row.paper is None else row.paper.psnr, 2)
        print(
            f"{row.dataset:<11} {row.method:<24} {row.scenes:>2}/{row.total:<2} {fmt(row.train_min, 2):>9} {fmt(row.vram_gib, 2):>5} {fmt(row.fps, 1):>7} "
            f"{fmt(row.psnr, 2):>6} {paper_psnr:>6} {fmt(row.ssim, 3):>6} {fmt(row.lpips_vgg, 3):>6} {fmt(row.deploy_mb, 2):>7} {fmt(row.primitives_k, 1):>6}"
        )
    for flag in table.flags:
        print(f"FLAG {flag}")
    print(f"wrote {config.runs / 'TABLE.json'} and {config.runs / 'table.rrd'}")
