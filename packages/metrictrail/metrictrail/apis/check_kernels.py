"""Run the vipe-src kernel smoke checks and print one line per check."""

from dataclasses import dataclass

import torch

from metrictrail.kernels import KernelCheck, missing_vipe_ext_submodules, run_all


@dataclass(frozen=True, slots=True)
class Config:
    """Kernel smoke check."""

    tolerance: float = 1e-4
    """Largest allowed absolute error against the torch reference."""


def main(config: Config) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device: the vipe-src kernels need a GPU")
    missing: list[str] = missing_vipe_ext_submodules()
    if missing:
        raise SystemExit(f"vipe_ext lacks submodules {missing}")
    results: list[KernelCheck] = run_all(torch.device("cuda"))
    for result in results:
        print(f"{result.module}.{result.op}: max abs error {result.max_abs_error:.3g}")
    failed: list[KernelCheck] = [result for result in results if not result.max_abs_error < config.tolerance]
    if failed:
        raise SystemExit(f"{len(failed)} kernel check(s) above {config.tolerance}: {failed}")
    print(f"all {len(results)} kernel checks pass ({torch.cuda.get_device_name()})")
