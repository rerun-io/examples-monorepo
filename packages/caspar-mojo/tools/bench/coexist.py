"""CASPAR (pycolmap) and a Mojo GPU kernel in one process, in both orders: they must share the CUDA device.

Usage: python coexist.py <dir holding mojo_smoke_ext.so> {caspar-first,mojo-first}
"""

import sys


def caspar() -> None:
    from colsfm.ba_capability import _caspar_probe_solve

    summary = _caspar_probe_solve("PINHOLE")
    assert summary is not None and summary.is_solution_usable(), summary
    print(f"caspar: residuals={summary.num_residuals} usable={summary.is_solution_usable()}")


def mojo() -> None:
    import mojo_smoke_ext

    name, err = mojo_smoke_ext.axpy_max_error(1_000_003)
    assert err == 0.0, err
    print(f"mojo: {name} axpy max error {err}")


if __name__ == "__main__":
    import pycolmap

    sys.path.insert(0, sys.argv[1])
    print(f"pycolmap {pycolmap.__version__}, CASPAR backend enum: {pycolmap.BundleAdjustmentBackend.CASPAR!r}")
    steps = [caspar, mojo, caspar] if sys.argv[2] == "caspar-first" else [mojo, caspar, mojo]
    for step in steps:
        step()
    print("coexist: OK")
