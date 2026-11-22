"""Re-freeze the `/fused_ssim/` records of `fixtures/hexels_references.rrd` from the CUDA fused-ssim package.

Reference: rahul-goel/fused-ssim at a7c48d6 (its `fused_ssim_cuda` extension, built with its own setup.py, which compiles
with --use_fast_math), the upstream revision hexels_mojo.fused_ssim follows. The inputs of each record (`img1`, `img2`,
`dl_dmap`) are kept; its outputs (`ssim_map`, the three partials, `grad` = backward with those partials) are recomputed.
Every other record is copied over unchanged, in file order. The reference build
(made once, outside the workspace's environments; this file's references were made with it): the rattler-build recipe
`recipes/fused-ssim-src` of commit 91b0c3f59 (branch open4dgs/speed-fused-ssim; sm_120 + sm_121, torch 2.12 cu130),
installed by pixi into the open4dgs env of that branch, whose site-packages provide `fused_ssim_cuda`; then, from
packages/hexels-mojo, with that extension on the path:

    PYTHONPATH=<dir holding fused_ssim_cuda*.so> python tests/freeze_fused_ssim_references.py
"""

import argparse
from pathlib import Path

import rerun as rr
import torch  # noqa: F401  # loads libtorch, which the fused_ssim_cuda extension links against
from fused_ssim_cuda import fusedssim, fusedssim_backward  # pyrefly: ignore[missing-import]
from hexels_references import REFERENCE_FILE, encode, load  # pyrefly: ignore[missing-import]

C1: float = 0.01**2
C2: float = 0.03**2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REFERENCE_FILE, help="output file (default: the committed fixture, in place)")
    args = parser.parse_args()
    existing = load()
    rec = rr.RecordingStream("hexels_mojo_references")
    rec.save(str(args.out))
    refrozen = 0
    for path, record in existing.items():
        values: dict = record.tensors | record.texts
        if path.startswith("/fused_ssim/"):
            img1, img2, dl_dmap = (record.tensor(n).cuda() for n in ("img1", "img2", "dl_dmap"))
            ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = fusedssim(C1, C2, img1, img2, True)
            grad = fusedssim_backward(C1, C2, img1, img2, dl_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
            new = {"ssim_map": ssim_map, "dm_dmu1": dm_dmu1, "dm_dsigma1_sq": dm_dsigma1_sq, "dm_dsigma12": dm_dsigma12, "grad": grad}
            assert set(new) <= set(values), (path, sorted(values))
            values = {k: (new[k].cpu() if k in new else v) for k, v in values.items()}
            refrozen += 1
        rec.log(path, rr.AnyValues(**encode(values)), static=True)
    rec.flush()
    print(f"re-froze {refrozen} /fused_ssim records; copied {len(existing) - refrozen} others to {args.out}")


if __name__ == "__main__":
    main()
