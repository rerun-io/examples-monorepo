"""Freeze the upstream CUDA outputs that hexels_mojo is tested against (run once, in an env that still has them).

Needs the hexels-src CUDA extensions (`gsplat` from hexel-kernel, `fused_ssim_cuda`, `simple_knn._C`) and nvcc
for upstream's JIT prune scorer: the hexels env before hexels_mojo replaced them. Writes
`fixtures/hexels_references.rrd` in the layout of `hexels_references.py`.

    python tests/freeze_hexels_references.py --root <hexels-src checkout>
"""

import argparse
import sys
from pathlib import Path

import rerun as rr
import torch
from hexels_references import REFERENCE_FILE, encode  # pyrefly: ignore[missing-import]
from torch import Tensor


def _hexels(n: int, height: int, width: int, generator: torch.Generator) -> dict[str, Tensor]:
    """Random hexels over a (height, width) image, in the parameter ranges of a 2D fit."""
    u = lambda *shape: torch.rand(*shape, generator=generator)  # noqa: E731
    return {
        "xy": (u(n, 2) * torch.tensor([width, height])).float(),
        "scales": (1.0 + 4.0 * u(n, 2)).float(),
        "rotations": ((u(n, 1) - 0.5) * 6.283).float(),
        "colors": u(n, 3).float(),
        "gradients": ((u(n, 2, 3) - 0.5) * 0.2).float(),
        "softness": (4.0 + 12.0 * u(n)).float(),
    }


def _prune_records(root: Path) -> dict[str, dict[str, Tensor]]:
    sys.path.insert(0, str(root))
    from gsplat_hexels import rasterize_hexels_sum  # pyrefly: ignore[missing-import]
    from scripts.image_compression_allocation import load_prune_scores  # pyrefly: ignore[missing-import]

    scorer = load_prune_scores()
    records: dict[str, dict[str, Tensor]] = {}
    for name, n, height, width, rich, seed in (("rich", 600, 72, 96, True, 0), ("plain", 400, 64, 80, False, 1)):
        generator = torch.Generator().manual_seed(seed)
        h = {k: v.cuda() for k, v in _hexels(n, height, width, generator).items()}
        target = torch.rand(height, width, 3, generator=generator).cuda()
        image, denominator, _, ids = rasterize_hexels_sum(
            h["xy"], h["scales"], h["rotations"], h["colors"], height, width,
            color_grads=h["gradients"] if rich else None, softness=h["softness"] if rich else 10.0,
            return_aux=True, return_topk=True,
        )  # fmt: skip
        image_packed = torch.cat((image, denominator[..., None]), dim=-1).contiguous()
        scores, max_error = scorer.removal_scores_cuda(
            h["xy"], h["scales"], h["rotations"], h["colors"], h["gradients"], h["softness"], image_packed, target, ids.int().contiguous(), rich
        )
        records[f"/prune/{name}"] = h | {"image": image_packed, "target": target, "selected": ids.int().contiguous(), "rich": torch.tensor([int(rich)]),
                                         "scores": scores, "max_error": max_error.reshape(1)}  # fmt: skip
    return records


def _ssim_records() -> dict[str, dict[str, Tensor]]:
    from fused_ssim_cuda import fusedssim, fusedssim_backward  # pyrefly: ignore[missing-import]

    records: dict[str, dict[str, Tensor]] = {}
    for name, shape, seed in (("noise", (2, 3, 67, 93), 2), ("smooth", (1, 3, 96, 128), 3)):
        generator = torch.Generator().manual_seed(seed)
        img2 = torch.rand(*shape, generator=generator)
        if name == "smooth":
            img2 = torch.nn.functional.avg_pool2d(img2, 7, stride=1, padding=3)
        img1 = (img2 + 0.1 * torch.randn(*shape, generator=generator)).clamp(0, 1)
        dl_dmap = torch.randn(*shape, generator=generator)
        img1, img2, dl_dmap = img1.float().cuda(), img2.float().cuda(), dl_dmap.float().cuda()
        ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = fusedssim(0.01**2, 0.03**2, img1, img2, True)
        grad = fusedssim_backward(0.01**2, 0.03**2, img1, img2, dl_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        records[f"/fused_ssim/{name}"] = {"img1": img1, "img2": img2, "dl_dmap": dl_dmap, "ssim_map": ssim_map, "dm_dmu1": dm_dmu1,
                                          "dm_dsigma1_sq": dm_dsigma1_sq, "dm_dsigma12": dm_dsigma12, "grad": grad}  # fmt: skip
    return records


def _knn_records() -> dict[str, dict[str, Tensor]]:
    from simple_knn._C import distCUDA2  # pyrefly: ignore[missing-import]

    generator = torch.Generator().manual_seed(4)
    uniform = torch.rand(5000, 3, generator=generator) * 4.0 - 1.0
    centers = torch.randn(40, 3, generator=generator) * 5.0
    clustered = centers[torch.randint(0, 40, (30000,), generator=generator)] + 0.05 * torch.randn(30000, 3, generator=generator)
    duplicates = torch.cat((uniform[:1500], uniform[:500]))
    records: dict[str, dict[str, Tensor]] = {}
    for name, points in (("uniform", uniform), ("clustered", clustered), ("duplicates", duplicates)):
        points = points.float().cuda().contiguous()
        records[f"/distcuda2/{name}"] = {"points": points, "dists": distCUDA2(points)}
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="hexels-src checkout ($PREFIX/share/hexels-src)")
    args = parser.parse_args()
    records = _prune_records(args.root) | _ssim_records() | _knn_records()
    rec = rr.RecordingStream("hexels_mojo_references")
    rec.save(str(REFERENCE_FILE))
    for path, values in records.items():
        rec.log(path, rr.AnyValues(**encode({k: v.cpu() for k, v in values.items()})), static=True)
    rec.flush()
    print(f"wrote {len(records)} records to {REFERENCE_FILE}")


if __name__ == "__main__":
    main()
