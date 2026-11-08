"""Freeze hexel-kernel's 3D hexel rasterizer forward (`gsplat.cuda_hexels3d._C.rasterize_hexels3d_forward_cached`).

Needs hexel-kernel (still built by recipes/hexels-src). Writes `fixtures/raster3d_references.rrd` (same record layout
as `hexels_references.py`), one `/raster3d/<case>` record per case: inputs, the image and radii, and
the saved intermediates (transforms, tile ranges, sorted ids, final transmittance, last index) so the port can be
checked stage by stage.

    python tests/freeze_raster3d_references.py
"""

import math
from pathlib import Path

import rerun as rr
import torch
from hexels_references import encode  # pyrefly: ignore[missing-import]
from torch import Tensor

RASTER3D_FILE: Path = Path(__file__).parent / "fixtures" / "raster3d_references.rrd"
# (name, P, H, W, color_model, view_model, use_view_grads, softness kind, conservative_bounds, seed)
CASES = (
    ("linear_view_grads", 3000, 96, 128, 1, 1, True, "per_hexel", False, 30),
    ("flat_scalar_conservative", 1500, 80, 96, 0, 0, False, "scalar", True, 31),
    ("quadratic_bounded_lobe_triplet", 1500, 80, 96, 3, 2, False, "triplet", False, 32),
)
SAVED = ("radii_saved", "transmats", "opacities_out", "softnesses_out", "view_dirs", "normals", "points_xy_center", "ranges",
         "values_sorted", "final_transmittance", "last_index")  # fmt: skip


def _camera(width: int, height: int, fov_x: float) -> tuple[Tensor, Tensor, Tensor, float, float]:
    """3DGS conventions: column-major world-to-view (identity at the origin, looking down +z) and full projection."""
    fov_y = 2.0 * math.atan(math.tan(fov_x / 2.0) * height / width)
    znear, zfar = 0.01, 100.0
    tan_x, tan_y = math.tan(fov_x / 2.0), math.tan(fov_y / 2.0)
    proj = torch.zeros(4, 4)
    proj[0, 0], proj[1, 1] = 1.0 / tan_x, 1.0 / tan_y
    proj[3, 2], proj[2, 2], proj[2, 3] = 1.0, zfar / (zfar - znear), -(zfar * znear) / (zfar - znear)
    view = torch.eye(4)
    view[:3, 3] = torch.tensor([0.1, -0.05, 0.3])
    world_view = view.transpose(0, 1)
    full = world_view @ proj.transpose(0, 1)
    campos = torch.linalg.inv(world_view)[3, :3]
    return world_view.contiguous(), full.contiguous(), campos.contiguous(), tan_x, tan_y


def _case(cuda, name, n, h, w, color_model, view_model, use_view_grads, softness_kind, conservative, seed) -> dict[str, Tensor]:
    g = torch.Generator().manual_seed(seed)
    u = lambda *s: torch.rand(*s, generator=g)  # noqa: E731
    z = 2.0 + 4.0 * u(n, 1)
    xy = (u(n, 2) - 0.5) * 2.0 * z * torch.tensor([0.6, 0.45])
    means = torch.cat((xy, z), dim=1)
    scales = 0.02 + 0.12 * u(n, 2)
    rotations = torch.randn(n, 4, generator=g)
    colors = u(n, 3)
    color_grads = (u(n, 2, 3) - 0.5) * 0.4
    color_quads = (u(n, 3, 3) - 0.5) * 0.2 if color_model >= 2 else torch.empty(0)
    view_grads = (u(n, 3, 3) - 0.5) * 0.2 if use_view_grads else torch.empty(0)
    specular_ks = 0.3 * u(n, 3) if view_model == 2 else torch.empty(0)
    specular_log_s = 4.0 * u(n) if view_model == 2 else torch.empty(0)
    opacities = 0.05 + 0.95 * u(n)
    softnesses = {"per_hexel": 2.0 + 10.0 * u(n), "triplet": 2.0 + 10.0 * u(n, 3), "scalar": torch.empty(0)}[softness_kind]
    view, proj, campos, tan_x, tan_y = _camera(w, h, math.radians(60.0))
    background = torch.tensor([0.1, 0.2, 0.3])
    inputs = {
        "means3d": means, "scales": scales, "rotations": rotations, "colors": colors, "color_grads": color_grads,
        "color_quads": color_quads, "view_grads": view_grads, "specular_ks": specular_ks, "specular_log_s": specular_log_s,
        "opacities": opacities, "softnesses": softnesses, "viewmatrix": view, "projmatrix": proj, "campos": campos,
        "background": background,
    }  # fmt: skip
    cuda_inputs = [t.float().contiguous().cuda() for t in inputs.values()]
    out_color, radii, saved = cuda.rasterize_hexels3d_forward_cached(
        *cuda_inputs, w, h, tan_x, tan_y, 6.0, 1.0, color_model, view_model, use_view_grads, False, conservative, False
    )
    record = {k: v.float() for k, v in inputs.items()}
    record["config"] = torch.tensor([h, w, color_model, view_model, int(use_view_grads), int(conservative)], dtype=torch.int32)
    record["camera"] = torch.tensor([tan_x, tan_y, 6.0, 1.0])
    record["out_color"], record["radii"] = out_color, radii.int()
    for key, value in zip(SAVED, saved, strict=True):
        if value.numel():
            record[key] = value.int() if value.dtype in (torch.int32, torch.int64) else value
    return record


def main() -> None:
    import gsplat.cuda_hexels3d._C as cuda  # pyrefly: ignore[missing-import]

    records = {f"/raster3d/{case[0]}": _case(cuda, *case) for case in CASES}
    rec = rr.RecordingStream("hexels_mojo_raster3d_references")
    rec.save(str(RASTER3D_FILE))
    for path, values in records.items():
        rec.log(path, rr.AnyValues(**encode({k: v.cpu() for k, v in values.items()})), static=True)
    rec.flush()
    print(f"wrote {len(records)} records to {RASTER3D_FILE}")


if __name__ == "__main__":
    main()
