"""Quality/Fast preset artifact gate: grade a dumped streaming run against the
original DAG's golden artifacts across all four artifact classes.

Runs in two parts so GPU time is paid once:
  1. ``tools/dump_artifacts.py`` runs the pipeline and writes per-tick NPZs.
  2. this tool (CPU-only) reads that dump + the golden artifacts and prints a
     PASS/FAIL table for the chosen preset.

Gate encodings were grounded by the 2026-06-10 four-agent dig over the real
running_jumping data and confirmed by Pablo:

  * 2D keypoints (quality only): per-(frame,cam) mean reprojection error over
    jointly-visible markers, normalized by the golden person-bbox diagonal,
    p95 over frames <= 2%.
  * triangulated 3D (quality only): per-frame mean error over markers valid in
    BOTH, p95 over frames <= 2% of the skeleton scale H.
  * SMPL-X parameters gate the IDENTIFIABLE surface realizations (raw per-joint
    pose geodesic is non-identifiable — twist is unconstrained by reprojection
    and the original LBFGS lands in a different valid basin): pose -> PVE,
    betas -> shape-space vertex displacement, trans -> millimeters. Raw
    geodesics are reported as diagnostics only.
  * masks: causal 720p masks cannot match the bidirectional 4K golden at IoU
    0.98 (the original's own re-run floor was 0.95), so the gate is the
    causal-achievable mean/p5/min IoU band.

Exit code 0 = PASS, 1 = FAIL.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tyro
from jaxtyping import Bool, Float64
from numpy import ndarray

Preset = Literal["quality", "fast"]


@dataclass(frozen=True, slots=True)
class GateResult:
    """One artifact gate's measured value vs its threshold."""

    name: str
    """Human-readable gate name."""
    value: float
    """Measured statistic."""
    threshold: float
    """PASS bound."""
    unit: str
    """Unit string for display (mm, %, IoU)."""
    higher_is_better: bool
    """True for IoU-style gates (value >= threshold passes)."""
    applies: bool
    """Whether this gate counts toward the preset verdict (kpt2d/tri3d are
    quality-only)."""

    @property
    def passed(self) -> bool:
        return self.value >= self.threshold if self.higher_is_better else self.value <= self.threshold


@dataclass
class ValidateArtifactsConfig:
    run_dir: Path = Path("/tmp/qfdig/current_run")
    """Dumped streaming run (tools/dump_artifacts.py output)."""
    preset: Preset = "quality"
    """Which gate set to apply."""
    golden_ma2d_dir: Path = Path(
        "/home/pablo/0Dev/repos/mamma/.claude/worktrees/baseline-3a4bc75/output/ma_2d/baseline-rj2/outdoors/running_jumping"
    )
    """Original per-camera 2D landmark NPZs (4K px)."""
    golden_ma3d_dir: Path = Path(
        "/home/pablo/0Dev/repos/mamma/.claude/worktrees/baseline-3a4bc75/output/ma_3d/baseline-rj2/outdoors/running_jumping"
    )
    """Original ma_3d NPZs (verts_joints + smplx_params)."""
    golden_masks_npz: Path = Path("/tmp/qfdig/golden_masks_720p.npz")
    """Pre-staged golden masks downscaled to engine resolution (720p packed bool)."""
    smplx_model_npz: Path = Path("data/body_models/smplx/SMPLX_NEUTRAL.npz")
    """For shapedirs (betas -> shape-space displacement)."""
    body_id: int = 0
    """Person id (golden artifacts are body_id 00)."""
    n_betas: int = 16
    """Number of shape coefficients the fitter uses (golden is (1,16))."""
    skip_first_frames: int = 8
    """Warmup frames excluded from every gate (parity with validate_dynamic)."""
    src_to_engine_scale: float = 1.0 / 3.0
    """Golden 4K px -> engine 720p px (multiply golden xy by this)."""


def _aa_to_R(aa: Float64[ndarray, "n 3"]) -> Float64[ndarray, "n 3 3"]:
    """Axis-angle -> rotation matrices via Rodrigues (vectorized)."""
    theta: Float64[ndarray, "n 1"] = np.linalg.norm(aa, axis=-1, keepdims=True)
    k: Float64[ndarray, "n 3"] = np.where(theta > 1e-12, aa / np.maximum(theta, 1e-12), 0.0)
    n: int = aa.shape[0]
    cross: Float64[ndarray, "n 3 3"] = np.zeros((n, 3, 3))
    cross[:, 0, 1], cross[:, 0, 2] = -k[:, 2], k[:, 1]
    cross[:, 1, 0], cross[:, 1, 2] = k[:, 2], -k[:, 0]
    cross[:, 2, 0], cross[:, 2, 1] = -k[:, 1], k[:, 0]
    sin_t: Float64[ndarray, "n 1 1"] = np.sin(theta)[..., None]
    cos_t: Float64[ndarray, "n 1 1"] = np.cos(theta)[..., None]
    return np.eye(3)[None] + sin_t * cross + (1.0 - cos_t) * (cross @ cross)


def _geodesic_deg(a: Float64[ndarray, "n 3"], b: Float64[ndarray, "n 3"]) -> Float64[ndarray, "n"]:
    """Relative rotation angle (deg) between two axis-angle sets."""
    rel: Float64[ndarray, "n 3 3"] = np.einsum("nij,nik->njk", _aa_to_R(a), _aa_to_R(b))
    trace: Float64[ndarray, "n"] = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(trace))


def _skeleton_scale_m(golden_joints: Float64[ndarray, "f j 3"]) -> float:
    """Per-clip H = median over frames of the joint-bbox diagonal (meters)."""
    extent: Float64[ndarray, "f 3"] = golden_joints.max(axis=1) - golden_joints.min(axis=1)
    return float(np.median(np.linalg.norm(extent, axis=1)))


def main(config: ValidateArtifactsConfig) -> int:
    cams: list[str] = list(np.load(config.run_dir / "masks.npz")["camera_names"])
    n_cams: int = len(cams)
    gvj = np.load(config.golden_ma3d_dir / f"verts_joints_body_id-{config.body_id:02d}.npz")
    gsm = np.load(config.golden_ma3d_dir / f"smplx_params_body_id-{config.body_id:02d}.npz", allow_pickle=True)
    golden_joints: Float64[ndarray, "f j 3"] = gvj["pred_joints"].astype(np.float64)
    n_golden: int = golden_joints.shape[0]
    skip: int = config.skip_first_frames
    frames: ndarray = np.arange(skip, n_golden)
    h_m: float = _skeleton_scale_m(golden_joints)
    gate_pct2_mm: float = 0.02 * h_m * 1000.0
    print(f"preset={config.preset}  H={h_m:.3f} m  (2%H={gate_pct2_mm:.1f} mm, 5%H={2.5 * gate_pct2_mm:.1f} mm)  skip_first={skip}")

    gates: list[GateResult] = []
    diagnostics: list[str] = []

    # ----------------------------------------------------------- 2D keypoints
    lmc = np.load(config.run_dir / "landmarks.npz")
    o_xy_all: Float64[ndarray, "fo c j 2"] = lmc["joints2d"][:, :, :, :2].astype(np.float64)
    o_vis_all: Float64[ndarray, "fo c j"] = lmc["visibility"].astype(np.float64)
    o_lm_pos: dict[int, int] = {int(f): i for i, f in enumerate(lmc["frame_indices"])}
    g_xy: Float64[ndarray, "f c j 2"] = np.zeros((n_golden, n_cams, 512, 2))
    g_vis: Float64[ndarray, "f c j"] = np.zeros((n_golden, n_cams, 512))
    for ci, cam in enumerate(cams):
        d = np.load(config.golden_ma2d_dir / f"{cam}.npz", allow_pickle=True)
        g_xy[:, ci] = d["landmarks"][:, config.body_id, :, :2].astype(np.float64) * config.src_to_engine_scale
        g_vis[:, ci] = d["visibilities"][:, config.body_id].astype(np.float64)
    # Frames the dump never recorded (subject missing in some camera that tick)
    # are excluded — mirrors the SMPL-X `common` guard below; a bare o_lm_pos[f]
    # would KeyError. Missing frames stay NaN in e_fc and drop out at nanmean.
    present_lm: Bool[ndarray, "f"] = np.array([f in o_lm_pos for f in range(n_golden)])
    rows: ndarray = np.array([o_lm_pos.get(f, 0) for f in range(n_golden)])
    o_xy: Float64[ndarray, "f c j 2"] = o_xy_all[rows]
    o_vis: Float64[ndarray, "f c j"] = o_vis_all[rows]
    both_vis: Bool[ndarray, "f c j"] = (g_vis > 0.5) & (o_vis > 0.5)
    golden_vis: Bool[ndarray, "f c j"] = g_vis > 0.5
    px_err: Float64[ndarray, "f c j"] = np.linalg.norm(o_xy - g_xy, axis=-1)
    e_fc: Float64[ndarray, "f c"] = np.full((n_golden, n_cams), np.nan)
    for f in range(n_golden):
        if not present_lm[f]:
            continue
        for c in range(n_cams):
            gm: Bool[ndarray, "j"] = golden_vis[f, c]
            if gm.sum() < 2:
                continue
            pts: Float64[ndarray, "k 2"] = g_xy[f, c][gm]
            diag: float = float(np.hypot(*(pts.max(axis=0) - pts.min(axis=0))))
            sel: Bool[ndarray, "j"] = both_vis[f, c]
            if sel.sum() >= 8 and diag > 1.0:
                e_fc[f, c] = px_err[f, c][sel].mean() / diag
    e_f: Float64[ndarray, "f"] = np.nanmean(e_fc, axis=1)[frames]
    e_f = e_f[~np.isnan(e_f)]
    kpt_p95_pct: float = float(np.percentile(e_f, 95) * 100.0)
    gates.append(GateResult("kpt2d p95 (norm reproj err)", kpt_p95_pct, 2.0, "%", False, config.preset == "quality"))
    diagnostics.append(f"kpt2d: mean {e_f.mean() * 100:.2f}%  p95 {kpt_p95_pct:.2f}%  max {e_f.max() * 100:.2f}%")

    # ----------------------------------------------------------- triangulated 3D
    tric = np.load(config.run_dir / "triangulated.npz")
    t_pos: dict[int, int] = {int(f): i for i, f in enumerate(tric["frame_indices"])}
    present_tri: Bool[ndarray, "f"] = np.array([f in t_pos for f in range(n_golden)])
    trows: ndarray = np.array([t_pos.get(f, 0) for f in range(n_golden)])
    t_pts: Float64[ndarray, "f j 3"] = tric["points"][trows].astype(np.float64)
    t_val: Bool[ndarray, "f j"] = tric["valid"][trows]
    g_tri: Float64[ndarray, "f j 3"] = gsm["triangulated_3d_pts"].astype(np.float64)
    both3d: Bool[ndarray, "f j"] = t_val & (np.linalg.norm(g_tri, axis=-1) > 1e-6)
    e3: Float64[ndarray, "f j"] = np.linalg.norm(t_pts - g_tri, axis=-1)
    ef3: Float64[ndarray, "f"] = np.full(n_golden, np.nan)
    for f in range(n_golden):
        if present_tri[f] and both3d[f].sum() >= 8:
            ef3[f] = e3[f][both3d[f]].mean()
    ef3g: Float64[ndarray, "f"] = ef3[frames]
    ef3g = ef3g[~np.isnan(ef3g)]
    tri_p95_mm: float = float(np.percentile(ef3g, 95) * 1000.0)
    gates.append(GateResult("tri3d p95 (both-valid)", tri_p95_mm, gate_pct2_mm, "mm", False, config.preset == "quality"))
    diagnostics.append(f"tri3d: mean {ef3g.mean() * 1000:.1f}mm  p95 {tri_p95_mm:.1f}mm  max {ef3g.max() * 1000:.1f}mm  (gate {gate_pct2_mm:.1f}mm)")

    # ----------------------------------------------------------- SMPL-X (identifiable)
    fits = np.load(config.run_dir / "smplx_fits.npz")
    f_pos: dict[int, int] = {int(f): i for i, f in enumerate(fits["frame_indices"])}
    common: list[int] = [f for f in range(skip, n_golden) if f in f_pos]
    fr: ndarray = np.array([f_pos[f] for f in common])
    o_pose: Float64[ndarray, "F 165"] = fits["pose"][fr].astype(np.float64)
    o_trans: Float64[ndarray, "F 3"] = fits["trans"][fr].astype(np.float64)
    o_betas: Float64[ndarray, "nb"] = fits["betas"].astype(np.float64)
    o_verts: Float64[ndarray, "F v 3"] = fits["vertices"][fr].astype(np.float64)
    g_pose: Float64[ndarray, "F 165"] = gsm["smplx_pose"][common].astype(np.float64)
    g_trans: Float64[ndarray, "F 3"] = gsm["smplx_translation"][common].astype(np.float64)
    g_betas: Float64[ndarray, "nb"] = gsm["smplx_betas"][0].astype(np.float64)
    g_verts: Float64[ndarray, "F v 3"] = gvj["pred_vertices"][common].astype(np.float64)
    n_common: int = len(common)

    # pose gate := PVE (the surface realization of pose)
    pve: Float64[ndarray, "F"] = np.linalg.norm(o_verts - g_verts, axis=-1).mean(axis=1) * 1000.0
    pve_p95_mm: float = float(np.percentile(pve, 95))
    pve_gate_mm: float = 27.0 if config.preset == "quality" else 30.0
    gates.append(GateResult("smplx pose->PVE p95", pve_p95_mm, pve_gate_mm, "mm", False, True))

    # betas gate := shape-space vertex displacement
    shapedirs: Float64[ndarray, "v 3 nsd"] = np.load(config.smplx_model_npz, allow_pickle=True)["shapedirs"].astype(np.float64)
    nb: int = min(config.n_betas, o_betas.shape[0], g_betas.shape[0])
    dbeta: Float64[ndarray, "nb"] = o_betas[:nb] - g_betas[:nb]
    vtx_disp: Float64[ndarray, "v 3"] = shapedirs[:, :, :nb] @ dbeta
    betas_disp_mm: float = float(np.linalg.norm(vtx_disp, axis=-1).max() * 1000.0)
    betas_gate_mm: float = gate_pct2_mm if config.preset == "quality" else 2.5 * gate_pct2_mm
    gates.append(GateResult("smplx betas->shape disp (max vtx)", betas_disp_mm, betas_gate_mm, "mm", False, True))

    # trans gate := absolute mm
    dtr: Float64[ndarray, "F"] = np.linalg.norm(o_trans - g_trans, axis=-1) * 1000.0
    trans_p95_mm: float = float(np.percentile(dtr, 95))
    trans_gate_mm: float = gate_pct2_mm if config.preset == "quality" else 2.5 * gate_pct2_mm
    gates.append(GateResult("smplx trans p95", trans_p95_mm, trans_gate_mm, "mm", False, True))

    # pose geodesic diagnostics (NOT gated)
    groups: dict[str, tuple[int, int]] = {"global": (0, 3), "body": (3, 66), "jaw": (66, 69), "eyes": (69, 75), "hands": (75, 165)}
    geo_bits: list[str] = []
    for gname, (a, b) in groups.items():
        nj: int = (b - a) // 3
        ang: Float64[ndarray, "F nj"] = _geodesic_deg(
            g_pose[:, a:b].reshape(n_common * nj, 3), o_pose[:, a:b].reshape(n_common * nj, 3)
        ).reshape(n_common, nj)
        geo_bits.append(f"{gname} p95 {np.percentile(ang.mean(axis=1), 95):.1f}deg")
    diagnostics.append("smplx pose geodesic (diag, non-identifiable): " + ", ".join(geo_bits))
    diagnostics.append(f"smplx: PVE p95 {pve_p95_mm:.1f}mm  trans p95 {trans_p95_mm:.1f}mm  betas-disp {betas_disp_mm:.1f}mm")

    # ----------------------------------------------------------- masks
    gmask = np.load(config.golden_masks_npz)
    cmask = np.load(config.run_dir / "masks.npz")
    g_pk: ndarray = gmask["packed"]  # materialize once (savez_compressed re-decompresses per index otherwise)
    c_pk: ndarray = cmask["packed"]
    c_pos: dict[int, int] = {int(f): i for i, f in enumerate(cmask["frame_indices"])}
    hw: int = int(gmask["mask_hw"][0]) * int(gmask["mask_hw"][1])
    iou: Float64[ndarray, "f c"] = np.full((n_golden, n_cams), np.nan)
    for f in range(n_golden):
        if f not in c_pos:
            continue  # frame absent from the dump; iou[f] stays NaN and is dropped below
        cf: int = c_pos[f]
        for c in range(n_cams):
            a_m: Bool[ndarray, "hw"] = np.unpackbits(g_pk[f, c])[:hw].astype(bool)
            b_m: Bool[ndarray, "hw"] = np.unpackbits(c_pk[cf, c])[:hw].astype(bool)
            union: int = int(np.logical_or(a_m, b_m).sum())
            iou[f, c] = 1.0 if union == 0 else float(np.logical_and(a_m, b_m).sum()) / union
    iou_g: Float64[ndarray, "n"] = iou[frames].ravel()
    iou_g = iou_g[~np.isnan(iou_g)]  # drop (frame,cam) pairs from frames absent in the dump
    mean_iou: float = float(iou_g.mean())
    p5_iou: float = float(np.percentile(iou_g, 5))
    p1_iou: float = float(np.percentile(iou_g, 1))
    min_iou: float = float(iou_g.min())
    # Tail gate uses p1, not raw min: over ~1900 (frame,cam) pairs the single
    # worst is hostage to one causal fast-motion frame (the jump apex, which the
    # offline bidirectional golden smooths with future frames) — the dig flagged
    # min/max as "hostage to one-frame dropouts; prefer p1". A min>=0.50 floor
    # still trips on a true collapse. mean/p5 carry the "within X%" intent.
    if config.preset == "quality":
        gates.append(GateResult("masks mean IoU", mean_iou, 0.95, "IoU", True, True))
        gates.append(GateResult("masks p5 IoU", p5_iou, 0.90, "IoU", True, True))
        gates.append(GateResult("masks p1 IoU", p1_iou, 0.80, "IoU", True, True))
        gates.append(GateResult("masks min IoU (collapse floor)", min_iou, 0.50, "IoU", True, True))
    else:
        gates.append(GateResult("masks mean IoU", mean_iou, 0.90, "IoU", True, True))
        gates.append(GateResult("masks p1 IoU", p1_iou, 0.70, "IoU", True, True))
        gates.append(GateResult("masks min IoU (collapse floor)", min_iou, 0.50, "IoU", True, True))
    per_cam: str = ", ".join(f"{cams[c]} {iou[frames, c].mean():.3f}" for c in range(n_cams))
    diagnostics.append(f"masks: mean {mean_iou:.3f}  p5 {p5_iou:.3f}  p1 {p1_iou:.3f}  min {min_iou:.3f}  | per-cam {per_cam}")

    # ----------------------------------------------------------- verdict
    print("\nDIAGNOSTICS")
    for line in diagnostics:
        print(f"  {line}")
    print(f"\n{config.preset.upper()} PRESET GATES (vs original DAG, {n_golden} golden frames, skip {skip}):")
    print(f"  {'gate':<36}{'value':>10}{'thresh':>10}  result")
    applied_pass: bool = True
    for g in gates:
        cmp: str = ">=" if g.higher_is_better else "<="
        status: str = "PASS" if g.passed else "FAIL"
        tag: str = "" if g.applies else "  (diag)"
        print(f"  {g.name:<36}{g.value:>9.2f}{g.unit:<1} {cmp}{g.threshold:>7.2f}{g.unit:<1}  {status}{tag}")
        if g.applies and not g.passed:
            applied_pass = False
    print(f"\nRESULT ({config.preset}): {'PASS' if applied_pass else 'FAIL'}")
    return 0 if applied_pass else 1


if __name__ == "__main__":
    sys.exit(main(tyro.cli(ValidateArtifactsConfig)))
