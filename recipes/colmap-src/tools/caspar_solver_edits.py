"""Source of the two CASPAR solver patches: edits applied to COLMAP's generated f32 Symforce-Caspar solver.

usage: python caspar_solver_edits.py {beta,reduced} <colmap checkout>

`beta` makes `caspar-pcg-beta-recurrence.patch`; `reduced` (on a checkout that has the beta edit) makes
`caspar-reduced-camera-system.patch`. Each edit asserts the exact text it replaces, so a regenerated solver that moved
fails here instead of patching the wrong place. The camera-side buffer spans are read from the generated allocation
code, not listed by node type: a new camera node type is covered without an edit. The regeneration commands and the
check that the patches still match this script are in `patches/PROVENANCE.md`.

Follow-up: these edits belong in Symforce-Caspar's `solver.cc.jinja` / `caspar_generate.py`, so the generator emits
them; until that lands upstream this script is their tracked source.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SOLVER_DIR = "src/thirdparty/Symforce-Caspar/generated/f32"
ELIMINATED = "Point"
"""The node type the reduced camera system eliminates; every other node type is a camera-side block."""


def sub(path: Path, old: str, new: str) -> None:
    """Replace `old`, which must occur exactly once, by `new`."""
    text: str = path.read_text()
    found: int = text.count(old)
    if found != 1:
        raise SystemExit(f"{path}: expected one `{old[:70]}`, found {found}")
    path.write_text(text.replace(old, new))


def beta(root: Path) -> None:
    """PCG beta_k = rho_k / rho_{k-1}. Symforce 0.12's template divides by `alpha_numerator`, which only `DoAlphaFirst`
    writes, i.e. by rho_0 for every k >= 1: the directions lose conjugacy from the third iteration on."""
    tools_h, tools_cu = root / SOLVER_DIR / "solver_tools.h", root / SOLVER_DIR / "solver_tools.cu"
    sub(tools_h, "template <typename T>\nvoid BetaFromNumDenom(const T* beta_num, const T* beta_denum, T* beta);",
        "// beta = beta_num / beta_denum, then beta_denum = beta_num: the next iteration divides by this one's r.z.\n"
        "template <typename T>\nvoid BetaFromNumDenom(const T* beta_num, T* beta_denum, T* beta);")
    sub(tools_h, "extern template void BetaFromNumDenom<float>(const float* beta_num, const float* beta_denum,\n"
        "                                             float* beta);",
        "extern template void BetaFromNumDenom<float>(const float* beta_num, float* beta_denum, float* beta);")
    sub(tools_h, "extern template void BetaFromNumDenom<double>(const double* beta_num, const double* beta_denum,\n"
        "                                              double* beta);",
        "extern template void BetaFromNumDenom<double>(const double* beta_num, double* beta_denum, double* beta);")
    sub(tools_cu, "__global__ void BetaFromNumDenomKernel(const T* beta_num, const T* beta_denum, T* beta) {\n"
        "  *beta = *beta_num / *beta_denum;\n}",
        "__global__ void BetaFromNumDenomKernel(const T* beta_num, T* beta_denum, T* beta) {\n"
        "  *beta = *beta_num / *beta_denum;\n  *beta_denum = *beta_num;\n}")
    sub(tools_cu, "void BetaFromNumDenom(const T* beta_num, const T* beta_denum, T* beta) {",
        "void BetaFromNumDenom(const T* beta_num, T* beta_denum, T* beta) {")
    sub(tools_cu, "template void BetaFromNumDenom<float>(const float* beta_num, const float* beta_denum, float* beta);",
        "template void BetaFromNumDenom<float>(const float* beta_num, float* beta_denum, float* beta);")
    sub(tools_cu, "template void BetaFromNumDenom<double>(const double* beta_num, const double* beta_denum,\n"
        "                                       double* beta);",
        "template void BetaFromNumDenom<double>(const double* beta_num, double* beta_denum, double* beta);")


def family_nodes(solver_cc: str, family: str) -> list[str]:
    """Node types of one per-node vector family (`p`, `r_k`, ...) in allocation order, checked against the layout rule
    the span operations rely on: one block per node type (plus a zero-size end marker in some families), allocated back
    to back in `r_k`'s node order, with the eliminated type strictly inside. Its camera-side part is then exactly the two
    spans around the eliminated block."""
    allocations: list[tuple[str, str]] = re.findall(r"\n  nodes__(\w+?)__(\w+?)_(?:end__)? =\s*assign_and_increment", solver_cc)
    positions: list[int] = [index for index, (_, name) in enumerate(allocations) if name == family]
    if not positions or positions != list(range(positions[0], positions[0] + len(positions))):
        raise SystemExit(f"generated layout changed: `{family}` blocks are not allocated back to back")
    nodes: list[str] = list(dict.fromkeys(allocations[index][0] for index in positions))
    reference: list[str] = list(dict.fromkeys(node for node, name in allocations if name == "r_k"))
    if nodes != reference or len(positions) not in (len(nodes), 2 * len(nodes)):
        raise SystemExit(f"generated layout changed: `{family}` is not one block per node type in r_k's order")
    if ELIMINATED not in nodes[1:-1]:
        raise SystemExit(f"generated layout changed: {ELIMINATED} is not strictly inside `{family}`")
    return nodes


def reduced(root: Path) -> None:
    """Reduced camera system, option-gated: eliminate the Point blocks inside the PCG (implicit Schur complement, exact
    damped point blocks, back-substitution) and stop the PCG on the Ceres / rootba Q criterion."""
    solver_dir: Path = root / SOLVER_DIR
    solver_cc: Path = solver_dir / "solver.cc"

    # --- parameters: off at this layer; COLMAP's adapter turns them on. Validated in `solve` before the device loop. ---
    sub(solver_dir / "solver_params.h", "  T solver_rel_decrease_min = 1.0f;\n",
        "  T solver_rel_decrease_min = 1.0f;\n"
        "  // Reduced camera system: eliminate the Point nodes inside the PCG and stop it on the Q criterion with\n"
        "  // reduced_pcg_eta in (0, 1), after at most reduced_pcg_iter_max (>= 1) iterations. Excludes the per-iteration\n"
        "  // score exits (pcg_rel_decrease_min, pcg_rel_score_exit must stay -1); pcg_iter_max and pcg_rel_error_exit\n"
        "  // apply to the full system only.\n"
        "  int reduced_camera_system = 0;\n  int reduced_pcg_iter_max = 500;\n  T reduced_pcg_eta = 0.1f;\n")
    sub(solver_dir / "solver_params.h", "        solver_rel_decrease_min(static_cast<T>(other.solver_rel_decrease_min)) {}",
        "        solver_rel_decrease_min(static_cast<T>(other.solver_rel_decrease_min)),\n"
        "        reduced_camera_system(other.reduced_camera_system),\n"
        "        reduced_pcg_iter_max(other.reduced_pcg_iter_max),\n"
        "        reduced_pcg_eta(static_cast<T>(other.reduced_pcg_eta)) {}")

    # --- solver_tools: Scale and Subtract over a float range ---
    tools_h, tools_cu = solver_dir / "solver_tools.h", solver_dir / "solver_tools.cu"
    sub(tools_h, "template <typename T>\nT ReadCuMem(const T* const data);\n",
        "template <typename T>\nT ReadCuMem(const T* const data);\n\n"
        "// x[i] *= a over [start, end).\ntemplate <typename T>\nvoid Scale(T* start, T* end, T a);\n\n"
        "// out[i] = a[i] - b[i] over [a_start, a_end); out may alias b.\ntemplate <typename T>\n"
        "void Subtract(const T* a_start, const T* a_end, const T* b, T* out);\n")
    sub(tools_h, "extern template double ReadCuMem<double>(const double* const data);\n",
        "extern template double ReadCuMem<double>(const double* const data);\n"
        "extern template void Scale<float>(float* start, float* end, float a);\n"
        "extern template void Scale<double>(double* start, double* end, double a);\n"
        "extern template void Subtract<float>(const float* a_start, const float* a_end, const float* b, float* out);\n"
        "extern template void Subtract<double>(const double* a_start, const double* a_end, const double* b, double* out);\n")
    sub(tools_cu, "// Explicit instantiations\n",
        "template <typename T>\n__global__ void ScaleKernel(T* x, size_t n, T a) {\n"
        "  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;\n  if (i < n) x[i] *= a;\n}\n\n"
        "template <typename T>\nvoid Scale(T* start, T* end, T a) {\n  const size_t n = end - start;\n"
        "  if (n > 0) ScaleKernel<T><<<(n + 255) / 256, 256>>>(start, n, a);\n}\n\n"
        "template <typename T>\n__global__ void SubtractKernel(const T* a, const T* b, T* out, size_t n) {\n"
        "  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;\n  if (i < n) out[i] = a[i] - b[i];\n}\n\n"
        "template <typename T>\nvoid Subtract(const T* a_start, const T* a_end, const T* b, T* out) {\n"
        "  const size_t n = a_end - a_start;\n  if (n > 0) SubtractKernel<T><<<(n + 255) / 256, 256>>>(a_start, b, out, n);\n}\n\n"
        "// Explicit instantiations\n")
    sub(tools_cu, "template double ReadCuMem<double>(const double* const data);\n",
        "template double ReadCuMem<double>(const double* const data);\n"
        "template void Scale<float>(float* start, float* end, float a);\n"
        "template void Scale<double>(double* start, double* end, double a);\n"
        "template void Subtract<float>(const float* a_start, const float* a_end, const float* b, float* out);\n"
        "template void Subtract<double>(const double* a_start, const double* a_end, const double* b, double* out);\n")

    # --- solver.h: the mode flag and the helpers ---
    sub(solver_dir / "solver.h", "  void DoJtjpDirect();\n",
        "  void DoJtjpDirect();\n"
        "  // Reduced camera system (params_.reduced_camera_system): see solver.cc.\n"
        "  bool reduced_ = false;  // this solve eliminates the Point nodes\n"
        "  void RcsPointSolve(float* in, float* out);\n"
        "  void RcsCopyCameras(const float* src_first, const float* src_point, const float* src_point_end,\n"
        "                      const float* src_last_end, float* dst_first, float* dst_point_end);\n"
        "  void RcsZeroCameras(float* first, float* point, float* point_end, float* last_end);\n"
        "  void RcsReduceRhs();\n  void RcsMatvec();\n  void RcsBackSubstitute();\n")

    # --- solver.cc: helpers. Camera-side ranges are two spans around the Point block, read from the layout. ---
    text: str = solver_cc.read_text()
    families: tuple[str, ...] = ("p", "z", "step", "r_k", "r_0", "w")
    nodes: list[str] = family_nodes(text, "r_k")
    for family in families:
        family_nodes(text, family)
    first, last = nodes[0], nodes[-1]
    after: str = nodes[nodes.index(ELIMINATED) + 1]

    def point_end(family: str) -> str:
        # The Point block ends where the next node type's block starts (any alignment gap is unused by the kernels).
        return f"nodes__{after}__{family}_"

    def cameras(family: str) -> str:
        return f"nodes__{first}__{family}_, nodes__{ELIMINATED}__{family}_, {point_end(family)}, nodes__{last}__{family}_end__"

    def points(family: str) -> str:
        return f"nodes__{ELIMINATED}__{family}_, {point_end(family)}"

    helpers: str = f'''
// ---------------------------------------------------------------------------------------------------------------
// Reduced camera system. The PCG above runs on the full normal equations, cameras and points together, so a long
// trajectory's low-frequency (drift) modes barely move in pcg_iter_max iterations. With reduced_camera_system the
// Point nodes are eliminated: the PCG runs on S = H_cc - H_cp H_pp^-1 H_pc (implicit, damped), the points are
// back-substituted after it, and the PCG stops on the Q criterion (Ceres / rootba) instead of the residual norm.
// Everything reuses the generated kernels: the matvec is DoJtjpDirect (it accumulates the coupling terms into w,
// which enters holding the block-diagonal M p), and PointNormalize is the exact damped H_pp^-1 per point. Keeping
// the Point part of r_k at zero keeps the Point parts of z, p, Mp and step at zero through the unchanged PCG updates.
// The camera side of a vector is every block before and after the Point block ({first} .. {last}, from the layout).
// ---------------------------------------------------------------------------------------------------------------
void GraphSolver::RcsPointSolve(float* in, float* out) {{
  PointNormalize(nodes__Point__precond_diag_, Point_num_, nodes__Point__precond_tril_, Point_num_, in, Point_num_,
                 solver__current_diag_, out, Point_num_, Point_num_);
}}

void GraphSolver::RcsCopyCameras(const float* src_first, const float* src_point, const float* src_point_end,
                                 const float* src_last_end, float* dst_first, float* dst_point_end) {{
  Copy(src_first, src_point, dst_first);
  Copy(src_point_end, src_last_end, dst_point_end);
}}

void GraphSolver::RcsZeroCameras(float* first, float* point, float* point_end, float* last_end) {{
  Zero(first, point);
  Zero(point_end, last_end);
}}

// r_k = r_0 reduced: g_c - H_cp H_pp^-1 g_p on the cameras, 0 on the points; Mp = r_k (M p_0 = r_0 in the PCG).
void GraphSolver::RcsReduceRhs() {{
  Copy(marker__r_0_start_, marker__r_0_end_, marker__r_k_start_);
  RcsPointSolve(nodes__Point__r_k_, nodes__Point__z_);
  Scale({points("z")}, -1.0f);
  RcsZeroCameras({cameras("p")});
  Copy({points("z")}, nodes__Point__p_);
  Copy(marker__r_k_start_, marker__r_k_end_, marker__w_start_);
  DoJtjpDirect();
  Copy(marker__w_start_, marker__w_end_, marker__r_k_start_);
  Zero({points("r_k")});
  Copy(marker__r_k_start_, marker__r_k_end_, marker__Mp_start_);
  Zero({points("p")});
  Zero({points("z")});
  Zero({points("w")});
}}

// w = S p on the cameras and 0 on the points; w enters holding M p, p has a zero Point part.
void GraphSolver::RcsMatvec() {{
  DoJtjpDirect();  // w_points = H_pc p
  RcsPointSolve(nodes__Point__w_, nodes__Point__z_);
  Scale({points("z")}, -1.0f);
  RcsCopyCameras({cameras("p")}, nodes__{first}__z_, {point_end("z")});
  RcsZeroCameras({cameras("p")});
  Copy({points("z")}, nodes__Point__p_);
  DoJtjpDirect();  // w_cameras += H_cp (-H_pp^-1 H_pc p)
  RcsCopyCameras({cameras("z")}, nodes__{first}__p_, {point_end("p")});
  Zero({points("p")});
  Zero({points("z")});
  Zero({points("w")});
}}

// step_points = H_pp^-1 (g_p - H_pc step_cameras).
void GraphSolver::RcsBackSubstitute() {{
  RcsCopyCameras({cameras("step")}, nodes__{first}__p_, {point_end("p")});
  Zero({points("p")});
  Zero(marker__w_start_, marker__w_end_);
  DoJtjpDirect();
  Subtract({points("r_0")}, nodes__Point__w_, nodes__Point__w_);
  RcsPointSolve(nodes__Point__w_, nodes__Point__step_);
  Zero({points("w")});
}}

'''
    sub(solver_cc, "void GraphSolver::DoAlphaFirst() {", helpers + "void GraphSolver::DoAlphaFirst() {")

    # --- solve(): validate before the device loop, pick the product once, guard the divisions ---
    sub(solver_cc, "  float up_scale = params_.diag_scaling_up;\n",
        "  float up_scale = params_.diag_scaling_up;\n"
        "  if (params_.reduced_camera_system != 0) {\n"
        "    if (params_.reduced_pcg_iter_max < 1 || !(params_.reduced_pcg_eta > 0.0f && params_.reduced_pcg_eta < 1.0f)) {\n"
        "      throw std::invalid_argument(\"reduced camera system: reduced_pcg_iter_max must be >= 1 and reduced_pcg_eta in (0, 1)\");\n"
        "    }\n"
        "    if (params_.pcg_rel_decrease_min != -1.0f || params_.pcg_rel_score_exit != -1.0f) {\n"
        "      throw std::invalid_argument(\"reduced camera system: pcg_rel_decrease_min and pcg_rel_score_exit must be -1\");\n"
        "    }\n"
        "  }\n"
        "  // With no Point nodes there is nothing to eliminate: the full system is the camera system.\n"
        "  reduced_ = params_.reduced_camera_system != 0 && Point_num_ > 0;\n"
        "  const int pcg_iter_cap = reduced_ ? params_.reduced_pcg_iter_max : params_.pcg_iter_max;\n"
        "  void (GraphSolver::*matvec)() = reduced_ ? &GraphSolver::RcsMatvec : &GraphSolver::DoJtjpDirect;\n")
    sub(solver_cc, "    score_best_pcg = score_best;\n    for (pcg_iter_ = 0; pcg_iter_ < params_.pcg_iter_max; pcg_iter_++) {",
        "    score_best_pcg = score_best;\n"
        "    double pcg_q = 0.0;  // the reduced PCG's model decrease, with this iteration's rho and alpha\n"
        "    double pcg_rho = 0.0;\n    double pcg_alpha = 0.0;\n"
        "    if (reduced_) {\n      RcsReduceRhs();\n    }\n"
        "    for (pcg_iter_ = 0; pcg_iter_ < pcg_iter_cap; pcg_iter_++) {")
    # Iteration 0. Reduced: alpha_0 = rho_0 / p.Sp is 0/0 when the reduced right-hand side is zero (no camera gradient,
    # or no camera block); the camera step is then zero and the points are back-substituted alone.
    sub(solver_cc, "        Copy(marker__r_k_start_, marker__r_k_end_, marker__w_start_);\n        DoJtjpDirect();\n"
        "        DoAlphaFirst();\n        DoUpdateStepFirst();\n",
        "        Copy(marker__r_k_start_, marker__r_k_end_, marker__w_start_);\n        (this->*matvec)();\n"
        "        DoAlphaFirst();\n"
        "        if (reduced_) {\n"
        "          pcg_rho = ReadCuMem(solver__alpha_numerator_);\n"
        "          pcg_alpha = ReadCuMem(solver__alpha_);\n"
        "          if (!(pcg_rho > 0.0) || !std::isfinite(pcg_alpha)) {\n"
        "            Zero(nodes__" + first + "__step_, nodes__" + last + "__step_end__);\n"
        "            break;\n"
        "          }\n"
        "        }\n"
        "        DoUpdateStepFirst();\n")
    # Iteration k. Reduced: stop before a division by zero (rho_k = 0: converged exactly) and keep the last finite
    # iterate when alpha_k breaks down; DoBeta already wrote rho_k (and, with the beta fix, the next divisor).
    sub(solver_cc, "        DoBeta();\n        DoUpdateP();\n        DoUpdateMp();\n        DoJtjpDirect();\n"
        "        DoAlpha();\n        DoUpdateStep();\n",
        "        DoBeta();\n"
        "        if (reduced_) {\n"
        "          pcg_rho = ReadCuMem(solver__beta_numerator_);\n"
        "          if (!(pcg_rho > 0.0) || !std::isfinite(pcg_rho)) {\n            break;\n          }\n"
        "        }\n"
        "        DoUpdateP();\n        DoUpdateMp();\n        (this->*matvec)();\n        DoAlpha();\n"
        "        if (reduced_) {\n"
        "          pcg_alpha = ReadCuMem(solver__alpha_);\n"
        "          if (!(pcg_alpha > 0.0) || !std::isfinite(pcg_alpha)) {\n            break;\n          }\n"
        "        }\n"
        "        DoUpdateStep();\n")
    sub(solver_cc,
        "      if (pcg_r_kp1_norm2_ < pcg_r_0_norm2_ * params_.pcg_rel_error_exit) {\n        break;\n      }\n    }\n"
        "    pcg_iter_ = std::min(pcg_iter_, params_.pcg_iter_max - 1);\n",
        "      if (reduced_) {\n"
        "        // Q(x_k) = Q(x_{k-1}) - alpha_k rho_k / 2; stop once k (Q_{k-1} - Q_k) / -Q_k < eta.\n"
        "        const double decrease = 0.5 * pcg_alpha * pcg_rho;\n"
        "        pcg_q -= decrease;\n"
        "        if (pcg_q < 0.0 && (pcg_iter_ + 1) * decrease < -pcg_q * params_.reduced_pcg_eta) {\n          break;\n        }\n"
        "      } else if (pcg_r_kp1_norm2_ < pcg_r_0_norm2_ * params_.pcg_rel_error_exit) {\n        break;\n      }\n    }\n"
        "    pcg_iter_ = std::min(pcg_iter_, pcg_iter_cap - 1);\n"
        "    if (reduced_) {\n      RcsBackSubstitute();\n    }\n")
    # The residual norms feed only the full system's stop rule: the reduced PCG skips their blocking reads.
    sub(solver_cc, "  pcg_r_0_norm2_ = ReadCuMem(solver__r_0_norm2_tot_);\n  pcg_r_kp1_norm2_ = ReadCuMem(solver__r_kp1_norm2_tot_);\n",
        "  if (!reduced_) {\n    pcg_r_0_norm2_ = ReadCuMem(solver__r_0_norm2_tot_);\n"
        "    pcg_r_kp1_norm2_ = ReadCuMem(solver__r_kp1_norm2_tot_);\n  }\n")
    sub(solver_cc, "  pcg_r_kp1_norm2_ = ReadCuMem(solver__r_kp1_norm2_tot_);\n}\n",
        "  if (!reduced_) {\n    pcg_r_kp1_norm2_ = ReadCuMem(solver__r_kp1_norm2_tot_);\n  }\n}\n")
    if "#include <cmath>" not in solver_cc.read_text():
        sub(solver_cc, "#include <algorithm>\n", "#include <algorithm>\n#include <cmath>\n")

    # --- COLMAP: options, validation, mapping, CLI registration, binding (the adapter) ---
    estimators: Path = root / "src/colmap/estimators"
    sub(estimators / "bundle_adjustment_caspar.h", "  double solver_rel_decrease_min = 1.0;\n",
        "  double solver_rel_decrease_min = 1.0;\n"
        "  // Eliminate the 3D points inside the solver's PCG (reduced camera system) and stop the PCG on the Q criterion\n"
        "  // with reduced_pcg_eta in (0, 1) after at most reduced_pcg_iter_max (>= 1) iterations; pcg_iter_max and\n"
        "  // pcg_rel_error_exit then do not apply, and pcg_rel_decrease_min / pcg_rel_score_exit must stay -1.\n"
        "  // false = the full-system PCG.\n"
        "  bool reduced_camera_system = true;\n  int reduced_pcg_iter_max = 500;\n  double reduced_pcg_eta = 0.1;\n")
    sub(estimators / "bundle_adjustment_caspar.cc", "      params.solver_rel_decrease_min = co.solver_rel_decrease_min;\n",
        "      params.solver_rel_decrease_min = co.solver_rel_decrease_min;\n"
        "      if (co.reduced_camera_system) {\n"
        "        THROW_CHECK_GE(co.reduced_pcg_iter_max, 1);\n"
        "        THROW_CHECK(std::isfinite(co.reduced_pcg_eta) && co.reduced_pcg_eta > 0.0 && co.reduced_pcg_eta < 1.0)\n"
        "            << \"reduced_pcg_eta must be in (0, 1)\";\n"
        "        THROW_CHECK(co.pcg_rel_decrease_min == -1.0 && co.pcg_rel_score_exit == -1.0)\n"
        "            << \"reduced_camera_system excludes pcg_rel_decrease_min and pcg_rel_score_exit\";\n"
        "      }\n"
        "      params.reduced_camera_system = co.reduced_camera_system ? 1 : 0;\n"
        "      params.reduced_pcg_iter_max = co.reduced_pcg_iter_max;\n"
        "      params.reduced_pcg_eta = co.reduced_pcg_eta;\n")
    sub(root / "src/colmap/controllers/option_manager.cc",
        "  AddDefaultOption(\"BundleAdjustmentCaspar.gpu_index\",\n",
        "  AddDefaultOption(\"BundleAdjustmentCaspar.reduced_camera_system\",\n"
        "                   &bundle_adjustment->caspar->reduced_camera_system);\n"
        "  AddDefaultOption(\"BundleAdjustmentCaspar.reduced_pcg_iter_max\",\n"
        "                   &bundle_adjustment->caspar->reduced_pcg_iter_max);\n"
        "  AddDefaultOption(\"BundleAdjustmentCaspar.reduced_pcg_eta\",\n"
        "                   &bundle_adjustment->caspar->reduced_pcg_eta);\n"
        "  AddDefaultOption(\"BundleAdjustmentCaspar.gpu_index\",\n")
    sub(root / "src/pycolmap/estimators/bundle_adjustment.cc",
        '                         "Minimum relative solver decrease.")\n',
        '                         "Minimum relative solver decrease.")\n'
        '          .def_readwrite("reduced_camera_system",\n                         &CasparBAOpts::reduced_camera_system,\n'
        '                         "Eliminate the 3D points inside the PCG (reduced camera system).")\n'
        '          .def_readwrite("reduced_pcg_iter_max",\n                         &CasparBAOpts::reduced_pcg_iter_max,\n'
        '                         "PCG iteration cap with reduced_camera_system (>= 1).")\n'
        '          .def_readwrite("reduced_pcg_eta",\n                         &CasparBAOpts::reduced_pcg_eta,\n'
        '                         "Q-criterion forcing term with reduced_camera_system, in (0, 1).")\n')


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("beta", "reduced"):
        raise SystemExit(__doc__)
    {"beta": beta, "reduced": reduced}[sys.argv[1]](Path(sys.argv[2]))
    print(f"{sys.argv[1]}: applied to {sys.argv[2]}")
