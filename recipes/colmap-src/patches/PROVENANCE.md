# Provenance of the Caspar adapter patches

Both patches apply, in this order, to colmap/colmap `137f21680c79d3f2fffbb1d11deb367162aa6f83` (main, 2026-11-15, after
FlexGlue #4908..#4969). `SHA256SUMS` hashes all 274 files they touch (verified by the recipe's Step 0 before the build).

| Patch | Kind | Content |
|---|---|---|
| `add-caspar-adapters.patch` | hand-written, 4 files (56 KB) | `OpenCVFisheyeAdapter` and `FullOpenCVAdapter` in `caspar_model_adapter.h`, their dispatch in `bundle_adjustment_caspar.cc`, their residuals plus the SumStore padding-thread fix in `Symforce-Caspar/caspar_generate.py`, two FAQ lines |
| `add-caspar-adapters-generated-f32.patch` | Symforce output (2.7 MB) | 266 new fp32 kernel files (224 OPENCV_FISHEYE, 42 FULL_OPENCV) and the 4 regenerated aggregates (`solver.{h,cc}`, `caspar_mappings.{h,cu}`) |

## History

The adapters were first written against COLMAP 4.2.0 (`be5e2916`): OPENCV_FISHEYE on the fork branch
`caspar-opencv-fisheye` @ c03b1f38 (pablovela5620/pyCuSFM, `packages/pycolmap-caspar-fisheye`), FULL_OPENCV on
`caspar-full-opencv` @ 06871fdc + fc4a8e6c (local clone; the setter calls use Symforce's `FullOpencvSplit...` names).
For FlexGlue the recipe moved to COLMAP main. The hand-written changes applied there without conflict (upstream changed
only SPDX headers, `#ifdef CASPAR_ENABLED` guards, and a warning in place of a throw for invalid camera parameters); the
port is branch `caspar-adapters-main` @ 391f7467 (generated kernels @ 700de612) in a local clone of colmap main.

FULL_OPENCV follows upstream PR #4657 (fixed-calibration THIN_PRISM_FISHEYE): FULL_OPENCV's 12 parameters do not fit a
tunable merged or split calib in Caspar's 48 KB shared-memory budget, so only two factors are generated, with the
calibration as constant sequential data (pose + point tunable; point tunable). Any other variant (refined intrinsics,
refined `sensor_from_rig`, fixed points) fails with `LOG(FATAL_THROW)` instead of silently dropping observations.

## Generator

Upstream PR #4799 says it regenerated the kernels with SymForce main plus open SymForce PRs. Checked on 2026-11-15:
plain `symforce==0.12.0` (PyPI) running upstream main's unmodified `caspar_generate.py` reproduces upstream main's
`generated/f32` **byte for byte (0 of 488 files differ)** when the output directory is outside the checkout (clang-format
then falls back to its default style, as upstream's tree is formatted). SymForce main @ 2bcd467 plus PRs #472/#473/#475
does not reproduce it (zero-initialised registers, a different solver loop), so 0.12.0 is the generator.

```bash
# generator env: pyCuSFM packages/pycolmap-caspar-fisheye/generator/pixi.{toml,lock} (symforce 0.12.0, clang-format 21.1.2)
python src/thirdparty/Symforce-Caspar/caspar_generate.py /tmp/caspar-gen/f32 f32   # from the patched checkout; ~100 s
```

Only the new files (`kernel_*OpenCVFisheye*`, `kernel_*opencv_fisheye*`, `kernel_FullOpenCVPose_*`, `kernel_full_opencv_*`)
and the four aggregates are taken. Our SumStore padding guard (the hazard reported on upstream PR #4611) also changes
the stock PINHOLE / SIMPLE_RADIAL kernels' `SumStore` lines; those stay as upstream ships them, as before.

Cut and checked:

```bash
git diff --full-index 137f2168 HEAD -- doc/faq.rst src/colmap/estimators/bundle_adjustment_caspar.cc \
  src/colmap/estimators/caspar/caspar_model_adapter.h src/thirdparty/Symforce-Caspar/caspar_generate.py   # hand-written
git diff --full-index 137f2168 HEAD -- src/thirdparty/Symforce-Caspar/generated/f32                      # generated
git diff --name-only 137f2168 HEAD | xargs sha256sum > SHA256SUMS
# fresh checkout of 137f2168: git apply both patches; sha256sum -c SHA256SUMS -> 274/274
```

Drop an adapter when upstream CASPAR supports its model.

## caspar-pcg-beta-recurrence.patch and caspar-reduced-camera-system.patch

Generated, not hand-edited: `tools/caspar_solver_edits.py` is their tracked source. On a checkout of the pin with the
three patches above applied (in recipe order):

```bash
git add -A src doc && git commit -qm base
python recipes/colmap-src/tools/caspar_solver_edits.py beta <checkout> && git -C <checkout> commit -qam beta
python recipes/colmap-src/tools/caspar_solver_edits.py reduced <checkout>
git -C <checkout> diff --full-index base beta  > patches/caspar-pcg-beta-recurrence.patch
git -C <checkout> diff --full-index beta      > patches/caspar-reduced-camera-system.patch
(cd <checkout> && sha256sum <the files listed in SHA256SUMS>) > patches/SHA256SUMS
```

The script asserts every text it replaces, and it reads the camera-side buffer spans from the generated allocation code
(the blocks before and after `Point` in each per-node vector), so a regenerated solver with a new camera node type is
covered without an edit, and a changed layout stops the script instead of producing a silently partial patch.

- **PCG beta** (`caspar-pcg-beta-recurrence.patch`, 2 files). Symforce 0.12's `solver.cc.jinja` calls
  `BetaFromNumDenom(beta_numerator, alpha_numerator)`, and only `DoAlphaFirst` writes `alpha_numerator`, so every
  iteration after the first divides by `rho_0`: the PCG loses conjugacy from its third iteration on.
  `BetaFromNumDenom` now stores `rho_k` as the next denominator. colsfm `tests/test_caspar_solver.py` checks it: a
  full-system PCG capped at 10 iterations per step ends 2.0e-5 above Ceres' minimum with the fix, 8.4e-5 without.
- **Reduced camera system** (`caspar-reduced-camera-system.patch`, 9 files; `SolverParams::reduced_camera_system`,
  off in Caspar, on in COLMAP's `CasparBundleAdjustmentOptions`): the Point nodes are eliminated inside the PCG. The
  right-hand side becomes `g_c - H_cp H_pp^-1 g_p`; the matvec (chosen once per solve) is the generated
  `DoJtjpDirect` run twice (point part of `p` zeroed, then `p = (0, -H_pp^-1 H_pc p)` accumulating into `w`, which
  enters holding `M p`), with `PointNormalize` as the exact damped `H_pp^-1`; points are back-substituted after the
  PCG; the PCG stops on the Ceres/rootba Q criterion (`reduced_pcg_eta` 0.1, at most `reduced_pcg_iter_max` 500).
  A zero reduced right-hand side (no camera gradient or no camera block) gives a zero camera step; `rho_k = 0`
  (exact convergence) stops before the next division; a non-finite or non-positive `alpha` stops before the step
  update, keeping the last finite iterate. `reduced_pcg_iter_max >= 1`, `reduced_pcg_eta` in (0, 1) and no
  per-iteration score exit are checked in COLMAP's adapter and again in `solve()`. The residual-norm readbacks only
  feed the full system's stop rule, so the reduced PCG skips them. New helpers: `Scale` and `Subtract` in
  solver_tools; `BundleAdjustmentCaspar.reduced_*` registered in the COLMAP CLI.
- Measurements. KITTI 06 (colsfm mapping rounds, f32): CASPAR alone went from 1,184 mm Sim(3) ATE to 816 mm; beta alone
  reaches 1,172 mm; the reduced system without the beta fix 1,086 mm; CASPAR + Ceres polish 776 mm, Ceres alone
  782 mm. Ceres f64 with CASPAR's own algorithm (CGNR, 20 CG iterations per step) stalls the same way, so f32 was never
  the cause. Full colsfm pipeline with the polish off vs main with it (Sim(3) ATE): KITTI 06 790.4 -> 774.7 mm, MSD
  MGO05 6.29 -> 6.36, EuRoC V1_01 33.89 -> 33.84, TartanAir P004 cube median 9.02 -> 9.40 over 5 runs each. Clip 007:
  the polish cost 467 s (146 iterations at 4 threads).
- Not done: CASPAR applies no robust loss (its factors take no weight input; IRLS would regenerate every factor).

Follow-up: move both edits into Symforce-Caspar's generator (`solver.cc.jinja` and `caspar_generate.py`), so the solver
is emitted with them and this script goes away. The beta fix is a one-line upstream report (draft, not filed).

