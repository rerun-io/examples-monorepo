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

## caspar-reduced-camera-system.patch

Hand-written on top of the three patches above (applied last), 8 files, +244 -16. Reproduce it with
`python rcs_edit.py <checkout>` (colsfm-speed-5x scratch, scripts/rcs_edit.py) on a checkout that has the other
patches applied, then `git diff --full-index -- src`.

- **PCG beta.** Symforce 0.12's `solver.cc.jinja` calls `BetaFromNumDenom(beta_numerator, alpha_numerator)`, and only
  `DoAlphaFirst` writes `alpha_numerator`, so every iteration after the first divides by `rho_0`: the PCG loses
  conjugacy after its second iteration. `BetaFromNumDenom` now stores `rho_k` as the next denominator.
- **Reduced camera system** (`SolverParams::reduced_camera_system`, off in Caspar, on in COLMAP's
  `CasparBundleAdjustmentOptions`): the Point nodes are eliminated inside the PCG. The right-hand side becomes
  `g_c - H_cp H_pp^-1 g_p`; the matvec is the generated `DoJtjpDirect` run twice (point part of `p` zeroed, then
  `p = (0, -H_pp^-1 H_pc p)` accumulating into `w`, which enters holding `M p`), with `PointNormalize` as the exact
  damped `H_pp^-1`; points are back-substituted after the PCG; the PCG stops on the Ceres/rootba Q criterion
  (`reduced_pcg_eta` 0.1, at most `reduced_pcg_iter_max` 500). New helpers: `Scale` and `Subtract` in solver_tools.
- Why both: on KITTI 06 (colsfm mapping rounds, f32) CASPAR alone went from 1,184 mm Sim(3) ATE to 816 mm; beta
  alone reaches 1,172 mm; the reduced system without the beta fix 1,086 mm. Ceres f64 with CASPAR's own algorithm
  (CGNR, 20 CG iterations per step) stalls the same way, so f32 was never the cause.
