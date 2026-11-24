# Provenance of the colmap-src patches

The five patches apply, in `recipe.yaml`'s order, to colmap/colmap `137f21680c79d3f2fffbb1d11deb367162aa6f83` (main,
2026-11-15, after FlexGlue #4908..#4969). `SHA256SUMS` hashes all 284 files they touch (verified by the recipe's Step 0
before the build); re-hash with the `git diff --name-only` command below, all five patches committed.

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
git diff --name-only 137f2168 HEAD | xargs sha256sum > SHA256SUMS   # with all five patches committed
# fresh checkout of 137f2168: git apply the five patches; sha256sum -c SHA256SUMS -> 284/284
```

Drop an adapter when upstream CASPAR supports its model.

## OPENCV_FISHEYE ray angle (2026-11-25)

`opencv_fisheye_core` takes the off-axis angle from the unit ray, `theta = atan2(|(x, y)|, z)` with
`(x, y, z) = P / |P|`, instead of `atan(|(X, Y)| / Z)`. Equal for Z > 0 (7e-4 px max on a TartanAir 195 deg model); the
X/Z form mirrors a point at Z = 0, so a step that carried a rim observation (89-90 deg) across the image plane raised
the cost x20-x77, every LM trial was rejected and CASPAR returned its start values as CONVERGENCE
(`CONVERGED_DIAG_EXIT`). Rays past 90 deg now project to their Kannala-Brandt pixel; COLMAP's own `ImgFromCam` still
refuses them, and colsfm's post-solve guards remove those observations. The ray is normalised before the f32 epsilons
(`sqrt(r^2 + 1e-6)`, `atan2`'s 1e-6 on Z): on metric X, Y and Z they moved the pixel by up to 2 px for points 1 cm away,
so the factor depended on the scene scale. Root cause: `loop-and-quality/a7/caspar-fisheye-rootcause.md` (NAS scratch).

Both adapter patches were re-cut as above: 45 regenerated `opencv_fisheye*` kernels, the aggregates unchanged, every
other generated file byte for byte as the generator env gives it. Source: local branch `caspar-fisheye-fix-ray-main` @
`573b507f` in `~/0Dev/forks/colmap-caspar-fisheye` on pablo-dl-server (pushed nowhere): `4c5ea2aa` hand-written,
`8a608f5b` generated, then the other three patches, one commit each. colsfm `tests/test_caspar_solver.py` checks it: the
rim rig at scene scales 1, 0.01 and 0.002 ends at Ceres' cost, both solvers scored on one set of observations (in front
at the start and in Ceres' solution).

## caspar-diag-exit-no-convergence.patch, caspar-rig-refine.patch, caspar-rig-refine-generated-f32.patch

Applied after the five patches above (`refine_sensor_from_rig` on CASPAR, 2026-11-24).

- **Diag exit** (`caspar-diag-exit-no-convergence.patch`, hand-written, 1 file): `CONVERGED_DIAG_EXIT` (the LM damping
  passed `diag_exit_value` after uphill steps) is reported as `NO_CONVERGENCE`, not `CONVERGENCE`. It hid the Mission
  office fisheye stall (observations near 90 deg off-axis).
- **Rig refinement, hand-written** (`caspar-rig-refine.patch`, 5 files): `caspar_generate.py` gets one tunable
  `<Model>SensorFromRig` Pose3 node per model and 40 "rig" factors (sensor_from_rig tunable), registered after every
  existing factor: calib {merged, focal+extra (principal point fixed), fixed} x pose {var, fixed} x point {var, fixed}
  for SIMPLE_RADIAL, PINHOLE and OPENCV_FISHEYE, fixed calib for FULL_OPENCV. The adapter
  (`bundle_adjustment_caspar.{h,cc}`, `caspar_model_adapter.h`) uses a rig factor for a non-reference image when, as in
  the Ceres BA, `refine_sensor_from_rig` is set, the sensor is not constant in the config and the rig's reference
  sensor has an image in the problem; one SensorFromRig node per (rig, sensor) and model, written back to the `Rig`.
  The reference sensor stays identity (gauge); external observations keep a constant sensor_from_rig (as Ceres).
  `LOG(FATAL_THROW)` remains for principal-point-only refinement with a rig and for FULL_OPENCV with refined
  intrinsics (no factor generated). The rig setters are one macro line per calibration state
  (`CASPAR_RIG_ADAPTER_MEMBERS`). `bundle_adjustment_caspar_test.cc`: the old "throws on variable sensor_from_rig"
  test becomes a rig-vs-Ceres test plus a principal-point-only throw test.
- **Shared memory.** A factor kernel's static shared memory is its largest accessor buffer (`inout_shared`, 16 KB at
  1024 threads) plus one 8 KB SharedIndex array per tunable node plus, per `AddSum` output, a `__shared__` sum buffer.
  The six factors with four tunable nodes (pose, sensor_from_rig, calib, point) needed 49,156 B in `score` and
  `res_jac_first`: 4 B over the 48 KB static limit. A smaller block is not an option (SharedIndices groups every
  factor's indices in 1024-thread blocks). For those six factors only, the generator's `AddSumDirect` adds the block's
  residual sum to global memory itself, with the same reduction and the same one atomicAdd per block, without the 4 B
  buffer: every kernel now fits (24 at exactly 49,152 B). All other kernels regenerate byte for byte as before.
- **Generated** (`caspar-rig-refine-generated-f32.patch`): 424 new kernel files and the 4 aggregates. `solver.{h,cc}`
  are regenerated and then carry the reduced-camera-system edits again: its seven `solver.cc` hunks and one `solver.h`
  hunk applied by context (`patch -p1`), the one whose context moved (`DoUpdateR` ends with
  `SimpleRadialSensorFromRig` now) by hand, and the span helper block re-rendered with `tools/caspar_rcs_helpers.py`
  (the camera span now ends at `SimpleRadialSensorFromRig`). The beta fix lives in `solver_tools`, which did not change.

```bash
# regenerate (generator env as above), outside the checkout:
python src/thirdparty/Symforce-Caspar/caspar_generate.py /tmp/caspar-gen-rig/rig f32
# take the new files + caspar_mappings.{h,cu}; take solver.{h,cc} and re-apply the RCS solver hunks, then
python <recipe>/tools/caspar_rcs_helpers.py <checkout>   # exit 0 once the printed block is in solver.cc
```

- **Measurements** (2026-11-24, RTX 5090 dev build sm_120, f32; CASPAR under /tmp/gpu-5090.lock, Ceres 2.x on 8 CPU
  threads of a loaded machine, TRIVIAL loss = CASPAR's L2, same start, poses compared after a Sim(3) fit of the frame
  centres since each solver picks its own gauge points):
  - Mission office H10b, 2-lens OPENCV_FISHEYE, obs <= 80 deg (606 images, 200k points, rig + poses + points + all
    intrinsics): Ceres 62.95 s, cost 1.085148e6; CASPAR RCS (diag_exit_value 1e12) 0.87 s, 1.085161e6, rig 0.00014 deg /
    2.8e-6 from Ceres (baseline 0.03048 both), poses 0.0005 deg max; full PCG 0.77 s, rig 0.006 deg. The full problem
    (obs up to 90 deg) still stalls on CASPAR (cost 1.4378e6 vs start 1.4380e6, rig unchanged; Ceres 39.2 s, 1.3509e6):
    the hemisphere-edge / no-robust-loss limit, now reported as NO_CONVERGENCE.
  - EuRoC V1_01 stereo (FULL_OPENCV, intrinsics fixed, 400 images, 71k points), sensor_from_rig perturbed 1 deg / 2 cm:
    Ceres 14.2 s; CASPAR RCS 0.65 s, rig 0.00001 deg from Ceres, both back to 0.025 deg / baseline 0.1109 m vs factory
    0.1101 m. TWO_CAMS_FROM_WORLD gauge: FULL_OPENCV has no fixed-point factor (pre-existing), so THREE_POINTS throws.
  - Synthetic 2-camera rigs (12 frames, 400 points, rig perturbed 2 deg / 3 cm): PINHOLE {none, focal+extra, all},
    OPENCV_FISHEYE {none, focal+extra}: CASPAR within 0.0065 deg (rig) and 0.007 deg (poses) of Ceres, cost within 1e-5.
    SIMPLE_RADIAL focal+extra: 0.0001 deg. OPENCV_FISHEYE with every intrinsic refined is weak (a principal-point shift
    trades against a small rotation, ~1 px per 0.15 deg): CASPAR RCS reaches Ceres' cost (2.511456e3) with the rig
    0.002-0.03 deg off across runs (f32 atomics are not deterministic); the full PCG stops at 2.51160e3 (+6e-5) with
    0.15-0.21 deg. Tests: caspar-rig/tests/ on the NAS (Mission office scratch), 10/10 passing.
  - CASPAR reports NO_CONVERGENCE at these minima: its f32 steps get rejected until the damping passes diag_exit_value.

## caspar-pcg-beta-recurrence.patch and caspar-reduced-camera-system.patch

Hand-written patches, cut like the others. One part depends on the generated layout: the span helpers in `solver.cc`
(`RcsReduceRhs`, `RcsMatvec`, `RcsBackSubstitute`, ...) name the node types before and after `Point`.
`tools/caspar_rcs_helpers.py` reads that layout from the allocation code, refuses one the span copies would get wrong
(an allocation between the blocks, a block size or alignment that differs from `r_k`'s), and checks that `solver.cc`
holds the helper block it renders. On a checkout of the pin with the three patches above committed as `base`:

```bash
git -C <checkout> apply <recipe>/patches/caspar-pcg-beta-recurrence.patch && git -C <checkout> commit -qam beta
git -C <checkout> apply <recipe>/patches/caspar-reduced-camera-system.patch      # then edit, if needed
python <recipe>/tools/caspar_rcs_helpers.py <checkout>   # after a solver regeneration: paste the block it prints
git -C <checkout> diff --full-index base beta > <recipe>/patches/caspar-pcg-beta-recurrence.patch
git -C <checkout> diff --full-index beta      > <recipe>/patches/caspar-reduced-camera-system.patch
```

- **PCG beta** (`caspar-pcg-beta-recurrence.patch`, 2 files). Symforce 0.12's `solver.cc.jinja` calls
  `BetaFromNumDenom(beta_numerator, alpha_numerator)`, and only `DoAlphaFirst` writes `alpha_numerator`, so every
  iteration after the first divides by `rho_0`: the PCG loses conjugacy from its third iteration on.
  `BetaFromNumDenom` now stores `rho_k` as the next denominator. colsfm `tests/test_caspar_solver.py` checks it: a
  full-system PCG capped at 10 iterations per step ends 2.0e-5 above Ceres' minimum with the fix, 8.4e-5 without.
- **Reduced camera system** (`caspar-reduced-camera-system.patch`, 9 files; `reduced_camera_system`, off by default
  in Caspar's `SolverParams` and in COLMAP's `CasparBundleAdjustmentOptions`; colsfm turns it on): the Point nodes are
  eliminated inside the PCG. The right-hand side becomes `g_c - H_cp H_pp^-1 g_p`; the matvec (chosen once per solve)
  is the generated `DoJtjpDirect` run twice (point part of `p` zeroed, then `p = (0, -H_pp^-1 H_pc p)` accumulating
  into `w`, which enters holding `M p`), with `PointNormalize` as the exact damped `H_pp^-1`; points are
  back-substituted after the PCG; the PCG stops on the Ceres/rootba Q criterion (`reduced_pcg_eta` 0.1, at most
  `reduced_pcg_iter_max` 500). The mode sets that policy even with no Point node, where only the elimination is
  skipped (a direct solver caller; COLMAP's adapter keeps constant points as Point nodes).
  `rho` and `alpha` must be finite and positive at every iteration: otherwise the PCG stops before the step update,
  with a zero camera step at the first iteration (a zero reduced right-hand side: no camera gradient or no camera
  block) and the last iterate after it. `reduced_pcg_iter_max >= 1`, `reduced_pcg_eta` in (0, 1) and no
  per-iteration score exit are checked in COLMAP's adapter (double) and again in `solve()` on the f32 values the
  solver uses (0.99999999 passes the first and rounds to 1.0f). The residual-norm readbacks only
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
is emitted with them and `tools/caspar_rcs_helpers.py` goes away. The beta fix is a one-line upstream report (draft, not filed).

