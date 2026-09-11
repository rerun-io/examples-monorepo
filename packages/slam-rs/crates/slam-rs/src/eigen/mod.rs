//! The Eigen and Sophus ports, in one place.
//!
//! **One stance holds for every file here: this reproduces Eigen's operation
//! order on purpose.** Not its results — its order. A dot product summed in a
//! different association gives a different last bit, and in this port that bit
//! reaches decisions: the rank test `|beta| > sqrt(epsilon)` in the
//! marginalization QR (`marg_helper.cpp:301`), `inc.array().isFinite().all()`
//! and `step_norminf < 1e-4` in the LM loop (`sqrt_keypoint_vio.cpp:1424`,
//! `:1477`), and the `0 < inv_dist < 3` triangulation gate (`:534`). A landmark
//! either exists or does not; a keyframe is either evicted or not. So these are
//! not "numerics utilities" to be swapped for nalgebra's equivalents (D44), and
//! **nothing here is rewritten, only moved** — each file cites the Eigen source
//! it was ported from, line for line.
//!
//! | module | ported from | what needs it |
//! |---|---|---|
//! | `qr` | `Householder.h`, `Jacobi.h`'s `makeGivens`, `Redux.h` | the landmark blocks' QR and the marginalization's flat QR |
//! | `ldlt` | `LDLT.h`, `TriangularSolverVector.h` | the LM step's solve |
//!
//! Two of the four modules left in S33, once the last bit stopped being the
//! reference. `blas` — Eigen's `gemv` blocking, its vectorised `redux` tree and
//! `head<3>().norm()` — is gone: the LDLT's trailing updates are nalgebra
//! `gemv`/`gemv_tr` over views, the QR's column norm is `norm_squared`, the
//! prior's cost is a left fold over an iterator, and a three-coefficient norm is
//! `Vector3::norm`. What decides an answer's *shape* — the LDLT's panel
//! structure, the pivot on the un-updated diagonal, the QR's rank policy — is
//! untouched; only the association inside one panel or one sum is now the
//! library's.
//!
//! The 4x4 `JacobiSVD` port also left in S33: `crate::ba_base::triangulate` calls
//! [`nalgebra::linalg::SVD`] and promotes the solve to `f64`, so the DLT null
//! vector is now more accurate than the C++'s rather than identical to it. The
//! Givens half of `Eigen::JacobiRotation` moved to [`qr`] with its callers.
//!
//! Two members of the family live elsewhere on purpose: [`crate::lie::eigen_maxi`]
//! and [`crate::lie::LieScalar::eigen_redux3`] are per-scalar constants and
//! reductions, so they sit on the scalar trait. `eigen_redux3` has one
//! production caller left — the keyframe-eviction baseline in
//! `crate::estimator` — and outlives `blas` for that reason alone.

pub(crate) mod qr;
