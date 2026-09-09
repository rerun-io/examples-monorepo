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
//! | `blas` | `GeneralMatrixVector.h`, and the three-coefficient reductions | the LDLT solves and the prior's cost |
//! | `qr` | `Householder.h`, `Jacobi.h`'s `makeGivens`, `Redux.h` | the landmark blocks' QR and the marginalization's flat QR |
//! | `ldlt` | `LDLT.h`, `TriangularSolverVector.h` | the LM step's solve |
//! | `svd` | `JacobiSVD.h`, `RealSvd2x2.h`, `Jacobi.h`'s `makeJacobi` | the DLT triangulation |
//!
//! Two members of the family live elsewhere on purpose: `LieScalar::eigen_maxi`
//! and `LieScalar::eigen_redux3` are per-scalar constants and reductions, so
//! they sit on the trait that carries the packet width
//! ([`crate::lie::LieScalar::EIGEN_PACKET_SIZE`]).

pub(crate) mod blas;
pub(crate) mod qr;
pub(crate) mod svd;

pub use blas::norm3;
