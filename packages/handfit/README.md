# handfit

Native warm and cold UmeTrack hand-pose fitting. The Rust core uses SymForce-generated
f64 landmark and lens Jacobians and a local Levenberg–Marquardt solver. The
Python runtime API needs numpy only. `handfit::scale` ports handtrack's joint hand-scale calibration (MEgATrack §3.6).

From the workspace root:

```bash
export CARGO_HOME="$PWD/.cargo-home" CARGO_NET_OFFLINE=true
pixi run -e handfit-codegen --frozen handfit-codegen
pixi run -e handfit-dev --frozen handfit-build
pixi run -e handfit-dev --frozen handfit-cargo-test
pixi run -e handfit-dev --frozen gate
pixi run -e handtrack-dev --frozen python -m pytest packages/handtrack/tests/test_native_fit.py -m golden
pixi run -e handtrack --frozen python packages/handtrack/tools/native_fit_bench.py --golden <fit_calls.pt> --report <report.md>   # torch vs native
pixi run -e handfit --frozen python packages/handfit/tools/golden_bench.py --golden <golden_fit.npz>             # numpy only (Mac, cap)
pixi run -e handfit --frozen python packages/handfit/tools/export_cold_golden.py --golden <golden_fit.npz> --output <golden_cold.txt>
cargo run --release --example cold_bench -- <golden_cold.txt> --stages --baseline <results.txt>   # cold fit, no Python (cross-builds)
```

`cold_bench` times `cold::initial_pose` on the golden set's 21 cold rows and splits it into the hypotheses, the palm-only
solves and the full solves, as `initial_pose` reports them in its diagnostics. `--write` stores every row's result and `--baseline` compares a run with a stored one (wrist,
landmarks, energies, converged flags, terminations, hypothesis choice); it fails when a converged flag or a termination
changes or a wrist moves more than 0.01 mm. Its views are built as `HandFitter.fit` builds them, so its results equal the
binding's.

`handtrack/tools/native_fit_bench.py` replays the recorded oracle through torch, native backends in one process. It writes accuracy
quantiles, cold acquisition decisions, stage selections, warm joint-angle outliers,
and full-replay timings to the requested Markdown report.
`--golden`, `--report`, and `--repeats` select the fixture, destination and
repetition count. The golden pytest reads `HANDFIT_GOLDEN` and skips with the
missing path when the fixture is absent.

The drop-in entry point is `handtrack.fit.native.fit_pose`, with the same
signature and `list[FitResult]` result as the torch implementation. It sends warm and cold hands through one GIL-released native call. `fit_pose_central_difference` is the diagnostic entry
point. Neither entry point changes the tracker default.

For direct numpy callers, `handfit.HandFitter` takes the numeric model arrays,
topology, phi and solver configuration once. Register calibrations with
`add_camera`, then pass a complete frame to `fit`. Dtypes, shapes, units and
configuration order are documented in `handfit/_core.pyi`. The native boundary
copies inputs before releasing the GIL, validates shapes and topology, and
returns owned numpy arrays. The last two angles are carried through for warm hands and zero for cold hands.
A view with no wrist weight uses the palm centre as its distance reference;
zero weights mask NaN observations. A cold hand with fewer than three observed points in each view returns
a neutral pose, NaN energies, and `no_evidence`. A warm hand without evidence
runs its temporal-prior-only solve. Pass `has_previous` to identify cold rows;
it defaults to all warm for existing numpy callers.

The model geometry must already be scaled to the subject. Phi rescales observed
distances; it does not scale the model again. Model topology must match the
codegen asset, including all four joint topology tables and all bone slots.
Numeric axes, pivots, landmark positions, weights and limits may vary.
The shim caches up to eight model/config handles. Treat models as immutable
while cached; clear `handtrack.fit.native._CACHE` after editing a model in place.

Cold fits align a neutral palm to each view, then try the 24 cube rotations.
Palm-only rigid LM selects distinct wrists for full neutral/open finger fits.
Exact energy ties break by index. Outputs include rigid/full iteration counts,
chosen wrist indices and the winning full start for diagnostics.

A palm-only (rigid) solve holds the fingers, so its landmarks are fixed hand-frame points moved by the wrist: it runs no
forward kinematics after the first evaluation, its Jacobian has the six rigid columns (`-R hat(p)` and the identity) and its
LM loop runs over six coordinates. Every solve builds the Jacobian only at accepted poses (a rejected step needs only its
energy), keeps it by residual row in buffers zeroed once per solve, adds `J^T J` per landmark over that landmark's own
columns, and solves the damped system by Cholesky (LU when a pivot is not positive). The arithmetic is reordered, not
changed: on the golden set's 624 hands the float32 outputs equal the previous ones (one cold joint angle moves by one float32
step), and so do every converged flag, termination and iteration count (2026-10-01). The cold fit takes about a quarter of
its previous time and the warm fit about half.

The solver keeps the reference's damping, limits, active-set tolerance, projected
prediction, rotation retraction and stationary test. Residuals and linear
algebra use f64; steps are cast to f32 before retraction, and Python outputs are
f32. Central differences use the reference step sizes with f64 residuals, so
this mode isolates Jacobian effects without claiming bit-exact torch replay.

Three repairs live in the generator for SymForce 0.12's Rust backend: parentheses
around power bases, bool types for relational temporaries, and a supported
constructor for the 63-vector. Regeneration applies them automatically and stops
with an error when SymForce's output no longer matches them; do not edit the
generated Rust files by hand. Random Jacobian tests cover both sides,
palm blending, the FK clamp, both lenses, all distortion terms, and negative z.
