# Canonical GN Mojo Artifact

This document is the single artifact to keep if the GN migration work is collapsed down to one markdown file. It is written to stand on its own. A new contributor should be able to read only this file, open the named paths, run the named commands, and continue the work without needing any other handoff document. This document follows the repository rules in `.agents/PLANS.md`.

## Purpose / Big Picture

The goal is to keep exactly one Mojo GN implementation in `mast3r-slam`: the idiomatic one. The older “pragmatic” Mojo variant and the MAX exploration only matter as sources of lessons learned. The user-visible result after the final cleanup should be simple: `mast3r_slam` defaults to one Mojo GN backend, CUDA remains only as the oracle and fallback, and the real workload benchmarks on the checked-in `normal-apt` and `livingroom` rays fixtures remain accurate and fast.

Today, that goal is almost reached. The idiomatic public GN path is real code and is already within the required window on the real rays fixtures. The remaining gap is documentation and cleanup discipline, not a missing core implementation.

## Progress

- [x] (2026-04-09 17:31 CDT) The GN backend was split into dedicated Mojo files instead of being mixed into the matching kernel module.
- [x] (2026-04-09 23:59 CDT) Real-fixture capture and benchmark tooling was added at the live GN backend boundary.
- [x] (2026-04-10 00:15 CDT) The rays idiomatic public path was proven to be within the requested tolerance on real `normal-apt` and `livingroom` rays fixtures.
- [x] (2026-04-10 07:09 CDT) The idiomatic public GN path stopped being a pure alias and became explicit public-loop code in `gn.mojo` for rays, points, and calib.
- [x] (2026-04-10 07:12 CDT) The GN correctness suite passed after the public-loop rewrite: `13 passed`.
- [x] (2026-04-10 07:18 CDT) The idiomatic public rays path remained within 10% of current Mojo on the checked-in real rays fixtures.
- [x] (2026-04-10 07:18 CDT) The idiomatic public points and calib paths remained within 10% of current Mojo on the direct synthetic public benchmark.
- [ ] Delete or archive the older non-canonical Mojo surfaces after this document is accepted as the source of truth.
- [ ] Decide whether to keep the MAX custom-op prototype in-tree or move it to archival documentation only.
- [ ] Add a real calibrated fixture path only if an in-repo dataset/configuration actually exercises `solve_GN_calib` end to end.

## Surprises & Discoveries

- Observation: the real release gate had to move away from synthetic microbenchmarks.
  Evidence: the synthetic `rays_step` microbenchmark often made CUDA look better, while the real captured GN fixture and bounded example runs showed the current Mojo public path meeting or beating the practical target.

- Observation: the rays path is the only GN surface that the checked-in `normal-apt` and `livingroom` signoff runs exercise by default.
  Evidence: `packages/mast3r-slam/config/base.yaml` and `packages/mast3r-slam/config/fast.yaml` both set `use_calib: False`, so bounded example captures only emitted `rays-000.pt`.

- Observation: rerunning the example videos with `config/calib.yaml` did not produce live `calib` fixtures in the current repo state.
  Evidence: bounded capture directories for `verify-normal-apt-calib` and `verify-livingroom-calib` completed without fixture files being written.

- Observation: the public GN orchestration bugs mattered more than expected once the idiomatic path stopped aliasing.
  Evidence: the first real idiomatic calib loop failed until two issues were fixed in `gn.mojo`: calib uses `ii/jj` at argument slots `4/5`, and `gauss_newton_calib_step` must receive `Q_thresh`.

- Observation: the standalone idiomatic ray-step kernel never beat the current validated Mojo kernel on the real rays fixture.
  Evidence: earlier idiomatic kernel experiments remained slower than the current Mojo rays step even after several local optimizations.

- Observation: MAX custom ops are viable, but not the canonical answer here.
  Evidence: the MAX `pose_retr` experiments worked, but small-op overhead stayed high and the public GN signoff was achieved without requiring MAX.

## Decision Log

- Decision: the canonical implementation to keep is the idiomatic public GN path, not the earlier “pragmatic” Mojo export path.
  Rationale: the idiomatic public path now exists as real code, it preserves the verified performance envelope on real rays fixtures, and it is the version we want future contributors to reason about.
  Date/Author: 2026-04-10 / Codex

- Decision: CUDA remains in the repository as the correctness oracle and fallback even if the older Mojo path is later removed.
  Rationale: parity and regression checking still depend on a trusted external implementation.
  Date/Author: 2026-04-10 / Codex

- Decision: the signoff metric is based on real captured rays fixtures and bounded example runs, not synthetic microbenchmarks.
  Rationale: synthetic benches were useful diagnostically but did not reflect practical GN behavior closely enough.
  Date/Author: 2026-04-10 / Codex

- Decision: points and calib may remain CUDA-step-backed internally until there is a real need and a real fixture path to replace them.
  Rationale: the current live workload and validation corpus center on rays, and the idiomatic public surface is already stable and measured.
  Date/Author: 2026-04-10 / Codex

- Decision: this file should replace the older scatter of handoff notes when someone asks “what matters now?”
  Rationale: the repo had accumulated multiple intermediate plans and exploratory notes; future work should start from one canonical narrative.
  Date/Author: 2026-04-10 / Codex

## Outcomes & Retrospective

The important technical outcome is that the GN Mojo work is no longer an open-ended migration experiment. There is a clear canonical path: idiomatic public GN orchestration in Mojo, real rays fixture signoff, CUDA as oracle, and a small, explicit remaining validation gap for calibrated fixtures. The most valuable lesson from the first Mojo version is not its code shape. It is the benchmark and profiling knowledge it produced: real-fixture measurement matters more than synthetic numbers, and a clean public orchestration layer matters more than preserving every first-pass kernel variant.

The one thing not yet proven by a real checked-in example run is live calibrated fixture signoff on `normal-apt` and `livingroom`. That is a dataset/configuration problem, not a known correctness or performance regression in the idiomatic public GN path.

## Context and Orientation

The GN backend in this repository means the Gauss-Newton pose optimization code used by `mast3r-slam`. In practice, it is the code that computes per-edge Hessian and gradient blocks, solves for pose updates, and retracts those updates back into the pose tensors.

The canonical repository paths are these:

- `packages/mast3r-slam/mast3r_slam/backend/mojo/gn.mojo`
  This is the main file to read first. It contains the public GN orchestration in Mojo, the public idiomatic wrappers, the solve logic, and the pose-retraction dispatch.

- `packages/mast3r-slam/mast3r_slam/backend/mojo/gn_kernels.mojo`
  This contains the validated low-level Mojo GN kernels. This is still the production-performance baseline for the rays step.

- `packages/mast3r-slam/mast3r_slam/backend/mojo/mast3r_slam_mojo_backends.mojo`
  This is the shared-library export surface for Python. If a Mojo GN function is meant to be callable from Python, it must be exported here.

- `packages/mast3r-slam/mast3r_slam/gn_backends.py`
  This is the Python backend selector. It decides whether a public GN call goes to Mojo or CUDA, and it honors the environment switches used by tests and benchmarks.

- `packages/mast3r-slam/mast3r_slam/global_opt.py`
  This is the live caller. It captures GN fixtures, calls the backend entrypoints, and reflects what the application actually uses.

- `packages/mast3r-slam/tools/bench_gn_real_fixtures.py`
  This is the primary benchmark tool that matters. It compares CUDA, current Mojo, and idiomatic Mojo on captured `.pt` fixtures.

- `packages/mast3r-slam/tests/test_gn_step_api.py`
  This is the main GN correctness smoke suite. It proves the public and step surfaces still work after edits.

- `packages/mast3r-slam/mast3r_slam/backend/src/gn_kernels.cu`
  This is the CUDA oracle. It matters for parity and fallback, not as the future implementation target.

The non-canonical files are still worth knowing about so a reader can recognize them if they appear in blame or old commits:

- `packages/mast3r-slam/mast3r_slam/backend/mojo/gn_kernels_idiomatic.mojo`
  This is an experimental standalone idiomatic ray-step kernel. It did not become the winning implementation.

- `packages/mast3r-slam/mast3r_slam/backend/max_ops/gn_ops/__init__.mojo`
  This is the MAX custom-op exploration. It is educational, not required for the canonical path.

- `packages/mast3r-slam/mast3r_slam/mojo_gn.py`
  This is an earlier Python-side GN orchestration experiment and should not be considered the long-term architecture.

## Canonical External References

These are the only external references that should be linked from future GN work unless a newer verified reference is clearly better.

- Layouts overview:
  https://docs.modular.com/mojo/manual/layout/layouts/

- LayoutTensor guide:
  https://docs.modular.com/mojo/manual/layout/tensors/

- GPU block and warp operations:
  https://docs.modular.com/mojo/manual/gpu/block-and-warp/

- MAX PyTorch custom ops:
  https://docs.modular.com/max/develop/custom-kernels-pytorch/

- GPU threading vs SIMD puzzle:
  https://puzzles.modular.com/puzzle_23/gpu-thread-vs-simd.html

- Puzzle 23 overview:
  https://puzzles.modular.com/puzzle_23/puzzle_23.html

- Structured Mojo Kernels Part 1:
  https://www.modular.com/blog/structured-mojo-kernels-part-1-peak-performance-half-the-code

- Structured Mojo Kernels Part 2:
  https://www.modular.com/blog/structured-mojo-kernels-part-2-the-three-pillars

- Structured Mojo Kernels Part 3:
  https://www.modular.com/blog/structured-mojo-kernels-part-3-composition-in-practice

- Structured Mojo Kernels Part 4:
  https://www.modular.com/blog/structured-mojo-kernels-part-4-portability-and-the-road-ahead

These links matter because they capture the specific ideas that shaped the final GN direction: `LayoutTensor` as the right abstraction once storage exists, block/warp reductions instead of manual synchronization trees where possible, and a structured separation between low-level kernels and higher-level orchestration.

## Plan of Work

If future cleanup happens, the order should be conservative.

First, keep the public idiomatic GN path exactly as it is and do not touch the live rays signoff. That means preserving the public idiomatic wrappers in `gn.mojo`, the exports in `mast3r_slam_mojo_backends.mojo`, and the routing in `gn_backends.py`.

Second, archive or delete the older experimental implementation surfaces only after the benchmarks and tests are rerun. The first candidates are the exploratory notes and code paths that are no longer part of the canonical execution story, especially the standalone experimental idiomatic ray kernel and the Python-side GN orchestration prototype.

Third, if someone wants to port the points or calib step surfaces fully into idiomatic Mojo, they should do it only after establishing a real reproducible fixture path that actually exercises those surfaces. Without that, the work is too easy to overfit to synthetic cases.

Fourth, if there is a later cleanup to make the idiomatic path the only Mojo path, the code should be reorganized around clear names rather than “current” versus “idiomatic” dual naming. Once the older path is retired, the idiomatic one should simply become `gauss_newton_*_impl`.

## Concrete Steps

All commands below are the ones that should still work after cleanup if the canonical implementation remains correct.

Build the shared library from the repository root:

    pixi run -e mast3r-slam-dev _build-mojo-kernels

Run the GN smoke suite:

    pixi run -e mast3r-slam-dev pytest \
      packages/mast3r-slam/tests/test_gn_fixture_utils.py \
      packages/mast3r-slam/tests/test_gn_step_api.py -q

Benchmark the checked-in real rays fixtures from the package directory:

    cd packages/mast3r-slam
    pixi run -e mast3r-slam-dev python tools/bench_gn_real_fixtures.py \
      artifacts/gn-fixtures/verify-normal-apt/rays-000.pt \
      --warmup 5 --runs 20

    cd packages/mast3r-slam
    pixi run -e mast3r-slam-dev python tools/bench_gn_real_fixtures.py \
      artifacts/gn-fixtures/verify-livingroom/rays-000.pt \
      --warmup 5 --runs 20

If you need a quick synthetic sanity check for points and calib public surfaces:

    cd packages/mast3r-slam
    pixi run -e mast3r-slam-dev python - <<'PY'
    import torch
    import mast3r_slam_mojo_backends as m
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(0)
    Twc = torch.randn(4, 8, device=device, generator=gen, dtype=torch.float32).contiguous()
    Twc[:, 6] = 1.0
    Twc[:, 7] = 1.0
    Xs = torch.randn(4, 64, 3, device=device, generator=gen, dtype=torch.float32).contiguous()
    Cs = (torch.rand(4, 64, 1, device=device, generator=gen, dtype=torch.float32) + 0.5).contiguous()
    ii = torch.tensor([0, 1, 2], device=device, dtype=torch.long)
    jj = torch.tensor([1, 2, 3], device=device, dtype=torch.long)
    idx = torch.randint(0, 64, (3, 64), device=device, generator=gen, dtype=torch.long).contiguous()
    valid = (torch.rand(3, 64, 1, device=device, generator=gen) > 0.2).contiguous()
    Q = (torch.rand(3, 64, 1, device=device, generator=gen, dtype=torch.float32) * 2.0 + 1.0).contiguous()
    K = torch.tensor([[500.0, 0.0, 32.0], [0.0, 500.0, 32.0], [0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    def bench(fn, warmup=10, runs=50):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        vals = []
        for _ in range(runs):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            vals.append(s.elapsed_time(e))
        t = torch.tensor(vals, dtype=torch.float64)
        return float(t.median())
    pts_args = (Twc.clone(), Xs, Cs, ii, jj, idx, valid, Q, 0.05, 0.0, 1.5, 3, 1e-8)
    cal_args = (Twc.clone(), Xs, Cs, K, ii, jj, idx, valid, Q, 64, 64, -10, 1e-6, 1.0, 10.0, 0.0, 1.5, 3, 1e-8)
    print(bench(lambda: m.gauss_newton_points_impl(pts_args)))
    print(bench(lambda: m.gauss_newton_points_impl_idiomatic(pts_args)))
    print(bench(lambda: m.gauss_newton_calib_impl(cal_args)))
    print(bench(lambda: m.gauss_newton_calib_impl_idiomatic(cal_args)))
    PY

## Validation and Acceptance

This canonical artifact is still valid only if the following remain true.

The shared library builds successfully.

The GN tests pass.

The idiomatic public rays path stays within 10% of the current Mojo public rays path on both checked-in real rays fixtures:

- `artifacts/gn-fixtures/verify-normal-apt/rays-000.pt`
- `artifacts/gn-fixtures/verify-livingroom/rays-000.pt`

The accepted measured public rays numbers at the time this artifact was written were:

- normal-apt:
  current `43.789 ms`, idiomatic `43.931 ms`, ratio `1.003x`

- livingroom:
  current `43.798 ms`, idiomatic `43.713 ms`, ratio `0.998x`

The accepted measured direct public synthetic numbers for the non-live surfaces were:

- points:
  current `0.269 ms`, idiomatic `0.268 ms`, ratio `0.998x`

- calib:
  current `0.264 ms`, idiomatic `0.267 ms`, ratio `1.014x`

Accuracy remains acceptable only if the public rays fixture benchmark still reports the tiny drift envelope already observed:

- translation max abs around `1e-7`
- quaternion max abs around `1e-8`
- scale max abs around `1e-8`
- `dx` max abs around `1e-7`

## Idempotence and Recovery

The build, test, and benchmark commands above are safe to rerun. The real rays fixture benchmarks are read-only. If a future cleanup breaks parity or throughput, the first recovery step is not to reintroduce multiple Mojo implementations blindly. Instead, restore the last known good `gn.mojo`, rerun the two real rays benchmarks, and compare against the accepted numbers recorded in this document.

If a future contributor wants to remove old files, they should do it in separate commits from functional changes. That keeps recovery simple: restore removed files only if the cleanup itself breaks discovery or validation.

## Artifacts and Notes

The most important measured outputs to preserve mentally are these.

From the real normal-apt rays fixture:

    mojo-current public   43.789 ms
    mojo-idiomatic public 43.931 ms
    ratio                 1.003x

From the real livingroom rays fixture:

    mojo-current public   43.798 ms
    mojo-idiomatic public 43.713 ms
    ratio                 0.998x

From the direct synthetic public comparison:

    points_public current=0.269 idiomatic=0.268 ratio=0.998
    calib_public  current=0.264 idiomatic=0.267 ratio=1.014

The final architectural position is straightforward:

- The idiomatic public GN path is the one that matters.
- The earlier Mojo variant is only useful as historical learning.
- CUDA stays as oracle and fallback.
- MAX is optional research, not required production architecture.

Revision note: this file was created to replace the scattered GN migration notes with one canonical markdown artifact that assumes the idiomatic Mojo GN path is the only version worth carrying forward.
