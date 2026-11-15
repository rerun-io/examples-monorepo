# tests/colsfm

The test suite for the `colsfm` package. `colsfm/README.md` is the module map this
mirrors: one test module per production concern, plus the shared scene builders every
synthetic test draws on.

There are three ways to run it, and they select different things: the default run, the
type-checked run that adds beartype, and the `perf` run that adds the wall-clock bounds
the default run deliberately skips.

## The default run

```shell
pixi run colsfm-test
```

which is `python -m pytest tests/colsfm -q` inside the `colsfm` environment.

Several modules read shipped cuSFM runs under `data/`. Each skips with the missing path
named rather than failing, so a checkout without the data is still green — and a run that
reports more skips than usual is telling you which datasets are absent.

**The suite does not run the default configuration, on purpose.** Since 2026-09-06 a run
with no flags asks for RaCo, LightGlue+ and CASPAR (NOTES.md decision 17), which need the
gitignored graphs under `data/cusfm_models/` and a `CASPAR_ENABLED` pycolmap — neither of
which a fresh clone has. `test_pipeline.py` therefore builds every run from
`PYCOLMAP_ABLATION`, the `--features-backend pycolmap --matching-backend pycolmap
--ba-backend ceres --no-optimize-extrinsics` path, and states that once. What the defaults
*are* is pinned separately, as pure assertions on `PipelineOptions` and on `resolve_run`'s
`ResolvedRun`, which need no engine and no GPU. The engines themselves are exercised by
`test_raco_backend.py` and `test_tensorrt_backends.py` behind their own skips, and CASPAR by
`test_mapping_ba_backend.py`, which asserts *both* branches — the GPU solve where the build
has it, and the loud Ceres fallback with its reason in `MappingStats` where it does not.

## The type-checked run

`colsfm/__init__.py` turns on beartype's claw only when `PIXI_DEV_MODE=1`:

```python
if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype.claw import beartype_this_package

    beartype_this_package()
```

`colsfm-test` does not set it, so **the default run checks no annotation at run time**.
Every jaxtyping shape and dtype in the package goes unverified there, whatever the static
checker made of it. To get the checked run:

```shell
PIXI_DEV_MODE=1 pixi run -e colsfm python -m pytest tests/colsfm -q
```

The claw wraps every call in the package, so the checked run is the slower one: 160 s
here, against 95–126 s for the default run across three goes on the same machine. One
test opts out on purpose:
`test_a_thousand_node_graph_solves_within_one_second` carries a `skipif` on
`PIXI_DEV_MODE`, because the claw sits in the Ceres inner loop and a deadline measured
through it means nothing. Its numerical half,
`test_a_thousand_node_graph_closes_its_loop`, does run type-checked.

### What the checked run currently reports

**533 passed, 14 skipped** (2026-09-06), the same as the default run and 156 s against its
103 s. It was 5 failed / 522 passed until the two annotation defects it exists to find were
fixed: `colsfm.colmap_text_model.parse_colmap_points_text` declared `Int[ndarray,
"n_points 3"]` and returned `uint8` RGB, and a `publish_atomically` test expected a
`TypeError` beartype pre-empts with `BeartypeCallHintParamViolation`.

```shell
pixi run colsfm-test-typed
```

## The performance tests

Tests whose assertion is a wall-clock or throughput bound carry `@pytest.mark.perf`. They
measure the machine as much as the code, so an ordinary run collects them and skips them:

```shell
# just the timing bounds
pixi run -e colsfm python -m pytest tests/colsfm -q -m perf

# the whole suite, timing bounds included
COLSFM_PERF_TESTS=1 pixi run colsfm-test
```

`conftest.py` owns both halves of that: `pytest_configure` registers the marker and
`pytest_collection_modifyitems` skips it unless `-m perf` or `COLSFM_PERF_TESTS=1` asked
for it. A skip rather than a deselect, so the default run still reports that the
performance tests exist and were not run.

| `perf` test | The bound |
|---|---|
| `test_pose_graph.py::test_a_thousand_node_graph_solves_within_one_second` | 1000 nodes and 1049 edges solve in under 1.0 s. |
| `test_retrieval.py::test_brute_force_build_stays_inside_the_runtime_budget` | 226 images in 20 s, 900 in 60 s, on CPU NumPy. |
| `test_retrieval.py::test_the_vocabulary_backend_builds_and_queries_1000_images_inside_the_budget` | 1000 images build in 90 s, all 1000 queries in 30 s. |
| `test_mapping_galileo_parity.py::test_the_galileo_mapping_is_no_slower_than_ten_times_the_blob` | The mapping stage stays inside 10x the blob's own `runtime.csv` figure. |
| `test_raco_backend.py::test_the_fixed_engine_beats_the_dynamic_one_at_the_profile_maximum` | The fixed-shape TensorRT engine is the faster one at the profile maximum. |

Each of these sits next to a default-run test that asserts what the same work *computed*:
the pose graph closes its loop, the vocabulary tree fills every leaf, the Galileo map
matches the blob's. A busy machine can fail the bound without touching the numbers.

## How the modules are organised

### The mapping suite

`test_mapping.py` was one 1 987-line file covering five subjects that change for different
reasons. It is now five, and reading one of them no longer means reading all five:

| Module | What it owns |
|---|---|
| `mapping_helpers.py` | Not a test module. The synthetic rig every mapping scene is built from: known poses, known points, noisy observations, the COLMAP database that carries them, and the bookkeeping that matches a triangulated track back to the point that generated it. |
| `test_mapping.py` | What the mapper does to a scene. Correspondence loading, point recovery against ground truth, the outer loop's two stopping rules, pose recovery from a perturbed trajectory, single-thread determinism, the per-round callback. Synthetic input only. |
| `test_mapping_ba_backend.py` | Which bundle adjuster a run gets. The three CASPAR fallbacks (camera model, a bundle adjustment asked to free `sensor_from_rig` — i.e. `--no-regularised-extrinsics` — and a build without `CASPAR_ENABLED`), the capability probe and its cache, the solver-option cast, and the single Ceres bundle adjustment that finishes a CASPAR run. |
| `test_mapping_point_filters.py` | The two pre-solve guards, `filter_degenerate_points` and `filter_projection_failures`. Pure functions over a triangulated model; no solver, no backend, no dataset. |
| `test_mapping_extrinsics.py` | `--optimize_extrinsics`: what an unregularised rig refinement recovers from a 20 mm perturbation, and the drift it costs on weakly covisible cameras. Real Galileo matches, because a synthetic rig cannot settle the question. |
| `test_mapping_galileo_parity.py` | The integration run: the shipped cuSFM blob's own keyframes, matches and input poses, so the only difference from cuSFM's output is the mapper. The slowest mapping module, and the only one that needs a dataset on disk. |

`test_ba_backend.py` is a separate, older file and stays that way: it states the backend
policy as a pure contract, with no rig, no database and no GPU.
`test_mapping_ba_backend.py` is the same policy observed where it is applied.

### Shared code

| Module | What it holds |
|---|---|
| `conftest.py` | Fixtures and helpers used by more than one module: the Galileo data paths and their parsed metadata, `pose_delta` and `project_and_mask`, the blob `Keyframe` reader, the synthetic cuSFM-layout run builder, the isaac config, and the `perf` marker. |
| `loop_helpers.py` | The synthetic scene the loop-closure and loop-pose tests share. |
| `mapping_helpers.py` | The synthetic rig the mapping modules share, described above. |

`tests/colsfm` has no `__init__.py` and is therefore on `sys.path`, which is why these are
imported as `from conftest import ...` and `from mapping_helpers import ...`. Fixtures stay
in the test modules that use them rather than in the helper modules — building the mapping
rig costs 8 ms and its database 39 ms, so a per-module fixture is cheaper than a shared one
is worth.
