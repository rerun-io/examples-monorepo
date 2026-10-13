---
name: verify-block
description: >
  Prove that a block of a dagkit pipeline (Model, Op, Node, Graph) still does its job: a VERIFY card, a runnable
  check, hypothesis properties per layer, rrd goldens, a metric band against catalog GT (or a pre-port golden) and
  Rerun pixel evidence for the PR. Use when porting a block or package onto dagkit, adding a model (add-model's
  last step), changing a block's math, or rewiring a graph. Works from Claude Code and Codex.
---

# Verify a block

A block is done when its three tiers pass. Read this file before the first edit: step 3 comes BEFORE any code change.

## Rules that always bind

- **Reference first.** Commit the card and `verify.py`, record `pre` + `pre2` and the `pre` shots, then edit.
  A golden recorded after the first edit proves nothing. Short clips first; the full clip last.
- **Band, never byte identity.** `post` must stay inside a band set from measured run-to-run noise. Never widen
  a band or floor after you see `post`; a band change needs new `pre` runs and a reason in the PR.
- **Pixels, not logs. Fail closed.** A green exit code does not prove what the viewer shows. A missing golden, a
  missing or non-finite metric, or an empty `noise` without `noise_from` fails (exit 1).
- **RRD first.** Goldens, metrics and shots go into `.rrd` files; `VERIFY.toml` holds only bands, segment ids,
  metric names, shot times and test ids.
- **Shared catalog.** Read freely; never remove, replace or re-register a layer; never restart a catalog server
  or kill a process by name (`pkill rerun` also kills the shared catalog server).
- **Lanes.** Unit: no GPU, no catalog. `verify` runs in the prod env (`-e <pkg>`, beartype off); the rest in `-dev`.
- **Public repo.** No hostnames, personal paths or session ids in the card, `verify.py`, commits or PR body.

## The three tiers (all must pass)

| tier | checks | lane | run |
|---|---|---|---|
| Model | parity vs the upstream reference (owned copy: fp32 bit-identical; other kernels: tolerance), torch ↔ TensorRT parity, owned outputs (a 2nd call leaves the 1st result unchanged) | golden (stored upstream reference data), else integration | `tests-golden`, `tests-integration` |
| Op / Node | hypothesis properties of Types, Ops, Nodes and runners (section 4) | unit | `tests` |
| Graph | metric band vs GT on a catalog segment (or vs the pre-port golden) + Rerun pixels vs the `pre` shots | prod env | `verify`, `verify-full` |

## When to use: which tiers a change needs

| change | Model tier | Op / Node tier (hypothesis) | Graph tier (band + pixels) |
|---|---|---|---|
| port a package or block onto dagkit | owned outputs if a wrapper is new | Types, Ops, Nodes, runner equivalence | pre-port golden or GT band |
| add a model (add-model, last step) | parity vs upstream, torch ↔ TensorRT | Types + the wrapper node | GT band; without GT, the `2-predictor` golden |
| change an Op's or a Model's math | parity if a Model | properties of that Op | band of every graph that uses it |
| rewire a graph (ports, runner, placement) | - | `validate()`, sequential == threaded | band + pixels |
| change a Type or its Rerun mapping | - | serde, `to_rerun` structure, `rerun_roundtrip` | pixels |

## The loop

1. Choose the reference (section 1) and write the card (section 2) plus `verify.py` (section 3).
2. Commit them on the stage branch BEFORE you edit the block. That commit's sha7 goes into the `pre` Producer.
3. Run `verify-pre` (`pre` + `pre2` per short case, `pre` per full case); take the `pre` shots (section 5).
4. Write each `Case.noise` from |pre − pre2|, then commit.
5. Write the property tests (section 4). Each must fail once on a broken invariant, then pass.
6. Port or change the block. Keep the unit lane green at each step (`tdd` skill).
7. Run `pixi run -e <pkg>-dev --frozen gate` (lint, typecheck, deadcode, tests, tests-integration), then
   `tests-golden` (gate skips golden; Model reference checks live there).
8. Run `pixi run -e <pkg> --frozen verify` (short). The bands must pass. Take the `post` shots and compare them
   with the `pre` shots.
9. Run `verify-full`, then write the PR body (section 7).

## 1. Reference: a GT segment, or the pre-port golden

Use catalog GT when it exists; registering cheap public GT beats any golden. Only the datasets below; any other
dataset needs the owner's approval. Check `CatalogClient(url).dataset_names()` before you rely on a row.

| GT kind | datasets with that GT on the catalog | blocks it verifies |
|---|---|---|
| camera poses / SLAM | msd-g2, msd-index, msd-odyssey, dataforge-lamaria, dataforge-aria_gen2_pilot, dataforge-hot3d-aria, dataforge-hot3d-quest3 | SLAM / VIO, trackers, robocap SLAM |
| rig calibration | dataforge-{hocap, assembly101, egoexo4d, epfl, mamma-eval, mamma-markerless-*, mamma-syn, mamma-syn-bedlam} | exo-calib, mamma, monoprior calibration |
| 3D body (SMPL-X / markers) | dataforge-{egoexo4d, epfl, mamma-eval, mamma-markerless-*, mamma-syn, mamma-syn-bedlam} | mamma, posekit, sam3d-body, lamp |
| 3D hands (MANO / UmeTrack) | dataforge-{aria_gen2_pilot, hot3d-aria, hot3d-quest3, umetrack, show3d, hocap, assembly101, epfl} | robocap-live, handtrack, hand fitting |
| masks | dataforge-{mamma-eval, mamma-syn, mamma-syn-bedlam} | SAM2 / SAM3, mamma tracker |
| 2D keypoints | dataforge-{aria_gen2_pilot, hot3d-aria, hot3d-quest3, umetrack, egoexo4d, epfl, mamma-*} | posekit, keypoint nets, exo-calib kp2d |
| depth | none registered: use the pre-port golden | monoprior, prompt-da, mvs, vistadream |
| none | robocap (its `slam_rs` layer is our own output, not GT) | use the pre-port golden |

Choose the segment by a rule (segment kind), never by result: the first sorted segment id that matches. Write
the rule and the id into the card. Query with the `rerun-catalog-queries` skill (one round trip per call):

```python
from datafusion import col
from rerun.catalog import CatalogClient

ds = CatalogClient("rerun+http://127.0.0.1:51235").get_dataset("dataforge-assembly101")
ids = sorted(ds.segment_table().select(col("rerun_segment_id")).to_arrow_table().column(0).to_pylist())
def gt_cams(seg: str) -> set[str]:   # cameras with static GT pose AND intrinsics in THIS segment (static rows only)
    row = ds.filter_segments(seg).filter_contents(["/world/**"]).reader(index=None).to_arrow_table().to_pylist()[0]
    pose = {k.split(":")[0] for k, v in row.items() if k.endswith(":Transform3D:translation") and v is not None}
    return pose & {k.split(":")[0].removesuffix("/pinhole") for k, v in row.items() if k.endswith(":Pinhole:image_from_camera") and v is not None}
seg = next(s for s in ids if len(gt_cams(s)) >= 4)    # rule: >= 4 cameras with static GT pose + intrinsics
```
`schema()` is the DATASET schema even after `filter_segments`: use it to find column names, never to prove that a
segment has GT. Read the rows of the segment, as above.

No GT: the pre-port output is the reference (`reference = "pre-port"`). An already broken pre-port pipeline is the
baseline: smallest fix in an uncommitted scratch copy, Producer version `<sha7>+<fix>`, stated in the PR.

## 2. The card: `packages/<pkg>/VERIFY.toml`

Owned TOML, read by `dagkit.verify` with pyserde (`deny_unknown_fields`, `type_check=coerce`). An unknown key is an error.

| table | fields |
|---|---|
| top level `VerifyCard` | `package`, `[[graph]]`, `[[tests]]` |
| `[[graph]]` `GraphBlock` | `name`, `run` (`"pkg.verify:run"`), `score` (`"pkg.verify:score"`), `cases`, `metrics`, `shots` |
| `[[graph.cases]]` `Case` | `name`, `clip` (`short`/`full`), `reference` (`gt`/`pre-port`), `dataset` + `segment` OR `input` (repo-relative file), `window_s` (`[start, stop]`, absolute seconds on the case clock), `params` (Config overrides, string values), `noise` (metric → measured \|pre − pre2\|), `noise_from` (short case whose noise a full case reuses) |
| `[[graph.metrics]]` `MetricBand` | `name`, `unit`, `better` (`lower`/`higher`), `floor` (ABSOLUTE smallest tolerance; write "5 % of pre" as the number) |
| `[[graph.shots]]` `Shot` | `view` (a view's `origin` from `rerun_get_viewer_state`, e.g. `/`, or `full`), `t_s` (absolute seconds on the case clock) |
| `[[tests]]` `TestBlock` | `name`, `tier` (`model`/`op`/`node`/`types`), `lane` (`unit`/`integration`/`golden`), `tests` (test ids), `properties` (ids of the property tests that own the bounds) |

```toml
package = "exo-calib"

[[graph]]
name = "calibrate"
run = "exo_calib.verify:run"
score = "exo_calib.verify:score"
metrics = [{ name = "rot_deg_mean", unit = "deg", better = "lower", floor = 0.1 },
           { name = "trans_cm_mean", unit = "cm", better = "lower", floor = 1.0 }]
shots = [{ view = "/", t_s = 0.0 }, { view = "/", t_s = 30.0 }]

[[graph.cases]]
name = "short"                     # rule: first sorted segment with a static exo rig (>= 4 cams) and GT poses
clip = "short"
reference = "gt"
dataset = "dataforge-assembly101"
segment = "<segment id from section 1>"
params = { frame_budget = "200" }
noise = { rot_deg_mean = 0.02, trans_cm_mean = 0.3 }

[[graph.cases]]
name = "full"
clip = "full"
reference = "gt"
dataset = "dataforge-assembly101"
segment = "<same segment id>"
params = { frame_budget = "1000" }
noise_from = "short"               # run once pre-port; reuses the short case's noise, stated in the PR

[[tests]]
name = "triangulation"
tier = "op"
lane = "unit"
tests = ["tests/test_ops_triangulate.py"]
properties = ["tests/test_ops_triangulate.py::test_recovers_projected_points"]
```

Band: `tolerance = max(2 × noise[metric], floor)`. `lower` passes when `post ≤ pre + tolerance`, `higher` when
`post ≥ pre − tolerance`; noise 0 (deterministic) gives the floor. Keep GPU nondeterminism as in production
(no extra seeding). Typical floors: ATE 5 % of pre (as a number), MPJPE / PVE 2 mm, rotation 0.1°, translation
1 cm, focal 0.5 %, 2D keypoints 0.5 px, PSNR 0.3 dB, mask IoU 0.01. A failed band: find the cause (Type
mapping, dtype, frame offset, realtime drops; verify runs lossless). Never "fix" it with byte identity.

## 3. The check: `<module>/verify.py` (App layer) and the goldens

```python
def run(case: Case, phase: Literal["pre", "pre2", "post"], recording: rr.RecordingStream) -> None: ...
def score(case: Case, golden: Path) -> dict[str, float]: ...
```

- `run` writes the graph's outputs to `recording` at their normal entity paths, on the case clock (catalog
  cases: the segment's `video_time`). At `pre`/`pre2` it drives the OLD pipeline and logs that pipeline's
  outputs at the paths the ported graph will use. The port later changes only the body of `run`.
- `score` reads only the golden rrd, plus GT from the catalog. It never runs the pipeline. At `post` the CLI
  re-scores the `pre`, `pre2` and `post` rrds with the same `score`, so a scoring change cannot hide a regression.
- Goldens: `$DAGKIT_GOLDENS/<package>/<block>/<case>.<phase>.rrd` (a durable local directory, never committed);
  `recording_id` = segment (local input: case name); static `/producers/verify-<block>` and `/verify/<block>/metrics`.
- Pixi tasks in the package's prod feature (`[feature.<pkg>.tasks]`, single-line commands):

```toml
verify-pre = { cmd = "python -m dagkit.verify VERIFY.toml --phase pre && python -m dagkit.verify VERIFY.toml --phase pre2 --clip short", cwd = "packages/<pkg>" }
verify = { cmd = "python -m dagkit.verify VERIFY.toml --phase post --clip short", cwd = "packages/<pkg>" }
verify-full = { cmd = "python -m dagkit.verify VERIFY.toml --phase post --clip full", cwd = "packages/<pkg>" }
```

CLI: `python -m dagkit.verify VERIFY.toml --phase pre|pre2|post [--block B] [--clip short|full] [--register]`
prints a band table and exits 1 on a failed band. `--register` adds the golden as layer `verify-<block>-<phase>-<sha7>`
on the shared catalog: `post` only (pre and post share entity paths and would overlap), and only on request.

## 4. Property tests (hypothesis) per layer

| layer | checklist (each item is one test; its id goes into `[[tests]].properties`) |
|---|---|
| Types | pyserde round-trip equal (dtype and shape kept); unknown field refused; `to_rerun` structure: paths under the root, STATIC flags, component dtypes and shapes; `rerun_roundtrip` equal (integration lane: it starts an in-process server) |
| Op | geometric invariants: `triangulate(project(X)) ≈ X` (≥ 2 non-degenerate views); SE3 / Sim3 compose with inverse = identity; invariance to view order, weight scale or a Sim3 of the estimate; an injected outlier is dropped; degenerate or invalid input follows THAT Op's stated policy (shared DLT kernel: NaN + False; mamma's adapter: finite 0 + False) |
| Node | `dagkit.testing.check_node_contract` with fake Models: `reset` idempotent (reset twice, same outputs), outputs monotonic per port, `flush` drains (a second `flush` emits nothing), `emits_late` nodes advance the watermark |
| Graph / runner | `validate()` passes; `run_sequential == run_threaded(mode="lossless")` on a synthetic source (Block / AsOf edges, no mutable shared resource); DropOldest: received ⊆ produced, in order, drops counted |

- Arrays: `hypothesis.extra.numpy.arrays` with an explicit dtype and bounded finite floats; `assume(...)` drops
  degenerate geometry; tolerances live in the test (rtol 1e-6 f64, 1e-3 f32). `@settings(deadline=None)` and a
  bounded `max_examples` keep the unit lane in seconds. Break each invariant once locally and see it go red.

## 5. Pixel evidence (Rerun MCP, `rerun-viewer-validation` skill)

Output and GT must be in ONE recording. A local `.rrd` opened in the viewer becomes its own dataset and never merges
with a catalog segment: register both into your OWN viewer's catalog. Local-input cases open the golden directly.

1. Probe a free port (`ss -ltn`); `setsid <env>/lib/python3.12/site-packages/rerun_sdk/rerun_cli/rerun --headless
   --port <port>` from the package env; save the pid; `rerun_connect` to `http://127.0.0.1:<port>`.
2. One scratch dataset per block, case and phase in the viewer's catalog. Copy ONLY input + GT layers (e.g.
   `base`, `gt`); never a layer that writes the block's output paths (old outputs, `verify-*`), or it masks gaps:

```python
src = CatalogClient("rerun+http://127.0.0.1:51235").get_dataset(case.dataset)
row = src.segment_table().filter(col("rerun_segment_id") == case.segment).select(
    col("rerun_layer_names"), col("rerun_storage_urls")).to_arrow_table().to_pylist()[0]
keep = [(u, n) for u, n in zip(row["rerun_storage_urls"], row["rerun_layer_names"], strict=True) if n in {"base", "gt"}]  # input + GT only
scratch = CatalogClient(f"rerun+http://127.0.0.1:{port}").create_dataset(f"verify-{block}-{case.name}-{phase}")
scratch.register([u for u, _ in keep] + [golden.as_uri()], layer_name=[n for _, n in keep] + ["golden"]).wait()
url = scratch.segment_url(case.segment)    # → rerun_open_url; url = …/dataset/<id>?segment_id=<segment>
```

3. Poll `rerun_get_viewer_state` (≤ 60 s) until `active_store_id` == `Recording:<id from url>:<segment>` and its
   timeline has a range; else click the segment in Sources (`query_tree content_contains=<segment>`). Pass
   that store id to every `rerun_set_time_cursor`.
4. Per `Shot`: `rerun_set_time_cursor` (`{"time": t_s × 1e9}`, timeline `video_time`, `play: false`) →
   `wait_for min_steps ≥ 3` → `rerun_save_screenshot` (`view_id` of the view with that origin; none for `full`) → read the PNG.
   Software-rendered video can take 20–40 s to show a frame; judge a frame blank only after that.
5. Same viewport (headless 1920×1080) and blueprint for both phases. Store the shots in `<case>.<phase>.shots.rrd`
   (same recording id): `rec.log(f"/verify/{block}/shots/{view}@{t_s}", rr.EncodedImage(path=png), static=True)`.
6. Compare `post` with `pre` per shot (same objects in the same place, no new error badges, nothing missing;
   one line per shot in the PR). Stop the viewer by process group (`kill -TERM -- -<pgid>`); check the port.

## 6. Artifacts: rrd first

| artifact | where |
|---|---|
| graph outputs, Producer, metrics per phase | `$DAGKIT_GOLDENS/<package>/<block>/<case>.<phase>.rrd` |
| reference and post screenshots | `<case>.<phase>.shots.rrd` (static `EncodedImage`) |
| bands, segment ids, metric names, shot times, test ids | `packages/<pkg>/VERIFY.toml` (committed) |
| band table, PNGs | PR body (pasted / attached), never committed; no JSON metric files: `score` reads the rrds |

## 7. PR body

```markdown
Gate: <package>-dev gate on <hardware class, e.g. x86_64-cuda> @ <sha7>
Verify: <package> verify + verify-full @ <sha7>: <n>/<n> bands pass (pre-port reference @ <sha7 of the card commit>)

| case | metric | pre | pre2 | post | band | result |
|---|---|---|---|---|---|---|
| short | rot_deg_mean | 0.412 | 0.431 | 0.420 | ≤ pre + 0.100 | pass |

| shot | pre | post |
|---|---|---|
| `/` @ 0.0 s | ![pre](<attachment>) | ![post](<attachment>) |

Tests: unit <n>, integration <n>, golden <n> passed (Types, Ops, Nodes, Graph build, Model parity, rerun_roundtrip).
Reference: <segment id> of <dataset> (rule: <segment kind>) | pre-port golden (broken pre-port: <yes/no + fix>).
```
Report a failing band as failing; never move a band in the PR that it caught.
