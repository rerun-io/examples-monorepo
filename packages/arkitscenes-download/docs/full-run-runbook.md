# Full-corpus run: runbook + lessons learned

How the full ARKitScenes corpus (5,047 attempted → **5,015 published**) went from Apple's
CDN to a queryable Rerun catalog in July 2026 — every step, every dead end, and the
recipe to redo any part of it. Companion to [architecture.md](architecture.md), which
covers the per-sequence pipeline itself; this doc covers **running it at corpus scale**.

**Final state (2026-07-23):**

- **HF (canonical, public):** `pablovela5620/arkitscenes-rrd` — layer-first layout
  `<layer>/<video_id>.rrd` + `blueprints/`, ~1.9 TB.
- **S3:** `rerun-datasets-scratch-…-us-east-1-an/arkitscenes.2026.07.22/` and the
  curated bucket copy (us-west-2), byte-verified against HF.
- **Local working set (this deployment):** `/mnt/nas/datasets/arkitscenes/arkitscenes.2026.07.22/`
  (single canonical dir, 7 layers + blueprints, fixed gt).
- **Catalogs:** local OSS server on `:51235` (5,015 segments × 7 layers) and the
  internal cloud stack (`arkitscenes`, registered via `register_prefix` per layer).
- Missing 32 = 24 upstream-incomplete (no annotation) + 8 integrity-gate rejects.
- Modal spend: **~$265** total, ~5 h wall-clock — vs ~3 days projected for the
  local-only run.

---

## 1. The steps, in order (and what each one taught us)

### Step 0 — Local reference run first

Launched the then-existing local orchestrator (since deleted — the local flow is now
the three quickstart tools, and corpus scale is Modal-only). It shipped 298 sequences
before being retired, at ~37 MB/s off Apple's CDN → **~70 h projected**.

**Why it mattered anyway:** it became the reference implementation. The Modal
benchmark was validated against sequences the local run had already produced
(byte-profile-identical output proved the cloud path correct), and its per-sequence
integrity gates (PTS preservation, frame-count) carried over unchanged.

### Step 1 — Modal image: pixi inside the container

`arkitscenes_download/modal_jobs/__init__.py`. Instead of approximating the env with
pip, the image installs pixi, copies `pixi.toml` + `pixi.lock` + the two editable
packages (`arkitscenes-download`, `simplecv`), and runs
`pixi install --frozen -e arkitscenes-download`. Build-time asserts grep the env's
ffmpeg for `av1_nvenc` and `libsvtav1` so a broken image can't reach workers.

- **Byte-identical deps to local**, including conda-forge ffmpeg. NVENC works fine
  against Modal's injected L4 driver.
- Workers subprocess into the env's python via the same `tools/apps/download.py` /
  `ingest_sequence.py` shims `pipeline.py` uses — zero conversion-code fork.
- **Gotcha:** there is no pixi activation in the container — `_run_tool` must put the
  env's `bin/` on `PATH` and set `FFMPEG_PATH` explicitly, or the ingest can't find
  ffmpeg.
- Package source rides Modal's runtime mount, so **code fixes don't rebuild the image**
  (the ~10–15 min pixi-install layer stays cached).
- HF token ships as an **ephemeral per-run secret** read from `hf auth token` at launch
  (abc-130k pattern); nothing stored on Modal.

### Step 2 — Benchmark before committing money

`modal run -m arkitscenes_download.modal_jobs.convert_sequences::benchmark` — same 5
locally-validated sequences through both hardware legs, into `bench/` scratch prefixes.

| | L4 / `av1_nvenc` | 16-core / `libsvtav1` |
|---|---|---|
| Mean per sequence | **67.5 s** | 103 s |
| Cost per sequence | **$0.042** (~$214 corpus) | $0.073 (~$367) |
| Output | identical to local reference | smaller but forks encoder provenance |

**Verdict: GPU wins every axis.** CPU's only theoretical win is width (Modal caps this
plan at ~50 GPUs but CPU limits are counted in cores) — irrelevant here because upload,
not compute, was the critical path. Don't trust the in-worker nvenc smoke-probe (buggy,
reports False); read the `encoder` provenance property off the produced RRD instead.

### Step 3 — THE big lesson: never point a fleet at an HF repo

First full-run launch: 64 workers, each committing its own sequence via
`upload_folder`. **HF hard-throttles commits per repo** (~6/min observed; the limit is
deliberately undocumented — see §3). ~90% of billed GPU time went to 429 backoff
(~$45 burned). Dropping to 8 workers didn't help: every commit still 429'd.

**Root cause is structural, not tunable:** every repo upload is a git-like commit to
one serialized history. N parallel writers contend for one throttled resource.

### Step 4 — The fix: staging volume + single drain

Rearchitected in `convert_sequences.py`:

- **32 L4 converters** → Modal Volume `arkitscenes-rrd-staging` (atomic tmp+rename).
  They never talk to HF → run flat-out; GPU cost becomes fixed GPU-seconds.
- **One `drain_to_hf` function** (`max_containers=1`, 24 h timeout) loops
  `volume.reload()` + `HfApi().upload_large_folder` — batched commits, built-in 429
  handling, resume cache persisted on the volume.
- **Skip logic = destination-exists** (staged OR on HF). No done-files. Every relaunch
  is free; we restarted three times at zero cost.

Result: remaining ~4,800 sequences converted in ~3 h; drain kept pace at ~1,000
sequences/hour; conversion and upload finished together. Launch command (idempotent):

```bash
cd packages/arkitscenes-download
pixi run --frozen -e arkitscenes-download modal run --detach \
  -m arkitscenes_download.modal_jobs.convert_sequences::full_run \
  --encoder gpu --confirm
```

### Step 5 — Destination sync (HF → network storage)

`hf download` onto NFSv3-mounted storage, then atomic dir swap. Measured facts:

- Bottleneck is the **destination's spinning-disk write path under parallel scattered writes** (~44 MB/s at
  8 workers), not the internet (≥58 MB/s single-stream headroom measured concurrently).
- `--max-workers 3` made it **worse** (29 MB/s) — keep 8.
- **uid-squash trap:** hf's resume metadata
  (`.cache/huggingface/download/**/*.metadata`) lands mode-0000 through NFS
  uid-squash. Before ANY sync restart:
  `find <dir>/.cache/huggingface/download -name '*.metadata' -exec chmod 644 {} +`
  or the restart re-downloads everything. Same trap for `write_rrd` over NFS
  (`chmod 644 *.rrd` after writing).

### Step 6 — HF → S3 (scratch, then curated)

`arkitscenes_download/modal_jobs/transfer_to_s3.py`. Key design points:

- **Credentials:** no local AWS anywhere — Modal's OIDC identity
  (`MODAL_IDENTITY_TOKEN` → STS `assume_role_with_web_identity` on `modal-oidc-role`)
  is trusted by the scratch bucket policy, and that role can also *read* curated.
- **Discovery = ONE `list_repo_files` call** building work items; workers never touch
  the HF metadata API.
- **Second HF rate limit found:** the per-file **xet read-token endpoint** 429s under
  ~28 req/s (32 workers) — and the storm penalty-boxed the local token too. 16 workers
  + exponential backoff in-worker + `modal.Retries` = clean ~45 min run.
- scratch → curated is an **S3 Batch copy** (reality's `aws-batch-copy` skill, run by a
  human under `assume` — SSO tokens live in the macOS keychain, so never poll via
  non-interactive ssh+assume; poll from Modal boto3 instead). The
  `batch-reports/.../manifest.json` present at submission is a placeholder, not
  completion.

### Step 7 — Layer-first layout migration (the registration unlock)

Original layout was sequence-first (`rrd/<video_id>/<layer>.rrd`). Migrated everything
to **layer-first**: `<layer>/<video_id>.rrd` + `blueprints/` (S3 adds a dated dataset
prefix: `arkitscenes.<date>/<layer>/<video_id>.rrd`).

- Why: hub registration becomes **one `register_prefix` call per layer** (7 calls)
  instead of 35k per-file ops; selective per-layer downloads; layer evolution =
  directory ops.
- How (HF side): **30,090 server-side `CommitOperationCopy` ops** — zero bytes moved
  (xet copy-by-hash), 31 commits @1,000 ops pinned to a source revision, zero 429s —
  then ONE atomic cutover commit (delete old tree + new README), with a rollback SHA
  recorded first.
- Same content-addressing is why **"Duplicate this dataset" is instant and survives
  deletion of the original** — a duplicate holds its own references to the same chunks.

### Step 8 — The gt registration bug (and 132× fix)

Registering the catalog, 6 layers took 1.5–3 min each; **gt took >60 min** (killed) and
on the cloud stack **corrupted the dataset server-side** (every later registration of
any file failed schema deserialization; rebuilt fresh, ~1 min).

- **Root cause:** `ingest/gt.py` logged each box at `/world/gt/boxes/box-<uid>-<label>`
  → ~30k globally-unique entity paths corpus-wide. The OSS server re-merges the
  cumulative Arrow schema once per recording with linear-scan field lookups → ~cubic.
  Validated: same chunks with 44 shared paths = **136× faster**.
- **Fix:** slot paths (`box_00..box_NN`) + uid preserved as `rr.AnyValues(uid=…)` on
  the same entity. Converter patched (46 tests pass); all 5,015 gt files rewritten and
  propagated to all copies (local/HF/S3), byte-verified. Broken gt = 1 h+ + corruption; fixed gt =
  **11 s**.
- **Rule going forward: never mint unbounded unique entity paths.** Identity belongs in
  component values, not path names.
- Repro + issue text: `rerun-io/rerun-schema-width-register-repro`.
- Registration API lessons (all encoded in `ingest/catalog.py`):
  - `register()` is async; a dropped client gRPC call does NOT stop the server —
    poll the segment table for completion instead of resubmitting. Resubmits are
    near full price: the SKIP duplicate check runs *after* the per-file work
    (~300 ms/file at 5k entries).
  - Never bulk-REPLACE: each overwrite drops the server's schema cache → O(all-sources)
    recompute per file. Measured 12× slower than SKIP at just 400 files, growing with N.
  - Giant calls make the server refuse new connections while it grinds; expect the
    client call to drop and poll through it (catalog.py), or use layer-first
    `register_prefix`.
- **Verify layer completeness, never segment counts.** "5,015 segments" once hid
  4,500 base-only segments. The only real check: N segments AND N with all 7 layers.

---

## 2. The rules (what to remember above all)

1. **Repos are for publishing; object stores are for writing.** An HF dataset repo must
   only ever be written by a single batching process. Fleet output goes to a Modal
   volume, an HF bucket, or S3 first.
2. **Destination-exists is the only state you need.** Idempotent workers + free
   restarts fall out automatically.
3. **Decouple compute from upload.** Billed GPU-seconds must never wait on a network or
   a rate limit.
4. **Validate the cloud path against a local reference** before spending: same
   sequences, byte-profile comparison, provenance columns read from the output itself.
5. **Gate the expensive step behind an explicit flag** (`full_run` refuses without
   `--confirm`) and benchmark $/item first.
6. **Never mint unbounded unique entity paths** in RRDs — schema width is a
   registration-time cost multiplier and (on the cloud stack) a corruption vector.
7. **Layer-first layouts** for any multi-layer dataset destined for a catalog:
   `register_prefix` per layer beats per-file registration by orders of magnitude.

## 3. HF rate limits, as actually encountered

Three distinct surfaces, only one of them documented with numbers:

| Surface | Limit | Where it bit |
|---|---|---|
| Documented buckets (API / resolvers / pages, 5-min windows) | e.g. free user: 1,000 API / 5,000 resolves per 5 min | background noise only |
| **Repo commits** | *undocumented, changeable* ("granular user action" class); ~6/min observed | Step 3 — the $45 lesson |
| **Xet read-token endpoint** (per-file, on download) | 429 storm at ~28 req/s sustained | Step 6 — S3 transfer at 32 workers |

Mitigations that worked: single drain with `upload_large_folder` (uploads); ≤16 workers
+ in-worker exponential backoff (downloads); one-call discovery so workers never touch
the metadata API. `huggingface_hub` ≥1.2 parses the `RateLimit` headers and waits
precisely. A PRO account raises the documented buckets and is HF's stated answer for
granular-action quota errors.

## 4. Storage-destination cheat sheet

| | HF dataset repo | HF bucket | S3 |
|---|---|---|---|
| Write model | git commits — serialized + throttled | objects (Xet batch / `s3.hf.co` gateway) | objects |
| Parallel fleet writes | ❌ never | ✅ | ✅ |
| Storage cost | free (public) | ~$12–18/TB/mo | ~$23/TB/mo + egress |
| Public page / card / citability | ✅ | ❌ | ❌ |
| Registerable by OSS `rerun server` | ❌ (download first; `file://` only) | not yet (S3 gateway exists; server gate is `to_file_path()`) | internal stack: ✅ |

Facts worth keeping: repos and buckets share one Xet CAS (copies between them are
zero-transfer, by hash); dataset repos are **not** S3-reachable at the protocol level
(tested live); HF buckets **are** (`s3.hf.co`, path-style, `us-east-1`, HFAK keys,
`object_store` ≥ the 2026-07-10 redirect fix / PR #790).

## 5. Known small gotchas (alphabetical-ish)

- Layout migration on the same filesystem = **hardlinks** (`ln`), not copies: both
  layouts coexist at zero bytes while you verify, and cutover is deleting the old
  names. Before any `rm -rf` of a supposedly-redundant tree, prove it:
  `find <tree> -type f -links 1` must return nothing.
- OSS server `-d <dir>` assigns every file layer `base` — it cannot load a layered
  dataset. Register layers explicitly against a plain `rerun server`.
- `rerun` console entrypoints are Python wrappers that spawn the real binary —
  `Popen.terminate()` kills only the wrapper and leaks the server. Spawn with
  `start_new_session=True` and kill the process group.
- `spawn_map` takes **parallel iterators** (map-style), not tuple rows — tuple rows
  crash every worker instantly.
- Modal dashboards / `modal app list` task counts can be stale; verify completion from
  the data (volume/HF counts), not the task count.
- In-worker nvenc smoke-probe unreliable — trust the RRD `encoder` property.
- NFS uid-squash lands files mode-0000 (`hf download` metadata AND `write_rrd` output)
  — chmod before reuse.
- Spinning-disk destinations collapse under parallel scattered writes; fewer workers is not automatically
  better (3 was worse than 8 — measure, don't guess).
- Dual HF identity: personal token is the machine default (pipelines read
  `hf auth token`); work-account ops are scoped per-command via
  `HF_TOKEN=$HF_TOKEN_WORK` — never rc-export `HF_TOKEN` (it silently overrides
  `hf auth switch`).
- ~1.3% of sequences legitimately fail the transcode integrity gates (PTS/frame-count)
  — same sequences fail identically local and cloud; retry once for transients, then
  accept the deterministic rejects.

## 6. Next time, condensed

Re-ingest (schema v2, new codec) or a new corpus:

1. Validate per-sequence code locally on ~5 sequences; keep their outputs as reference.
2. Benchmark both hardware legs on those 5 via `::benchmark`; read $/seq off the table.
3. `full_run --encoder gpu --confirm` → converters into the staging volume, drain into
   the (public, layer-first) HF repo.
4. `hf download` → local storage (8 workers; over NFS, chmod the metadata cache before any restart).
5. `transfer_to_s3.py` if the internal stack needs it (16 workers, Modal OIDC creds).
6. Register: 7 × `register_prefix` (cloud), or OSS: `pixi run arkitscenes-download-serve`
   + `tools/apps/register_catalog.py --rrd-dir <layer-first root>` (resumable, self-verifying).

Adding an inference layer (e.g. PromptDA) is strictly easier: input is the existing
RRDs (no CDN, no NVENC requirement), output is one new `<layer>/` directory — but it
has **no upload throttle to hide compute behind**, so this is the workload where the
50-GPU cap can actually bind.
