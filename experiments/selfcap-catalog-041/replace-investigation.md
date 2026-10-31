# REPLACE registration investigation

## Result

The apparent 0.41 stall is CPU-bound schema recomputation under the catalog-wide
write lock, not a deadlock, file-descriptor exhaustion, or a lost task error. The
same replacement scaling exists in 0.38.1. No calls were made to port 51235 during
this investigation. Both isolated test servers were stopped after testing.

In 0.41, `Dataset::add_source()` reads the current schema, replaces a source, then
invalidates the schema cache for `SourceInsertOutcome::Overwritten`. The next
source replacement reconstructs the schema from every registered source. Even
replacing a file with itself takes this path. For B replacements over M resident
sources of width W, the work is approximately O(B*M*W); a full replacement batch
therefore has quadratic growth with dataset size.

`register_with_dataset()` holds the server's global write guard through source
loading and registration. Other catalog reads wait for the entire batch, so the
viewer appears stuck and even listing other datasets times out.

Released source:
- [Schema reconstruction](https://github.com/rerun-io/rerun/blob/0.41.0/crates/store_app/re_server/src/store/dataset.rs#L283)
- [Overwrite invalidation](https://github.com/rerun-io/rerun/blob/0.41.0/crates/store_app/re_server/src/store/dataset.rs#L654)
- [Global write guard](https://github.com/rerun-io/rerun/blob/0.41.0/crates/store_app/re_server/src/rerun_cloud/mod.rs#L1225)

## Evidence

1. Sixteen combinations of server 0.38.1/0.41.0, client 0.38.1/0.41.0, and URI
   pattern (same, alternating files, atomic same-path content change, distinct
   files) each completed 200 replacements of three tiny segments. No fixed N
   triggered an indefinite stall. The basic same-URI server retained 13 FDs.
2. Add-only, skip, and unregister/add controls completed. Against 0.41, both
   clients raised explicit duplicate errors from `wait()`; the error listed all
   rejected segment/layers. Against 0.38, duplicate ERROR is rejected by the
   registration RPC. Neither case swallowed an error into an endless wait.
3. A scratch dataset on test port 51241, loaded only from the saved SelfCap
   inventory, contained 5,494 sources. Replacing its 690 base sources reproduced
   the apparent freeze without a viewer. GDB twice sampled a running Tokio worker
   in `Dataset::schema -> schema_of_store -> arrow_fields -> to_arrow_field`.
4. Debug logs continued advancing about every 0.78 seconds. Strace showed an old
   file being closed and another progress log about every 0.77 seconds, rather
   than a blocking disk operation. Other workers slept normally. A catalog-list
   probe timed out after 3 seconds while the batch owned the write guard.
5. FDs decreased from 6,217 to 6,103 as old sources were replaced; the process limit
   was 500,000. RSS was about 3.1 GB. CPU time kept increasing (105 CPU seconds at
   the recorded 222-second process age). Raising the FD limit cannot solve this.
6. The registration call had not returned a handle: the client log stopped after
   `REPLACE BEGIN`, before `HANDLE RETURNED`. The expensive work precedes the task
   response. The large batch was stopped after collecting these signals, rather
   than waiting an estimated nine minutes for 690 replacements at that rate.

## Three-file reproduction

Run `repro_replace_scaling.py` on the server host with either SDK:

```bash
REPRO_URL=rerun+http://127.0.0.1:51241 python repro_replace_scaling.py
# Same control on a separate 0.38.1 server:
REPRO_URL=rerun+http://127.0.0.1:51238 python repro_replace_scaling.py
```

The script writes three small files, each with 64 scalar property fields, and
registers them under increasing numbers of layers. It refuses port 51235.
Registered files are retained and never overwritten by subsequent runs.

| Layers | Sources | 0.38 REPLACE (s) | 0.41 REPLACE (s) |
|---:|---:|---:|---:|
| 1 | 3 | 0.006 | 0.007 |
| 8 | 24 | 0.103 | 0.115 |
| 32 | 96 | 0.940 | 0.950 |
| 64 | 192 | 3.149 | 3.276 |

A separate final run of the committed reproduction measured 3.207 seconds for
64-layer REPLACE, versus 0.323 seconds for unregister plus insertion, including
both removal and reinsertion. Small request overhead masks the curve at N=1.

## Workarounds and limits

- **Skip unchanged registrations:** compare `(segment_id, layer_name, URI)` and
  register only changed references. This is the preferred immediate workaround
  and is already implemented by the snapshot restore tool.
- **Smaller update batches:** reduce how long unrelated catalog operations are
  blocked. This does not remove the total schema-recomputation work.
- **Unregister then register:** avoids repeated overwrite invalidation and was
  about 10 times faster in the synthetic test. It introduces an availability gap
  and needs recovery if insertion fails; do not apply it to a live shared corpus
  without an explicit maintenance plan.
- **Distinct URIs:** necessary for safe file publication, but do not eliminate
  this cost when the segment/layer identity is still overwritten.
- **Higher ulimit:** irrelevant to this reproduced stall; no FD leak was seen.

A server fix could reuse the dataset schema when the old/new source schemas are
identical, track distinct source schemas with reference counts, and shorten the
catalog-wide write-lock scope. These are proposals, not changes made here.

The prior blanket no-go assessment is narrowed: this is not evidence of a new
0.41 compatibility regression. A swap still needs a plan for replacement batches
and a fresh snapshot after the writer ends. No upgrade was performed.

## Upstream and artifacts

[Issue #12778](https://github.com/rerun-io/rerun/issues/12778) covers quadratic
registration. [PR #12774](https://github.com/rerun-io/rerun/pull/12774) discusses
cache refresh for insert/skip and explicitly leaves REPLACE slow. Its API status
was closed/unmerged; the relevant cache behavior is present in the 0.41 release.
A focused follow-up on the remaining overwrite path and global read starvation
is worthwhile. `replace-issue-draft.md` is ready; nothing was filed.

Local evidence is under `/tmp/rerun-replace-repro/`: `matrix.log`,
`schema-scale-038.log`, `schema-scale-041.log`, `final-repro.log`,
`task-controls041.log`, `gdb-scale.txt`, `gdb-scale-second.txt`,
`strace-scale.txt`, and `stall-status.txt`. The wider matrix driver is `repro.py`;
the private-file scale probe is `scale.py` and is not required for the public repro.
