# agent-traces

Convert Claude Code and Codex session transcripts into Rerun recordings: one `.rrd` per session.
You can then open a session in the Rerun viewer, or register many sessions on a Rerun catalog and query them together.

Recordings hold full transcripts and tool outputs, which may include secrets. Treat converted files as sensitive.

## What a recording holds

One fact/type table applies to every content entity; a declared fact gets a typed column when any row supplies it.
Absent cells are null; `agent_id` is always present and is an empty string for the main agent.
Read the catalog with `event` to retain repeated `wall` timestamps; use `wall` for time windows, UTC day grouping, and the viewer timeline.

| Entity (family) | Typed columns | Notes |
| --- | --- | --- |
| Temporal rows | `wall` timestamp; `event` integer sequence | Transcript timestamps stay unchanged, including repeats. A unique event number follows wall order with a stable emission-order tie-break across all families, including turns. The catalog reader keeps only one row per index value. |
| All content rows | `agent_id`; `turn_id`, `call_id`, `model`, `effort` when supplied | IDs, model and effort are strings. Child `turn_id` inherits the main turn of an explicit spawn call; without that link, it uses the main [human prompt, next human prompt) interval containing the row. It is empty when no parent turn is known. `metadata_json` keeps source line, prompt/message IDs and provider extras. Large payload strings stay in `input_json`, `tool_use_result_json`, and Codex `native_json`, never in `metadata_json`. |
| `conversation/<role>` | TextLog text, severity, colour; shared facts above | Full user, assistant, injected, inter-agent, thinking, and compaction text. Child rows share the entity, start with `[a<n>] `, and use a darker role colour. Encrypted inter-agent parts retain a size placeholder. |
| `conversation/current` | Markdown text; shared facts | Latest main-agent message with role and time; thinking stays in its own view. |
| `tools` | Strings `tool`, `phase`, `kind`, `input_json`, `tool_use_result_json`; bool `is_error`; float64 `elapsed_ms`; shared facts | One row per model call or output part; `phase` is `call` or `result`, and calls set `is_error=false`. Tool names are values, including `mcp/<server>/<tool>`. Unknown elapsed time is null. Spawned `child_agent_id` stays in `metadata_json`. Output replays are deduplicated only when an explicit output ID is present; repeated text at later source lines is retained. Codex calls keep their raw arguments or scripts and raw outputs. |
| `elapsed/tools/<kind>` | Float scalar; string `tool`; shared facts | Finite call-to-result times only. Both providers use these kinds: `shell`, `file_read`, `file_edit`, `web_search`, `mcp`, `subagent`, `other`, `plan`, `image`; unknown tools use `other`. Child samples use `/children` siblings, with separate main/children legend names. |
| `executions/<kind>` | TextLog; bool `is_error`; strings `input_json`, `native_json`; shared facts | Codex native command, file-change, MCP, search, and image-view details. `executions/command:is_error` marks native command failures; model tool results do not. Only explicit `call_id` joins a model invocation; native item IDs stay in `metadata_json`. |
| `context/<kind>` | TextLog; shared facts | Codex base instructions, developer messages, and environment context. |
| `usage/<counter>` | Float scalar; shared facts including string `model` and `effort` | Input, output, cache-read, cache-creation, and thinking tokens, counted once per request. Input means uncached input. Codex subtracts cached input from input, with a zero floor; cached input stays in `cache_read_tokens`, including turn totals and `total_cache_read_tokens` session properties. Child samples use `/children` siblings and retain the child's model and effort. Unknown request model or effort is null. Claude finalizes usage at the last record for each assistant message and excludes synthetic model markers. |
| `media/images` | Encoded bytes, MIME type; shared facts | Image source stays in `metadata_json`. Media bytes use size placeholders in tool metadata; all other metadata keys are retained. Codex tool-output images keep the call ID and use the same media placeholders as message images. |
| `lifecycle/<kind>` | TextLog; shared facts | Hook, compaction, API-error, and workflow text. Claude API errors use `lifecycle/api_errors`. |
| `lifecycle/workflows`, `lifecycle/workflow_scripts` | TextLog; shared facts | Claude `workflows/wf_*.json` sidecars retain the full run document: status, errors, results, phases, logs and token totals. Scripts from `workflows/scripts/*.js` are shown verbatim, never executed. Both enter the inventory and fingerprint. Runs use their reported time; scripts and untimed runs use the earliest timed event across the session, children and workflows. Sidecars without any timed event cause a source-naming error. |
| `turns` | TextLog; strings `turn_id`, `model`, `effort`, `agent_id`; int64 `turn_index`, `n_tool_calls`, `n_assistant_messages`, `n_images`, all token counters; float64 `elapsed_ms` | One row per human prompt; only human prompts start or name turns. Elapsed time runs from the prompt to the last assistant, thinking, tool or usage activity. Rows at or after the next human prompt join the next turn; injected/context rows do not extend elapsed time. Source line and prompt ID stay in `metadata_json`. Missing turn model/effort is an empty string. |
| `turns/elapsed_ms`, `turns/output_tokens`, `turns/tool_calls` | Float scalar; shared facts | Main-session turn summaries. |
| `agents` property table | `n`, `agent_id`, `agent_type`, `description` | Static child inventory; time-ordered `n` maps `[a<n>]` to the source identity. Claude discovers workflow subagents recursively; `.meta.json` labels enter the inventory and fingerprint. |
| Session properties | Typed session ID, agent (`claude` or `codex`), profile, host, working directory, git branch, CLI versions, models, counts | Static, like scalar series labels. Start time is the first session event. Claude replays increment `property:skipped:replayed_record`. Codex totals come from the last available thread snapshot and are null when unknown. |

Claude tool references retain their names. Offloaded text is read without newline translation; missing files leave the preview and increment `property:skipped:offloaded_output_missing`.

Each file carries a default layout with only the views that have data: Conversation beside Current message, plus Thinking, Tools, Executions, Context, Lifecycle, token plots, Images, Tool elapsed, and Turns. Tools comes before Executions in the detail tabs. The views include shared child text and `/children` scalar series. The catalog uses the same layout function with all views.

The recordings are written with rerun-sdk 0.38.1. Open them with a viewer of that version or newer.

## Convert your sessions

From the workspace root:

```bash
pixi run -e agent-traces agent-traces-convert-all --home ~/.claude --out ~/agent-traces
pixi run -e agent-traces agent-traces-convert-all --home ~/.codex --out ~/agent-traces
pixi run -e agent-traces rerun ~/agent-traces/claude/<session-id>.rrd
```

`convert-all` writes `<out>/<profile>/<session-id>.rrd` and a `manifest.json` per profile. An exclusive `manifest.lock` protects each run from manifest load through its last save. A concurrent run waits and prints a waiting message.
Run it again at any time. It converts a session again only when the session's files changed, when you pass a different `--host`, or when a new version of this package changes what a recording holds.
Filter with `--project`, `--session-id`, or `--since YYYY-MM-DD`. `--project` matches a substring of the working directory: the project directory name for Claude, or `session_meta.cwd` for Codex.

`convert-all` exits 1 if any session fails; policy skips and unchanged inputs do not cause failure. `register` exits 1 if a registration fails or a recording is missing. Successful work remains saved in either case.

To convert one transcript, use `agent-traces-convert --session <file>.jsonl --out <file>.rrd`. Single-file Codex conversion reads every rollout header in the home to find children. Invalid input layouts return exit status 1 without creating output files.

## Profiles

The profile is the home folder's name without the dot: `~/.claude` gives `claude`, and `~/.codex` gives `codex`.
A second home, for example one set with `CLAUDE_CONFIG_DIR` or `CODEX_HOME`, gets its own profile from its folder name.
Use `--profile` to choose a different name.

## Register on a catalog

A `rerun server` has no access control: anyone who can reach its port can read every recording. Treat catalogs serving these recordings as sensitive; bind the server to localhost or keep it on a private network.

Start a catalog server in another terminal (default port 51234), then register one or more output folders:

```bash
pixi run -e agent-traces rerun server --host 127.0.0.1
pixi run -e agent-traces agent-traces-register --catalog-url <catalog-url> --out ~/agent-traces/<computer-a> ~/agent-traces/<computer-b>
```

Profiles are the union across the supplied roots. Each profile becomes one dataset, `agent-traces-<profile>`, with one registration batch and one segment per session. Roots are processed in argument order; if a session appears in several roots, the first copy wins and later copies count as skipped duplicates, including with `--replace`. Blueprints are written once per profile under the first root that contains it.
`host`, `profile`, and `agent` are segment-table columns, so a query can group or filter by them. The table starts with recording name, wall start, turns, host, agent, and profile; skipped-record counters are hidden. Every registration refreshes both default layouts and retires older blueprint registrations. After unregistering them, it deletes local files named `agent-traces-<32 hex>.rbl` or `agent-traces-table-<32 hex>.rbl` whose parent directory has the registered profile's name, including files under other output roots. Only `file://` URLs with an empty or `localhost` authority qualify; other files, including `agent-traces.rbl`, are left alone. Existing datasets are reused.
The server opens each recording through its `file://` path, so it must be able to read the output folder. All files are written readable by every user (mode 644).

`register` skips sessions that the dataset already has. After you convert changed sessions again, add `--replace` to update them.
`rerun server` keeps registrations in memory. After a server restart, run `register` again.

## Query the catalog

Set `catalog_url` and `dataset_name` for your catalog, then run these blocks in order. Each content query applies `filter_contents` before one reader across all segments and projects only small typed columns. Read with `index="event"`: a wall-indexed reader collapses repeated timestamps. Use `wall` for time-window filters before projection and for UTC day grouping. The returned `wall` values are UTC without a timezone marker; use matching timezone-naive UTC bounds. Filter the relevant entity’s `agent_id` to `''` for main-agent-only counts or text searches, otherwise child copies also count. Do not use latest-at filling for event totals.

Session facts come from the segment table; no content scan is needed. The `optional` schema guard handles absent properties and content columns without inventing values for unknown facts.

```python
import pyarrow as pa
from datafusion import col, lit, functions as F
from rerun.catalog import CatalogClient

client = CatalogClient(catalog_url)
dataset = client.get_dataset(dataset_name)

def optional(frame, name, dtype):
    return col(name)[0] if name in frame.schema().names else lit(None).cast(dtype)

segments = dataset.segment_table()
sessions = segments.select(
    "rerun_segment_id",
    optional(segments, "property:session:n_turns", pa.int64()).alias("turns"),
    optional(segments, "property:session:n_tool_calls", pa.int64()).alias("tool_calls"),
    optional(segments, "property:session:n_subagents", pa.int64()).alias("subagents"),
    optional(segments, "property:session:models", pa.string()).alias("models"),
    optional(segments, "property:session:total_cost_usd", pa.float64()).alias("cost_usd"),
).to_arrow_table()
```

Output tokens per model per UTC day include main and child requests. The unique event index puts each sample on its own row, so `coalesce` selects its main or child column. The guard also supports datasets with no child columns. Cache only this small projection: both this aggregate and the next recipe consume it.

```python
usage = dataset.filter_contents(["/usage/output_tokens/**"]).reader(index="event")
main_tokens = optional(usage, "/usage/output_tokens:Scalars:scalars", pa.float64())
child_tokens = optional(usage, "/usage/output_tokens/children:Scalars:scalars", pa.float64())
usage_rows = usage.select(
    "rerun_segment_id", F.date_trunc("day", col("wall")).alias("day"),
    F.coalesce(main_tokens, child_tokens).alias("tokens"),
    F.coalesce(optional(usage, "/usage/output_tokens:model", pa.string()),
               optional(usage, "/usage/output_tokens/children:model", pa.string()), lit("")).alias("model"),
    F.coalesce(child_tokens, lit(0.0)).alias("child_tokens"),
).filter(col("tokens").is_not_null()).cache()
output_per_model_day = usage_rows.aggregate(
    group_by=[col("rerun_segment_id"), col("model"), col("day")],
    aggs=[F.sum(col("tokens")).alias("output_tokens")],
).to_arrow_table()
```

Subagent share of output tokens reuses the cached projection, with no second server read. Substitute another counter in the usage query to measure that counter. Zero total tokens gives a null share.

```python
subagent_share = usage_rows.aggregate(
    group_by=[col("rerun_segment_id")],
    aggs=[(F.sum(col("child_tokens")) / F.nullif(F.sum(col("tokens")), lit(0.0))).alias("share")],
).to_arrow_table()
```

Calls, error rate, median latency, and approximate p90 latency per tool use only `tools` facts. The first aggregate combines multi-part results by agent and call ID; the second groups by tool within each segment. Error rate uses completed calls, and latency uses their known values. Calls without results still count. Codex native command failures belong to `executions/command:is_error`, not model tool results, and Codex model tools have unknown latency. The schema guard supplies a typed null when no tool reports latency; empty columns need not be requested from the reader.

```python
tools = dataset.filter_contents(["/tools"]).reader(index="event")
tool_rows = tools.select(
    "rerun_segment_id", col("/tools:tool")[0].alias("tool"), col("/tools:phase")[0].alias("phase"),
    col("/tools:agent_id")[0].alias("agent_id"), col("/tools:call_id")[0].alias("call_id"),
    col("/tools:is_error")[0].alias("is_error"),
    optional(tools, "/tools:elapsed_ms", pa.float64()).alias("elapsed_ms"),
).filter(col("phase").is_not_null())
calls = tool_rows.aggregate(
    group_by=[col("rerun_segment_id"), col("tool"), col("agent_id"), col("call_id")],
    aggs=[F.count(col("call_id"), distinct=True, filter=col("phase") == lit("call")).alias("calls"),
          F.bool_or(col("phase") == lit("result")).alias("completed"),
          F.bool_or(col("is_error"), filter=col("phase") == lit("result")).alias("error"),
          F.max(col("elapsed_ms"), filter=col("phase") == lit("result")).alias("latency")],
)
tool_summary = calls.aggregate(
    group_by=[col("rerun_segment_id"), col("tool")],
    aggs=[F.sum(col("calls")).alias("calls"),
          F.avg(col("error").cast(pa.float64()), filter=col("completed")).alias("error_rate"),
          F.median(col("latency")).alias("median_latency_ms"),
          F.approx_percentile_cont(col("latency"), 0.9).alias("p90_latency_ms")],
).to_arrow_table()
```

## Sessions from another computer

Copy that computer's home folder to this one in any way you like. Copy at least `projects/` for Claude, or `sessions/` and `archived_sessions/` for Codex.
Then convert the copy with `--host` set to the computer's name, so the recordings say where the sessions ran:

```bash
pixi run -e agent-traces agent-traces-convert-all --home <copy>/.claude --out ~/agent-traces/<computer> --host <computer>
```

Register every computer's output folder in one command; matching profiles join the same datasets:

```bash
pixi run -e agent-traces agent-traces-register --catalog-url <catalog-url> --out ~/agent-traces/<computer-a> ~/agent-traces/<computer-b>
```
If one session appears on two computers, the copy registered first is kept and the other is counted as a skipped duplicate.

## Limits

- Codex rollouts from CLI versions older than 0.150 are skipped. `convert-all` prints the count per version. Pre-envelope files are skipped as `property:skipped:legacy_rollout`. Children without native completion items retain their message records. Unknown records and response subtypes are counted by full tag.
- Codex rollouts do not record how long each command ran, so Codex tool rows have no elapsed time.
- Codex subagent rollouts go into their parent's recording. When the parent is skipped or fails, its subagents are skipped too, and the summary says why.
- In Rerun 0.38.1, a TextLog view can show an empty panel when the time cursor is after a very tall last row. Move the cursor onto the row or use Current message.
- If `convert-all` reports an unsupported manifest version, delete that `manifest.json`. The next run converts every session in that profile again.
- Costs are not computed. The Claude CLI's own session total is kept as a property when the transcript has one.


## Validation

Run `pixi run -e agent-traces-dev --frozen gate` for static checks, unit tests, and the catalog integration test. The integration test starts its own server on a free local port and stops that process afterward.
Run `pixi run -e agent-traces-dev --frozen tests-golden` for the saved child-row pixel check in the environment's headless Rerun 0.38.1 viewer. The tests report a skip when the required binary or graphics adapter is absent.
