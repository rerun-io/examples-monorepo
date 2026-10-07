# agent-traces

Convert Claude Code session transcripts into Rerun recordings: one `.rrd` per session.
You can then open a session in the Rerun viewer.

Recordings hold full transcripts and tool outputs, which may include secrets. Treat converted files as sensitive.

## What a recording holds

One fact/type table applies to every content entity; a declared fact gets a typed column when any row supplies it.
Absent cells are null; `agent_id` is always present and is an empty string for the main agent.
Read the catalog with `event` to retain repeated `wall` timestamps; use `wall` for time windows, UTC day grouping, and the viewer timeline.

| Entity (family) | Typed columns | Notes |
| --- | --- | --- |
| Temporal rows | `wall` timestamp; `event` integer sequence | Transcript timestamps stay unchanged, including repeats. A unique event number follows wall order with a stable emission-order tie-break across all families, including turns. The catalog reader keeps only one row per index value. |
| All content rows | `agent_id`; `turn_id`, `call_id`, `model`, `effort` when supplied | IDs, model and effort are strings. Child `turn_id` inherits the main turn of an explicit spawn call; without that link, it uses the main [human prompt, next human prompt) interval containing the row. It is empty when no parent turn is known. `metadata_json` keeps source line, prompt/message IDs and provider extras. Large payload strings stay in `input_json`, `tool_use_result_json`, never in `metadata_json`. |
| `conversation/<role>` | TextLog text, severity, colour; shared facts above | Full user, assistant, injected, thinking, and compaction text. Child rows share the entity, start with `[a<n>] `, and use a darker role colour. |
| `conversation/current` | Markdown text; shared facts | Latest main-agent message with role and time; thinking stays in its own view. |
| `tools` | Strings `tool`, `phase`, `kind`, `input_json`, `tool_use_result_json`; bool `is_error`; float64 `elapsed_ms`; shared facts | One row per model call or output part; `phase` is `call` or `result`, and calls set `is_error=false`. Tool names are values, including `mcp/<server>/<tool>`. Unknown elapsed time is null. Spawned `child_agent_id` stays in `metadata_json`. |
| `elapsed/tools/<kind>` | Float scalar; string `tool`; shared facts | Finite call-to-result times only. Kinds: `shell`, `file_read`, `file_edit`, `web_search`, `mcp`, `subagent`, `other`; unknown tools use `other`. Child samples use `/children` siblings, with separate main/children legend names. |
| `usage/<counter>` | Float scalar; shared facts including string `model` and `effort` | Input, output, cache-read, cache-creation, and thinking tokens, counted once per request. Input means uncached input. Child samples use `/children` siblings and retain the child's model and effort. Unknown request model or effort is null. Claude finalizes usage at the last record for each assistant message and excludes synthetic model markers. |
| `media/images` | Encoded bytes, MIME type; shared facts | Image source stays in `metadata_json`. Media bytes use size placeholders in tool metadata; all other metadata keys are retained. |
| `lifecycle/<kind>` | TextLog; shared facts | Hook, compaction, API-error, and workflow text. Claude API errors use `lifecycle/api_errors`. |
| `lifecycle/workflows`, `lifecycle/workflow_scripts` | TextLog; shared facts | Claude `workflows/wf_*.json` sidecars retain the full run document: status, errors, results, phases, logs and token totals. Scripts from `workflows/scripts/*.js` are shown verbatim, never executed. Both enter the inventory and fingerprint. Runs use their reported time; scripts and untimed runs use the earliest timed event across the session, children and workflows. Sidecars without any timed event cause a source-naming error. |
| `turns` | TextLog; strings `turn_id`, `model`, `effort`, `agent_id`; int64 `turn_index`, `n_tool_calls`, `n_assistant_messages`, `n_images`, all token counters; float64 `elapsed_ms` | One row per human prompt; only human prompts start or name turns. Elapsed time runs from the prompt to the last assistant, thinking, tool or usage activity. Rows at or after the next human prompt join the next turn; injected/context rows do not extend elapsed time. Source line and prompt ID stay in `metadata_json`. Missing turn model/effort is an empty string. |
| `turns/elapsed_ms`, `turns/output_tokens`, `turns/tool_calls` | Float scalar; shared facts | Main-session turn summaries. |
| `agents` property table | `n`, `agent_id`, `agent_type`, `description` | Static child inventory; time-ordered `n` maps `[a<n>]` to the source identity. Claude discovers workflow subagents recursively; `.meta.json` labels enter the inventory and fingerprint. |
| Session properties | Typed session ID, agent (`claude`), profile, host, working directory, git branch, CLI versions, models, counts | Static, like scalar series labels. Start time is the first session event. Claude replays increment `property:skipped:replayed_record`. |

Claude tool references retain their names. Offloaded text is read without newline translation; missing files leave the preview and increment `property:skipped:offloaded_output_missing`.

Each file carries a default layout with only the views that have data: Conversation beside Current message, plus Thinking, Tools, Lifecycle, token plots, Images, Tool elapsed, and Turns. The views include shared child text and `/children` scalar series.

The recordings are written with rerun-sdk 0.38.1. Open them with a viewer of that version or newer.

## Convert your sessions

To convert one transcript, use `agent-traces-convert --session <file>.jsonl --out <file>.rrd`. Invalid input layouts return exit status 1 without creating output files.

## Profiles

The profile is the home folder's name without the dot: `~/.claude` gives `claude`.
A second home, for example one set with `CLAUDE_CONFIG_DIR`, gets its own profile from its folder name.
Use `--profile` to choose a different name.

## Sessions from another computer

Copy that computer's home folder to this one in any way you like. Copy at least `projects/` for Claude.
Then convert the copy with `--host` set to the computer's name, so the recordings say where the sessions ran.


## Limits

- In Rerun 0.38.1, a TextLog view can show an empty panel when the time cursor is after a very tall last row. Move the cursor onto the row or use Current message.
- Costs are not computed. The Claude CLI's own session total is kept as a property when the transcript has one.


## Validation

Run `pixi run -e agent-traces-dev --frozen gate` for static checks, and unit tests.
Run `pixi run -e agent-traces-dev --frozen tests-golden` for the saved child-row pixel check in the environment's headless Rerun 0.38.1 viewer. The tests report a skip when the required binary or graphics adapter is absent.
