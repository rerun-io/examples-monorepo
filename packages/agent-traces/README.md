# agent-traces

Convert Claude Code and Codex session transcripts into Rerun recordings: one `.rrd` per session.
You can then open a session in the Rerun viewer, or register many sessions on a Rerun catalog and query them together.

## What a recording holds

- One timeline, `wall`, from each transcript record's timestamp.
- `conversation/`: user, assistant, thinking, and compaction text, in full.
- `tools/<name>`: one row per tool call and one per result, with the full output once. MCP tools go under `tools/mcp/<server>/<tool>`. `tools/elapsed_ms/<name>` plots the time from call to result.
- `usage/`: tokens per model request, counted once per request.
- `media/images`: screenshots and pasted images from the transcript.
- `lifecycle/`: hooks, compaction boundaries, API errors, and similar events.
- `turns`: one row per human prompt, with elapsed time, tool calls, and token totals.
- `agents/<id>/...`: the same tree for each subagent, inside the parent session's recording.
- Properties: session id, agent (`claude` or `codex`), profile, host, working directory, git branch, CLI versions, models, and counts.

Each file carries a default layout, so it opens ready to read.
The recordings are written with rerun-sdk 0.38.1. Open them with a viewer of that version or newer.

## Convert your sessions

From the workspace root:

```bash
pixi run -e agent-traces agent-traces-convert-all --home ~/.claude --out ~/agent-traces
pixi run -e agent-traces agent-traces-convert-all --home ~/.codex --out ~/agent-traces
pixi run -e agent-traces rerun ~/agent-traces/claude/<session-id>.rrd
```

`convert-all` writes `<out>/<profile>/<session-id>.rrd` and a `manifest.json` per profile.
Run it again at any time. It converts a session again only when the session's files changed, when you pass a different `--host`, or when a new version of this package changes what a recording holds.
Filter with `--project`, `--session-id`, or `--since YYYY-MM-DD`.

To convert one transcript, use `agent-traces-convert --session <file>.jsonl --out <file>.rrd`.

## Profiles

The profile is the home folder's name without the dot: `~/.claude` gives `claude`, and `~/.codex` gives `codex`.
A second home, for example one set with `CLAUDE_CONFIG_DIR` or `CODEX_HOME`, gets its own profile from its folder name.
Use `--profile` to choose a different name.

## Register on a catalog

Start a catalog server in another terminal (default port 51234), then register the output folder:

```bash
pixi run -e agent-traces rerun server
pixi run -e agent-traces agent-traces-register --catalog-url rerun+http://localhost:51234 --out ~/agent-traces
```

Each profile becomes one dataset, `agent-traces-<profile>`, with one segment per session.
`host`, `profile`, and `agent` are segment-table columns, so a query can group or filter by them.
The server opens each recording through its `file://` path, so it must be able to read the output folder. All files are written readable by every user (mode 644).

`register` skips sessions that the dataset already has. After you convert changed sessions again, add `--replace` to update them.
`rerun server` keeps registrations in memory. After a server restart, run `register` again.

## Sessions from another computer

Copy that computer's home folder to this one in any way you like. Copy at least `projects/` for Claude, or `sessions/` and `archived_sessions/` for Codex.
Then convert the copy with `--host` set to the computer's name, so the recordings say where the sessions ran:

```bash
pixi run -e agent-traces agent-traces-convert-all --home <copy>/.claude --out ~/agent-traces/<computer> --host <computer>
```

Register each computer's output folder separately; its sessions join the same `agent-traces-<profile>` datasets.
If one session appears on two computers, the copy registered first is kept and the other is counted as a skipped duplicate.

## Limits

- Codex rollouts from CLI versions older than 0.150 are skipped. `convert-all` prints the count per version.
- Codex rollouts do not record how long each command ran, so Codex tool rows have no elapsed time.
- Codex subagent rollouts go into their parent's recording. When the parent is skipped or fails, its subagents are skipped too, and the summary says why.
- A line that is not valid JSON, usually left by an interrupted write, is skipped with a warning that names the file and line. The recording counts it as `damaged-line` in its skipped records. A line that is valid JSON but not a known record shape still fails the session, and `convert-all` reports it and continues.
- If `convert-all` reports an unsupported manifest version, delete that `manifest.json`. The next run converts every session in that profile again.
- Costs are not computed. The Claude CLI's own session total is kept as a property when the transcript has one.
