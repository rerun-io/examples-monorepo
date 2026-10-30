# SelfCap night2 production runner

Operational code for the authorized ep001 batch. State, configuration, reports and large files live under `/tmp/selfcap-night2`; no runtime data belongs here. Run from a detached, pinned worktree with package imports from that same worktree. The SOMA executable has its own explicit pinned worktree and isolated state root. Change revisions only between jobs.

`runner.py status` reads the durable SQLite state. Worker units run `runner.py worker ROLE INDEX`; `runner.py controller` owns retries and milestones. Deployment verifies all stage imports and a real S2 registration before opening S2 claims.

Config writers must publish a complete JSON document with a unique sibling
temporary file followed by `Path.replace`; in Python, use `runner.write_config`.
The runner retries partial legacy reads and then uses its last complete snapshot.
Atomic replacement prevents partial reads; it does not merge concurrent edits.

Run the isolated operational regression suite from the repository root:

```bash
pixi run --as-is -e selfcap-curate-dev python -m pytest -q ops/selfcap-night2/tests
```

These tests use temporary caches and databases, synthetic recordings, and fake
transport boundaries. They do not start production workers or publish layers.
