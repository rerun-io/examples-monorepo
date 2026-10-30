# Deferred SelfCap 0.41 catalog test

This isolates the 0.41 runtime from the monorepo's Rerun pins. The shared Spark
catalog was upgraded to 0.41.0 on 2026-10-30 after the writer finished and Pablo
authorized the swap. Its stable runtime is `~/dataforge-dev/rerun-041`, and the
persistent `rerun-dev-catalog.service` restores a saved snapshot on start.
The separate `rerun-catalog.service` on port 9876 was not changed.

Before the pause instruction arrived, the earlier authorized steps published 690
summary ranks and the visible rank column to `selfcap-post-ep1-v2` on `:51235`.
No other source layer or source dataset was modified. Those registrations remain;
undoing them would itself change the live catalog. No more source writes are planned
until Pablo asks.

## Commands for the resumed session

Run these on Spark, from the updated checkout, in a host-side terminal. Review each
step with Pablo; these commands are not scheduled or started automatically.

```bash
# Optional later refresh: changes only SelfCap summary files/registrations and blueprints.
pixi run -e dataforge --frozen python packages/dataforge/tools/apps/selfcap_blueprints.py \
  --catalog-name selfcap-post-ep1-v2 \
  --output-root /home/pablo/dataforge-dev/selfcap-post-ep1-v2 --refresh-quality

# Read-only inventory of all visible datasets, layers, assets, blueprints, and tables.
pixi run -e dataforge --frozen python packages/dataforge/tools/apps/catalog_snapshot.py \
  --catalog-url rerun+http://127.0.0.1:51235 \
  --target /home/pablo/dataforge-dev/catalog-snapshots/catalog-after-run.json

# Installs only this experiment's environment and starts only the test unit on :51241.
bash experiments/selfcap-catalog-041/start-test-server.sh

# Restore the SelfCap files onto the test server, with no copies.
pixi run -e dataforge --frozen python packages/dataforge/tools/apps/catalog_restore.py \
  --snapshot-path /home/pablo/dataforge-dev/catalog-snapshots/catalog-after-run.json \
  --catalog-url rerun+http://127.0.0.1:51241 --names selfcap-post-ep1-v2

experiment_root="$PWD/experiments/selfcap-catalog-041"
pixi run --manifest-path "$experiment_root/pixi.toml" python "$experiment_root/blueprints.py" \
  --output-root /home/pablo/dataforge-dev/selfcap-041-test/blueprints

# A 0.41 viewer, on a machine with this experiment installed:
"$experiment_root/.pixi/envs/default/bin/rerun" rerun+http://dgx-spark:51241

# Stops only this experiment:
systemctl --user stop selfcap-catalog-041-test
```

Omit `--names` from catalog_restore to restore all entries. Restored dataset IDs
change, so links containing old dataset IDs must be updated. The inventory uses
successive metadata reads, not an atomic snapshot; take it after the writer finishes.
Keep original snapshots rather than overwriting them.

`blueprints.py` requires 0.41.0 and refuses port 51235 unless `--allow-shared` is supplied.
It preserves the membership of an existing refined dataset by default. Pass
`--populate-refined` only for the isolated experiment that intentionally shares all
source registrations and uses a saved filter. It adds saved ascending
Quality rank and start point 0.5. The refined entry shares all layer file references
and applies a saved `refined = true` table filter; its underlying registrations are
not a physical refined subset. The original recording blueprint is reused.

## Later REPLACE investigation

The apparent server stall was subsequently traced to CPU-bound, quadratic schema
recomputation under the global catalog write lock, not a deadlock or FD leak.
Both 0.38.1 and 0.41.0 show the same scaling. See
[the investigation](replace-investigation.md), the public three-file
[reproduction](repro_replace_scaling.py), and the unfiled
[issue draft](replace-issue-draft.md). Both isolated test servers were stopped.

## Evidence collected before the pause

The 0.38.1 CatalogClient could list, read schemas, query a reader, and register a
summary layer on a scratch dataset on the 0.41 server. The server registered
SelfCap files plus samples from EgoExo4D, MSD G2, and Robocap. The 0.41 viewer
rendered SelfCap rank-ordered cards and the refined filter. Existing 0.38 EgoExo4D
card and recording blueprints and the Robocap card blueprint loaded without
regeneration. The Robocap recording blueprint check was interrupted by the pause.

The source inventory is saved on Spark at:
`/home/pablo/dataforge-dev/catalog-snapshots/catalog-20261030-041-test.json`.
It contains 29 datasets, 8,192 segment/layer references, 72 blueprint references,
and no separate tables or assets. A two-dataset test restore (5,534 segment/layers)
took 21.7 s; a warm recovery took 9.5 s. Linear extrapolation of the first result is
about 32 s for all 8,192 references, excluding cold I/O and per-entry overhead.

**Initial observation, clarified by the later investigation:** A repeated full REPLACE registration on the
test server while its viewer was open stalled, and even catalog listing timed out
at 20 s. Closing the viewer did not recover it; only the isolated test unit was
restarted. The restore tool now skips identical layer URIs and blueprint references;
an unchanged EgoExo4D restore with a connected viewer completed in 0.043 s. This
avoids that repeat-registration path but does not establish that the server issue
is fixed. Re-test changed-URI replacement and mixed read/write behavior before a swap.

The rank tool tests, snapshot guards, lint, typecheck and dead-code checks passed.
The package unit lane passed 916 tests. One full gate hit a beartype checker-generation
race in unchanged concurrent MSD encoding (19 failures, 8 setup errors). Retrying
integration passed 137 tests, with 69 skips. The final 0.41 experiment wrapper has
been syntax/lint checked; its equivalent prototype supplied the viewer evidence.

Rerun 0.39 added the Python table-blueprint API, 0.40 added saved sorting and filters,
and 0.41 added preview start-point control, defaulting to the midpoint:
[0.39](https://rerun.io/docs/changelog/changeset-0-39),
[0.40](https://rerun.io/docs/changelog/changeset-0-40),
[0.41](https://rerun.io/docs/changelog/changeset-0-41).
