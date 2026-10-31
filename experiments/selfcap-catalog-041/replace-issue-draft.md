# Draft — do not file automatically

Title: OSS server: REPLACE repeatedly rebuilds dataset schema and blocks unrelated catalog reads

This is a follow-up to #12778 and the remaining REPLACE limitation described in
#12774. Tested with released 0.41.0 on Linux aarch64; 0.38.1 shows similar scaling.
Both Python client versions reproduce it.

## Reproduction

Start an isolated server and run the attached `repro_replace_scaling.py` on the
same host. The script uses only three generated RRDs, each with 64 scalar property
fields. It varies the number of layers per segment and times replacing every
source, then unregistering and reinserting those same sources.

```
REPRO_URL=rerun+http://127.0.0.1:51241 python repro_replace_scaling.py
```

Measured 0.41 REPLACE times: 3 sources 0.007 s; 24 sources 0.115 s; 96 sources
0.950 s; 192 sources 3.276 s. In a repeat, replacing 192 sources took 3.207 s,
while unregistering and reinserting them took 0.323 s.

A larger existing dataset with 5,494 sources took approximately 0.78 s per
replacement in a 690-source batch. During the batch, a separate catalog-list
request timed out. No viewer was required.

## Signals

Two native backtraces sampled the active worker in:

```
ComponentColumnDescriptor::to_arrow_field
ChunkColumnDescriptors::arrow_fields
re_server::store::source::schema_of_store
Dataset::schema
RerunCloudHandler::register_with_dataset
```

The debug log continued advancing, and strace showed file-close/progress-log
activity at approximately 0.77 s intervals. FD use decreased as old sources were
released (6,217 to 6,103, limit 500,000). This is not FD exhaustion or a deadlock.
The registration RPC had not returned a handle, so this is not an unreported
failure in client `wait()`. Explicit duplicate errors are correctly surfaced by
both clients against the 0.41 server.

## Likely cause

`Dataset::add_source()` invalidates `cached_schema` for every Overwritten source.
The next replacement calls `schema()`, merging schemas from all resident sources
again. This also occurs when the old and new source schemas are identical.
`register_with_dataset()` holds the catalog-wide write guard across the whole
operation, so unrelated datasets' reads queue behind it.

Could identical-schema replacement preserve the cached schema, or could source
schemas be tracked by hash/refcount? Shortening the catalog-wide write-lock scope
would also prevent the resulting catalog-wide read starvation.

## Workarounds tested

Skipping identical segment/layer/URI references avoids the unnecessary work.
Smaller batches bound each blocking interval but retain the total cost.
Unregister/reinsert is faster but introduces an availability gap. Increasing the
FD limit or publishing the same identity under a distinct URI does not address
the schema-cache cost.
