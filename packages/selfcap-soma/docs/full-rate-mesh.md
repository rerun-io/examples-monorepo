# Full-rate mesh and exact video timestamps

The batch mesh is now written at every fitted frame, rather than every third
frame. d31 has 4,367 mesh samples and 4,367 SOMA joint samples. A layer export
rejects a decimated or shifted mesh before publication. The upper-body runner
also writes full-rate meshes.

The exact-time check found a separate numerical problem: Rerun's numeric-array
duration constructor multiplies seconds by 1e9 then truncates. Two successive
writes could move 123 d31 samples 1–2 ns earlier. This is not the centimetre-scale
motion error, but it violated exact timestamp equality. The writer now rounds
back to integer nanoseconds and passes `timedelta64[ns]` arrays, including the
2D overlays and per-frame pose tensors.

The three-way validation compares every published mesh/joint/2D-overlay
sample against the source triangulation and the display `/world/skeleton`
points. No half-frame offset or new hand-latency shift is introduced. The
existing Quest-to-video mapping and 48 ms hand correction are unchanged.

Full-rate storage is intentionally larger: about 407 MB per layer rather than
163 MB. The first full-rate timing measured 9.0 s for the fit recording write
and 22.2 s for mesh normals/projection/layer writing, versus about 3.1 + 9.0 s
previously. Total local run was 51.4 s. Runtime varies with the shared host load.

For 690 similar segments, the published layers alone need about 281 GB decimal.
Keeping all intermediate fit recordings adds roughly 343 GB, so both together
exceed the approximately 392 GiB currently free on the local filesystem.
Choose a larger output volume or an explicit post-publication intermediate
retention policy; no automatic deletion or silent decimation was added.

Batch checkpoint version is `soma-batch-v5-exact-time-full-mesh`, invalidating
old sparse-mesh outputs. Tests reject omitted frames and 1 ns shifts, and
cover the integer-preserving TimeColumn conversion.

Final exact-time run: 407,055,097 bytes; fit write 10.33 s; layer write 23.48 s;
total before publication 55.85 s under the current shared load. All 4,367
mesh, joint, seven overlay and both display-source timestamp arrays match
exactly. Their sorted little-endian int64 SHA-256 is
`0ecef1e403de1ea87196bd029b79bcc85caf534341e814718ab445c04fd32cdd`,
matching the runner's source proof. Local artifact:
`/tmp/soma-fit/exact-full-rate/selfcap-post-ep1-v2/selfcap__2025-12-03__d31bf1da__ep001/`.
