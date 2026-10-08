# gsplat-viewer

```bash
target/release/gsplat-rust-renderer recording.rrd
# Without a display:
target/release/gsplat-rust-renderer recording.rrd --headless --port 9877
```

For complete headless frame timings, use `gsplat probe` (the `gsplat-cli` `probe` feature);
see the [CLI reference](../gsplat-cli/README.md) for flags and [architecture](../../docs/architecture.md) for viewer limits.
