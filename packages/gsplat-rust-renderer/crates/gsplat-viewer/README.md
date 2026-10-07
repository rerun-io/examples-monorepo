# gsplat-viewer

```bash
target/release/gsplat-rust-renderer recording.rrd
# Without a display:
target/release/gsplat-rust-renderer recording.rrd --headless --port 9877
# Build with --features probe to time complete headless frames:
target/release/gsplat-viewer-probe recording.rrd --out frames.json --window-size 1920x1080
```

The frame probe requires one 3D view and writes the measured camera path and a PNG
beside its report. See [architecture](../../docs/architecture.md) for viewer limits.
