# gsplat-viewer

The gsplat extension of the canonical viewer (`packages/canonical-viewer`): `setup::EXTENSION` registers the compute
splat, Hexels3D, SOG, SOG LOD and animated SOG visualizers. Open recordings with `rerun-canonical`:

```bash
rerun-canonical recording.rrd
# Without a display:
rerun-canonical recording.rrd --headless --port 9877
```

For complete headless frame timings, use the canonical viewer's `viewer-probe` (`packages/canonical-viewer`);
see the [CLI reference](../gsplat-cli/README.md) for flags and [architecture](../../docs/architecture.md) for viewer limits.
