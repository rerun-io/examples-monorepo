# Node App Rerun Validation

Date: 2026-05-17

## Numeric Checks

HoCap command, run from `packages/mv-api`:

```bash
pixi run -e mv-api-dev mv-api-full-app \
  --rr-config.save artifacts/node-validation/hocap-node-app.rrd \
  --rr-config.headless \
  --max-frames 1 \
  --camera-source dataset
```

Result:

- `node_num_rows=10932.0`
- `node_processed_timestamps=1085`

LG command, run from `packages/mv-api`:

```bash
pixi run -e mv-api-dev mv-api-full-app \
  --rr-config.save artifacts/node-validation/lg-node-app.rrd \
  --rr-config.headless \
  --max-frames 1 \
  --camera-source estimated \
  synced-videos \
  --root-directory /mnt/8tb/data/exoego-self-collected/lg/ServerAssembly_4Views_11-3-25
```

Result:

- `node_num_rows=18100.0`
- `node_processed_timestamps=4513`

Use `rerun rrd stats <rrd>` for entity-count checks against the node app output.

## Native Viewer Screenshots

Whole node-app screenshots:

- `hocap-node-full.png`
- `lg-node-full.png`

Per-view crops from those native Rerun screenshots:

- `views/hocap_ego_hololens_kv5h72.png`
- `views/hocap_exo_037522251142.png`
- `views/hocap_exo_043422252387.png`
- `views/hocap_exo_046122250168.png`
- `views/hocap_exo_105322251225.png`
- `views/hocap_exo_105322251564.png`
- `views/hocap_exo_108222250342.png`
- `views/hocap_exo_115422250549.png`
- `views/hocap_exo_117222250549.png`
- `views/lg_ego_TOP.png`
- `views/lg_exo_FRONT.png`
- `views/lg_exo_LEFT.png`
- `views/lg_exo_RIGHT.png`

Native screenshot command shape:

```bash
xvfb-run -a -s "-screen 0 1920x1080x24" \
  /home/pablo/0Dev/work/rerun-projects/examples-monorepo/.pixi/envs/mv-api-dev/bin/rerun \
  --window-size 1920x1080 \
  --screenshot-to artifacts/node-validation/screenshots/<name>.png \
  artifacts/node-validation/<node-rrd>.rrd
```

Rerun saved the screenshots but exited with the known Xvfb/WGPU validation panic after saving:

```text
Surface width and height must be within the maximum supported texture size. Requested was (40000, 40000).
```

The saved PNGs were verified with `file`, `ls -lh`, and manual visual inspection through Codex image viewing.
