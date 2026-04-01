# pyvrs-viewer

Python port of [rerun-io/cpp-example-vrs](https://github.com/rerun-io/cpp-example-vrs) using [pyvrs](https://github.com/facebookresearch/pyvrs).

Reads VRS files and logs camera streams (via `rr.EncodedImage` / `rr.VideoStream`) and IMU data (via `rr.Scalar` / `rr.Arrows3D`) to Rerun `.rrd` files.

[![Rerun](https://img.shields.io/badge/rerun-0.30+-blue)](https://rerun.io)

## Usage

```bash
# Save to .rrd file
pixi run -e pyvrs-viewer vrs-to-rrd-quest

# Or with a custom VRS file
pixi run -e pyvrs-viewer -- python tools/demos/vrs_to_rrd.py --vrs-path /path/to/file.vrs --rr-config.save output.rrd
```
