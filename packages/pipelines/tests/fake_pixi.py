"""A stand-in for ``pixi run --manifest-path <m> --frozen -e <env> <task> <args>``: the stage tasks of capture_to_splat.

It writes the task's layer like a real stage (``<--output-dir>/<segment>.rrd``, atomically, with its run record) and
registers it on ``--catalog``/``--catalog-url``. ``FAKE_STAGE_FAIL=<task>`` exits 3 before writing; ``fake-scratch --write <file>`` writes the file and publishes nothing; ``FAKE_STAGE_NOOP=<task>``
exits 0 without writing;
``FAKE_STAGE_KILL=<task>`` kills itself while the layer's temp file is open.
"""

import os
import signal
import sys
from pathlib import Path

import rerun as rr
from rerun.catalog import CatalogClient
from simplecv.catalog_layers import atomic_recording, register_layer
from simplecv.run_record import start_run, write_run_record

LAYER_OF_TASK: dict[str, str] = {
    "posekit-lens-masks": "lens_mask", "slam-rs-catalog-layer": "slam_rs", "dataforge-insta360-face-masks": "masks", "colsfm-catalog": "colsfm",
}

arguments: list[str] = sys.argv[1:]
task: str = arguments[arguments.index("-e") + 2]
options: dict[str, str] = {name: value for name, value in zip(arguments, arguments[1:], strict=False) if name.startswith("--")}
if os.environ.get("FAKE_STAGE_FAIL") == task:
    print(f"{task}: failing on purpose")
    sys.exit(3)
if task == "fake-scratch":  # a task whose output is scratch of its step: it publishes nothing
    Path(arguments[arguments.index("--write") + 1]).write_text(task)
    sys.exit(0)
segment: str = options.get("--segment") or options["--source.segment"]
catalog_url: str = options.get("--catalog") or options.get("--catalog-url") or options["--source.catalog-url"]
if os.environ.get("FAKE_STAGE_NOOP") == task:
    sys.exit(0)
run = start_run(arguments, source=Path(__file__))
layer: str = LAYER_OF_TASK[task]
path: Path = Path(options["--layer-path"]) if "--layer-path" in options else Path(options["--output-dir"]) / f"{segment}.rrd"
with atomic_recording(path, application_id="fake-stage", recording_id=segment, send_properties=False) as recording:
    recording.log(f"/{layer}", rr.TextLog(f"{task} {' '.join(arguments)}"), static=True)
    if os.environ.get("FAKE_STAGE_KILL") == task:
        recording.flush()
        os.kill(os.getpid(), signal.SIGKILL)
    write_run_record(recording, run.finish(layer, segment))
register_layer(CatalogClient(catalog_url).get_dataset(name=segment.split("__", 1)[0]), [path], layer)
print(f"{task}: registered {path} as {layer}")
