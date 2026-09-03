"""Owned typed LAMP inference subset from https://github.com/facebookresearch/LAMP.

Taken from upstream revision ``db3e4bf9992874a85946b92e9c8933bba396bc44``
under CC-BY-NC 4.0. The code and public weights are for **non-commercial use
only**; see ``LICENSE`` for the full terms.

File mapping:

- ``core/se3.py`` <- ``lamp/core/se3.py``
- ``core/types.py`` <- ``lamp/core/types.py``
- ``models/blocks.py`` <- ``lamp/models/blocks.py``
- ``models/lifter.py`` <- ``lamp/models/lifter.py``
- ``models/model.py`` <- ``lamp/models/model.py``
- ``models/model_loader.py`` <- ``lamp/models/model_loader.py``
- ``models/model_utils.py`` <- ``lamp/models/model_utils.py``
- ``tracking/smoothing.py`` <- ``lamp/tracking/smoothing.py``
- ``tracking/snippets.py`` <- ``lamp/tracking/snippets.py``
- ``tracking/tracker.py`` <- ``lamp/tracking/tracker.py``
- ``tracking/tracking_utils.py`` <- ``lamp/tracking/tracking_utils.py``
The Aria-specific ``io/sensor_io.py`` reader is not carried by the owned fork;
``lamptrack.cameras.RigCamera`` supplies its small inference calibration seam.

Local changes: import paths, static typing, Google-style documentation, and
removal of Aria-only loaders and dead code. Model modules, parameter names,
state-dict keys, and numerical operations remain unchanged.
"""
