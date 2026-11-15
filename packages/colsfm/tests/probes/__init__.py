"""Runnable probes that establish what pycolmap 4.2.0 offers the `colsfm` pipeline.

Every probe is self-contained, builds its own synthetic data, and asserts the
behaviour that ``docs/spec/pycolmap-capabilities.md`` documents.  Run them with::

    pixi run -e colsfm colsfm-probe

beartype is enabled unconditionally here rather than behind ``PIXI_DEV_MODE``:
the whole point of the suite is to check what the bindings actually accept, so
the annotations must be enforced whenever it runs.
"""

from beartype.claw import beartype_this_package

beartype_this_package()
