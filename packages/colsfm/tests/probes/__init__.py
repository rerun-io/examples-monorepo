"""Runnable probes that establish what the pinned pycolmap offers the `colsfm` pipeline.

Every probe is self-contained, builds its own synthetic data, and asserts the
behaviour the bindings show. They run in the unit lane (`pixi run -e colsfm-dev tests`).

The probes enable beartype on themselves (the probe package is not in `beartype_packages`):
the point of the suite is to check what the bindings actually accept, so their
annotations are enforced whenever it runs.
"""

from beartype.claw import beartype_this_package

beartype_this_package()
