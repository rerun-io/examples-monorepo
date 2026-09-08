"""What the boundary and logging suites share: the synthetic rig, and the recording reader.

One 200x200 kb4 camera per index with every distortion coefficient zero, the
Index device's frozen noise model, and one blocky-noise scene that can be shifted
by whole pixels: enough to detect on, track through and refuse the wrong thing,
and small enough that the whole Python suite stays inside a few seconds.

Factories rather than values, because a test picks the camera count, the baseline
and the shift. Session-scoped, because a factory holds no per-test state and a
``@given`` test may not request a function-scoped fixture — each call still hands
back a fresh calibration or frontend, so a test may mutate what it is given.
Fixtures rather than a module the tests import from each other: pytest injects
these, so no test module has to be on another one's import path. That is also
where :func:`read_rows` belongs — the two logging suites check their rungs
against a real recording, and one reader means one account of what a row is,
and :func:`manifest` — five suites read the frozen reference set and parsing it
once a session is both cheaper and one account of what "the manifest" means.
"""

import pytest

from slam_rs.reference import ReferenceManifest, load_manifest


@pytest.fixture(scope="session")
def manifest() -> ReferenceManifest:
    """The frozen reference set, parsed once for the whole session."""
    return load_manifest()
