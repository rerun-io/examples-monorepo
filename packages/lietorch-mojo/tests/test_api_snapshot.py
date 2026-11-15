"""lietorch_mojo's public surface against a stored snapshot of upstream lietorch (no lietorch needed).

The snapshot `fixtures/lietorch_api.json` is written by `tests/api_surface.py`
from the compiled lietorch. Every upstream name must exist here with the same
kind, the same signature (parameter names, kinds, defaults) and, for class
constants, the same value. lietorch_mojo may add names; it may not drop or
change one.
"""

import inspect
import json

import pytest
from api_surface import FIXTURE, api_surface
from conftest import require_mojo_extension

require_mojo_extension()

from lietorch_mojo import backends  # noqa: E402

SNAPSHOT: dict = json.loads(FIXTURE.read_text())


def _diff_entry(where: str, upstream: dict, ours: dict | None) -> list[str]:
    if ours is None:
        return [f"{where}: missing"]
    if upstream.get("kind") != "class" or ours.get("kind") != "class":
        return [] if upstream == ours else [f"{where}: upstream {upstream} != ours {ours}"]
    problems: list[str] = []
    if upstream["bases"] != ours["bases"]:
        problems.append(f"{where}: bases {upstream['bases']} != {ours['bases']}")
    for name, member in upstream["members"].items():
        mine: dict | None = ours["members"].get(name)
        if mine != member:
            problems.append(f"{where}.{name}: upstream {member} != ours {mine}")
    return problems


@pytest.mark.parametrize("module", list(SNAPSHOT["modules"]))
def test_module_surface_matches_upstream_snapshot(module: str) -> None:
    ours: dict = api_surface("lietorch_mojo")[module]
    prefix: str = "lietorch" if module == "." else f"lietorch{module}"
    problems: list[str] = []
    for name, entry in SNAPSHOT["modules"][module].items():
        problems += _diff_entry(f"{prefix}.{name}", entry, ours.get(name))
    assert not problems, "\n".join(problems)


def test_backends_match_upstream_snapshot() -> None:
    """Every lietorch_backends function exists in lietorch_mojo.backends with the same argument count."""
    ours: dict[str, int] = {
        name: len(inspect.signature(getattr(backends, name)).parameters) for name in SNAPSHOT["backends"] if hasattr(backends, name)
    }
    assert ours == SNAPSHOT["backends"]


def test_group_constants_are_in_the_snapshot() -> None:
    """Guard on the snapshot itself: the constants the group kernels rely on are recorded."""
    groups: dict = SNAPSHOT["modules"][".groups"]
    expected: dict[str, tuple[int, int, int]] = {"SO3": (1, 3, 4), "RxSO3": (2, 4, 5), "SE3": (3, 6, 7), "Sim3": (4, 7, 8)}
    for name, (gid, manifold, embedded) in expected.items():
        members: dict = groups[name]["members"]
        assert members["group_id"]["value"] == gid
        assert members["manifold_dim"]["value"] == manifold
        assert members["embedded_dim"]["value"] == embedded
        assert len(members["id_elem"]["value"]["values"]) == embedded
