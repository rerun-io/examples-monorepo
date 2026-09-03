"""Workspace guards: a sibling package named in a member's pyproject must resolve to the in-repo path, never to a PyPI namesake.

Pixi resolves each editable member's ``[project].dependencies`` through uv. A bare
sibling name is an ordinary registry requirement unless the member pins it with
``[tool.uv.sources]``, and PyPI does host unrelated packages called ``monopriors``,
``simplecv`` and ``sam3``. The first test checks the pins; the second checks the
lock, which is the only place a fallback would show up.
"""

import re
from pathlib import Path

import tomllib

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
PACKAGES_DIR: Path = REPO_ROOT / "packages"


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _workspace_packages() -> dict[str, Path]:
    """Normalized distribution name -> package directory, for every packages/*/pyproject.toml."""
    packages: dict[str, Path] = {}
    for pyproject in PACKAGES_DIR.glob("*/pyproject.toml"):
        name: str | None = tomllib.loads(pyproject.read_text()).get("project", {}).get("name")
        if name:
            packages[_normalize(name)] = pyproject.parent
    return packages


def _requirement_name(requirement: str) -> str:
    return _normalize(re.split(r"[<>=!~\[; @]", requirement, maxsplit=1)[0].strip())


def test_member_pyprojects_pin_workspace_siblings_with_uv_sources() -> None:
    packages: dict[str, Path] = _workspace_packages()
    problems: list[str] = []
    for name, package_dir in sorted(packages.items()):
        manifest: dict = tomllib.loads((package_dir / "pyproject.toml").read_text())
        sources: dict[str, dict] = {_normalize(k): v for k, v in manifest.get("tool", {}).get("uv", {}).get("sources", {}).items()}
        for requirement in manifest.get("project", {}).get("dependencies", []):
            sibling: str = _requirement_name(requirement)
            if sibling not in packages or sibling == name:
                continue
            source: dict | None = sources.get(sibling)
            if source is None or "path" not in source:
                problems.append(f"{package_dir.name}: `{sibling}` needs `[tool.uv.sources] {sibling} = {{ path = ..., editable = true }}`")
                continue
            if (package_dir / source["path"]).resolve() != packages[sibling].resolve():
                problems.append(f"{package_dir.name}: `{sibling}` source path {source['path']!r} does not point at packages/{packages[sibling].name}")
    assert not problems, "\n".join(problems)


def test_lock_never_resolves_a_workspace_package_from_an_index() -> None:
    packages: dict[str, Path] = _workspace_packages()
    lock_text: str = (REPO_ROOT / "pixi.lock").read_text()
    package_section: str = lock_text.split("\npackages:\n", maxsplit=1)[1]
    from_index: list[str] = []
    for entry in re.finditer(r"^- pypi: (\S+)\n((?:  .*\n)+)", package_section, re.M):
        locator: str = entry.group(1)
        name_match: re.Match[str] | None = re.search(r"^  name: (\S+)$", entry.group(2), re.M)
        if name_match and _normalize(name_match.group(1)) in packages and locator.startswith(("http://", "https://")):
            from_index.append(f"{name_match.group(1)} <- {locator}")
    assert not from_index, "workspace packages resolved from an index (pin them with [tool.uv.sources]):\n" + "\n".join(from_index)
