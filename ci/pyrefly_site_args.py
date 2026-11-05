"""Print `--site-package-path` arguments that put the running environment first, for the `typecheck` task.

`pyrefly.toml` lists every `*-dev` env's site-packages so a file can import what only a sibling env
installs, but pyrefly takes the first listed env that has a module: an env checked against another
env's copy of a dependency (pyserde, numpy) reports errors its own copy does not have. The CLI flag
replaces the config list, so this prints the env's own directories (its site-packages and the
directories its `.pth` files add, as pyrefly would query from the interpreter) followed by the
config list, without duplicates or paths that do not exist.
"""

import os
import site
import sys

import tomllib

REPO_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def own_dirs() -> list[str]:
    """Site-packages directories and `.pth` additions on this interpreter's `sys.path`, in order."""
    site_packages: set[str] = set(site.getsitepackages())
    stdlib: str = os.path.join(sys.base_prefix, "lib")
    return [path for path in sys.path if os.path.isdir(path) and (path in site_packages or not path.startswith(stdlib))]


def config_dirs() -> list[str]:
    """The `site-package-path` entries of `pyrefly.toml` that exist on this machine."""
    with open(os.path.join(REPO_ROOT, "pyrefly.toml"), "rb") as stream:
        entries: list[str] = tomllib.load(stream)["site-package-path"]
    return [path for path in (os.path.join(REPO_ROOT, entry) for entry in entries) if os.path.isdir(path)]


if __name__ == "__main__":
    ordered: list[str] = list(dict.fromkeys(os.path.realpath(path) for path in own_dirs() + config_dirs()))
    print(" ".join(f"--site-package-path {path}" for path in ordered))
