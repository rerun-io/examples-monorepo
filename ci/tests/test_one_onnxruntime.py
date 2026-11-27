"""ONNX Runtime guard: every environment resolves at most one ONNX Runtime build.

Two builds in one process share ``libonnxruntime_providers_shared.so``: both open it with ``RTLD_GLOBAL``, so the
first one loaded serves every CUDA provider, and the other runtime's provider calls into a foreign core. That is how
colsfm-dev segfaulted with conda ``onnxruntime-cpp 1.30.0`` (COLMAP) beside PyPI ``onnxruntime-gpu 1.29.0``.

The Python bindings statically link their own core, so "one build" cannot mean one file: the rule is that the conda
``onnxruntime`` (Python) and ``onnxruntime-cpp`` packages come from the same feedstock build (version, build number
and CUDA/CPU variant), and that nothing installs ONNX Runtime from PyPI beside them.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

REPO_ROOT: Path = Path(__file__).resolve().parents[2]

ORT_CONDA_NAMES: Final[frozenset[str]] = frozenset({"onnxruntime", "onnxruntime-cpp", "onnxruntime-novec", "onnxruntime-novec-cpp"})
CONDA_FILE: Final[re.Pattern[str]] = re.compile(r"^(?P<name>.+)-(?P<version>[^-]+)-(?P<build>[^-]+)\.(?:conda|tar\.bz2)$")
BUILD_STRING: Final[re.Pattern[str]] = re.compile(r"_(?P<number>\d+)_(?P<variant>cpu|cuda\d+)$")
PYPI_ORT_NAMES: Final[frozenset[str]] = frozenset(
    {"onnxruntime", "onnxruntime_gpu", "onnxruntime_directml", "onnxruntime_openvino", "onnxruntime_qnn", "onnxruntime_training"}
)
"""PyPI distributions that ship an ONNX Runtime core (not extensions such as ``onnxruntime_extensions``), wheel-normalized."""


@dataclass(frozen=True, slots=True)
class OrtBuild:
    """One conda ONNX Runtime package as the lock names it."""

    name: str
    """Package name, e.g. ``onnxruntime-cpp``."""
    version: str
    """Upstream version, e.g. ``1.30.0``."""
    number: str
    """Feedstock build number, e.g. ``201``."""
    variant: str
    """``cpu`` or ``cuda<NNN>``."""

    def same_build(self, other: "OrtBuild") -> bool:
        return (self.version, self.number, self.variant) == (other.version, other.number, other.variant)


def _locked_packages() -> dict[tuple[str, str], list[str]]:
    """(environment, lock platform) -> the conda/pypi locators listed for it."""
    lock_text: str = (REPO_ROOT / "pixi.lock").read_text()
    environments: str = lock_text.split("\nenvironments:\n", maxsplit=1)[1].split("\npackages:\n", maxsplit=1)[0]
    locked: dict[tuple[str, str], list[str]] = {}
    environment: str | None = None
    platform: str | None = None
    for line in environments.splitlines():
        env_match: re.Match[str] | None = re.fullmatch(r"  ([A-Za-z0-9_-]+):", line)
        platform_match: re.Match[str] | None = re.fullmatch(r"      ([a-z0-9-]+):", line)
        entry_match: re.Match[str] | None = re.fullmatch(r"      - (conda|pypi): (\S+)", line)
        if env_match:
            environment, platform = env_match.group(1), None
        elif platform_match and environment is not None:
            platform = platform_match.group(1)
            locked[(environment, platform)] = []
        elif entry_match and environment is not None and platform is not None:
            locked[(environment, platform)].append(f"{entry_match.group(1)}:{entry_match.group(2)}")
    return locked


def _ort_build(file_name: str) -> OrtBuild | None:
    conda_match: re.Match[str] | None = CONDA_FILE.match(file_name)
    if conda_match is None or conda_match.group("name") not in ORT_CONDA_NAMES:
        return None
    build_match: re.Match[str] | None = BUILD_STRING.search(conda_match.group("build"))
    assert build_match is not None, f"{file_name}: unexpected ONNX Runtime build string"
    return OrtBuild(conda_match.group("name"), conda_match.group("version"), build_match.group("number"), build_match.group("variant"))


def _problems(locators: list[str]) -> list[str]:
    """What breaks the one-build rule in one environment and platform."""
    problems: list[str] = []
    builds: list[OrtBuild] = []
    for locator in locators:
        kind, _, url = locator.partition(":")
        file_name: str = url.rsplit("/", maxsplit=1)[-1]
        if kind == "pypi" and file_name.split("-", maxsplit=1)[0].lower().replace(".", "_") in PYPI_ORT_NAMES:
            problems.append(f"ONNX Runtime from PyPI ({file_name}); take the conda build")
        elif kind == "conda" and (build := _ort_build(file_name)) is not None:
            builds.append(build)
    if len({build.name.removesuffix("-cpp") for build in builds}) > 1:
        problems.append("both the novec and the regular ONNX Runtime")
    if builds and not all(build.same_build(builds[0]) for build in builds):
        problems.append("different ONNX Runtime builds: " + ", ".join(f"{b.name} {b.version} #{b.number} {b.variant}" for b in builds))
    return problems


def test_every_environment_resolves_one_onnxruntime_build() -> None:
    locked: dict[tuple[str, str], list[str]] = _locked_packages()
    # A lock-format change must fail here, not silently scan nothing.
    assert any(_ort_build(locator.rsplit("/", maxsplit=1)[-1]) for locators in locked.values() for locator in locators), (
        "the lock scan found no conda ONNX Runtime package; the pixi.lock layout changed"
    )
    failing: dict[str, list[str]] = {}
    for (environment, platform), locators in sorted(locked.items()):
        problems: list[str] = _problems(locators)
        if problems:
            failing.setdefault(environment, []).extend(f"{platform}: {problem}" for problem in problems)
    assert not failing, "environments with more than one ONNX Runtime build:\n" + "\n".join(
        f"  {env}: {'; '.join(sorted(set(problems)))}" for env, problems in sorted(failing.items())
    )


def test_the_rule_rejects_the_colsfm_pair_and_accepts_one_feedstock_build() -> None:
    prefix: str = "conda:https://prefix.dev/conda-forge/linux-64/"
    cpp: str = prefix + "onnxruntime-cpp-1.30.0-h714bfea_201_cuda130.conda"
    python: str = prefix + "onnxruntime-1.30.0-py312h6654455_201_cuda130.conda"
    wheel: str = "pypi:https://files.pythonhosted.org/packages/87/7b/x/onnxruntime_gpu-1.29.0-cp312-cp312-manylinux_2_28_x86_64.whl"
    assert _problems([cpp, wheel]) == ["ONNX Runtime from PyPI (onnxruntime_gpu-1.29.0-cp312-cp312-manylinux_2_28_x86_64.whl); take the conda build"]
    assert _problems([cpp, prefix + "onnxruntime-1.30.0-py312h6654455_200_cuda130.conda"]) != []
    assert _problems([cpp, prefix + "onnxruntime-1.30.0-py312h16c82da_2_cpu.conda"]) != []
    assert _problems([cpp, python, "pypi:./packages/simplecv", prefix + "onnx-1.21.0-py312h1234567_0.conda"]) == []
    extensions: str = "pypi:https://files.pythonhosted.org/packages/1/2/x/onnxruntime_extensions-0.14.0-cp312-cp312-manylinux_2_28_x86_64.whl"
    assert _problems([cpp, python, extensions]) == [], "an extension package ships no ONNX Runtime core"
