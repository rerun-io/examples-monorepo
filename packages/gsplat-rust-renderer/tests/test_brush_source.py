"""The Pixi source package and activation link preserve one patched Brush tree."""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
from serde import serde
from serde.yaml import from_yaml


@serde
@dataclass(frozen=True, slots=True)
class Source:
    """Source fields owned by the rattler-build recipe schema."""

    git: str
    rev: str
    patches: list[str]


@serde
@dataclass(frozen=True, slots=True)
class Recipe:
    """The recipe's pinned source; the build backend owns its remaining fields."""

    source: Source


def test_brush_recipe_revision() -> None:
    """The observer patch is applied to the reviewed upstream revision."""
    source = Path(__file__).resolve().parents[2] / "brush-src"
    recipe = from_yaml(Recipe, (source / "recipe.yaml").read_text())
    assert recipe.source.git == "https://github.com/ArthurBrussee/brush.git"
    assert recipe.source.rev == "1388f74c6fe0236f68ee4915564bf00e9d2e3747"
    assert recipe.source.patches == ["patches/brush-1388f74c-process-observer.patch"]
    assert (source / recipe.source.patches[0]).is_file()


def test_activation_preserves_a_live_link_and_repairs_a_dangling_link(tmp_path: Path) -> None:
    """Switching prod/dev environments must not change Cargo's source identity."""
    script = Path(__file__).resolve().parents[1] / "activate.sh"
    prefixes = [tmp_path / "prod", tmp_path / "dev"]
    for prefix in prefixes:
        (prefix / "share/brush-src").mkdir(parents=True)
    link = tmp_path / "packages/gsplat-rust-renderer/target/brush-src"
    for prefix in prefixes:
        result = subprocess.run(
            ["sh", str(script)],
            env=dict(os.environ, PIXI_PROJECT_ROOT=str(tmp_path), CONDA_PREFIX=str(prefix)),
            check=True, capture_output=True, text=True,
        )
        assert result.stdout == result.stderr == ""
        assert link.resolve() == prefixes[0] / "share/brush-src"
    (prefixes[0] / "share/brush-src").rmdir()
    subprocess.run(
        ["sh", str(script)],
        env=dict(os.environ, PIXI_PROJECT_ROOT=str(tmp_path), CONDA_PREFIX=str(prefixes[1])),
        check=True,
    )
    assert link.resolve() == prefixes[1] / "share/brush-src"


@pytest.mark.integration
def test_installed_brush_contains_the_reversible_observer_patch(tmp_path: Path) -> None:
    """The packaged files carry precisely the context expected by the shipped patch."""
    root = Path(__file__).resolve().parents[1]
    tree = root / "target/brush-src"
    if not tree.is_dir():
        pytest.skip("missing installed Brush source; activate a gsplat Pixi environment")
    for excluded in (".git", "apps", ".github", ".zed", "build_env.sh", "conda_build.sh", "conda_build.bat", ".source_info.json", ".ok"):
        assert not (tree / excluded).exists()
    for name in ("message.rs", "train_stream.rs"):
        path = Path("crates/brush-process/src") / name
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_bytes((tree / path).read_bytes())
    patch = root.parent / "brush-src/patches/brush-1388f74c-process-observer.patch"
    environment = dict(os.environ, GIT_CEILING_DIRECTORIES=str(tmp_path.parent))
    subprocess.run(["git", "apply", "--reverse", str(patch)], cwd=tmp_path, env=environment, check=True)
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=tmp_path, env=environment, check=True)
