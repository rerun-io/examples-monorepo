"""One Rerun viewer: only packages/canonical-viewer builds a re_viewer App or a viewer binary.

Every other package reaches the viewer as an extension (`canonical_viewer_extension::Extension`, listed in
`canonical_viewer::EXTENSIONS`), so native, headless and web runs, tests and measurements all carry every extension and
every Rerun patch. A second App drifts: it misses the other extensions and the canonical device request, and its pictures
and timings stop matching what people open (AGENTS.md, "One Rerun viewer").
"""

import re
import subprocess
from pathlib import Path
from typing import Any

import tomllib

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
CANONICAL: str = "packages/canonical-viewer/"
RULE: str = (
    "Only packages/canonical-viewer may build a re_viewer App or a viewer binary; register an Extension in "
    "canonical_viewer::EXTENSIONS instead (AGENTS.md, 'One Rerun viewer')"
)
# What starts a viewer in Rust: an App value (`App::new`, an egui_kittest harness of App), Rerun's viewer entry points,
# or eframe's own native loop, in a file that uses re_viewer or the rerun crate's viewer.
VIEWER_CONSTRUCTION: re.Pattern[str] = re.compile(
    r"\bApp::new\s*\(|Harness::<\s*re_viewer::App\s*>|\bre_viewer::run_native_app\b|\brerun::run\s*\(|\beframe::run_native\s*\("
)
# Cargo features that link a viewer binary: the rerun crate's native and web viewers, re_viewer's web build.
VIEWER_FEATURES: dict[str, frozenset[str]] = {
    "rerun": frozenset({"native_viewer", "web_viewer"}),
    "re_viewer": frozenset({"default_web"}),
}


def _tracked(pattern: str) -> list[str]:
    listed: str = subprocess.run(["git", "ls-files", "-z", "--", pattern], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout
    return [name for name in listed.split("\0") if name and not name.startswith(CANONICAL)]


def viewer_constructions(name: str, text: str) -> list[str]:
    """`path:line` of every viewer construction in a Rust file that uses re_viewer or the rerun viewer."""
    if not re.search(r"\bre_viewer\b|\brerun::run\b", text):
        return []
    return [f"{name}:{number}" for number, line in enumerate(text.splitlines(), 1) if VIEWER_CONSTRUCTION.search(line)]


def viewer_features(name: str, manifest: dict[str, Any]) -> list[str]:
    """`path: crate/feature` for every viewer feature a Cargo manifest enables: in a dependency entry at any depth
    (`[dependencies]`, `[target.*.dependencies]`, `[workspace.dependencies]`, ...) or in a `[features]` list."""
    found: list[str] = []

    def walk(table: dict[str, Any]) -> None:
        for key, value in table.items():
            if not isinstance(value, dict):
                continue
            if key in VIEWER_FEATURES and isinstance(value.get("features"), list):
                found.extend(f"{name}: {key}/{feature}" for feature in value["features"] if feature in VIEWER_FEATURES[key])
            else:
                walk(value)

    walk(manifest)
    for entries in manifest.get("features", {}).values():
        for entry in entries:
            crate, _, feature = entry.partition("/")
            if feature in VIEWER_FEATURES.get(crate.removesuffix("?"), frozenset()):
                found.append(f"{name}: {entry}")
    return found


def test_only_the_canonical_viewer_constructs_a_viewer() -> None:
    found: list[str] = [hit for name in _tracked("*.rs") for hit in viewer_constructions(name, (REPO_ROOT / name).read_text(errors="replace"))]
    assert not found, f"{RULE}: {found}"


def test_only_the_canonical_viewer_links_a_viewer_binary() -> None:
    found: list[str] = [hit for name in _tracked("*Cargo.toml") for hit in viewer_features(name, tomllib.loads((REPO_ROOT / name).read_text()))]
    assert not found, f"{RULE}: {found}"


def test_the_guard_sees_a_viewer_outside_the_canonical_viewer() -> None:
    planted: str = "use re_viewer::App;\nfn main() {\n    let app = re_viewer::App::new(token, info, env, options, cc, None, rt);\n}\n"
    assert viewer_constructions("packages/demo/src/main.rs", planted) == ["packages/demo/src/main.rs:3"]
    manifest: dict[str, Any] = tomllib.loads(
        '[dependencies]\nrerun = { workspace = true, features = ["sdk", "native_viewer"] }\n'
        '[target.\'cfg(target_arch = "wasm32")\'.dependencies]\nre_viewer = { workspace = true, features = ["default_web"] }\n'
        '[features]\napp = ["rerun/web_viewer"]\n'
    )
    assert viewer_features("packages/demo/Cargo.toml", manifest) == [
        "packages/demo/Cargo.toml: rerun/native_viewer",
        "packages/demo/Cargo.toml: re_viewer/default_web",
        "packages/demo/Cargo.toml: rerun/web_viewer",
    ]
