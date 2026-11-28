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
# Rust: comments and string literals are not code; whitespace around path and generic punctuation carries no meaning.
COMMENTS_AND_STRINGS: re.Pattern[str] = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"', re.S)
TIGHT_PUNCTUATION: re.Pattern[str] = re.compile(r"\s*(::|[<>,()=])\s*")
# Cargo: the tables that declare dependencies, and the features that link a native or web viewer binary.
DEPENDENCY_TABLES: tuple[str, ...] = ("dependencies", "dev-dependencies", "build-dependencies")
VIEWER_FEATURES: dict[str, frozenset[str]] = {
    "rerun": frozenset({"native_viewer", "web_viewer"}),
    "re_viewer": frozenset({"default_web"}),
}


def _tracked(pattern: str) -> list[str]:
    listed: str = subprocess.run(["git", "ls-files", "-z", "--", pattern], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout
    return [name for name in listed.split("\0") if name]


def viewer_constructions(text: str) -> list[str]:
    """Every place a Rust source starts a viewer: Rerun's or eframe's native loop, `rerun::run`, or, in a source that names
    re_viewer, an App (`new`, `with_commands`, an egui_kittest Harness of App, under any `use ... as` or `type` alias)."""
    code: str = TIGHT_PUNCTUATION.sub(r"\1", " ".join(COMMENTS_AND_STRINGS.sub(" ", text).split()))
    patterns: list[str] = [r"\brun_native(?:_app)?\(", r"\brerun::run\("]
    if "re_viewer" in code:
        names: set[str] = {"App", *re.findall(r"\bApp as (\w+)", code), *re.findall(r"\btype (\w+)=(?:\w+::)*App\b", code)}
        app: str = "|".join(sorted(names))
        patterns += [rf"\b(?:{app})::(?:new|with_commands)\(", rf"\bHarness::<(?:\w+::)*(?:{app})\b"]
    return [match.group(0) for match in re.finditer("|".join(patterns), code)]


def viewer_features(manifest: dict[str, Any], workspace_packages: dict[str, str]) -> list[str]:
    """`crate/feature` for every viewer feature a Cargo manifest enables: on a dependency (renamed with `package = ...`,
    here or in its workspace) or through a `[features]` entry that forwards to one."""
    tables: list[dict[str, Any]] = [manifest.get(table, {}) for table in DEPENDENCY_TABLES]
    tables += [target.get(table, {}) for target in manifest.get("target", {}).values() for table in DEPENDENCY_TABLES]
    tables.append(manifest.get("workspace", {}).get("dependencies", {}))
    found: list[str] = []
    packages: dict[str, str] = {}
    for table in tables:
        for key, entry in table.items():
            spec: dict[str, Any] = entry if isinstance(entry, dict) else {}
            crate: str = spec.get("package") or (workspace_packages.get(key, key) if spec.get("workspace") else key)
            packages[key] = crate
            found += [f"{crate}/{feature}" for feature in spec.get("features", []) if feature in VIEWER_FEATURES.get(crate, ())]
    for entries in manifest.get("features", {}).values():
        for entry in entries:
            alias, _, feature = entry.partition("/")
            crate = packages.get(alias.removesuffix("?"), alias.removesuffix("?"))
            if feature in VIEWER_FEATURES.get(crate, ()):
                found.append(f"{crate}/{feature} (via {entry})")
    return found


def test_only_the_canonical_viewer_constructs_a_viewer() -> None:
    found: list[str] = [
        f"{name}: {hit}"
        for name in _tracked("*.rs")
        if not name.startswith(CANONICAL)
        for hit in viewer_constructions((REPO_ROOT / name).read_text(errors="replace"))
    ]
    assert not found, f"{RULE}: {found}"


def test_only_the_canonical_viewer_links_a_viewer_binary() -> None:
    manifests: dict[str, dict[str, Any]] = {name: tomllib.loads((REPO_ROOT / name).read_text()) for name in _tracked("*Cargo.toml")}
    # `workspace = true` entries take their package from the workspace manifest; a rename there applies to every member.
    workspace_packages: dict[str, str] = {
        key: entry["package"]
        for manifest in manifests.values()
        for key, entry in manifest.get("workspace", {}).get("dependencies", {}).items()
        if isinstance(entry, dict) and "package" in entry
    }
    found: list[str] = [
        f"{name}: {hit}"
        for name, manifest in manifests.items()
        if not name.startswith(CANONICAL)
        for hit in viewer_features(manifest, workspace_packages)
    ]
    assert not found, f"{RULE}: {found}"


def test_the_guard_sees_every_spelling_of_a_viewer() -> None:
    planted: str = """use re_viewer::{App as ViewerApp, external::eframe};
// A comment may name re_viewer::App::new(...) freely.
const HELP: &str = "App::new(";
fn start(cc: &eframe::CreationContext<'_>) {
    let one = re_viewer::App::new(token, info, env, options, cc, None, rt);
    let two = ViewerApp::with_commands(
        token, info, env, options, cc, None, rt, commands,
    );
    let harness = egui_kittest::Harness::<
        re_viewer::App,
    >::builder();
}
type Viewer = re_viewer::App;
fn three() { Viewer::new(token, info, env, options, cc, None, rt); eframe::run_native("x", options, creator); }
"""
    assert viewer_constructions(planted) == [
        "App::new(",
        "ViewerApp::with_commands(",
        "Harness::<re_viewer::App",
        "Viewer::new(",
        "run_native(",
    ]
    assert viewer_constructions("struct App;\nfn main() { App::new(); }\n") == [], "an App unrelated to re_viewer"
    manifest: dict[str, Any] = tomllib.loads(
        '[package.metadata.rerun]\nfeatures = ["native_viewer"]\n'
        '[dependencies]\nrerun = { workspace = true, features = ["sdk", "native_viewer"] }\n'
        'viewer = { package = "re_viewer", features = ["default_web"], optional = true }\n'
        "[target.'cfg(target_arch = \"wasm32\")'.dependencies]\nweb = { workspace = true }\n"
        '[features]\napp = ["viewer?/default_web", "web/web_viewer"]\n'
    )
    assert viewer_features(manifest, {"web": "rerun"}) == [
        "rerun/native_viewer",
        "re_viewer/default_web",
        "re_viewer/default_web (via viewer?/default_web)",
        "rerun/web_viewer (via web/web_viewer)",
    ]
