"""One Rerun viewer: only packages/canonical-viewer builds a re_viewer App or a viewer binary.

Every other package reaches the viewer as an extension (`canonical_viewer_extension::Extension`, listed in
`canonical_viewer::EXTENSIONS`), so native, headless and web runs, tests and measurements all carry every extension and
every Rerun patch. A second App drifts: it misses the other extensions and the canonical device request, and its pictures
and timings stop matching what people open (AGENTS.md, "One Rerun viewer").
"""

import posixpath
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
# Rust lexical forms that are not code: raw strings (`r#"..."#`, `br"..."`) and char literals (`'"'`, `'\u{7f}'`).
# A quote followed by an identifier and no closing quote is a lifetime, which stays code.
RAW_STRING: re.Pattern[str] = re.compile(r'b?r(#*)"')
CHAR_LITERAL: re.Pattern[str] = re.compile(r"'(?:\\(?:u\{[0-9a-fA-F]+\}|x[0-9a-fA-F]{2}|.)|[^\\'\n])'")
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


def rust_code(text: str) -> str:
    """`text` as the compiler's tokens see it: comments (nested block comments too) and string, raw-string, byte and
    char literals become one space each, and whitespace collapses, tight around path and generic punctuation."""
    out: list[str] = []
    i: int = 0
    while i < len(text):
        raw: re.Match[str] | None = RAW_STRING.match(text, i) if i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_") else None
        char: re.Match[str] | None = CHAR_LITERAL.match(text, i)
        if text.startswith("//", i):
            end: int = text.find("\n", i)
            i = len(text) if end < 0 else end
        elif text.startswith("/*", i):
            depth, i = 1, i + 2
            while i < len(text) and depth:
                if text.startswith("/*", i):
                    depth, i = depth + 1, i + 2
                elif text.startswith("*/", i):
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
        elif raw:
            end = text.find('"' + raw.group(1), raw.end())
            i = len(text) if end < 0 else end + 1 + len(raw.group(1))
        elif text[i] == '"':
            i += 1
            while i < len(text) and text[i] != '"':
                i += 2 if text[i] == "\\" else 1
            i += 1
        elif char:
            i = char.end()
        else:
            out.append(text[i])
            i += 1
            continue
        out.append(" ")
    return TIGHT_PUNCTUATION.sub(r"\1", " ".join("".join(out).split()))


def viewer_constructions(text: str) -> list[str]:
    """Every place a Rust source starts a viewer: Rerun's or eframe's native loop, `rerun::run`, or, in a source that names
    re_viewer, an App (`new`, `with_commands`, an egui_kittest Harness of App, under any `use ... as` or `type` alias)."""
    if not re.search(r"re_viewer|run_native|rerun", text):
        return []
    code: str = rust_code(text)
    patterns: list[str] = [r"\brun_native(?:_app)?\(", r"\brerun::run\("]
    if "re_viewer" in code:
        names: set[str] = {"App", *re.findall(r"\bApp as (\w+)", code), *re.findall(r"\btype (\w+)=(?:\w+::)*App\b", code)}
        app: str = "|".join(sorted(names))
        patterns += [rf"\b(?:{app})::(?:new|with_commands)\(", rf"\bHarness::<(?:\w+::)*(?:{app})\b"]
    return [match.group(0) for match in re.finditer("|".join(patterns), code)]


def workspace_renames(name: str, manifests: dict[str, dict[str, Any]]) -> dict[str, str]:
    """`package = ...` renames in the `[workspace.dependencies]` of the workspace that owns manifest `name`: the one its
    `package.workspace` names, else the nearest manifest with a `[workspace]` table at or above it."""
    explicit: str | None = manifests[name].get("package", {}).get("workspace")
    directories: list[str] = [posixpath.join(posixpath.dirname(name), explicit)] if explicit else [posixpath.dirname(name)]
    while not explicit and directories[-1]:
        directories.append(posixpath.dirname(directories[-1]))
    for directory in directories:
        workspace: dict[str, Any] | None = manifests.get(posixpath.normpath(posixpath.join(directory, "Cargo.toml")), {}).get("workspace")
        if workspace is not None:
            return {
                key: entry["package"] for key, entry in workspace.get("dependencies", {}).items() if isinstance(entry, dict) and "package" in entry
            }
    return {}


def viewer_features(manifest: dict[str, Any], renames: dict[str, str]) -> list[str]:
    """`crate/feature` for every viewer feature a Cargo manifest enables: on a dependency (renamed with `package = ...`,
    here or, for `workspace = true`, in `renames`) or through a `[features]` entry that forwards to one."""
    tables: list[dict[str, Any]] = [manifest.get(table, {}) for table in DEPENDENCY_TABLES]
    tables += [target.get(table, {}) for target in manifest.get("target", {}).values() for table in DEPENDENCY_TABLES]
    tables.append(manifest.get("workspace", {}).get("dependencies", {}))
    found: list[str] = []
    packages: dict[str, str] = {}
    for table in tables:
        for key, entry in table.items():
            spec: dict[str, Any] = entry if isinstance(entry, dict) else {}
            crate: str = spec.get("package") or (renames.get(key, key) if spec.get("workspace") else key)
            packages[key] = crate
            found += [f"{crate}/{feature}" for feature in spec.get("features", []) if feature in VIEWER_FEATURES.get(crate, ())]
    for entries in manifest.get("features", {}).values():
        for entry in entries:
            alias, _, feature = entry.partition("/")
            crate = packages.get(alias.removesuffix("?"), alias.removesuffix("?"))
            if feature in VIEWER_FEATURES.get(crate, ()):
                found.append(f"{crate}/{feature} (via {entry})")
    return found


def manifest_violations(manifests: dict[str, dict[str, Any]]) -> list[str]:
    """`path: crate/feature` for every viewer feature enabled outside packages/canonical-viewer."""
    return [
        f"{name}: {hit}"
        for name, manifest in manifests.items()
        if not name.startswith(CANONICAL)
        for hit in viewer_features(manifest, workspace_renames(name, manifests))
    ]


def test_only_the_canonical_viewer_constructs_a_viewer() -> None:
    found: list[str] = [
        f"{name}: {hit}"
        for name in _tracked("*.rs")
        if not name.startswith(CANONICAL)
        for hit in viewer_constructions((REPO_ROOT / name).read_text(errors="replace"))
    ]
    assert not found, f"{RULE}: {found}"


def test_only_the_canonical_viewer_links_a_viewer_binary() -> None:
    found: list[str] = manifest_violations({name: tomllib.loads((REPO_ROOT / name).read_text()) for name in _tracked("*Cargo.toml")})
    assert not found, f"{RULE}: {found}"


def test_the_guard_reads_rust_as_code() -> None:
    planted: str = """use re_viewer::{App as ViewerApp, external::eframe};
fn start<'a>(cc: &'a eframe::CreationContext<'_>) {
    let quote = '"';
    let one = re_viewer::App::new(token, info, env, options, cc, None, rt);
    let label = "done";
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
    text_only: str = """use re_viewer::App;
// A line comment: App::new(
/* A block comment /* nested: App::new( */ still a comment: App::new( */
const A: &str = "App::new(";
const B: &str = r#"App::new(" and "#;
const C: &[u8] = br"App::new(";
const D: char = '\\'';
struct App;
"""
    assert viewer_constructions(text_only) == []
    assert viewer_constructions("struct App;\nfn main() { App::new(); }\n") == [], "an App unrelated to re_viewer"


def test_the_guard_reads_cargo_dependencies() -> None:
    member: dict[str, Any] = tomllib.loads(
        '[package]\nname = "demo"\n[package.metadata.rerun]\nfeatures = ["native_viewer"]\n'
        '[dependencies]\nrerun = { workspace = true, features = ["sdk", "native_viewer"] }\n'
        'viewer = { package = "re_viewer", features = ["default_web"], optional = true }\n'
        "[target.'cfg(target_arch = \"wasm32\")'.dependencies]\nweb = { workspace = true }\n"
        '[features]\napp = ["viewer?/default_web", "web/web_viewer"]\n'
    )
    workspace_a: dict[str, Any] = tomllib.loads('[workspace]\n[workspace.dependencies]\nweb = { package = "rerun", version = "1" }\n')
    workspace_b: dict[str, Any] = tomllib.loads('[workspace]\n[workspace.dependencies]\nweb = { package = "re_renderer", version = "1" }\n')
    manifests: dict[str, dict[str, Any]] = {
        "packages/a/Cargo.toml": workspace_a,
        "packages/a/crates/demo/Cargo.toml": member,
        "packages/b/Cargo.toml": workspace_b,
    }
    assert manifest_violations(manifests) == [
        "packages/a/crates/demo/Cargo.toml: rerun/native_viewer",
        "packages/a/crates/demo/Cargo.toml: re_viewer/default_web",
        "packages/a/crates/demo/Cargo.toml: re_viewer/default_web (via viewer?/default_web)",
        "packages/a/crates/demo/Cargo.toml: rerun/web_viewer (via web/web_viewer)",
    ], "workspace b's `web` rename must not shadow workspace a's"
    explicit: dict[str, Any] = {**member, "package": {"name": "demo", "workspace": "../../../b"}}
    assert workspace_renames("packages/a/crates/demo/Cargo.toml", {**manifests, "packages/a/crates/demo/Cargo.toml": explicit}) == {
        "web": "re_renderer"
    }
