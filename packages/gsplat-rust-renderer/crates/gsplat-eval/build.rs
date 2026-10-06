//! Embed the lockfile and checkout identity of this build, not the run directory.
mod build_support;
use std::{env, fs, path::PathBuf};
fn main() {
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("../..");
    let lock_path = root.join("Cargo.lock");
    println!("cargo:rerun-if-changed={}", lock_path.display());
    println!(
        "cargo:rerun-if-changed={}",
        root.join("Cargo.toml").display()
    );
    // Track all source changes, including dirty tracked files and new files.
    // Git's index/HEAD paths live in the shared worktree metadata directory.
    for path in ["src", "crates", "shader", "tests"] {
        println!("cargo:rerun-if-changed={}", root.join(path).display());
    }
    println!("cargo:rerun-if-env-changed=GSPLAT_SOURCE_SHA");
    println!("cargo:rerun-if-changed=build_support.rs");
    let git = |args: &[&str]| build_support::git(&root, args);
    for name in ["HEAD", "index"] {
        if let Some(path) = git(&["rev-parse", "--git-path", name]) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    if let Some(head) =
        git(&["rev-parse", "--git-path", "HEAD"]).and_then(|path| fs::read_to_string(path).ok())
        && let Some(branch) = head.trim().strip_prefix("ref: ")
        && let Some(path) = git(&["rev-parse", "--git-path", branch])
    {
        println!("cargo:rerun-if-changed={path}");
    }
    let lock: toml::Value =
        toml::from_str(&fs::read_to_string(lock_path).unwrap()).expect("parse Cargo.lock");
    let packages = lock["package"].as_array().unwrap();
    let mut rows = Vec::new();
    for name in [
        "brush-render",
        "burn",
        "cubecl",
        "wgpu",
        "wgpu-core",
        "naga",
        "re_renderer",
    ] {
        let found: Vec<_> = packages
            .iter()
            .filter(|p| p["name"].as_str() == Some(name))
            .collect();
        assert_eq!(found.len(), 1, "expected exactly one {name}");
        let p = found[0];
        rows.push(format!(
            "({name:?}, {:?}, {:?})",
            p["version"].as_str().unwrap(),
            p.get("source")
                .and_then(|s| s.as_str())
                .unwrap_or(if name == "brush-render" {
                    "Brush 1388f74c + packages/brush-src/patches/brush-1388f74c-process-observer.patch"
                } else {
                    "workspace"
                })
        ));
    }
    let manifest: toml::Value =
        toml::from_str(&fs::read_to_string(root.join("Cargo.toml")).unwrap())
            .expect("parse workspace manifest");
    let profile = env::var("PROFILE").unwrap();
    let code = format!(
        "pub const BUILD_GIT: &str = {:?};\npub const BUILD_PROFILE: &str = {:?};\npub const RELEASE_SETTINGS: &str = {:?};\npub const RESOLVED: &[(&str, &str, &str)] = &[{}];\n",
        build_support::source_sha(&root, env::var("GSPLAT_SOURCE_SHA").ok().as_deref()),
        profile,
        manifest["profile"]["release"].to_string(),
        rows.join(",")
    );
    fs::write(
        PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("versions.rs"),
        code,
    )
    .unwrap();
}
