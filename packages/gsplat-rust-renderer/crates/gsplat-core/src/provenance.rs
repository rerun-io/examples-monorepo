//! Runtime source identity and dependency fingerprint shared by metric reports (gsplat-cli, the canonical viewer's
//! viewer-probe).
use serde::Serialize;
use sha2::{Digest as _, Sha256};
use std::{path::Path, process::Command};

#[derive(Clone, Debug, Serialize)]
pub struct Provenance {
    pub crate_version: String,
    pub source_sha: Option<String>,
    pub cargo_lock_sha256: Option<String>,
    pub brush: String,
}

impl Provenance {
    /// The identity of the Cargo workspace at `root`: its git commit (or `GSPLAT_SOURCE_SHA`) and its Cargo.lock, and
    /// the Brush revision and patch named by `brush_recipe` (the text of `recipes/brush-src/recipe.yaml`, which the
    /// calling binary embeds).
    pub fn capture(root: &Path, brush_recipe: &str) -> Self {
        let source_sha = std::env::var("GSPLAT_SOURCE_SHA")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .or_else(|| {
                let output = Command::new("git")
                    .current_dir(root)
                    .args(["rev-parse", "HEAD"])
                    .output()
                    .ok()?;
                output
                    .status
                    .success()
                    .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
            });
        let revision = brush_recipe
            .lines()
            .find_map(|line| line.trim().strip_prefix("rev: "))
            .expect("Brush recipe revision");
        let patch = brush_recipe
            .lines()
            .find_map(|line| line.trim().strip_prefix("- patches/"))
            .expect("Brush observer patch");
        let patch = format!("patches/{patch}");
        Self {
            crate_version: env!("CARGO_PKG_VERSION").into(),
            source_sha,
            cargo_lock_sha256: std::fs::read(root.join("Cargo.lock"))
                .ok()
                .map(|bytes| format!("{:x}", Sha256::digest(bytes))),
            brush: format!("Brush {revision} + recipes/brush-src/{patch}"),
        }
    }
}
