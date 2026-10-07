//! Runtime source identity and dependency fingerprint shared by metric reports.
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::{path::Path, process::Command};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    pub crate_version: String,
    pub source_sha: Option<String>,
    pub cargo_lock_sha256: Option<String>,
    pub brush: String,
}
impl Default for Provenance {
    fn default() -> Self {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let source_sha = std::env::var("GSPLAT_SOURCE_SHA")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .or_else(|| {
                let output = Command::new("git")
                    .current_dir(&root)
                    .args(["rev-parse", "HEAD"])
                    .output()
                    .ok()?;
                output
                    .status
                    .success()
                    .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
            });
        Self {
            crate_version: env!("CARGO_PKG_VERSION").into(),
            source_sha,
            cargo_lock_sha256: std::fs::read(root.join("Cargo.lock"))
                .ok()
                .map(|bytes| format!("{:x}", Sha256::digest(bytes))),
            brush:
                "Brush 1388f74c + packages/brush-src/patches/brush-1388f74c-process-observer.patch"
                    .into(),
        }
    }
}
