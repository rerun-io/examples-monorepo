//! Build provenance shared by evaluation and benchmarking.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
include!(concat!(env!("OUT_DIR"), "/versions.rs"));
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DependencyVersion {
    pub version: String,
    pub source: String,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Versions {
    pub evaluator: String,
    pub ours: String,
    pub profile: String,
    pub release_settings: String,
    pub dependencies: BTreeMap<String, DependencyVersion>,
    pub environment: BTreeMap<String, Option<String>>,
}
impl Default for Versions {
    fn default() -> Self {
        Self {
            evaluator: env!("CARGO_PKG_VERSION").into(),
            ours: BUILD_GIT.into(),
            profile: BUILD_PROFILE.into(),
            release_settings: RELEASE_SETTINGS.into(),
            dependencies: RESOLVED
                .iter()
                .map(|(name, version, source)| {
                    (
                        (*name).into(),
                        DependencyVersion {
                            version: (*version).into(),
                            source: (*source).into(),
                        },
                    )
                })
                .collect(),
            environment: ["BURN_DEVICE", "CUBECL_WGPU_MAX_TASKS", "WGPU_BACKEND"]
                .into_iter()
                .map(|name| (name.into(), std::env::var(name).ok()))
                .collect(),
        }
    }
}
