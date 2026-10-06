//! Build-time source identity for both checkouts and exported source archives.
use std::{path::Path, process::Command};

pub fn git(root: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .current_dir(root)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8(output.stdout).ok()?.trim().to_owned())
}

pub fn source_sha(root: &Path, explicit: Option<&str>) -> String {
    explicit
        .filter(|value| !value.trim().is_empty())
        .map(str::to_owned)
        .or_else(|| git(root, &["describe", "--always", "--dirty"]))
        .unwrap_or_else(|| "unknown".into())
}
