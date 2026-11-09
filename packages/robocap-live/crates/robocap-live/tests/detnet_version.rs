//! One setting picks the DetNet version: its folder in the models bundle and the camera turn its weights expect.

use std::path::{Path, PathBuf};

use robocap_live::hands::DetNetInput;
use robocap_live::nets::DetNetVersion;

#[test]
fn the_default_is_mix1_upright_and_names_round_trip() {
    assert_eq!(DetNetVersion::default(), DetNetVersion::Mix1Upright);
    assert_eq!(DetNetVersion::ALL.map(DetNetVersion::name), ["mix1-upright", "0929"]);
    for version in DetNetVersion::ALL {
        assert_eq!(DetNetVersion::from_name(version.name()), Some(version));
    }
    assert_eq!(DetNetVersion::from_name("upright"), None);
}

#[test]
fn the_camera_turn_follows_the_version() {
    let turned = |version: DetNetVersion| -> Vec<bool> { (0..6).map(|camera| DetNetInput::from(version).half_turned(camera)).collect() };
    assert_eq!(turned(DetNetVersion::Mix1Upright), [false, false, true, true, false, false]);
    assert_eq!(turned(DetNetVersion::V0929), [false; 6]);
}

#[test]
fn a_version_reads_its_folder_and_a_flat_bundle_falls_back_to_the_root() -> std::io::Result<()> {
    let root: PathBuf = std::env::temp_dir().join(format!("robocap-live-detnet-version-{}", std::process::id()));
    std::fs::create_dir_all(root.join("detnet/0929"))?;
    assert_eq!(DetNetVersion::V0929.dir(&root), root.join("detnet/0929"));
    assert_eq!(DetNetVersion::Mix1Upright.dir(&root), root, "no detnet/mix1-upright folder: the flat layout");
    assert_eq!(DetNetVersion::V0929.dir(Path::new("/nonexistent")), Path::new("/nonexistent"));
    std::fs::remove_dir_all(&root)
}
