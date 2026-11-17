// The X4 Air clip on disk, shared through `include!` by `insta360_pano.rs` and kornia-staging-gpu's
// `tests/stitch.rs`.

const LENS: kornia_image::ImageSize = kornia_image::ImageSize {
    width: 1920,
    height: 1920,
};
/// The clip (`KORNIA_STAGING_INSV`) and an `ffmpeg`, or `None` with the missing asset named.
fn clip() -> Option<(String, kornia_staging_io::insta360::Insta360Clip)> {
    let Ok(path) = std::env::var("KORNIA_STAGING_INSV") else {
        eprintln!("SKIP: KORNIA_STAGING_INSV is unset (it names the X4 Air clip VID_20261116_125200_00_003.insv)");
        return None;
    };
    if !std::path::Path::new(&path).is_file() {
        eprintln!("SKIP: asset {path} (VID_20261116_125200_00_003.insv) is absent");
        return None;
    }
    if std::process::Command::new("ffmpeg")
        .arg("-version")
        .output()
        .is_err()
    {
        eprintln!("SKIP: no ffmpeg on PATH to decode the HEVC tracks");
        return None;
    }
    let clip = kornia_staging_io::insta360::Insta360Clip::open(&path).unwrap();
    assert_eq!(clip.lens_track, [1, 0], "lens 0 is track 1 (recon/09)");
    Some((path, clip))
}

/// Luma (plane 0, no range conversion) of frame `frame` of video stream `track`, decoded by `ffmpeg`.
fn decode(path: &str, track: usize, frame: usize) -> kornia_image::Image<u8, 1> {
    let t = format!("{:.6}", (frame as f64 * 1001.0 - 100.0) / 30000.0);
    let out = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-ss", &t, "-i", path])
        .args(["-map", &format!("0:v:{track}"), "-frames:v", "1"])
        .args(["-f", "rawvideo", "-pix_fmt", "yuvj420p", "-"])
        .output()
        .expect("ffmpeg runs");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    kornia_image::Image::new(LENS, out.stdout[..1920 * 1920].to_vec()).unwrap()
}
