//! The reader on the real X4 Air clip. Asset-gated: `KORNIA_STAGING_INSV` names the file; run with
//! `cargo test -p kornia-staging-io --test insta360_clip -- --ignored`.
use kornia_staging_io::insta360::Insta360Clip;

#[test]
#[ignore = "needs the X4 Air clip VID_20261116_125200_00_003.insv (KORNIA_STAGING_INSV)"]
fn the_x4_air_clip_reads_whole() -> Result<(), Box<dyn std::error::Error>> {
    let Ok(path) = std::env::var("KORNIA_STAGING_INSV") else {
        eprintln!("SKIP: KORNIA_STAGING_INSV is unset (it names the X4 Air clip VID_20261116_125200_00_003.insv)");
        return Ok(());
    };
    if !std::path::Path::new(&path).is_file() {
        eprintln!("SKIP: asset {path} (VID_20261116_125200_00_003.insv) is absent");
        return Ok(());
    }
    let clip = Insta360Clip::open(&path)?;
    let m = &clip.trailer.metadata;
    assert_eq!(m.camera_type, "Insta360 X4 Air");
    assert_eq!(clip.tracks.len(), 2);
    for track in &clip.tracks {
        assert_eq!(track.samples.len(), 7724);
        assert_eq!(
            (
                track.width,
                track.height,
                track.nal_length_size,
                track.parameter_sets.len()
            ),
            (1920, 1920, 4, 3)
        );
        let sync: Vec<usize> = track
            .samples
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_sync)
            .map(|(i, _)| i)
            .collect();
        assert!(sync.len() == 258 && sync.iter().all(|i| i % 30 == 0));
    }
    assert!(clip.tracks[0]
        .samples
        .iter()
        .zip(&clip.tracks[1].samples)
        .all(|(a, b)| a.pts == b.pts));
    assert_eq!(clip.tracks[0].samples[7723].pts, 7723 * 1001);
    assert_eq!(clip.trailer.imu.len(), 257_600);
    let imu = &clip.trailer.imu;
    let rate = (imu.len() - 1) as f64 / ((imu[imu.len() - 1].t_us - imu[0].t_us) as f64 * 1e-6);
    assert!((rate - 997.2).abs() < 0.5, "{rate}");
    assert_eq!((clip.first_frame_entry, clip.frames()), (8, 7724));
    assert_eq!(
        clip.frame_exposures(false)[0].t_us,
        m.first_frame_timestamp_us
    );
    assert_eq!(
        (
            clip.frame_exposures(false).len(),
            clip.frame_exposures(true).len()
        ),
        (7724, 7724)
    );
    assert_eq!(clip.lens_track, [1, 0]);
    // Every IRAP sample converts with its parameter sets in front.
    let mut reader = clip.sample_reader()?;
    let mut out = Vec::new();
    assert!(reader.read_annex_b(0, 0, &mut out)?);
    assert_eq!(&out[..5], &[0, 0, 0, 1, 0x40]);
    out.clear();
    assert!(!reader.read_annex_b(1, 1, &mut out)?);
    Ok(())
}
