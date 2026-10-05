//! Read-only catalog/fixture contract check. Run explicitly with the two env vars below.
use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use super::{Clip, catalog, read_imu};

#[test]
#[ignore = "requires SLAM_RS_CATALOG_URL, SLAM_RS_TEST_FIXTURE, SLAM_RS_TEST_OUTPUT, and dav1d for G2"]
fn catalog_inputs_match_python_fixture() -> anyhow::Result<()> {
    let url = std::env::var("SLAM_RS_CATALOG_URL")?;
    let fixture = PathBuf::from(std::env::var("SLAM_RS_TEST_FIXTURE")?);
    let output = PathBuf::from(std::env::var("SLAM_RS_TEST_OUTPUT")?);
    std::fs::create_dir_all(&output)?;
    let reference: Clip =
        serde_json::from_str(&std::fs::read_to_string(fixture.join("clip.json"))?)?;
    let input = catalog::load(&url, &reference.segment_id, Some(reference.framesets))?;
    assert_eq!(input.clip.frame_t_ns, reference.frame_t_ns);
    assert_eq!(input.clip.resolution_wh, reference.resolution_wh);
    let expected_imu = read_imu(&fixture.join("imu.csv")).map_err(anyhow::Error::msg)?;
    let mut checked_imu = 0;
    let mut max_imu_difference = 0.0_f64;
    for row in &input.imu {
        // The loader keeps its trailing interpolation margin; replay stops at the
        // first sample after the last frame, just like the fixture writer.
        if row.t_ns > expected_imu.last().unwrap().t_ns {
            break;
        }
        let index = expected_imu
            .binary_search_by_key(&row.t_ns, |v| v.t_ns)
            .expect("IMU timestamp missing from fixture");
        let expected = &expected_imu[index];
        for (actual, expected) in row
            .gyro
            .iter()
            .chain(&row.accel)
            .zip(expected.gyro.iter().chain(&expected.accel))
        {
            max_imu_difference = max_imu_difference.max((actual - expected).abs());
        }
        checked_imu += 1;
    }
    assert_eq!(
        checked_imu,
        expected_imu.len(),
        "same initial IMU history and one-sample lead"
    );
    assert!(
        max_imu_difference < 1e-12,
        "IMU difference {max_imu_difference}"
    );
    std::fs::write(
        output.join("catalog-calib.json"),
        input.calibration.to_json_string()?,
    )?;

    let raw = fixture.join("frames.u8");
    let mut xz = None;
    let mut reader: Box<dyn Read> = if raw.exists() {
        Box::new(std::fs::File::open(raw)?)
    } else {
        let mut child = Command::new("xz")
            .args(["-T2", "-dc"])
            .arg(fixture.join("frames.u8.xz"))
            .stdout(Stdio::piped())
            .spawn()?;
        let reader = child.stdout.take().unwrap();
        xz = Some(child);
        Box::new(reader)
    };
    let mut counts = [0usize; 4];
    let mut sums = [0u64; 4];
    let mut maxima = [0u8; 4];
    let mut offset = 0;
    for _ in 0..input.clip.framesets {
        for (camera, &(width, height)) in input.clip.resolution_wh.iter().enumerate() {
            let actual = &input.pixels[offset..offset + width * height];
            let mut expected = vec![0; actual.len()];
            reader.read_exact(&mut expected)?;
            for (&a, &b) in actual.iter().zip(&expected) {
                let difference = a.abs_diff(b);
                counts[camera] += usize::from(difference != 0);
                sums[camera] += u64::from(difference);
                maxima[camera] = maxima[camera].max(difference);
            }
            offset += actual.len();
        }
    }
    if let Some(mut xz) = xz {
        assert!(xz.wait()?.success());
    }
    if reference.segment_id.starts_with("msd-g2__") {
        assert_eq!(counts, [0; 4], "G2 decoded pixels must match the Python fixture");
    }
    let per_camera: Vec<_> = (0..4)
        .map(|camera| {
            let (w, h) = input.clip.resolution_wh[camera];
            let pixels = w * h * input.clip.framesets;
            serde_json::json!({ "camera": camera, "pixels": pixels, "different": counts[camera],
            "mae": sums[camera] as f64 / pixels as f64, "max_abs": maxima[camera] })
        })
        .collect();
    let report = serde_json::json!({ "segment": reference.segment_id, "framesets": input.clip.framesets,
        "imu_checked": checked_imu, "max_imu_difference": max_imu_difference, "pixels": per_camera });
    std::fs::write(
        output.join("comparison.json"),
        serde_json::to_string_pretty(&report)?,
    )?;
    println!("{report}");
    Ok(())
}
