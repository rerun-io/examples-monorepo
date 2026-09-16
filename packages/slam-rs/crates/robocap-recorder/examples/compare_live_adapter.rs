//! Compare identical calibrated inputs through the live adapter and direct Vio.
use anyhow::{Result, ensure};
use robocap_recorder::{
    ImuChannel, LiveSlam, LiveSlamOptions, SlamInput, SlamReport, fast_profile,
};
use serde::Deserialize;
use slam_rs::{
    ImageView, Vio, calib::Calibration, config::VioConfig, frontend::flow::FrontendOptions,
};
use std::{fs, path::Path};

#[derive(Deserialize)]
struct Clip {
    frame_t_ns: Vec<i64>,
    num_cameras: usize,
    resolution_wh: Vec<[usize; 2]>,
}
struct Imu {
    t: i64,
    gyro: [f64; 3],
    accel: [f64; 3],
}

fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    ensure!(
        args.len() == 3,
        "usage: compare_live_adapter CLIP_DIRECTORY CONFIG_JSON OUTPUT_CSV"
    );
    let directory = Path::new(&args[0]);
    let clip: Clip = serde_json::from_str(&fs::read_to_string(directory.join("clip.json"))?)?;
    ensure!(clip.num_cameras == 4, "comparison needs four cameras");
    let calibration =
        Calibration::<f64>::from_json_str(&fs::read_to_string(directory.join("calib.json"))?)?;
    let mut config = VioConfig::from_json_str(&fs::read_to_string(&args[1])?)?;
    fast_profile(&mut config);
    let mut direct = Vio::<f32>::with_backend(
        config.clone(),
        calibration.clone(),
        FrontendOptions {
            threads: 4,
            ..Default::default()
        },
        slam_rs::Backend::Cpu,
    )?;
    let mut live = LiveSlam::with_configuration(calibration, config, LiveSlamOptions::default())?;
    let imu = fs::read_to_string(directory.join("imu.csv"))?
        .lines()
        .filter(|line| !line.starts_with('#') && !line.is_empty())
        .map(|line| -> Result<Imu> {
            let fields = line.split(',').collect::<Vec<_>>();
            ensure!(fields.len() == 7, "invalid IMU row");
            Ok(Imu {
                t: fields[0].parse()?,
                gyro: [fields[1].parse()?, fields[2].parse()?, fields[3].parse()?],
                accel: [fields[4].parse()?, fields[5].parse()?, fields[6].parse()?],
            })
        })
        .collect::<Result<Vec<_>>>()?;
    for row in &imu {
        direct.push_imu(row.t, row.gyro, row.accel)?;
    }
    let images = (0..clip.frame_t_ns.len())
        .map(|frame| {
            (0..4)
                .map(|camera| -> Result<Vec<u8>> {
                    let bytes =
                        fs::read(directory.join(format!("frame_{frame:03}_cam{camera}.pgm")))?;
                    let [width, height] = clip.resolution_wh[camera];
                    let header = format!("P5\n{width} {height}\n255\n");
                    ensure!(
                        bytes.starts_with(header.as_bytes())
                            && bytes.len() == header.len() + width * height,
                        "invalid dumped image"
                    );
                    Ok(bytes[header.len()..].to_vec())
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let mut reports = Vec::<SlamReport>::new();
    let mut cursor = 0;
    for (frame, &t) in clip.frame_t_ns.iter().enumerate() {
        // Deliver both original sensor channels before this frame, including
        // the first later gyro and its accelerometer interpolation bracket.
        while cursor < imu.len() && imu[cursor].t <= t + 3_000_000 {
            let row = &imu[cursor];
            for (channel, xyz) in [(ImuChannel::Gyro, row.gyro), (ImuChannel::Accel, row.accel)] {
                if let Some(report) = live.push(
                    SlamInput::Imu {
                        channel,
                        timestamp_ns: row.t,
                        xyz,
                    },
                    t + 3_000_000,
                )? {
                    reports.push(report);
                }
            }
            cursor += 1;
        }
        for (camera, pixels) in images[frame].iter().enumerate() {
            if let Some(report) = live.push(
                SlamInput::Frame {
                    camera,
                    timestamp_ns: t,
                    pixels: pixels.clone(),
                },
                t + 3_000_000,
            )? {
                reports.push(report);
            }
        }
    }
    let mut compared = 0;
    let mut max_difference = 0.0_f64;
    let mut csv = String::from("#timestamp [ns], p_x, p_y, p_z, q_w, q_x, q_y, q_z\n");
    for report in &reports {
        let frame = clip
            .frame_t_ns
            .binary_search(&report.timestamp_ns)
            .map_err(|_| anyhow::anyhow!("adapter changed the original frame time"))?;
        let views = images[frame]
            .iter()
            .enumerate()
            .map(|(camera, pixels)| {
                let [width, height] = clip.resolution_wh[camera];
                ImageView {
                    width,
                    height,
                    stride: width,
                    data: pixels,
                }
            })
            .collect::<Vec<_>>();
        let result = direct.track(report.timestamp_ns, &views)?;
        if let Some(pose) = report.pose {
            let difference = pose
                .iter()
                .zip(result.world_from_rig)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            max_difference = max_difference.max(difference);
            ensure!(
                difference < 1e-5,
                "first disagreement at frame {frame}, time {}: live={pose:?}, direct={:?}, difference={difference}",
                report.timestamp_ns,
                result.world_from_rig
            );
            compared += 1;
            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{}\n",
                report.timestamp_ns, pose[0], pose[1], pose[2], pose[6], pose[3], pose[4], pose[5]
            ));
        }
    }
    ensure!(compared > 20, "too few supported poses: {compared}");
    fs::write(&args[2], csv)?;
    println!(
        "reports={} compared={compared} max_pose_component_difference={max_difference}",
        reports.len()
    );
    Ok(())
}
