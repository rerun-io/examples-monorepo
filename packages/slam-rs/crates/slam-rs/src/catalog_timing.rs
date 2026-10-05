//! Frameset association and IMU pairing shared by catalog readers.
use crate::calib::catalog::CatalogError;

/// Camera selection, decoding and clock rules for a supported catalog rig.
#[derive(Debug, Clone, Copy)]
pub struct RigProfile {
    pub camera_names: [&'static str; 4],
    pub rig_cameras: usize,
    pub codec: [u8; 4],
    pub tolerance_ns: i64,
    pub nominal_fps: i64,
    pub downscale: u32,
    pub interpolate_accel: bool,
    pub video_time_is_absolute: bool,
}

pub const ROBOCAP: RigProfile = RigProfile {
    camera_names: ["left_front", "right_front", "left", "right"],
    rig_cameras: 6,
    codec: *b"avc1",
    tolerance_ns: 1_000_000,
    nominal_fps: 30,
    downscale: 3,
    interpolate_accel: true,
    video_time_is_absolute: true,
};
pub const MSD_G2: RigProfile = RigProfile {
    camera_names: ["cam0", "cam1", "cam2", "cam3"],
    rig_cameras: 4,
    codec: *b"av01",
    tolerance_ns: 0,
    nominal_fps: 54,
    downscale: 1,
    interpolate_accel: false,
    video_time_is_absolute: false,
};

/// One gyroscope sample with acceleration on the same timestamp.
#[derive(Debug, PartialEq)]
pub struct ImuRow {
    pub t_ns: i64,
    pub gyro: [f64; 3],
    pub accel: [f64; 3],
}

/// A median timestamp and the selected frame indices in camera order.
pub type Frameset = (i64, Vec<usize>);

/// Nearest unused image, ties to the later image; retain future images on failure.
pub fn frame_nearest_anchor(
    times: &[i64],
    cursor: usize,
    anchor: i64,
    tolerance: i64,
) -> (Option<usize>, usize) {
    let mut index = cursor;
    if index >= times.len() {
        return (None, cursor);
    }
    while index + 1 < times.len()
        && times[index + 1].abs_diff(anchor) <= times[index].abs_diff(anchor)
    {
        index += 1;
    }
    if times[index].abs_diff(anchor) > tolerance as u64 {
        return (
            None,
            if times[index] < anchor {
                index + 1
            } else {
                cursor
            },
        );
    }
    (Some(index), cursor)
}

/// Match every camera to camera zero, consuming complete groups once and using their integer median timestamp.
pub fn match_framesets(cameras: &[&[i64]], tolerance: i64) -> Result<Vec<Frameset>, CatalogError> {
    if cameras.is_empty() {
        return Err(CatalogError("a frameset needs at least one camera".into()));
    }
    for (index, times) in cameras.iter().enumerate() {
        if times.is_empty() {
            return Err(CatalogError(format!(
                "camera {index} has no frames, so it is not part of this recording"
            )));
        }
    }
    let overlap_start = cameras.iter().map(|t| t[0]).max().unwrap_or(0);
    let overlap_end = cameras.iter().map(|t| t[t.len() - 1]).min().unwrap_or(0);
    let mut cursors = vec![0; cameras.len()];
    let mut sets: Vec<Frameset> = Vec::new();
    let (mut interior, mut drops) = (0usize, 0usize);
    for (anchor_index, &anchor) in cameras[0].iter().enumerate() {
        let inside = (overlap_start..=overlap_end).contains(&anchor);
        interior += usize::from(inside);
        let mut selected = vec![anchor_index];
        for camera in 1..cameras.len() {
            let (index, cursor) =
                frame_nearest_anchor(cameras[camera], cursors[camera], anchor, tolerance);
            cursors[camera] = cursor;
            let Some(index) = index else {
                break;
            };
            selected.push(index);
        }
        if selected.len() != cameras.len() {
            drops += usize::from(inside);
            continue;
        }
        for camera in 1..cameras.len() {
            cursors[camera] = selected[camera] + 1;
        }
        let mut times: Vec<i64> = cameras
            .iter()
            .zip(&selected)
            .map(|(times, &index)| times[index])
            .collect();
        times.sort();
        let middle = times.len() / 2;
        let t = if times.len() % 2 == 1 {
            times[middle]
        } else {
            times[middle - 1] + (times[middle] - times[middle - 1]) / 2
        };
        if let Some(previous) = sets.last().filter(|s| s.0 >= t) {
            return Err(CatalogError(format!(
                "frameset timestamps are not strictly increasing: {t} follows {}",
                previous.0
            )));
        }
        sets.push((t, selected));
    }
    let allowed = interior.div_ceil(1000).max(1);
    if drops > allowed {
        return Err(CatalogError(format!(
            "{drops} of {interior} interior framesets are incomplete, more than the {allowed} basalt allows: the cameras are not one recording within {tolerance} ns"
        )));
    }
    Ok(sets)
}

/// Require identical clocks, or interpolate acceleration within its coverage, retaining the first duplicate sample.
pub fn pair_imu(
    mut gyro: Vec<(i64, [f64; 3])>,
    mut accel: Vec<(i64, [f64; 3])>,
    interpolate: bool,
) -> Result<Vec<ImuRow>, CatalogError> {
    gyro.sort_by_key(|v| v.0);
    accel.sort_by_key(|v| v.0);
    if !gyro.windows(2).all(|w| w[0].0 < w[1].0) {
        return Err(CatalogError(
            "IMU gyro timestamps are not strictly increasing".into(),
        ));
    }
    if !interpolate {
        if !gyro.iter().map(|v| v.0).eq(accel.iter().map(|v| v.0)) {
            return Err(CatalogError(format!(
                "{} gyro and {} accel samples are not on identical timestamps; pair them before feeding",
                gyro.len(),
                accel.len()
            )));
        }
        return Ok(gyro
            .into_iter()
            .zip(accel)
            .map(|((t_ns, gyro), (_, accel))| ImuRow { t_ns, gyro, accel })
            .collect());
    }
    accel.dedup_by_key(|v| v.0);
    if gyro.is_empty() || accel.len() < 2 {
        return Err(CatalogError(format!(
            "pairing needs a gyroscope sample and two accelerometer samples to interpolate between; got {} gyro and {} accel samples",
            gyro.len(),
            accel.len()
        )));
    }
    let gyro_start = gyro[0].0;
    let gyro_end = gyro[gyro.len() - 1].0;
    let mut cursor = 0;
    let mut rows = Vec::new();
    for (t_ns, gyro) in gyro {
        if t_ns < accel[0].0 || t_ns > accel[accel.len() - 1].0 {
            continue;
        }
        while cursor + 1 < accel.len() - 1 && accel[cursor + 1].0 <= t_ns {
            cursor += 1;
        }
        let (ta, a) = accel[cursor];
        let (tb, b) = accel[cursor + 1];
        // Match numpy.interp's evaluation order and exact endpoint values. The
        // Python feed has always used it; both readers must now give the same bits.
        let accel = if t_ns == ta {
            a
        } else if t_ns == tb {
            b
        } else {
            std::array::from_fn(|axis| {
                let slope = (b[axis] - a[axis]) / (tb - ta) as f64;
                slope * (t_ns - ta) as f64 + a[axis]
            })
        };
        rows.push(ImuRow { t_ns, gyro, accel });
    }
    if rows.is_empty() {
        return Err(CatalogError(format!(
            "the two inertial channels do not overlap, so nothing pairs: the gyroscope spans {gyro_start}..{gyro_end} ns and the accelerometer {}..{} ns",
            accel[0].0,
            accel[accel.len() - 1].0
        )));
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_unused_frames_use_integer_medians() -> Result<(), CatalogError> {
        assert_eq!(
            match_framesets(
                &[
                    &[100, 200, 300],
                    &[99, 202, 300],
                    &[101, 201, 302],
                    &[100, 199, 301]
                ],
                2
            )?,
            vec![(100, vec![0; 4]), (200, vec![1; 4]), (300, vec![2; 4])]
        );
        assert_eq!(
            match_framesets(&[&[100], &[90, 110]], 20)?,
            vec![(105, vec![0, 1])]
        );
        Ok(())
    }

    #[test]
    fn accel_pairing_keeps_first_duplicates_and_drops_uncovered_gyro() -> Result<(), CatalogError> {
        let gyro = vec![
            (0, [1.0; 3]),
            (10, [2.0; 3]),
            (20, [3.0; 3]),
            (30, [4.0; 3]),
            (40, [5.0; 3]),
        ];
        let accel = vec![
            (10, [2.0, 4.0, 6.0]),
            (10, [90.0; 3]),
            (30, [6.0, 8.0, 10.0]),
        ];
        let paired = pair_imu(gyro, accel, true)?;
        assert_eq!(
            paired.iter().map(|s| s.t_ns).collect::<Vec<_>>(),
            [10, 20, 30]
        );
        assert_eq!(paired[0].accel, [2.0, 4.0, 6.0]);
        assert_eq!(paired[1].accel, [4.0, 6.0, 8.0]);
        assert_eq!(paired[1].gyro, [3.0; 3]);
        assert_eq!(paired[2].accel, [6.0, 8.0, 10.0]);
        Ok(())
    }

    #[test]
    fn interpolation_preserves_the_python_feeds_float64_rounding() -> Result<(), CatalogError> {
        let rows = pair_imu(
            vec![(2, [0.0; 3]), (3, [0.0; 3])],
            vec![(0, [0.1; 3]), (3, [0.8; 3])],
            true,
        )?;
        // numpy.interp([2, 3], [0, 3], [0.1, 0.8]); divide before
        // multiplying by the timestamp delta, and preserve exact endpoints.
        assert_eq!(rows[0].accel[0].to_bits(), 0x3fe2_2222_2222_2223);
        assert_eq!(rows[1].accel, [0.8; 3]);
        Ok(())
    }

    #[test]
    fn synced_channels_require_the_same_timestamps() {
        assert!(pair_imu(vec![(1, [0.0; 3])], vec![(2, [0.0; 3])], false).is_err());
    }
}
