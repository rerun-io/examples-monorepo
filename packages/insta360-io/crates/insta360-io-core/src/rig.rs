//! The calibrated rig of one clip: two lens cameras posed in the IMU frame, frame and IMU times on one clock, the
//! per-frame exposure, and the lens tracks cut into GOPs.
//!
//! Rig frame = IMU frame. Lens `k` is posed as `T_imu_lens = T_imu_P * T_P_lens`: `T_P_lens` is the factory rotation
//! and centre in the vendor rig frame P (V6 lens string when the clip has one, else V3), `T_imu_P = (R_imu_P,
//! R_imu_P * lever_arm)` comes from the device profile. Frame `i` is stamped at its effective time on the IMU clock:
//! the exposure-record-4 stamp plus the profile's `slope * shutter_i + constant` (mid-exposure of the middle row).
//! IMU times are the trailer's, without the vendor's `gyro_timestamp` subtraction: the profiles were fitted on that
//! clock. Every time is in nanoseconds on the trailer clock.
use crate::profile::{
    builtin_profiles, lens_type, select_profile, CaptureMode, DeviceProfile, ProfileError,
};
use kornia_staging_3d::camera::formats::insta360::FactoryLens;
use kornia_staging_io::insta360::{Insta360Clip, Insta360Error};
use nalgebra::{Isometry3, Matrix3, Rotation3, Translation3, UnitQuaternion};
use std::ops::Range;
use std::path::Path;

/// Why a rig could not be built from a clip.
#[derive(Debug, thiserror::Error)]
pub enum RigError {
    /// The file is not a readable two-lens clip.
    #[error(transparent)]
    Clip(#[from] Insta360Error),
    /// No device profile fits the clip.
    #[error(transparent)]
    Profile(#[from] ProfileError),
    /// The clip's records do not support a rig.
    #[error("{0}")]
    Invalid(String),
}

/// A clip read and checked, with its device profile applied.
#[derive(Clone, Debug)]
pub struct InsvRig {
    /// The clip.
    pub clip: Insta360Clip,
    /// The profile the clip's mode selected.
    pub profile: DeviceProfile,
    /// What the clip states about its mode.
    pub mode: CaptureMode,
    /// Rolling-shutter readout of each lens image: the clip's own when it states one, else the profile's.
    pub readout_s: f64,
    /// Lens 0 and lens 1: V6 when the clip has a V6 string, else V3.
    pub lenses: [FactoryLens; 2],
    /// `T_imu_lens` of lens 0 and lens 1.
    pub imu_from_lens: [Isometry3<f64>; 2],
    /// `R_imu_P`, the profile's measured rotation orthonormalised.
    pub imu_from_pano: Matrix3<f64>,
    /// Frame times, ns (frames `0..frames`).
    pub frame_ns: Vec<i64>,
    /// Exposure time per frame of lens 0 (record 4) and lens 1 (record 12; record 4 without it).
    pub exposure_s: [Vec<f64>; 2],
    /// IMU sample times, ns.
    pub imu_ns: Vec<i64>,
    /// Specific force, m/s^2, IMU axes.
    pub accel: Vec<[f64; 3]>,
    /// Angular rate, rad/s, IMU axes.
    pub gyro: Vec<[f64; 3]>,
}

impl InsvRig {
    /// Open `path` with the built-in profiles, keeping the first `frame_limit` frames (and the IMU up to 1 s past the
    /// last of them) when given.
    ///
    /// # Errors
    /// A malformed clip, fewer than two frames, no fitting profile, or frame times the profile's model does not keep
    /// increasing.
    pub fn open(path: impl AsRef<Path>, frame_limit: Option<usize>) -> Result<Self, RigError> {
        Self::from_clip(Insta360Clip::open(path)?, &builtin_profiles(), frame_limit)
    }

    /// The rig of an open clip under `profiles`.
    ///
    /// # Errors
    /// As [`InsvRig::open`].
    pub fn from_clip(
        clip: Insta360Clip,
        profiles: &[DeviceProfile],
        frame_limit: Option<usize>,
    ) -> Result<Self, RigError> {
        let vin0 = clip.frame_exposures(false);
        if vin0.len() < 2 {
            return Err(RigError::Invalid(format!(
                "{} frames; a clip needs two or more",
                vin0.len()
            )));
        }
        let mut intervals: Vec<i64> = vin0.windows(2).map(|w| w[1].t_us - w[0].t_us).collect();
        intervals.sort_unstable();
        let m = &clip.trailer.metadata;
        let v3 = m
            .offset_v3
            .as_deref()
            .ok_or_else(|| RigError::Invalid("the metadata has no V3 lens string".into()))?;
        let mode = CaptureMode {
            camera_type: m.camera_type.clone(),
            lens_type: lens_type(v3).map_err(RigError::Invalid)?,
            frame_rate_hz: 1e6 / median(&intervals),
            readout_s: m.rolling_shutter_ms.map(|ms| ms * 1e-3),
        };
        let profile = select_profile(profiles, &mode)?.clone();
        let frames = frame_limit.map_or(vin0.len(), |limit| limit.min(vin0.len()));
        let record_t: Vec<i64> = vin0[..frames].iter().map(|e| e.t_us).collect();
        let exposure_s = [false, true].map(|vin1| {
            clip.frame_exposures(vin1)[..frames]
                .iter()
                .map(|e| e.exposure_s)
                .collect::<Vec<_>>()
        });
        let frame_ns = profile
            .frame_times_ns(&record_t, &exposure_s[0])
            .map_err(RigError::Invalid)?;
        let horizon = frame_limit.map_or(i64::MAX, |_| frame_ns[frames - 1] + 1_000_000_000);
        let imu: Vec<_> = clip
            .trailer
            .imu
            .iter()
            .filter(|s| s.t_us * 1000 <= horizon)
            .collect();
        let imu_from_pano = profile.imu_from_pano();
        let lever_arm = profile.lever_arm();
        let lenses = clip.lenses_v6.unwrap_or(clip.lenses);
        let imu_from_lens = lenses.map(|lens| {
            let rotation = imu_from_pano * lens.pano_from_lens.rotation.matrix();
            let centre = imu_from_pano * (lens.pano_from_lens.translation.vector + lever_arm);
            Isometry3::from_parts(
                Translation3::from(centre),
                UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(rotation)),
            )
        });
        Ok(Self {
            readout_s: mode.readout_s.unwrap_or(profile.readout_s),
            imu_ns: imu.iter().map(|s| s.t_us * 1000).collect(),
            accel: imu.iter().map(|s| s.accel_m_s2).collect(),
            gyro: imu.iter().map(|s| s.gyro_rad_s).collect(),
            clip,
            profile,
            mode,
            lenses,
            imu_from_lens,
            imu_from_pano,
            frame_ns,
            exposure_s,
        })
    }

    /// Frames of the rig.
    pub fn frames(&self) -> usize {
        self.frame_ns.len()
    }

    /// The earliest frame or IMU time, ns: zero of the reader's timeline.
    pub fn start_ns(&self) -> i64 {
        self.imu_ns
            .first()
            .map_or(self.frame_ns[0], |&t| t.min(self.frame_ns[0]))
    }

    /// Frames of lens `lens` cut into GOPs, as Rerun's `Mp4Reader` chunks a stream: each range is a keyframe plus
    /// every sample up to the next keyframe; a stream that does not open on a keyframe starts its first range anyway.
    pub fn gops(&self, lens: usize) -> Vec<Range<usize>> {
        let samples = &self.clip.tracks[self.clip.lens_track[lens]].samples[..self.frames()];
        gop_ranges(samples.iter().map(|s| s.is_sync))
    }

    /// Samples `range` of lens `lens` as Annex B access units (parameter sets on IRAPs), packed: the bytes of every
    /// sample back to back, `range.len() + 1` offsets into them, and each sample's IRAP flag.
    ///
    /// # Errors
    /// A sample outside the track, a read failure or a malformed sample.
    pub fn read_packed(&self, lens: usize, range: Range<usize>) -> Result<Packed, RigError> {
        let track = &self.clip.tracks[self.clip.lens_track[lens]];
        let mut reader = self.clip.sample_reader()?;
        let bytes: u64 = track.samples[range.clone()]
            .iter()
            .map(|s| u64::from(s.size))
            .sum();
        let mut packed = Packed {
            // The parameter sets in front of each IRAP add a few hundred bytes.
            data: Vec::with_capacity(bytes as usize + 1024),
            offsets: Vec::with_capacity(range.len() + 1),
            keyframes: Vec::with_capacity(range.len()),
        };
        packed.offsets.push(0);
        for index in range {
            packed
                .keyframes
                .push(reader.read_annex_b(track, index, &mut packed.data)?);
            let end = i32::try_from(packed.data.len())
                .map_err(|_| RigError::Invalid("a GOP of 2 GiB or more".into()))?;
            packed.offsets.push(end);
        }
        Ok(packed)
    }
}

/// Samples read back to back.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Packed {
    /// Every sample's bytes, in order.
    pub data: Vec<u8>,
    /// Sample `i` is `data[offsets[i]..offsets[i + 1]]`.
    pub offsets: Vec<i32>,
    /// Whether each sample is an IRAP picture.
    pub keyframes: Vec<bool>,
}

/// GOP ranges of a stream with these keyframe flags (see [`InsvRig::gops`]).
pub fn gop_ranges(keyframes: impl IntoIterator<Item = bool>) -> Vec<Range<usize>> {
    let flags: Vec<bool> = keyframes.into_iter().collect();
    let starts: Vec<usize> = (0..flags.len()).filter(|&i| i == 0 || flags[i]).collect();
    starts
        .iter()
        .enumerate()
        .map(|(n, &start)| start..starts.get(n + 1).copied().unwrap_or(flags.len()))
        .collect()
}

fn median(sorted: &[i64]) -> f64 {
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2] as f64
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) as f64 / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn gops_open_on_keyframes_and_the_first_opens_on_frame_zero() {
        assert_eq!(
            gop_ranges([true, false, false, true, false]),
            vec![0..3, 3..5]
        );
        assert_eq!(gop_ranges([false, false, true]), vec![0..2, 2..3]);
        assert_eq!(gop_ranges([true, true]), vec![0..1, 1..2]);
        assert!(gop_ranges([]).is_empty());
    }

    proptest! {
        /// The GOPs tile the stream in order, and every GOP but a leading one opens on a keyframe and holds no other.
        #[test]
        fn gops_tile_the_stream(flags in prop::collection::vec(any::<bool>(), 0..200)) {
            let gops = gop_ranges(flags.iter().copied());
            let mut next = 0;
            for (n, gop) in gops.iter().enumerate() {
                prop_assert_eq!(gop.start, next);
                prop_assert!(gop.end > gop.start);
                prop_assert!(flags[gop.start] || n == 0);
                prop_assert!(flags[gop.start + 1..gop.end].iter().all(|k| !k));
                next = gop.end;
            }
            prop_assert_eq!(next, flags.len());
        }
    }
}
