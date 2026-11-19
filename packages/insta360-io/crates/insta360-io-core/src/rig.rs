//! The rig of one clip: two lens cameras, frame and IMU times on one clock, the per-frame exposure, and the lens tracks
//! cut into GOPs.
//!
//! With a device profile (the body and video mode one was measured in), rig frame = IMU frame. Lens `k` is posed as
//! `T_imu_lens = T_imu_P * T_P_lens`: `T_P_lens` is the factory rotation and centre in the vendor rig frame P (V6 lens
//! string when the clip has one, else V3), `T_imu_P = (R_imu_P, R_imu_P * lever_arm)` comes from the profile. Frame `i`
//! is stamped at its effective time on the IMU clock: the exposure-record-4 stamp plus the profile's `slope *
//! shutter_i + constant` (mid-exposure of the middle row).
//!
//! Without one (a body nobody measured, or a mode outside every measured one), the clip is still read, and
//! [`InsvRig::profile`] says why there is no profile: rig frame = the vendor frame P (the lens poses are the clip's
//! own factory calibration), frames sit at the raw record-4 stamps, the readout is the clip's own if it states one,
//! and the IMU has no pose in the rig. Such a rig must not feed visual-inertial estimation.
//!
//! IMU times are the trailer's, without the vendor's `gyro_timestamp` subtraction: the profiles were fitted on that
//! clock. Every time is in nanoseconds on the trailer clock.
use crate::profile::{builtin_profiles, select_profile, CaptureMode, DeviceProfile};
use kornia_staging_3d::camera::formats::insta360::FactoryLens;
use kornia_staging_io::insta360::{Insta360Clip, Insta360Error};
use nalgebra::{Isometry3, Matrix3, Rotation3, Translation3, UnitQuaternion, Vector3};
use std::ops::Range;
use std::path::Path;

/// Why a rig could not be built from a clip.
#[derive(Debug, thiserror::Error)]
pub enum RigError {
    /// The file is not a readable two-lens clip.
    #[error(transparent)]
    Clip(#[from] Insta360Error),
    /// The clip's records do not support a rig.
    #[error("{0}")]
    Invalid(String),
    /// A lens or sample range outside the rig.
    #[error("{0}")]
    OutOfRange(String),
}

/// A clip read and checked, with its device profile applied when one fits.
#[derive(Clone, Debug)]
pub struct InsvRig {
    /// The clip.
    pub clip: Insta360Clip,
    /// The profile the clip's mode selected, or why no measured profile fits: the body has none, the clip lies
    /// outside its measured modes, or its lenses differ in type.
    pub profile: Result<DeviceProfile, String>,
    /// Rolling-shutter readout of each lens image: the clip's own when it states one, else the profile's; `None`
    /// without either.
    pub readout_s: Option<f64>,
    /// Lens 0 and lens 1: V6 when the clip has a V6 string, else V3.
    pub lenses: [FactoryLens; 2],
    /// `T_rig_lens` of lens 0 and lens 1: in the IMU frame with a profile, in the vendor frame P without.
    pub rig_from_lens: [Isometry3<f64>; 2],
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
    /// A malformed clip, fewer than two frames, a frame limit of 0, a lens track whose pictures are presented out of
    /// decode order (B-frames), or frame times that do not keep increasing.
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
        if frame_limit == Some(0) {
            return Err(RigError::Invalid(
                "frame limit 0; a rig needs a frame".into(),
            ));
        }
        // Frame `i` is sample `i` in decode order, so a track that shows its pictures in another order (B-frames)
        // would pair pictures with the wrong exposure records.
        for (lens, &track) in clip.lens_track.iter().enumerate() {
            if clip.tracks[track]
                .samples
                .windows(2)
                .any(|w| w[1].pts < w[0].pts)
            {
                return Err(RigError::Invalid(format!(
                    "lens {lens}: pictures are presented out of decode order (B-frames); the reader pairs samples with \
                     exposure records in decode order"
                )));
            }
        }
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
        let lenses = clip.lenses_v6.unwrap_or(clip.lenses);
        let lens_type = match lenses.map(|lens| lens.lens_type) {
            [a, b] if a == b => Ok(a),
            [a, b] => Err(format!(
                "lenses of types {a} and {b}; a profile needs one lens type"
            )),
        };
        let clip_readout_s = m.rolling_shutter_ms.map(|ms| ms * 1e-3);
        let profile = choose_profile(
            profiles,
            &m.camera_type,
            lens_type,
            1e6 / median(&intervals),
            clip_readout_s,
        );
        let frames = frame_limit.map_or(vin0.len(), |limit| limit.min(vin0.len()));
        let record_t: Vec<i64> = vin0[..frames].iter().map(|e| e.t_us).collect();
        let exposure_s = [false, true].map(|vin1| {
            clip.frame_exposures(vin1)[..frames]
                .iter()
                .map(|e| e.exposure_s)
                .collect::<Vec<_>>()
        });
        let frame_ns = match &profile {
            Ok(p) => p.frame_times_ns(&record_t, &exposure_s[0]),
            Err(_) => {
                let raw: Vec<i64> = record_t.iter().map(|t| t * 1000).collect();
                if raw.windows(2).any(|w| w[1] <= w[0]) {
                    Err("the exposure-record stamps are not increasing".to_owned())
                } else {
                    Ok(raw)
                }
            }
        }
        .map_err(RigError::Invalid)?;
        let horizon = frame_limit.map_or(i64::MAX, |_| frame_ns[frames - 1] + 1_000_000_000);
        let imu: Vec<_> = clip
            .trailer
            .imu
            .iter()
            .filter(|s| s.t_us * 1000 <= horizon)
            .collect();
        // Without a profile the rig frame is P itself: the identity rotation and no lever arm.
        let (rig_from_pano, lever_arm) = profile.as_ref().map_or_else(
            |_| (Matrix3::identity(), Vector3::zeros()),
            |p| (p.imu_from_pano(), p.lever_arm()),
        );
        let rig_from_lens = lenses.map(|lens| {
            let rotation = rig_from_pano * lens.pano_from_lens.rotation.matrix();
            let centre = rig_from_pano * (lens.pano_from_lens.translation.vector + lever_arm);
            Isometry3::from_parts(
                Translation3::from(centre),
                UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(rotation)),
            )
        });
        Ok(Self {
            readout_s: clip_readout_s.or(profile.as_ref().ok().map(|p| p.readout_s)),
            imu_ns: imu.iter().map(|s| s.t_us * 1000).collect(),
            accel: imu.iter().map(|s| s.accel_m_s2).collect(),
            gyro: imu.iter().map(|s| s.gyro_rad_s).collect(),
            clip,
            profile,
            lenses,
            rig_from_lens,
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

    /// The track index of lens `lens`.
    fn track(&self, lens: usize) -> Result<usize, RigError> {
        self.clip
            .lens_track
            .get(lens)
            .copied()
            .ok_or_else(|| RigError::OutOfRange(format!("lens {lens}; a rig has lenses 0 and 1")))
    }

    /// Frames of lens `lens` cut into GOPs, as Rerun's `Mp4Reader` chunks a stream: each range is a keyframe plus
    /// every sample up to the next keyframe; a stream that does not open on a keyframe starts its first range anyway.
    ///
    /// # Errors
    /// A lens other than 0 and 1.
    pub fn gops(&self, lens: usize) -> Result<Vec<Range<usize>>, RigError> {
        let samples = &self.clip.tracks[self.track(lens)?].samples[..self.frames()];
        Ok(gop_ranges(samples.iter().map(|s| s.is_sync)))
    }

    /// Samples `range` of lens `lens` as Annex B access units (parameter sets on IRAPs), packed: the bytes of every
    /// sample back to back, `range.len() + 1` offsets into them, and each sample's IRAP flag.
    ///
    /// # Errors
    /// A lens other than 0 and 1 or a range outside the rig's frames ([`RigError::OutOfRange`]), a read failure or a
    /// malformed sample.
    pub fn read_packed(&self, lens: usize, range: Range<usize>) -> Result<Packed, RigError> {
        let track_index = self.track(lens)?;
        if range.start > range.end || range.end > self.frames() {
            return Err(RigError::OutOfRange(format!(
                "samples {range:?} of a rig of {} frames",
                self.frames()
            )));
        }
        let track = &self.clip.tracks[track_index];
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
                .push(reader.read_annex_b(track_index, index, &mut packed.data)?);
            let end = i32::try_from(packed.data.len())
                .map_err(|_| RigError::Invalid("a GOP of 2 GiB or more".into()))?;
            packed.offsets.push(end);
        }
        Ok(packed)
    }
}

/// The profile measured in the clip's mode, or why there is none: the body has no profile, the clip lies outside
/// every measured mode of its body, or its two lenses are of different types.
pub fn choose_profile(
    profiles: &[DeviceProfile],
    camera_type: &str,
    lens_type: Result<u32, String>,
    frame_rate_hz: f64,
    readout_s: Option<f64>,
) -> Result<DeviceProfile, String> {
    let mode = CaptureMode {
        camera_type: camera_type.to_owned(),
        lens_type: lens_type?,
        frame_rate_hz,
        readout_s,
    };
    select_profile(profiles, &mode)
        .cloned()
        .map_err(|e| e.to_string())
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
    use crate::profile::builtin_profiles;
    use kornia_staging_3d::camera::formats::insta360::parse_lens_string;
    use kornia_staging_io::container::mp4::{Sample, VideoCodec, VideoTrack};
    use kornia_staging_io::insta360::{Exposure, Insta360Metadata, Trailer};
    use proptest::prelude::*;

    #[test]
    fn a_clip_without_a_fitting_profile_is_read_without_one_and_says_why() {
        let profiles = builtin_profiles();
        let profile = choose_profile(&profiles, "Insta360 X5", Ok(113), 29.9688, Some(0.021244));
        assert_eq!(profile.map(|p| p.name), Ok("x5".to_owned()));
        // A measured body in an unmeasured mode (the public X5 sample records 25 fps).
        let note =
            choose_profile(&profiles, "Insta360 X5", Ok(113), 25.0, Some(0.021244)).unwrap_err();
        assert!(
            note.contains("frame rate 25.0000 Hz is not 29.97"),
            "{note}"
        );
        // A body nobody measured.
        let note = choose_profile(&profiles, "Insta360 X3", Ok(97), 29.97, None).unwrap_err();
        assert!(
            note.contains("no device profile for a \"Insta360 X3\""),
            "{note}"
        );
        // Lens strings without a single lens type.
        let note = choose_profile(
            &profiles,
            "Insta360 X5",
            Err("mixed lens types".into()),
            29.97,
            None,
        );
        assert_eq!(note.map(|p| p.name), Err("mixed lens types".to_owned()));
    }

    /// A lens string of `version` 3 or 6 for two lenses of these types (3840 x 1920 calibration frame).
    fn lens_string(version: u32, types: [u32; 2]) -> String {
        let coefficients = if version == 3 { 5 } else { 13 };
        let mut tokens = vec!["2".to_owned()];
        for (lens, lens_type) in types.iter().enumerate() {
            let cx = 960.0 + 1920.0 * lens as f64;
            tokens.extend(
                [
                    2.0, 1000.0, 1000.0, cx, 960.0, 0.0, 0.0, 90.0, 0.0, 0.0, 0.0,
                ]
                .map(|v: f64| v.to_string()),
            );
            tokens.extend(std::iter::repeat_n("0.0".to_owned(), coefficients));
            tokens.extend(["3840".to_owned(), "1920".to_owned(), lens_type.to_string()]);
        }
        tokens.push((version << 16).to_string());
        tokens.join("_")
    }

    /// An X4 Air clip in the measured mode with `frames` frames whose lens samples have these presentation times; no
    /// file holds its samples.
    fn x4_air_clip(
        frames: usize,
        v3_types: [u32; 2],
        v6_types: Option<[u32; 2]>,
        pts: &[i64],
    ) -> Insta360Clip {
        let parse = |text: &str| -> [FactoryLens; 2] {
            parse_lens_string(text, None, [1920, 1920])
                .unwrap()
                .try_into()
                .unwrap()
        };
        let (v3, v6) = (
            lens_string(3, v3_types),
            v6_types.map(|t| lens_string(6, t)),
        );
        let samples: Vec<Sample> = (0..frames)
            .map(|i| Sample {
                offset: 0,
                size: 1,
                dts: 1001 * i as i64,
                pts: pts[i],
                is_sync: i % 30 == 0,
            })
            .collect();
        let track = VideoTrack {
            track_id: 1,
            codec: VideoCodec::H265,
            width: 1920,
            height: 1920,
            timescale: 30000,
            nal_length_size: 4,
            parameter_sets: Vec::new(),
            samples,
        };
        Insta360Clip {
            path: "VID_x_00_001.insv".into(),
            trailer: Trailer {
                metadata: Insta360Metadata {
                    camera_type: "Insta360 X4 Air".into(),
                    offset_v3: Some(v3.clone()),
                    offset_v6: v6.clone(),
                    rolling_shutter_ms: Some(15.883),
                    ..Default::default()
                },
                imu: Vec::new(),
                exposures_vin0: (0..frames as i64)
                    .map(|i| Exposure {
                        t_us: 33_367 * i,
                        exposure_s: 0.001,
                    })
                    .collect(),
                exposures_vin1: Vec::new(),
            },
            lenses: parse(&v3),
            lenses_v6: v6.as_deref().map(parse),
            lens_track: [0, 1],
            tracks: vec![track.clone(), track],
            track_paths: vec!["VID_x_00_001.insv".into(); 2],
            first_frame_entry: 0,
        }
    }

    fn rig(clip: Insta360Clip) -> Result<InsvRig, RigError> {
        InsvRig::from_clip(clip, &builtin_profiles(), None)
    }

    fn in_order(frames: usize) -> Vec<i64> {
        (0..frames as i64).map(|i| 1001 * i).collect()
    }

    #[test]
    fn the_profile_is_selected_on_the_lenses_the_rig_emits() {
        let measured = rig(x4_air_clip(4, [131, 131], Some([131, 131]), &in_order(4))).unwrap();
        assert_eq!(measured.profile.map(|p| p.name), Ok("x4-air".to_owned()));
        // The emitted cameras come from the V6 string; its lens types decide, whatever the V3 string says.
        let unmeasured = rig(x4_air_clip(4, [131, 131], Some([132, 132]), &in_order(4))).unwrap();
        let note = unmeasured.profile.unwrap_err();
        assert!(note.contains("lens type 132"), "{note}");
        let mixed = rig(x4_air_clip(4, [131, 131], Some([132, 131]), &in_order(4))).unwrap();
        let note = mixed.profile.unwrap_err();
        assert!(note.contains("lenses of types 132 and 131"), "{note}");
        // Without a V6 string the V3 lenses are emitted.
        let v3_only = rig(x4_air_clip(4, [131, 131], None, &in_order(4))).unwrap();
        assert!(v3_only.profile.is_ok());
    }

    #[test]
    fn a_track_whose_pictures_are_presented_out_of_decode_order_is_refused() {
        // I P B B in decode order, shown I B B P: the reader pairs samples with exposure records in decode order.
        let reordered = [0, 3003, 1001, 2002];
        for limit in [None, Some(1)] {
            let error = InsvRig::from_clip(
                x4_air_clip(4, [131, 131], None, &reordered),
                &builtin_profiles(),
                limit,
            )
            .unwrap_err()
            .to_string();
            assert!(
                error.contains("lens 0") && error.contains("out of decode order"),
                "{error}"
            );
        }
        // A constant composition offset keeps the order.
        let shifted: Vec<i64> = in_order(4).iter().map(|t| t + 2002).collect();
        assert!(rig(x4_air_clip(4, [131, 131], None, &shifted)).is_ok());
    }

    #[test]
    fn a_zero_frame_limit_and_reads_outside_the_rig_are_errors_not_panics() {
        let zero = InsvRig::from_clip(
            x4_air_clip(4, [131, 131], None, &in_order(4)),
            &builtin_profiles(),
            Some(0),
        );
        assert!(matches!(zero, Err(RigError::Invalid(ref why)) if why.contains("frame limit 0")));
        let rig = InsvRig::from_clip(
            x4_air_clip(4, [131, 131], None, &in_order(4)),
            &builtin_profiles(),
            Some(3),
        )
        .unwrap();
        assert_eq!(rig.gops(1).unwrap(), vec![0..3]);
        assert!(matches!(rig.gops(2), Err(RigError::OutOfRange(_))));
        for (lens, (start, stop)) in [(2, (0, 1)), (0, (2, 1)), (0, (0, 4)), (1, (3, 4))] {
            let range = start..stop;
            assert!(
                matches!(
                    rig.read_packed(lens, range.clone()),
                    Err(RigError::OutOfRange(_))
                ),
                "lens {lens} {range:?}"
            );
        }
    }

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
