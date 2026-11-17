//! The clip: the two HEVC lens tracks, the trailer and the parsed factory lenses, read and checked once.
use super::trailer::{read_trailer, Exposure, Trailer};
use super::{malformed, Insta360Error};
use crate::container::hevc::sample_to_annex_b;
use crate::container::mp4::{read_hevc_tracks, VideoTrack};
use kornia_staging_3d::camera::formats::insta360::{parse_lens_string, FactoryLens};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// An Insta360 dual-fisheye clip: trailer, factory lenses and the two HEVC lens tracks.
///
/// Frame `i` of the clip is sample `i` of both lens tracks and exposure entry `first_frame_entry + i` of both sensor
/// records; the clip holds the frames present in both tracks ([`Insta360Clip::frames`]).
#[derive(Clone, Debug)]
pub struct Insta360Clip {
    /// File the clip was read from.
    pub path: PathBuf,
    /// Metadata, IMU and exposure records.
    pub trailer: Trailer,
    /// V3 factory calibration per lens, in lens order, at the track resolution with the window crop applied.
    pub lenses: [FactoryLens; 2],
    /// V6 factory calibration per lens when the clip carries it.
    pub lenses_v6: Option<[FactoryLens; 2]>,
    /// Video track index (into [`Insta360Clip::tracks`]) of each lens.
    pub lens_track: [usize; 2],
    /// HEVC video tracks in file order.
    pub tracks: Vec<VideoTrack>,
    /// Index of the exposure entry that is frame 0: the entry stamped `first_frame_timestamp`.
    pub first_frame_entry: usize,
}

/// The two lenses of a lens string, or the reason it does not describe a dual fisheye.
fn two_lenses(
    text: &str,
    field: &str,
    m: &super::Insta360Metadata,
    size: [u32; 2],
) -> Result<[FactoryLens; 2], Insta360Error> {
    let lenses = parse_lens_string(text, m.window_crop, size)?;
    <[FactoryLens; 2]>::try_from(lenses).map_err(|all| {
        Insta360Error::Unsupported(format!(
            "the {field} lens string has {} lenses, a dual fisheye has 2",
            all.len()
        ))
    })
}

impl Insta360Clip {
    /// Read the container index and the trailer. Samples stay on disk; read them with [`Insta360Clip::sample_reader`].
    ///
    /// # Errors
    /// Besides malformed input: two lens tracks of different sizes, a lens string that is not two supported lenses,
    /// and a vin0 or vin1 exposure record (absent or short) without an entry for every frame are refused by name.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Insta360Error> {
        let path = path.as_ref().to_path_buf();
        let mut file = BufReader::new(File::open(&path)?);
        let tracks = read_hevc_tracks(&mut file)?;
        let trailer = read_trailer(&mut file)?;
        if tracks.len() != 2 {
            return Err(malformed(format!(
                "expected 2 HEVC lens tracks, found {}",
                tracks.len()
            )));
        }
        let size = [u32::from(tracks[0].width), u32::from(tracks[0].height)];
        if [u32::from(tracks[1].width), u32::from(tracks[1].height)] != size {
            return Err(Insta360Error::Unsupported(format!(
                "lens tracks of {}x{} and {}x{} pixels; both lenses must share one size",
                size[0], size[1], tracks[1].width, tracks[1].height
            )));
        }
        let m = &trailer.metadata;
        let v3 = m
            .offset_v3
            .as_deref()
            .ok_or_else(|| malformed("metadata has no V3 lens string (field 54)"))?;
        let lenses = two_lenses(v3, "V3", m, size)?;
        let lenses_v6 = m
            .offset_v6
            .as_deref()
            .map(|s| two_lenses(s, "V6", m, size))
            .transpose()?;
        let first = m.first_frame_timestamp_us;
        let exposures = &trailer.exposures_vin0;
        // In i128, so times anywhere in i64 cannot overflow the difference.
        let distance = |i: usize| (i128::from(exposures[i].t_us) - i128::from(first)).abs();
        let nearest = (0..exposures.len()).min_by_key(|&i| distance(i));
        let period_us = 1e6 / m.frame_rate.unwrap_or(30.0);
        let first_frame_entry = nearest
            .filter(|&i| (distance(i) as f64) < 0.5 * period_us)
            .ok_or_else(|| {
                malformed(format!(
                    "no exposure entry within half a frame of first_frame_timestamp {first}"
                ))
            })?;
        let clip = Self {
            path,
            lens_track: m.lens_track(),
            trailer,
            lenses,
            lenses_v6,
            tracks,
            first_frame_entry,
        };
        let frames = clip.frames();
        for (name, record) in [
            ("vin0 (record 4)", &clip.trailer.exposures_vin0),
            ("vin1 (record 12)", &clip.trailer.exposures_vin1),
        ] {
            if record.len() < first_frame_entry + frames {
                return Err(Insta360Error::Unsupported(format!(
                    "the {name} exposure record holds {} entries; frames 0..{frames} need {}",
                    record.len(),
                    first_frame_entry + frames
                )));
            }
        }
        Ok(clip)
    }

    /// Frames of the clip: those present in both lens tracks.
    pub fn frames(&self) -> usize {
        self.tracks[0]
            .samples
            .len()
            .min(self.tracks[1].samples.len())
    }

    /// Exposure entries of the clip's frames in one sensor record (frame `i` is entry `first_frame_entry + i`).
    pub fn frame_exposures(&self, vin1: bool) -> &[Exposure] {
        let record = if vin1 {
            &self.trailer.exposures_vin1
        } else {
            &self.trailer.exposures_vin0
        };
        &record[self.first_frame_entry..self.first_frame_entry + self.frames()]
    }

    /// Open the file for reading samples.
    pub fn sample_reader(&self) -> Result<SampleReader, Insta360Error> {
        Ok(SampleReader {
            file: File::open(&self.path)?,
            scratch: Vec::new(),
        })
    }
}

/// Reads video samples of a clip as Annex B access units, one at a time.
#[derive(Debug)]
pub struct SampleReader {
    file: File,
    scratch: Vec<u8>,
}

impl SampleReader {
    /// Read sample `index` of `track`, convert it to Annex B with parameter sets on IRAP samples, and append it to
    /// `out`. Returns whether the sample is an IRAP picture.
    pub fn read_annex_b(
        &mut self,
        track: &VideoTrack,
        index: usize,
        out: &mut Vec<u8>,
    ) -> Result<bool, Insta360Error> {
        let sample = track
            .samples
            .get(index)
            .ok_or_else(|| malformed(format!("sample {index} of {}", track.samples.len())))?;
        self.scratch.resize(sample.size as usize, 0);
        self.file.seek(SeekFrom::Start(sample.offset))?;
        self.file.read_exact(&mut self.scratch)?;
        Ok(sample_to_annex_b(
            &self.scratch,
            track.nal_length_size,
            &track.parameter_sets,
            out,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::{bytes_field, exposures, metadata, trailer_file, varint_field, V3};
    use super::*;
    use crate::container::mp4::testing::{bx, trak, TrackSpec};

    const PARAMETER_SETS: &[(u8, &[u8])] = &[(32, &[0x40, 0x01])];
    const ENTRIES: [u64; 5] = [1_000_100, 1_000_150, 1_000_200, 1_033_567, 1_066_934];

    /// An `hvc1` track of `samples` (1 or 2) six-byte samples, `width` pixels wide.
    fn track(id: u32, samples: usize, width: u16) -> Vec<u8> {
        let spec = TrackSpec {
            track_id: id,
            sizes: &[6, 6][..samples],
            chunk_offsets: &[8, 14][..samples],
            stsc: &[[1, 1, 1]],
            stts: &[[samples as u32, 1001]],
            ctts: &[[samples as u32, 0]],
            stss: Some(&[1]),
            media_time: 0,
            co64: false,
            parameter_sets: PARAMETER_SETS,
        };
        let mut trak = trak(&spec);
        let at = trak.windows(4).position(|w| w == b"hvc1").unwrap() + 4 + 24;
        trak[at..at + 2].copy_from_slice(&width.to_be_bytes());
        trak
    }

    /// Open a clip of the given tracks and trailer records, written under a directory named for `name`; the
    /// directory is removed unless the clip opens (then the test removes it).
    fn open(
        name: &str,
        tracks: &[Vec<u8>],
        records: &[(u8, Vec<u8>)],
    ) -> Result<Insta360Clip, Insta360Error> {
        let mut file = bx(b"mdat", &[0, 0, 0, 2, 0x26, 0x01, 0, 0, 0, 2, 0x02, 0x01]);
        file.extend(bx(b"moov", &tracks.concat()));
        file.extend(bx(b"inst", &trailer_file(records)[1000..]));
        let dir = std::env::temp_dir().join(format!("insta360-test-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("clip.insv");
        std::fs::write(&path, &file).unwrap();
        let clip = Insta360Clip::open(&path);
        if clip.is_err() {
            std::fs::remove_dir_all(dir).unwrap();
        }
        clip
    }

    fn two_tracks() -> Vec<Vec<u8>> {
        vec![track(1, 2, 1920), track(2, 2, 1920)]
    }

    fn records(metadata: Vec<u8>, vin1: Option<&[u64]>) -> Vec<(u8, Vec<u8>)> {
        let mut records = vec![(1, metadata), (3, Vec::new()), (4, exposures(&ENTRIES))];
        records.extend(vin1.map(|times| (12, exposures(times))));
        records
    }

    fn unsupported(result: Result<Insta360Clip, Insta360Error>, reason: &str) {
        match result {
            Err(Insta360Error::Unsupported(text)) => assert!(text.contains(reason), "{text}"),
            other => panic!("expected an unsupported clip ({reason}), got {other:?}"),
        }
    }

    #[test]
    fn frame_zero_is_the_exposure_entry_stamped_first_frame_timestamp() {
        let clip = open(
            "base",
            &two_tracks(),
            &records(metadata(true), Some(&ENTRIES)),
        )
        .unwrap();
        assert_eq!(clip.first_frame_entry, 2);
        assert_eq!(
            clip.frame_exposures(false)
                .iter()
                .map(|e| e.t_us)
                .collect::<Vec<_>>(),
            vec![1_000_200, 1_033_567]
        );
        assert_eq!(clip.lens_track, [1, 0]);
        assert!(clip.lenses_v6.is_none());
        let mut reader = clip.sample_reader().unwrap();
        let mut out = Vec::new();
        assert!(reader.read_annex_b(&clip.tracks[0], 0, &mut out).unwrap());
        assert_eq!(out, [0, 0, 0, 1, 0x40, 0x01, 0, 0, 0, 1, 0x26, 0x01]);
        std::fs::remove_dir_all(clip.path.parent().unwrap()).unwrap();
        // A first-frame time of i64::MIN and an exposure at i64::MAX: the distance between them does not overflow,
        // and the clip is refused because no exposure is near the first frame.
        let mut far_metadata = metadata(true);
        varint_field(24, 1 << 63, &mut far_metadata);
        let far = vec![
            (1, far_metadata),
            (3, Vec::new()),
            (4, exposures(&[i64::MAX as u64])),
            (12, exposures(&[i64::MAX as u64])),
        ];
        assert!(matches!(
            open("far", &two_tracks(), &far),
            Err(Insta360Error::Malformed(_))
        ));
    }

    #[test]
    fn unequal_lens_tracks_hold_the_frames_present_in_both() {
        let tracks = [track(1, 2, 1920), track(2, 1, 1920)];
        let clip = open("unequal", &tracks, &records(metadata(true), Some(&ENTRIES))).unwrap();
        assert_eq!(clip.frames(), 1);
        assert_eq!(clip.frame_exposures(false).len(), 1);
        assert_eq!(clip.frame_exposures(true).len(), 1);
        std::fs::remove_dir_all(clip.path.parent().unwrap()).unwrap();
    }

    #[test]
    fn an_absent_or_short_vin1_record_is_refused_by_name() {
        unsupported(
            open("no-vin1", &two_tracks(), &records(metadata(true), None)),
            "vin1",
        );
        // Frame 0 is entry 2, so two frames need four entries.
        unsupported(
            open(
                "short-vin1",
                &two_tracks(),
                &records(metadata(true), Some(&ENTRIES[..3])),
            ),
            "vin1",
        );
    }

    #[test]
    fn unsupported_lens_strings_and_capture_geometry_are_refused_by_name() {
        // Lens tracks of two sizes.
        let tracks = [track(1, 2, 1920), track(2, 2, 960)];
        unsupported(
            open("sizes", &tracks, &records(metadata(true), Some(&ENTRIES))),
            "share one size",
        );
        // A non-finite token: the later field 54 replaces the first.
        let mut bad = metadata(true);
        bytes_field(
            54,
            V3.replacen("_3842.370_", "_inf_", 1).as_bytes(),
            &mut bad,
        );
        match open("inf", &two_tracks(), &records(bad, Some(&ENTRIES))) {
            Err(Insta360Error::LensString(error)) => {
                assert!(error.to_string().contains("\"inf\""), "{error}")
            }
            other => panic!("expected a lens-string error, got {other:?}"),
        }
        // A one-lens string.
        let one_lens: String = V3
            .split('_')
            .skip(1)
            .take(19)
            .fold("1".to_owned(), |s, t| s + "_" + t)
            + "_197632";
        let mut single = metadata(true);
        bytes_field(54, one_lens.as_bytes(), &mut single);
        unsupported(
            open("one", &two_tracks(), &records(single, Some(&ENTRIES))),
            "has 1 lenses",
        );
    }
}
