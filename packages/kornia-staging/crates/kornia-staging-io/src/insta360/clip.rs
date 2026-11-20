//! The clip: the two lens tracks (H.264 or HEVC), the trailer and the parsed factory lenses, read and checked once.
use super::trailer::{read_trailer, Exposure, Trailer};
use super::{malformed, Insta360Error};
use crate::container::mp4::{read_video_tracks, VideoTrack};
use crate::container::nal::sample_to_annex_b;
use kornia_staging_3d::camera::formats::insta360::{parse_lens_string, FactoryLens};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// An Insta360 dual-fisheye clip: trailer, factory lenses and the two lens tracks.
///
/// Recent bodies (X4 and later) record both lenses into one file, one video track each. Older ones (ONE X2, ONE
/// R/RS, X3) record one lens per file: `…_00_<seq>.insv` holds lens 0 and the trailer, `…_10_<seq>.insv` lens 1; the
/// clip opens from either and reads both.
///
/// Frame `i` of the clip is sample `i` of both lens tracks and exposure entry `first_frame_entry + i` of both sensor
/// records; the clip holds the frames present in both tracks ([`Insta360Clip::frames`]).
#[derive(Clone, Debug)]
pub struct Insta360Clip {
    /// The file holding the trailer (the `_00_` file of a two-file clip).
    pub path: PathBuf,
    /// Metadata, IMU and exposure records.
    pub trailer: Trailer,
    /// V3 factory calibration per lens (V6 on a body that writes no V3 string, the X6), in lens order, at the track
    /// resolution with the window crop applied.
    pub lenses: [FactoryLens; 2],
    /// V6 factory calibration per lens when the clip carries it.
    pub lenses_v6: Option<[FactoryLens; 2]>,
    /// Video track index (into [`Insta360Clip::tracks`]) of each lens.
    pub lens_track: [usize; 2],
    /// The lens video tracks, in file order (a two-file clip: the `_00_` file's, then the `_10_` file's).
    pub tracks: Vec<VideoTrack>,
    /// The file each track's samples are in.
    pub track_paths: Vec<PathBuf>,
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

/// The member `lens` (`0` or `1`) of the two-file clip `path` belongs to, from its name: the field `00` or `10`
/// after the time (`VID_<date>_<time>_<lens>_<seq>.insv`). `None` when the name has no such field.
pub fn lens_file(path: &Path, lens: usize) -> Option<PathBuf> {
    let name = path.file_name()?.to_str()?;
    let mut parts: Vec<&str> = name.split('_').collect();
    let field = parts.iter().position(|p| *p == "00" || *p == "10")?;
    let replacement = ["00", "10"][lens];
    parts[field] = replacement;
    Some(path.with_file_name(parts.join("_")))
}

impl Insta360Clip {
    /// Read the container index and the trailer. Samples stay on disk; read them with [`Insta360Clip::sample_reader`].
    ///
    /// # Errors
    /// Besides malformed input: a file that is neither two lens tracks nor one member of a two-file clip, lens tracks
    /// of different sizes or codecs, a lens string that is not two supported lenses, and a vin0 or vin1 exposure
    /// record without an entry for every frame are refused by name. A clip without a vin1 record (the X5 writes only
    /// record 4) gives both lenses the vin0 entries.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Insta360Error> {
        let given = path.as_ref().to_path_buf();
        // A two-file clip's trailer is in its `_00_` file, whichever member was given.
        let path = match lens_file(&given, 0) {
            Some(first) if first != given && first.exists() => first,
            _ => given,
        };
        let mut file = BufReader::new(File::open(&path)?);
        let mut tracks = read_video_tracks(&mut file)?;
        let trailer = read_trailer(&mut file)?;
        let mut track_paths = vec![path.clone(); tracks.len()];
        let per_file = tracks.len() == 1;
        if per_file {
            let second = lens_file(&path, 1).filter(|p| *p != path).ok_or_else(|| {
                Insta360Error::Unsupported(format!(
                    "{} holds one video track and its name has no `_00_` lens field to find the other lens's file",
                    path.display()
                ))
            })?;
            let other =
                read_video_tracks(&mut BufReader::new(File::open(&second).map_err(|e| {
                    Insta360Error::Unsupported(format!(
                        "{} holds one lens; the other lens's file {} could not be opened: {e}",
                        path.display(),
                        second.display()
                    ))
                })?))?;
            if other.len() != 1 {
                return Err(malformed(format!(
                    "{} holds {} video tracks, a lens file holds 1",
                    second.display(),
                    other.len()
                )));
            }
            tracks.extend(other);
            track_paths.push(second);
        }
        if tracks.len() != 2 {
            return Err(malformed(format!(
                "expected 2 lens video tracks (H.264 or HEVC), found {}",
                tracks.len()
            )));
        }
        let size = [u32::from(tracks[0].width), u32::from(tracks[0].height)];
        if [u32::from(tracks[1].width), u32::from(tracks[1].height)] != size
            || tracks[0].codec != tracks[1].codec
        {
            return Err(Insta360Error::Unsupported(format!(
                "lens tracks of {}x{} {:?} and {}x{} {:?}; both lenses must share one size and codec",
                size[0], size[1], tracks[0].codec, tracks[1].width, tracks[1].height, tracks[1].codec
            )));
        }
        let m = &trailer.metadata;
        let lenses_v6 = m
            .offset_v6
            .as_deref()
            .map(|s| two_lenses(s, "V6", m, size))
            .transpose()?;
        let lenses = match (m.offset_v3.as_deref(), lenses_v6) {
            (Some(v3), _) => two_lenses(v3, "V3", m, size)?,
            (None, Some(v6)) => v6,
            (None, None) => {
                return Err(malformed(
                    "metadata has no V3 (field 54) or V6 (field 111) lens string",
                ))
            }
        };
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
            // One lens per file: the files name the lens order.
            lens_track: if per_file { [0, 1] } else { m.lens_track() },
            trailer,
            lenses,
            lenses_v6,
            tracks,
            track_paths,
            first_frame_entry,
        };
        let frames = clip.frames();
        for (name, record) in [
            ("vin0 (record 4)", clip.trailer.exposures_vin0.as_slice()),
            ("vin1 (record 12)", clip.vin1_record()),
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

    /// Exposure entries of the clip's frames in one sensor record (frame `i` is entry `first_frame_entry + i`); vin1
    /// is the vin0 record when the clip has no record 12.
    pub fn frame_exposures(&self, vin1: bool) -> &[Exposure] {
        let record = if vin1 {
            self.vin1_record()
        } else {
            &self.trailer.exposures_vin0
        };
        &record[self.first_frame_entry..self.first_frame_entry + self.frames()]
    }

    /// Record 12, or record 4 when the clip has no record 12.
    fn vin1_record(&self) -> &[Exposure] {
        if self.trailer.exposures_vin1.is_empty() {
            &self.trailer.exposures_vin0
        } else {
            &self.trailer.exposures_vin1
        }
    }

    /// Open the clip's files for reading samples.
    pub fn sample_reader(&self) -> Result<SampleReader<'_>, Insta360Error> {
        Ok(SampleReader {
            clip: self,
            files: self
                .track_paths
                .iter()
                .map(File::open)
                .collect::<Result<_, _>>()?,
            scratch: Vec::new(),
        })
    }
}

/// Reads video samples of a clip as Annex B access units, one at a time.
#[derive(Debug)]
pub struct SampleReader<'a> {
    clip: &'a Insta360Clip,
    files: Vec<File>,
    scratch: Vec<u8>,
}

impl SampleReader<'_> {
    /// Read sample `index` of track `track` (an index into [`Insta360Clip::tracks`]), convert it to Annex B with the
    /// parameter sets on random-access samples, and append it to `out`. Returns whether the sample is a random-access
    /// picture (an H.264 IDR, an HEVC IRAP).
    pub fn read_annex_b(
        &mut self,
        track: usize,
        index: usize,
        out: &mut Vec<u8>,
    ) -> Result<bool, Insta360Error> {
        let t = self
            .clip
            .tracks
            .get(track)
            .ok_or_else(|| malformed(format!("track {track} of {}", self.clip.tracks.len())))?;
        let sample = t
            .samples
            .get(index)
            .ok_or_else(|| malformed(format!("sample {index} of {}", t.samples.len())))?;
        self.scratch.resize(sample.size as usize, 0);
        let file = &mut self.files[track];
        file.seek(SeekFrom::Start(sample.offset))?;
        file.read_exact(&mut self.scratch)?;
        Ok(sample_to_annex_b(
            t.codec,
            &self.scratch,
            t.nal_length_size,
            &t.parameter_sets,
            out,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::{
        bytes_field, exposures, metadata, metadata_with_lens, trailer_file, varint_field, V3, V6,
    };
    use super::*;
    use crate::container::mp4::testing::{bx, trak, TrackSpec};

    const PARAMETER_SETS: &[(u8, &[u8])] = &[(32, &[0x40, 0x01])];
    const ENTRIES: [u64; 5] = [1_000_100, 1_000_150, 1_000_200, 1_033_567, 1_066_934];

    /// An `hvc1` track of `samples` (1 or 2) six-byte samples, `width` pixels wide.
    fn track(id: u32, samples: usize, width: u16) -> Vec<u8> {
        let spec = TrackSpec {
            h264: false,
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
        assert!(reader.read_annex_b(0, 0, &mut out).unwrap());
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
    fn a_two_file_clip_opens_from_either_member_and_reads_lens_1_from_its_own_file() {
        let dir =
            std::env::temp_dir().join(format!("insta360-test-{}-two-file", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Lens 0 and the trailer in `_00_`, lens 1 in `_10_` (the ONE X2 / ONE R / X3 layout); the samples differ.
        let mut first = bx(b"mdat", &[0, 0, 0, 2, 0x26, 0x01, 0, 0, 0, 2, 0x02, 0x01]);
        first.extend(bx(b"moov", &track(1, 2, 1920)));
        first.extend(bx(
            b"inst",
            &trailer_file(&records(metadata(true), Some(&ENTRIES)))[1000..],
        ));
        let mut second = bx(b"mdat", &[0, 0, 0, 2, 0x26, 0x07, 0, 0, 0, 2, 0x02, 0x07]);
        second.extend(bx(b"moov", &track(1, 2, 1920)));
        let (lens0, lens1) = (
            dir.join("VID_20231126_111301_00_004.insv"),
            dir.join("VID_20231126_111301_10_004.insv"),
        );
        std::fs::write(&lens0, &first).unwrap();
        std::fs::write(&lens1, &second).unwrap();
        for given in [&lens0, &lens1] {
            let clip = Insta360Clip::open(given).unwrap();
            assert_eq!(
                (clip.path.as_path(), clip.lens_track),
                (lens0.as_path(), [0, 1])
            );
            assert_eq!(clip.track_paths, vec![lens0.clone(), lens1.clone()]);
            let mut reader = clip.sample_reader().unwrap();
            let mut out = Vec::new();
            reader.read_annex_b(1, 0, &mut out).unwrap();
            assert_eq!(out[out.len() - 2..], [0x26, 0x07]);
        }
        // Without its `_10_` partner, a one-track file is refused by name.
        std::fs::remove_file(&lens1).unwrap();
        unsupported(Insta360Clip::open(&lens0), "other lens's file");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_body_without_a_v3_string_takes_its_lenses_from_v6() {
        let clip = open(
            "v6-only",
            &two_tracks(),
            &records(metadata_with_lens(true, 111, V6), Some(&ENTRIES)),
        )
        .unwrap();
        assert_eq!(Some(clip.lenses), clip.lenses_v6);
        std::fs::remove_dir_all(clip.path.parent().unwrap()).unwrap();
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
    fn an_absent_vin1_record_gives_both_lenses_the_vin0_entries() {
        let clip = open("no-vin1", &two_tracks(), &records(metadata(true), None)).unwrap();
        assert_eq!(clip.frame_exposures(true), clip.frame_exposures(false));
        assert_eq!(clip.frame_exposures(true).len(), 2);
        std::fs::remove_dir_all(clip.path.parent().unwrap()).unwrap();
    }

    #[test]
    fn a_short_vin1_record_is_refused_by_name() {
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
