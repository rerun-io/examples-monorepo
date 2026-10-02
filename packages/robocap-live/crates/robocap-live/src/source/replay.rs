//! Replay of a `robocap-live-dump/1` directory (SPEC "The dump format") as a [`FrameSource`].
//!
//! Events come out in time order: the IMU samples up to a frameset's time, then the frameset (on equal times the IMU sample
//! first). Options:
//! - `realtime` paces events by `t_ns` against a monotonic clock (late events go out at once and are counted);
//! - `looping` restarts at the end with all times shifted forward by the clip span plus [`LOOP_GAP_NS`], so time stays strictly
//!   increasing and consumers see the restart as a gap (SLAM resets on it);
//! - `preload` reads the whole clip into RAM first (the cap's eMMC is slower than 30 framesets/s x 6 x 2 MB).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use super::{FrameSource, SourceError, SourceEvent};
use crate::frame::{DumpMeta, FULL_SIZE, FrameError, FrameReader, Frameset, ImuSample, Rig, read_imu};

/// Time inserted between the end of one loop and the start of the next.
pub const LOOP_GAP_NS: i64 = 500_000_000;
/// Bytes of one `reference_world_from_rig.bin` record: `i64 t_ns` + 16 `f64` (row-major).
pub const REFERENCE_RECORD_BYTES: usize = 8 + 16 * 8;

/// How to replay.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReplayConfig {
    /// Pace events by their timestamps.
    pub realtime: bool,
    /// Restart at the end, times shifted forward.
    pub looping: bool,
    /// Read every frameset into RAM before the first event.
    pub preload: bool,
}

/// Counters of a replay.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReplayCounts {
    /// Framesets yielded.
    pub framesets: u64,
    /// IMU samples yielded.
    pub imu: u64,
    /// Completed loops.
    pub loops: u64,
    /// Realtime events that were already due by more than 5 ms when produced (the reader is too slow).
    pub late: u64,
}

enum Frames {
    Reader(FrameReader),
    Preloaded { framesets: Vec<Frameset>, next: usize },
}

/// A dump directory as a time-ordered event source.
pub struct ReplaySource {
    dir: PathBuf,
    rig: Rig,
    meta: DumpMeta,
    options: ReplayConfig,
    frames: Frames,
    imu: Vec<ImuSample>,
    imu_next: usize,
    pending: Option<Frameset>,
    frames_done: bool,
    first_t_ns: i64,
    span_ns: i64,
    loop_index: i64,
    frames_per_loop: u64,
    pacing: Option<(Instant, i64)>,
    stop: Arc<AtomicBool>,
    /// What was yielded so far.
    pub counts: ReplayCounts,
}

fn invalid(message: String) -> SourceError {
    SourceError::Frame(FrameError::Invalid(message))
}

/// Read `reference_world_from_rig.bin`: per frame `(t_ns, world_from_rig row-major)`.
///
/// # Errors
///
/// [`SourceError::Frame`] when the file is missing or not whole records.
pub fn read_reference_poses(dir: &Path) -> Result<Vec<(i64, [f64; 16])>, SourceError> {
    let path = dir.join("reference_world_from_rig.bin");
    let bytes = std::fs::read(&path).map_err(|source| FrameError::Io { path: path.clone(), source })?;
    if bytes.len() % REFERENCE_RECORD_BYTES != 0 {
        return Err(invalid(format!("{}: {} bytes is not whole {REFERENCE_RECORD_BYTES}-byte records", path.display(), bytes.len())));
    }
    let mut poses = Vec::with_capacity(bytes.len() / REFERENCE_RECORD_BYTES);
    for record in bytes.chunks_exact(REFERENCE_RECORD_BYTES) {
        let word = |at: usize| {
            let mut out = [0u8; 8];
            out.copy_from_slice(&record[at..at + 8]);
            out
        };
        let t_ns = i64::from_le_bytes(word(0));
        let matrix: [f64; 16] = std::array::from_fn(|i| f64::from_le_bytes(word(8 + 8 * i)));
        poses.push((t_ns, matrix));
    }
    Ok(poses)
}

impl ReplaySource {
    /// Open a dump directory (`meta.json`, `rig.json`, `frames.bin`, `imu.bin`). `stop` ends the replay at the next event.
    ///
    /// # Errors
    ///
    /// [`SourceError::Frame`] when a file is missing or malformed, the frames are not 1920x1080, `meta.json` and `rig.json`
    /// name different devices, or preloading fails.
    pub fn open(dir: &Path, options: ReplayConfig, stop: Arc<AtomicBool>) -> Result<Self, SourceError> {
        let meta = DumpMeta::load(dir)?;
        if meta.size() != FULL_SIZE {
            return Err(invalid(format!("{}: frames are {}x{}, expected 1920x1080", dir.join("meta.json").display(), meta.width, meta.height)));
        }
        let rig = Rig::load(&dir.join("rig.json"))?;
        if meta.device != rig.device {
            return Err(invalid(format!("{}: meta.json says device {:?}, rig.json {:?}", dir.display(), meta.device, rig.device)));
        }
        let imu = read_imu(&dir.join("imu.bin"))?;
        if imu.windows(2).any(|pair| pair[1].t_ns <= pair[0].t_ns) {
            return Err(invalid(format!("{}: IMU timestamps are not strictly increasing", dir.join("imu.bin").display())));
        }
        let reader = FrameReader::open(&dir.join("frames.bin"), FULL_SIZE)?;
        let frames = if options.preload { preload(reader, &stop)? } else { Frames::Reader(reader) };
        let first_t_ns = imu.first().map_or(meta.first_t_ns, |sample| sample.t_ns.min(meta.first_t_ns));
        let last_t_ns = imu.last().map_or(meta.last_t_ns, |sample| sample.t_ns.max(meta.last_t_ns));
        let frames_per_loop = match &frames {
            Frames::Preloaded { framesets, .. } => framesets.len() as u64,
            Frames::Reader(_) => meta.frames,
        };
        Ok(Self {
            dir: dir.to_path_buf(),
            rig,
            meta,
            options,
            frames,
            imu,
            imu_next: 0,
            pending: None,
            frames_done: false,
            first_t_ns,
            span_ns: last_t_ns - first_t_ns + LOOP_GAP_NS,
            loop_index: 0,
            frames_per_loop,
            pacing: None,
            stop,
            counts: ReplayCounts::default(),
        })
    }

    /// The dump's `meta.json`.
    pub fn meta(&self) -> &DumpMeta {
        &self.meta
    }

    /// The time shift between consecutive loops (`t` in loop `k` is the dump's `t + k * span`).
    pub fn loop_span_ns(&self) -> i64 {
        self.span_ns
    }

    /// The earliest time in the dump (frames or IMU).
    pub fn first_t_ns(&self) -> i64 {
        self.first_t_ns
    }

    fn shift(&self) -> i64 {
        self.loop_index * self.span_ns
    }

    fn next_frameset_raw(&mut self) -> Result<Option<Frameset>, SourceError> {
        match &mut self.frames {
            Frames::Reader(reader) => Ok(reader.next_frameset()?),
            Frames::Preloaded { framesets, next } => {
                let out = framesets.get(*next).cloned();
                *next += 1;
                Ok(out)
            }
        }
    }

    fn restart(&mut self) -> Result<(), SourceError> {
        self.loop_index += 1;
        self.counts.loops += 1;
        self.imu_next = 0;
        self.frames_done = false;
        match &mut self.frames {
            Frames::Reader(reader) => *reader = FrameReader::open(&self.dir.join("frames.bin"), FULL_SIZE)?,
            Frames::Preloaded { next, .. } => *next = 0,
        }
        Ok(())
    }

    fn shifted(&self, mut frameset: Frameset) -> Frameset {
        let shift = self.shift();
        if shift != 0 {
            frameset.t_ns += shift;
            frameset.index += self.loop_index as u64 * self.frames_per_loop;
            for frame in frameset.cameras.iter_mut().flatten() {
                frame.meta.pts_ns += shift;
                frame.meta.seq = frameset.index;
            }
        }
        frameset
    }

    fn pace(&mut self, t_ns: i64) {
        if !self.options.realtime {
            return;
        }
        let (start, t0) = *self.pacing.get_or_insert((Instant::now(), t_ns));
        let due = start + Duration::from_nanos(u64::try_from(t_ns - t0).unwrap_or(0));
        let now = Instant::now();
        if due > now {
            std::thread::sleep(due - now);
        } else if now - due > Duration::from_millis(5) {
            self.counts.late += 1;
        }
    }
}

fn preload(mut reader: FrameReader, stop: &AtomicBool) -> Result<Frames, SourceError> {
    let started = Instant::now();
    let mut framesets = Vec::new();
    while let Some(frameset) = reader.next_frameset()? {
        framesets.push(frameset);
        if stop.load(Ordering::Relaxed) {
            return Err(SourceError::Stopped("stopped while preloading".into()));
        }
    }
    eprintln!("robocap-live: preloaded {} framesets in {:.1} s", framesets.len(), started.elapsed().as_secs_f64());
    Ok(Frames::Preloaded { framesets, next: 0 })
}

impl FrameSource for ReplaySource {
    fn rig(&self) -> &Rig {
        &self.rig
    }

    fn next_event(&mut self) -> Result<Option<SourceEvent>, SourceError> {
        loop {
            if self.stop.load(Ordering::Relaxed) {
                return Ok(None);
            }
            if self.pending.is_none() && !self.frames_done {
                match self.next_frameset_raw()? {
                    Some(frameset) => self.pending = Some(self.shifted(frameset)),
                    None => self.frames_done = true,
                }
            }
            let shift = self.shift();
            let imu = self.imu.get(self.imu_next).map(|sample| ImuSample { t_ns: sample.t_ns + shift, ..*sample });
            let frame_t = self.pending.as_ref().map(|frameset| frameset.t_ns);
            if let Some(sample) = imu.filter(|sample| frame_t.is_none_or(|t| sample.t_ns <= t)) {
                self.imu_next += 1;
                self.pace(sample.t_ns);
                self.counts.imu += 1;
                return Ok(Some(SourceEvent::Imu(sample)));
            }
            if let Some(t) = frame_t {
                self.pace(t);
                let Some(frameset) = self.pending.take() else { continue };
                self.counts.framesets += 1;
                return Ok(Some(SourceEvent::Frameset(frameset)));
            }
            if !self.options.looping {
                return Ok(None);
            }
            self.restart()?;
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::Arc;

    use kornia_image::Image;

    use super::*;
    use crate::frame::{CAMERA_NAMES, CameraFrame, DUMP_FORMAT, FrameMeta, FrameWriter, Luma, NUM_CAMERAS, RigCamera, imu_to_bytes};

    /// Write a tiny dump: `frames` framesets 33.3 ms apart from t0 = 1 s (camera 3 missing in frameset 1), IMU at 200 Hz from
    /// 0.99 s to past the last frame, and reference poses (translation x = index).
    pub(crate) fn write_test_dump(dir: &Path, frames: u64) -> Result<(), Box<dyn std::error::Error>> {
        write_test_dump_with(dir, frames, &|index, camera| index == 1 && camera == 3)
    }

    /// [`write_test_dump`] with `missing(index, camera)` choosing the absent frames.
    pub(crate) fn write_test_dump_with(dir: &Path, frames: u64, missing: &dyn Fn(u64, usize) -> bool) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all(dir)?;
        let t0 = 1_000_000_000i64;
        let period = 33_333_333i64;
        let mut writer = FrameWriter::create(&dir.join("frames.bin"), FULL_SIZE)?;
        let mut reference = Vec::new();
        for index in 0..frames {
            let t = t0 + index as i64 * period;
            let mut cameras: [Option<CameraFrame>; NUM_CAMERAS] = Default::default();
            for (camera, slot) in cameras.iter_mut().enumerate() {
                if missing(index, camera) {
                    continue;
                }
                let image: Luma = Arc::new(Image::from_size_val(FULL_SIZE, (index as u8).wrapping_mul(7).wrapping_add(camera as u8))?);
                *slot = Some(CameraFrame { meta: FrameMeta { seq: index, pts_ns: t + camera as i64 * 1000, source_id: camera as u32, turned_180: false }, full: image });
            }
            writer.write(&Frameset { index, t_ns: t, cameras })?;
            reference.extend_from_slice(&t.to_le_bytes());
            let mut matrix = [0.0f64; 16];
            for d in 0..4 {
                matrix[5 * d] = 1.0;
            }
            matrix[3] = index as f64;
            for value in matrix {
                reference.extend_from_slice(&value.to_le_bytes());
            }
        }
        writer.finish()?;
        std::fs::write(dir.join("reference_world_from_rig.bin"), reference)?;
        let last = t0 + (frames as i64 - 1) * period;
        let mut imu = Vec::new();
        let mut t = t0 - 10_000_000;
        while t <= last + 10_000_000 {
            imu.extend_from_slice(&imu_to_bytes(&ImuSample { t_ns: t, gyro: [0.0; 3], accel: [0.0, 0.0, 9.81] }));
            t += 5_000_000;
        }
        std::fs::write(dir.join("imu.bin"), imu)?;
        let meta = DumpMeta {
            format: DUMP_FORMAT.into(),
            source: "test".into(),
            segment: "test".into(),
            device: "cap_a".into(),
            frames,
            width: 1920,
            height: 1080,
            cameras: CAMERA_NAMES.iter().map(|name| name.to_string()).collect(),
            first_t_ns: t0,
            last_t_ns: last,
        };
        std::fs::write(dir.join("meta.json"), serde_json::to_string(&meta)?)?;
        let identity = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        let rig = Rig {
            cameras: CAMERA_NAMES
                .iter()
                .map(|name| RigCamera {
                    name: name.to_string(),
                    width: 1920,
                    height: 1080,
                    cam_from_rig: identity,
                    focal: [600.0, 600.0],
                    principal: [960.0, 540.0],
                    fisheye62: Some([0.0; 8]),
                })
                .collect(),
            source: "test".into(),
            device: "cap_a".into(),
        };
        std::fs::write(dir.join("rig.json"), serde_json::to_string(&rig)?)?;
        Ok(())
    }

    fn events(source: &mut ReplaySource, limit: usize) -> Result<Vec<SourceEvent>, SourceError> {
        let mut out = Vec::new();
        while out.len() < limit {
            match source.next_event()? {
                Some(event) => out.push(event),
                None => break,
            }
        }
        Ok(out)
    }

    fn time(event: &SourceEvent) -> i64 {
        match event {
            SourceEvent::Imu(sample) => sample.t_ns,
            SourceEvent::Frameset(frameset) => frameset.t_ns,
        }
    }

    #[test]
    fn a_dump_replays_in_time_order_with_imu_first() -> Result<(), Box<dyn std::error::Error>> {
        let dir = std::env::temp_dir().join(format!("robocap-live-replay-order-{}", std::process::id()));
        write_test_dump(&dir, 3)?;
        for preload in [false, true] {
            let options = ReplayConfig { preload, ..Default::default() };
            let mut source = ReplaySource::open(&dir, options, Arc::new(AtomicBool::new(false)))?;
            let all = events(&mut source, 1000)?;
            assert!(all.windows(2).all(|pair| time(&pair[0]) <= time(&pair[1])), "time order");
            let framesets: Vec<&Frameset> = all.iter().filter_map(|e| if let SourceEvent::Frameset(f) = e { Some(f) } else { None }).collect();
            assert_eq!(framesets.iter().map(|f| f.index).collect::<Vec<_>>(), vec![0, 1, 2]);
            assert!(framesets[1].cameras[3].is_none() && framesets[1].cameras[2].is_some());
            assert_eq!(framesets[2].cameras[5].as_ref().map(|f| f.full.as_slice()[0]), Some(14 + 5));
            // The IMU sample at the frameset's exact time (1.0 s) comes before frameset 0.
            let first_frame = all.iter().position(|e| matches!(e, SourceEvent::Frameset(_))).unwrap_or(usize::MAX);
            assert_eq!(first_frame, 3, "IMU at 0.990, 0.995, 1.000 s precede the frameset at 1.000 s");
            assert_eq!(source.counts.imu as usize + 3, all.len());
            assert_eq!(read_reference_poses(&dir)?.len(), 3);
        }
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn a_dump_whose_meta_and_rig_name_different_devices_is_refused() -> Result<(), Box<dyn std::error::Error>> {
        let dir = std::env::temp_dir().join(format!("robocap-live-replay-device-{}", std::process::id()));
        write_test_dump(&dir, 1)?;
        let meta = DumpMeta { device: "cap_b".into(), ..DumpMeta::load(&dir)? };
        std::fs::write(dir.join("meta.json"), serde_json::to_string(&meta)?)?;
        assert!(ReplaySource::open(&dir, ReplayConfig::default(), Arc::new(AtomicBool::new(false))).is_err(), "the rig says cap_a");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn looping_shifts_times_forward_and_keeps_them_strictly_increasing() -> Result<(), Box<dyn std::error::Error>> {
        let dir = std::env::temp_dir().join(format!("robocap-live-replay-loop-{}", std::process::id()));
        write_test_dump(&dir, 3)?;
        let options = ReplayConfig { looping: true, preload: true, ..Default::default() };
        let mut source = ReplaySource::open(&dir, options, Arc::new(AtomicBool::new(false)))?;
        let span = source.loop_span_ns();
        assert_eq!(span, 1_075_000_000 - 990_000_000 + LOOP_GAP_NS, "IMU 0.990 .. 1.075 s spans the frames");
        let all = events(&mut source, 200)?;
        let imu: Vec<i64> = all.iter().filter_map(|e| if let SourceEvent::Imu(s) = e { Some(s.t_ns) } else { None }).collect();
        assert!(imu.windows(2).all(|pair| pair[1] > pair[0]), "IMU strictly increasing across loops");
        let framesets: Vec<&Frameset> = all.iter().filter_map(|e| if let SourceEvent::Frameset(f) = e { Some(f) } else { None }).collect();
        assert!(framesets.len() >= 6);
        assert_eq!((framesets[3].index, framesets[3].t_ns), (3, 1_000_000_000 + span));
        assert_eq!(framesets[3].cameras[1].as_ref().map(|f| f.meta.pts_ns), Some(1_000_000_000 + span + 1000));
        assert!(source.counts.loops >= 1);
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn realtime_pacing_follows_the_timestamps() -> Result<(), Box<dyn std::error::Error>> {
        let dir = std::env::temp_dir().join(format!("robocap-live-replay-pace-{}", std::process::id()));
        write_test_dump(&dir, 4)?;
        let options = ReplayConfig { realtime: true, preload: true, ..Default::default() };
        let mut source = ReplaySource::open(&dir, options, Arc::new(AtomicBool::new(false)))?;
        let started = Instant::now();
        let all = events(&mut source, 1000)?;
        let elapsed = started.elapsed().as_secs_f64();
        let span = (time(&all[all.len() - 1]) - time(&all[0])) as f64 / 1e9;
        assert!(elapsed >= span - 0.002 && elapsed < span + 0.05, "elapsed {elapsed:.3} s for a {span:.3} s clip");
        assert_eq!(source.counts.late, 0);
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }
}
