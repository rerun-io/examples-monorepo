//! Timestamp-tolerance grouping for runtime camera counts, independent of device sequence numbers.

use std::collections::VecDeque;
use std::time::Duration;

use crate::{CameraFrame, Frameset, SensorError};

/// Runtime matcher geometry and explicit source policy.
#[derive(Clone, Copy, Debug)]
pub struct MatcherConfig {
    /// Camera count, greater than zero.
    pub cameras: usize,
    /// Maximum difference from a group's first-arriving timestamp, nanoseconds.
    pub tolerance_ns: i64,
    /// Emit/resolve incomplete groups after a newer timestamp exceeds this bound.
    pub max_wait_ns: i64,
}

/// Counters of the matcher.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MatcherCounts {
    /// Framesets with every configured camera.
    pub complete: u64,
    /// Emitted groups with cameras missing.
    pub partial: u64,
    /// Frames that arrived for a camera slot already filled.
    pub duplicate: u64,
    /// Frames older than the last emitted frameset.
    pub late: u64,
}

struct Group {
    anchor_ns: i64,
    cameras: Vec<Option<CameraFrame>>,
}

/// Groups per-camera frames into [`Frameset`]s in time order.
///
/// Each camera must deliver in timestamp order. Cross-camera arrivals may interleave.
/// Late frames are counted and dropped, duplicate slots keep their first frame,
/// and settled or expired groups emit with any missing cameras left empty.
/// ```
/// use kornia_staging_sensors::{FramesetMatcher, MatcherConfig};
/// let mut matcher = FramesetMatcher::new(MatcherConfig { cameras: 7, tolerance_ns: 3_000_000,
///     max_wait_ns: 100_000_000 })?;
/// let mut remaining = Vec::new();
/// matcher.flush(&mut remaining);
/// assert!(remaining.is_empty());
/// # Ok::<(), kornia_staging_sensors::SensorError>(())
/// ```
pub struct FramesetMatcher {
    observer: Option<EmitObserver>,
    config: MatcherConfig,
    groups: VecDeque<Group>,
    newest_per_camera: Vec<Option<i64>>,
    newest_ns: Option<i64>,
    last_emitted_ns: Option<i64>,
    next_index: u64,
    /// What happened so far.
    pub counts: MatcherCounts,
}

type EmitObserver = Box<dyn Fn(&Frameset, i64, Duration) + Send>;

impl FramesetMatcher {
    /// Observe each emission without changing grouping, including flush emissions.
    /// # Arguments
    /// * `observer` - receives the frameset, first-arrival anchor and newest-member timestamp minus that anchor;
    ///   must not block or retain images.
    pub fn observe_emits(&mut self, observer: impl Fn(&Frameset, i64, Duration) + Send + 'static) {
        self.observer = Some(Box::new(observer));
    }
    /// Construct a matcher for a runtime number of cameras.
    /// # Arguments
    /// * `config` - camera count and nanosecond bounds.
    /// # Errors
    /// Rejects zero camera count or negative time bounds.
    pub fn new(config: MatcherConfig) -> Result<Self, SensorError> {
        if config.cameras == 0 || config.tolerance_ns < 0 || config.max_wait_ns < 0 {
            return Err(SensorError::InvalidConfig(
                "positive camera count and nonnegative matcher bounds required",
            ));
        }
        Ok(Self {
            observer: None,
            newest_per_camera: vec![None; config.cameras],
            config,
            groups: VecDeque::new(),
            newest_ns: None,
            last_emitted_ns: None,
            next_index: 0,
            counts: MatcherCounts::default(),
        })
    }

    /// Add a frame using its metadata camera slot and timestamp; append finalized framesets to `out`.
    /// # Arguments
    /// * `frame` - captured image and monotonic timestamp.
    /// * `out` - receives finalized groups in order.
    /// # Errors
    /// Invalid camera index.
    pub fn push(&mut self, frame: CameraFrame, out: &mut Vec<Frameset>) -> Result<(), SensorError> {
        let camera = frame.meta.camera_slot;
        let t = frame.meta.timestamp_ns;
        if camera >= self.config.cameras {
            return Err(SensorError::InvalidCamera {
                camera_slot: camera,
                cameras: self.config.cameras,
            });
        }
        if self
            .last_emitted_ns
            .is_some_and(|last| t as i128 <= last as i128 + self.config.tolerance_ns as i128)
        {
            self.counts.late += 1;
            return Ok(());
        }
        self.newest_per_camera[camera] =
            Some(self.newest_per_camera[camera].map_or(t, |newest| newest.max(t)));
        self.newest_ns = Some(self.newest_ns.map_or(t, |newest| newest.max(t)));
        match self
            .groups
            .iter_mut()
            .find(|group| group.anchor_ns.abs_diff(t) <= self.config.tolerance_ns as u64)
        {
            Some(group) if group.cameras[camera].is_some() => {
                self.counts.duplicate += 1;
            }
            Some(group) => group.cameras[camera] = Some(frame),
            None => {
                let at = self
                    .groups
                    .iter()
                    .position(|group| group.anchor_ns > t)
                    .unwrap_or(self.groups.len());
                let mut cameras = vec![None; self.config.cameras];
                cameras[camera] = Some(frame);
                self.groups.insert(
                    at,
                    Group {
                        anchor_ns: t,
                        cameras,
                    },
                );
            }
        }
        while let Some(front) = self.groups.front() {
            let anchor = front.anchor_ns;
            // A camera that already delivered a newer frame will not deliver one for this group (per-camera order).
            let settled =
                front
                    .cameras
                    .iter()
                    .zip(self.newest_per_camera.iter())
                    .all(|(slot, newest)| {
                        slot.is_some()
                            || newest.is_some_and(|newest| {
                                newest as i128 > anchor as i128 + self.config.tolerance_ns as i128
                            })
                    });
            let stale = self.newest_ns.is_some_and(|newest| {
                newest as i128 - anchor as i128 > self.config.max_wait_ns as i128
            });
            if !(settled || stale) {
                break;
            }
            let Some(group) = self.groups.pop_front() else {
                break;
            };
            out.push(self.emit(group));
        }
        Ok(())
    }

    /// Emit every pending group (end of capture).
    /// # Arguments
    /// * `out` - receives remaining framesets, including partial groups.
    pub fn flush(&mut self, out: &mut Vec<Frameset>) {
        while let Some(group) = self.groups.pop_front() {
            out.push(self.emit(group));
        }
    }

    fn emit(&mut self, group: Group) -> Frameset {
        let present = group.cameras.iter().filter(|slot| slot.is_some()).count();
        if present == self.config.cameras {
            self.counts.complete += 1;
        } else {
            self.counts.partial += 1;
        }
        let timestamp_ns = group
            .cameras
            .iter()
            .flatten()
            .map(|frame| frame.meta.timestamp_ns)
            .min()
            .unwrap_or(group.anchor_ns);
        let latest = group
            .cameras
            .iter()
            .flatten()
            .map(|frame| frame.meta.timestamp_ns)
            .max()
            .unwrap_or(group.anchor_ns);
        self.last_emitted_ns = Some(self.last_emitted_ns.map_or(latest, |last| last.max(latest)));
        let index = self.next_index;
        self.next_index += 1;
        let frameset = Frameset {
            index,
            timestamp_ns,
            cameras: group.cameras,
        };
        if let Some(observe) = &self.observer {
            observe(
                &frameset,
                group.anchor_ns,
                Duration::from_nanos(latest.saturating_sub(group.anchor_ns).max(0) as u64),
            );
        }
        frameset
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use kornia_image::{Image, ImageSize};

    use super::*;
    use crate::CaptureMeta;

    fn config(cameras: usize, tolerance_ns: i64, max_wait_ns: i64) -> MatcherConfig {
        MatcherConfig {
            cameras,
            tolerance_ns,
            max_wait_ns,
        }
    }

    fn frame(camera: usize, t: i64) -> Result<CameraFrame, kornia_image::ImageError> {
        let image = Image::new(
            ImageSize {
                width: 2,
                height: 1,
            },
            vec![camera as u8; 2],
        )?;
        Ok(CameraFrame {
            meta: CaptureMeta {
                sequence: 0,
                timestamp_ns: t,
                camera_slot: camera,
            },
            full: Arc::new(image),
        })
    }

    #[test]
    fn frames_within_the_tolerance_form_framesets_in_time_order(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let ms = 1_000_000;
        let mut matcher = FramesetMatcher::new(config(6, 3 * ms, 100 * ms))?;
        let mut out = Vec::new();
        // Trigger instants 0, 33, 66 ms; cameras deliver with up to 2 ms of skew and in a scrambled order across cameras.
        for camera in [3, 0, 5, 1, 2] {
            matcher.push(frame(camera, camera as i64 * 400_000)?, &mut out)?;
        }
        assert!(out.is_empty(), "camera 4 has not delivered yet");
        matcher.push(frame(4, 2 * ms)?, &mut out)?;
        assert_eq!(out.len(), 1);
        assert_eq!((out[0].index, out[0].timestamp_ns), (0, 0));
        assert!(out[0].cameras.iter().all(Option::is_some));
        // Camera 2 skips instant 33: the set is emitted once camera 2 delivers instant 66.
        for camera in [0, 1, 3, 4, 5] {
            matcher.push(frame(camera, 33 * ms + camera as i64)?, &mut out)?;
        }
        assert_eq!(out.len(), 1);
        matcher.push(frame(2, 66 * ms)?, &mut out)?;
        assert_eq!(out.len(), 2);
        assert!(out[1].cameras[2].is_none() && out[1].cameras[0].is_some());
        assert_eq!(out[1].timestamp_ns, 33 * ms);
        // A duplicate slot and a frame older than the emitted sets are counted and dropped.
        matcher.push(frame(2, 66 * ms + 1)?, &mut out)?;
        matcher.push(frame(0, 30 * ms)?, &mut out)?;
        assert_eq!((matcher.counts.duplicate, matcher.counts.late), (1, 1));
        // A camera that stops: its group goes out partial once frames 100 ms newer exist.
        matcher.push(frame(0, 170 * ms)?, &mut out)?;
        assert_eq!(out.len(), 3);
        assert_eq!(out[2].timestamp_ns, 66 * ms);
        assert_eq!((matcher.counts.complete, matcher.counts.partial), (1, 2));
        matcher.flush(&mut out);
        assert_eq!(out.len(), 4);
        assert!(out
            .windows(2)
            .all(|pair| pair[0].timestamp_ns < pair[1].timestamp_ns
                && pair[0].index + 1 == pair[1].index));
        Ok(())
    }

    #[test]
    fn diagnostic_emits_keep_first_arrival_anchor_and_member_timestamps(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut matcher = FramesetMatcher::new(config(2, 3, 10))?;
        let sink = seen.clone();
        matcher.observe_emits(move |set, anchor, wait| {
            sink.lock()
                .unwrap()
                .push((set.index, anchor, set.timestamp_ns, wait));
        });
        let mut out = Vec::new();
        matcher.push(frame(0, 12)?, &mut out)?;
        matcher.push(frame(1, 10)?, &mut out)?;
        matcher.push(frame(0, 30)?, &mut out)?;
        matcher.push(frame(1, 35)?, &mut out)?;
        matcher.flush(&mut out);
        let seen = seen.lock().unwrap();
        assert_eq!(
            seen.iter().map(|x| (x.0, x.1, x.2)).collect::<Vec<_>>(),
            [(0, 12, 10), (1, 30, 30), (2, 35, 35)]
        );
        assert!(seen.iter().all(|row| row.3 == Duration::ZERO));
        assert_eq!((matcher.counts.complete, matcher.counts.partial), (1, 2));
        Ok(())
    }

    #[test]
    fn runtime_two_camera_policies_and_extreme_times_are_explicit(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut matcher = FramesetMatcher::new(config(2, 3, 10))?;
        let mut out = Vec::new();
        matcher.push(frame(0, i64::MIN)?, &mut out)?;
        let mut replacement = frame(0, i64::MIN + 1)?;
        replacement.meta.sequence = 7;
        matcher.push(replacement, &mut out)?;
        matcher.push(frame(1, i64::MIN + 2)?, &mut out)?;
        assert_eq!(out[0].cameras.len(), 2);
        assert_eq!(out[0].cameras[0].as_ref().unwrap().meta.sequence, 0);
        matcher.push(frame(0, i64::MAX)?, &mut out)?;
        matcher.flush(&mut out);
        assert_eq!(out.len(), 2);
        assert_eq!(matcher.counts.duplicate, 1);
        assert!(FramesetMatcher::new(config(0, 0, 0)).is_err());
        assert!(FramesetMatcher::new(config(2, -1, 0)).is_err());
        assert!(matches!(
            matcher.push(frame(2, 20)?, &mut out),
            Err(SensorError::InvalidCamera { .. })
        ));
        Ok(())
    }
}
