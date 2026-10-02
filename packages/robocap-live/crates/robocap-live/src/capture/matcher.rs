//! Frameset assembly for the live cameras: frames whose timestamps lie within a tolerance (3 ms) of each other belong to one
//! trigger instant. The cameras start at different V4L2 sequence numbers, so timestamps, not sequences, decide (PR #270).

use std::collections::VecDeque;

use crate::frame::{CameraFrame, Frameset, NUM_CAMERAS};

/// Counters of the matcher.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MatcherCounts {
    /// Framesets with all six cameras.
    pub complete: u64,
    /// Framesets emitted with cameras missing.
    pub partial: u64,
    /// Frames that arrived for a camera slot already filled.
    pub duplicate: u64,
    /// Frames older than the last emitted frameset.
    pub late: u64,
}

struct Group {
    anchor_ns: i64,
    cameras: [Option<CameraFrame>; NUM_CAMERAS],
}

/// Groups per-camera frames into [`Frameset`]s in time order.
pub struct FramesetMatcher {
    tolerance_ns: i64,
    max_wait_ns: i64,
    groups: VecDeque<Group>,
    newest_per_camera: [Option<i64>; NUM_CAMERAS],
    newest_ns: Option<i64>,
    last_emitted_ns: Option<i64>,
    next_index: u64,
    /// What happened so far.
    pub counts: MatcherCounts,
}

impl FramesetMatcher {
    /// `tolerance_ns`: frames this close to a group's first frame join it. `max_wait_ns`: a group still incomplete when a frame
    /// this much newer has arrived is emitted with the cameras it has (a camera that stopped delivering).
    pub fn new(tolerance_ns: i64, max_wait_ns: i64) -> Self {
        Self {
            tolerance_ns,
            max_wait_ns,
            groups: VecDeque::new(),
            newest_per_camera: [None; NUM_CAMERAS],
            newest_ns: None,
            last_emitted_ns: None,
            next_index: 0,
            counts: MatcherCounts::default(),
        }
    }

    /// Add camera `camera`'s frame (its `meta.pts_ns` is the timestamp) and append every frameset that became final to `out`.
    pub fn push(&mut self, camera: usize, frame: CameraFrame, out: &mut Vec<Frameset>) {
        let t = frame.meta.pts_ns;
        if camera >= NUM_CAMERAS || self.last_emitted_ns.is_some_and(|last| t <= last + self.tolerance_ns) {
            self.counts.late += 1;
            return;
        }
        self.newest_per_camera[camera] = Some(self.newest_per_camera[camera].map_or(t, |newest| newest.max(t)));
        self.newest_ns = Some(self.newest_ns.map_or(t, |newest| newest.max(t)));
        match self.groups.iter_mut().find(|group| (group.anchor_ns - t).abs() <= self.tolerance_ns) {
            Some(group) if group.cameras[camera].is_some() => self.counts.duplicate += 1,
            Some(group) => group.cameras[camera] = Some(frame),
            None => {
                let at = self.groups.iter().position(|group| group.anchor_ns > t).unwrap_or(self.groups.len());
                let mut cameras: [Option<CameraFrame>; NUM_CAMERAS] = Default::default();
                cameras[camera] = Some(frame);
                self.groups.insert(at, Group { anchor_ns: t, cameras });
            }
        }
        while let Some(front) = self.groups.front() {
            let anchor = front.anchor_ns;
            // A camera that already delivered a newer frame will not deliver one for this group (per-camera order).
            let settled = front.cameras.iter().zip(self.newest_per_camera.iter()).all(|(slot, newest)| {
                slot.is_some() || newest.is_some_and(|newest| newest > anchor + self.tolerance_ns)
            });
            let stale = self.newest_ns.is_some_and(|newest| newest - anchor > self.max_wait_ns);
            if !(settled || stale) {
                break;
            }
            let Some(group) = self.groups.pop_front() else { break };
            out.push(self.emit(group));
        }
    }

    /// Emit every pending group (end of capture).
    pub fn flush(&mut self, out: &mut Vec<Frameset>) {
        while let Some(group) = self.groups.pop_front() {
            out.push(self.emit(group));
        }
    }

    fn emit(&mut self, group: Group) -> Frameset {
        let present = group.cameras.iter().filter(|slot| slot.is_some()).count();
        if present == NUM_CAMERAS {
            self.counts.complete += 1;
        } else {
            self.counts.partial += 1;
        }
        let t_ns = group.cameras.iter().flatten().map(|frame| frame.meta.pts_ns).min().unwrap_or(group.anchor_ns);
        let latest = group.cameras.iter().flatten().map(|frame| frame.meta.pts_ns).max().unwrap_or(group.anchor_ns);
        self.last_emitted_ns = Some(self.last_emitted_ns.map_or(latest, |last| last.max(latest)));
        let index = self.next_index;
        self.next_index += 1;
        Frameset { index, t_ns, cameras: group.cameras }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use kornia_image::{Image, ImageSize};

    use super::*;
    use crate::frame::FrameMeta;

    fn frame(camera: usize, t: i64) -> Result<CameraFrame, kornia_image::ImageError> {
        let image = Image::new(ImageSize { width: 2, height: 1 }, vec![camera as u8; 2])?;
        Ok(CameraFrame { meta: FrameMeta { seq: 0, pts_ns: t, source_id: camera as u32, turned_180: false }, full: Arc::new(image) })
    }

    #[test]
    fn frames_within_the_tolerance_form_framesets_in_time_order() -> Result<(), kornia_image::ImageError> {
        let ms = 1_000_000;
        let mut matcher = FramesetMatcher::new(3 * ms, 100 * ms);
        let mut out = Vec::new();
        // Trigger instants 0, 33, 66 ms; cameras deliver with up to 2 ms of skew and in a scrambled order across cameras.
        for camera in [3, 0, 5, 1, 2] {
            matcher.push(camera, frame(camera, camera as i64 * 400_000)?, &mut out);
        }
        assert!(out.is_empty(), "camera 4 has not delivered yet");
        matcher.push(4, frame(4, 2 * ms)?, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!((out[0].index, out[0].t_ns), (0, 0));
        assert!(out[0].cameras.iter().all(Option::is_some));
        // Camera 2 skips instant 33: the set is emitted once camera 2 delivers instant 66.
        for camera in [0, 1, 3, 4, 5] {
            matcher.push(camera, frame(camera, 33 * ms + camera as i64)?, &mut out);
        }
        assert_eq!(out.len(), 1);
        matcher.push(2, frame(2, 66 * ms)?, &mut out);
        assert_eq!(out.len(), 2);
        assert!(out[1].cameras[2].is_none() && out[1].cameras[0].is_some());
        assert_eq!(out[1].t_ns, 33 * ms);
        // A duplicate slot and a frame older than the emitted sets are counted and dropped.
        matcher.push(2, frame(2, 66 * ms + 1)?, &mut out);
        matcher.push(0, frame(0, 30 * ms)?, &mut out);
        assert_eq!((matcher.counts.duplicate, matcher.counts.late), (1, 1));
        // A camera that stops: its group goes out partial once frames 100 ms newer exist.
        matcher.push(0, frame(0, 170 * ms)?, &mut out);
        assert_eq!(out.len(), 3);
        assert_eq!(out[2].t_ns, 66 * ms);
        assert_eq!((matcher.counts.complete, matcher.counts.partial), (1, 2));
        matcher.flush(&mut out);
        assert_eq!(out.len(), 4);
        assert!(out.windows(2).all(|pair| pair[0].t_ns < pair[1].t_ns && pair[0].index + 1 == pair[1].index));
        Ok(())
    }
}
