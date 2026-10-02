//! The stages' bounded queues (drop-oldest or blocking) and the store of SLAM's published poses.

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crate::slam::SlamPose;

/// What a full queue does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QueuePolicy {
    /// Drop the oldest waiting item (and count it).
    DropOldest,
    /// Block the producer.
    Block,
}

struct QueueState<T> {
    items: VecDeque<T>,
    closed: bool,
    dropped: u64,
}

/// A bounded MPSC queue with a drop-oldest or blocking policy, closable from either side.
pub(crate) struct StageQueue<T> {
    state: Mutex<QueueState<T>>,
    changed: Condvar,
    capacity: usize,
    policy: QueuePolicy,
}

/// Lock `mutex`, also after a panic in another stage (the data is counters and queues, valid at every step).
pub(super) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl<T> StageQueue<T> {
    /// A queue of `capacity` (at least 1) items.
    pub fn new(capacity: usize, policy: QueuePolicy) -> Self {
        Self {
            state: Mutex::new(QueueState { items: VecDeque::new(), closed: false, dropped: 0 }),
            changed: Condvar::new(),
            capacity: capacity.max(1),
            policy,
        }
    }

    /// Add an item; returns false when the queue is closed (the item is dropped).
    pub fn push(&self, item: T) -> bool {
        let mut evicted = Vec::new();
        self.push_evicting(item, &mut evicted)
    }

    /// [`StageQueue::push`], handing the items the drop-oldest policy evicts to `evicted` instead of dropping them (they are
    /// still counted as dropped).
    pub fn push_evicting(&self, item: T, evicted: &mut Vec<T>) -> bool {
        let mut state = lock(&self.state);
        while state.items.len() >= self.capacity && !state.closed {
            match self.policy {
                QueuePolicy::DropOldest => {
                    if let Some(old) = state.items.pop_front() {
                        evicted.push(old);
                    }
                    state.dropped += 1;
                }
                QueuePolicy::Block => state = self.changed.wait(state).unwrap_or_else(std::sync::PoisonError::into_inner),
            }
        }
        if state.closed {
            return false;
        }
        state.items.push_back(item);
        self.changed.notify_all();
        true
    }

    /// The oldest item, waiting for one; `None` once the queue is closed and empty.
    pub fn pop(&self) -> Option<T> {
        let mut state = lock(&self.state);
        loop {
            if let Some(item) = state.items.pop_front() {
                self.changed.notify_all();
                return Some(item);
            }
            if state.closed {
                return None;
            }
            state = self.changed.wait(state).unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    /// No more items will be pushed; waiting consumers drain what is left and then get `None`.
    pub fn close(&self) {
        lock(&self.state).closed = true;
        self.changed.notify_all();
    }

    /// Items dropped by the drop-oldest policy.
    pub fn dropped(&self) -> u64 {
        lock(&self.state).dropped
    }
}


struct PoseState {
    poses: VecDeque<SlamPose>,
    /// SLAM has consumed (processed or skipped) every frameset up to this time.
    progress_ns: i64,
    closed: bool,
}

/// The SLAM poses published so far, for the hands and output stages.
pub(crate) struct PoseStore {
    state: Mutex<PoseState>,
    changed: Condvar,
}

impl Default for PoseStore {
    fn default() -> Self {
        Self { state: Mutex::new(PoseState { poses: VecDeque::new(), progress_ns: i64::MIN, closed: false }), changed: Condvar::new() }
    }
}

impl PoseStore {
    /// Publish a pose (and SLAM's progress to its time).
    pub fn publish(&self, pose: SlamPose) {
        let mut state = lock(&self.state);
        state.progress_ns = state.progress_ns.max(pose.t_ns);
        state.poses.push_back(pose);
        if state.poses.len() > 128 {
            state.poses.pop_front();
        }
        self.changed.notify_all();
    }

    /// SLAM consumed the frameset at `t_ns` without a pose (rate skip, missing input).
    pub fn progress(&self, t_ns: i64) {
        let mut state = lock(&self.state);
        state.progress_ns = state.progress_ns.max(t_ns);
        self.changed.notify_all();
    }

    /// SLAM ended; waiters stop waiting.
    pub fn close(&self) {
        lock(&self.state).closed = true;
        self.changed.notify_all();
    }

    /// The newest usable pose at or before `t_ns`, after waiting until SLAM has passed `t_ns` (up to `wait`, or without a
    /// bound when `None`). Returns the pose (or the newest pose at or before `t_ns` of any status, or none) and the wait.
    pub fn pose_for(&self, t_ns: i64, wait: Option<Duration>) -> (Option<SlamPose>, Duration) {
        let started = Instant::now();
        let mut state = lock(&self.state);
        while state.progress_ns < t_ns && !state.closed {
            match wait {
                None => state = self.changed.wait(state).unwrap_or_else(std::sync::PoisonError::into_inner),
                Some(limit) => {
                    let elapsed = started.elapsed();
                    if elapsed >= limit {
                        break;
                    }
                    state = self.changed.wait_timeout(state, limit - elapsed).unwrap_or_else(std::sync::PoisonError::into_inner).0;
                }
            }
        }
        let at_or_before = state.poses.iter().rev().filter(|pose| pose.t_ns <= t_ns);
        let pose = at_or_before.clone().find(|pose| pose.ok).or_else(|| at_or_before.clone().next()).copied();
        (pose, started.elapsed())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;

    use nalgebra::Isometry3;

    use super::*;
    use crate::slam::SlamStatus;

    #[test]
    fn a_drop_oldest_queue_keeps_the_newest_and_counts() {
        let queue = StageQueue::new(1, QueuePolicy::DropOldest);
        assert!(queue.push(1) && queue.push(2) && queue.push(3));
        assert_eq!(queue.dropped(), 2);
        assert_eq!(queue.pop(), Some(3));
        queue.close();
        assert!(!queue.push(4));
        assert_eq!(queue.pop(), None);
    }

    #[test]
    fn a_blocking_queue_hands_every_item_over() {
        let queue = Arc::new(StageQueue::new(2, QueuePolicy::Block));
        let producer = {
            let queue = queue.clone();
            thread::spawn(move || {
                for i in 0..100 {
                    queue.push(i);
                }
                queue.close();
            })
        };
        let mut got = Vec::new();
        while let Some(i) = queue.pop() {
            got.push(i);
        }
        let _ = producer.join();
        assert_eq!(got, (0..100).collect::<Vec<_>>());
        assert_eq!(queue.dropped(), 0);
    }

    #[test]
    fn the_pose_store_returns_the_newest_ok_pose_at_or_before_after_progress() {
        let store = Arc::new(PoseStore::default());
        let pose = |index: u64, t: i64, ok: bool| SlamPose {
            world_from_rig: Isometry3::translation(index as f64, 0.0, 0.0),
            ok,
            compute_ms: 1.0,
            ..SlamPose::untracked(index, t, if ok { SlamStatus::Tracking } else { SlamStatus::NoVisualFeatures })
        };
        store.publish(pose(0, 100, true));
        store.publish(pose(1, 200, false));
        let (found, _) = store.pose_for(250, Some(Duration::from_millis(5)));
        assert_eq!(found.map(|p| p.index), Some(0), "newest ok at or before 250 (the timeout passed)");
        let waiter = {
            let store = store.clone();
            thread::spawn(move || store.pose_for(300, None).0.map(|p| p.index))
        };
        thread::sleep(Duration::from_millis(20));
        store.publish(pose(2, 300, true));
        assert_eq!(waiter.join().ok().flatten(), Some(2));
    }
}
