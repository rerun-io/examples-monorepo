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

    /// Clone the first queued item that matches without removing it or waiting for more input.
    /// A drop-oldest producer may replace it later; SLAM treats it only as a lookahead hint.
    pub fn peek_matching(&self, matches: impl Fn(&T) -> bool) -> Option<T>
    where
        T: Clone,
    {
        lock(&self.state).items.iter().find(|item| matches(item)).cloned()
    }
}

struct PoseState {
    poses: VecDeque<SlamPose>,
    /// World boundaries, including the index of any pending frame discarded by a reset.
    resets: VecDeque<u64>,
    /// SLAM has consumed (processed or skipped) every frameset up to this index.
    progress_index: Option<u64>,
    closed: bool,
}

/// The SLAM poses published so far, for the hands and output stages.
pub(crate) struct PoseStore {
    state: Mutex<PoseState>,
    changed: Condvar,
    lossless: bool,
}

impl PoseStore {
    const KEEP: usize = 128;

    /// Live readers can use an older usable pose as soon as SLAM consumes their frame.
    /// Lossless readers wait for publication and prefer the exact frame, including a not-yet-tracking pose.
    pub fn new(lossless: bool) -> Self {
        Self {
            state: Mutex::new(PoseState { poses: VecDeque::new(), resets: VecDeque::new(), progress_index: None, closed: false }),
            changed: Condvar::new(),
            lossless,
        }
    }

    /// A reset starts a new world: old-world poses must not be used for new frames.
    pub fn reset(&self, index: u64) {
        let mut state = lock(&self.state);
        // Published poses precede the boundary; preserve them for queued old-world readers.
        state.resets.push_back(index);
        if state.resets.len() > Self::KEEP {
            state.resets.pop_front();
        }
    }

    /// SLAM consumed `index`. In lossless mode, do not pass an accepted frame still awaiting publication.
    pub fn consumed(&self, index: u64, pending_index: Option<u64>) {
        let progress = if self.lossless {
            pending_index.map_or(Some(index), |pending| pending.checked_sub(1).map(|before| index.min(before)))
        } else {
            Some(index)
        };
        if let Some(index) = progress {
            self.progress(index);
        }
    }

    /// Publish a pose (and SLAM's progress to its index).
    pub fn publish(&self, pose: SlamPose) {
        let mut state = lock(&self.state);
        state.progress_index = state.progress_index.max(Some(pose.index));
        state.poses.push_back(pose);
        if state.poses.len() > Self::KEEP {
            state.poses.pop_front();
        }
        self.changed.notify_all();
    }

    /// SLAM consumed the frameset at `index` without a pose (rate skip, missing input).
    pub fn progress(&self, index: u64) {
        let mut state = lock(&self.state);
        state.progress_index = state.progress_index.max(Some(index));
        self.changed.notify_all();
    }

    /// SLAM ended; waiters stop waiting.
    pub fn close(&self) {
        lock(&self.state).closed = true;
        self.changed.notify_all();
    }

    /// Wait until SLAM has passed `index` (up to `wait`, unbounded when `None`), then return a pose and the wait.
    /// Lossless prefers the exact frame, of any status; live prefers the newest usable pose at or before `index`.
    /// Both fall back to the newest pose of any status in the same world, or none.
    pub fn pose_for(&self, index: u64, wait: Option<Duration>) -> (Option<SlamPose>, Duration) {
        let started = Instant::now();
        let mut state = lock(&self.state);
        while state.progress_index.is_none_or(|progress| progress < index) && !state.closed {
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
        let world_start = state.resets.iter().rev().find(|&&reset| reset <= index).copied().unwrap_or(0);
        let at_or_before = state.poses.iter().rev().filter(|pose| pose.index <= index && pose.index >= world_start);
        let pose = if self.lossless {
            at_or_before.clone().next()
        } else {
            at_or_before.clone().find(|pose| pose.ok).or_else(|| at_or_before.clone().next())
        }
        .copied();
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
        let store = Arc::new(PoseStore::new(false));
        let pose = |index: u64, t: i64, ok: bool| SlamPose {
            world_from_rig: Isometry3::translation(index as f64, 0.0, 0.0),
            ok,
            compute_ms: 1.0,
            ..SlamPose::untracked(index, t, if ok { SlamStatus::Tracking } else { SlamStatus::NoVisualFeatures })
        };
        store.publish(pose(0, 100, true));
        store.publish(pose(1, 200, false));
        let (found, _) = store.pose_for(2, Some(Duration::from_millis(5)));
        assert_eq!(found.map(|p| p.index), Some(0), "newest ok at or before input 2 (the timeout passed)");
        let waiter = {
            let store = store.clone();
            thread::spawn(move || store.pose_for(2, None).0.map(|p| p.index))
        };
        thread::sleep(Duration::from_millis(20));
        store.publish(pose(2, 300, true));
        assert_eq!(waiter.join().ok().flatten(), Some(2));
    }

    #[test]
    fn live_consumption_does_not_wait_for_the_lagged_pose() {
        let store = PoseStore::new(false);
        store.publish(SlamPose { ok: true, ..SlamPose::untracked(7, 100, SlamStatus::Tracking) });
        store.consumed(19, Some(19));
        let (pose, waited) = store.pose_for(19, Some(Duration::from_secs(1)));
        assert_eq!(pose.map(|p| (p.index, p.t_ns)), Some((7, 100)));
        assert!(waited < Duration::from_millis(100));
    }

    #[test]
    fn lossless_consumption_waits_for_publication_even_if_the_exact_pose_is_not_ok() -> Result<(), Box<dyn std::error::Error>> {
        let store = Arc::new(PoseStore::new(true));
        store.publish(SlamPose { ok: true, ..SlamPose::untracked(7, 100, SlamStatus::Tracking) });
        store.consumed(19, Some(19));
        let (tx, rx) = std::sync::mpsc::channel();
        let waiter = {
            let store = store.clone();
            thread::spawn(move || tx.send(store.pose_for(19, None).0.map(|p| (p.index, p.t_ns))))
        };
        assert!(matches!(rx.recv_timeout(Duration::from_millis(30)), Err(std::sync::mpsc::RecvTimeoutError::Timeout)));
        store.publish(SlamPose::untracked(19, 200, SlamStatus::NoVisualFeatures));
        assert_eq!(rx.recv_timeout(Duration::from_secs(1))?, Some((19, 200)));
        waiter.join().map_err(|_| "pose waiter panicked")??;
        store.reset(20);
        assert!(store.pose_for(7, Some(Duration::ZERO)).0.is_some(), "queued readers of the old world retain their published poses");
        assert_eq!(store.pose_for(19, Some(Duration::ZERO)).0.map(|p| p.index), Some(19));
        assert!(store.pose_for(20, Some(Duration::ZERO)).0.is_none(), "new-world readers cannot use old-world poses");
        Ok(())
    }

    #[test]
    fn lookahead_peeks_a_selected_frame_without_consuming_it() {
        let queue = StageQueue::new(3, QueuePolicy::DropOldest);
        for t in [1, 2, 3] {
            queue.push(t);
        }
        assert_eq!(queue.peek_matching(|&t| t >= 2), Some(2));
        assert_eq!(queue.pop(), Some(1));
        assert_eq!(queue.pop(), Some(2));
        assert_eq!(queue.pop(), Some(3));
        assert_eq!(queue.peek_matching(|_| true), None);
    }
}
