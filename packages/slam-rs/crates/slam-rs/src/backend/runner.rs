//! The back end's two modes over one synchronous step (DESIGN §2, PR a2).
//!
//! [`BackendRunner`] wraps any [`BackendWork`] (the map; from b2 also loop
//! closing). The work has two parts:
//!
//! - [`BackendWork::step`] takes every [`MapStamp`], in order. It is never
//!   skipped: a lost stamp would lose a keyframe or keep an evicted one active.
//! - [`BackendWork::query`] is the loop query of one new keyframe. It may be
//!   dropped when the back end is behind.
//!
//! [`BackendMode::Offline`] runs both inline: tracking waits, every query runs,
//! and two runs give the same map. [`BackendMode::Realtime`] runs them on one
//! thread. Tracking only sends: the channel has no bound, so a send never waits
//! (the stamps are small and the step is fast; the queries are the slow part).
//! The thread merges every waiting stamp before it starts a query, and it keeps
//! at most `max_queued_queries` queries. One more drops the oldest, which the
//! next keyframe makes stale anyway, and counts it.
//!
//! The work sits behind a mutex so a caller can read it ([`BackendRunner::read`]).
//! A read waits at most for the step or query in progress; `submit` never locks.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;

use super::{MapEvent, StepTiming};
use crate::config::BackendMode;
use crate::estimator::MapStamp;
use crate::types::FrameId;

/// What the back end does with the window's stamps.
pub trait BackendWork: Send + 'static {
    /// Take one stamp. Every stamp reaches this, in order.
    fn step(&mut self, stamp: MapStamp) -> Vec<MapEvent>;

    /// Look for a loop at `keyframe`, which a step added. Realtime may drop it.
    fn query(&mut self, keyframe: FrameId) -> Vec<MapEvent>;
}

/// What the back end has done so far.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BackendCounters {
    /// Stamps taken.
    pub stamps: u64,
    /// Loop queries run.
    pub queries_run: u64,
    /// Loop queries dropped because the back end was behind (Realtime only).
    pub queries_dropped: u64,
    /// Most stamps ever sent and not yet taken (Realtime; 0 Offline).
    pub max_pending_stamps: u64,
    /// Wall time of the steps.
    pub step: StepTiming,
    /// Wall time of the queries.
    pub query: StepTiming,
}

/// The work and what is waiting for it, behind the runner's mutex.
struct State<B> {
    work: B,
    counters: BackendCounters,
    /// Keyframes whose loop query has not run yet, oldest first.
    queries: VecDeque<FrameId>,
}

impl<B: BackendWork> State<B> {
    /// Step one stamp and queue a query for each keyframe it added; past
    /// `max_queries` queued, drop the oldest.
    fn step(&mut self, stamp: MapStamp, max_queries: usize) {
        let started = std::time::Instant::now();
        let events: Vec<MapEvent> = self.work.step(stamp);
        self.counters.step.record(crate::duration_ns(started));
        self.counters.stamps += 1;
        for event in events {
            if let MapEvent::KeyframeAdded(id) = event {
                self.queries.push_back(id);
            }
        }
        while self.queries.len() > max_queries {
            self.queries.pop_front();
            self.counters.queries_dropped += 1;
        }
    }

    /// Run the oldest queued query; `false` if none was queued.
    fn query_one(&mut self) -> bool {
        let Some(keyframe) = self.queries.pop_front() else {
            return false;
        };
        let started = std::time::Instant::now();
        self.work.query(keyframe);
        self.counters.query.record(crate::duration_ns(started));
        self.counters.queries_run += 1;
        true
    }
}

/// Lock the state. A poisoned lock means the back-end thread panicked; that
/// panic is re-raised by [`BackendRunner::flush`], so a read here re-raises it.
fn lock<B>(state: &Mutex<State<B>>) -> MutexGuard<'_, State<B>> {
    match state.lock() {
        Ok(guard) => guard,
        Err(_) => panic!("the back-end thread panicked"),
    }
}

enum Message {
    Stamp(MapStamp),
    /// Answer once every earlier stamp is stepped and every queued query ran.
    Flush(mpsc::SyncSender<()>),
}

/// The Realtime thread and the tracking side of its channel.
struct Thread {
    sender: Option<Sender<Message>>,
    handle: Option<JoinHandle<()>>,
    /// Stamps sent and not yet taken.
    pending: Arc<AtomicU64>,
    max_pending: u64,
}

/// The back end in one of its two modes.
pub struct BackendRunner<B: BackendWork> {
    mode: BackendMode,
    state: Arc<Mutex<State<B>>>,
    /// `Some` in Realtime.
    thread: Option<Thread>,
}

impl<B: BackendWork> BackendRunner<B> {
    /// Run `work` in `mode`. Realtime starts the back-end thread here and keeps
    /// at most `max_queued_queries` loop queries waiting (Offline ignores it).
    ///
    /// # Errors
    ///
    /// The operating system's refusal to start the thread.
    pub fn new(work: B, mode: BackendMode, max_queued_queries: usize) -> std::io::Result<Self> {
        let state = Arc::new(Mutex::new(State {
            work,
            counters: BackendCounters::default(),
            queries: VecDeque::new(),
        }));
        let mut thread: Option<Thread> = None;
        if mode == BackendMode::Realtime {
            let (sender, receiver) = mpsc::channel::<Message>();
            let pending = Arc::new(AtomicU64::new(0));
            let handle = std::thread::Builder::new()
                .name("slam-rs-backend".into())
                .spawn({
                    let state = Arc::clone(&state);
                    let pending = Arc::clone(&pending);
                    move || run_realtime(&state, &receiver, &pending, max_queued_queries)
                })?;
            thread = Some(Thread {
                sender: Some(sender),
                handle: Some(handle),
                pending,
                max_pending: 0,
            });
        }
        Ok(Self {
            mode,
            state,
            thread,
        })
    }

    /// Where the back end runs.
    pub fn mode(&self) -> BackendMode {
        self.mode
    }

    /// Hand the back end one stamp. Offline steps it and runs its queries
    /// before returning. Realtime sends it and returns at once.
    pub fn submit(&mut self, stamp: MapStamp) {
        let Some(thread) = self.thread.as_mut() else {
            let mut state = lock(&self.state);
            state.step(stamp, usize::MAX);
            while state.query_one() {}
            return;
        };
        let pending: u64 = thread.pending.fetch_add(1, Ordering::Relaxed) + 1;
        thread.max_pending = thread.max_pending.max(pending);
        if let Some(sender) = thread.sender.as_ref()
            && sender.send(Message::Stamp(stamp)).is_err()
        {
            // The thread is gone (it panicked); `flush` re-raises its panic.
            thread.pending.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Wait until the back end has taken every stamp submitted so far and run
    /// every query still queued. Immediate in Offline.
    ///
    /// # Panics
    ///
    /// Re-raises a panic of the back-end thread.
    pub fn flush(&mut self) {
        let Some(thread) = self.thread.as_mut() else {
            return;
        };
        let (ack, done) = mpsc::sync_channel::<()>(1);
        let delivered: bool = thread
            .sender
            .as_ref()
            .is_some_and(|sender| sender.send(Message::Flush(ack)).is_ok());
        if delivered && done.recv().is_ok() {
            return;
        }
        // The thread ended before it answered: it panicked.
        thread.sender = None;
        if let Some(handle) = thread.handle.take()
            && let Err(panic) = handle.join()
        {
            std::panic::resume_unwind(panic);
        }
    }

    /// Read the work and the counters. Realtime waits for the step or query in
    /// progress, never for the stamps still queued.
    pub fn read<R>(&self, read: impl FnOnce(&B, &BackendCounters) -> R) -> R {
        let state = lock(&self.state);
        let mut counters: BackendCounters = state.counters;
        if let Some(thread) = &self.thread {
            counters.max_pending_stamps = thread.max_pending;
        }
        read(&state.work, &counters)
    }
}

/// The mode, the work and what it has taken; never the wall-clock timings or
/// the Realtime queue depth, which change run to run (`Vio`'s `Debug` is a
/// determinism fingerprint).
impl<B: BackendWork + std::fmt::Debug> std::fmt::Debug for BackendRunner<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut out = f.debug_struct("BackendRunner");
        out.field("mode", &self.mode);
        match self.state.lock() {
            Ok(state) => out
                .field("work", &state.work)
                .field("stamps", &state.counters.stamps)
                .field("queries_run", &state.counters.queries_run)
                .field("queries_dropped", &state.counters.queries_dropped),
            Err(_) => out.field("work", &"<the back-end thread panicked>"),
        };
        out.finish()
    }
}

impl<B: BackendWork> Drop for BackendRunner<B> {
    /// Close the channel and join the thread. It steps the stamps it already
    /// has and leaves the queued queries.
    fn drop(&mut self) {
        let Some(thread) = self.thread.as_mut() else {
            return;
        };
        thread.sender = None;
        if let Some(handle) = thread.handle.take()
            && handle.join().is_err()
        {
            log::error!("the back-end thread panicked");
        }
    }
}

/// The Realtime loop: step every stamp waiting, then one query, until the
/// channel closes. It blocks only when it has nothing to do.
fn run_realtime<B: BackendWork>(
    state: &Mutex<State<B>>,
    receiver: &Receiver<Message>,
    pending: &AtomicU64,
    max_queries: usize,
) {
    loop {
        let message = match receiver.try_recv() {
            Ok(message) => message,
            Err(TryRecvError::Disconnected) => return,
            Err(TryRecvError::Empty) => {
                if lock(state).query_one() {
                    continue;
                }
                match receiver.recv() {
                    Ok(message) => message,
                    Err(_) => return,
                }
            }
        };
        match message {
            Message::Stamp(stamp) => {
                lock(state).step(stamp, max_queries);
                pending.fetch_sub(1, Ordering::Relaxed);
            }
            Message::Flush(ack) => {
                while lock(state).query_one() {}
                // The flusher may have stopped waiting; nothing to do then.
                let _ = ack.send(());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use std::collections::BTreeMap;
    use std::sync::atomic::AtomicBool;
    use std::time::Duration;

    use super::*;
    use crate::estimator::KeyframeStamp;

    /// Records what it was given; each step and query sleeps `delay`. A gate,
    /// when set, holds every step until opened.
    #[derive(Debug, Default)]
    struct Recorder {
        steps: Vec<i64>,
        queries: Vec<FrameId>,
        delay: Duration,
        gate: Option<Arc<AtomicBool>>,
    }

    impl BackendWork for Recorder {
        fn step(&mut self, stamp: MapStamp) -> Vec<MapEvent> {
            if let Some(gate) = &self.gate {
                while !gate.load(Ordering::Acquire) {
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
            std::thread::sleep(self.delay);
            let MapStamp::Keyframe(stamp) = stamp else {
                return Vec::new();
            };
            self.steps.push(stamp.t_ns);
            vec![MapEvent::KeyframeAdded(stamp.t_ns)]
        }

        fn query(&mut self, keyframe: FrameId) -> Vec<MapEvent> {
            std::thread::sleep(self.delay);
            self.queries.push(keyframe);
            Vec::new()
        }
    }

    fn stamp(t_ns: i64) -> MapStamp {
        MapStamp::Keyframe(Box::new(KeyframeStamp {
            t_ns,
            keyframe_poses: BTreeMap::new(),
            landmarks: Vec::new(),
            keypoints: Vec::new(),
        }))
    }

    #[test]
    fn offline_steps_and_queries_every_stamp_before_submit_returns() {
        let mut runner = BackendRunner::new(Recorder::default(), BackendMode::Offline, 1).unwrap();
        for t_ns in 0..10 {
            runner.submit(stamp(t_ns));
            runner.read(|work, counters| {
                assert_eq!(work.steps.len() as i64, t_ns + 1);
                assert_eq!(work.queries.last(), Some(&t_ns));
                assert_eq!(counters.stamps as i64, t_ns + 1);
            });
        }
        runner.read(|work, counters| {
            assert_eq!(work.queries, (0..10).collect::<Vec<_>>());
            assert_eq!(counters.queries_run, 10);
            assert_eq!(counters.queries_dropped, 0);
            assert_eq!(counters.max_pending_stamps, 0);
        });
    }

    #[test]
    fn realtime_with_time_to_spare_does_what_offline_does() {
        let mut runner = BackendRunner::new(Recorder::default(), BackendMode::Realtime, 4).unwrap();
        for t_ns in 0..20 {
            runner.submit(stamp(t_ns));
            std::thread::sleep(Duration::from_millis(1));
        }
        runner.flush();
        runner.read(|work, counters| {
            assert_eq!(work.steps, (0..20).collect::<Vec<_>>());
            assert_eq!(work.queries, (0..20).collect::<Vec<_>>());
            assert_eq!(counters.stamps, 20);
            assert_eq!(counters.queries_dropped, 0);
        });
    }

    /// A step held closed while tracking submits 50 stamps: no submit waits,
    /// every stamp is stepped in order, and the queries beyond the bound are
    /// dropped (oldest first) and counted.
    #[test]
    fn a_slow_realtime_back_end_drops_old_queries_never_stamps_and_never_blocks() {
        let gate = Arc::new(AtomicBool::new(false));
        let work = Recorder {
            delay: Duration::from_millis(2),
            gate: Some(Arc::clone(&gate)),
            ..Recorder::default()
        };
        let mut runner = BackendRunner::new(work, BackendMode::Realtime, 3).unwrap();
        let started = std::time::Instant::now();
        for t_ns in 0..50 {
            runner.submit(stamp(t_ns));
        }
        // 50 steps of 2 ms each would take 100 ms; the gate holds them all.
        assert!(started.elapsed() < Duration::from_millis(50));
        gate.store(true, Ordering::Release);
        runner.flush();
        runner.read(|work, counters| {
            assert_eq!(work.steps, (0..50).collect::<Vec<_>>());
            assert_eq!(counters.stamps, 50);
            assert_eq!(counters.max_pending_stamps, 50);
            assert!(counters.queries_dropped >= 47 - 3, "{counters:?}");
            assert_eq!(counters.queries_run + counters.queries_dropped, 50);
            assert_eq!(work.queries.len() as u64, counters.queries_run);
            // What survived is the newest, in order.
            assert_eq!(work.queries.last(), Some(&49));
            assert!(work.queries.windows(2).all(|pair| pair[0] < pair[1]));
        });
    }

    #[test]
    fn dropping_a_realtime_runner_joins_its_thread() {
        let work = Recorder {
            delay: Duration::from_millis(1),
            ..Recorder::default()
        };
        let mut runner = BackendRunner::new(work, BackendMode::Realtime, 2).unwrap();
        for t_ns in 0..5 {
            runner.submit(stamp(t_ns));
        }
        let state = Arc::clone(&runner.state);
        drop(runner);
        // The thread stepped what it already had, then ended and let go.
        assert_eq!(Arc::strong_count(&state), 1);
        assert_eq!(lock(&state).work.steps, (0..5).collect::<Vec<_>>());
    }

    #[derive(Debug)]
    struct Panics;

    impl BackendWork for Panics {
        fn step(&mut self, _: MapStamp) -> Vec<MapEvent> {
            panic!("step failed");
        }
        fn query(&mut self, _: FrameId) -> Vec<MapEvent> {
            Vec::new()
        }
    }

    #[test]
    fn a_panic_on_the_back_end_thread_reaches_flush_and_spares_submit() {
        let mut runner = BackendRunner::new(Panics, BackendMode::Realtime, 2).unwrap();
        runner.submit(stamp(0));
        std::thread::sleep(Duration::from_millis(20));
        runner.submit(stamp(1));
        let flushed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| runner.flush()));
        let message: String = flushed
            .unwrap_err()
            .downcast_ref::<&str>()
            .map(|text| (*text).to_owned())
            .unwrap_or_default();
        assert_eq!(message, "step failed");
        drop(runner);
    }
}
