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
//! Readers never wait for the work. [`BackendRunner::view`] hands out a copy
//! ([`BackendWork::View`]) with the counters at the moment it was taken, one
//! consistent read. In Realtime the back-end thread takes that copy itself,
//! once a reader has asked for one, each time it has stepped every stamp
//! waiting and after a query that changed the work; it posts it on a board
//! whose lock only guards an `Arc` swap. The work's own lock stays the thread's
//! for a whole step or query (in b2 a loop query and PGO). From outside,
//! [`BackendRunner::read`] (tests, diagnostics) waits for it and
//! [`BackendRunner::try_read`] (the map delta) gives up when it is held. Offline
//! steps inside `submit`, so a view taken from the work there never waits.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};
use std::thread::JoinHandle;

use super::{MapEvent, StepTiming};
use crate::config::BackendMode;
use crate::estimator::MapStamp;
use crate::types::FrameId;

/// What the back end does with the window's stamps.
pub trait BackendWork: Send + 'static {
    /// What readers get: a copy of the work that stays valid while it goes on.
    type View: Send + Sync + 'static;

    /// Take one stamp. Every stamp reaches this, in order.
    fn step(&mut self, stamp: MapStamp) -> Vec<MapEvent>;

    /// Look for a loop at `keyframe`, which a step added. Realtime may drop it.
    fn query(&mut self, keyframe: FrameId) -> Vec<MapEvent>;

    /// The query of `keyframe` was dropped (Realtime, behind): release what it
    /// held. Nothing by default.
    fn drop_query(&mut self, _keyframe: FrameId) {}

    /// Copy the work for readers.
    fn view(&self) -> Self::View;
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
    /// Wall time of the views the Realtime thread posted for readers.
    pub publish: StepTiming,
    /// Wall time of the loops that ran a pose-graph optimization (PR c2).
    pub pgo: StepTiming,
}

/// A copy of the work and the counters when it was taken: one moment.
#[derive(Debug)]
pub struct BackendView<V> {
    /// The copy.
    pub view: Arc<V>,
    /// What the back end had done when it was taken.
    pub counters: BackendCounters,
}

impl<V> Clone for BackendView<V> {
    fn clone(&self) -> Self {
        Self {
            view: Arc::clone(&self.view),
            counters: self.counters,
        }
    }
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
            if let Some(dropped) = self.queries.pop_front() {
                self.work.drop_query(dropped);
            }
            self.counters.queries_dropped += 1;
        }
    }

    /// Run the oldest queued query: `None` if none was queued, else whether
    /// it changed the work.
    fn query_one(&mut self) -> Option<bool> {
        let keyframe: FrameId = self.queries.pop_front()?;
        let started = std::time::Instant::now();
        let events: Vec<MapEvent> = self.work.query(keyframe);
        self.counters.query.record(crate::duration_ns(started));
        self.counters.queries_run += 1;
        for event in &events {
            if let MapEvent::LoopVerified {
                pgo_ns: Some(ns), ..
            } = event
            {
                self.counters.pgo.record(*ns);
            }
        }
        Some(!events.is_empty())
    }

    /// Take a view now, with the counters it goes with.
    fn view(&self) -> BackendView<B::View> {
        BackendView {
            view: Arc::new(self.work.view()),
            counters: self.counters,
        }
    }
}

/// What the Realtime thread posts for readers. Its lock guards copies and an
/// `Arc` swap, never a step.
struct Board<V> {
    posted: Mutex<Posted<V>>,
    /// A reader has asked for a view since the last one posted: the next step
    /// or query that changes the work posts one (and clears it). Views are
    /// O(map), and readers take one a second at most, so the thread copies
    /// the map only when one was asked for.
    wanted: AtomicBool,
}

struct Posted<V> {
    /// The newest view.
    latest: BackendView<V>,
    /// The counters after the newest step or query (newer than `latest`'s).
    counters: BackendCounters,
}

impl<V> Board<V> {
    fn posted(&self) -> MutexGuard<'_, Posted<V>> {
        match self.posted.lock() {
            Ok(guard) => guard,
            Err(_) => panic!("the back-end thread panicked"),
        }
    }

    /// Post the counters and, on `force` or when `take_view` and a reader
    /// asked, a new view.
    fn post<B: BackendWork<View = V>>(&self, state: &mut State<B>, take_view: bool, force: bool) {
        let mut latest: Option<BackendView<V>> = None;
        if force || (take_view && self.wanted.swap(false, Ordering::AcqRel)) {
            let started = std::time::Instant::now();
            let view: B::View = state.work.view();
            state.counters.publish.record(crate::duration_ns(started));
            latest = Some(BackendView {
                view: Arc::new(view),
                counters: state.counters,
            });
        }
        let mut posted = self.posted();
        posted.counters = state.counters;
        if let Some(latest) = latest {
            posted.latest = latest;
        }
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

/// Stamps waiting past which a Realtime run warns, once: steps average about
/// 0.6 ms (MIPB08), so 64 waiting means the thread has fallen well behind.
const WARN_PENDING_STAMPS: u64 = 64;

enum Message {
    Stamp(MapStamp),
    /// Answer once every earlier stamp is stepped and every queued query ran.
    Flush(mpsc::SyncSender<()>),
}

/// The Realtime thread and the tracking side of its channel.
struct Thread<B: BackendWork> {
    sender: Option<Sender<Message>>,
    handle: Option<JoinHandle<()>>,
    /// Stamps sent and not yet taken.
    pending: Arc<AtomicU64>,
    max_pending: u64,
    board: Arc<Board<B::View>>,
}

/// The back end in one of its two modes.
pub struct BackendRunner<B: BackendWork> {
    mode: BackendMode,
    state: Arc<Mutex<State<B>>>,
    /// `Some` in Realtime.
    thread: Option<Thread<B>>,
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
        let mut thread: Option<Thread<B>> = None;
        if mode == BackendMode::Realtime {
            let (sender, receiver) = mpsc::channel::<Message>();
            let pending = Arc::new(AtomicU64::new(0));
            let first: BackendView<B::View> = lock(&state).view();
            let board = Arc::new(Board {
                posted: Mutex::new(Posted {
                    counters: first.counters,
                    latest: first,
                }),
                wanted: AtomicBool::new(false),
            });
            let handle = std::thread::Builder::new()
                .name("slam-rs-backend".into())
                .spawn({
                    let state = Arc::clone(&state);
                    let pending = Arc::clone(&pending);
                    let board = Arc::clone(&board);
                    move || run_realtime(&state, &board, &receiver, &pending, max_queued_queries)
                })?;
            thread = Some(Thread {
                sender: Some(sender),
                handle: Some(handle),
                pending,
                max_pending: 0,
                board,
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
            while state.query_one().is_some() {}
            return;
        };
        let pending: u64 = thread.pending.fetch_add(1, Ordering::Relaxed) + 1;
        if pending > WARN_PENDING_STAMPS && thread.max_pending <= WARN_PENDING_STAMPS {
            log::warn!(
                "the back-end thread is {pending} map stamps behind tracking (a slow host?); stamps are never dropped, so the backlog grows until it catches up"
            );
        }
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

    /// A copy of the work and the counters it goes with; never waits for a
    /// step or query. Offline it is taken now (the steps run inside `submit`).
    /// Realtime it is the newest one the thread posted, and the call asks the
    /// thread to post a new one at the next change (so a reader that reads
    /// once a second sees a map at most one read old); after [`Self::flush`]
    /// it is current.
    pub fn view(&self) -> BackendView<B::View> {
        let Some(thread) = &self.thread else {
            return lock(&self.state).view();
        };
        thread.board.wanted.store(true, Ordering::Release);
        let mut latest: BackendView<B::View> = thread.board.posted().latest.clone();
        latest.counters.max_pending_stamps = thread.max_pending;
        latest
    }

    /// Read the work in place, for a tracking-side reader that needs less than
    /// a whole [`Self::view`] (the map's changes since its last read). Offline
    /// it runs at once. Unlike [`Self::read`], Realtime never waits for the
    /// back-end thread: `None` while a step or query holds the work, and the
    /// reader asks again later.
    ///
    /// # Panics
    ///
    /// Re-raises a panic of the back-end thread, as [`Self::view`] does.
    pub fn try_read<R>(&self, f: impl FnOnce(&B) -> R) -> Option<R> {
        if self.thread.is_none() {
            return Some(f(&lock(&self.state).work));
        }
        match self.state.try_lock() {
            Ok(state) => Some(f(&state.work)),
            Err(TryLockError::WouldBlock) => None,
            Err(TryLockError::Poisoned(_)) => panic!("the back-end thread panicked"),
        }
    }

    /// What the back end has done so far; never waits for a step or query.
    pub fn counters(&self) -> BackendCounters {
        let Some(thread) = &self.thread else {
            return lock(&self.state).counters;
        };
        let mut counters: BackendCounters = thread.board.posted().counters;
        counters.max_pending_stamps = thread.max_pending;
        counters
    }

    /// Change the work once every stamp submitted so far is taken (the global
    /// BA at the end of a sequence). Realtime then posts a fresh view, so
    /// readers see the change.
    ///
    /// # Panics
    ///
    /// Re-raises a panic of the back-end thread.
    pub fn update<R>(&mut self, change: impl FnOnce(&mut B) -> R) -> R {
        self.flush();
        let mut state = lock(&self.state);
        let out: R = change(&mut state.work);
        if let Some(thread) = &self.thread {
            thread.board.post(&mut state, true, true);
        }
        out
    }

    /// Read the work itself and the counters. Realtime waits for the step or
    /// query in progress (never for the stamps still queued), so tracking
    /// reads [`Self::view`] instead; this is for tests and diagnostics.
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
/// channel closes. It blocks only when it has nothing to do. It posts the
/// counters after each step and query, and a view once no stamp is waiting
/// (after a query only if it changed the work); a flush always posts one.
fn run_realtime<B: BackendWork>(
    state: &Mutex<State<B>>,
    board: &Board<B::View>,
    receiver: &Receiver<Message>,
    pending: &AtomicU64,
    max_queries: usize,
) {
    let query = || {
        let mut state = lock(state);
        let changed: Option<bool> = state.query_one();
        if let Some(changed) = changed {
            board.post(&mut state, changed, false);
        }
        changed.is_some()
    };
    loop {
        let message = match receiver.try_recv() {
            Ok(message) => message,
            Err(TryRecvError::Disconnected) => return,
            Err(TryRecvError::Empty) => {
                if query() {
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
                let mut state = lock(state);
                state.step(stamp, max_queries);
                let waiting: u64 = pending.fetch_sub(1, Ordering::Relaxed) - 1;
                board.post(&mut state, waiting == 0, false);
            }
            Message::Flush(ack) => {
                while query() {}
                board.post(&mut lock(state), true, true);
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
        type View = Vec<i64>;

        fn view(&self) -> Vec<i64> {
            self.steps.clone()
        }

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
            images: None,
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

    /// A step held closed on the back-end thread (holding the work's lock):
    /// a view and the counters come back at once, at the last posted moment,
    /// and once the step ends and the runner is flushed the view is current.
    #[test]
    fn a_held_step_never_makes_a_view_or_the_counters_wait() {
        let gate = Arc::new(AtomicBool::new(true));
        let work = Recorder {
            gate: Some(Arc::clone(&gate)),
            ..Recorder::default()
        };
        let mut runner = BackendRunner::new(work, BackendMode::Realtime, 4).unwrap();
        let first = runner.view();
        assert!(first.view.is_empty());
        runner.submit(stamp(0));
        runner.flush();
        gate.store(false, Ordering::Release);
        runner.submit(stamp(1));
        // Wait until the thread is inside the held step: it is idle after the
        // flush, so the only one to take the work's lock now is step 1.
        let started = std::time::Instant::now();
        while runner.state.try_lock().is_ok() {
            assert!(
                started.elapsed() < Duration::from_secs(5),
                "the step never started"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        let reading = std::time::Instant::now();
        let held = runner.view();
        let counters: BackendCounters = runner.counters();
        assert!(reading.elapsed() < Duration::from_millis(50));
        assert!(runner.state.try_lock().is_err(), "the step is still held");
        assert_eq!(*held.view, vec![0]);
        assert_eq!(held.counters.stamps, 1);
        assert_eq!(counters.stamps, 1);
        gate.store(true, Ordering::Release);
        runner.flush();
        let current = runner.view();
        assert_eq!(*current.view, vec![0, 1]);
        assert_eq!(current.counters.stamps, 2);
        assert_eq!(current.counters.queries_run, 2);
        assert!(
            current.counters.publish.steps >= 2,
            "{:?}",
            current.counters
        );
    }

    /// Realtime copies the map only when a reader asked: without a `view` call
    /// only the flush posts one.
    #[test]
    fn a_realtime_view_is_posted_only_when_a_reader_asks() {
        let mut runner = BackendRunner::new(Recorder::default(), BackendMode::Realtime, 4).unwrap();
        for t_ns in 0..5 {
            runner.submit(stamp(t_ns));
        }
        runner.flush();
        assert_eq!(runner.counters().publish.steps, 1, "the flush's post alone");
        assert_eq!(*runner.view().view, vec![0, 1, 2, 3, 4]);
        for t_ns in 5..8 {
            runner.submit(stamp(t_ns));
        }
        runner.flush();
        // At most one post for the one request, then the flush's.
        assert!((2..=3).contains(&runner.counters().publish.steps));
    }

    /// Offline takes a view on demand, never posts one, and the view's
    /// counters are the ones of that moment.
    #[test]
    fn an_offline_view_is_taken_on_demand() {
        let mut runner = BackendRunner::new(Recorder::default(), BackendMode::Offline, 1).unwrap();
        for t_ns in 0..3 {
            runner.submit(stamp(t_ns));
        }
        let view = runner.view();
        assert_eq!(*view.view, vec![0, 1, 2]);
        assert_eq!(view.counters, runner.counters());
        assert_eq!(view.counters.stamps, 3);
        assert_eq!(view.counters.publish.steps, 0);
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
        type View = ();

        fn view(&self) {}

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
