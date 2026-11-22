//! GPU timestamps of the splat frame and the motion pass, logged every few seconds as a viewer line.
//!
//! Off unless [`ENABLED`] is set before the device is created (the web viewer sets it for `?profile=1`); it then
//! needs the adapter's `TIMESTAMP_QUERY`. One readback is in flight per view: frames encoded while it is mapped
//! are not timed. Browsers may quantise timestamps (Chromium: 100 us unless its developer flag is set).
use re_viewer::external::{parking_lot::Mutex, re_log, re_renderer::external::bytemuck};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Set before device creation to request `TIMESTAMP_QUERY` and time every view.
pub static ENABLED: AtomicBool = AtomicBool::new(false);

/// Ten frame slots (`gsplat_core::QUERY_COUNT`) plus the motion pass begin/end.
const SLOTS: u32 = gsplat_core::QUERY_COUNT + 2;
const LOG_EVERY_S: f64 = 3.0;

pub(crate) fn requested() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

#[derive(Default)]
struct Samples {
    frame: Vec<f64>,
    motion: Vec<f64>,
    since: Option<web_time::Instant>,
}

static SAMPLES: Mutex<Samples> = Mutex::new(Samples {
    frame: Vec::new(),
    motion: Vec::new(),
    since: None,
});

fn median_p90(values: &mut [f64]) -> (f64, f64) {
    values.sort_by(f64::total_cmp);
    let at = |q: f64| values[((values.len() - 1) as f64 * q).round() as usize];
    (at(0.5), at(0.9))
}

fn record(ticks: &[u64], period_ns: f32, motion: bool) {
    let ms = |a: u64, b: u64| (b - a) as f64 * f64::from(period_ns) / 1e6;
    let (t0, t9) = (ticks[0], ticks[9]);
    let mut samples = SAMPLES.lock();
    if t9 > t0 {
        samples.frame.push(ms(t0, t9));
    }
    // The motion pass runs before the frame's first pass, only when the clip time changed.
    let (m0, m1) = (ticks[10], ticks[11]);
    if motion && m0 > 0 && m1 > m0 && m1 <= t0 {
        samples.motion.push(ms(m0, m1));
    }
    let now = web_time::Instant::now();
    let since = *samples.since.get_or_insert(now);
    if now.duration_since(since).as_secs_f64() < LOG_EVERY_S || samples.frame.is_empty() {
        return;
    }
    let n = samples.frame.len();
    let (frame_median, frame_p90) = median_p90(&mut samples.frame);
    let motion = if samples.motion.is_empty() {
        "no time change".to_owned()
    } else {
        let m = samples.motion.len();
        let (median, p90) = median_p90(&mut samples.motion);
        format!("median {median:.3} ms, p90 {p90:.3} ms ({m} passes)")
    };
    re_log::info!(
        "gsplat profile: GPU splat frame median {frame_median:.2} ms, p90 {frame_p90:.2} ms ({n} frames); motion pass {motion}"
    );
    *samples = Samples {
        since: Some(now),
        ..Samples::default()
    };
}

/// Per-view query set and readback.
pub(crate) struct ViewProfile {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    busy: Arc<AtomicBool>,
    last_time: Option<u32>,
}

impl ViewProfile {
    pub(crate) fn new(device: &wgpu::Device) -> Option<Self> {
        if !requested() || !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let bytes = u64::from(SLOTS) * 8;
        Some(Self {
            queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("gsplat profile"),
                ty: wgpu::QueryType::Timestamp,
                count: SLOTS,
            }),
            resolve: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gsplat profile resolve"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gsplat profile readback"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            busy: Arc::new(AtomicBool::new(false)),
            last_time: None,
        })
    }

    /// The query set for this frame, unless the previous readback is still mapped.
    pub(crate) fn begin(&self) -> Option<wgpu::QuerySet> {
        (!self.busy.load(Ordering::Acquire)).then(|| self.queries.clone())
    }

    /// Resolve the frame's timestamps after it was encoded with [`Self::begin`]'s set and map them on submit.
    pub(crate) fn end(&mut self, encoder: &mut wgpu::CommandEncoder, period_ns: f32, time: f32) {
        let motion = self.last_time.is_some_and(|t| t != time.to_bits());
        self.last_time = Some(time.to_bits());
        encoder.resolve_query_set(&self.queries, 0..SLOTS, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.readback, 0, u64::from(SLOTS) * 8);
        self.busy.store(true, Ordering::Release);
        let (buffer, busy) = (self.readback.clone(), self.busy.clone());
        encoder.map_buffer_on_submit(&self.readback, wgpu::MapMode::Read, .., move |result| {
            if result.is_ok() {
                if let Ok(view) = buffer.get_mapped_range(..) {
                    let ticks: Vec<u64> = bytemuck::cast_slice(&view).to_vec();
                    drop(view);
                    record(&ticks, period_ns, motion);
                }
                buffer.unmap();
            }
            busy.store(false, Ordering::Release);
        });
    }
}
