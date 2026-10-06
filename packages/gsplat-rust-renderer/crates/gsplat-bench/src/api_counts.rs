//! Count wgpu API events only during separate attribution frames.
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Counts {
    pub queue_submits: u64,
    pub blocking_device_polls: u64,
    /// CPU time from a mapped count read to the next submit, inside Brush's GPU window.
    pub readback_to_submit_ms: f64,
}

#[derive(Default)]
struct Counter {
    submits: AtomicU64,
    waits: AtomicU64,
    last_readback_ns: AtomicU64,
    readback_to_submit_ns: AtomicU64,
    epoch: OnceLock<Instant>,
}
impl log::Log for Counter {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.target().starts_with("wgpu_core::device::")
    }
    fn log(&self, record: &log::Record<'_>) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let message = record.args().to_string();
        if record.target() == "wgpu_core::device::queue" && message == "Queue::submit" {
            self.submits.fetch_add(1, Ordering::Relaxed);
            let readback = self.last_readback_ns.swap(0, Ordering::Relaxed);
            if readback != 0 {
                let now = self.epoch.get_or_init(Instant::now).elapsed().as_nanos() as u64;
                self.readback_to_submit_ns
                    .fetch_add(now - readback, Ordering::Relaxed);
            }
        } else if record.target() == "wgpu_core::device::global"
            && message.starts_with("Device::poll Wait")
        {
            self.waits.fetch_add(1, Ordering::Relaxed);
        } else if record.target() == "wgpu_core::device::global"
            && message.starts_with("Buffer::get_mapped_range")
        {
            let now = self.epoch.get_or_init(Instant::now).elapsed().as_nanos() as u64;
            self.last_readback_ns.store(now, Ordering::Relaxed);
        }
    }
    fn flush(&self) {}
}

static COUNTER: Counter = Counter {
    submits: AtomicU64::new(0),
    waits: AtomicU64::new(0),
    last_readback_ns: AtomicU64::new(0),
    readback_to_submit_ns: AtomicU64::new(0),
    epoch: OnceLock::new(),
};

pub fn install() -> anyhow::Result<()> {
    log::set_logger(&COUNTER).map_err(|error| anyhow::anyhow!("API counter logger: {error}"))?;
    log::set_max_level(log::LevelFilter::Off);
    Ok(())
}

/// Restore disabled logging even if a diagnostic render fails.
pub struct Observation;
impl Observation {
    pub fn begin() -> Self {
        COUNTER.submits.store(0, Ordering::Relaxed);
        COUNTER.waits.store(0, Ordering::Relaxed);
        COUNTER.last_readback_ns.store(0, Ordering::Relaxed);
        COUNTER.readback_to_submit_ns.store(0, Ordering::Relaxed);
        COUNTER.epoch.get_or_init(Instant::now);
        log::set_max_level(log::LevelFilter::Trace);
        Self
    }
    pub fn finish(self) -> Counts {
        Counts {
            queue_submits: COUNTER.submits.load(Ordering::Relaxed),
            blocking_device_polls: COUNTER.waits.load(Ordering::Relaxed),
            readback_to_submit_ms: COUNTER.readback_to_submit_ns.load(Ordering::Relaxed) as f64
                / 1e6,
        }
    }
}
impl Drop for Observation {
    fn drop(&mut self) {
        log::set_max_level(log::LevelFilter::Off);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use log::Log as _;
    #[test]
    fn counts_submits_once_and_excludes_nonblocking_polls() {
        let counter = Counter::default();
        for (target, message) in [
            ("wgpu_core::device::queue", "Queue::submit"),
            (
                "wgpu_core::device::queue",
                "Queue::submit returned submit index 1",
            ),
            (
                "wgpu_core::device::global",
                "Device::poll Wait { submission_index: None, timeout: None }",
            ),
            ("wgpu_core::device::global", "Device::poll Poll"),
            ("other", "Queue::submit"),
        ] {
            counter.log(
                &log::Record::builder()
                    .target(target)
                    .args(format_args!("{message}"))
                    .build(),
            );
        }
        assert_eq!(counter.submits.load(Ordering::Relaxed), 1);
        assert_eq!(counter.waits.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    #[ignore = "integration: GPU"]
    async fn observes_real_submits_waits_and_mapped_readback() {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        install().unwrap();
        let observation = Observation::begin();
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&buffer, 0, None);
        let (tx, rx) = std::sync::mpsc::channel();
        encoder.map_buffer_on_submit(&buffer, wgpu::MapMode::Read, .., move |result| {
            tx.send(result).unwrap();
        });
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        assert_eq!(&*buffer.get_mapped_range(..).unwrap(), &[0; 8]);
        buffer.unmap();
        queue.submit([]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let counts = observation.finish();
        assert_eq!(counts.queue_submits, 2, "{counts:?}");
        assert_eq!(counts.blocking_device_polls, 2, "{counts:?}");
        assert!(counts.readback_to_submit_ms > 0.0, "{counts:?}");
        assert_eq!(log::max_level(), log::LevelFilter::Off);
    }
}
