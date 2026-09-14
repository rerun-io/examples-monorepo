use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use gstreamer::{self as gst, prelude::*};
use gstreamer_app::{AppSink, AppSrc};

/// Owned bytes and timing from one GStreamer buffer, before any persistence.
pub struct CapturedBuffer {
    /// Pipeline presentation time. This is not yet a device-clock timestamp.
    pub pts_ns: u64,
    pub keyframe: bool,
    pub bytes: Vec<u8>,
}

/// Owns a pipeline and returns its samples without a second unbounded queue.
pub struct SamplePipeline {
    pipeline: gst::Pipeline,
    sink: AppSink,
}

impl SamplePipeline {
    /// Clone the named input handle. The producer must respect its queue limit.
    pub fn source(&self, name: &str) -> Result<AppSrc> {
        self.pipeline
            .by_name(name)
            .context("named appsrc missing")?
            .downcast::<AppSrc>()
            .map_err(|_| anyhow::anyhow!("named input is not appsrc"))
    }
    /// Launch an application-supplied pipeline containing a named appsink.
    pub fn start(description: &str, sink_name: &str) -> Result<Self> {
        gst::init()?;
        let pipeline = gst::parse::launch(description)?
            .downcast::<gst::Pipeline>()
            .map_err(|_| anyhow::anyhow!("capture description must form a pipeline"))?;
        let sink = pipeline
            .by_name(sink_name)
            .context("named appsink missing")?
            .downcast::<AppSink>()
            .map_err(|_| anyhow::anyhow!("named element is not an appsink"))?;
        ensure!(
            sink.max_buffers() > 0 && !sink.is_drop(),
            "capture requires a bounded appsink without silent dropping"
        );
        let owner = Self { pipeline, sink };
        owner.pipeline.set_state(gst::State::Playing)?;
        Ok(owner)
    }

    /// None means drained EOS. A timeout or pipeline fault is an error.
    pub fn next(&mut self, timeout: Duration) -> Result<Option<CapturedBuffer>> {
        let timeout = gst::ClockTime::from_nseconds(u64::try_from(timeout.as_nanos())?);
        let Some(sample) = self.sink.try_pull_sample(timeout) else {
            if let Some(bus) = self.pipeline.bus() {
                for message in bus.iter() {
                    if let gst::MessageView::Error(error) = message.view() {
                        bail!(
                            "capture pipeline error: {} ({:?})",
                            error.error(),
                            error.debug()
                        );
                    }
                }
            }
            if self.sink.is_eos() {
                return Ok(None);
            }
            bail!("capture sample timeout");
        };
        let buffer = sample.buffer().context("capture sample has no buffer")?;
        ensure!(
            !buffer.flags().contains(gst::BufferFlags::CORRUPTED),
            "capture buffer is marked corrupt"
        );
        let pts_ns = buffer
            .pts()
            .context("capture sample has no timestamp")?
            .nseconds();
        let mapped = buffer
            .map_readable()
            .context("capture buffer cannot be mapped")?;
        Ok(Some(CapturedBuffer {
            pts_ns,
            keyframe: !buffer.flags().contains(gst::BufferFlags::DELTA_UNIT),
            bytes: mapped.as_slice().to_vec(),
        }))
    }
}

impl Drop for SamplePipeline {
    fn drop(&mut self) {
        if let Err(error) = self.pipeline.set_state(gst::State::Null) {
            eprintln!("capture pipeline did not stop cleanly: {error}");
        }
    }
}
