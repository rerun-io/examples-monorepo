//! Frame dispatch, uploads, and fallible device downloads.

use kornia_staging_gpu::runtime::GpuError;

/// A frame can use CubeCL's existing server thread for all host-side GPU work.
/// Nested client operations then execute directly, without channel round trips.
pub struct FrameExecutor<R: cubecl::prelude::Runtime> {
    pub(super) client: cubecl::prelude::ComputeClient<R>,
}

impl<R: cubecl::prelude::Runtime> crate::frontend::stages::FrameExecutor for FrameExecutor<R> {
    fn run(
        self,
        body: impl FnOnce() -> Result<(), crate::frontend::flow::FrontendError> + Send,
    ) -> Result<(), crate::frontend::flow::FrontendError> {
        kornia_staging_gpu::transfer::execute_exclusive(&self.client, "frontend dispatch", body)
    }
}

/// All deferred GPU work for a frame, in upload/launch order.
pub(super) enum Launch {
    Pyramid(kornia_staging_gpu::pyramid::PyramidLaunch),
    Corners(kornia_staging_gpu::features::CornerLaunch),
    Klt(kornia_staging_gpu::optical_flow::FusedLaunch),
    Stereo(super::frontend::onewait::StereoLaunch),
}

impl Launch {
    fn run<R: cubecl::prelude::Runtime>(self, client: &cubecl::prelude::ComputeClient<R>) {
        match self {
            // SAFETY: The frame prepares and runs these launches on its exclusive
            // client stream, before any later frame can reuse the upload.
            Self::Pyramid(launch) => unsafe { launch.run(client) },
            // SAFETY: The frame queues this work after its pyramid on the same client stream.
            Self::Corners(launch) => unsafe { launch.run(client) },
            // SAFETY: The frame runs validated KLT work on its preparation stream.
            Self::Klt(launch) => unsafe { launch.run(client) },
            Self::Stereo(launch) => launch.run(client),
        }
    }
}

#[derive(Default)]
struct LaunchState {
    active: bool,
    commands: Vec<Launch>,
}

/// Shared by the frame's pyramid, corner and KLT stages. Outside a frame,
/// ordinary dispatches run immediately; per-camera KLT waits until collection.
#[derive(Clone, Default)]
pub struct LaunchList(std::sync::Arc<std::sync::Mutex<LaunchState>>);

impl LaunchList {
    pub(super) fn begin(&self) -> Result<FrameBatch, crate::frontend::flow::FrontendError> {
        let mut state = self.0.lock().unwrap_or_else(|error| error.into_inner());
        if state.active {
            return Err(crate::frontend::flow::FrontendError::NestedGpuFrame);
        }
        state.active = true;
        Ok(FrameBatch(self.clone()))
    }

    pub(super) fn dispatch<R: cubecl::prelude::Runtime>(
        &self,
        client: &cubecl::prelude::ComputeClient<R>,
        launch: Launch,
    ) -> Result<(), GpuError> {
        let mut state = self.0.lock().unwrap_or_else(|error| error.into_inner());
        if state.active {
            state.commands.push(launch);
        } else {
            drop(state);
            kornia_staging_gpu::runtime::guarded(
                GpuError::DeviceLost {
                    what: "queued launch",
                },
                || {
                    launch.run(client);
                    Ok(())
                },
            )?;
        }
        Ok(())
    }

    pub(super) fn clear(&self) {
        self.0
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .commands
            .clear();
    }

    pub(super) fn defer(&self, launch: Launch) {
        self.0
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .commands
            .push(launch);
    }

    pub(super) fn flush<R: cubecl::prelude::Runtime>(
        &self,
        client: &cubecl::prelude::ComputeClient<R>,
    ) -> Result<(), GpuError> {
        let commands = std::mem::take(
            &mut self
                .0
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .commands,
        );
        kornia_staging_gpu::runtime::guarded(
            GpuError::DeviceLost {
                what: "queued launch",
            },
            || {
                #[cfg(test)]
                super::runtime::fire_if_armed("queued launch");
                for command in commands {
                    command.run(client);
                }
                Ok(())
            },
        )
    }
}

/// Dropping an unfinished frame cancels its unsubmitted work.
pub(super) struct FrameBatch(LaunchList);

impl FrameBatch {
    pub(super) fn finish<R: cubecl::prelude::Runtime>(
        self,
        client: &cubecl::prelude::ComputeClient<R>,
    ) -> Result<(), crate::frontend::flow::FrontendError> {
        self.0.flush(client).map_err(Into::into)
    }
}

impl Drop for FrameBatch {
    fn drop(&mut self) {
        let mut state = self.0.0.lock().unwrap_or_else(|error| error.into_inner());
        state.active = false;
        state.commands.clear();
    }
}

pub(super) use kornia_staging_gpu::transfer::read_failed;

/// Submit the current readback, then queue independent work before its wait.
#[cfg(feature = "gpu-core")]
pub(super) fn read_with_lookahead<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    launches: &LaunchList,
    handles: Vec<cubecl::server::Handle>,
    what: &'static str,
    after_copy: impl FnOnce() -> Result<(), GpuError>,
) -> Result<Vec<cubecl::bytes::Bytes>, GpuError> {
    launches.flush(client)?;
    #[cfg(test)]
    if super::runtime::armed(super::runtime::BLOCKING_READ) {
        return Err(GpuError::DeviceReadFailed { what });
    }
    kornia_staging_gpu::transfer::read_with_lookahead(client, handles, what, after_copy)
}
