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
        let result = super::guarded(
            GpuError::DeviceLost {
                what: "frontend dispatch",
            },
            || {
                self.client
                    .exclusive(body)
                    .map_err(|error| read_failed("frontend dispatch", &error))
            },
        );
        result.map_err(crate::frontend::flow::FrontendError::from)?
    }
}

/// All deferred GPU work for a frame, in upload/launch order.
pub(super) enum Launch {
    Pyramid(kornia_staging_gpu::pyramid::PyramidLaunch),
    Corners(kornia_staging_gpu::features::CornerLaunch),
    Klt(super::track::FusedLaunch),
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
            Self::Klt(launch) => launch.run(client),
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
    pub(super) fn begin(&self) -> Result<FrameBatch, GpuError> {
        let mut state = self.0.lock().unwrap_or_else(|error| error.into_inner());
        if state.active {
            return Err(GpuError::NestedFrame);
        }
        state.active = true;
        Ok(FrameBatch(self.clone()))
    }

    pub(super) fn dispatch<R: cubecl::prelude::Runtime>(
        &self,
        client: &cubecl::prelude::ComputeClient<R>,
        launch: Launch,
    ) {
        let mut state = self.0.lock().unwrap_or_else(|error| error.into_inner());
        if state.active {
            state.commands.push(launch);
        } else {
            drop(state);
            launch.run(client);
        }
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
    ) {
        #[cfg(test)]
        super::runtime::fire_if_armed("queued launch");
        let commands = std::mem::take(
            &mut self
                .0
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .commands,
        );
        for command in commands {
            command.run(client);
        }
    }
}

/// Dropping an unfinished frame cancels its unsubmitted work.
pub(super) struct FrameBatch(LaunchList);

impl FrameBatch {
    pub(super) fn finish<R: cubecl::prelude::Runtime>(
        self,
        client: &cubecl::prelude::ComputeClient<R>,
    ) -> Result<(), crate::frontend::flow::FrontendError> {
        super::guarded(
            GpuError::DeviceLost {
                what: "frame dispatch",
            },
            || {
                self.0.flush(client);
                Ok(())
            },
        )
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

/// Download every handle together and map device errors at one boundary.
#[cfg(feature = "gpu-core")]
pub(super) fn read_blocking<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    launches: &LaunchList,
    handles: Vec<cubecl::server::Handle>,
    what: &'static str,
) -> Result<Vec<cubecl::bytes::Bytes>, GpuError> {
    read_with_lookahead(client, launches, handles, what, || Ok(()))
}

/// Submit the current readback, then queue independent work before its wait.
#[cfg(feature = "gpu-core")]
pub(super) fn read_with_lookahead<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    launches: &LaunchList,
    handles: Vec<cubecl::server::Handle>,
    what: &'static str,
    after_copy: impl FnOnce() -> Result<(), GpuError>,
) -> Result<Vec<cubecl::bytes::Bytes>, GpuError> {
    launches.flush(client);
    #[cfg(test)]
    if super::runtime::armed(super::runtime::BLOCKING_READ) {
        return Err(GpuError::DeviceReadFailed { what });
    }
    // `read_async` sends the copy and submits it before returning the future.
    // Work queued now goes in a later submission, independent of this copy.
    let pending = client.read_async(handles);
    after_copy()?;
    cubecl::future::reader::read_sync(pending).map_err(|error| read_failed(what, &error))
}
