//! Frame dispatch, uploads, and fallible device downloads.

use super::GpuError;

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
        result.map_err(crate::frontend::tracker::TrackerError::from)?
    }
}

/// All deferred GPU work for a frame, in upload/launch order.
#[expect(
    clippy::large_enum_variant,
    reason = "the Stereo launch lands in the one-wait commit; keep the final dispatch layout"
)]
pub(super) enum Launch {
    Pyramid(super::pyramid::PyramidLaunch),
    Corners(super::detect::batch::CornerLaunch),
    Klt(super::track::FusedLaunch),
}

impl Launch {
    fn run<R: cubecl::prelude::Runtime>(self, client: &cubecl::prelude::ComputeClient<R>) {
        match self {
            Self::Pyramid(launch) => launch.run(client),
            Self::Corners(launch) => launch.run(client),
            Self::Klt(launch) => launch.run(client),
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
    ) -> Result<(), crate::frontend::tracker::TrackerError> {
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

/// Storage binding alignment and maximum size, in bytes.
pub(super) fn binding_limits<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
) -> (usize, usize) {
    let memory = &client.properties().memory;
    (
        (memory.alignment as usize).max(256),
        memory.max_page_size as usize,
    )
}

/// Upload a host slice through one aligned allocation.
pub(super) fn upload<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    bytes: &[u8],
) -> cubecl::server::Handle {
    // Copy directly into aligned owned storage; create_from_slice copies
    // through two Vecs before allocating this same aligned storage.
    let mut data = cubecl::bytes::Bytes::from_elems(Vec::<u8>::new());
    data.extend_from_byte_slice(bytes);
    client.create(data)
}

/// A failed device read as a typed error, with the runtime's own reason logged.
///
/// [`GpuError`] is `Copy`, so it cannot carry the `ServerError`'s reason and
/// backtrace; the warning is where they are kept, and the returned variant is
/// what the stage errors carry to the caller.
pub(super) fn read_failed(what: &'static str, error: &cubecl::server::ServerError) -> GpuError {
    log::warn!("reading {what} from the device failed: {error}");
    GpuError::DeviceReadFailed { what }
}

/// Download every handle together and map device errors at one boundary.
#[cfg(feature = "gpu-core")]
pub(super) fn read_blocking<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    launches: &LaunchList,
    handles: Vec<cubecl::server::Handle>,
    what: &'static str,
) -> Result<Vec<cubecl::bytes::Bytes>, GpuError> {
    launches.flush(client);
    #[cfg(test)]
    if super::runtime::armed(super::runtime::BLOCKING_READ) {
        return Err(GpuError::DeviceReadFailed { what });
    }
    // `read_async` sends the copy and submits it before returning the future.
    let pending = client.read_async(handles);
    cubecl::future::reader::read_sync(pending).map_err(|error| read_failed(what, &error))
}

/// One frame on the device, and how many pixels it holds.
///
/// The upload owns one aligned copy of the slice before submitting it to
/// CubeCL. It is exactly as long as the frame and nothing more: an unstrided
/// frame goes straight out of the caller's buffer with no staging copy at all,
/// and only a strided one — dav1d's shape — is repacked row by row into
/// `scratch`, which the caller owns so the per-frame path never allocates.
///
/// Both the pyramid builder's level-0 upload and the corner scanner's own frame
/// upload are this, which is why it is here and not in either.
#[cfg(feature = "gpu-core")]
pub(super) fn upload_frame<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    image: &crate::image::ImageU16,
    scratch: &mut Vec<u16>,
) -> (cubecl::server::Handle, usize) {
    use cubecl::prelude::CubeElement;

    let (width, height): (usize, usize) = (image.width(), image.height());
    let pixels: usize = width * height;
    if image.stride() == width {
        return (
            upload(client, u16::as_bytes(&image.data()[..pixels])),
            pixels,
        );
    }
    scratch.clear();
    scratch.reserve(pixels);
    for y in 0..height {
        scratch.extend_from_slice(image.row(y));
    }
    (upload(client, u16::as_bytes(scratch)), pixels)
}
