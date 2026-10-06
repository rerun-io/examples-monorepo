//! Frontend scheduling and lazy input adapters for staged GPU feature extraction.
use crate::frontend::{
    detect::FrameCornerScan,
    flow::FrontendError,
    input::{FrameImage, FrameImages},
};
use cubecl::prelude::*;
use kornia_image::Image;
use kornia_staging_gpu::features::{GpuCornerScan as DeviceScan, ScanError, ScanInput};
use kornia_staging_gpu::runtime::GpuError;
use kornia_staging_imgproc::features::{
    BandRequest, CellSelect, CornerScan, FastCorner, Occupancy, SelectionStatus,
};

/// Reusable GPU scanner scheduled by the frontend's frame queue.
pub struct GpuCornerScan<R: Runtime> {
    pub(super) inner: DeviceScan<R>,
    client: ComputeClient<R>,
    launches: super::submission::LaunchList,
}
impl<R: Runtime> GpuCornerScan<R> {
    pub fn new(
        client: ComputeClient<R>,
        launches: super::submission::LaunchList,
    ) -> Result<Self, GpuError> {
        Ok(Self {
            inner: DeviceScan::new(client.clone())?,
            client,
            launches,
        })
    }
    fn with_input<T>(
        &mut self,
        camera: usize,
        image: FrameImage<'_>,
        operation: impl FnOnce(&mut DeviceScan<R>, ScanInput<'_>) -> Result<T, ScanError>,
    ) -> Result<T, FrontendError> {
        if self.inner.has_level0(camera, image.size()) {
            operation(&mut self.inner, ScanInput::Uploaded(image.size())).map_err(Into::into)
        } else {
            image
                .with_dense(|image| operation(&mut self.inner, ScanInput::Dense(image)))?
                .map_err(Into::into)
        }
    }
}
impl<R: Runtime> FrameCornerScan for GpuCornerScan<R> {
    fn scan_frame(&mut self, camera: usize, image: FrameImage<'_>) -> Result<(), FrontendError> {
        self.inner.abort_scan();
        self.launches.flush(&self.client)?;
        self.with_input(camera, image, |scan, input| scan.scan_input(camera, input))
    }
    fn select_frame(
        &mut self,
        camera: usize,
        image: FrameImage<'_>,
        select: &CellSelect,
        _eligibility: Option<(&Occupancy<'_>, &[bool])>,
        out: &mut Vec<Option<FastCorner>>,
    ) -> Result<SelectionStatus, FrontendError> {
        let outcome = (|| {
            self.launches.flush(&self.client)?;
            self.with_input(camera, image, |scan, input| {
                scan.select_input(camera, input, select, out)
            })
        })();
        if outcome.is_err() {
            self.inner.abort_scan();
            self.inner.abort_selection();
            out.clear();
        }
        outcome
    }
    fn submit_cells(
        &mut self,
        images: FrameImages<'_>,
        selects: &[Option<CellSelect>],
    ) -> Result<(), FrontendError> {
        let client = &self.client;
        let launches = &self.launches;
        self.inner.submit_cells(
            images.iter().map(|image| image.size()),
            selects,
            |scan, camera, select| {
                launches.flush(client)?;
                let image = images.get(camera);
                if scan.has_level0(camera, image.size()) {
                    scan.submit_input(camera, ScanInput::Uploaded(image.size()), select)?;
                } else {
                    image.with_dense(|image| {
                        scan.submit_input(camera, ScanInput::Dense(image), select)
                    })??;
                }
                Ok(())
            },
            |work| {
                launches.dispatch(client, super::submission::Launch::Corners(work))?;
                Ok(())
            },
        )
    }
    fn take_cells(&mut self) -> Result<(), FrontendError> {
        let outcome = (|| {
            self.launches.flush(&self.client)?;
            self.inner.take_cells().map_err(Into::into)
        })();
        if outcome.is_err() {
            self.inner.abort_selection();
        }
        outcome
    }
}
impl<R: Runtime> CornerScan for GpuCornerScan<R> {
    type Error = FrontendError;
    fn scan(&mut self, camera: usize, image: &Image<u16, 1>) -> Result<(), FrontendError> {
        self.scan_frame(camera, FrameImage::dense(image))
    }
    fn select_cells(
        &mut self,
        camera: usize,
        image: &Image<u16, 1>,
        select: &CellSelect,
        eligibility: Option<(&Occupancy<'_>, &[bool])>,
        out: &mut Vec<Option<FastCorner>>,
    ) -> Result<SelectionStatus, FrontendError> {
        self.select_frame(camera, FrameImage::dense(image), select, eligibility, out)
    }
    fn band(&mut self, request: BandRequest) -> Result<&[FastCorner], FrontendError> {
        self.inner.band(request).map_err(Into::into)
    }
}
impl<R: Runtime> std::fmt::Debug for GpuCornerScan<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt(f)
    }
}
