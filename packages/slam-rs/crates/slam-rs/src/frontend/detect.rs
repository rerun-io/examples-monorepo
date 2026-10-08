//! Application frame preparation around the shared cell detector policy.
use super::flow::FrontendError;
use super::input::{FrameImage, FrameImages};
use kornia_staging_imgproc::features::{
    CellSelect, CornerScan, FastCorner, Occupancy, SelectionStatus,
};

/// Frame scheduling and image ownership remain in the application.
pub trait FrameCornerScan: CornerScan<Error: Into<FrontendError>> {
    fn fork_frame(&self) -> Option<Box<Self>> where Self: Sized { None }
    fn scan_frame(&mut self, camera: usize, image: FrameImage<'_>) -> Result<(), FrontendError> {
        image
            .with_dense(|image| self.scan(camera, image))
            .map_err(FrontendError::Image)?.map_err(Into::into)
    }
    fn select_frame(
        &mut self,
        camera: usize,
        image: FrameImage<'_>,
        select: &CellSelect,
        eligibility: Option<(&Occupancy<'_>, &[bool])>,
        out: &mut Vec<Option<FastCorner>>,
    ) -> Result<SelectionStatus, FrontendError> {
        image
            .with_dense(|image| self.select_cells(camera, image, select, eligibility, out))
            .map_err(FrontendError::Image)?.map_err(Into::into)
    }
    fn submit_cells(
        &mut self,
        _images: FrameImages<'_>,
        _selects: &[Option<CellSelect>],
    ) -> Result<(), Self::Error> {
        Ok(())
    }
    fn take_cells(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl FrameCornerScan for kornia_staging_imgproc::features::CpuCornerScan {
    fn fork_frame(&self) -> Option<Box<Self>> { Some(Box::new(self.fresh())) }
}
