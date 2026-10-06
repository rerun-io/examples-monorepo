//! The tracker's [`Perception`] on the real networks: the DetNet letterbox + decode ([`detect`]) and perspective-crop
//! KeyNet estimator ([`PerspectiveKeyNet`]), the stages robocap_track.py hands handtrack's `Tracker`.

use nalgebra::Isometry3;

use super::HandsError;
use super::detect::{Detections, detect};
use super::estimator::{KeypointEstimate, PerspectiveKeyNet, ViewRequest};
use super::letterbox::BarLetterbox;
use super::tracker::Perception;
use crate::frame::Luma;
use crate::frame::{NUM_CAMERAS, Rig};
use crate::nets::HandNets;
use kornia_staging_sensors::CameraFrame;

/// DetNet + perspective KeyNet behind [`Perception`].
pub struct NetsPerception {
    estimator: PerspectiveKeyNet,
    letterbox: BarLetterbox,
}

impl NetsPerception {
    /// For a RoboCap rig (1920x1080 KB4 cameras); the tracker sets the hand scale ([`Perception::set_phi`]).
    ///
    /// # Errors
    ///
    /// The estimator's errors for a rig it cannot model.
    pub fn new(rig: &Rig) -> Result<Self, HandsError> {
        Ok(Self {
            estimator: PerspectiveKeyNet::new(rig, 1.0)?,
            letterbox: BarLetterbox::robocap(),
        })
    }
}

impl Perception for NetsPerception {
    fn detect(
        &mut self,
        nets: &mut dyn HandNets,
        _cameras: &[usize],
        small: &[&Luma],
    ) -> Result<Vec<Detections>, HandsError> {
        let images: Vec<&kornia_image::Image<u8, 1>> =
            small.iter().map(|image| image.as_ref()).collect();
        detect(nets, &self.letterbox, &images)
    }

    fn estimate(
        &mut self,
        nets: &mut dyn HandNets,
        full: &[Option<&CameraFrame>; NUM_CAMERAS],
        turned_180: &[bool; NUM_CAMERAS],
        world_from_rig: &Isometry3<f64>,
        views: &[ViewRequest],
    ) -> Result<(Vec<KeypointEstimate>, f64), HandsError> {
        self.estimator
            .estimate(nets, full, turned_180, world_from_rig, views)
    }

    fn set_phi(&mut self, phi: f64) {
        self.estimator.set_phi(phi);
    }
}
