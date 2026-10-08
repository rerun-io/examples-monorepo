//! Direct recording of RoboCap capture streams.

#[cfg(target_os = "linux")]
mod camera;
mod capture;
mod device_profile;
#[cfg(feature = "live-slam")]
pub use device_profile::slam_luma;
pub use device_profile::{
    CAMERAS, CameraSpec, DeviceProfile, FRAME_HEIGHT, FRAME_WIDTH, SENSORS, SLAM_CPUS,
    SLAM_DOWNSCALE, SLAM_PIXELS, STREAMS, SensorChannel,
};
mod display;
pub use display::DisplayAssets;
#[cfg(target_os = "linux")]
mod device;
mod durable_rrd;
#[cfg(target_os = "linux")]
#[cfg(feature = "live-slam")]
mod live_slam;
#[cfg(feature = "gstreamer-capture")]
mod pipeline;
mod segments;
#[cfg(feature = "live-slam")]
pub use live_slam::{
    ImuChannel, LiveSlam, LiveSlamOptions, SLAM_CAMERAS, SlamInput, SlamReport, SlamStatus,
};
#[cfg(feature = "live-slam")]
mod slam_process;
#[cfg(feature = "live-slam")]
pub use slam_process::{SlamProcess, SlamSender, slam_worker};

#[cfg(target_os = "linux")]
pub use camera::{Camera, CameraFrame};
pub use capture::{
    CalibrationSource, CaptureIdentity, DirectWriter, MotionKind, MotionSample, VideoSample,
};
#[cfg(target_os = "linux")]
pub use device::{FrameTrigger, monotonic_ns, start_imu};
pub use durable_rrd::DurableRrdSink;
#[cfg(feature = "gstreamer-capture")]
pub use pipeline::{CapturedBuffer, SamplePipeline};
pub use segments::SegmentedWriter;

#[cfg(all(target_os = "linux", feature = "gstreamer-capture"))]
pub mod session;
