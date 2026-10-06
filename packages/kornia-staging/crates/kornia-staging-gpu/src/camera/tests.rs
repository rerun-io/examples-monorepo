use super::*;
use crate::{
    runtime::{guarded, GpuError},
    transfer,
};
use cubecl::prelude::*;
use kornia_staging_3d::camera::{ProjectionReject, UnprojectError};
impl Brown8 {
    /// Project camera-frame points; each output carries the CPU rejection reason.
    ///
    /// # Arguments
    /// * `client` - Target GPU client and stream.
    /// * `points` - Camera-frame `(x,y,z)` points.
    /// * `out` - Reused result storage in input order.
    ///
    /// # Errors
    /// Device failures are returned separately from per-point projection rejects.
    fn project<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        points: &[[f32; 3]],
        out: &mut Vec<Result<[f32; 2], ProjectionReject>>,
    ) -> Result<(), GpuError> {
        out.clear();
        let bytes = self.run(
            client,
            points.as_flattened(),
            points.len(),
            |params, source, output, count| unsafe {
                project_batch::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(count.div_ceil(64) as u32, 1, 1),
                    CubeDim::new_1d(64),
                    BufferArg::from_raw_parts(params, BROWN8_PARAMETERS),
                    BufferArg::from_raw_parts(source, count * 3),
                    BufferArg::from_raw_parts(output, count * 4),
                    count,
                );
            },
        )?;
        for point in f32::from_bytes(&bytes).as_chunks::<4>().0 {
            let result = match point[3] {
                status if status == brown8::VALID as f32 => Ok([point[0], point[1]]),
                status if status == brown8::NON_FINITE as f32 => Err(ProjectionReject::NonFinite),
                status if status == brown8::BELOW_DEPTH as f32 => {
                    Err(ProjectionReject::BelowMinDepth)
                }
                status if status == brown8::OUTSIDE_DOMAIN as f32 => {
                    Err(ProjectionReject::OutsideDomain)
                }
                _ => {
                    out.clear();
                    return Err(GpuError::DeviceReadFailed {
                        what: "camera status",
                    });
                }
            };
            out.push(result);
        }
        Ok(())
    }

    /// Invert pixels to unit bearings with the CPU damped Newton and radius rules.
    ///
    /// # Arguments
    /// * `client` - Target GPU client and stream.
    /// * `pixels` - Image coordinates with top-left pixel centre `(0,0)`.
    /// * `out` - Reused typed results in input order.
    ///
    /// # Errors
    /// Device failures are returned separately from per-pixel inverse rejects.
    fn unproject<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        pixels: &[[f32; 2]],
        out: &mut Vec<Result<[f32; 3], UnprojectError>>,
    ) -> Result<(), GpuError> {
        out.clear();
        let bytes = self.run(
            client,
            pixels.as_flattened(),
            pixels.len(),
            |params, source, output, count| unsafe {
                unproject_batch::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(count.div_ceil(64) as u32, 1, 1),
                    CubeDim::new_1d(64),
                    BufferArg::from_raw_parts(params, BROWN8_PARAMETERS),
                    BufferArg::from_raw_parts(source, count * 2),
                    BufferArg::from_raw_parts(output, count * 4),
                    count,
                );
            },
        )?;
        for point in f32::from_bytes(&bytes).as_chunks::<4>().0 {
            let result = match point[3] {
                status if status == brown8::VALID as f32 => Ok([point[0], point[1], point[2]]),
                status if status == brown8::NON_FINITE as f32 => Err(UnprojectError::NonFinite),
                status if status == brown8::OUTSIDE_DOMAIN as f32 => {
                    Err(UnprojectError::OutsideDomain)
                }
                status if status == brown8::SINGULAR as f32 => Err(UnprojectError::Singular),
                status if status == brown8::NO_CONVERGENCE as f32 => {
                    Err(UnprojectError::NoConvergence)
                }
                _ => {
                    out.clear();
                    return Err(GpuError::DeviceReadFailed {
                        what: "camera status",
                    });
                }
            };
            out.push(result);
        }
        Ok(())
    }

    fn run<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        input: &[f32],
        count: usize,
        launch: impl FnOnce(
            cubecl::server::Handle,
            cubecl::server::Handle,
            cubecl::server::Handle,
            usize,
        ),
    ) -> Result<cubecl::bytes::Bytes, GpuError> {
        if count == 0 {
            return Ok(cubecl::bytes::Bytes::from_elems(Vec::<f32>::new()));
        }
        assert!(
            count <= u32::MAX as usize / 4
                && count.div_ceil(64) <= crate::kernels::MAX_CUBES_PER_DIM as usize
        );
        let expected = count * 4 * size_of::<f32>();
        guarded(
            GpuError::DeviceLost {
                what: "camera operation",
            },
            || {
                let params = transfer::upload(client, f32::as_bytes(&self.parameters))?;
                let source = transfer::upload(client, f32::as_bytes(input))?;
                let output = client.empty(expected);
                launch(params, source, output.clone(), count);
                let reads = transfer::read_buffers(client, vec![output], "camera output")?;
                let [bytes]: [cubecl::bytes::Bytes; 1] =
                    reads.try_into().map_err(|_| GpuError::DeviceReadFailed {
                        what: "camera output",
                    })?;
                if bytes.len() != expected {
                    return Err(GpuError::ShortRead {
                        what: "camera output",
                        actual: bytes.len(),
                        expected,
                    });
                }
                Ok(bytes)
            },
        )
    }
}

#[cube(launch_unchecked)]
pub(crate) fn project_batch(params: &[f32], input: &[f32], out: &mut [f32], count: usize) {
    let index = ABSOLUTE_POS;
    if index >= count {
        terminate!();
    }
    let mut result = Array::<f32>::new(3usize);
    let status = brown8::project(
        params,
        0usize,
        input[index * 3usize],
        input[index * 3usize + 1usize],
        input[index * 3usize + 2usize],
        &mut result,
    );
    out[index * 4usize] = result[0usize];
    out[index * 4usize + 1usize] = result[1usize];
    out[index * 4usize + 2usize] = 0.0f32;
    out[index * 4usize + 3usize] = f32::cast_from(status);
}

#[cube(launch_unchecked)]
pub(crate) fn unproject_batch(params: &[f32], input: &[f32], out: &mut [f32], count: usize) {
    let index = ABSOLUTE_POS;
    if index >= count {
        terminate!();
    }
    let mut result = Array::<f32>::new(3usize);
    let status = brown8::unproject(
        params,
        0usize,
        input[index * 2usize],
        input[index * 2usize + 1usize],
        &mut result,
    );
    out[index * 4usize] = result[0usize];
    out[index * 4usize + 1usize] = result[1usize];
    out[index * 4usize + 2usize] = result[2usize];
    out[index * 4usize + 3usize] = f32::cast_from(status);
}

#[test]
fn calibration_validates_once_and_preserves_device_coefficients() {
    let parameters = [400., 410., 320., 240., 0.1, 0., 0., 0., 0., 0., 0., 0.];
    let camera = Brown8::new(parameters, Some(2.0)).unwrap();
    assert_eq!(&camera.device_parameters()[..12], &parameters);
    approx::assert_abs_diff_eq!(camera.device_parameters()[12], 2.0);
    for radius in [Some(0.0), Some(-1.0), Some(f32::NAN)] {
        assert!(Brown8::new(parameters, radius).is_err());
    }
    for (index, value) in [(0, 0.0), (1, -1.0), (4, f32::INFINITY)] {
        let mut invalid = parameters;
        invalid[index] = value;
        assert!(Brown8::new(invalid, None).is_err());
    }
}

#[cfg(feature = "wgpu")]
mod parity;
