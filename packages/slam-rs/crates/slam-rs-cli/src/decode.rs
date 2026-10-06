//! Catalog replay conversion over borrowed decoder planes.
use crate::catalog::{Packet, RigProfile};
use anyhow::{Result, bail, ensure};
use kornia_image::ImageSize;
use kornia_staging_io::video::{ColorRange, PacketCodec, decode_packets};

pub(super) fn decode<'a>(
    packets: &[Packet],
    profile: &RigProfile,
    width: usize,
    height: usize,
    mut output: impl FnMut(usize) -> Option<&'a mut [u8]>,
) -> Result<()> {
    let geometry = LumaConverter::new(ImageSize { width, height }, profile.downscale as usize)?;
    let library = std::env::var("SLAM_RS_DAV1D_LIB").unwrap_or_else(|_| "libdav1d.so.6".into());
    // Preserve the catalog's avc1-or-AV1 selection and replay range behavior.
    let codec = if profile.codec == *b"avc1" {
        PacketCodec::H264
    } else {
        PacketCodec::Av1 { library: &library }
    };
    let count = decode_packets(
        packets.iter().map(|p| p.data.as_ref()),
        codec,
        |index, y, size, stride, range| {
            geometry.validate_plane(y, size, stride)?;
            // Legacy replay copies H.264 luma without range expansion, regardless of metadata.
            let bypass_range = codec == PacketCodec::H264 || range == ColorRange::Full;
            geometry.copy_validated(y, stride, bypass_range, output(index))
        },
    )?;
    ensure!(
        count == packets.len(),
        "decoded {count} frames from {} packets",
        packets.len()
    );
    Ok(())
}

// Fixed limited-range conversion, shared by every decoded frame.
const STUDIO_RANGE: [u8; 256] = {
    let mut values = [0; 256];
    let mut i = 0;
    while i < 256 {
        let value = ((i as i32 - 16) * 255 + 109).div_euclid(219);
        values[i] = if value < 0 {
            0
        } else if value > 255 {
            255
        } else {
            value as u8
        };
        i += 1;
    }
    values
};

/// Validated output geometry.
struct LumaConverter {
    output: ImageSize,
    input: ImageSize,
    downscale: usize,
}
impl LumaConverter {
    /// Set output size and integer downscale before decoding.
    /// # Arguments
    /// * `output` - destination dimensions.
    /// * `downscale` - positive integer reduction factor.
    /// # Errors
    /// Rejects zero or overflowing geometry.
    fn new(output: ImageSize, downscale: usize) -> Result<Self> {
        let input = output
            .width
            .checked_mul(downscale)
            .zip(output.height.checked_mul(downscale));
        let Some((width, height)) = input else {
            bail!("geometry overflow");
        };
        if width == 0 || height == 0 || output.width.checked_mul(output.height).is_none() {
            bail!("zero or overflowing geometry");
        }
        Ok(Self {
            output,
            input: ImageSize { width, height },
            downscale,
        })
    }

    fn validate_plane(&self, y: &[u8], size: ImageSize, stride: usize) -> Result<()> {
        kornia_staging_io::video::validate_y_plane(y, size, stride)?;
        if size != self.input {
            bail!("decoded geometry disagrees with output");
        }
        Ok(())
    }

    fn copy_validated(
        &self,
        y: &[u8],
        stride: usize,
        full_range: bool,
        out: Option<&mut [u8]>,
    ) -> Result<()> {
        let size = self.input;
        if let Some(gray) = out {
            if gray.len() != self.output.width * self.output.height {
                bail!("output buffer size mismatch");
            }
            if self.downscale > 1 {
                kornia_staging_imgproc::resize::resize_area_u8_into::<1>(
                    y,
                    (size.width, size.height),
                    stride,
                    gray,
                    (self.output.width, self.output.height),
                )?;
            } else {
                for (source, row) in y
                    .chunks(stride)
                    .take(size.height)
                    .zip(gray.chunks_exact_mut(self.output.width))
                {
                    if full_range {
                        row.copy_from_slice(&source[..size.width]);
                    } else {
                        for (target, &pixel) in row.iter_mut().zip(source) {
                            *target = STUDIO_RANGE[usize::from(pixel)];
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
impl LumaConverter {
    fn convert(
        &self,
        y: &[u8],
        size: ImageSize,
        stride: usize,
        full_range: bool,
        out: Option<&mut [u8]>,
    ) -> Result<()> {
        self.validate_plane(y, size, stride)?;
        self.copy_validated(y, stride, full_range, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h264_replay_preserves_native_luma_without_range_conversion() -> Result<()> {
        let packets = [Packet {
            t: 0,
            data: include_bytes!(
                "../../../../kornia-staging/crates/kornia-staging-io/tests/data/black-16.h264"
            )
            .to_vec()
            .into(),
        }];
        for downscale in [1, 2] {
            let profile = RigProfile {
                downscale,
                ..slam_rs::catalog_timing::ROBOCAP
            };
            let width = 16 / downscale as usize;
            let mut pixels = vec![0; width * width];
            let mut destination = Some(pixels.as_mut_slice());
            decode(&packets, &profile, width, width, |index| {
                assert_eq!(index, 0);
                destination.take()
            })?;
            assert!(destination.is_none());
            assert_eq!(pixels, vec![16; width * width]);
        }
        Ok(())
    }

    #[test]
    fn invalid_picture_does_not_request_output_storage() {
        let geometry = LumaConverter::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            1,
        )
        .unwrap();
        let mut requests = 0;
        let result = decode_packets(
            [include_bytes!(
                "../../../../kornia-staging/crates/kornia-staging-io/tests/data/black-16.h264"
            )
            .as_slice()],
            PacketCodec::H264,
            |_, y, size, stride, range| {
                geometry.validate_plane(y, size, stride)?;
                requests += 1;
                geometry.copy_validated(y, stride, range == ColorRange::Full, None)
            },
        );
        assert!(result.is_err());
        assert_eq!(requests, 0);
    }

    #[test]
    fn limited_range_downscale_retains_the_observed_lut_gap() -> Result<()> {
        let y = [16, 16, 99, 99, 16, 16];
        let size = ImageSize {
            width: 2,
            height: 2,
        };
        let mut full = [255; 4];
        LumaConverter::new(size, 1)?.convert(&y, size, 4, false, Some(&mut full))?;
        assert_eq!(full, [0; 4]);
        let mut small = [0];
        LumaConverter::new(
            ImageSize {
                width: 1,
                height: 1,
            },
            2,
        )?
        .convert(&y, size, 4, false, Some(&mut small))?;
        assert_eq!(
            small,
            [16],
            "SF-18: downscale >1 bypasses the limited-range LUT"
        );
        Ok(())
    }
    #[test]
    fn full_range_copies_visible_rows_and_checks_destination() -> Result<()> {
        let size = ImageSize {
            width: 2,
            height: 2,
        };
        let converter = LumaConverter::new(size, 1)?;
        let mut out = [0; 4];
        converter.convert(&[1, 2, 99, 99, 3, 4], size, 4, true, Some(&mut out))?;
        assert_eq!(out, [1, 2, 3, 4]);
        assert!(
            converter
                .convert(&[1, 2, 3], size, 2, true, Some(&mut out))
                .is_err()
        );
        assert!(
            converter
                .convert(&[1, 2, 3, 4], size, 2, true, Some(&mut [0; 3]))
                .is_err()
        );
        assert!(
            converter
                .convert(
                    &[1; 8],
                    ImageSize {
                        width: 4,
                        height: 2
                    },
                    4,
                    true,
                    None
                )
                .is_err()
        );
        converter.convert(&[1, 2, 3, 4], size, 2, true, None)?;
        assert!(LumaConverter::new(size, 0).is_err());
        Ok(())
    }
}
