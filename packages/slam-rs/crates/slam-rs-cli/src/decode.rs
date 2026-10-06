//! One decoder per camera; visible Y bytes only, with no decoder work in replay.
use std::ffi::{CString, c_void};

use anyhow::{Context, Result, ensure};
use openh264::formats::YUVSource;

use crate::catalog::{Packet, RigProfile};
unsafe extern "C" {
    fn catalog_av1_open(path: *const libc::c_char) -> *mut c_void;
    fn catalog_av1_send(decoder: *mut c_void, bytes: *const u8, length: usize) -> i32;
    fn catalog_av1_get(
        decoder: *mut c_void,
        bytes: *mut *const u8,
        width: *mut i32,
        height: *mut i32,
        stride: *mut isize,
        full_range: *mut i32,
    ) -> i32;
    fn catalog_av1_close(decoder: *mut c_void);
}

struct Av1(*mut c_void);
impl Drop for Av1 {
    fn drop(&mut self) {
        // SAFETY: constructed only from a successful open; owned once until drop.
        unsafe {
            catalog_av1_close(self.0);
        }
    }
}

pub(super) fn decode<'a>(
    packets: &[Packet],
    profile: &RigProfile,
    width: usize,
    height: usize,
    mut output: impl FnMut(usize) -> Option<&'a mut [u8]>,
) -> Result<()> {
    let luma_range: [u8; 256] = std::array::from_fn(|value| {
        ((value as i32 - 16) * 255 + 109)
            .div_euclid(219)
            .clamp(0, 255) as u8
    });
    let mut count = 0;
    let mut frame = |y: &[u8], w: usize, h: usize, stride: usize, full_range: bool| -> Result<()> {
        ensure!(
            w > 0 && h > 0 && stride >= w && y.len() >= (h - 1) * stride + w,
            "invalid Y plane"
        );
        let downscale = profile.downscale as usize;
        ensure!(
            w % downscale == 0
                && h % downscale == 0
                && (w / downscale, h / downscale) == (width, height),
            "decoded {w}x{h} disagrees with calibration"
        );
        if let Some(gray) = output(count) {
            ensure!(
                gray.len() == width * height,
                "output buffer disagrees with calibration"
            );
            if profile.downscale > 1 {
                kornia_staging_imgproc::resize::resize_area_u8_into::<1>(
                    y,
                    (w, h),
                    stride,
                    gray,
                    (width, height),
                )?;
            } else {
                for (source, row) in y.chunks(stride).take(h).zip(gray.chunks_exact_mut(width)) {
                    if full_range {
                        row.copy_from_slice(&source[..w]);
                    } else {
                        for (target, &pixel) in row.iter_mut().zip(source) {
                            *target = luma_range[usize::from(pixel)];
                        }
                    }
                }
            }
        }
        count += 1;
        Ok(())
    };
    if profile.codec == *b"avc1" {
        let mut decoder = openh264::decoder::Decoder::with_api_config(
            openh264::OpenH264API::from_source(),
            openh264::decoder::DecoderConfig::new()
                .flush_after_decode(openh264::decoder::Flush::NoFlush),
        )?;
        for packet in packets {
            if let Some(picture) = decoder.decode(&packet.data)? {
                let (w, h) = picture.dimensions();
                frame(picture.y(), w, h, picture.strides().0, true)?;
            }
        }
        for picture in decoder.flush_remaining()? {
            let (w, h) = picture.dimensions();
            frame(picture.y(), w, h, picture.strides().0, true)?;
        }
    } else {
        let library = CString::new(
            std::env::var("SLAM_RS_DAV1D_LIB").unwrap_or_else(|_| "libdav1d.so.6".into()),
        )?;
        // SAFETY: C string lives through open. The bridge copies all settings.
        let pointer = unsafe { catalog_av1_open(library.as_ptr()) };
        ensure!(
            !pointer.is_null(),
            "cannot open dav1d 1.x; install libdav1d.so.6 or set SLAM_RS_DAV1D_LIB"
        );
        let decoder = Av1(pointer);
        for packet in packets {
            // SAFETY: decoder is live, packet bytes live for this synchronous copy.
            let status =
                unsafe { catalog_av1_send(decoder.0, packet.data.as_ptr(), packet.data.len()) };
            ensure!(status == 0, "dav1d send failed: {status}");
            drain(&decoder, &mut frame)?;
        }
        drain(&decoder, &mut frame)?;
    }
    ensure!(
        count == packets.len(),
        "decoded {count} frames from {} packets",
        packets.len()
    );
    Ok(())
}

fn drain(
    decoder: &Av1,
    output: &mut impl FnMut(&[u8], usize, usize, usize, bool) -> Result<()>,
) -> Result<()> {
    loop {
        let (mut bytes, mut width, mut height, mut stride, mut full_range) =
            (std::ptr::null(), 0, 0, 0, 0);
        // SAFETY: each out pointer is writable; picture stays owned by decoder
        // until the next get/close. output consumes its borrowed plane first.
        let status = unsafe {
            catalog_av1_get(
                decoder.0,
                &mut bytes,
                &mut width,
                &mut height,
                &mut stride,
                &mut full_range,
            )
        };
        if status == -libc::EAGAIN {
            break;
        }
        ensure!(status == 0, "dav1d get failed: {status}");
        ensure!(
            !bytes.is_null() && width > 0 && height > 0 && stride >= width as isize,
            "invalid dav1d picture"
        );
        let length = (stride as usize)
            .checked_mul(height as usize - 1)
            .and_then(|n| n.checked_add(width as usize))
            .context("Y plane size overflow")?;
        // SAFETY: dav1d guarantees stride * (height - 1) + width readable bytes
        // for this 8-bit Y plane; the C bridge rejects other bit depths.
        let y = unsafe { std::slice::from_raw_parts(bytes, length) };
        output(
            y,
            width as usize,
            height as usize,
            stride as usize,
            full_range != 0,
        )?;
    }
    Ok(())
}
