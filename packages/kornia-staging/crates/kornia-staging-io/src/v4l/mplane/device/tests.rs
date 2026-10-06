use super::*;
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Default)]
struct Driver {
    calls: Vec<String>,
    fail_at: Option<usize>,
    errno: Option<i32>,
    format_edit: Option<fn(&mut PixFormat)>,
    query_edit: Option<fn(&mut Buffer, &mut [Plane])>,
    dequeue_edit: Option<fn(&mut Buffer, &mut [Plane])>,
    allocated: Option<u32>,
    poll_result: Option<Option<i16>>,
    #[allow(clippy::vec_box)] // Simulated mmap addresses must survive Vec growth.
    pixels: Vec<Box<[u8; 64]>>,
}

#[derive(Clone, Default)]
struct Fake(Rc<RefCell<Driver>>);

impl Fake {
    fn call(&self, name: String) -> io::Result<()> {
        let mut driver = self.0.borrow_mut();
        driver.calls.push(name);
        if driver.fail_at == Some(driver.calls.len() - 1) {
            Err(io::Error::from_raw_os_error(
                driver.errno.unwrap_or(libc::EINTR),
            ))
        } else {
            Ok(())
        }
    }
}

// SAFETY: the fake owns its mappings and bounds every simulated kernel write.
unsafe impl Syscalls for Fake {
    fn open(&self, path: &CStr) -> io::Result<c_int> {
        assert_eq!(path, c"/test-camera");
        self.call("open".into())?;
        Ok(123)
    }

    unsafe fn ioctl<T>(&self, fd: c_int, request: c_ulong, arg: &mut T) -> io::Result<()> {
        assert_eq!(fd, 123);
        let ptr = std::ptr::from_mut(arg).cast::<c_void>();
        // SAFETY: Device pairs every request with its UAPI record and live plane array.
        unsafe {
            match request {
                G_FMT | S_FMT => {
                    let format = &mut *ptr.cast::<Format>();
                    assert_eq!(format.kind, CAPTURE_MPLANE);
                    let mut pix = format.fmt.pix_mp;
                    if request == G_FMT {
                        self.call("G_FMT".into())?;
                        pix.width = 320;
                        format.fmt.pix_mp = pix;
                    } else {
                        self.call(format!("S_FMT({})", { pix.width }))?;
                        if pix.width == 8 {
                            if let Some(edit) = self.0.borrow().format_edit {
                                edit(&mut pix);
                            }
                            format.fmt.pix_mp = pix;
                        }
                    }
                }
                REQBUFS => {
                    let req = &mut *ptr.cast::<RequestBuffers>();
                    assert_eq!((req.kind, req.memory), (CAPTURE_MPLANE, MEMORY_MMAP));
                    self.call(format!("REQBUFS({})", req.count))?;
                    if req.count != 0 {
                        if let Some(count) = self.0.borrow().allocated {
                            req.count = count;
                        }
                    }
                }
                QUERYBUF | QBUF | DQBUF => {
                    let buffer = &mut *ptr.cast::<Buffer>();
                    assert_eq!(
                        (buffer.kind, buffer.memory, buffer.length),
                        (CAPTURE_MPLANE, MEMORY_MMAP, 2)
                    );
                    let planes = std::slice::from_raw_parts_mut(buffer.m.planes, 2);
                    if request == QUERYBUF {
                        self.call(format!("QUERYBUF({})", buffer.index))?;
                        for (p, plane) in planes.iter_mut().enumerate() {
                            plane.length = 64;
                            plane.m.mem_offset = buffer.index * 8192 + p as u32 * 4096;
                        }
                        if let Some(edit) = self.0.borrow().query_edit {
                            edit(buffer, planes);
                        }
                    } else if request == QBUF {
                        assert_eq!(
                            planes.iter().map(|p| p.length).collect::<Vec<_>>(),
                            [64, 64]
                        );
                        self.call(format!("QBUF({})", buffer.index))?;
                    } else {
                        self.call("DQBUF".into())?;
                        buffer.index = 1;
                        buffer.timestamp = libc::timeval {
                            tv_sec: 2,
                            tv_usec: 345,
                        };
                        buffer.sequence = 17;
                        buffer.flags = 0x2000;
                        for plane in planes.iter_mut() {
                            plane.data_offset = 3;
                            plane.bytesused = 35;
                        }
                        if let Some(edit) = self.0.borrow().dequeue_edit {
                            edit(buffer, planes);
                        }
                    }
                }
                STREAMON | STREAMOFF => {
                    assert_eq!(*ptr.cast::<u32>(), CAPTURE_MPLANE);
                    self.call(
                        if request == STREAMON {
                            "STREAMON"
                        } else {
                            "STREAMOFF"
                        }
                        .into(),
                    )?;
                }
                _ => panic!("unexpected ioctl {request:#x}"),
            }
        }
        Ok(())
    }

    fn map(&self, fd: c_int, length: usize, offset: u32) -> io::Result<*mut c_void> {
        assert_eq!((fd, length), (123, 64));
        self.call(format!("mmap({offset})"))?;
        let mut pixels = Box::new([42; 64]);
        let ptr = pixels.as_mut_ptr().cast();
        self.0.borrow_mut().pixels.push(pixels);
        Ok(ptr)
    }

    unsafe fn unmap(&self, address: *mut c_void, length: usize) {
        assert_eq!(length, 64);
        let index = self
            .0
            .borrow()
            .pixels
            .iter()
            .position(|p| p.as_ptr().cast::<c_void>() == address)
            .unwrap();
        let _ = self.call(format!("munmap({index})"));
    }

    fn poll(&self, fd: c_int, timeout: u16) -> io::Result<Option<i16>> {
        assert_eq!(fd, 123);
        self.call(format!("poll({timeout})"))?;
        Ok(self.0.borrow().poll_result.unwrap_or(Some(libc::POLLIN)))
    }

    fn close(&self, fd: c_int) {
        assert_eq!(fd, 123);
        let _ = self.call("close".into());
    }
}

fn requested() -> RawFormat {
    RawFormat {
        width: 8,
        height: 4,
        fourcc: u32::from_le_bytes(*b"NM12"),
        planes: 2,
        stride: [8, 8, 0, 0, 0, 0, 0, 0],
        bytes: [32, 16, 0, 0, 0, 0, 0, 0],
    }
}

#[test]
fn c_open_capture_queue_and_teardown_sequence_is_preserved() {
    let driver = Fake::default();
    let camera =
        Device::open_with_syscalls(c"/test-camera", 2, requested(), driver.clone()).unwrap();
    assert_eq!(camera.count(), 2);
    let frame = camera.dequeue(200).unwrap().unwrap();
    assert_eq!(
        (frame.index, frame.timestamp_ns, frame.sequence, frame.flags),
        (1, 2_000_345_000, 17, 0x2000)
    );
    assert_eq!(&frame.lengths[..2], &[32, 32]);
    let pixels = driver.0.borrow();
    assert_eq!(frame.planes[0], pixels.pixels[2].as_ptr().wrapping_add(3));
    assert_eq!(frame.planes[1], pixels.pixels[3].as_ptr().wrapping_add(3));
    drop(pixels);
    camera.queue(frame.index).unwrap();
    drop(camera);
    assert_eq!(
        driver.0.borrow().calls,
        [
            "open",
            "G_FMT",
            "S_FMT(8)",
            "REQBUFS(2)",
            "QUERYBUF(0)",
            "mmap(0)",
            "mmap(4096)",
            "QBUF(0)",
            "QUERYBUF(1)",
            "mmap(8192)",
            "mmap(12288)",
            "QBUF(1)",
            "STREAMON",
            "poll(200)",
            "DQBUF",
            "QBUF(1)",
            "STREAMOFF",
            "munmap(0)",
            "munmap(1)",
            "munmap(2)",
            "munmap(3)",
            "REQBUFS(0)",
            "S_FMT(320)",
            "close",
        ]
    );
}

#[test]
fn each_open_failure_preserves_errno_step_and_partial_cleanup() {
    let opening = [
        "open",
        "G_FMT",
        "S_FMT(8)",
        "REQBUFS(2)",
        "QUERYBUF(0)",
        "mmap(0)",
        "mmap(4096)",
        "QBUF(0)",
        "QUERYBUF(1)",
        "mmap(8192)",
        "mmap(12288)",
        "QBUF(1)",
        "STREAMON",
    ];
    let steps = [
        "open",
        "G_FMT/S_FMT",
        "G_FMT/S_FMT",
        "REQBUFS",
        "QUERYBUF",
        "mmap",
        "mmap",
        "QBUF",
        "QUERYBUF",
        "mmap",
        "mmap",
        "QBUF",
        "STREAMON",
    ];
    for (failure, step) in steps.iter().enumerate() {
        let driver = Fake::default();
        driver.0.borrow_mut().fail_at = Some(failure);
        let error = Device::open_with_syscalls(c"/test-camera", 2, requested(), driver.clone())
            .unwrap_err();
        assert_eq!(error.step, *step);
        assert_eq!(error.source.raw_os_error(), Some(libc::EINTR));
        let mut expected: Vec<String> = opening[..=failure].iter().map(|s| (*s).into()).collect();
        // Expected successful mappings before each failure, worked from the original C loop.
        let mapped = [0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 3, 4, 4][failure];
        expected.extend((0..mapped).map(|i| format!("munmap({i})")));
        if failure > 0 {
            expected.push("REQBUFS(0)".into());
        }
        if failure > 1 {
            expected.push("S_FMT(320)".into());
        }
        if failure > 0 {
            expected.push("close".into());
        }
        assert_eq!(driver.0.borrow().calls, expected, "failure at {failure}");
    }
}

#[test]
fn buffer_count_validation_precedes_open_and_queue_checks_the_allocated_count() {
    for count in [0, 33] {
        let driver = Fake::default();
        let error = Device::open_with_syscalls(c"/test-camera", count, requested(), driver.clone())
            .unwrap_err();
        assert_eq!(error.step, "validate");
        assert_eq!(error.source.raw_os_error(), Some(libc::EINVAL));
        assert!(driver.0.borrow().calls.is_empty());
    }
    let driver = Fake::default();
    let camera =
        Device::open_with_syscalls(c"/test-camera", 2, requested(), driver.clone()).unwrap();
    driver.0.borrow_mut().calls.clear();
    assert_eq!(
        camera.queue(2).unwrap_err().raw_os_error(),
        Some(libc::EINVAL)
    );
    assert!(driver.0.borrow().calls.is_empty());
}

#[test]
fn format_and_allocated_buffer_bounds_match_the_c_checks() {
    let edits: [fn(&mut PixFormat); 6] = [
        |f| f.width += 1,
        |f| f.height += 1,
        |f| f.pixelformat = 0,
        |f| f.num_planes = 1,
        |f| f.plane_fmt[1].bytesperline = 9,
        |f| f.plane_fmt[1].sizeimage = 15,
    ];
    for edit in edits {
        let driver = Fake::default();
        driver.0.borrow_mut().format_edit = Some(edit);
        let error = Device::open_with_syscalls(c"/test-camera", 2, requested(), driver.clone())
            .unwrap_err();
        assert_eq!(
            (error.step, error.source.raw_os_error()),
            ("G_FMT/S_FMT", Some(libc::ENOTSUP))
        );
        assert_eq!(
            driver.0.borrow().calls,
            [
                "open",
                "G_FMT",
                "S_FMT(8)",
                "REQBUFS(0)",
                "S_FMT(320)",
                "close"
            ]
        );
    }
    for count in [0, 33] {
        let driver = Fake::default();
        driver.0.borrow_mut().allocated = Some(count);
        let error = Device::open_with_syscalls(c"/test-camera", 2, requested(), driver.clone())
            .unwrap_err();
        assert_eq!(
            (error.step, error.source.raw_os_error()),
            ("REQBUFS", Some(libc::EOVERFLOW))
        );
        assert_eq!(
            &driver.0.borrow().calls[4..],
            ["REQBUFS(0)", "S_FMT(320)", "close"]
        );
    }
    // Larger sizeimage is accepted, and the allocated count may differ from the request.
    let driver = Fake::default();
    driver.0.borrow_mut().allocated = Some(1);
    driver.0.borrow_mut().format_edit = Some(|f| f.plane_fmt[0].sizeimage = 64);
    let camera =
        Device::open_with_syscalls(c"/test-camera", 2, requested(), driver.clone()).unwrap();
    assert_eq!(camera.count(), 1);
}

#[test]
fn short_query_buffers_clean_up_only_successful_mappings() {
    type Edit = fn(&mut Buffer, &mut [Plane]);
    for (edit, step, suffix) in [
        (
            (|b: &mut Buffer, _: &mut [Plane]| b.length = 1) as Edit,
            "QUERYBUF",
            vec!["REQBUFS(0)", "S_FMT(320)", "close"],
        ),
        (
            (|_: &mut Buffer, p: &mut [Plane]| p[0].length = 31) as Edit,
            "QUERYBUF",
            vec!["REQBUFS(0)", "S_FMT(320)", "close"],
        ),
        (
            (|_: &mut Buffer, p: &mut [Plane]| p[1].length = 15) as Edit,
            "mmap",
            vec!["mmap(0)", "munmap(0)", "REQBUFS(0)", "S_FMT(320)", "close"],
        ),
    ] {
        let driver = Fake::default();
        driver.0.borrow_mut().query_edit = Some(edit);
        let error = Device::open_with_syscalls(c"/test-camera", 2, requested(), driver.clone())
            .unwrap_err();
        assert_eq!(
            (error.step, error.source.raw_os_error()),
            (step, Some(libc::EIO))
        );
        assert_eq!(&driver.0.borrow().calls[5..], suffix);
    }
}

#[test]
fn bad_dequeues_requeue_before_eio_and_queue_errors_take_precedence() {
    type Edit = fn(&mut Buffer, &mut [Plane]);
    let edits: [Edit; 5] = [
        |b, _| b.length = 1,
        |b, _| b.flags |= BUFFER_ERROR,
        |_, p| p[0].data_offset = 36,
        |_, p| p[0].bytesused = 65,
        |_, p| p[1].bytesused = 18,
    ];
    for edit in edits {
        for queue_fails in [false, true] {
            let driver = Fake::default();
            let camera =
                Device::open_with_syscalls(c"/test-camera", 2, requested(), driver.clone())
                    .unwrap();
            {
                let mut state = driver.0.borrow_mut();
                state.calls.clear();
                state.dequeue_edit = Some(edit);
                if queue_fails {
                    state.fail_at = Some(2);
                }
            }
            let error = camera.dequeue(0).err().unwrap();
            assert_eq!(
                error.raw_os_error(),
                Some(if queue_fails { libc::EINTR } else { libc::EIO })
            );
            assert_eq!(driver.0.borrow().calls, ["poll(0)", "DQBUF", "QBUF(1)"]);
        }
    }
    let driver = Fake::default();
    let camera =
        Device::open_with_syscalls(c"/test-camera", 2, requested(), driver.clone()).unwrap();
    driver.0.borrow_mut().dequeue_edit = Some(|b, _| b.index = 2);
    driver.0.borrow_mut().calls.clear();
    assert_eq!(
        camera.dequeue(0).err().unwrap().raw_os_error(),
        Some(libc::EIO)
    );
    assert_eq!(driver.0.borrow().calls, ["poll(0)", "DQBUF"]);
}

#[test]
fn timeout_poll_flags_eintr_and_eagain_are_not_retried() {
    let driver = Fake::default();
    let camera =
        Device::open_with_syscalls(c"/test-camera", 2, requested(), driver.clone()).unwrap();
    driver.0.borrow_mut().calls.clear();
    driver.0.borrow_mut().poll_result = Some(None);
    assert!(camera.dequeue(u16::MAX).unwrap().is_none());
    assert_eq!(driver.0.borrow().calls, ["poll(65535)"]);
    for event in [libc::POLLERR, libc::POLLHUP, libc::POLLNVAL] {
        driver.0.borrow_mut().calls.clear();
        driver.0.borrow_mut().poll_result = Some(Some(event | libc::POLLIN));
        assert_eq!(
            camera.dequeue(200).err().unwrap().raw_os_error(),
            Some(libc::EIO)
        );
        assert_eq!(driver.0.borrow().calls, ["poll(200)"]);
    }
    for (failure, errno) in [(0, libc::EINTR), (1, libc::EINTR), (1, libc::EAGAIN)] {
        {
            let mut state = driver.0.borrow_mut();
            state.calls.clear();
            state.poll_result = None;
            state.fail_at = Some(failure);
            state.errno = Some(errno);
        }
        assert_eq!(
            camera.dequeue(200).err().unwrap().raw_os_error(),
            Some(errno)
        );
        assert_eq!(
            &driver.0.borrow().calls,
            &(["poll(200)", "DQBUF"][..=failure])
        );
    }
}

#[test]
fn teardown_continues_after_each_cleanup_failure() {
    let expected = [
        "STREAMOFF",
        "munmap(0)",
        "munmap(1)",
        "munmap(2)",
        "munmap(3)",
        "REQBUFS(0)",
        "S_FMT(320)",
        "close",
    ];
    for failure in 0..expected.len() {
        let driver = Fake::default();
        let camera =
            Device::open_with_syscalls(c"/test-camera", 2, requested(), driver.clone()).unwrap();
        driver.0.borrow_mut().calls.clear();
        driver.0.borrow_mut().fail_at = Some(failure);
        drop(camera);
        assert_eq!(driver.0.borrow().calls, expected);
    }
}
