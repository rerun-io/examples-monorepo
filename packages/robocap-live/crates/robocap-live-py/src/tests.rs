//! Embed Python to test the actual `HandsLayer.push` boundary with a delayed image consumer.
//!
//! The retained images share the queued images' storage. They are read only after Python has mutated the source, even if
//! the pipeline has already finished. This test hook is absent from the extension built for users.

use std::cell::RefCell;

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use super::{FULL_SIZE, NUM_CAMERAS};
use kornia_staging_sensors::CameraFrame;

thread_local! {
    static SUBMITTED: RefCell<[Option<CameraFrame>; NUM_CAMERAS]> = RefCell::new(Default::default());
}

pub(super) fn retain_submitted(cameras: &[Option<CameraFrame>; NUM_CAMERAS]) {
    SUBMITTED.with(|submitted| *submitted.borrow_mut() = cameras.clone());
}

#[pyfunction]
fn _submitted_frame(py: Python<'_>, camera: usize) -> PyResult<Bound<'_, PyArray2<u8>>> {
    SUBMITTED.with(|submitted| {
        let submitted = submitted.borrow();
        let frame = submitted[camera]
            .as_ref()
            .expect("test submitted this camera");
        PyArray1::from_slice(py, frame.full.as_slice()).reshape([FULL_SIZE.height, FULL_SIZE.width])
    })
}

#[test]
fn python_delayed_consumer() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "robocap_live._core")?;
        super::_core(&module)?;
        module.add_function(wrap_pyfunction!(_submitted_frame, &module)?)?;
        let sys = py.import("sys")?;
        sys.getattr("modules")?
            .set_item("robocap_live._core", module)?;
        let package = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        sys.getattr("path")?
            .call_method1("insert", (0, package.to_str().unwrap()))?;
        let test = package.join("tests/test_core.py");
        let result: i32 = py
            .import("pytest")?
            .call_method1(
                "main",
                (vec![test.to_str().unwrap(), "-q", "-k", "delayed_consumer"],),
            )?
            .extract()?;
        sys.getattr("stdout")?.call_method0("flush")?;
        SUBMITTED.with(|submitted| *submitted.borrow_mut() = Default::default());
        assert_eq!(result, 0, "Python snapshot tests failed");
        Ok(())
    })
}
