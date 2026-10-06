#![cfg(feature = "wgpu")]
use kornia_staging_gpu::runtime::gpu_client;
/// This runtime stores every element width the kernels bind.
///
/// The one test that would fire on a fleet machine before any of the others
/// mean anything. The `cubecl-wgpu` WGSL compiler on `u16`/`u8` can
/// panic on cubecl's own worker thread, so the launch reports success and every
/// read comes back as zeros. `probe_storage` copies a known pattern on the
/// device and refuses the runtime if it does not survive; measured on this
/// host, the portable lane without `cubecl-wgpu/spirv` fails exactly here.
#[test]
fn the_runtime_stores_every_element_width_the_kernels_bind() {
    kornia_staging_gpu::runtime::probe_storage(&gpu_client().unwrap()).unwrap();
}

/// A host with no GPU is a typed error, in a subprocess that really has none.
///
/// A missing adapter must not raise `pyo3_runtime.PanicException`, even though
/// CubeCL unwraps its own bring-up on its worker thread and the process's
/// documented contract is a `ValueError` and never a Rust panic (decision D32).
/// It cannot be tested in-process — a client is a per-process singleton and the
/// environment is read once — so each case re-runs *this test binary* with one
/// variable changed and reads what the child printed.
///
/// The child asserts, so a child that stopped reaching the probe fails rather
/// than passing quietly; each case additionally says whether rustc's own
/// `panicked at` belongs in the child's output — for the case the probe answers
/// it must not appear, and for the one that reaches cubecl's own unwrap it
/// must, because the caught panic's message is the only account of a failure no
/// probe anticipated.
mod absent_gpu {
    use std::process::{Command, Output};

    /// Names the child answers to, so the parent can tell it which case to run.
    const CASE: &str = "KORNIA_GPU_ABSENT_CASE";

    /// Run this test binary again as the child of `case`, with `variables` set.
    ///
    /// `--test-threads=1` and `--nocapture` so the child's `println!` reaches
    /// the parent whatever the harness would otherwise do with it.
    fn child(test: &str, case: &str, variables: &[(&str, &str)]) -> String {
        let exe: std::path::PathBuf = std::env::current_exe().unwrap();
        run(Command::new(exe), test, case, variables)
    }

    /// Run `command` as the child of `case` and return everything it printed.
    fn run(mut command: Command, test: &str, case: &str, variables: &[(&str, &str)]) -> String {
        command
            .args(["--exact", test, "--nocapture", "--test-threads=1"])
            .env(CASE, case);
        for (name, value) in variables {
            command.env(name, value);
        }
        let output: Output = command.output().unwrap();
        let text: String = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.status.success(), "the {case} child failed:\n{text}");
        text
    }

    /// `panicked at` is rustc's own header and nothing else prints it, so it is
    /// how a case says whether a panic happened at all — which is not the same
    /// question as whether one escaped. A case that reaches the runtime's own
    /// unwrap panics and is caught; a case the probe answers never panics. Both
    /// return the typed error, and each test below says which it is.
    fn panicked(text: &str) -> bool {
        text.contains("panicked at")
    }

    /// The case this process is the child of, or `None` if it is the parent.
    fn case() -> Option<String> {
        std::env::var(CASE).ok()
    }

    /// Why this host has no client — `unwrap_err` cannot say it, because a
    /// `ComputeClient` is not `Debug`.
    fn client_error() -> kornia_staging_gpu::runtime::GpuError {
        match kornia_staging_gpu::runtime::gpu_client() {
            Err(error) => error,
            Ok(_) => panic!("this child was supposed to have no GPU, and it built a client"),
        }
    }

    /// The shape both cases share, run once as the child and once as the
    /// parent.
    ///
    /// In the child: the client's error is `expected`, and it is printed with
    /// the `CHILD` prefix the parent greps for. In the parent: `spawn` starts
    /// the child, its output contains `expected_text`, and whether rustc's own
    /// `panicked at` appears is exactly `expect_panic` — the two questions each
    /// shim's doc answers for its own case. `None` from `spawn` is a skip and
    /// not a pass.
    fn assert_absent_gpu_case(
        expected: kornia_staging_gpu::runtime::GpuError,
        expected_text: &str,
        expect_panic: bool,
        spawn: impl FnOnce() -> Option<String>,
    ) {
        if case().is_some() {
            let error: kornia_staging_gpu::runtime::GpuError = client_error();
            assert_eq!(error, expected);
            println!("CHILD {error}");
            return;
        }
        let Some(text) = spawn() else {
            println!("SKIPPED: this case's child could not be started, so the case is not a pass");
            return;
        };
        assert!(text.contains(expected_text), "{text}");
        assert_eq!(
            panicked(&text),
            expect_panic,
            "the child's `panicked at` lines are not this case's (expected {expect_panic}):\n{text}"
        );
    }

    /// No Vulkan ICD: the loader enumerates nothing and wgpu has no adapter.
    #[cfg(feature = "wgpu")]
    #[cfg_attr(
        target_os = "macos",
        ignore = "requires the Vulkan ICD loader; Metal ignores VK_DRIVER_FILES"
    )]
    #[test]
    fn a_wgpu_host_with_no_adapter_is_a_typed_error() {
        const NAME: &str = "absent_gpu::a_wgpu_host_with_no_adapter_is_a_typed_error";
        // The probe answers this one too.
        assert_absent_gpu_case(
            kornia_staging_gpu::runtime::GpuError::NoAdapter { backend: "vulkan" },
            "CHILD wgpu found no vulkan adapter",
            false,
            || {
                Some(child(
                    NAME,
                    "no-adapter",
                    &[
                        ("VK_DRIVER_FILES", "/nonexistent/no-such-icd.json"),
                        // Older Vulkan loaders recognize only the legacy name.
                        ("VK_ICD_FILENAMES", "/nonexistent/no-such-icd.json"),
                    ],
                ))
            },
        );
    }

    /// `CUBECL_WGPU_DEFAULT_DEVICE` naming an index the host does not have.
    ///
    /// The case the adapter probe cannot see: cubecl-wgpu selects by
    /// enumeration here rather than by power preference, and panics on its own
    /// thread. It is what the `catch_unwind` in `gpu_client` is for, and it is
    /// the only test that exercises it.
    #[cfg(feature = "wgpu")]
    #[test]
    fn a_wgpu_device_index_past_the_end_is_a_typed_error() {
        const NAME: &str = "absent_gpu::a_wgpu_device_index_past_the_end_is_a_typed_error";
        // This one reaches cubecl's own unwrap, so the runtime's own message
        // must survive to stderr: the child caught the panic and returned the
        // typed error, and without the quiet panic hook the one clue about a
        // case no probe anticipated is still printed rather than swallowed.
        assert_absent_gpu_case(
            kornia_staging_gpu::runtime::GpuError::ClientPanicked { runtime: "wgpu" },
            "CHILD building the wgpu client panicked",
            true,
            || {
                Some(child(
                    NAME,
                    "bad-index",
                    &[("CUBECL_WGPU_DEFAULT_DEVICE", "DiscreteGpu(99)")],
                ))
            },
        );
    }
}
