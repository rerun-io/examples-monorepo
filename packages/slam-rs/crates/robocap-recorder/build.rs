fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Reuse the earlier cap binary's linkage: carry Rust's unwinder from the
    // toolchain archives instead of requiring a compiler runtime on the cap.
    let target = std::env::var("TARGET")?;
    if target.contains("linux") {
        println!("cargo:rerun-if-changed=native/camera.c");
        cc::Build::new()
            .file("native/camera.c")
            .warnings_into_errors(true)
            .compile("cap_camera");
    }
    if target == "aarch64-unknown-linux-gnu" {
        let compiler_key = format!("CC_{}", target.replace('-', "_"));
        println!("cargo:rerun-if-env-changed={compiler_key}");
        println!("cargo:rerun-if-env-changed=CC");
        let compiler = std::env::var(&compiler_key)
            .or_else(|_| std::env::var("CC"))
            .unwrap_or_else(|_| "cc".to_owned());
        let output = std::process::Command::new(compiler)
            .arg("-print-file-name=libgcc_eh.a")
            .output()?;
        let archive = std::path::PathBuf::from(std::str::from_utf8(&output.stdout)?.trim());
        if !output.status.success() || !archive.is_file() {
            return Err(std::io::Error::other(
                "compiler did not resolve the static unwinder archive",
            )
            .into());
        }
        let directory = archive
            .parent()
            .ok_or_else(|| std::io::Error::other("unwinder archive has no directory"))?;
        println!("cargo:rustc-link-search=native={}", directory.display());
        println!("cargo:rustc-link-lib=static=gcc_eh");
        println!("cargo:rustc-link-lib=static=gcc");
    }
    Ok(())
}
