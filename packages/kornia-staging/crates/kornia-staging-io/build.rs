fn main() {
    #[cfg(feature = "decoder")]
    if std::env::var("CARGO_CFG_TARGET_FAMILY").as_deref() == Ok("unix") {
        println!("cargo:rerun-if-env-changed=DAV1D_INCLUDE_DIR");
        println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
        println!("cargo:rerun-if-changed=src/video/dav1d.c");
        let mut build = cc::Build::new();
        build.file("src/video/dav1d.c").warnings_into_errors(true);
        if let Some(include) = std::env::var_os("DAV1D_INCLUDE_DIR")
            .map(std::path::PathBuf::from)
            .or_else(|| {
                std::env::var_os("CONDA_PREFIX")
                    .map(|prefix| std::path::PathBuf::from(prefix).join("include"))
            })
        {
            build.include(include);
        }
        build.compile("kornia_staging_dav1d");
        if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
            println!("cargo:rustc-link-lib=dl");
        }
    }
}
