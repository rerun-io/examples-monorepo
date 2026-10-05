fn main() {
    #[cfg(feature = "catalog")]
    {
        println!("cargo:rerun-if-env-changed=DAV1D_INCLUDE_DIR");
        println!("cargo:rerun-if-changed=native/dav1d.c");
        let mut build = cc::Build::new();
        build.file("native/dav1d.c").warnings_into_errors(true);
        if let Some(include) = std::env::var_os("DAV1D_INCLUDE_DIR") {
            build.include(include);
        }
        build.compile("catalog_dav1d");
        println!("cargo:rustc-link-lib=dl");
    }
}
