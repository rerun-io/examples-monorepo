//! macOS: the CPython symbols an extension module calls are resolved at load time by the interpreter that imports it,
//! so the cdylib is linked with them left undefined (what maturin passes for you; a plain `cargo build` does not).
//! Scoped to Apple targets and to this crate's cdylib, as slam-rs-py's build.rs.
fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_VENDOR").as_deref() == Ok("apple") {
        println!("cargo::rustc-link-arg-cdylib=-undefined");
        println!("cargo::rustc-link-arg-cdylib=dynamic_lookup");
    }
}
