//! The one weakened link check the extension module needs, scoped to it.
//!
//! On macOS the CPython symbols an extension module calls are resolved at load
//! time by the interpreter that imports it, so the cdylib has to be linked with
//! them left undefined; maturin and setuptools-rust pass these flags for you and
//! a plain `cargo build` does not, and the link fails on ~200 `_Py*` symbols.
//! Here rather than in `.cargo/config.toml`, where the same two flags were
//! per-triple and workspace-wide: they applied to every crate built for
//! `aarch64-apple-darwin`, the six test binaries included, and an Intel-Mac lane
//! would have needed a third copy. `TARGET_VENDOR` covers every Apple target at
//! once, and `link-arg-cdylib` reaches only this crate's cdylib.
//!
//! The two arguments are clang's own pair — `-undefined` takes its value as the
//! next argument — which is the spelling the workspace config used and the one
//! PyO3's guide gives.
fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_VENDOR").as_deref() == Ok("apple") {
        println!("cargo::rustc-link-arg-cdylib=-undefined");
        println!("cargo::rustc-link-arg-cdylib=dynamic_lookup");
    }
}
