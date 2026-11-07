fn main() {
    // On macOS an extension module must leave Python symbols undefined for the
    // interpreter to resolve at import time; this emits the required
    // `-undefined dynamic_lookup` linker arguments (no-op on other platforms).
    pyo3_build_config::add_extension_module_link_args();
}
