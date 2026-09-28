//! Build the FMI 2 variadic logger callback shim.
fn main() {
    cc::Build::new()
        .file("src/fmi2_logger.c")
        .compile("raichu_fmi2_logger");
}
