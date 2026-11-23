fn main() {
    println!("cargo:rerun-if-changed=src/kernels.c");

    // Only attempt recompiling PTX if clang is available
    if let Ok(status) = std::process::Command::new("clang")
        .args([
            "-O3",
            "--target=nvptx64-nvidia-cuda",
            "-march=sm_86",
            "-S",
            "src/kernels.c",
            "-o",
            "src/kernels.ptx.tmp",
        ])
        .status()
        && status.success()
        && let Ok(content) = std::fs::read_to_string("src/kernels.ptx.tmp")
    {
        let patched = content.replace(".visible .func", ".visible .entry");
        let _ = std::fs::write("src/kernels.ptx", patched);
        let _ = std::fs::remove_file("src/kernels.ptx.tmp");
    }
}
