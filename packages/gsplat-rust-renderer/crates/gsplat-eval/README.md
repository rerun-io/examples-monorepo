# gsplat-eval

`pixi run -e gsplat-rust-renderer --frozen gsplat-eval dirs --render RENDER --gt GT --out report.json`

The default `brush` convention calls pinned Brush code on its Burn wgpu device:
RGBA8 GT, transparent-alpha byte premultiplication onto black, rounded 8-bit
render values, capped PSNR, and Brush's padded SSIM. `--lpips` loads Brush's
compiled VGG weights once and scores the same prepared RGB images.

`--convention published` preserves the Python checkpoint guard: white
composition with byte truncation, f32 PSNR reduction, and float64 valid-window
SSIM. Images are paired by strict relative PNG paths. Both trees must contain
the same nonempty path set; dimensions must match and be at least 11x11.
The report contains per-view values, their arithmetic mean, the convention,
and pinned tool versions. This convention is only for the checkpoint guard.

The library exposes `Evaluator::evaluate_pair` and `evaluate_directories`.
The evaluator accepts `DynamicImage`, including floating-point render output.
Brush treats rendered alpha as already composited on black. Published input
images pass through the historical 8-bit file convention.

Unit tests run with the workspace Rust gate. GPU checks are explicit:
`cargo test --locked --workspace --test evaluation -- --ignored`.
For all downloaded checkpoint sets, run the Python golden test with
`GSPLAT_EVAL_BIN` naming the built binary and `GSPLAT_CHECKPOINT_ROOT` naming
the directory containing scene folders:
`pytest -m golden tests/test_rust_evaluator.py -s`.
Each image must agree within 1e-6 dB PSNR and 1e-7 SSIM.

`dirs` reads PNGs, whose saved RGB values are already clipped to [0,1]. Those
scores need not equal Brush's in-memory evaluation of highlights above 1.
`Evaluator::evaluate_pair` accepts an unclipped float render against byte GT
and matches Brush `eval_stats`. `evaluate_renders` compares two premultiplied
RGBA float renders without quantization or clipping: RGB on black, alpha,
and RGB over white, using Brush PSNR and its padded SSIM formula.
Reports embed resolved Cargo sources, build-time git identity, release settings,
and backend environment variables. PNG previews are evidence, not metric inputs.
