# Published metric references

These 1,600 per-image values are the W0-A evaluator results at acfb99eb,
verified against the former Python evaluator on all eight 200-image
nerfbaselines 3dgs-mcmc Blender checkpoints. Maximum observed disagreement
was 7.11e-15 dB PSNR and 1.69e-11 SSIM. Source evidence:
`~/gsplat-modern-work/reports/w0a/published/<scene>.json` and `w0a.md`.

The golden lane compares Rust against these fixed references at 1e-6 dB /
1e-7 SSIM. Do not regenerate expectations from the implementation under test.
The checkpoints remain external downloaded assets.
