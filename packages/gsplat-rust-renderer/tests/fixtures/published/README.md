# Published metric references

These 1,600 per-image values were recorded at commit `acfb99eb` and verified
against the former Python evaluator on all eight 200-image nerfbaselines
3dgs-mcmc Blender checkpoints. Maximum observed disagreement was
7.11e-15 dB PSNR and 1.69e-11 SSIM. The JSON files beside this document are
the fixed reference data.

The golden lane compares Rust against these references at 1e-6 dB / 1e-7 SSIM.
Do not regenerate expectations from the implementation under test.
The checkpoints remain external downloaded assets.
