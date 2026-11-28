# caspar-mojo

COLMAP's CASPAR GPU bundle adjustment (Levenberg-Marquardt + PCG on the reduced camera system, f32) ported from
~700 Symforce-generated CUDA kernels to a small set of generic Mojo GPU kernels. The compiler instantiates each
kernel per camera model and free-parameter set. numpy arrays in and out (`caspar_mojo.problem`); no torch.

## Build and test

```bash
pixi run -e caspar-mojo-dev --frozen caspar-mojo-build      # builds caspar_mojo/_caspar_mojo.so for the local GPU
pixi run -e caspar-mojo-dev --frozen tests                  # builds first, then the unit lane
pixi run -e caspar-mojo-dev --frozen tests-integration      # GPU tests
```

Mojo builds GPU kernels for the GPU of the host that builds, so build on the host that runs (RTX 5090 sm_120,
RTX 3060 sm_86, GB10 sm_121 all work with no flags). The `colsfm` environments also carry this package, so
CASPAR (pycolmap) and caspar-mojo run in one process.

`tools/bench/` holds standalone probes: `launch_overhead.mojo` (launch, sync and scalar read-back cost; `mojo build`
it, then run the binary) and `mojo_smoke_ext.mojo` + `coexist.py` (CASPAR and a Mojo kernel in one process).

## Results

To be filled in.
