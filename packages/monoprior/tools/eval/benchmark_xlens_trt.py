"""CLI shim for the X-Lens eager / frozen-geometry / TensorRT rig benchmark."""

import tyro

from monopriors.apis.benchmark_xlens_trt import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
