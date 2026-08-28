"""CLI shim for the batched MoGe v2 TensorRT benchmark."""

import tyro

from monopriors.apis.benchmark_moge_v2_trt import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
