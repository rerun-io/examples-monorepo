import tyro

from simplecv.apis.benchmark_torchcodec_cuda_singleview import BenchmarkConfig, main

if __name__ == "__main__":
    main(tyro.cli(BenchmarkConfig, description="Benchmark TorchCodec CUDA singleview video decoding."))
