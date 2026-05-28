import tyro

from simplecv.apis.mano_optim_benchmark import ManoOptimBenchConfig, main

# Example usage
if __name__ == "__main__":
    main(
        tyro.cli(
            ManoOptimBenchConfig,
            description="Mano Optimization Benchmark",
        )
    )
