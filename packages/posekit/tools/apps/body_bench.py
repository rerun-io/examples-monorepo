import tyro

from posekit.apis.body_bench import BodyBenchConfig, main

if __name__ == "__main__":
    main(tyro.cli(BodyBenchConfig))
