import tyro

from posekit.apis.hands_bench import HandsBenchConfig, main

if __name__ == "__main__":
    main(tyro.cli(HandsBenchConfig))
