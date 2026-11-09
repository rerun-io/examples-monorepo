"""Fixed local-cache hand evaluation and timing."""
import tyro

from posekit.apis.hands_cached_bench import Config, main

if __name__ == '__main__':
    main(tyro.cli(Config))
