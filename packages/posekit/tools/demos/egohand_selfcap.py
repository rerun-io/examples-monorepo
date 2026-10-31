"""Run the restricted EgoHand SelfCap experiment."""
import tyro

from posekit.apis.egohand_selfcap import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
