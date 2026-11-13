import tyro

from posekit.apis.label_propose import LabelProposeConfig, main

if __name__ == "__main__":
    main(tyro.cli(LabelProposeConfig))
