import tyro

from posekit.apis.hands_eval import HandsEvalConfig, main

if __name__ == "__main__":
    main(tyro.cli(HandsEvalConfig))
