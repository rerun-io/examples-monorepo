import tyro

from posekit.apis.people_masks import PeopleMasksConfig, main

if __name__ == "__main__":
    main(tyro.cli(PeopleMasksConfig))
