import tyro

from posekit.apis.body3d import Body3dConfig, main

if __name__ == "__main__":
    main(tyro.cli(Body3dConfig))
