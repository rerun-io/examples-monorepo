import tyro

from dataforge.apis.insta360_face_masks import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
