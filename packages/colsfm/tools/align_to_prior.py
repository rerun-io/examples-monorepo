import tyro

from colsfm.apis.align_to_prior import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
