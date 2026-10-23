import tyro

from mamma.apis.capture import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config, description="Streaming multi-view SMPL-X capture of one catalog segment: track, dense landmarks, triangulate, fit."))
