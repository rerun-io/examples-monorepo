import tyro

from colsfm.apis.raco_export import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
