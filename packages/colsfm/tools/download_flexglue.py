import tyro

from colsfm.apis.download_flexglue import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
