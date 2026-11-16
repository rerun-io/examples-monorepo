import tyro

from colsfm.apis.download_galileo import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
