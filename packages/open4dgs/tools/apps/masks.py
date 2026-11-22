import tyro

from open4dgs.apis.masks import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
