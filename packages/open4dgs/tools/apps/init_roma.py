import tyro

from open4dgs.apis.init_roma import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
