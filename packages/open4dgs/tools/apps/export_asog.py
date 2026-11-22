import tyro

from open4dgs.apis.export_asog import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
