import tyro

from dataforge.apis.download import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
