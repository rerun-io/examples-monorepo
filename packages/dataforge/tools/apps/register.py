import tyro

from dataforge.apis.register import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
