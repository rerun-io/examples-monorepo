import tyro

from dataforge.apis.view import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
