import tyro

from dataforge.apis.soma_x import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
