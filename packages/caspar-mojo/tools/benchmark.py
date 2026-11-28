import tyro

from caspar_mojo.apis.benchmark import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
