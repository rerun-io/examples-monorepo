import tyro

from selfcap_soma.api import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
