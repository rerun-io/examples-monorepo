import tyro

from selfcap_soma.overlay import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
