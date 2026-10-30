import tyro

from selfcap_soma.feet_fit import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
