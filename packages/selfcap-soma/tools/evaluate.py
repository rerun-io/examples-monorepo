import tyro

from selfcap_soma.evaluate import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
