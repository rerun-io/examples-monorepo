import tyro

from selfcap_curate.apis.run import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
