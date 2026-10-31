import tyro

from selfcap_curate.apis.dev_cycle import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
