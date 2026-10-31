import tyro

from selfcap_curate.apis.post import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
