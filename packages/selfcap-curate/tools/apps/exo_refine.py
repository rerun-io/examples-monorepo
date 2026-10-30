import tyro

from selfcap_curate.apis.exo_refine import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
