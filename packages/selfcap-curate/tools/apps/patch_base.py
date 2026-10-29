import tyro

from selfcap_curate.apis.patch_base import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
