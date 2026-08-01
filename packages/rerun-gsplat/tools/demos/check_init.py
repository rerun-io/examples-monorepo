import tyro

from rerun_gsplat.apis.check_init import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
