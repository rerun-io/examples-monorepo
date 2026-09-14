import tyro

from dataforge.apis.robocap_live_display import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
