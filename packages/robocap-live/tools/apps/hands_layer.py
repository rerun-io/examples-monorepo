import tyro

from robocap_live.apis.hands_layer import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
