import tyro

from robocap_live.apis.hands_layer import Cli, main

if __name__ == "__main__":
    main(tyro.cli(Cli))
