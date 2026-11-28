import tyro

from colsfm.apis.ba_synth import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
