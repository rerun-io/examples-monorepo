import tyro

from monopriors.apis.rig_depth import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
