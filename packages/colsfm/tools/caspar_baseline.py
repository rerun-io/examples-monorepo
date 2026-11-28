import tyro

from colsfm.apis.caspar_baseline import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
