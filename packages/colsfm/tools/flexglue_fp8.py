import tyro

from colsfm.apis.flexglue_fp8 import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
