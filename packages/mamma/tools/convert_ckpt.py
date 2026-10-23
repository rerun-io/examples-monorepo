import tyro

from mamma.apis.convert_ckpt import ConvertConfig, main

if __name__ == "__main__":
    main(tyro.cli(ConvertConfig))
