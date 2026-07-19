import tyro

from arkitscenes_download.pipeline import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
