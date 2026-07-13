import tyro

from arkitscenes_download.ingest.batch import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
