import tyro

from arkitscenes_download.ingest.subdataset import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
