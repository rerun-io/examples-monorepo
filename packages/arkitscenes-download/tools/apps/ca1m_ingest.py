import tyro

from arkitscenes_download.ca1m.ingest import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
