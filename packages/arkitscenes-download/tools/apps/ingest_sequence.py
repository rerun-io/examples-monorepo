import tyro

from arkitscenes_download.ingest.cli import Config, ingest_sequence

if __name__ == "__main__":
    ingest_sequence(tyro.cli(Config))
