import tyro

from handtrack.apis.stream_rate import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
