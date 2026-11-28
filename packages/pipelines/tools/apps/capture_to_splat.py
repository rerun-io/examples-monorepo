import tyro

from pipelines.apis.capture_to_splat import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
