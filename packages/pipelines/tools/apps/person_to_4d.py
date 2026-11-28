import tyro

from pipelines.apis.person_to_4d import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
