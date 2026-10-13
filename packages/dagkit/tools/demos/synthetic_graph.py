import tyro

from dagkit.apis.synthetic_demo import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
