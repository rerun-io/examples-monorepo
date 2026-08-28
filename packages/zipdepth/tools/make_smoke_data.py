import tyro

from zipdepth.apis.smoke_data import SmokeDataConfig, build_smoke_data

if __name__ == "__main__":
    build_smoke_data(tyro.cli(SmokeDataConfig))
