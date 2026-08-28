import tyro

from zipdepth.apis.train_smoke import TrainSmokeConfig, train_smoke

if __name__ == "__main__":
    train_smoke(tyro.cli(TrainSmokeConfig))
