import tyro

from monopriors.apis.monoprior_inference import MonoPriorConfig, monoprior_from_img

if __name__ == "__main__":
    monoprior_from_img(tyro.cli(MonoPriorConfig))
