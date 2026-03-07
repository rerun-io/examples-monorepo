import tyro

from monopriors.apis.vda_inference import VDAConfig, vda_inference

if __name__ == "__main__":
    vda_inference(tyro.cli(VDAConfig))
