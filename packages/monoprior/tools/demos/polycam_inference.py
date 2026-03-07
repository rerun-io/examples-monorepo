import tyro

from monopriors.apis.polycam_inference import PolycamConfig, polycam_inference

if __name__ == "__main__":
    polycam_inference(tyro.cli(PolycamConfig))
