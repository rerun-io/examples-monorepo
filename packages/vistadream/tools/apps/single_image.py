import tyro

from vistadream.apis.single_image import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config, description="One image to a 3D Gaussian scene (MoGe depth, optional FLUX outpainting) and a camera path through it."))
