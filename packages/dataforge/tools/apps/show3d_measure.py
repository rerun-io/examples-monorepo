"""Compare SHOW3D video encoders."""
import tyro

from dataforge.apis.show3d_measure import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
