import tyro

from monopriors.apis.multiview_geometry import MultiviewGeometryCLIConfig, main

if __name__ == "__main__":
    main(tyro.cli(MultiviewGeometryCLIConfig))
