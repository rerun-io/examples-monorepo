import tyro

from handtrack.apis.export_nets_onnx import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
