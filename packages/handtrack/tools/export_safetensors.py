import tyro

from handtrack.apis.export_safetensors import ExportSafetensorsConfig, main

if __name__ == "__main__":
    main(tyro.cli(ExportSafetensorsConfig))
