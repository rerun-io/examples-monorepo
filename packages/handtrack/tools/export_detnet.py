import tyro

from handtrack.apis.export_detnet import ExportDetNetConfig, main

if __name__ == "__main__":
    main(tyro.cli(ExportDetNetConfig))
