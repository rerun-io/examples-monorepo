import tyro

from handtrack.apis.detnet_compare import DetNetCompareConfig, main

if __name__ == "__main__":
    main(tyro.cli(DetNetCompareConfig))
