"""CLI shim: export colsfm's registered cube-face frames, people masks and model as a spirula training folder."""

import tyro

from monopriors.apis.spirula_dataset import SpirulaDatasetConfig, main

if __name__ == "__main__":
    main(tyro.cli(SpirulaDatasetConfig))
