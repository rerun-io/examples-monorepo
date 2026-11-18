"""CLI shim: MoGe-2 TensorRT depth + normal priors for a cube-faces layer, in spirula's dataset layout."""

import tyro

from monopriors.apis.moge_spirula_priors import MoGeSpirulaPriorsConfig, main

if __name__ == "__main__":
    main(tyro.cli(MoGeSpirulaPriorsConfig))
