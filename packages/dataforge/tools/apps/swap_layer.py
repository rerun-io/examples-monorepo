import tyro

from dataforge.apis.swap_layer import SwapConfig, main

if __name__ == "__main__":
    main(tyro.cli(SwapConfig))
