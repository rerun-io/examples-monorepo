"""Compare layer recordings at their source timestamps."""

import tyro

from dataforge.apis.compare_layers import main

if __name__ == "__main__":
    tyro.cli(main)
