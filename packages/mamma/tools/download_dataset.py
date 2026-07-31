import sys

import tyro

from mamma.apis.download_dataset import Config, main

if __name__ == "__main__":
    sys.exit(main(tyro.cli(Config)))
