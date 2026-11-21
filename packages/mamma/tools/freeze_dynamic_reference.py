import sys

import tyro

from mamma.apis.freeze_dynamic_reference import Config, main

if __name__ == "__main__":
    sys.exit(main(tyro.cli(Config)))
