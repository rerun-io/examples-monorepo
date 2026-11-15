import tyro

from colsfm.apis.run import Command, main

if __name__ == "__main__":
    main(tyro.cli(Command))
