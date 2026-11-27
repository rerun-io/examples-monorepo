import tyro

from canonical_viewer.apis.build_web import BuildWebConfig, main

if __name__ == "__main__":
    main(tyro.cli(BuildWebConfig))
