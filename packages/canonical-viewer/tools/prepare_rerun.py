import tyro

from canonical_viewer.apis.prepare_rerun import PrepareRerunConfig, main

if __name__ == "__main__":
    main(tyro.cli(PrepareRerunConfig))
