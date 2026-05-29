import tyro

from simplecv.apis.preprocess_epfl_smart_kitchen import PreprocessConfig, main

if __name__ == "__main__":
    main(tyro.cli(PreprocessConfig, description="Build an AV1 EPFL-Smart-Kitchen mirror for SimpleCV."))
