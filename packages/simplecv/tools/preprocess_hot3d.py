import tyro

from simplecv.apis.preprocess_hot3d import PreprocessConfig, main

if __name__ == "__main__":
    main(tyro.cli(PreprocessConfig, description="Preprocess HOT3D VRS files to AV1 MP4."))
