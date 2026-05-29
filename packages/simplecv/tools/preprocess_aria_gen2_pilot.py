import tyro

from simplecv.apis.preprocess_aria_gen2_pilot import PreprocessConfig, main

if __name__ == "__main__":
    main(tyro.cli(PreprocessConfig, description="Preprocess Aria Gen2 Pilot VRS files to AV1 MP4."))
