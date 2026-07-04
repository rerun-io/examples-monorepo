import tyro

from simplecv.apis.preprocess_mamma import PreprocessConfig, main

if __name__ == "__main__":
    main(tyro.cli(PreprocessConfig, description="Re-encode downloaded MAMMA videos to AV1 yuv420 (videos_av1 mirror)."))
