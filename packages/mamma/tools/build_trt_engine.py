import tyro

from mamma.apis.build_trt_engine import BuildConfig, main

if __name__ == "__main__":
    main(tyro.cli(BuildConfig))
