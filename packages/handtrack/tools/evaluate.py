import tyro

from handtrack.apis.evaluate import EvaluateConfig, main

if __name__ == "__main__":
    main(tyro.cli(EvaluateConfig))
