import tyro

from handtrack.apis.detector_eval import DetectorEvalConfig, main

if __name__ == "__main__":
    main(tyro.cli(DetectorEvalConfig))
