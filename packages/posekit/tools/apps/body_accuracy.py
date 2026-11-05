import tyro

from posekit.apis.body_accuracy import Config, main

if __name__ == "__main__":
    report = main(tyro.cli(Config))
    raise SystemExit(0 if report.passed else 1)
