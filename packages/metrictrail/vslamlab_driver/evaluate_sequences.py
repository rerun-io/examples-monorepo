"""Run VSLAM-LAB's own ``evaluate_sequence`` on an experiment folder written by the MetricTrail harness.

Runs inside VSLAM-LAB's ``vslamlab`` pixi environment with the VSLAM-LAB checkout as the working directory and on
``PYTHONPATH`` (its ``evaluate_sequence`` shells out to ``pixi run -e vslamlab evo_config`` / ``evo_res`` there).
It imports nothing from the harness, so it works with VSLAM-LAB's own dependencies only.

Usage:
    cd <VSLAM-LAB> && PYTHONPATH=$PWD pixi run --frozen -e vslamlab python <this file> \
        --exp-folder <exp> --dataset rgbdtum --sequences rgbd_dataset_freiburg1_desk ...
"""

import argparse  # VSLAM-LAB's env has no tyro; this script is a VSLAM-LAB-side shim, not a harness CLI.
from pathlib import Path
from types import SimpleNamespace

from Datasets.get_dataset import get_dataset
from Evaluate.evaluate_functions import evaluate_sequence
from path_constants import VSLAMLAB_BENCHMARK


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-folder", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    args = parser.parse_args()
    exp_folder: Path = args.exp_folder.resolve()
    # The two attributes evaluate_sequence reads from vslamlab_utilities.Experiment.
    experiment = SimpleNamespace(folder=exp_folder, log_csv=exp_folder / "vslamlab_exp_log.csv")
    dataset = get_dataset(args.dataset, VSLAMLAB_BENCHMARK)
    for sequence in args.sequences:
        evaluate_sequence(experiment, dataset, sequence, overwrite=True)


if __name__ == "__main__":
    main()
