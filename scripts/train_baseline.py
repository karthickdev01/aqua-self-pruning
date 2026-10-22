"""Train the dense baseline and print the learning curve.

This is the "Part 2" evidence script: a dense MLP trained with the
from-scratch Adam optimiser, reporting the learning curve (loss and
accuracy over epochs) and a numerical-stability check.

Usage::

    python scripts/train_baseline.py --epochs 60 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aqua.train.experiment import ExperimentConfig, runExperiment  # noqa: E402
from aqua.utils.serialization import writeJson  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIRECTORY = REPOSITORY_ROOT / "results" / "experiments"


def parseArguments() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description="Train the dense AQUA baseline.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, nargs="+", default=[32, 32])
    parser.add_argument(
        "--output",
        type=str,
        default="baseline_learning_curve.json",
        help="File (in results/experiments) to write the history to.",
    )
    return parser.parse_args()


def main() -> int:
    """Train the baseline and print the learning curve."""
    arguments = parseArguments()
    config = ExperimentConfig(
        epochs=arguments.epochs,
        batchSize=arguments.batch_size,
        learningRate=arguments.learning_rate,
        targetSparsity=0.0,
        criterion="saliency",
        hiddenSizes=list(arguments.hidden),
        seed=arguments.seed,
        name="baseline",
        datasetKwargs={
            "sampleCount": 1200,
            "validationFraction": 0.25,
            "noiseStd": 0.20,
            "turns": 1.2,
        },
    )
    result = runExperiment(config)

    print("epoch  trainLoss  trainAcc  valLoss  valAcc   seconds")
    for record in result.history:
        print(
            f"{record['epoch']:>5}  {record['trainLoss']:>9.4f}  "
            f"{record['trainAccuracy']:>8.4f}  {record['validationLoss']:>7.4f}  "
            f"{record['validationAccuracy']:>6.4f}  {record['durationSeconds']:>7.3f}"
        )

    allFinite = all(
        record["isFinite"]
        and record["trainLoss"] == record["trainLoss"]
        and record["validationLoss"] == record["validationLoss"]
        for record in result.history
    )
    print()
    print(f"numerically stable throughout training: {allFinite}")
    print(f"final train accuracy:      {result.finalMetrics['trainAccuracy']:.4f}")
    print(f"final validation accuracy: {result.finalMetrics['validationAccuracy']:.4f}")
    print(f"total wall clock:          {result.wallClockSeconds:.2f} s")

    outputPath = RESULTS_DIRECTORY / arguments.output
    writeJson(outputPath, result.asDict())
    print(f"history written to {outputPath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
