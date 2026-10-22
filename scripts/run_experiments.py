"""Run the full experiment suite and write the raw results.

The suite covers:

* a sparsity sweep (0%, 50%, 75%, 90%, 95%) for every importance criterion,
* multiple random seeds per point,
* an optional re-growth ablation,
* cost measurement (MAC counts and latency) for every trained model.

Everything is written to ``results/experiments/`` as JSON and CSV so that
the numbers in ``README.md`` and ``DESIGN.md`` can be traced back to the raw
data.

Usage::

    python scripts/run_experiments.py --seeds 5 --epochs 60
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aqua.train.experiment import ExperimentConfig, runExperiment  # noqa: E402
from aqua.utils.serialization import ensureDirectory, writeCsv, writeJson  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIRECTORY = REPOSITORY_ROOT / "results" / "experiments"

DEFAULT_DATASET_KWARGS: Dict[str, Any] = {
    "sampleCount": 1200,
    "validationFraction": 0.25,
    "noiseStd": 0.20,
    "turns": 1.2,
}

DEFAULT_HIDDEN_SIZES = [32, 32]
DEFAULT_SPARSITIES = [0.0, 0.5, 0.75, 0.9, 0.95]
DEFAULT_CRITERIA = ["magnitude", "saliency", "signedTaylor", "gradientMagnitude"]


def parseArguments() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the AQUA pruning experiments.")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds per point.")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs per run.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--sparsities",
        type=float,
        nargs="+",
        default=DEFAULT_SPARSITIES,
        help="Target sparsity levels to sweep.",
    )
    parser.add_argument(
        "--criteria",
        nargs="+",
        default=DEFAULT_CRITERIA,
        help="Importance criteria to compare.",
    )
    parser.add_argument(
        "--include-regrowth",
        action="store_true",
        help="Also run the re-growth ablation for the saliency criterion.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="experiments",
        help="Prefix of the written result files.",
    )
    return parser.parse_args()


def _lookupLatency(latency: Dict[str, Any], path: str, batchSize: int) -> float:
    """Return the median latency of ``path`` at ``batchSize``.

    The batch-size keys of the latency dictionary are integers, but they are
    read back from CSV as strings.  This helper accepts either, so that the
    aggregation code cannot silently read a zero.
    """
    entries = latency.get(path, {})
    for key in (batchSize, str(batchSize)):
        measurement = entries.get(key)
        if measurement is not None:
            return float(measurement.get("medianSeconds", 0.0))
    return 0.0


def _buildRow(
    result: Any,
    criterion: str,
    targetSparsity: float,
    seed: int,
    withRegrowth: bool,
) -> Dict[str, Any]:
    """Flatten one experiment result into a table row."""
    cost = result.costMetrics
    latency = cost.get("latency", {})
    denseLatency = _lookupLatency(latency, "dense", 32)
    sparseLatency = _lookupLatency(latency, "sparseAware", 32)
    denseLatency1 = _lookupLatency(latency, "dense", 1)
    sparseLatency1 = _lookupLatency(latency, "sparseAware", 1)
    denseLatency256 = _lookupLatency(latency, "dense", 256)
    sparseLatency256 = _lookupLatency(latency, "sparseAware", 256)
    revived = sum(int(event.get("revivedCount", 0)) for event in result.regrowthEvents)

    return {
        "criterion": criterion,
        "targetSparsity": targetSparsity,
        "seed": seed,
        "withRegrowth": withRegrowth,
        "trainLoss": result.finalMetrics["trainLoss"],
        "trainAccuracy": result.finalMetrics["trainAccuracy"],
        "validationLoss": result.finalMetrics["validationLoss"],
        "validationAccuracy": result.finalMetrics["validationAccuracy"],
        "achievedSparsity": result.finalMetrics["achievedSparsity"],
        "activeParameters": result.finalMetrics["activeParameters"],
        "totalParameters": result.finalMetrics["totalParameters"],
        "totalScalarParameters": result.finalMetrics["totalScalarParameters"],
        "denseMacs": cost["denseMacs"],
        "activeMacs": cost["activeMacs"],
        "theoreticalMacReduction": cost["theoreticalMacReduction"],
        "denseLatencySeconds": denseLatency,
        "sparseLatencySeconds": sparseLatency,
        "measuredLatencyRatio": (sparseLatency / denseLatency) if denseLatency else 0.0,
        "denseLatencyBatch1Seconds": denseLatency1,
        "sparseLatencyBatch1Seconds": sparseLatency1,
        "measuredLatencyRatioBatch1": (sparseLatency1 / denseLatency1) if denseLatency1 else 0.0,
        "denseLatencyBatch256Seconds": denseLatency256,
        "sparseLatencyBatch256Seconds": sparseLatency256,
        "measuredLatencyRatioBatch256": (
            (sparseLatency256 / denseLatency256) if denseLatency256 else 0.0
        ),
        "sparseMatchesDense": cost["sparseMatchesDense"],
        "revivedConnections": revived,
        "wallClockSeconds": result.wallClockSeconds,
    }


def runSuite(arguments: argparse.Namespace) -> List[Dict[str, Any]]:
    """Run every configured experiment and return the flat result rows."""
    rows: List[Dict[str, Any]] = []
    ensureDirectory(RESULTS_DIRECTORY)

    for criterion in arguments.criteria:
        for targetSparsity in arguments.sparsities:
            for seed in range(arguments.seeds):
                config = ExperimentConfig(
                    epochs=arguments.epochs,
                    batchSize=arguments.batch_size,
                    learningRate=arguments.learning_rate,
                    targetSparsity=targetSparsity,
                    criterion=criterion,
                    hiddenSizes=list(DEFAULT_HIDDEN_SIZES),
                    seed=seed,
                    enableRegrowth=False,
                    name=f"{criterion}-s{targetSparsity:.2f}-seed{seed}",
                    datasetKwargs=dict(DEFAULT_DATASET_KWARGS),
                )
                result = runExperiment(config)
                row = _buildRow(result, criterion, targetSparsity, seed, False)
                rows.append(row)
                print(
                    f"[{criterion:<18}] target={targetSparsity:.2f} seed={seed} "
                    f"valAcc={row['validationAccuracy']:.4f} "
                    f"achieved={row['achievedSparsity']:.4f}",
                    flush=True,
                )

    if arguments.include_regrowth:
        for targetSparsity in [value for value in arguments.sparsities if value > 0.0]:
            for seed in range(arguments.seeds):
                config = ExperimentConfig(
                    epochs=arguments.epochs,
                    batchSize=arguments.batch_size,
                    learningRate=arguments.learning_rate,
                    targetSparsity=targetSparsity,
                    criterion="saliency",
                    hiddenSizes=list(DEFAULT_HIDDEN_SIZES),
                    seed=seed,
                    enableRegrowth=True,
                    name=f"regrowth-s{targetSparsity:.2f}-seed{seed}",
                    datasetKwargs=dict(DEFAULT_DATASET_KWARGS),
                )
                result = runExperiment(config)
                row = _buildRow(result, "saliency", targetSparsity, seed, True)
                rows.append(row)
                print(
                    f"[regrowth           ] target={targetSparsity:.2f} seed={seed} "
                    f"valAcc={row['validationAccuracy']:.4f} "
                    f"achieved={row['achievedSparsity']:.4f} "
                    f"revived={row['revivedConnections']}",
                    flush=True,
                )

    return rows


def aggregateRows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate the per-seed rows into mean/std groups."""
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        key = (row["criterion"], row["targetSparsity"], row["withRegrowth"])
        groups.setdefault(key, []).append(row)

    aggregates: List[Dict[str, Any]] = []
    for (criterion, targetSparsity, withRegrowth), group in sorted(
        groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])
    ):
        accuracies = [float(row["validationAccuracy"]) for row in group]
        sparsities = [float(row["achievedSparsity"]) for row in group]
        latencies = [float(row["measuredLatencyRatio"]) for row in group]
        macReductions = [float(row["theoreticalMacReduction"]) for row in group]
        aggregates.append(
            {
                "criterion": criterion,
                "targetSparsity": targetSparsity,
                "withRegrowth": withRegrowth,
                "seedCount": len(group),
                "meanValidationAccuracy": statistics.mean(accuracies),
                "stdValidationAccuracy": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
                "minValidationAccuracy": min(accuracies),
                "maxValidationAccuracy": max(accuracies),
                "meanAchievedSparsity": statistics.mean(sparsities),
                "stdAchievedSparsity": statistics.stdev(sparsities) if len(sparsities) > 1 else 0.0,
                "meanTheoreticalMacReduction": statistics.mean(macReductions),
                "meanMeasuredLatencyRatio": statistics.mean(latencies),
                "stdMeasuredLatencyRatio": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            }
        )
    return aggregates


def main() -> int:
    """Run the suite and write the artefacts."""
    arguments = parseArguments()
    rows = runSuite(arguments)
    aggregates = aggregateRows(rows)

    prefix = arguments.output_prefix
    writeCsv(RESULTS_DIRECTORY / f"{prefix}_raw.csv", rows)
    writeCsv(RESULTS_DIRECTORY / f"{prefix}_summary.csv", aggregates)
    writeJson(
        RESULTS_DIRECTORY / f"{prefix}_summary.json",
        {
            "settings": {
                "seeds": arguments.seeds,
                "epochs": arguments.epochs,
                "batchSize": arguments.batch_size,
                "learningRate": arguments.learning_rate,
                "sparsities": arguments.sparsities,
                "criteria": arguments.criteria,
                "includeRegrowth": arguments.include_regrowth,
                "hiddenSizes": DEFAULT_HIDDEN_SIZES,
                "dataset": "spiral",
                "datasetKwargs": DEFAULT_DATASET_KWARGS,
            },
            "aggregates": aggregates,
        },
    )

    print("\n=== Summary (mean validation accuracy) ===")
    for entry in aggregates:
        print(
            f"{entry['criterion']:<18} target={entry['targetSparsity']:.2f} "
            f"regrowth={entry['withRegrowth']!s:<5} "
            f"acc={entry['meanValidationAccuracy']:.4f} "
            f"+/- {entry['stdValidationAccuracy']:.4f} "
            f"(n={entry['seedCount']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
