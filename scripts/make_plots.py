"""Generate the plots for the AQUA submission.

The plots are:

* ``pareto_sparsity_accuracy.png`` -- the sparsity/accuracy Pareto curve
  for every criterion, with error bars over the seeds.
* ``criteria_comparison.png`` -- accuracy versus sparsity for the criteria
  side by side.
* ``learning_curves.png`` -- loss and accuracy against epoch for the dense
  baseline and for a 90% sparse run.
* ``cost_measurements.png`` -- theoretical multiply-accumulate reduction
  against measured wall-clock latency, per batch size.

Usage::

    python scripts/make_plots.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aqua.train.experiment import ExperimentConfig, runExperiment  # noqa: E402
from aqua.utils.serialization import readCsv  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIRECTORY = REPOSITORY_ROOT / "results" / "experiments"
PLOTS_DIRECTORY = REPOSITORY_ROOT / "results" / "plots"

CRITERION_LABELS = {
    "magnitude": "Magnitude (baseline)",
    "saliency": "Saliency |w x grad|",
    "signedTaylor": "Signed Taylor -w x grad",
    "gradientMagnitude": "Gradient magnitude |grad|",
}

COLOURS = {
    "magnitude": "#8c8c8c",
    "saliency": "#1f77b4",
    "signedTaylor": "#d62728",
    "gradientMagnitude": "#2ca02c",
}


def _readSummary(path: Path) -> List[Dict[str, str]]:
    """Read a summary CSV produced by ``run_experiments.py``."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'python scripts/run_experiments.py' first."
        )
    return readCsv(path)


def _groupByCriterion(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        if row.get("withRegrowth", "False") == "True":
            continue
        grouped.setdefault(row["criterion"], []).append(row)
    for entries in grouped.values():
        entries.sort(key=lambda row: float(row["targetSparsity"]))
    return grouped


def plotPareto(rows: List[Dict[str, str]]) -> Path:
    """Plot accuracy against achieved sparsity for every criterion."""
    grouped = _groupByCriterion(rows)
    figure, axis = plt.subplots(figsize=(8.0, 5.5))

    for criterion, entries in grouped.items():
        sparsities = [float(row["meanAchievedSparsity"]) for row in entries]
        accuracies = [float(row["meanValidationAccuracy"]) for row in entries]
        errors = [float(row["stdValidationAccuracy"]) for row in entries]
        axis.errorbar(
            sparsities,
            accuracies,
            yerr=errors,
            marker="o",
            markersize=6,
            capsize=4,
            linewidth=1.8,
            color=COLOURS.get(criterion, None),
            label=f"{CRITERION_LABELS.get(criterion, criterion)}",
        )

    axis.set_xlabel("Achieved sparsity (fraction of connections removed)")
    axis.set_ylabel("Validation accuracy")
    axis.set_title("Sparsity-accuracy Pareto curve\n(spiral dataset, mean +/- std over 5 seeds)")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="lower left", fontsize=9)
    axis.set_ylim(0.3, 1.0)
    figure.tight_layout()
    PLOTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    outputPath = PLOTS_DIRECTORY / "pareto_sparsity_accuracy.png"
    figure.savefig(outputPath, dpi=150)
    plt.close(figure)
    return outputPath


def plotCriteriaComparison(rows: List[Dict[str, str]]) -> Path:
    """Plot every criterion on its own axis for a direct comparison."""
    grouped = _groupByCriterion(rows)
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

    for criterion, entries in grouped.items():
        label = CRITERION_LABELS.get(criterion, criterion)
        colour = COLOURS.get(criterion, None)
        targets = [float(row["targetSparsity"]) for row in entries]
        accuracies = [float(row["meanValidationAccuracy"]) for row in entries]
        errors = [float(row["stdValidationAccuracy"]) for row in entries]

        axes[0].errorbar(
            targets, accuracies, yerr=errors, marker="s", capsize=4,
            linewidth=1.8, color=colour, label=label,
        )
        axes[1].errorbar(
            targets, accuracies, yerr=errors, marker="s", capsize=4,
            linewidth=1.8, color=colour, label=label,
        )

    axes[0].set_title("Accuracy vs target sparsity")
    axes[1].set_title("Accuracy vs target sparsity (zoomed to the interesting range)")
    for axis in axes:
        axis.set_xlabel("Target sparsity")
        axis.set_ylabel("Validation accuracy")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    axes[1].set_ylim(0.45, 1.0)
    figure.tight_layout()
    PLOTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    outputPath = PLOTS_DIRECTORY / "criteria_comparison.png"
    figure.savefig(outputPath, dpi=150)
    plt.close(figure)
    return outputPath


def plotLearningCurves() -> Path:
    """Plot the learning curves of the dense baseline and two 90% sparse runs."""
    datasetKwargs = {
        "sampleCount": 1200,
        "validationFraction": 0.25,
        "noiseStd": 0.20,
        "turns": 1.2,
    }
    runs = {
        "Dense (0% sparsity)": ExperimentConfig(
            epochs=60, targetSparsity=0.0, criterion="saliency", seed=0,
            hiddenSizes=[32, 32], datasetKwargs=dict(datasetKwargs),
        ),
        "Saliency pruned (90%)": ExperimentConfig(
            epochs=60, targetSparsity=0.9, criterion="saliency", seed=0,
            hiddenSizes=[32, 32], datasetKwargs=dict(datasetKwargs),
        ),
        "Magnitude pruned (90%)": ExperimentConfig(
            epochs=60, targetSparsity=0.9, criterion="magnitude", seed=0,
            hiddenSizes=[32, 32], datasetKwargs=dict(datasetKwargs),
        ),
    }

    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    for label, config in runs.items():
        result = runExperiment(config)
        epochs = [record["epoch"] for record in result.history]
        axes[0].plot(
            epochs, [record["trainLoss"] for record in result.history],
            label=f"{label} (train)", linewidth=1.6,
        )
        axes[0].plot(
            epochs, [record["validationLoss"] for record in result.history],
            linestyle="--", linewidth=1.4, label=f"{label} (validation)",
        )
        axes[1].plot(
            epochs, [record["validationAccuracy"] for record in result.history],
            label=label, linewidth=1.8,
        )

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Loss during training")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation accuracy")
    axes[1].set_title("Validation accuracy during training")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0.2, 1.0)

    figure.tight_layout()
    PLOTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    outputPath = PLOTS_DIRECTORY / "learning_curves.png"
    figure.savefig(outputPath, dpi=150)
    plt.close(figure)
    return outputPath


def plotCostMeasurements() -> Path:
    """Plot theoretical MAC reduction against measured latency change."""
    rows = _readSummary(EXPERIMENTS_DIRECTORY / "experiments_raw.csv")
    grouped = _groupByCriterion(rows)

    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

    for criterion, entries in grouped.items():
        targets = [float(row["targetSparsity"]) for row in entries]
        macReduction = [float(row["theoreticalMacReduction"]) for row in entries]
        label = CRITERION_LABELS.get(criterion, criterion)
        colour = COLOURS.get(criterion, None)
        axes[0].plot(targets, macReduction, marker="o", linewidth=1.8, color=colour, label=label)

    axes[0].plot([0, 1], [0, 1], linestyle=":", color="black", label="Ideal (ratio = sparsity)")
    axes[0].set_xlabel("Target sparsity")
    axes[0].set_ylabel("Multiply-accumulate reduction (theoretical)")
    axes[0].set_title("Theoretical work reduction\n(what the sparse path actually does)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    # The measured latency ratio for each of the three batch sizes, using the
    # saliency criterion (the primary one).
    saliencyEntries = grouped.get("saliency", [])
    if saliencyEntries:
        targets = [float(row["targetSparsity"]) for row in saliencyEntries]
        for column, marker, batchSize in (
            ("measuredLatencyRatioBatch1", "o", 1),
            ("measuredLatencyRatio", "s", 32),
            ("measuredLatencyRatioBatch256", "^", 256),
        ):
            ratios = [float(row[column]) for row in saliencyEntries]
            axes[1].plot(
                targets, ratios, marker=marker, linewidth=1.8,
                label=f"batch size {batchSize}",
            )

    axes[1].axhline(1.0, linestyle=":", color="black", label="No speedup (ratio = 1)")
    axes[1].set_xlabel("Target sparsity")
    axes[1].set_ylabel("Measured latency ratio (sparse / dense)")
    axes[1].set_title(
        "Measured wall-clock latency (saliency criterion)\n"
        "(lower is faster; > 1 means the sparse path is SLOWER)"
    )
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    figure.tight_layout()
    PLOTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    outputPath = PLOTS_DIRECTORY / "cost_measurements.png"
    figure.savefig(outputPath, dpi=150)
    plt.close(figure)
    return outputPath


def main() -> int:
    """Generate every plot."""
    parser = argparse.ArgumentParser(description="Generate the AQUA plots.")
    parser.add_argument(
        "--summary",
        type=str,
        default="experiments_summary.csv",
        help="Name of the summary CSV in results/experiments.",
    )
    arguments = parser.parse_args()

    PLOTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    rows = _readSummary(EXPERIMENTS_DIRECTORY / arguments.summary)

    generated = [
        plotPareto(rows),
        plotCriteriaComparison(rows),
        plotLearningCurves(),
    ]
    try:
        generated.append(plotCostMeasurements())
    except FileNotFoundError as error:
        print(f"Skipping the cost plot: {error}")

    for path in generated:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
