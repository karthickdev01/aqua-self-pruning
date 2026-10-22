"""Quick diagnostic sweep: compare pruning criteria across sparsity levels.

Usage::

    python scripts/sweep_criteria.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aqua.train.experiment import ExperimentConfig, runExperiment  # noqa: E402

DATASET_KWARGS = {
    "sampleCount": 1200,
    "validationFraction": 0.25,
    "noiseStd": 0.20,
    "turns": 1.2,
}


def main() -> int:
    """Run the sweep and print a summary table."""
    criteria = ["magnitude", "saliency", "signedTaylor", "gradientMagnitude"]
    sparsities = [0.0, 0.5, 0.75, 0.9, 0.95]
    seeds = [0, 1, 2, 3, 4]

    lines: list[str] = []
    for criterion in criteria:
        for sparsity in sparsities:
            accuracies = []
            for seed in seeds:
                config = ExperimentConfig(
                    epochs=60,
                    targetSparsity=sparsity,
                    seed=seed,
                    hiddenSizes=[32, 32],
                    criterion=criterion,
                    datasetKwargs=dict(DATASET_KWARGS),
                )
                result = runExperiment(config)
                accuracies.append(result.finalMetrics["validationAccuracy"])
            mean = statistics.mean(accuracies)
            stdev = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
            line = (
                f"{criterion:<18} sp={sparsity:.2f} mean={mean:.4f} sd={stdev:.4f} "
                f"vals={[round(value, 4) for value in accuracies]}"
            )
            lines.append(line)
            print(line, flush=True)

    outputPath = Path(__file__).resolve().parents[1] / "results" / "experiments" / "criterion_sweep.txt"
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    outputPath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
