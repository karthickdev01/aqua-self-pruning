"""Audit the submission against the challenge's requirements.

This script checks that every deliverable is present and internally
consistent.  It is a *submission* check, not a test of the code (that is
what ``pytest`` is for).

Usage::

    python scripts/verify_submission.py
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# The audit inspects the repository it is run against.  When it is executed
# as a script that is the repository the script lives in; when it is imported
# (for example by the test suite) the caller's current working directory is
# used, so the audit can be pointed at a different checkout.
def _resolveRepositoryRoot() -> Path:
    """Return the repository the audit should inspect."""
    if __name__ == "__main__":
        return Path(__file__).resolve().parents[1]
    return Path.cwd()


REPOSITORY_ROOT = _resolveRepositoryRoot()
RESULTS_DIRECTORY = REPOSITORY_ROOT / "results" / "experiments"
PLOTS_DIRECTORY = REPOSITORY_ROOT / "results" / "plots"

REQUIRED_FILES = [
    "README.md",
    "DESIGN.md",
    "LICENSE",
    "requirements.txt",
    ".github/workflows/ci.yml",
    "aqua/engine/tensor.py",
    "aqua/engine/ops.py",
    "aqua/nn/layers.py",
    "aqua/nn/mlp.py",
    "aqua/optim/adam.py",
    "aqua/prune/criteria.py",
    "aqua/prune/schedule.py",
    "aqua/prune/allocator.py",
    "aqua/prune/pruner.py",
    "aqua/prune/regrowth.py",
    "aqua/train/trainer.py",
    "aqua/train/experiment.py",
    "aqua/train/sparse.py",
    "aqua/train/cost.py",
    "aqua/data/datasets.py",
    "scripts/run_experiments.py",
    "scripts/make_plots.py",
    "scripts/train_baseline.py",
]

REQUIRED_PLOTS = [
    "pareto_sparsity_accuracy.png",
    "criteria_comparison.png",
    "learning_curves.png",
    "cost_measurements.png",
]

FORBIDDEN_IMPORTS = [
    "torch",
    "tensorflow",
    "jax",
    "keras",
    "transformers",
    "sklearn",
    "scipy",
]


class Audit:
    """Collects the checks and their outcomes."""

    def __init__(self) -> None:
        self.passed: List[str] = []
        self.failed: List[str] = []

    def check(self, condition: bool, message: str) -> bool:
        """Record the outcome of a single check."""
        if condition:
            self.passed.append(message)
        else:
            self.failed.append(message)
        return condition

    def report(self) -> int:
        """Print the report and return the process exit code."""
        for message in self.passed:
            print(f"  PASS  {message}")
        for message in self.failed:
            print(f"  FAIL  {message}")
        print()
        print(f"{len(self.passed)} passed, {len(self.failed)} failed")
        return 1 if self.failed else 0


def checkFiles(audit: Audit) -> None:
    """Check that every required file exists."""
    for relative in REQUIRED_FILES:
        path = REPOSITORY_ROOT / relative
        audit.check(path.exists(), f"file present: {relative}")


def checkPlots(audit: Audit) -> None:
    """Check that every required plot exists and is non-trivial."""
    for name in REQUIRED_PLOTS:
        path = PLOTS_DIRECTORY / name
        exists = path.exists() and path.stat().st_size > 10_000
        audit.check(exists, f"plot present and non-trivial: {name}")


def checkNoForbiddenImports(audit: Audit) -> None:
    """Check that no forbidden library is imported anywhere."""
    offenders: List[str] = []
    for path in REPOSITORY_ROOT.rglob("*.py"):
        if ".venv" in path.parts or ".git" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for name in FORBIDDEN_IMPORTS:
            if f"import {name}" in text or f"from {name}" in text:
                offenders.append(f"{path.relative_to(REPOSITORY_ROOT)} imports {name}")
    audit.check(
        not offenders,
        "no forbidden library is imported (torch/tf/jax/keras/sklearn/scipy)",
    )
    for offender in offenders:
        print(f"        -> {offender}")


def checkRequirements(audit: Audit) -> None:
    """Check that requirements.txt contains only the allowed dependencies."""
    requirementsPath = REPOSITORY_ROOT / "requirements.txt"
    if not audit.check(requirementsPath.exists(), "requirements.txt is present"):
        return
    text = requirementsPath.read_text(encoding="utf-8").lower()
    allowed = {"numpy", "matplotlib", "pytest"}
    listedLines = [
        line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    packages = {
        line.split(">=")[0].split("==")[0].strip() for line in listedLines
    }
    audit.check(
        packages <= allowed,
        f"requirements.txt only lists allowed packages: {sorted(packages)}",
    )
    listed = " ".join(listedLines)
    audit.check(
        "sklearn" not in listed and "scikit" not in listed,
        "requirements.txt does not list a dataset-loading library",
    )


def _findSummary(
    rows: List[Dict[str, str]], criterion: str, target: float, withRegrowth: bool
) -> Optional[Dict[str, str]]:
    """Find the summary row matching a criterion and target sparsity."""
    for row in rows:
        if row["criterion"] != criterion:
            continue
        if abs(float(row["targetSparsity"]) - target) > 1e-9:
            continue
        if (row.get("withRegrowth") == "True") != withRegrowth:
            continue
        return row
    return None


def checkResults(audit: Audit) -> None:
    """Check that the raw results are present and self-consistent."""
    rawPath = RESULTS_DIRECTORY / "experiments_raw.csv"
    summaryPath = RESULTS_DIRECTORY / "experiments_summary.csv"
    jsonPath = RESULTS_DIRECTORY / "experiments_summary.json"

    if not audit.check(rawPath.exists(), "raw experiment CSV present"):
        return
    if not audit.check(summaryPath.exists(), "summary CSV present"):
        return

    with open(rawPath, encoding="utf-8", newline="") as handle:
        rawRows = list(csv.DictReader(handle))
    with open(summaryPath, encoding="utf-8", newline="") as handle:
        summaryRows = list(csv.DictReader(handle))

    audit.check(len(rawRows) > 0, f"raw CSV has rows ({len(rawRows)})")
    audit.check(len(summaryRows) > 0, f"summary CSV has rows ({len(summaryRows)})")

    seedCounts = {int(row["seedCount"]) for row in summaryRows}
    audit.check(
        all(count >= 5 for count in seedCounts),
        f"every summary group has at least 5 seeds (found {sorted(seedCounts)})",
    )

    targets = {float(row["targetSparsity"]) for row in summaryRows}
    required = {0.0, 0.5, 0.75, 0.9, 0.95}
    audit.check(
        required <= targets,
        f"all required sparsity levels are swept (missing {sorted(required - targets)})",
    )

    criteria = {row["criterion"] for row in summaryRows}
    audit.check(
        {"magnitude", "saliency"} <= criteria,
        f"both magnitude and saliency criteria are compared (found {sorted(criteria)})",
    )

    mismatches = [
        row
        for row in summaryRows
        if abs(float(row["meanAchievedSparsity"]) - float(row["targetSparsity"])) > 0.01
    ]
    audit.check(not mismatches, "achieved sparsity matches the target within 1%")

    nonZeroLatency = [row for row in rawRows if float(row["denseLatencySeconds"]) > 0.0]
    audit.check(
        len(nonZeroLatency) == len(rawRows),
        f"every raw row carries a measured latency ({len(nonZeroLatency)}/{len(rawRows)})",
    )

    mismatched = [row for row in rawRows if row["sparseMatchesDense"] != "True"]
    audit.check(not mismatched, "sparse forward matched the dense forward in every run")

    if audit.check(jsonPath.exists(), "summary JSON present"):
        try:
            with open(jsonPath, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError) as error:
            audit.check(False, f"summary JSON is readable ({error})")
            payload = {}
        audit.check(
            "aggregates" in payload and len(payload["aggregates"]) > 0,
            "summary JSON contains the aggregates",
        )

    saliency90 = _findSummary(summaryRows, "saliency", 0.9, False)
    magnitude90 = _findSummary(summaryRows, "magnitude", 0.9, False)
    if saliency90 and magnitude90:
        gap = float(saliency90["meanValidationAccuracy"]) - float(magnitude90["meanValidationAccuracy"])
        audit.check(gap > 0.05, f"saliency beats magnitude at 90% sparsity by {gap:.4f}")
        saliencyStd = float(saliency90["stdValidationAccuracy"])
        magnitudeStd = float(magnitude90["stdValidationAccuracy"])
        audit.check(
            saliencyStd <= magnitudeStd,
            f"saliency has no larger variance at 90% ({saliencyStd:.4f} vs {magnitudeStd:.4f})",
        )


def checkReadmeNumbers(audit: Audit) -> None:
    """Check that the numbers quoted in the README match the summary CSV."""
    summaryPath = RESULTS_DIRECTORY / "experiments_summary.csv"
    readmePath = REPOSITORY_ROOT / "README.md"
    if not summaryPath.exists() or not readmePath.exists():
        return
    with open(summaryPath, encoding="utf-8", newline="") as handle:
        summaryRows = list(csv.DictReader(handle))
    readme = readmePath.read_text(encoding="utf-8")

    for criterion, target in (("saliency", 0.9), ("magnitude", 0.9), ("saliency", 0.95)):
        row = _findSummary(summaryRows, criterion, target, False)
        if row is None:
            continue
        value = float(row["meanValidationAccuracy"])
        audit.check(
            f"{value:.4f}" in readme,
            f"README quotes the measured {criterion} accuracy at {target:.0%} ({value:.4f})",
        )


def _isShallowClone() -> bool:
    """Whether this is a shallow clone (e.g. a CI checkout with --depth 1).

    A shallow clone contains only the tip commit, so the commit count is 1
    even when the real history is long.  The history check must not fail in
    that case -- it would report a problem that does not exist.
    """
    shallowFile = REPOSITORY_ROOT / ".git" / "shallow"
    return shallowFile.exists()


def checkGitHistory(audit: Audit) -> None:
    """Check that the commit history has not been squashed into one commit.

    A shallow clone (``actions/checkout`` uses ``--depth 1`` by default)
    only contains the tip commit, so the count is not meaningful there.  In
    that case the check is skipped rather than reporting a false failure.
    """
    try:
        output = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        count = int(output.stdout.strip())
    except Exception:  # pragma: no cover - git may be unavailable
        audit.check(False, "git history is readable")
        return

    if _isShallowClone():
        audit.check(True, f"git history not checked (shallow clone, {count} commit visible)")
        return

    audit.check(count >= 3, f"git history has multiple commits ({count})")


def main() -> int:
    """Run every audit check and report."""
    audit = Audit()
    print("AQUA submission audit\n")
    print("Files:")
    checkFiles(audit)
    print("\nPlots:")
    checkPlots(audit)
    print("\nDependencies:")
    checkNoForbiddenImports(audit)
    checkRequirements(audit)
    print("\nResults:")
    checkResults(audit)
    checkReadmeNumbers(audit)
    print("\nRepository:")
    checkGitHistory(audit)
    print()
    return audit.report()


if __name__ == "__main__":
    raise SystemExit(main())
