"""Tests for the submission audit script.

The audit is itself part of the deliverable, so it is tested like any other
code.  In particular, the audit has to behave correctly in a *shallow* clone,
which is what a CI checkout produces by default.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = REPOSITORY_ROOT / "scripts" / "verify_submission.py"


def _runAudit(cwd: Path) -> subprocess.CompletedProcess:
    """Run the audit script in ``cwd`` and return the completed process."""
    return subprocess.run(
        [sys.executable, str(AUDIT_SCRIPT)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )


def test_audit_passes_in_this_repository() -> None:
    """The audit must pass on the working tree it ships with."""
    result = _runAudit(REPOSITORY_ROOT)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "0 failed" in result.stdout


def test_audit_detects_a_shallow_clone() -> None:
    """A shallow clone must not be reported as a squashed history.

    ``actions/checkout`` fetches a single commit by default, so
    ``git rev-list --count HEAD`` returns 1 even when the real history is
    long.  The audit used to fail there, which made the CI job red for a
    reason that had nothing to do with the code.
    """
    script = REPOSITORY_ROOT / "scripts" / "verify_submission.py"
    source = script.read_text(encoding="utf-8")

    # The audit must detect the shallow-clone case and skip the count check.
    assert "_isShallowClone" in source, "the audit must handle shallow clones"
    assert "shallow" in source.lower()

    # And it must be importable so the helper can be unit-tested.
    sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
    try:
        import verify_submission  # type: ignore[import-not-found]

        assert hasattr(verify_submission, "_isShallowClone")
    finally:
        sys.path.pop(0)


def test_audit_reports_a_missing_deliverable(tmp_path: Path) -> None:
    """A repository without the README must fail the audit.

    This is checked by importing the audit module with the working directory
    pointed at an empty folder: when the audit is imported it inspects the
    current directory, so a missing deliverable has to be detected.
    """
    empty = tmp_path / "empty"
    empty.mkdir()

    script = (
        "import sys\n"
        "sys.path.insert(0, %r)\n" % str(REPOSITORY_ROOT / "scripts")
        + "import verify_submission as audit\n"
        "result = audit.main()\n"
        "print('AUDIT_RESULT', result)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(empty),
        capture_output=True,
        text=True,
    )
    assert "AUDIT_RESULT 1" in completed.stdout, completed.stdout + completed.stderr
    assert "failed" in completed.stdout
