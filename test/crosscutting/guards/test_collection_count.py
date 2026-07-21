"""P14 — Collection-count guard.

Asserts the suite never silently shrinks: a bad conftest, a broken
import, or a misplaced file during a restructure can deselect hundreds
of tests without failing CI.  This guard pins a floor on the number of
collected tests, per top-level test package and in total.

When tests are intentionally removed, lower the affected floor in
EXPECTED_MINIMUMS with a note in the commit message.  When adding
tests, no update is needed (floors are minimums, not exact counts).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.guard

REPO_ROOT = Path(__file__).resolve().parents[3]

# Floors set at the 2026-07-21 micro-to-macro restructure (originally
# 2026-07-19 macro-to-micro). Small slack (~2-5%) so trivial refactors
# don't trip the guard. Keys are counted prefixes: the five l0-l4 levels
# plus each module mirror under l0_modules.
EXPECTED_MINIMUMS = {
    "test/l4_system": 145,
    "test/l3_subsystems": 36,
    "test/l2_integration": 85,
    "test/l0_modules": 1760,
    "test/crosscutting": 80,
    "test/l0_modules/core": 120,
    "test/l0_modules/physics": 780,
    "test/l0_modules/simulation": 64,
    "test/l0_modules/viz": 390,
    "test/l0_modules/capture": 82,
    "test/l0_modules/analysis": 310,
}
TOTAL_MINIMUM = 2160


@pytest.fixture(scope="module")
def per_dir_counts() -> dict[str, int]:
    """Collect counts for every top-level test package in one pytest run."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "test", "--collect-only", "-q",
         "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=300,
    )
    counts: dict[str, int] = {}
    total = 0
    for line in result.stdout.splitlines():
        # Test-id lines look like
        # "test/l0_modules/core/test_types.py::TestX::test_y"
        if not line.startswith("test/") or "::" not in line:
            continue
        total += 1
        parts = line.split("::", 1)[0].split("/")
        # Count the level prefix (test/<level>) and, for l0_modules, also
        # the module prefix (test/l0_modules/<module>).
        counts["/".join(parts[:2])] = counts.get("/".join(parts[:2]), 0) + 1
        if parts[1] == "l0_modules" and len(parts) > 3:
            key = "/".join(parts[:3])
            counts[key] = counts.get(key, 0) + 1
    counts["__total__"] = total
    return counts


def test_total_collection_floor(per_dir_counts):
    """The whole suite collects at least TOTAL_MINIMUM tests."""
    total = per_dir_counts["__total__"]
    assert total >= TOTAL_MINIMUM, (
        f"Only {total} tests collected (floor {TOTAL_MINIMUM}). "
        "A conftest/import failure or misplaced files may have silently "
        "dropped tests. If the removal was intentional, lower TOTAL_MINIMUM."
    )


@pytest.mark.parametrize("package", sorted(EXPECTED_MINIMUMS))
def test_package_collection_floor(per_dir_counts, package):
    """Each top-level test package keeps at least its expected floor."""
    count = per_dir_counts.get(package, 0)
    floor = EXPECTED_MINIMUMS[package]
    assert count >= floor, (
        f"{package} collected {count} tests (floor {floor}). "
        "If tests moved, update EXPECTED_MINIMUMS to match the new layout."
    )
