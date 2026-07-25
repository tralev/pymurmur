"""P14.1 — Architecture structural scans: no module-level np.random,
no cKDTree in physics/forces, viz must not import simulation, no
print() in package sources.

Split out of test_architecture.py (file-size split). Imports
_collect_import_edges/_is_known_violation from test_architecture_edges.py
for the viz-no-simulation-import scan; the other scans are self-contained.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from test.crosscutting.guards.architecture.test_architecture_edges import (
    _collect_import_edges,
    _is_known_violation,
)

pytestmark = pytest.mark.guard


def test_no_module_level_numpy_random():
    """No .py file under pymurmur/ has top-level np.random.* calls."""
    pattern = re.compile(r"^\s{0,3}np\.random\.")
    failures: list[str] = []

    for py_file in sorted(Path("pymurmur").rglob("*.py")):
        lines = py_file.read_text().split("\n")
        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("import "):
                continue
            if stripped.startswith("from "):
                continue
            if pattern.search(line):
                failures.append(f"  {py_file}:{lineno}: {line.strip()}")

    if failures:
        msg = (
            f"\n❌ {len(failures)} top-level np.random.* call(s) found:\n"
            + "\n".join(failures)
            + "\n\nAll randomness must go through flock.rng. "
            + "Top-level np.random.* breaks determinism (P0.4).\n"
        )
        raise AssertionError(msg)

    print("✓ No top-level np.random.* calls found")


def test_no_cKDTree_in_forces():
    """No cKDTree construction in physics/forces/."""
    known = {
        "pymurmur/physics/forces/spatial.py",
        "pymurmur/physics/forces/spatial_helpers.py",  # extracted from spatial.py
        "pymurmur/physics/forces/vicsek.py",
        "pymurmur/physics/forces/angle.py",
    }
    failures: list[str] = []

    for py_file in Path("pymurmur/physics/forces").rglob("*.py"):
        rel = str(py_file)
        text = py_file.read_text()
        if "cKDTree(" in text or "KDTree(" in text:
            if rel in known:
                print(f"  ⚠️  Known: {rel} builds private cKDTree (P2.3)")
                continue
            failures.append(f"  {py_file}: contains cKDTree/KDTree construction")

    if failures:
        msg = (
            "\n❌ cKDTree construction found in forces/:\n"
            + "\n".join(failures)
            + "\n\nSpatial index construction belongs in physics/flock.py "
            + "(P0.2). Forces modules must use flock.index.\n"
        )
        raise AssertionError(msg)

    print("✓ No cKDTree constructions in physics/forces/")


def test_viz_no_simulation_import():
    """viz/ modules must not import from simulation/ at runtime."""
    edges = _collect_import_edges()

    failures: list[str] = []
    for source, target, lineno, in_tc in edges:
        if not source.startswith("pymurmur.viz."):
            continue
        if not target.startswith("pymurmur.simulation"):
            continue
        if in_tc:
            continue
        if _is_known_violation(source, target):
            continue
        failures.append(f"  {source}:{lineno} → {target}")

    if failures:
        msg = (
            f"\n❌ {len(failures)} viz → simulation runtime import(s) found:\n"
            + "\n".join(failures)
            + "\n\nviz/ modules must not import simulation/ modules at runtime.\n"
        )
        raise AssertionError(msg)

    print("✓ No runtime viz → simulation imports found")


def _scan_print_violations(root: Path, exempt: set[str]) -> list[str]:
    """G3: Scan *root* for `print(` calls, skipping comments/blanks and
    any file whose path (relative to cwd) is in *exempt*.  Extracted
    from the test body so the scan logic itself is unit-testable
    against a synthetic tree, not just the real `pymurmur/` package."""
    pattern = re.compile(r"print\(")
    failures: list[str] = []

    for py_file in sorted(root.rglob("*.py")):
        rel = str(py_file)
        if rel in exempt:
            continue
        text = py_file.read_text()
        for lineno, line in enumerate(text.split("\n"), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            if pattern.search(stripped):
                failures.append(f"  {rel}:{lineno}: {stripped}")

    return failures


def test_no_print_in_package_sources():
    """G3: No `print(` calls in pymurmur/ package sources (except
    __main__.py, logging, and cli-helper modules which legitimately
    use print for CLI output)."""
    # Files that are legitimately allowed to use print()
    exempt = {
        "pymurmur/__main__.py",
        "pymurmur/core/logging.py",
    }
    failures = _scan_print_violations(Path("pymurmur"), exempt)

    if failures:
        msg = (
            f"\n❌ {len(failures)} print() call(s) in package sources:\n"
            + "\n".join(failures)
            + "\n\nprint() should be replaced with proper logging "
            + "(core/logging.py) or be exempted in the test.\n"
        )
        raise AssertionError(msg)

    print("✓ No unexpected print() calls in package sources")


def test_print_scan_catches_real_violation(tmp_path):
    """G3: The scan mechanism itself catches a real `print(` call in a
    synthetic file — proving the check isn't vacuous (i.e. it wouldn't
    just as happily pass if `pymurmur/` genuinely had one)."""
    (tmp_path / "clean.py").write_text("x = 1\n# print('commented out, ignored')\n")
    (tmp_path / "dirty.py").write_text("def f():\n    print('oops')\n")

    failures = _scan_print_violations(tmp_path, exempt=set())

    assert len(failures) == 1, f"Expected exactly 1 violation, got: {failures}"
    assert "dirty.py" in failures[0] and "print('oops')" in failures[0]


def test_print_scan_respects_exempt_set(tmp_path):
    """G3: A file explicitly listed in the exempt set is not flagged,
    matching how __main__.py/core/logging.py are exempted for real."""
    dirty = tmp_path / "dirty.py"
    dirty.write_text("print('cli output')\n")

    failures = _scan_print_violations(tmp_path, exempt={str(dirty)})
    assert failures == []


# ── File-size ceiling ──────────────────────────────────────────────

