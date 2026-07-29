"""P14.3 — Strictly-3D guard.

Fails if any .py file under pymurmur/physics/ uses a 2D spatial array
shape. Uses targeted regex patterns to avoid false positives from
innocent uses of the integer 2.

Ported from the inline Python heredoc that used to live directly in
`.github/workflows/guard-rails.yml`'s `guard-rail-3d` job — every other
P14 guard has a real pytest file; this one didn't, which also meant the
job couldn't run inside Docker like its siblings.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.guard

PHYSICS_DIR = Path("pymurmur/physics")

# Patterns that indicate a 2D spatial array (not false positives):
#   np.zeros((N, 2))     — explicit 2D init
#   np.empty((N, 2))     — explicit 2D init
#   .reshape(-1, 2)      — reshape to 2 columns
#   shape=(N, 2)         — shape annotation
#   positions[:, :2]     — slicing to first 2 columns only
#   (…, 2) in docstrings / comments
PATTERNS = [
    (r"np\.(?:zeros|ones|empty|full|random\.[a-z_]+)\(\s*\(\s*[^,)]+\s*,\s*2\s*\)",
     "numpy array init with (…, 2) shape"),
    (r"\.reshape\s*\(\s*[^,)]+\s*,\s*2\s*\)",
     "reshape to 2 columns (e.g. .reshape(N, 2) or .reshape(-1, 2))"),
    (r"shape\s*=\s*\(\s*[^,)]+\s*,\s*2\s*\)",
     "shape=(…, 2) annotation"),
    # Comma is mandatory here — that's what distinguishes a genuine
    # multi-axis "first 2 columns" slice (positions[:, :2]) from a
    # single-axis stride slice ([::2], no comma), which must NOT match.
    (r"(?<!shape)\[\s*:\s*,\s*:2\s*\]",
     "position slicing to first 2 columns ([:, :2])"),
    # Bare [:2] ("first 2 rows/elements"); excludes .shape[:2] (unpacking
    # a shape tuple's first 2 dimensions has nothing to do with spatial
    # dimensionality).
    (r"(?<!shape)\[:2\]",
     "slicing to first 2 elements ([:2])"),
    (r"\(N,\s*2\)",
     "doc/comment reference to (N,2) shape"),
]

DEPTH_VALIDATION_PHRASES = [
    "depth > 0", "depth <= 0", "depth < 0",
    "depth <= 0.0", "depth > 0.0",
    "domain.depth > 0", "domain.depth <= 0",
]


def _scan_for_2d_arrays() -> list[str]:
    """Scan every .py file under pymurmur/physics/ for 2D-spatial-array
    patterns.  Returns one formatted failure string per match."""
    failures: list[str] = []
    for py_file in sorted(PHYSICS_DIR.rglob("*.py")):
        text = py_file.read_text()
        lines = text.split("\n")
        for lineno, line in enumerate(lines, start=1):
            for pattern, description in PATTERNS:
                if re.search(pattern, line):
                    failures.append(
                        f"{py_file}:{lineno}: {description}\n    {line.strip()}"
                    )
    return failures


def test_no_2d_spatial_arrays_in_physics():
    """P14.3: No .py file under pymurmur/physics/ uses a 2D spatial
    array shape — pymurmur is strictly 3D throughout."""
    failures = _scan_for_2d_arrays()
    assert not failures, (
        f"STRICTLY-3D GUARD FAILED ({len(failures)} issue(s)):\n"
        + "\n".join(f"  • {f}" for f in failures)
    )


def test_2d_array_pattern_is_actually_detected(tmp_path):
    """The scan mechanism itself catches a real 2D-array pattern — not
    just "physics/ happens to have none today"."""
    scratch_physics = tmp_path / "physics"
    scratch_physics.mkdir()
    (scratch_physics / "bad.py").write_text("positions = np.zeros((N, 2))\n")

    failures: list[str] = []
    for py_file in sorted(scratch_physics.rglob("*.py")):
        text = py_file.read_text()
        for lineno, line in enumerate(text.split("\n"), start=1):
            for pattern, description in PATTERNS:
                if re.search(pattern, line):
                    failures.append(f"{py_file}:{lineno}: {description}")

    assert failures, "scratch 2D-array pattern must be detected by the scan"


def test_domain_config_validates_positive_depth():
    """P14.3: DomainConfig validates depth > 0 wherever it's mentioned,
    somewhere in the core/config*.py module family.

    File-size split: config.py's cross-field validation logic
    (including the depth check) now lives in config_validation.py, not
    config.py itself -- scan every core/config*.py file, not just the
    one named config.py, so this check survives future splits too.

    A second, later split moved core/config.py to the core/config/
    package (core/config/config_validation.py etc.) -- the original
    Path("pymurmur/core").glob("config*.py") predates that move and has
    matched zero files since (a bare directory named "config" doesn't
    satisfy a "*.py" glob), silently no-opping this guard via the
    "no files found" skip below rather than ever checking anything.
    """
    config_files = sorted(Path("pymurmur/core/config").glob("*.py"))
    if not config_files:
        pytest.skip("no core/config*.py files found")

    combined_text = "\n".join(f.read_text() for f in config_files)
    if "depth" not in combined_text:
        pytest.skip("no config*.py file mentions 'depth'")

    has_validation = any(
        phrase in combined_text for phrase in DEPTH_VALIDATION_PHRASES
    )
    assert has_validation, (
        "core/config*.py: depth field has no positivity validation"
    )
