"""File-size ceiling guard — no file under pymurmur/ or test/ exceeds
600 lines unless explicitly documented in KNOWN_OVERSIZED_FILES.

Split out of test_architecture.py (file-size split).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.guard

MAX_FILE_LINES = 600

# Files over MAX_FILE_LINES that are known and intentionally deferred
# (not yet split) -- update the reason when a file is added or removed.
KNOWN_OVERSIZED_FILES: dict[str, str] = {
    "pymurmur/viz/renderer.py": (
        "VAO-building and drawing methods were already extracted to "
        "renderer_vao.py/renderer_draw.py mixins (880 -> 634 lines), "
        "the largest reduction available without splitting Renderer3D's "
        "own __init__/frame-lifecycle GL-context setup -- a single "
        "cohesive block of tightly-coupled state (buffers, programs, "
        "FBOs) that isn't naturally divisible without fragmenting "
        "small, related pieces just to hit the line count. Verified "
        "with a real headless GL context + visualizer launch after "
        "the mixin split."
    ),
}


def test_no_file_exceeds_line_limit():
    """Keep pymurmur/ and test/ modular -- every file should be small
    enough to read and reason about in one sitting (~600 lines). New
    files that grow past this should be split (see git history for
    the metrics.py/config.py/flock.py/evoflock.py/spatial.py/field.py
    production splits, and the Phase 2b test-suite splits, for the
    established "pure extraction + re-export shim" / "extraction by
    existing class or section boundary" patterns), not added to
    KNOWN_OVERSIZED_FILES -- that list is for pre-existing debt being
    paid down deliberately, not a place to grandfather new debt.
    """
    violations: list[str] = []
    known: list[str] = []

    for root in (Path("pymurmur"), Path("test")):
        for py_file in sorted(root.rglob("*.py")):
            if "__pycache__" in str(py_file):
                continue
            n_lines = sum(1 for _ in py_file.open())
            if n_lines <= MAX_FILE_LINES:
                continue
            rel = str(py_file)
            if rel in KNOWN_OVERSIZED_FILES:
                known.append(f"  {rel}: {n_lines} lines -- {KNOWN_OVERSIZED_FILES[rel]}")
            else:
                violations.append(f"  {rel}: {n_lines} lines (limit {MAX_FILE_LINES})")

    if known:
        print(f"\n⚠️  {len(known)} known oversized file(s) (deferred):")
        for k in known:
            print(k)

    if violations:
        msg = (
            f"\n❌ {len(violations)} file(s) exceed the {MAX_FILE_LINES}-line "
            f"guideline:\n"
            + "\n".join(violations)
            + "\n\nSplit the file (pure extraction, see recent git history "
            + "for the pattern), or if it's a deliberate, temporary "
            + "exception, add it to KNOWN_OVERSIZED_FILES with a reason.\n"
        )
        raise AssertionError(msg)

    print(
        f"✓ All pymurmur/ and test/ files within {MAX_FILE_LINES} lines "
        f"({len(known)} known exception(s) deferred)"
    )
