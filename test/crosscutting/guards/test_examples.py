"""G2 — examples/ doc-drift guard: committed example metrics captures
stay in sync with the live FlockMetrics schema.

examples/ (unlike gitignored output/) holds curated, committed
metrics captures demonstrating what the simulator's output actually
looks like. If a new metric field is added to FlockMetrics without
regenerating examples/ (via scripts/generate_examples.py), the
examples silently stop being an accurate reference. This guard
catches that the same way this session's other doc-drift guards
catch arch.md/test.md drift: parse the committed artifact, compare
against live ground truth.
"""

import json
from pathlib import Path

import pytest

from pymurmur.analysis.metrics.flock_metrics import FlockMetrics

pytestmark = pytest.mark.guard

EXAMPLES_DIR = Path("examples")


def _example_json_files() -> list[Path]:
    return sorted(EXAMPLES_DIR.glob("metrics_*.json"))


def test_examples_directory_has_metrics_files():
    """examples/ has at least one committed metrics_*.json capture."""
    files = _example_json_files()
    assert files, f"No metrics_*.json files found in {EXAMPLES_DIR}"


@pytest.mark.parametrize("path", _example_json_files(), ids=lambda p: p.name)
def test_example_metrics_schema_matches_flock_metrics(path: Path):
    """Each example's first frame has exactly FlockMetrics's field set.

    Catches: a new metric field shipping without any example
    reflecting it, or an example going stale after a field is
    renamed/removed.
    """
    data = json.loads(path.read_text())
    assert "metrics" in data and data["metrics"], f"{path.name}: no metrics entries"

    first_frame = data["metrics"][0]
    expected_fields = {f.name for f in FlockMetrics.__dataclass_fields__.values()}
    actual_fields = set(first_frame.keys())

    missing = expected_fields - actual_fields
    extra = actual_fields - expected_fields
    assert not missing and not extra, (
        f"{path.name}: schema drift from FlockMetrics — "
        f"missing={sorted(missing)} extra={sorted(extra)}. "
        f"Regenerate with: python scripts/generate_examples.py"
    )


@pytest.mark.parametrize("path", _example_json_files(), ids=lambda p: p.name)
def test_example_metrics_has_metadata(path: Path):
    """Each example JSON has the expected metadata block (seed, mode,
    num_boids, frame_count) — the same shape Recorder.save_metrics_json
    always produces."""
    data = json.loads(path.read_text())
    metadata = data.get("metadata", {})
    for key in ("seed", "mode", "num_boids", "frame_count"):
        assert key in metadata, f"{path.name}: metadata missing '{key}'"
    assert metadata["frame_count"] == len(data["metrics"]), (
        f"{path.name}: metadata.frame_count={metadata['frame_count']} "
        f"but {len(data['metrics'])} metrics entries present"
    )
