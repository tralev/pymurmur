"""P0.16 — Validate evolved.yaml artifact.

Loads `output/evolved.yaml` (from a previous EvoFlock SSGA run) and
asserts every parameter is within its documented range. Prevents silent
corruption of the evolved artifact.

Marked @pytest.mark.guard so `pytest -m guard` selects it.
"""

import math
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.guard

EVOLVED_PATH = Path("output/evolved.yaml")

# (param_name, min_valid, max_valid)
PARAM_RANGES = [
    ("separation_weight", 0.0, 20.0),
    ("alignment_weight", 0.0, 10.0),
    ("cohesion_weight", 0.0, 10.0),
    ("noise_scale", 0.0, 10.0),
    ("max_force", 0.01, 10.0),
    ("phi_p", 0.0, 1.0),
    ("phi_a", 0.0, 5.0),
    ("steric", 0.0, 5.0),
    ("predictive_avoid_weight", 0.0, 200.0),
    ("static_avoid_weight", 0.0, 200.0),
]


class TestEvolvedYaml:
    """Validate the output/evolved.yaml artifact."""

    def test_evolved_yaml_exists(self):
        """output/evolved.yaml exists and is a regular file."""
        assert EVOLVED_PATH.exists(), f"{EVOLVED_PATH} not found"
        assert EVOLVED_PATH.is_file(), f"{EVOLVED_PATH} is not a file"

    def test_evolved_yaml_parses(self):
        """output/evolved.yaml is valid YAML with evolved_params section."""
        with open(EVOLVED_PATH) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), f"Expected dict, got {type(data).__name__}"
        assert "evolved_params" in data, (
            f"Missing 'evolved_params' key; got: {list(data.keys())}"
        )
        params = data["evolved_params"]
        assert isinstance(params, dict)
        assert len(params) >= len(PARAM_RANGES), (
            f"Expected >= {len(PARAM_RANGES)} evolved params, got {len(params)}"
        )

    @pytest.mark.parametrize("param, lo, hi", PARAM_RANGES)
    def test_evolved_param_in_range(self, param, lo, hi):
        """Every evolved parameter is within its valid range."""
        with open(EVOLVED_PATH) as f:
            params = yaml.safe_load(f)["evolved_params"]
        assert param in params, f"Missing parameter: {param}"
        val = float(params[param])
        assert lo <= val <= hi, (
            f"{param}={val} outside [{lo}, {hi}]"
        )

    def test_evolved_yaml_no_nan_or_inf(self):
        """No evolved parameter is NaN or infinity."""
        with open(EVOLVED_PATH) as f:
            params = yaml.safe_load(f)["evolved_params"]
        for key, val in params.items():
            assert math.isfinite(float(val)), f"{key} is non-finite: {val}"

    def test_evolved_yaml_has_expected_keys(self):
        """All 10 expected parameter keys are present in evolved_params."""
        with open(EVOLVED_PATH) as f:
            params = yaml.safe_load(f)["evolved_params"]
        expected = {p for p, _, _ in PARAM_RANGES}
        actual = set(params.keys())
        missing = expected - actual
        assert not missing, f"Missing keys: {missing}"
