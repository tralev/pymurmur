"""Unit tests for analysis.presets — PRESETS dictionary and application."""

from copy import deepcopy

import pytest

from pymurmur.analysis.presets import PRESETS
from pymurmur.core.config import SimConfig


def test_all_presets_valid():
    """Every preset dict contains only valid SimConfig field names."""
    valid_fields = set(SimConfig().__dataclass_fields__.keys())
    for name, preset in PRESETS.items():
        unknown = set(preset.keys()) - valid_fields
        assert not unknown, f"Preset '{name}' has unknown fields: {unknown}"


def test_preset_apply_changes_mode():
    """Applying 'ball' preset sets mode to 'projection'."""
    cfg = SimConfig()
    cfg.mode = "spatial"  # start with something else
    for key, value in PRESETS["ball"].items():
        setattr(cfg, key, value)
    assert cfg.mode == "projection"


def test_preset_apply_changes_weights():
    """Applying 'acro' preset changes separation_weight and noise_scale."""
    cfg = SimConfig()
    assert cfg.separation_weight == 4.5  # default
    assert cfg.noise_scale == 0.0        # default

    for key, value in PRESETS["acro"].items():
        setattr(cfg, key, value)

    assert cfg.separation_weight == 2.0
    assert cfg.noise_scale == 1.5


def test_preset_does_not_mutate_default():
    """Applying a preset doesn't modify the PRESETS dict entries."""
    preset_copy = deepcopy(PRESETS)
    cfg = SimConfig()
    for key, value in PRESETS["ball"].items():
        setattr(cfg, key, value)
    assert PRESETS == preset_copy


def test_all_presets_run():
    """Every preset produces a working simulation (no crash in 10 steps)."""
    from pymurmur.simulation.engine import SimulationEngine

    for name, preset in PRESETS.items():
        cfg = SimConfig()
        cfg.num_boids = 10  # small for speed
        for key, value in preset.items():
            setattr(cfg, key, value)

        sim = SimulationEngine(cfg)
        try:
            sim.run_headless(steps=10)
        except Exception as e:
            pytest.fail(f"Preset '{name}' crashed: {e}")
