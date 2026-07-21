"""Unit tests for analysis.presets — PRESETS dictionary and application."""

from copy import deepcopy

import pytest

from pymurmur.analysis.presets import LETTER_PRESETS, PRESETS, apply_preset
from pymurmur.core.config import _ALL_FIELD_NAMES, _NESTED_ONLY, SimConfig


def test_all_presets_valid():
    """Every preset dict contains only valid SimConfig field names (I7.1).

    Nested-only fields (retired shims, e.g. phi_p) are valid preset keys —
    the preset appliers route them to the sub-config explicitly.
    """
    valid_fields = _ALL_FIELD_NAMES | set(_NESTED_ONLY.keys())
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


# ── P10.1: apply_preset function + L etter presets ──

class TestApplyPreset:
    """P10.1: apply_preset(config, key) — extracted, testable function."""

    def test_apply_known_key_returns_label(self):
        """apply_preset with known key returns the preset label."""
        cfg = SimConfig()
        label = apply_preset(cfg, "a")
        assert label == "3D Pearce Default"
        assert cfg.mode == "projection"

    def test_apply_unknown_key_returns_none(self):
        """apply_preset with unknown key returns None, config unchanged."""
        cfg = SimConfig()
        old_mode = cfg.mode
        label = apply_preset(cfg, "z")
        assert label is None
        assert cfg.mode == old_mode

    def test_apply_preset_applies_all_fields(self):
        """apply_preset applies mode, phi_p, phi_a, sigma."""
        cfg = SimConfig()
        apply_preset(cfg, "e")
        assert cfg.mode == "projection"
        assert cfg.projection.phi_p == pytest.approx(0.10)
        assert cfg.phi_a == pytest.approx(0.75)
        assert cfg.sigma == 6

    def test_apply_spatial_preset(self):
        """apply_preset with spatial key (d) applies spatial weights."""
        cfg = SimConfig()
        apply_preset(cfg, "d")
        assert cfg.mode == "spatial"
        assert cfg.separation_weight == pytest.approx(0.25)
        assert cfg.alignment_weight == pytest.approx(0.55)
        assert cfg.influence_count == 8

    def test_all_letter_presets_exist(self):
        """P10.1: All 8 letter keys are registered in LETTER_PRESETS."""
        expected = {"a", "b", "c", "d", "e", "f", "h", "w"}
        actual = set(LETTER_PRESETS.keys())
        assert actual == expected

    def test_letter_presets_match_spec(self):
        """P10.1: Letter presets match S5.1 spec values."""
        # a: 3D Pearce Default — 0.04/0.80/6/proj
        _, _, params_a = LETTER_PRESETS["a"]
        assert params_a["phi_p"] == pytest.approx(0.04)
        assert params_a["phi_a"] == pytest.approx(0.80)
        assert params_a["sigma"] == 6

        # b: Ball of Birds — 0.18/0.70/7/proj
        _, _, params_b = LETTER_PRESETS["b"]
        assert params_b["phi_p"] == pytest.approx(0.18)
        assert params_b["phi_a"] == pytest.approx(0.70)
        assert params_b["sigma"] == 7

        # w: Spiral Vortex — 0.08/0.82/10/spatial
        _, _, params_w = LETTER_PRESETS["w"]
        assert params_w["separation_weight"] == pytest.approx(0.08)
        assert params_w["alignment_weight"] == pytest.approx(0.82)
        assert params_w["cohesion_weight"] == pytest.approx(1.0)
        assert params_w["influence_count"] == 10

    def test_all_letter_entries_have_valid_structure(self):
        """P10.1: Every LETTER_PRESETS entry is a 3-tuple (label, desc, dict)."""
        for key, entry in LETTER_PRESETS.items():
            assert len(entry) == 3, (
                f"Letter preset '{key}' should be (label, desc, params), got {len(entry)} items"
            )
            label, desc, params = entry
            assert isinstance(label, str), f"'{key}': label must be str"
            assert isinstance(desc, str), f"'{key}': description must be str"
            assert isinstance(params, dict), f"'{key}': params must be dict"
            assert len(params) >= 3, f"'{key}': params should have at least 3 fields"

    def test_numbered_presets_match_input_control_keys(self):
        """P10.1: PRESETS dict keys match the ordered list used by keys 1-9."""
        expected_order = [
            "ball", "storm_cloud", "stream", "column",
            "acro", "spiral_vortex", "void",
        ]
        actual = list(PRESETS.keys())
        assert actual == expected_order, (
            "PRESETS keys order must match input_control index order"
        )
