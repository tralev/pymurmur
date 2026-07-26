"""Engine-level tests for projection_heading_inertia / vicsek_heading_inertia
— the §09/§11-style heading-blend terms wired directly into projection.py
and vicsek.py (not a plugin; two small opt-in per-mode steering
parameters, default 0.0).
"""

from __future__ import annotations

import numpy as np
import pytest

from pymurmur.simulation.engine import SimulationEngine

ALL_MODES = ["spatial", "field", "projection", "vicsek", "angle", "influencer", "marl"]


class TestHeadingInertiaByteIdenticalWhenDefault:
    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_projection_field_default_matches_baseline(self, default_config, mode):
        cfg_baseline = default_config
        cfg_baseline.mode = mode
        cfg_baseline.num_boids = 30
        cfg_baseline.seed = 5

        cfg_explicit = default_config.__class__()
        cfg_explicit.mode = mode
        cfg_explicit.num_boids = 30
        cfg_explicit.seed = 5
        cfg_explicit.projection_heading_inertia = 0.0
        cfg_explicit.vicsek_heading_inertia = 0.0

        eng_baseline = SimulationEngine(cfg_baseline)
        eng_explicit = SimulationEngine(cfg_explicit)
        for _ in range(10):
            eng_baseline.step()
            eng_explicit.step()

        np.testing.assert_array_equal(
            eng_baseline.flock.positions, eng_explicit.flock.positions,
        )
        np.testing.assert_array_equal(
            eng_baseline.flock.velocities, eng_explicit.flock.velocities,
        )


class TestHeadingInertiaCrossModeNonInterference:
    """Setting the new fields to nonzero must not affect the 5 modes
    that don't consult them at all."""

    @pytest.mark.parametrize(
        "mode", ["spatial", "field", "angle", "influencer", "marl"],
    )
    def test_unaffected_modes_ignore_heading_inertia(self, default_config, mode):
        cfg_baseline = default_config
        cfg_baseline.mode = mode
        cfg_baseline.num_boids = 30
        cfg_baseline.seed = 9

        cfg_with_fields_set = default_config.__class__()
        cfg_with_fields_set.mode = mode
        cfg_with_fields_set.num_boids = 30
        cfg_with_fields_set.seed = 9
        cfg_with_fields_set.projection_heading_inertia = 0.9
        cfg_with_fields_set.vicsek_heading_inertia = 0.9

        eng_baseline = SimulationEngine(cfg_baseline)
        eng_set = SimulationEngine(cfg_with_fields_set)
        for _ in range(10):
            eng_baseline.step()
            eng_set.step()

        np.testing.assert_array_equal(
            eng_baseline.flock.positions, eng_set.flock.positions,
        )
        np.testing.assert_array_equal(
            eng_baseline.flock.velocities, eng_set.flock.velocities,
        )


class TestHeadingInertiaEngineSmoke:
    @pytest.mark.parametrize("mode", ["projection", "vicsek"])
    @pytest.mark.parametrize("inertia", [0.0, 0.5, 1.0])
    def test_no_crash_across_inertia_values(self, default_config, mode, inertia):
        cfg = default_config
        cfg.mode = mode
        cfg.num_boids = 40
        cfg.seed = 2
        cfg.projection_heading_inertia = inertia
        cfg.vicsek_heading_inertia = inertia

        engine = SimulationEngine(cfg)
        for _ in range(15):
            engine.step()

        assert np.all(np.isfinite(engine.flock.positions))
        assert np.all(np.isfinite(engine.flock.velocities))
