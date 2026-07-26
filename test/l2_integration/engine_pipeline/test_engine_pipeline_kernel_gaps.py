"""Engine-level tests for the Round-C gap-analysis additions: separation
kernels "linear"/"nearest_only"/"bell_zone", cohesion kernel "bell_zone",
alignment kernels "fov_weighted"/"circular_mean_2d"/"bell_zone", and the
mode-agnostic velocity_damping friction term. All are opt-in — default
config must reproduce current behavior byte-for-byte.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymurmur.simulation.engine import SimulationEngine

ALL_MODES = ["spatial", "field", "projection", "vicsek", "angle", "influencer", "marl"]

NEW_SEPARATION_KERNELS = ["linear", "nearest_only", "bell_zone"]
NEW_ALIGNMENT_KERNELS = ["fov_weighted", "circular_mean_2d", "bell_zone"]


class TestByteIdenticalWhenDefault:
    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_new_fields_default_matches_baseline(self, default_config, mode):
        cfg_baseline = default_config
        cfg_baseline.mode = mode
        cfg_baseline.num_boids = 30
        cfg_baseline.seed = 5

        cfg_explicit = default_config.__class__()
        cfg_explicit.mode = mode
        cfg_explicit.num_boids = 30
        cfg_explicit.seed = 5
        cfg_explicit.alignment_kernel = "unweighted"
        cfg_explicit.kernel_zone_width = 10.0
        cfg_explicit.velocity_damping = 0.0

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


class TestSpatialModeSmoke:
    """New kernels only take effect in spatial mode (the only mode whose
    force functions consult separation_kernel/cohesion_kernel/alignment_kernel)."""

    @pytest.mark.parametrize("kernel", NEW_SEPARATION_KERNELS)
    def test_separation_kernel_no_crash(self, default_config, kernel):
        cfg = default_config
        cfg.mode = "spatial"
        cfg.num_boids = 40
        cfg.seed = 3
        cfg.separation_kernel = kernel
        cfg.separation_kernel_radius = 15.0
        cfg.kernel_zone_width = 8.0

        engine = SimulationEngine(cfg)
        for _ in range(15):
            engine.step()

        assert np.all(np.isfinite(engine.flock.positions))
        assert np.all(np.isfinite(engine.flock.velocities))

    def test_cohesion_bell_zone_no_crash(self, default_config):
        cfg = default_config
        cfg.mode = "spatial"
        cfg.num_boids = 40
        cfg.seed = 4
        cfg.cohesion_kernel = "bell_zone"
        cfg.separation_kernel_radius = 15.0
        cfg.kernel_zone_width = 8.0

        engine = SimulationEngine(cfg)
        for _ in range(15):
            engine.step()

        assert np.all(np.isfinite(engine.flock.positions))
        assert np.all(np.isfinite(engine.flock.velocities))

    @pytest.mark.parametrize("kernel", NEW_ALIGNMENT_KERNELS)
    def test_alignment_kernel_no_crash(self, default_config, kernel):
        cfg = default_config
        cfg.mode = "spatial"
        cfg.num_boids = 40
        cfg.seed = 7
        cfg.alignment_kernel = kernel

        engine = SimulationEngine(cfg)
        for _ in range(15):
            engine.step()

        assert np.all(np.isfinite(engine.flock.positions))
        assert np.all(np.isfinite(engine.flock.velocities))


class TestVelocityDampingCrossModeSmoke:
    """velocity_damping is mode-agnostic — threaded directly into every
    mode's shared boid.integrate() call, unlike field_inertia."""

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_no_crash_with_damping_enabled(self, default_config, mode):
        cfg = default_config
        cfg.mode = mode
        cfg.num_boids = 30
        cfg.seed = 11
        cfg.velocity_damping = 0.3

        engine = SimulationEngine(cfg)
        for _ in range(15):
            engine.step()

        assert np.all(np.isfinite(engine.flock.positions))
        assert np.all(np.isfinite(engine.flock.velocities))

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_damping_changes_trajectory(self, default_config, mode):
        cfg_baseline = default_config
        cfg_baseline.mode = mode
        cfg_baseline.num_boids = 30
        cfg_baseline.seed = 13

        cfg_damped = default_config.__class__()
        cfg_damped.mode = mode
        cfg_damped.num_boids = 30
        cfg_damped.seed = 13
        cfg_damped.velocity_damping = 0.5

        eng_baseline = SimulationEngine(cfg_baseline)
        eng_damped = SimulationEngine(cfg_damped)
        for _ in range(10):
            eng_baseline.step()
            eng_damped.step()

        assert not np.array_equal(
            eng_baseline.flock.velocities, eng_damped.flock.velocities,
        )
