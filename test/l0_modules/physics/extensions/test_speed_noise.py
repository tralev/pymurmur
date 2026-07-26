"""Unit tests for the SpeedNoise extension — noise-modulated per-boid
speed cap, publishing flock.speed_noise_mult for PhysicsFlock.integrate()
to compose into the max_speed array consumed by boid.integrate().
"""

from __future__ import annotations

import numpy as np

from pymurmur.core.config import SimConfig
from pymurmur.physics.extensions import ExtensionManager
from pymurmur.physics.extensions._base import StepContext
from pymurmur.physics.extensions.speed_noise import SpeedNoise
from pymurmur.physics.forces import compute_all_forces


class TestSpeedNoiseExtension:

    @staticmethod
    def _make_flock_and_ctx(n_boids=200, mode="field", **cfg_kwargs):
        cfg = SimConfig()
        cfg.num_boids = n_boids
        cfg.mode = mode
        cfg.speed_noise_enabled = True
        cfg.width = 1000.0
        cfg.height = 700.0
        cfg.depth = 400.0
        cfg.seed = 123
        for key, value in cfg_kwargs.items():
            setattr(cfg, key, value)

        from pymurmur.physics.flock import PhysicsFlock
        flock = PhysicsFlock(cfg)

        ctx = StepContext(
            frame=0,
            dt=1.0 / 60.0,
            rng=flock.rng,
            center=flock.center,
            config=cfg,
        )
        return flock, ctx, cfg

    def test_publishes_speed_noise_mult(self):
        flock, ctx, cfg = self._make_flock_and_ctx()
        SpeedNoise().apply(flock, ctx)
        assert flock.speed_noise_mult is not None
        assert flock.speed_noise_mult.shape == (cfg.num_boids,)
        assert flock.speed_noise_mult.dtype == np.float32

    def test_values_within_configured_range(self):
        flock, ctx, cfg = self._make_flock_and_ctx()
        SpeedNoise().apply(flock, ctx)
        assert flock.speed_noise_mult.min() >= cfg.speed_noise_min_mult - 1e-5
        assert flock.speed_noise_mult.max() <= cfg.speed_noise_max_mult + 1e-5

    def test_determinism(self):
        flock1, ctx1, _ = self._make_flock_and_ctx()
        flock2, ctx2, _ = self._make_flock_and_ctx()
        SpeedNoise().apply(flock1, ctx1)
        SpeedNoise().apply(flock2, ctx2)
        np.testing.assert_array_equal(
            flock1.speed_noise_mult, flock2.speed_noise_mult,
        )

    def test_non_degenerate_across_boids(self):
        flock, ctx, _ = self._make_flock_and_ctx()
        SpeedNoise().apply(flock, ctx)
        assert flock.speed_noise_mult.std() > 0.01

    def test_static_field_is_frame_invariant(self):
        """speed_noise_time_scale=0.0 (default): positions unchanged
        across frames -> multiplier unchanged."""
        flock, ctx, _ = self._make_flock_and_ctx()
        SpeedNoise().apply(flock, ctx)
        m0 = flock.speed_noise_mult.copy()
        ctx.frame = 100
        SpeedNoise().apply(flock, ctx)
        np.testing.assert_array_equal(m0, flock.speed_noise_mult)

    def test_temporal_drift_changes_output(self):
        flock, ctx, _ = self._make_flock_and_ctx(speed_noise_time_scale=5.0)
        SpeedNoise().apply(flock, ctx)
        m0 = flock.speed_noise_mult.copy()
        ctx.frame = 100
        SpeedNoise().apply(flock, ctx)
        assert not np.allclose(m0, flock.speed_noise_mult)

    def test_teardown_on_disable(self):
        """Disabling mid-run via ExtensionManager must reset
        flock.speed_noise_mult to None, not leave it stale."""
        flock, ctx, cfg = self._make_flock_and_ctx()
        mgr = ExtensionManager(cfg)
        mgr.pre_step(flock, ctx)
        assert flock.speed_noise_mult is not None

        cfg.speed_noise_enabled = False
        mgr.pre_step(flock, ctx)
        assert flock.speed_noise_mult is None

    def test_end_to_end_smoke_vicsek(self):
        self._run_smoke(mode="vicsek")

    def test_end_to_end_smoke_field(self):
        self._run_smoke(mode="field")

    @staticmethod
    def _run_smoke(mode):
        """No crash across ~10 steps, and speed variance is higher than
        an otherwise-identical disabled baseline."""
        flock, ctx, cfg = TestSpeedNoiseExtension._make_flock_and_ctx(mode=mode)
        mgr = ExtensionManager(cfg)
        for frame in range(10):
            ctx.frame = frame
            mgr.pre_step(flock, ctx)
            compute_all_forces(flock, cfg)
            flock.integrate(cfg, ctx.dt, speed_mode="band")
        speeds_enabled = np.linalg.norm(flock.velocities[flock.active], axis=1)

        flock2, ctx2, cfg2 = TestSpeedNoiseExtension._make_flock_and_ctx(
            mode=mode, speed_noise_enabled=False,
        )
        mgr2 = ExtensionManager(cfg2)
        for frame in range(10):
            ctx2.frame = frame
            mgr2.pre_step(flock2, ctx2)
            compute_all_forces(flock2, cfg2)
            flock2.integrate(cfg2, ctx2.dt, speed_mode="band")
        speeds_disabled = np.linalg.norm(flock2.velocities[flock2.active], axis=1)

        assert np.all(np.isfinite(speeds_enabled))
        assert speeds_enabled.std() > speeds_disabled.std()
