"""Unit tests for the NeighborAdaptiveSpeed extension — generalized
neighbor-count adaptive speed (isolated boids fly faster), publishing
flock.neighbor_adaptive_speed_mult for PhysicsFlock.integrate() to
compose into max_speed.
"""

from __future__ import annotations

import numpy as np

from pymurmur.core.config import SimConfig
from pymurmur.physics.extensions import ExtensionManager
from pymurmur.physics.extensions._base import StepContext
from pymurmur.physics.extensions.neighbor_adaptive_speed import NeighborAdaptiveSpeed
from pymurmur.physics.forces import compute_all_forces


class TestNeighborAdaptiveSpeedExtension:

    @staticmethod
    def _make_flock_and_ctx(n_boids=100, mode="spatial", rebuild_index=True, **cfg_kwargs):
        cfg = SimConfig()
        cfg.num_boids = n_boids
        cfg.mode = mode
        cfg.neighbor_adaptive_speed_enabled = True
        cfg.width = 1000.0
        cfg.height = 700.0
        cfg.depth = 400.0
        cfg.seed = 123
        for key, value in cfg_kwargs.items():
            setattr(cfg, key, value)

        from pymurmur.physics.flock import PhysicsFlock
        flock = PhysicsFlock(cfg)
        if rebuild_index and flock._index is not None:
            flock._index.rebuild(flock.positions, flock.active)

        ctx = StepContext(
            frame=0,
            dt=1.0 / 60.0,
            rng=flock.rng,
            center=flock.center,
            config=cfg,
        )
        return flock, ctx, cfg

    def test_publishes_multiplier(self):
        flock, ctx, cfg = self._make_flock_and_ctx()
        NeighborAdaptiveSpeed().apply(flock, ctx)
        assert flock.neighbor_adaptive_speed_mult is not None
        assert flock.neighbor_adaptive_speed_mult.shape == (cfg.num_boids,)
        assert flock.neighbor_adaptive_speed_mult.dtype == np.float32

    def test_multiplier_never_below_one(self):
        """Isolated boids speed up (mult > 1); no boid ever slows down
        below the baseline (mult >= 1) — matches angle.py's `if deficit
        > 0` gate (no penalty for having enough neighbors)."""
        flock, ctx, _ = self._make_flock_and_ctx()
        NeighborAdaptiveSpeed().apply(flock, ctx)
        assert np.all(flock.neighbor_adaptive_speed_mult >= 1.0 - 1e-5)

    def test_isolated_boid_gets_larger_multiplier(self):
        """A boid far from everyone else gets a bigger speed bonus than
        one embedded in a dense cluster."""
        cfg = SimConfig()
        cfg.num_boids = 20
        cfg.mode = "spatial"
        cfg.neighbor_adaptive_speed_enabled = True
        cfg.seed = 1

        from pymurmur.physics.flock import PhysicsFlock
        flock = PhysicsFlock(cfg)
        # Cluster boids 0..18 tightly together; boid 19 is far away, isolated.
        flock.positions[:19] = np.array([500.0, 350.0, 200.0], dtype=np.float32) \
            + np.random.default_rng(0).uniform(-2, 2, size=(19, 3)).astype(np.float32)
        flock.positions[19] = [50.0, 50.0, 50.0]
        flock._index.rebuild(flock.positions, flock.active)

        ctx = StepContext(frame=0, dt=1 / 60, rng=flock.rng, center=flock.center, config=cfg)
        NeighborAdaptiveSpeed().apply(flock, ctx)

        clustered_mult = flock.neighbor_adaptive_speed_mult[0]
        isolated_mult = flock.neighbor_adaptive_speed_mult[19]
        assert isolated_mult > clustered_mult

    def test_determinism(self):
        flock1, ctx1, _ = self._make_flock_and_ctx()
        flock2, ctx2, _ = self._make_flock_and_ctx()
        NeighborAdaptiveSpeed().apply(flock1, ctx1)
        NeighborAdaptiveSpeed().apply(flock2, ctx2)
        np.testing.assert_array_equal(
            flock1.neighbor_adaptive_speed_mult, flock2.neighbor_adaptive_speed_mult,
        )

    def test_no_index_ready_is_noop(self):
        """Modes that never rebuild the spatial index (field/influencer/
        marl) get multiplier 1.0 everywhere, not a crash."""
        flock, ctx, _ = self._make_flock_and_ctx(mode="field", rebuild_index=False)
        NeighborAdaptiveSpeed().apply(flock, ctx)
        np.testing.assert_array_equal(flock.neighbor_adaptive_speed_mult, 1.0)

    def test_teardown_on_disable(self):
        flock, ctx, cfg = self._make_flock_and_ctx()
        mgr = ExtensionManager(cfg)
        mgr.pre_step(flock, ctx)
        assert flock.neighbor_adaptive_speed_mult is not None

        cfg.neighbor_adaptive_speed_enabled = False
        mgr.pre_step(flock, ctx)
        assert flock.neighbor_adaptive_speed_mult is None

    def test_end_to_end_smoke_spatial(self):
        self._run_smoke(mode="spatial")

    def test_end_to_end_smoke_vicsek(self):
        self._run_smoke(mode="vicsek")

    @staticmethod
    def _run_smoke(mode):
        flock, ctx, cfg = TestNeighborAdaptiveSpeedExtension._make_flock_and_ctx(
            mode=mode, rebuild_index=False,
        )
        mgr = ExtensionManager(cfg)
        for frame in range(10):
            ctx.frame = frame
            if flock._index is not None:
                flock._index.rebuild(flock.positions, flock.active)
            mgr.pre_step(flock, ctx)
            compute_all_forces(flock, cfg)
            flock.integrate(cfg, ctx.dt, speed_mode="band")
        assert np.all(np.isfinite(flock.velocities))
        assert np.all(np.isfinite(flock.positions))
