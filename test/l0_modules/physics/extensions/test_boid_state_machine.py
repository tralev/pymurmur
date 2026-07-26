"""Unit tests for the BoidStateMachine extension — generic, threshold-
driven per-boid state (normal/isolated/crowded/threatened), each
mapping to a speed-cap multiplier composed the same way as the other
multiplier extensions this session.
"""

from __future__ import annotations

import numpy as np

from pymurmur.core.config import SimConfig
from pymurmur.physics.extensions import ExtensionManager
from pymurmur.physics.extensions._base import StepContext
from pymurmur.physics.extensions.boid_state_machine import (
    STATE_CROWDED,
    STATE_ISOLATED,
    STATE_NORMAL,
    STATE_THREATENED,
    BoidStateMachine,
)


class TestBoidStateMachineExtension:

    @staticmethod
    def _make_flock_and_ctx(n_boids=30, mode="spatial", rebuild_index=True, **cfg_kwargs):
        cfg = SimConfig()
        cfg.num_boids = n_boids
        cfg.mode = mode
        cfg.boid_state_machine_enabled = True
        cfg.width = 1000.0
        cfg.height = 700.0
        cfg.depth = 400.0
        cfg.seed = 7
        for key, value in cfg_kwargs.items():
            setattr(cfg, key, value)

        from pymurmur.physics.flock import PhysicsFlock
        flock = PhysicsFlock(cfg)
        if rebuild_index and flock._index is not None:
            flock._index.rebuild(flock.positions, flock.active)

        ctx = StepContext(
            frame=0, dt=1.0 / 60.0, rng=flock.rng, center=flock.center, config=cfg,
        )
        return flock, ctx, cfg

    def test_publishes_state_and_multiplier(self):
        flock, ctx, cfg = self._make_flock_and_ctx()
        BoidStateMachine().apply(flock, ctx)
        assert flock.boid_state.shape == (cfg.num_boids,)
        assert flock.boid_state.dtype == np.int8
        assert flock.boid_state_speed_mult.shape == (cfg.num_boids,)

    def test_isolated_boid_gets_isolated_state(self):
        cfg = SimConfig()
        cfg.num_boids = 20
        cfg.mode = "spatial"
        cfg.boid_state_machine_enabled = True
        cfg.boid_state_isolated_neighbor_threshold = 2.0
        cfg.boid_state_isolated_speed_mult = 1.5
        cfg.seed = 1

        from pymurmur.physics.flock import PhysicsFlock
        flock = PhysicsFlock(cfg)
        # Cluster 0..18 tightly; boid 19 is far away, isolated.
        flock.positions[:19] = np.array([500.0, 350.0, 200.0], dtype=np.float32) \
            + np.random.default_rng(0).uniform(-2, 2, size=(19, 3)).astype(np.float32)
        flock.positions[19] = [50.0, 50.0, 50.0]
        flock._index.rebuild(flock.positions, flock.active)

        ctx = StepContext(frame=0, dt=1 / 60, rng=flock.rng, center=flock.center, config=cfg)
        BoidStateMachine().apply(flock, ctx)

        assert flock.boid_state[19] == STATE_ISOLATED
        assert flock.boid_state_speed_mult[19] == 1.5

    def test_crowded_boid_gets_crowded_state(self):
        cfg = SimConfig()
        cfg.num_boids = 20
        cfg.mode = "spatial"
        cfg.boid_state_machine_enabled = True
        cfg.boid_state_crowded_neighbor_threshold = 5.0
        cfg.boid_state_crowded_speed_mult = 0.6
        cfg.boid_state_neighbor_radius = 1000.0  # generous, so all count as neighbors
        cfg.seed = 2

        from pymurmur.physics.flock import PhysicsFlock
        flock = PhysicsFlock(cfg)
        # Pack everyone tightly -> everyone has many neighbors.
        flock.positions[:] = np.array([500.0, 350.0, 200.0], dtype=np.float32) \
            + np.random.default_rng(0).uniform(-2, 2, size=(20, 3)).astype(np.float32)
        flock._index.rebuild(flock.positions, flock.active)

        ctx = StepContext(frame=0, dt=1 / 60, rng=flock.rng, center=flock.center, config=cfg)
        BoidStateMachine().apply(flock, ctx)

        assert (flock.boid_state == STATE_CROWDED).all()
        np.testing.assert_allclose(flock.boid_state_speed_mult, 0.6)

    def test_threatened_takes_priority_over_isolated(self):
        """A boid that's both isolated AND near a threat must end up
        STATE_THREATENED, not STATE_ISOLATED (priority order)."""
        flock, ctx, cfg = self._make_flock_and_ctx(n_boids=5)
        ctx.threat_prox = np.array([0.9, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        BoidStateMachine().apply(flock, ctx)
        assert flock.boid_state[0] == STATE_THREATENED

    def test_no_threat_prox_never_threatened(self):
        flock, ctx, cfg = self._make_flock_and_ctx()
        assert ctx.threat_prox is None
        BoidStateMachine().apply(flock, ctx)
        assert not np.any(flock.boid_state == STATE_THREATENED)

    def test_normal_state_multiplier_is_one(self):
        cfg = SimConfig()
        cfg.num_boids = 5
        cfg.mode = "spatial"
        cfg.boid_state_machine_enabled = True
        cfg.boid_state_isolated_neighbor_threshold = -1.0  # never isolated
        cfg.boid_state_crowded_neighbor_threshold = 1e9    # never crowded
        cfg.seed = 3

        from pymurmur.physics.flock import PhysicsFlock
        flock = PhysicsFlock(cfg)
        flock._index.rebuild(flock.positions, flock.active)
        ctx = StepContext(frame=0, dt=1 / 60, rng=flock.rng, center=flock.center, config=cfg)
        BoidStateMachine().apply(flock, ctx)

        assert (flock.boid_state == STATE_NORMAL).all()
        np.testing.assert_allclose(flock.boid_state_speed_mult, 1.0)

    def test_determinism(self):
        flock1, ctx1, _ = self._make_flock_and_ctx()
        flock2, ctx2, _ = self._make_flock_and_ctx()
        BoidStateMachine().apply(flock1, ctx1)
        BoidStateMachine().apply(flock2, ctx2)
        np.testing.assert_array_equal(flock1.boid_state, flock2.boid_state)
        np.testing.assert_array_equal(
            flock1.boid_state_speed_mult, flock2.boid_state_speed_mult,
        )

    def test_teardown_on_disable(self):
        flock, ctx, cfg = self._make_flock_and_ctx()
        mgr = ExtensionManager(cfg)
        mgr.pre_step(flock, ctx)
        assert flock.boid_state_speed_mult is not None
        assert np.any(flock.boid_state != 0)

        cfg.boid_state_machine_enabled = False
        mgr.pre_step(flock, ctx)
        assert flock.boid_state_speed_mult is None
        assert np.all(flock.boid_state == 0)

    def test_end_to_end_smoke_all_modes(self):
        for mode in ("spatial", "field", "projection", "vicsek", "angle", "influencer", "marl"):
            flock, ctx, cfg = self._make_flock_and_ctx(mode=mode, rebuild_index=False)
            cfg.predator_enabled = True
            mgr = ExtensionManager(cfg)

            from pymurmur.physics.forces import compute_all_forces
            for frame in range(5):
                ctx.frame = frame
                if flock._index is not None:
                    flock._index.rebuild(flock.positions, flock.active)
                mgr.pre_step(flock, ctx)
                compute_all_forces(flock, cfg)
                flock.integrate(cfg, ctx.dt, speed_mode="band")
            assert np.all(np.isfinite(flock.velocities)), mode
            assert np.all(np.isfinite(flock.positions)), mode
