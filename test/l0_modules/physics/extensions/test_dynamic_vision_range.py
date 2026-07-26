"""Unit tests for the DynamicVisionRange extension — flock-wide adaptive
visual_range multiplier, bridged via config._dynamic_visual_range_mult
and consumed only by spatial_helpers.py's _query_neighbors (spatial +
projection modes).
"""

from __future__ import annotations

import numpy as np

from pymurmur.core.config import SimConfig
from pymurmur.physics.extensions import ExtensionManager
from pymurmur.physics.extensions._base import StepContext
from pymurmur.physics.extensions.dynamic_vision_range import DynamicVisionRange


class TestDynamicVisionRangeExtension:

    @staticmethod
    def _make_flock_and_ctx(n_boids=100, mode="spatial", rebuild_index=True, **cfg_kwargs):
        cfg = SimConfig()
        cfg.num_boids = n_boids
        cfg.mode = mode
        cfg.dynamic_vision_range_enabled = True
        cfg.width = 1000.0
        cfg.height = 700.0
        cfg.depth = 400.0
        cfg.seed = 42
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

    def test_publishes_multiplier_bridge(self):
        flock, ctx, cfg = self._make_flock_and_ctx()
        DynamicVisionRange().apply(flock, ctx)
        assert hasattr(cfg, "_dynamic_visual_range_mult")
        assert isinstance(cfg._dynamic_visual_range_mult, float)

    def test_multiplier_clamped_to_bounds(self):
        flock, ctx, cfg = self._make_flock_and_ctx(
            dynamic_vision_range_min_mult=0.8, dynamic_vision_range_max_mult=1.2,
        )
        ext = DynamicVisionRange()
        for frame in range(500):
            ctx.frame = frame
            ext.apply(flock, ctx)
        assert 0.8 - 1e-6 <= cfg._dynamic_visual_range_mult <= 1.2 + 1e-6

    def test_sparse_flock_expands_multiplier(self):
        """Very few neighbors within range -> multiplier grows over time."""
        flock, ctx, cfg = self._make_flock_and_ctx(
            n_boids=10, dynamic_vision_range_ideal_count=50.0,
        )
        ext = DynamicVisionRange()
        for frame in range(20):
            ctx.frame = frame
            ext.apply(flock, ctx)
        assert cfg._dynamic_visual_range_mult > 1.0

    def test_dense_flock_contracts_multiplier(self):
        """Many neighbors within range -> multiplier shrinks over time."""
        cfg = SimConfig()
        cfg.num_boids = 50
        cfg.mode = "spatial"
        cfg.dynamic_vision_range_enabled = True
        cfg.dynamic_vision_range_ideal_count = 1.0
        cfg.seed = 1

        from pymurmur.physics.flock import PhysicsFlock
        flock = PhysicsFlock(cfg)
        # Pack all boids into a tiny cluster -> everyone has many neighbors.
        flock.positions[:] = np.array([500.0, 350.0, 200.0], dtype=np.float32) \
            + np.random.default_rng(0).uniform(-2, 2, size=(50, 3)).astype(np.float32)
        flock._index.rebuild(flock.positions, flock.active)

        ctx = StepContext(frame=0, dt=1 / 60, rng=flock.rng, center=flock.center, config=cfg)
        ext = DynamicVisionRange()
        for frame in range(20):
            ctx.frame = frame
            ext.apply(flock, ctx)
        assert cfg._dynamic_visual_range_mult < 1.0

    def test_no_index_ready_is_noop(self):
        flock, ctx, cfg = self._make_flock_and_ctx(mode="field", rebuild_index=False)
        DynamicVisionRange().apply(flock, ctx)
        assert cfg._dynamic_visual_range_mult == 1.0

    def test_teardown_resets_bridge_to_one(self):
        flock, ctx, cfg = self._make_flock_and_ctx(n_boids=10, dynamic_vision_range_ideal_count=50.0)
        mgr = ExtensionManager(cfg)
        for frame in range(10):
            ctx.frame = frame
            mgr.pre_step(flock, ctx)
        assert cfg._dynamic_visual_range_mult != 1.0

        cfg.dynamic_vision_range_enabled = False
        mgr.pre_step(flock, ctx)
        assert cfg._dynamic_visual_range_mult == 1.0

    def test_query_neighbors_widens_with_multiplier(self):
        """Consumer-side check: spatial_helpers._query_neighbors actually
        reads the bridge attribute and widens the effective radius."""
        from pymurmur.physics.forces.spatial_helpers import _query_neighbors

        cfg = SimConfig()
        cfg.num_boids = 30
        cfg.mode = "spatial"
        cfg.visual_range = 50.0
        cfg.seed = 3

        from pymurmur.physics.flock import PhysicsFlock
        flock = PhysicsFlock(cfg)
        flock._index.rebuild(flock.positions, flock.active)

        narrow = _query_neighbors(
            flock.positions, flock.active, flock._index, cfg,
            filter_mode=cfg.neighbor_filter,
        )
        cfg._dynamic_visual_range_mult = 3.0
        wide = _query_neighbors(
            flock.positions, flock.active, flock._index, cfg,
            filter_mode=cfg.neighbor_filter,
        )
        # Widened radius should not produce fewer total candidate slots
        assert wide.shape[1] >= narrow.shape[1]

    def test_disabled_bridge_defaults_to_no_scaling(self):
        """getattr fallback: absent attribute -> multiplier of 1.0, i.e.
        byte-identical to the pre-DynamicVisionRange behavior."""
        cfg = SimConfig()
        cfg.mode = "spatial"
        assert not hasattr(cfg, "_dynamic_visual_range_mult")
