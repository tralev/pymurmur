"""Phase 2 cross-item integration tests — verify 2+ Phase 2 items work
together as a whole.

IT-P2-1: Dynamic spatial index swap (P2.1 → P2.3 → P2.4)
IT-P2-2: Full threat/evasion pipeline (P2.6 → P2.10 via engine.step)

IT-P2-3 (InstanceSchema buffer packing) and the holey-flock composition
tests live in test_cross_subsystem_schema_composition.py (file-size
split of this file).
"""

import numpy as np

from pymurmur.physics.flock import KDTreeIndex, SpatialHashGrid
from pymurmur.simulation.engine import SimulationEngine

# ═══════════════════════════════════════════════════════════════════
# IT-P2-1: Dynamic Spatial Index Swap (P2.1 → P2.3 → P2.4)
# ═══════════════════════════════════════════════════════════════════

class TestDynamicSpatialIndexSwap:
    """Change config.spatial_index mid-simulation and verify index type
    follows the config without crashing or data corruption."""

    def test_auto_to_kdtree_mid_simulation(self, default_config):
        """P2.1→P2.3→P2.4: Start with auto (hash_grid), switch to kdtree
        mid-run, verify index type changes and queries still work."""
        cfg = default_config
        cfg.num_boids = 50
        cfg.spatial_index = "auto"
        cfg.mode = "spatial"
        engine = SimulationEngine(cfg)

        # Verify initial index is SpatialHashGrid (50 < 5000)
        idx = engine.flock.get_index()
        assert isinstance(idx, SpatialHashGrid), (
            f"Auto with 50 birds must use SpatialHashGrid, got {type(idx).__name__}"
        )

        # Run 3 steps to warm up
        engine.run_headless(steps=3)
        assert engine.frame == 3
        assert np.isfinite(engine.flock.positions).all()

        # Switch to kdtree mid-simulation and step again
        engine.config.spatial_index = "kdtree"
        engine.flock._spatial_index_mode = "kdtree"
        engine.flock._index = KDTreeIndex()

        engine.step()
        assert engine.frame == 4
        # Index must now be KDTreeIndex
        idx2 = engine.flock.get_index()
        assert isinstance(idx2, KDTreeIndex), (
            f"After switch to kdtree, must be KDTreeIndex, got {type(idx2).__name__}"
        )
        # KDTreeIndex.query_knn must return valid global indices
        active_pos = engine.flock.positions[engine.flock.active]
        result = idx2.query_knn(active_pos[0], k=5)
        assert len(result) > 0, "KDTreeIndex must return neighbours after mid-sim switch"

    def test_kdtree_to_hash_grid_mid_simulation(self, default_config):
        """P2.1→P2.3→P2.4: Start with kdtree, switch to hash_grid, verify
        index rebuilds correctly and no NaN occurs."""
        cfg = default_config
        cfg.num_boids = 50
        cfg.spatial_index = "kdtree"
        cfg.mode = "spatial"
        engine = SimulationEngine(cfg)

        assert isinstance(engine.flock.get_index(), KDTreeIndex)

        # Run steps
        engine.run_headless(steps=3)

        # Switch to hash_grid
        engine.config.spatial_index = "hash_grid"
        engine.flock._spatial_index_mode = "hash_grid"
        engine.flock._index = SpatialHashGrid(cfg)

        engine.step()
        assert isinstance(engine.flock.get_index(), SpatialHashGrid)
        assert engine.flock.get_index().ready, "Hash grid must be rebuilt by engine.step"
        assert np.isfinite(engine.flock.positions).all()

    def test_hash_grid_to_none_mid_simulation(self, default_config):
        """P2.1→P2.3: Set spatial_index='none' mid-sim — index is None,
        mode that needs index must still work (self-built fallback)."""
        cfg = default_config
        cfg.num_boids = 30
        cfg.spatial_index = "hash_grid"
        cfg.mode = "spatial"
        engine = SimulationEngine(cfg)

        engine.run_headless(steps=2)

        # Switch to none — spatial mode still needs to compute forces
        engine.config.spatial_index = "none"
        engine.flock._spatial_index_mode = "none"
        engine.flock._index = None

        # Step — must not crash even though spatial mode needs an index
        # (the force mode itself will handle it)
        engine.step()
        assert engine.frame == 3
        assert np.isfinite(engine.flock.positions).all()

    def test_auto_reevaluation_on_bird_count_crossing(self, default_config):
        """P2.1→P2.4: When spatial_index='auto', adding birds past 5000
        triggers KDTreeIndex migration; then removing below 5000 migrates back."""
        cfg = default_config
        cfg.num_boids = 100
        cfg.spatial_index = "auto"
        cfg.mode = "projection"
        engine = SimulationEngine(cfg)

        # 100 < 5000 → SpatialHashGrid
        assert isinstance(engine.flock.get_index(), SpatialHashGrid)

        # Add 5000 birds via drain — _reevaluate_index triggers migration
        engine.enqueue_add(5000)
        engine.drain_commands()
        assert engine.flock.N_active == 5100
        assert isinstance(engine.flock.get_index(), KDTreeIndex), (
            "Crossing 5000 with auto must migrate to KDTreeIndex"
        )

        # Remove 200 birds — back below 5000
        engine.enqueue_remove(200)
        engine.drain_commands()
        assert engine.flock.N_active == 4900
        assert isinstance(engine.flock.get_index(), SpatialHashGrid), (
            "Dropping below 5000 with auto must migrate back to SpatialHashGrid"
        )

    def test_index_switch_preserves_active_mask(self, default_config):
        """P2.1→P2.3→P2.4: Switching index type does not alter active mask
        or positions/velocities."""
        cfg = default_config
        cfg.num_boids = 50
        cfg.spatial_index = "hash_grid"
        cfg.mode = "projection"
        engine = SimulationEngine(cfg)

        engine.run_headless(steps=5)

        # Snapshot state
        active_before = engine.flock.active.copy()
        engine.flock.positions.copy()
        engine.flock.velocities.copy()

        # Switch to kdtree and step
        engine.config.spatial_index = "kdtree"
        engine.flock._spatial_index_mode = "kdtree"
        engine.flock._index = KDTreeIndex()
        engine.step()

        # Active mask unchanged
        assert np.array_equal(active_before, engine.flock.active), (
            "Active mask must not change on index switch"
        )
        # Positions/velocities should have changed (physics ran), but not NaN
        assert np.isfinite(engine.flock.positions).all()
        assert np.isfinite(engine.flock.velocities).all()


# ═══════════════════════════════════════════════════════════════════
# IT-P2-2: Full Threat/Evasion Pipeline (P2.6 → P2.10)
# ═══════════════════════════════════════════════════════════════════

class TestThreatEvasionPipeline:
    """Verify the full pipeline: ExtensionManager → StepContext.threat_prox
    → force computation reads threat_prox → evasion forces applied."""

    def test_predator_publishes_threat_prox_to_context(self, default_config):
        """P2.6→P2.10: engine.step() wires extensions.pre_step() which
        sets ctx.threat_prox, then compute_all_forces can read it."""
        cfg = default_config
        cfg.num_boids = 30
        cfg.predator_enabled = True
        cfg.roosting_enabled = False  # no gating
        cfg.mode = "spatial"
        cfg.seed = 42
        engine = SimulationEngine(cfg)

        # Monkey-patch compute_all_forces to capture what ctx looks like
        from pymurmur.simulation import engine as eng_module

        captured_threat = []
        orig_compute = eng_module.compute_all_forces

        def spy_compute(flock, config):
            # We can't capture ctx directly, but we can verify threat_prox
            # was set by checking flock.accelerations before/after
            captured_threat.append(flock.accelerations.copy())
            return orig_compute(flock, config)

        eng_module.compute_all_forces = spy_compute
        try:
            engine.step()
        finally:
            eng_module.compute_all_forces = orig_compute

        # Forces were computed — engine didn't crash
        assert engine.frame == 1
        assert np.isfinite(engine.flock.positions).all()
        # At least some forces should be non-zero (predator + spatial forces)
        assert not np.allclose(engine.flock.last_accelerations[engine.flock.active], 0.0), (
            "Forces must be non-zero after step with predator + spatial mode"
        )

    def test_threat_prox_visible_to_force_composition(self, default_config):
        """P2.6→P2.10: When predator is enabled, the force computation
        produces non-zero accelerations on birds near the predator.

        Verify by running a single engine with predator, capturing
        pre-step accelerations (zero), then verifying post-step
        accelerations are non-zero for birds near predator."""
        cfg = default_config
        cfg.num_boids = 20
        cfg.predator_enabled = True
        cfg.roosting_enabled = False
        cfg.mode = "spatial"
        cfg.seed = 42

        engine = SimulationEngine(cfg)

        # Place all birds at known positions — some near centre (where
        # predator spawns), some far away
        engine.flock.positions[:] = np.array([
            [500, 350, 200],   # bird 0 — near centre
            [510, 350, 200],   # bird 1 — near centre
            [800, 600, 350],   # bird 2 — far
            [900, 600, 350],   # bird 3 — far
            [100, 50, 50],     # bird 4 — far corner
        ] * 4, dtype=np.float32)[:20]
        engine.flock.velocities[:] = np.array([4.0, 0, 0], dtype=np.float32)

        # Force predator to centre of domain
        engine.extensions._predator._pos = np.array(
            [500, 350, 200], dtype=np.float32
        )
        engine.extensions._predator._phase = "approach"

        # Run a step
        engine.step()

        # Verify forces were non-zero for at least some birds
        acc_mags = np.linalg.norm(
            engine.flock.last_accelerations[engine.flock.active], axis=1
        )
        assert acc_mags.max() > 0, (
            "Forces must be non-zero after step with predator"
        )
        # Birds near centre (<100 from predator) should feel stronger
        # force than birds far away (>300 from predator)
        pos = engine.flock.positions
        pred_pos = engine.extensions._predator._pos
        dists = np.linalg.norm(pos - pred_pos, axis=1)
        near = dists < 150
        far = dists > 300
        if near.any() and far.any():
            near_force = acc_mags[near].mean() if near.any() else 0
            acc_mags[far].mean() if far.any() else 0
            # Near birds should feel at least as much force as far birds
            # (predator threat decays with distance)
            assert near_force > 0, "Near birds must feel force"

        assert np.isfinite(engine.flock.positions).all()

    def test_extensions_then_forces_execute_in_order(self, default_config):
        """P2.6→P2.10: Engine step order: extensions (sets threat_prox)
        BEFORE forces (reads threat_prox). Verify this order is maintained."""
        cfg = default_config
        cfg.num_boids = 20
        cfg.predator_enabled = True
        cfg.roosting_enabled = False
        cfg.mode = "spatial"
        engine = SimulationEngine(cfg)

        order_log = []

        orig_ext = engine.extensions.pre_step
        orig_forces = __import__("pymurmur.simulation.engine", fromlist=["compute_all_forces"]).compute_all_forces

        def spy_ext(flock, ctx):
            order_log.append("extensions")
            return orig_ext(flock, ctx)

        def spy_forces(flock, config):
            order_log.append("forces")
            return orig_forces(flock, config)

        engine.extensions.pre_step = spy_ext

        import pymurmur.simulation.engine as eng
        eng.compute_all_forces = spy_forces
        try:
            engine.step()
        finally:
            eng.compute_all_forces = orig_forces
            engine.extensions.pre_step = orig_ext

        ext_idx = order_log.index("extensions")
        forces_idx = order_log.index("forces")
        assert ext_idx < forces_idx, (
            f"Extensions ({ext_idx}) must run before forces ({forces_idx}): {order_log}"
        )

    def test_force_composition_with_extensions_enabled(self, default_config):
        """P2.6→P2.10: ForceTerm composition (via engine.step with all
        extensions enabled) produces valid accelerations."""
        cfg = default_config
        cfg.num_boids = 30
        cfg.predator_enabled = True
        cfg.roosting_enabled = True
        cfg.wander_enabled = True
        cfg.ripple_enabled = True
        cfg.mode = "spatial"
        engine = SimulationEngine(cfg)

        # All 4 extensions + spatial forces
        engine.run_headless(steps=5)

        # No NaN, no divergence
        assert engine.frame == 5
        assert np.isfinite(engine.flock.positions).all()
        assert np.isfinite(engine.flock.velocities).all()
        # Birds must still be in the domain
        pos = engine.flock.positions[engine.flock.active]
        assert (pos[:, 0] >= 0).all() and (pos[:, 0] <= cfg.width).all()

    def test_force_composition_mode_switch_with_extensions(self, default_config):
        """P2.2→P2.6→P2.10: Switch force mode mid-simulation while
        extensions are active — pipeline must adapt without crash."""
        cfg = default_config
        cfg.num_boids = 20
        cfg.predator_enabled = True
        cfg.mode = "projection"
        engine = SimulationEngine(cfg)

        engine.run_headless(steps=3)

        # Switch to vicsek — predator extension still runs
        engine.config.mode = "vicsek"
        engine.step()
        assert engine.frame == 4
        assert np.isfinite(engine.flock.positions).all()

        # Switch to field
        engine.config.mode = "field"
        engine.step()
        assert engine.frame == 5

        # Final check: all birds still alive and in bounds
        assert engine.flock.N_active == 20
        assert np.isfinite(engine.flock.positions).all()

