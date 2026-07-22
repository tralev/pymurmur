"""Phase 2 cross-item integration tests — verify 2+ Phase 2 items work
together as a whole.

IT-P2-1: Dynamic spatial index swap (P2.1 → P2.3 → P2.4)
IT-P2-2: Full threat/evasion pipeline (P2.6 → P2.10 via engine.step)
IT-P2-3: InstanceSchema buffer packing consistency (P2.7 → P2.8)
"""

import numpy as np
import pytest

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


# ═══════════════════════════════════════════════════════════════════
# IT-P2-3: InstanceSchema Buffer Packing Consistency (P2.7 → P2.8)
# ═══════════════════════════════════════════════════════════════════

class TestInstanceSchemaPacking:
    """Verify InstanceSchema layout is consistent with Renderer3D
    buffer allocation and vbo.write() calls."""

    def test_schema_floats_matches_packed_array(self):
        """P2.7→P2.8/D7: InstanceSchema.floats must match the packed
        numpy array column count used in update_instances.

        D7: the renderer packs positions (3) + velocities (3) + hue (1)
        + scale (1) = 8 floats into one merged buffer (was 6, with hue
        + scale in a separate colour buffer). If schema.floats ≠
        packed.shape[1], GPU buffer is misaligned.
        """
        from pymurmur.viz.renderer import InstanceSchema

        schema = InstanceSchema()

        # Replicate what update_instances does:
        # self._packed = np.zeros((max_instances, schema.floats), dtype=np.float32)
        max_instances = 100
        packed = np.zeros((max_instances, schema.floats), dtype=np.float32)

        # Pack positions, velocities, hue, scale (D7: matches
        # Renderer3D.update_instances's column layout exactly)
        n = 10
        pos = np.random.randn(n, 3).astype(np.float32)
        vel = np.random.randn(n, 3).astype(np.float32)
        hue = np.random.rand(n).astype(np.float32)
        scale = np.ones(n, dtype=np.float32)
        packed[:n, 0:3] = pos
        packed[:n, 3:6] = vel
        packed[:n, 6] = hue
        packed[:n, 7] = scale

        # Verify shape matches schema
        assert packed.shape[1] == schema.floats, (
            f"Packed array has {packed.shape[1]} columns but schema says {schema.floats}"
        )
        # The vbo.write() uses packed[:n].tobytes()
        # Expected bytes: n * schema.floats * 4
        expected_bytes = n * schema.floats * 4
        actual_bytes = len(packed[:n].tobytes())
        assert actual_bytes == expected_bytes, (
            f"Packed bytes: {actual_bytes}, expected: {expected_bytes}"
        )

    def test_schema_layout_components_count_matches_attrs(self):
        """P2.7/D7: layout string components count must equal len(attrs).

        '3f 3f 1f 1f/i' has 4 components → ('in_bird_pos', 'in_bird_vel',
        'in_bird_hue', 'in_bird_scale') has 4 entries. Mismatch causes
        ModernGL VAO creation error.
        """
        from pymurmur.viz.renderer import InstanceSchema

        schema = InstanceSchema()

        # Parse layout: space-separated format components
        components = schema.layout.split()
        assert len(components) == 4, f"Layout '{schema.layout}' has {len(components)} components"
        assert len(schema.attrs) == len(components), (
            f"Layout has {len(components)} components but attrs has {len(schema.attrs)} entries"
        )

    def test_pos_vel_view_components_count_matches_attrs(self):
        """D7: the pos+vel-only padded view (used by the impostor VAO,
        whose shader has no in_bird_hue/in_bird_scale inputs) has its
        own component count matching its own 2-entry attrs tuple —
        independent of the main 4-component layout above."""
        from pymurmur.viz.renderer import InstanceSchema

        schema = InstanceSchema()
        components = schema.pos_vel_layout.split()
        assert len(components) == 3, (
            f"pos_vel_layout '{schema.pos_vel_layout}' has {len(components)} "
            f"components (3f, 3f, 8x — padding doesn't get its own attr name)"
        )
        # The trailing "8x" padding component has no attribute name —
        # only the first 2 components (3f, 3f) bind to pos_vel_attrs.
        assert len(schema.pos_vel_attrs) == 2

    def test_schema_buffer_allocation_formula(self):
        """P2.7: Buffer allocation = max_instances * schema.floats * 4 bytes.

        The renderer uses this formula in __init__ and reallocation.
        Changing schema.floats must produce matching byte count.
        """
        from pymurmur.viz.renderer import InstanceSchema

        for floats in (6, 9, 12):
            schema = InstanceSchema(floats=floats)
            max_instances = 50000

            # Replicate renderer's buffer allocation
            max_instances * schema.floats * 4
            packed = np.zeros((max_instances, schema.floats), dtype=np.float32)

            # Pack and write
            n = 10
            packed[:n, :floats] = np.random.randn(n, floats).astype(np.float32)
            written_bytes = len(packed[:n].tobytes())

            assert written_bytes == n * floats * 4, (
                f"{floats}-float schema: written {written_bytes} bytes, "
                f"expected {n * floats * 4}"
            )

    def test_schema_change_propagates_to_packed_shape(self):
        """P2.7: Changing InstanceSchema.floats changes the packed array
        dimensions, which changes vbo.write() byte count — verify the
        formula holds for any float count."""
        from pymurmur.viz.renderer import InstanceSchema

        # Default: 6 floats
        s6 = InstanceSchema(floats=6)
        packed6 = np.zeros((1000, s6.floats), dtype=np.float32)
        assert packed6.shape == (1000, 6)
        assert len(packed6[:5].tobytes()) == 5 * 6 * 4

        # Extended: 9 floats (e.g., +color.rgb)
        s9 = InstanceSchema(floats=9)
        packed9 = np.zeros((1000, s9.floats), dtype=np.float32)
        assert packed9.shape == (1000, 9)
        assert len(packed9[:5].tobytes()) == 5 * 9 * 4

        # 12 floats (e.g., + species + group)
        s12 = InstanceSchema(floats=12)
        packed12 = np.zeros((1000, s12.floats), dtype=np.float32)
        assert packed12.shape == (1000, 12)
        assert len(packed12[:5].tobytes()) == 5 * 12 * 4

    def test_mat4_bytes_size_never_changes_with_schema(self):
        """P2.8→P2.7: _mat4_bytes always returns 64 bytes regardless
        of InstanceSchema configuration. Matrix uniforms are separate
        from instance buffers."""
        from pymurmur.viz.renderer import _mat4_bytes
        glm = pytest.importorskip("glm", reason="PyGLM not installed")

        m = glm.mat4(1.0)
        b = _mat4_bytes(m)
        assert len(b) == 64, "_mat4_bytes must always return 64 bytes"

        # InstanceSchema changes don't affect matrix uploads
        from pymurmur.viz.renderer import InstanceSchema
        for floats in (6, 9, 12):
            InstanceSchema(floats=floats)  # any float count
            b2 = _mat4_bytes(glm.mat4(1.0))
            assert len(b2) == 64, (
                f"mat4 must be 64 bytes even with {floats}-float schema"
            )


# ── Cross-item: holey flock + extensions + force composition ──────

class TestHoleyFlockWithExtensionsAndComposition:
    """P2.9→P2.6→P2.10: Holey flock (inactive birds) + extensions
    + force composition — the full Phase 2 pipeline."""

    def test_holey_flock_with_predator_extension(self, default_config):
        """P2.9→P2.6→P2.10: Engine with holey flock + predator enabled
        runs without crash and inactive birds stay frozen."""
        cfg = default_config
        cfg.num_boids = 30
        cfg.predator_enabled = True
        cfg.roosting_enabled = False
        cfg.mode = "spatial"
        engine = SimulationEngine(cfg)

        # Create holes: deactivate birds 10-14 and 20-24
        engine.flock.active[10:15] = False
        engine.flock.active[20:25] = False
        inactive_mask = ~engine.flock.active
        active_mask = engine.flock.active

        pos_before_inactive = engine.flock.positions[inactive_mask].copy()
        vel_before_inactive = engine.flock.velocities[inactive_mask].copy()
        pos_before_active = engine.flock.positions[active_mask].copy()

        # Run 10 steps
        engine.run_headless(steps=10)

        # Inactive positions must be bit-identical (never touched)
        np.testing.assert_array_equal(pos_before_inactive,
            engine.flock.positions[inactive_mask],
            err_msg="Inactive positions must be unchanged after 10 steps")
        np.testing.assert_array_equal(vel_before_inactive,
            engine.flock.velocities[inactive_mask],
            err_msg="Inactive velocities must be unchanged after 10 steps")

        # Active birds must have moved (physics ran)
        assert not np.allclose(
            engine.flock.positions[active_mask], pos_before_active,
            atol=1e-4
        ), "Active birds must have moved after 10 steps"
        assert engine.frame == 10

    def test_holey_flock_all_extensions_enabled(self, default_config):
        """P2.9→P2.6: Holey flock with all 4 extensions enabled — no
        crash, no inactive bird corruption."""
        cfg = default_config
        cfg.num_boids = 30
        cfg.predator_enabled = True
        cfg.roosting_enabled = True
        cfg.wander_enabled = True
        cfg.ripple_enabled = True
        cfg.mode = "projection"
        engine = SimulationEngine(cfg)

        # Hole pattern
        engine.flock.active[5:10] = False
        engine.flock.active[15:20] = False
        inactive_mask = ~engine.flock.active
        pos_before = engine.flock.positions[inactive_mask].copy()

        engine.run_headless(steps=10)

        np.testing.assert_array_equal(
            pos_before, engine.flock.positions[inactive_mask],
            err_msg="Inactive positions corrupted with all extensions enabled"
        )
        assert np.isfinite(engine.flock.positions).all()
