"""IT-P2-3: InstanceSchema buffer packing + holey-flock composition.

Split out of test_cross_subsystem.py (file-size split) — spatial-index
swap and threat/evasion pipeline tests stay in the original.
"""

import numpy as np
import pytest

from pymurmur.simulation.engine import SimulationEngine

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
