"""D14/D15 — Angle mode per-instance parameters and structured AngleConfig.

Split out of test_angle.py (file-size split).
"""

import numpy as np

from pymurmur.core.config import SimConfig
from pymurmur.physics.flock import PhysicsFlock
from pymurmur.physics.forces.angle import AngleMode


class TestD14AngleModePerInstance:
    """D14: _angle_last_cell is per-spatial-index, not class-level.

    Two engines with different N must each have their own
    _angle_last_cell array — no cross-contamination.
    """

    def test_two_engine_different_n_independent_last_cell(self):
        """D14: Engines with different N get independent _angle_last_cell.

        Engine A (N=10) and Engine B (N=20) each run angle compute.
        Their _angle_last_cell arrays must have the correct shapes
        for their own N and not interfere.
        """
        from pymurmur.simulation.engine import SimulationEngine

        cfg_a = SimConfig()
        cfg_a.mode = "angle"
        cfg_a.num_boids = 10
        cfg_a.boundary_mode = "toroidal"
        cfg_a.seed = 40

        cfg_b = SimConfig()
        cfg_b.mode = "angle"
        cfg_b.num_boids = 20
        cfg_b.boundary_mode = "toroidal"
        cfg_b.seed = 41

        engine_a = SimulationEngine(cfg_a)
        engine_b = SimulationEngine(cfg_b)

        # Step both engines once
        engine_a.step(1.0 / 60.0)
        engine_b.step(1.0 / 60.0)

        # Each engine's spatial index must have its own _angle_last_cell
        idx_a = engine_a.flock.get_index()
        idx_b = engine_b.flock.get_index()

        last_cell_a = getattr(idx_a, '_angle_last_cell', None)
        last_cell_b = getattr(idx_b, '_angle_last_cell', None)

        assert last_cell_a is not None, "Engine A must have _angle_last_cell"
        assert last_cell_b is not None, "Engine B must have _angle_last_cell"

        # Each must have the correct capacity for its own N
        assert last_cell_a.shape[0] >= cfg_a.num_boids, (
            f"Engine A _angle_last_cell shape {last_cell_a.shape} "
            f"doesn't cover N={cfg_a.num_boids}"
        )
        assert last_cell_b.shape[0] >= cfg_b.num_boids, (
            f"Engine B _angle_last_cell shape {last_cell_b.shape} "
            f"doesn't cover N={cfg_b.num_boids}"
        )

        # They must be independent arrays (not shared)
        assert last_cell_a is not last_cell_b, (
            "_angle_last_cell must be independent arrays, not shared"
        )

    def test_two_engine_same_n_independent_last_cell(self):
        """D14: Even with same N, engines get independent arrays."""
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.mode = "angle"
        cfg.num_boids = 8
        cfg.boundary_mode = "toroidal"
        cfg.seed = 42

        engine_1 = SimulationEngine(cfg)
        engine_2 = SimulationEngine(cfg)

        engine_1.step(1.0 / 60.0)
        engine_2.step(1.0 / 60.0)

        idx_1 = engine_1.flock.get_index()
        idx_2 = engine_2.flock.get_index()

        last_cell_1 = getattr(idx_1, '_angle_last_cell', None)
        last_cell_2 = getattr(idx_2, '_angle_last_cell', None)

        assert last_cell_1 is not None
        assert last_cell_2 is not None
        # Different engine → different array (even with same N)
        assert last_cell_1 is not last_cell_2, (
            "Same N engines must have independent _angle_last_cell arrays"
        )

    def test_different_n_no_corruption(self):
        """D14: Running small engine after large engine doesn't corrupt.

        After large engine (N=20) runs, small engine (N=5) must have
        _angle_last_cell with its own shape, not the large engine's.
        """
        from pymurmur.simulation.engine import SimulationEngine

        cfg_large = SimConfig()
        cfg_large.mode = "angle"
        cfg_large.num_boids = 20
        cfg_large.boundary_mode = "toroidal"
        cfg_large.seed = 100

        cfg_small = SimConfig()
        cfg_small.mode = "angle"
        cfg_small.num_boids = 5
        cfg_small.boundary_mode = "toroidal"
        cfg_small.seed = 101

        large = SimulationEngine(cfg_large)
        small = SimulationEngine(cfg_small)

        # Run large first, then small
        large.step(1.0 / 60.0)
        small.step(1.0 / 60.0)

        idx_small = small.flock.get_index()
        last_cell_small = getattr(idx_small, '_angle_last_cell', None)

        assert last_cell_small is not None
        # Small engine must have _angle_last_cell sized for its own N
        assert last_cell_small.shape[0] >= cfg_small.num_boids, (
            f"Small engine _angle_last_cell corrupted by large: "
            f"shape={last_cell_small.shape} vs N={cfg_small.num_boids}"
        )
        # Small engine must NOT have large engine's N (20)
        assert last_cell_small.shape[0] < 20, (
            f"Small engine got large engine's _angle_last_cell: "
            f"shape={last_cell_small.shape}"
        )

    def test_sequential_compute_same_index_persists(self):
        """D14: Multiple compute() calls on same index reuse _angle_last_cell.

        Repeated calls with the same index should not recreate the
        array unnecessarily — the per-index storage persists.
        """
        cfg = SimConfig()
        cfg.mode = "angle"
        cfg.num_boids = 5
        cfg.boundary_mode = "toroidal"

        flock = PhysicsFlock(cfg)
        flock.active[:] = True
        flock.accelerations[:] = 0.0
        idx = flock.get_index()
        idx.rebuild(flock.positions, flock.active)

        # First compute
        AngleMode.compute(
            flock.positions, flock.velocities, flock.accelerations,
            flock.active, idx, flock.rng, flock.last_theta, cfg,
        )
        first = getattr(idx, '_angle_last_cell', None)
        assert first is not None

        # Second compute — same index, same array object
        AngleMode.compute(
            flock.positions, flock.velocities, flock.accelerations,
            flock.active, idx, flock.rng, flock.last_theta, cfg,
        )
        second = getattr(idx, '_angle_last_cell', None)
        assert second is not None

        # Should be the same array object (persisted on index)
        assert first is second, (
            "_angle_last_cell must persist across compute() calls on same index"
        )

    def test_angle_mode_class_no_longer_has_last_cell(self):
        """D14: AngleMode class no longer has _last_cell class attribute."""
        assert not hasattr(AngleMode, '_last_cell'), (
            "AngleMode._last_cell must not exist as class-level attribute"
        )


# ═══════════════════════════════════════════════════════════════════
# D15: AngleConfig structured access (no getattr fallbacks)
# ═══════════════════════════════════════════════════════════════════


class TestD15AngleConfigStructured:
    """D15: Angle mode reads config via structured access, not getattr.

    All angle-specific knobs live in AngleConfig. Boundary-related
    fields (margin, mode, sphere_radius) live in BoundaryConfig.
    No getattr(config, ...) fallbacks remain in angle.py.
    """

    def test_no_getattr_config_fallbacks_in_angle_py(self):
        """D15: angle.py has zero getattr(config, ...) fallbacks
        except for the _coherence_factor runtime bridge (S2.B8).
        _coherence_factor is set dynamically by the ecology extension
        and read via getattr(config, '_coherence_factor', 1.0) — this
        is the sanctioned runtime-bridge pattern, not a config fallback."""
        from pathlib import Path
        src = Path(__file__).parents[4] / "pymurmur" / "physics" / "forces" / "angle.py"
        text = src.read_text()
        # getattr on index is fine (D14 per-index storage), but
        # getattr on config must not exist anywhere in angle.py,
        # except for _coherence_factor (runtime bridge from ecology).
        import re
        # Negative lookahead: exclude the sanctioned _coherence_factor bridge
        matches = re.findall(
            r'getattr\(\s*config\b(?!\s*,\s*[\'"]_coherence_factor)',
            text,
        )
        assert len(matches) == 0, (
            f"angle.py must not use getattr(config, ...) except "
            f"_coherence_factor bridge: found {matches}"
        )

    def test_angle_config_yaml_roundtrip(self, tmp_path):
        """D15: AngleConfig values survive YAML round-trip unchanged."""
        from pymurmur.core.config import SimConfig

        cfg = SimConfig()
        cfg.mode = "angle"
        cfg.angle.turn_rate = 90.0
        cfg.angle.max_turn_rate = 360.0
        cfg.angle.turn_threshold = 1.5
        cfg.angle.jitter_deg = 8.0
        cfg.angle.base_speed = 200.0
        cfg.angle.angle_neighbors = 5
        cfg.angle.sep_radius_bodies = 2.0
        cfg.angle.align_radius_bodies = 6.0
        cfg.angle.range_radius_bodies = 15.0
        cfg.boundary.boundary_margin = 75.0
        cfg.boundary.boundary_mode = "margin"
        cfg.boundary.boundary_sphere_radius = 400.0

        p = tmp_path / "angle_test.yaml"
        cfg.to_file(p)
        loaded = SimConfig.from_file(p)

        assert loaded.angle.turn_rate == 90.0
        assert loaded.angle.max_turn_rate == 360.0
        assert loaded.angle.turn_threshold == 1.5
        assert loaded.angle.jitter_deg == 8.0
        assert loaded.angle.base_speed == 200.0
        assert loaded.angle.angle_neighbors == 5
        assert loaded.angle.sep_radius_bodies == 2.0
        assert loaded.angle.align_radius_bodies == 6.0
        assert loaded.angle.range_radius_bodies == 15.0
        assert loaded.boundary.boundary_margin == 75.0
        assert loaded.boundary.boundary_mode == "margin"
        assert loaded.boundary.boundary_sphere_radius == 400.0

    def test_angle_config_defaults_match_spec(self):
        """D15: AngleConfig defaults match the documented spec values."""
        from pymurmur.core.config import SimConfig

        cfg = SimConfig()
        assert cfg.angle.turn_rate == 120.0
        assert cfg.angle.max_turn_rate == 200.0
        assert cfg.angle.turn_threshold == 0.5
        assert cfg.angle.jitter_deg == 4.0
        assert cfg.angle.base_speed == 150.0
        assert cfg.angle.angle_neighbors == 7
        assert cfg.angle.sep_radius_bodies == 1.0
        assert cfg.angle.align_radius_bodies == 5.0
        assert cfg.angle.range_radius_bodies == 12.0

    def test_boundary_fields_read_from_boundary_config(self):
        """D15: boundary_mode/fps/sphere_radius read from structured config.

        The fields previously accessed via getattr(config, ...) are now
        in BoundaryConfig / VizConfig and accessible via dot notation.
        """
        from pymurmur.core.config import SimConfig

        cfg = SimConfig()
        # These must exist as proper fields (no AttributeError)
        _ = cfg.boundary.boundary_mode
        _ = cfg.boundary.boundary_sphere_radius
        _ = cfg.fps
        assert cfg.boundary.boundary_mode == "toroidal"
        assert cfg.boundary.boundary_sphere_radius == 300.0
        assert cfg.fps > 0

    def test_angle_mode_uses_structured_config_at_runtime(self):
        """D15: Running angle mode uses structured config (no crash)."""
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.mode = "angle"
        cfg.num_boids = 10
        cfg.boundary.boundary_mode = "toroidal"
        cfg.angle.turn_rate = 90.0
        cfg.angle.base_speed = 100.0
        cfg.seed = 42

        engine = SimulationEngine(cfg)
        for _ in range(5):
            engine.step(1.0 / 60.0)

        assert np.isfinite(engine.flock.positions).all()
        assert np.isfinite(engine.flock.velocities).all()



# ═══════════════════════════════════════════════════════════════════
# Part IV Cross-Item Integration: Angle mode as a whole
# ═══════════════════════════════════════════════════════════════════

