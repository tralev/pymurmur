"""Unit tests for simulation engine P8.10 — fixed-timestep accumulator + lerp.

Covers: accumulator behaviour, dt_phys config field, lerp correctness,
30fps vs 60fps determinism, spike clamping, render_positions, and
backward compatibility with existing step() callers.
"""

from copy import copy
from unittest.mock import patch

import numpy as np
import pytest

from pymurmur.core.config import SimConfig
from pymurmur.simulation.engine import SimulationEngine

# ── P8.10a: Config field ──────────────────────────────────────

def test_dt_phys_field_exists(default_config):
    """dt_phys field exists on FlockConfig with default 1/60."""
    cfg = default_config
    assert hasattr(cfg, "dt_phys")
    assert cfg.dt_phys == pytest.approx(1.0 / 60.0)

def test_dt_phys_configurable(default_config):
    """dt_phys can be set via SimConfig constructor."""
    from pymurmur.core.config import SimConfig
    cfg = SimConfig(dt_phys=0.01)
    assert cfg.dt_phys == 0.01

def test_dt_phys_in_to_file(default_config, tmp_path):
    """dt_phys appears in YAML output."""
    out = tmp_path / "cfg.yaml"
    default_config.to_file(str(out))
    assert "dt_phys" in out.read_text()


# ── P8.10b: Accumulator ───────────────────────────────────────

class TestAccumulator:
    """Fixed-timestep accumulator drains frame_dt into dt_phys steps."""

    def test_accumulator_initialised_zero(self, default_config):
        """_accumulator starts at 0."""
        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)
        assert engine._accumulator == 0.0

    def test_one_step_at_dt_phys(self, default_config):
        """frame_dt == dt_phys → 1 physics step, accumulator ~0."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.dt_phys = 1.0 / 60.0
        engine = SimulationEngine(cfg)
        engine.step(1.0 / 60.0)
        assert engine.frame == 1
        assert engine._accumulator < 1e-10

    def test_two_steps_at_half_dt_phys(self, default_config):
        """frame_dt = 2× dt_phys → 2 physics steps."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.dt_phys = 1.0 / 60.0
        engine = SimulationEngine(cfg)
        engine.step(2.0 / 60.0)
        assert engine.frame == 2
        assert engine._accumulator < 1e-10

    def test_partial_accumulator_remains(self, default_config):
        """frame_dt = 1.5× dt_phys → 1 step, accumulator = 0.5× dt_phys."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.dt_phys = 1.0 / 60.0
        engine = SimulationEngine(cfg)
        engine.step(1.5 / 60.0)
        assert engine.frame == 1
        assert engine._accumulator == pytest.approx(0.5 / 60.0)

    def test_accumulator_carries_over(self, default_config):
        """Residual accumulator carries to next step call."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.dt_phys = 1.0 / 60.0
        engine = SimulationEngine(cfg)

        # First call: 0.7× dt_phys, no step fired
        engine.step(0.7 / 60.0)
        assert engine.frame == 0
        assert engine._accumulator == pytest.approx(0.7 / 60.0)

        # Second call: 0.5× dt_phys, total 1.2× → 1 step, residual 0.2×
        engine.step(0.5 / 60.0)
        assert engine.frame == 1
        assert engine._accumulator == pytest.approx(0.2 / 60.0)

    def test_big_frame_dt_many_steps(self, default_config):
        """frame_dt = 3× dt_phys (below clamp) → 3 physics steps."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.dt_phys = 1.0 / 60.0
        engine = SimulationEngine(cfg)
        # 3× dt_phys = 0.05s, just below the 1/20=0.05s clamp
        engine.step(3.0 / 60.0)
        assert engine.frame == 3
        assert engine._accumulator < 1e-10


# ── P8.10c: Spike clamping ────────────────────────────────────

class TestSpikeClamp:
    """frame_dt is clamped at 1/20s to avoid spiral-of-death."""

    def test_frame_dt_clamped(self, default_config):
        """frame_dt > 0.05s gets clamped to 1/20."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.dt_phys = 1.0 / 60.0
        engine = SimulationEngine(cfg)

        # frame_dt = 0.1s → clamped to 1/20 = 0.05s → 3 steps
        engine.step(0.1)
        # 0.05 / (1/60) = 3.0 → exactly 3 steps
        assert engine.frame == 3
        assert engine._accumulator < 1e-10

    def test_frame_dt_below_clamp_not_affected(self, default_config):
        """frame_dt below clamp threshold passes through unchanged."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.dt_phys = 1.0 / 60.0
        engine = SimulationEngine(cfg)

        engine.step(0.04)  # below 1/20
        # No clamping, 0.04 / (1/60) = 2.4 → 2 steps, residual 0.4*dt_phys
        assert engine.frame == 2
        assert engine._accumulator == pytest.approx(0.4 / 60.0)


# ── P8.10d: render_positions / lerp ───────────────────────────

class TestRenderPositions:
    """render_positions is lerped between prev_positions and positions."""

    def test_render_positions_none_when_accumulator_clean(self, default_config):
        """When accumulator drains cleanly, render_positions is None (no copy)."""
        cfg = default_config
        cfg.num_boids = 20
        engine = SimulationEngine(cfg)

        engine.step(1.0 / 60.0)
        # Exact dt match: accumulator drained → render_positions = None
        # (visualizer reads flock.positions directly, saving a copy)
        assert engine.render_positions is None

    def test_render_positions_none_when_accumulator_zero(self, default_config):
        """When accumulator=0 (exact dt_phys), render_positions is None.

        Optimization: None signals the visualizer to read flock.positions
        directly, saving an array copy when positions == prev_positions.
        """
        cfg = default_config
        cfg.num_boids = 20
        cfg.dt_phys = 1.0 / 60.0
        engine = SimulationEngine(cfg)

        engine.step(1.0 / 60.0)
        # With exact dt_phys, accumulator should be 0
        assert engine._accumulator < 1e-10
        # Optimization: None → visualizer reads flock.positions directly
        assert engine.render_positions is None

    def test_render_positions_lerps_when_accumulator_nonzero(self, default_config):
        """With partial accumulator, render_positions is between prev and cur."""
        cfg = default_config
        cfg.num_boids = 20
        cfg.dt_phys = 1.0 / 60.0
        engine = SimulationEngine(cfg)

        # Run one exact step to establish state
        engine.step(1.0 / 60.0)
        prev = engine.flock.prev_positions.copy()
        cur = engine.flock.positions.copy()

        # Run a partial step → accumulator nonzero
        engine.step(0.5 / 60.0)
        # 0.5 × dt_phys: accumulator = 0.5 * dt_phys, no new physics step
        # render_positions should be lerp(prev, cur, 0.5)
        assert engine._accumulator == pytest.approx(0.5 / 60.0)

        expected = prev + 0.5 * (cur - prev)
        np.testing.assert_array_almost_equal(
            engine.render_positions, expected, decimal=5
        )

    def test_render_positions_dtype_when_nonzero_accumulator(self, default_config):
        """When accumulator is nonzero, render_positions has float32 dtype."""
        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)
        # Partial step: 0.5× dt_phys → nonzero accumulator → render_positions set
        engine.step(0.5 / 60.0)
        assert engine.render_positions is not None
        assert engine.render_positions.dtype == np.float32


# ── P8.10e: 30fps vs 60fps determinism ────────────────────────

class TestFramerateIndependence:
    """Same elapsed time → same physics state regardless of framerate."""

    def test_30fps_vs_60fps_same_physics(self, default_config):
        """After 1 second, 30fps and 60fps runs have identical positions."""
        cfg_60 = default_config
        cfg_60.num_boids = 50
        cfg_60.seed = 42
        cfg_60.dt_phys = 1.0 / 60.0

        cfg_30 = copy(cfg_60)
        cfg_30.seed = 42  # same seed for determinism

        # Run at 60fps for 1 second (60 frames)
        engine_60 = SimulationEngine(cfg_60)
        for _ in range(60):
            engine_60.step(1.0 / 60.0)

        # Run at 30fps for 1 second (30 frames)
        engine_30 = SimulationEngine(cfg_30)
        for _ in range(30):
            engine_30.step(1.0 / 30.0)

        # Both should have 60 physics steps → identical positions
        assert engine_60.frame == 60
        assert engine_30.frame == 60
        np.testing.assert_array_almost_equal(
            engine_60.flock.positions, engine_30.flock.positions, decimal=5
        )

    def test_variable_framerate_same_physics(self, default_config):
        """Variable framerate summing to same elapsed time → same physics steps."""
        cfg_a = default_config
        cfg_a.num_boids = 30
        cfg_a.seed = 99
        cfg_a.dt_phys = 1.0 / 60.0

        cfg_b = copy(cfg_a)
        cfg_b.seed = 99

        engine_a = SimulationEngine(cfg_a)
        engine_b = SimulationEngine(cfg_b)

        # Engine A: consistent 60fps for 0.5s (30 frames × 1/60)
        for _ in range(30):
            engine_a.step(1.0 / 60.0)

        # Engine B: mixed framerate totalling 0.5s
        # 10 frames at 1/30 = 0.333s + 10 frames at 1/60 = 0.167s = 0.5s
        for _ in range(10):
            engine_b.step(1.0 / 30.0)
        for _ in range(10):
            engine_b.step(1.0 / 60.0)

        # Both should have 30 physics steps
        assert engine_a.frame == 30
        assert engine_b.frame == 30

    def test_accumulator_preserves_physics_total_elapsed(self, default_config):
        """Variable framerate to 1.0s = 60 physics steps regardless."""
        cfg = default_config
        cfg.num_boids = 30
        cfg.seed = 7
        cfg.dt_phys = 1.0 / 60.0

        engine = SimulationEngine(cfg)
        # Feed frames of varying dt that total 1.0s
        # 20 frames at 1/30 = 20/30 = 0.666s
        for _ in range(20):
            engine.step(1.0 / 30.0)
        # 20 frames at 1/60 = 20/60 = 0.333s
        for _ in range(20):
            engine.step(1.0 / 60.0)

        # Total elapsed = 0.666 + 0.333 = 1.0s → 60 physics steps
        assert engine.frame == 60


# ── P8.10f: Backward compatibility ────────────────────────────

class TestBackwardCompatibility:
    """Existing callers of step() and run_headless() still work."""

    def test_step_default_dt_still_works(self, default_config):
        """step() with no args uses default frame_dt=1/60."""
        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)
        # Just verify no crash
        engine.step()
        assert engine.frame == 1

    def test_run_headless_still_works(self, default_config):
        """run_headless produces the same number of physics steps."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.dt_phys = 1.0 / 60.0
        engine = SimulationEngine(cfg)
        engine.run_headless(steps=5)
        assert engine.frame == 5

    def test_seeded_run_headless_deterministic(self, default_config):
        """Two engines with same seed produce identical positions via accumulator."""
        cfg_a = default_config
        cfg_a.num_boids = 30
        cfg_a.seed = 42
        cfg_a.dt_phys = 1.0 / 60.0

        cfg_b = copy(cfg_a)
        cfg_b.seed = 42

        engine_a = SimulationEngine(cfg_a)
        engine_b = SimulationEngine(cfg_b)

        for _ in range(30):
            engine_a.step(1.0 / 60.0)
            engine_b.step(1.0 / 60.0)

        np.testing.assert_array_almost_equal(
            engine_a.flock.positions, engine_b.flock.positions
        )


# ── P8.10g: _draw_birds_with_lerp None path ───────────────────

@pytest.mark.gpu
class TestDrawBirdsLerpNonePath:
    """When render_positions is None, draw_birds is called without override."""

    def test_none_path_calls_draw_birds_without_override(self, gpu_available):
        """_draw_birds_with_lerp(None) → draw_birds(flock) — no positions_override.

        The P8.10 optimisation sets render_positions=None when the accumulator
        drains cleanly. The visualizer must then call draw_birds(flock) without
        the positions_override kwarg, so the renderer reads flock.positions
        directly instead of using a stale copy.
        """
        from pymurmur.viz.visualizer import Visualizer

        cfg = SimConfig(num_boids=10, dt_phys=1.0/60.0, width=200, height=200, depth=200)
        sim = SimulationEngine(cfg)
        # Run one exact step — accumulator drains cleanly → render_positions=None
        sim.step(1.0/60.0)
        assert sim.render_positions is None  # precondition

        viz = Visualizer(sim, cfg, headless=True, width=100, height=100)

        with patch.object(viz.renderer, "draw_birds", wraps=viz.renderer.draw_birds) as mock_draw:
            viz.headless_frame()

        # Must be called exactly once (single-view, no dual_view)
        assert mock_draw.call_count == 1

        # First positional arg must be the flock
        call = mock_draw.call_args
        assert call is not None
        args, kwargs = call
        assert args[0] is sim.flock

        # No positions_override kwarg
        assert "positions_override" not in kwargs, (
            "draw_birds must NOT receive positions_override when render_positions is None"
        )

    def test_lerp_path_calls_draw_birds_with_override(self, gpu_available):
        """_draw_birds_with_lerp(array) → draw_birds(flock, positions_override=arr).

        When render_positions is a numpy array (partial accumulator), the
        visualizer must pass it as positions_override to the renderer.
        """
        from pymurmur.viz.visualizer import Visualizer

        cfg = SimConfig(num_boids=10, dt_phys=1.0/60.0, width=200, height=200, depth=200)
        sim = SimulationEngine(cfg)
        # Run one full step, then a partial step to get non-None render_positions
        sim.step(1.0/60.0)
        sim.step(0.5/60.0)  # partial accumulator → lerp positions
        assert sim.render_positions is not None  # precondition

        viz = Visualizer(sim, cfg, headless=True, width=100, height=100)

        with patch.object(viz.renderer, "draw_birds", wraps=viz.renderer.draw_birds) as mock_draw:
            viz.headless_frame()

        assert mock_draw.call_count == 1
        _, kwargs = mock_draw.call_args
        assert "positions_override" in kwargs, (
            "draw_birds must receive positions_override when render_positions is set"
        )
        np.testing.assert_array_equal(kwargs["positions_override"], sim.render_positions)

    def test_none_path_no_kwarg_spillage(self, gpu_available):
        """None→lerp→None cycle: kwarg absent when render_positions is None.

        After a partial-step frame produces lerp positions, draining the
        accumulator with another partial step restores render_positions=None
        and the visualizer must not leak positions_override.
        """
        from pymurmur.viz.visualizer import Visualizer

        cfg = SimConfig(num_boids=10, dt_phys=1.0/60.0, width=200, height=200, depth=200)
        sim = SimulationEngine(cfg)
        viz = Visualizer(sim, cfg, headless=True, width=100, height=100)

        # Frame 1: exact dt → None path (acc drained to 0)
        sim.step(1.0/60.0)
        assert sim.render_positions is None
        with patch.object(viz.renderer, "draw_birds") as mock:
            viz.headless_frame()
        _, kw1 = mock.call_args
        assert "positions_override" not in kw1

        # Frame 2: 0.5× dt → lerp path (acc = 0.5/60, no physics step)
        sim.step(0.5/60.0)
        assert sim.render_positions is not None
        with patch.object(viz.renderer, "draw_birds") as mock:
            viz.headless_frame()
        _, kw2 = mock.call_args
        assert "positions_override" in kw2

        # Frame 3: another 0.5× dt → acc = 1.0/60, 1 physics step,
        # acc drained to 0 → back to None path (no spillage)
        sim.step(0.5/60.0)
        assert sim._accumulator < 1e-10
        assert sim.render_positions is None
        with patch.object(viz.renderer, "draw_birds") as mock:
            viz.headless_frame()
        _, kw3 = mock.call_args
        assert "positions_override" not in kw3

    def test_engine_step_followed_by_lerp_draw_consistency(self, gpu_available):
        """render_positions after step() flows correctly into draw_birds.

        Integration: step() → render_positions → _draw_birds_with_lerp →
        draw_birds(). Verifies the full pipe from engine to renderer.
        """
        from pymurmur.viz.visualizer import Visualizer

        cfg = SimConfig(num_boids=20, dt_phys=1.0/60.0, width=200, height=200, depth=200)
        sim = SimulationEngine(cfg)
        viz = Visualizer(sim, cfg, headless=True, width=100, height=100)

        # Step: accumulator clean → render_positions=None
        sim.step(1.0/60.0)
        rpos_before = sim.render_positions
        assert rpos_before is None  # clean drain at exact dt

        with patch.object(viz.renderer, "draw_birds") as mock:
            viz.headless_frame()

        # Verify the visualizer respected the engine's None state
        _, kwargs = mock.call_args
        assert "positions_override" not in kwargs
