"""GPU-dependent tests for viz.renderer — P10.3 HUD GL helpers, P8.6
quality governor degradation/recovery actions.

Requires ModernGL GPU context. All GPU-dependent tests are gated behind
@pytest.mark.gpu and skipped when gpu_available is False.

Split out of test_renderer.py (file-size split). D17 headless FBO
depth and G6 GPU context loss tests moved to
test_renderer_fbo_context_loss.py (file-size split of this file).
"""

import pytest

# ── P10.3 HUD GL helpers (2026-07-19 audit gap) ──────────────────

@pytest.mark.gpu
class TestHudGLHelpers:
    """draw_hud_rect / hud_begin / hud_end execute without GL errors."""

    def test_hud_rect_renders(self, gpu_available):
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.renderer import Renderer3D

        r = Renderer3D(width=128, height=128, headless=True)
        r.begin_frame(OrbitCamera())
        r.hud_begin()
        r.draw_hud_rect(10, 10, 40, 12, (0.8, 0.2, 0.2))
        r.draw_hud_rect(0, 0, 128, 4, (0.2, 0.8, 0.2))
        r.hud_end()
        r.end_frame()
        img = r.capture_frame()
        assert img is not None

    def test_hud_begin_resets_viewport_to_full_window(self, gpu_available):
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.renderer import Renderer3D

        r = Renderer3D(width=128, height=96, headless=True)
        r.hud_begin()
        assert r.ctx.viewport == (0, 0, 128, 96)


# ── P8.6: Quality governor degradation/recovery actions ─────────

@pytest.mark.gpu
class TestQualityGovernorActions:
    """P8.6: _apply_quality_actions() degradation ladder and recovery.

    Covers all 3 degradation levels (trails→scale→count) and both
    recovery levels (scale restore→full heal).  Uses monkeypatched
    governor methods to trigger actions without waiting for real
    frame-timing windows (1.8 s degrade, 3.6 s recover).
    """

    @pytest.fixture
    def viz_with_mock_gov(self, gpu_available, default_config):
        """Visualizer with a mockable QualityGovernor."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=True)
        return viz

    # ── Degradation ─────────────────────────────────────────────

    def test_degrade_level_1_disables_trails(self, viz_with_mock_gov, monkeypatch):
        """Level 1 degradation → disable_trails() called on renderer."""
        viz = viz_with_mock_gov

        # Track whether disable_trails was called
        called = []
        monkeypatch.setattr(viz.renderer, "disable_trails",
                            lambda: called.append(True))

        # Configure governor internals: level=1, should_degrade=True
        viz._governor._degradation_level = 1
        viz._governor.should_degrade = lambda: True
        viz._governor.should_recover = lambda: False

        viz._apply_quality_actions()
        assert len(called) == 1, "disable_trails must be called on level 1"

    def test_degrade_level_2_reduces_render_scale(self, viz_with_mock_gov, monkeypatch):
        """Level 2 degradation → render_scale reduced by RENDER_SCALE_STEP (0.15)."""
        viz = viz_with_mock_gov
        original_scale = viz.renderer.render_scale
        step = viz._governor.RENDER_SCALE_STEP

        viz._governor._degradation_level = 2
        viz._governor.should_degrade = lambda: True
        viz._governor.should_recover = lambda: False

        viz._apply_quality_actions()
        assert viz.renderer.render_scale == pytest.approx(original_scale - step)

    def test_degrade_level_2_floor_respected(self, viz_with_mock_gov, monkeypatch):
        """Level 2 won't reduce scale below RENDER_SCALE_FLOOR (0.75)."""
        viz = viz_with_mock_gov
        floor = viz._governor.RENDER_SCALE_FLOOR
        # Start just above floor
        viz.renderer.render_scale = floor + 0.05

        viz._governor._degradation_level = 2
        viz._governor.should_degrade = lambda: True
        viz._governor.should_recover = lambda: False

        viz._apply_quality_actions()
        assert viz.renderer.render_scale == pytest.approx(floor)

    def test_degrade_level_3_enqueues_remove(self, monkeypatch):
        """Level 3 degradation → enqueue_remove called with ~18% of flock.

        Uses a dedicated flock with 1000 birds so the count-floor (512)
        doesn't gate the removal path.
        """
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        cfg = SimConfig(num_boids=1000, seed=1)
        sim = SimulationEngine(cfg)
        viz = Visualizer(sim, cfg, headless=True)
        N = viz.sim.flock.N_active
        frac = viz._governor.COUNT_STEP_FRACTION
        expected_remove = int(N * frac)  # 180 birds at 18%

        viz._governor._degradation_level = 3
        viz._governor.should_degrade = lambda: True
        viz._governor.should_recover = lambda: False

        viz._apply_quality_actions()
        viz.sim.drain_commands()
        assert viz.sim.flock.N_active == N - expected_remove

    def test_degrade_level_3_count_floor_respected(self, viz_with_mock_gov, monkeypatch):
        """Level 3 won't reduce below COUNT_FLOOR (512)."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        # Use a small flock that would go below floor at −18%
        cfg = SimConfig(num_boids=600, seed=1)
        sim = SimulationEngine(cfg)
        viz = Visualizer(sim, cfg, headless=True)
        floor = viz._governor.COUNT_FLOOR

        viz._governor._degradation_level = 3
        viz._governor.should_degrade = lambda: True
        viz._governor.should_recover = lambda: False

        viz._apply_quality_actions()
        viz.sim.drain_commands()
        assert viz.sim.flock.N_active == floor

    # ── Recovery ────────────────────────────────────────────────

    def test_recover_level_1_increases_render_scale(self, viz_with_mock_gov, monkeypatch):
        """Recovery from level 2→1 restores render_scale by one step.

        render_scale setter clamps to [0.75, 1.0], so we start at 0.80
        (a safe margin above the floor) and expect 0.80+0.15=0.95.
        """
        viz = viz_with_mock_gov
        step = viz._governor.RENDER_SCALE_STEP
        viz.renderer.render_scale = 0.80  # above floor, below ceiling

        # Patch both governor methods to guarantee recovery path
        viz._governor.should_degrade = lambda: False
        viz._governor.should_recover = lambda: True
        viz._governor._degradation_level = 1

        viz._apply_quality_actions()
        assert viz.renderer.render_scale == pytest.approx(0.80 + step)

    def test_recover_level_1_scale_capped_at_1(self, viz_with_mock_gov, monkeypatch):
        """Recovery won't raise render_scale above 1.0."""
        viz = viz_with_mock_gov
        viz.renderer.render_scale = 0.95

        viz._governor.should_degrade = lambda: False
        viz._governor.should_recover = lambda: True
        viz._governor._degradation_level = 1

        viz._apply_quality_actions()
        assert viz.renderer.render_scale == pytest.approx(1.0)

    def test_recover_level_0_full_heal(self, viz_with_mock_gov, monkeypatch):
        """Recovery to level 0 resets render_scale to 1.0 + re-enables trails."""
        viz = viz_with_mock_gov
        viz.renderer.render_scale = 0.75
        # Default config has trails='off' — override to test enable_trails path
        viz.config.trails = "velocity"

        viz._governor.should_degrade = lambda: False
        viz._governor.should_recover = lambda: True
        viz._governor._degradation_level = 0

        # Track enable_trails call
        enable_calls = []
        monkeypatch.setattr(viz.renderer, "enable_trails",
                            lambda m, _l: enable_calls.append((m, _l)))

        viz._apply_quality_actions()
        assert viz.renderer.render_scale == pytest.approx(1.0)
        assert len(enable_calls) == 1, "enable_trails must be called on full heal"

    def test_recover_level_0_no_trails_when_config_off(self, viz_with_mock_gov, monkeypatch):
        """Full heal skips enable_trails when config.trails is 'off'."""
        viz = viz_with_mock_gov
        viz.config.trails = "off"  # explicitly off — default is already off
        viz.renderer.render_scale = 0.75

        viz._governor.should_degrade = lambda: False
        viz._governor.should_recover = lambda: True
        viz._governor._degradation_level = 0

        enable_calls = []
        monkeypatch.setattr(viz.renderer, "enable_trails",
                            lambda m, _l: enable_calls.append((m, _l)))

        viz._apply_quality_actions()
        assert viz.renderer.render_scale == pytest.approx(1.0)
        assert len(enable_calls) == 0, "enable_trails skipped when config.trails='off'"

    # ── No-op cases ─────────────────────────────────────────────

    def test_no_action_when_neither_degrade_nor_recover(self, viz_with_mock_gov, monkeypatch):
        """Neither should_degrade nor should_recover → no side effects."""
        viz = viz_with_mock_gov
        viz._governor.should_degrade = lambda: False
        viz._governor.should_recover = lambda: False

        # Capture any calls
        disable_calls = []
        monkeypatch.setattr(viz.renderer, "disable_trails",
                            lambda: disable_calls.append(True))
        monkeypatch.setattr(viz.renderer, "enable_trails",
                            lambda m, _l: disable_calls.append(True))

        viz.renderer.render_scale = 0.85
        viz._apply_quality_actions()
        # No change to render_scale, no trail calls
        assert viz.renderer.render_scale == 0.85
        assert len(disable_calls) == 0

