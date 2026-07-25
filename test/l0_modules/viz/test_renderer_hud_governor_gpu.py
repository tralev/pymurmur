"""GPU-dependent tests for viz.renderer — P10.3 HUD GL helpers, P8.6 quality governor degradation/recovery actions, D17 headless FBO depth, G6 GPU context loss graceful degradation.

Requires ModernGL GPU context. All GPU-dependent tests are gated behind
@pytest.mark.gpu and skipped when gpu_available is False.

Split out of test_renderer.py (file-size split).
"""

import numpy as np
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


# ═══════════════════════════════════════════════════════════════════
# D17: Headless FBO depth attachment
# ═══════════════════════════════════════════════════════════════════


class TestD17HeadlessFBODepth:
    """D17: Headless FBO has depth renderbuffer so overlapping birds
    resolve correctly (nearer wins) rather than by draw order."""

    def test_headless_renderer_has_depth_attachment(self):
        """D17: Renderer3D(headless=True) adds depth_renderbuffer to FBO."""
        import inspect

        from pymurmur.viz.renderer import Renderer3D
        src = inspect.getsource(Renderer3D.__init__)
        assert "depth_renderbuffer" in src, (
            "Headless FBO must create a depth_renderbuffer"
        )
        assert "depth_attachment" in src, (
            "Headless FBO must have depth_attachment parameter"
        )

    def test_headless_fbo_code_path_has_depth(self):
        """D17: The headless FBO creation block includes depth attachment."""
        import inspect

        from pymurmur.viz.renderer import Renderer3D

        # Extract the headless FBO creation section
        src = inspect.getsource(Renderer3D.__init__)
        # Find the headless block
        headless_start = src.find("# Headless FBO")
        assert headless_start > 0, "Headless FBO comment not found"
        headless_section = src[headless_start:]
        fbo_end = headless_section.find("else:")
        headless_block = headless_section[:fbo_end] if fbo_end > 0 else headless_section

        assert "depth_renderbuffer" in headless_block, (
            f"Headless FBO block must contain depth_renderbuffer:\n{headless_block}"
        )
        assert "depth_attachment" in headless_block, (
            f"Headless FBO block must contain depth_attachment:\n{headless_block}"
        )
        assert "color_attachments" in headless_block, (
            f"Headless FBO block must still have color_attachments:\n{headless_block}"
        )

    def test_depth_rb_attribute_stored(self):
        """D17: _depth_rb is stored as instance attribute for lifetime management."""
        import inspect

        from pymurmur.viz.renderer import Renderer3D

        src = inspect.getsource(Renderer3D.__init__)
        assert "self._depth_rb" in src, (
            "Headless FBO depth renderbuffer must be stored as self._depth_rb"
        )


# ── G6: GPU context loss → graceful degradation ───────────────

class TestG6GLContextLoss:
    """G6: Losing GL context mid-render degrades to headless/mpl
    fallback instead of crashing.  Uses monkeypatch to simulate
    GL context loss and verifies clean degradation + warning."""

    def test_gl_loss_flag_initial_false(self):
        """gl_lost is False by default."""
        from pymurmur.viz.renderer import Renderer3D
        r = Renderer3D(800, 600, headless=True)
        assert r.gl_lost is False

    def test_simulate_gl_loss_sets_flag(self):
        """simulate_gl_loss() sets gl_lost=True."""
        from pymurmur.viz.renderer import Renderer3D
        r = Renderer3D(800, 600, headless=True)
        r.simulate_gl_loss()
        assert r.gl_lost is True

    def test_headless_frame_survives_gl_loss(self, default_config):
        """G6: headless_frame() returns a PIL Image even after
        GL context loss (blank fallback), without raising."""
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=True)

        # Simulate GL loss
        viz.renderer.simulate_gl_loss()
        viz._gl_warned = False  # reset for warning detection

        # headless_frame should return a PIL Image without crashing
        img = viz.headless_frame()
        from PIL import Image
        assert isinstance(img, Image.Image), (
            f"headless_frame must return PIL Image after GL loss, got {type(img)}"
        )
        assert img.size == (default_config.window_width, default_config.window_height)

    def test_frame_survives_gl_loss_no_crash(self, default_config):
        """G6: frame() does not crash when GL context is lost."""

        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=True)

        # Set gl_lost flag before frame() — should return early
        viz.renderer.gl_lost = True

        # frame() should not crash
        viz.frame()  # no exception

    def test_fallback_warning_emitted_once(self, default_config):
        """G6: RuntimeWarning is emitted with "GPU context lost" message
        on the first frame after GL loss, and only once."""
        import warnings
        from unittest.mock import patch

        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=True)

        # Simulate GL loss by making begin_frame raise
        def raise_gl_error(*args, **kwargs):
            viz.renderer.gl_lost = True  # so subsequent frames skip
            raise RuntimeError("Mock GL context loss")

        with patch.object(viz.renderer, "begin_frame", side_effect=raise_gl_error):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                # First frame — should emit warning
                viz.headless_frame()
                # Second frame — gl_lost is already True, no new warning
                viz.headless_frame()

                gl_warnings = [x for x in w
                               if "GPU context lost" in str(x.message)]
                assert len(gl_warnings) == 1, (
                    f"Expected exactly 1 GPU context lost warning, "
                    f"got {len(gl_warnings)}"
                )
                assert issubclass(gl_warnings[0].category, RuntimeWarning)

    @pytest.mark.parametrize("exc_type", [TypeError, AttributeError, ImportError, NameError])
    def test_programming_errors_are_not_swallowed_as_gl_loss(self, default_config, exc_type):
        """G6: A real programming error (bug in the render call, not a
        GPU/driver failure) propagates instead of being silently
        treated as GL context loss.

        `_render_safe`'s whitelist (`isinstance(e, (AttributeError,
        TypeError, ImportError, NameError))`) exists specifically so a
        genuine bug doesn't get masked as "graceful GPU degradation" —
        this was previously implemented but untested.
        """
        from unittest.mock import patch

        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=True)

        def raise_programming_error(*args, **kwargs):
            raise exc_type("a real bug, not a GPU failure")

        with patch.object(viz.renderer, "begin_frame", side_effect=raise_programming_error):
            with pytest.raises(exc_type, match="a real bug"):
                viz.headless_frame()

        # Must NOT have been treated as GL loss
        assert viz.renderer.gl_lost is False

    def test_simulation_continues_after_gl_loss(self, default_config):
        """G6: Simulation physics continue after GL context loss.
        Rendering is skipped but engine.step() still works."""
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        default_config.num_boids = 20
        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=True)

        # Run a few normal frames
        for _ in range(3):
            sim.step()
            viz.headless_frame()

        frame_after_normal = sim.frame
        pos_after_normal = sim.flock.positions.copy()

        # Simulate GL loss
        viz.renderer.simulate_gl_loss()

        # Run more frames — physics should still advance
        for _ in range(3):
            sim.step()
            viz.headless_frame()  # should return blank, not crash

        frame_after_loss = sim.frame
        assert frame_after_loss == frame_after_normal + 3, (
            "Frame counter must advance after GL loss"
        )
        # Positions should have changed (physics still running)
        assert not np.array_equal(pos_after_normal, sim.flock.positions), (
            "Positions must change — physics continues after GL loss"
        )
