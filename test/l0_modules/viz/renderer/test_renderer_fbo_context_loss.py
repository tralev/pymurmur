"""GPU-dependent tests for viz.renderer — D17 headless FBO depth,
G6 GPU context loss graceful degradation.

Requires ModernGL GPU context. GPU-dependent tests are gated behind
@pytest.mark.gpu and skipped when gpu_available is False.

Split out of test_renderer_hud_governor_gpu.py (file-size split) —
HUD GL helpers and quality-governor action tests stay in the original.
"""

import numpy as np
import pytest

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

    @pytest.mark.gpu
    def test_nearer_bird_wins_depth_test(self, gpu_available):
        """D17: with two birds on the camera's optical axis at different
        depths, the pixel at their shared screen location shows the
        nearer bird's colour — not the farther bird's, and not whichever
        was submitted last in the instance buffer."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.renderer import Renderer3D

        # bird_mesh="tetra": forced opaque geometry, no distance-fade —
        # the default "auto" mesh resolves to point-sprite impostors for
        # small N, whose depth-cue opacity fade (P8.2) makes a sufficiently
        # far bird nearly transparent and confounds a pure depth-test signal.
        cfg = SimConfig(num_boids=2, boid_size=20.0, bird_mesh="tetra")
        flock = PhysicsFlock(cfg)
        # Both on the camera's optical axis (Y=Z=0) -> same screen pixel
        # regardless of depth. far bird is index 1 (submitted after near
        # in the single instanced draw call) so a broken depth test would
        # show the far bird winning by draw order, not just coincidentally
        # agreeing with the correct answer.
        flock.positions[0] = [900.0, 0.0, 0.0]   # near
        flock.velocities[0] = [0.0, 1.0, 0.0]    # heading-theme hue A
        flock.positions[1] = [700.0, 0.0, 0.0]   # far
        flock.velocities[1] = [1.0, 0.0, 0.0]    # heading-theme hue B

        def render(active_mask):
            flock.active[:] = active_mask
            r = Renderer3D(width=200, height=200, headless=True,
                           point_sprites=False, theme="heading", bird_mesh="tetra")
            cam = OrbitCamera(target=(0.0, 0.0, 0.0))
            cam.distance, cam.elevation, cam.azimuth = 1000.0, 0.0, 0.0
            r.begin_frame(cam)
            r.draw_birds(flock)
            r.end_frame()
            img = r.capture_frame()
            pixel = img.getpixel((100, 100))
            r.release()
            return pixel

        color_near_alone = render([True, False])
        color_far_alone = render([False, True])
        assert color_near_alone != color_far_alone, (
            "sanity check: the two heading colours must be distinguishable"
        )

        color_combined = render([True, True])
        assert color_combined == color_near_alone, (
            f"D17: nearer bird must win the depth test at the shared pixel — "
            f"got {color_combined}, expected the near bird's colour "
            f"{color_near_alone} (far bird's was {color_far_alone})"
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
