"""GPU-dependent tests for viz.renderer — Renderer3D, Visualizer.

Requires ModernGL GPU context. All tests are gated behind
@pytest.mark.gpu and skipped when gpu_available is False.
"""

import pytest


@pytest.mark.gpu
class TestRenderer3D:
    """Tests requiring a ModernGL GPU context (standalone or windowed)."""

    def test_renderer_init(self, gpu_available):
        """Renderer3D(width, height) creates context without error."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.renderer import Renderer3D
        r = Renderer3D(800, 600, headless=True)
        assert r is not None

    def test_renderer_headless_init(self, gpu_available):
        """Renderer3D(width, height, headless=True) creates FBO."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.renderer import Renderer3D
        r = Renderer3D(800, 600, headless=True)
        assert r.headless is True

    def test_renderer_update_instances(self, gpu_available, small_flock):
        """update_instances() returns correct active count."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.renderer import Renderer3D
        r = Renderer3D(800, 600, headless=True)
        count = r.update_instances(small_flock)
        assert count == small_flock.N_active

    def test_renderer_begin_frame(self, gpu_available):
        """begin_frame(camera) clears and computes matrices."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.renderer import Renderer3D
        from pymurmur.viz.camera import OrbitCamera
        r = Renderer3D(800, 600, headless=True)
        cam = OrbitCamera()
        r.begin_frame(cam)
        # No error = pass

    def test_renderer_draw_birds_no_error(self, gpu_available, small_flock):
        """draw_birds(flock) completes without GL error."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.renderer import Renderer3D
        from pymurmur.viz.camera import OrbitCamera
        r = Renderer3D(800, 600, headless=True)
        r.begin_frame(OrbitCamera())
        r.draw_birds(small_flock)
        r.end_frame()

    def test_renderer_draw_grid_no_error(self, gpu_available):
        """draw_grid() completes without GL error."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.renderer import Renderer3D
        from pymurmur.viz.camera import OrbitCamera
        r = Renderer3D(800, 600, headless=True)
        r.begin_frame(OrbitCamera())
        r.draw_grid()
        r.end_frame()

    def test_renderer_capture_frame(self, gpu_available):
        """capture_frame() returns a PIL Image with correct dimensions."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.renderer import Renderer3D
        from pymurmur.viz.camera import OrbitCamera
        r = Renderer3D(800, 600, headless=True)
        r.begin_frame(OrbitCamera())
        r.end_frame()
        img = r.capture_frame()
        assert img is not None
        assert img.size == (800, 600)

    def test_renderer_buffer_growth(self, gpu_available):
        """Adding more birds than max_instances triggers growth."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.renderer import Renderer3D
        from pymurmur.physics.flock import PhysicsFlock
        from pymurmur.core.config import SimConfig

        r = Renderer3D(800, 600, headless=True, instance_buffer_chunk=10)
        old_max = r._max_instances

        # Create a flock larger than the initial chunk
        cfg = SimConfig()
        cfg.num_boids = 25  # > chunk of 10
        flock = PhysicsFlock(cfg)
        n = r.update_instances(flock)

        assert n == 25
        assert r._max_instances > old_max  # buffer grew

    def test_renderer_windowed_context(self, gpu_available):
        """Renderer3D creates a windowed (non-headless) context."""
        if not gpu_available:
            pytest.skip("GPU not available")
        import os
        if not os.environ.get("DISPLAY"):
            pytest.skip("No display available for windowed context")
        from pymurmur.viz.renderer import Renderer3D
        try:
            r = Renderer3D(800, 600, headless=False)
            assert r.headless is False
            assert r._fbo is None
        except Exception:
            pytest.skip("Windowed context creation failed (no display)")

    def test_renderer_windowed_branches_ci(self, gpu_available, monkeypatch):
        """Cover windowed init branches (lines 49, 90) without a display.

        Monkeypatches moderngl.create_context to use a standalone context
        while still passing headless=False. This exercises the else branches
        for context creation and FBO init.
        """
        if not gpu_available:
            pytest.skip("GPU not available")
        import moderngl
        from pymurmur.viz.renderer import Renderer3D

        # Use a real standalone context to satisfy the windowed code path
        real_ctx = moderngl.create_context(standalone=True, require=330)

        def _mock_create(standalone=False, require=330):
            return real_ctx

        monkeypatch.setattr(moderngl, "create_context", _mock_create)

        r = Renderer3D(800, 600, headless=False)
        assert r.headless is False       # line 45 branch
        assert r._fbo is None            # line 90 (else: no FBO in windowed)
        assert r.ctx is real_ctx         # line 49 (windowed context creation)

    @pytest.mark.skip(reason="Not directly testable — single memcpy is an implementation detail")
    def test_renderer_single_memcpy(self):
        """update_instances() uses vbo.write() with a single call."""
        pass

    def test_renderer_zero_birds(self, gpu_available):
        """Rendering with 0 active birds doesn't crash."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.renderer import Renderer3D
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.physics.flock import PhysicsFlock
        from pymurmur.core.config import SimConfig
        cfg = SimConfig()
        cfg.num_boids = 0
        flock = PhysicsFlock(cfg)
        r = Renderer3D(800, 600, headless=True)
        r.begin_frame(OrbitCamera())
        r.draw_birds(flock)
        r.end_frame()


@pytest.mark.gpu
class TestVisualizerIntegration:
    """Integration tests for the Visualizer + renderer + camera wiring."""

    def test_visualizer_init(self, gpu_available, default_config):
        """Visualizer(sim, config) creates renderer, camera."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=True)
        assert viz.renderer is not None
        assert viz.camera is not None
        assert viz.paused is False

    def test_visualizer_headless_frame(self, gpu_available, default_config):
        """headless_frame() returns a PIL Image."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=True)
        img = viz.headless_frame()
        assert img is not None
        assert img.size == (default_config.window_width, default_config.window_height)

    def test_visualizer_run_one_frame(self, gpu_available, default_config):
        """headless_frame() processes at least one simulation step."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        sim = SimulationEngine(default_config)
        frame_before = sim.frame
        viz = Visualizer(sim, default_config, headless=True)
        viz.headless_frame()
        assert sim.frame == frame_before + 1

    def test_visualizer_paused_skips_step(self, gpu_available, default_config):
        """Paused visualizer does not advance the simulation."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=True)
        viz.paused = True
        frame_before = sim.frame
        viz.headless_frame()
        assert sim.frame == frame_before  # no step taken

    def test_renderer_camera_wiring(self, gpu_available, default_config):
        """Renderer3D + OrbitCamera + SimulationEngine wire without error."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.renderer import Renderer3D
        from pymurmur.viz.camera import OrbitCamera

        sim = SimulationEngine(default_config)
        renderer = Renderer3D(800, 600, headless=True)
        camera = OrbitCamera()
        renderer.begin_frame(camera)
        sim.step(1.0 / 60)
        renderer.draw_birds(sim.flock)
        renderer.end_frame()

    def test_headless_frame_capture(self, gpu_available, default_config):
        """headless FBO readback produces PIL Image."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.renderer import Renderer3D
        from pymurmur.viz.camera import OrbitCamera

        sim = SimulationEngine(default_config)
        r = Renderer3D(800, 600, headless=True)
        camera = OrbitCamera()
        r.begin_frame(camera)
        r.draw_birds(sim.flock)
        r.end_frame()
        img = r.capture_frame()
        assert img is not None

    def test_visualizer_add_birds_integration(self, gpu_available, default_config):
        """Flock add_boids + Visualizer integration: no crash, N_active increases."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        sim = SimulationEngine(default_config)
        old_n = sim.flock.N_active
        viz = Visualizer(sim, default_config, headless=True)

        # Simulate main-loop deferred add: pending_add → flock.add_boids
        pending = 5
        added = sim.flock.add_boids(pending, default_config)
        default_config.num_boids = sim.flock.N_active
        pending -= added

        assert sim.flock.N_active == old_n + 5
        assert pending == 0  # all were added

        # Render a frame after add — must not crash
        viz.headless_frame()

    def test_visualizer_remove_birds_integration(self, gpu_available, default_config):
        """Flock remove_boids + Visualizer integration: no crash, N_active decreases."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        sim = SimulationEngine(default_config)
        old_n = sim.flock.N_active
        viz = Visualizer(sim, default_config, headless=True)

        # Simulate main-loop deferred remove: pending_remove → flock.remove_boids
        pending = 5
        removed = sim.flock.remove_boids(pending)
        default_config.num_boids = sim.flock.N_active
        pending -= removed

        assert sim.flock.N_active == old_n - 5
        assert pending == 0  # all were removed

        # Render a frame after remove — must not crash
        viz.headless_frame()

    def test_visualizer_reset_then_step(self, gpu_available, default_config):
        """After reset, sim can continue stepping without error."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=True)

        # Advance, then reset
        viz.headless_frame()
        viz.headless_frame()
        sim.reset()
        assert sim.frame == 0

        # After reset, should be able to continue
        viz.headless_frame()
        assert sim.frame == 1

    def test_visualizer_sim_reset(self, gpu_available, default_config):
        """sim.reset() restores frame counter to 0 and keeps flock."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=True)

        # Advance a few frames
        for _ in range(3):
            viz.headless_frame()
        assert sim.frame == 3

        # Simulate what main loop does when pending_reset is True
        sim.reset()
        assert sim.frame == 0
        assert sim.flock.N_active == default_config.num_boids

    def test_visualizer_windowed_frame(self, gpu_available, default_config):
        """frame() renders to screen in windowed mode (lines 60-64)."""
        if not gpu_available:
            pytest.skip("GPU not available")
        import os
        if not os.environ.get("DISPLAY"):
            pytest.skip("No display available for windowed context")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        sim = SimulationEngine(default_config)
        try:
            viz = Visualizer(sim, default_config, headless=False)
            frame_before = sim.frame
            viz.frame()
            assert sim.frame == frame_before + 1
        except Exception:
            pytest.skip("Windowed context creation failed (no display)")

    # ── frame() coverage via monkeypatch ──────────────────────────

    def test_visualizer_frame_headless_bypass(self, gpu_available, default_config,
                                               monkeypatch):
        """Cover frame() (lines 60-64) without a display via monkeypatched context."""
        if not gpu_available:
            pytest.skip("GPU not available")
        import moderngl
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        # Use standalone context for the windowed path
        real_ctx = moderngl.create_context(standalone=True, require=330)
        monkeypatch.setattr(moderngl, "create_context",
                            lambda standalone=False, require=330: real_ctx)

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=False)
        assert viz.renderer.headless is False

        frame_before = sim.frame
        viz.frame()  # lines 60-64 — no return value, no display
        assert sim.frame == frame_before + 1

    # ── headless_frame() edge cases ────────────────────────────────

    def test_visualizer_headless_frame_paused_toggle(self, gpu_available,
                                                      default_config):
        """Toggle paused mid-sequence: frames resume when unpaused."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=True)

        # Step 1-2: normal
        viz.headless_frame()
        viz.headless_frame()
        assert sim.frame == 2

        # Step 3: paused — no advance
        viz.paused = True
        viz.headless_frame()
        assert sim.frame == 2

        # Step 4: paused again — still no advance
        viz.headless_frame()
        assert sim.frame == 2

        # Step 5: unpause — advance resumes
        viz.paused = False
        viz.headless_frame()
        assert sim.frame == 3

    def test_visualizer_headless_frame_multi_advance(self, gpu_available,
                                                      default_config):
        """10 consecutive headless_frame() calls advance frame counter by 10."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=True)

        for _ in range(10):
            img = viz.headless_frame()
            assert img is not None

        assert sim.frame == 10

    def test_visualizer_headless_frame_zero_birds(self, gpu_available):
        """headless_frame() with 0 birds doesn't crash."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        cfg = SimConfig()
        cfg.num_boids = 0
        sim = SimulationEngine(cfg)
        viz = Visualizer(sim, cfg, headless=True)

        img = viz.headless_frame()
        assert img is not None
        assert sim.frame == 1  # step still advances even with 0 birds

    # ── __init__ property validation ───────────────────────────────

    def test_visualizer_camera_target_from_config(self, gpu_available, default_config):
        """Camera target is centred on the simulation volume."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=True)

        tx, ty, tz = viz.camera.target
        assert tx == default_config.width / 2
        assert ty == default_config.height / 2
        assert tz == default_config.depth / 2

    def test_visualizer_buffer_chunk_passthrough(self, gpu_available, default_config):
        """instance_buffer_chunk from config is passed to Renderer3D."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        default_config.instance_buffer_chunk = 7777
        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=True)

        assert viz.renderer._chunk == 7777
        assert viz.renderer._max_instances == 7777

    # ── run() loop body via monkeypatched pygame ──────────────────

    # ── helpers for run() tests ───────────────────────────────────

    @staticmethod
    def _patch_for_headless_run(monkeypatch):
        """Monkeypatch moderngl + pygame so run() works without a display."""
        import moderngl
        import pygame

        ctx = moderngl.create_context(standalone=True, require=330)
        monkeypatch.setattr(moderngl, "create_context",
                            lambda standalone=False, require=330: ctx)
        monkeypatch.setattr(pygame, "init", lambda: None)
        monkeypatch.setattr(pygame.display, "set_mode", lambda *a, **kw: None)
        monkeypatch.setattr(pygame.display, "set_caption", lambda *a: None)
        monkeypatch.setattr(pygame.display, "flip", lambda: None)
        monkeypatch.setattr(pygame, "quit", lambda: None)
        monkeypatch.setattr(pygame.time, "Clock",
                            lambda: type("C", (), {"tick": lambda s, f: 16})())

    @staticmethod
    def _make_one_shot_handle_events(input_ctrl, monkeypatch):
        """Make handle_events return True once, then False (exit loop)."""
        calls = [0]
        def _handle_once():
            calls[0] += 1
            return calls[0] == 1
        monkeypatch.setattr(input_ctrl, "handle_events", _handle_once)
        return calls

    # ── run() loop body via monkeypatched pygame ──────────────────

    def test_visualizer_run_one_cycle(self, gpu_available, default_config,
                                       monkeypatch):
        """Exercise run() loop body once via monkeypatched pygame + moderngl.

        Covers lines 68-113: deferred add, reset, grid toggle,
        paused stepping, metrics title — all without a display.
        """
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl

        self._patch_for_headless_run(monkeypatch)

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=False)
        camera = OrbitCamera()

        input_ctrl = InputControl(default_config, camera)
        input_ctrl.pending_add = 5       # test deferred add
        input_ctrl.pending_reset = True  # test reset (processed before step)
        input_ctrl.show_grid = True      # test grid branch (lines 94-95)

        calls = self._make_one_shot_handle_events(input_ctrl, monkeypatch)

        viz.run(input_ctrl)

        # Reset consumed, then step advanced frame
        assert input_ctrl.pending_reset is False
        assert sim.frame >= 1

        # Deferred add: all 5 birds added (fresh flock after reset)
        assert input_ctrl.pending_add == 0

        # Loop exited after one iteration
        assert calls[0] == 2  # first=True (run), second=False (exit)

    def test_visualizer_run_paused_no_step(self, gpu_available, default_config,
                                            monkeypatch):
        """run() with input_ctrl.paused=True skips sim.step but still renders."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl

        self._patch_for_headless_run(monkeypatch)

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=False)
        frame_before = sim.frame

        input_ctrl = InputControl(default_config, OrbitCamera())
        input_ctrl.paused = True

        calls = self._make_one_shot_handle_events(input_ctrl, monkeypatch)

        viz.run(input_ctrl)

        assert sim.frame == frame_before  # paused → no step
        assert calls[0] == 2

    def test_visualizer_run_reset_handling(self, gpu_available, default_config,
                                            monkeypatch):
        """run() resets simulation when pending_reset is True."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl

        self._patch_for_headless_run(monkeypatch)

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=False)

        input_ctrl = InputControl(default_config, OrbitCamera())
        input_ctrl.pending_reset = True

        calls = self._make_one_shot_handle_events(input_ctrl, monkeypatch)

        viz.run(input_ctrl)

        assert input_ctrl.pending_reset is False  # flag consumed
        assert sim.frame >= 1                      # reset→0, step→1

    def test_visualizer_run_remove_birds(self, gpu_available, default_config,
                                          monkeypatch):
        """run() processes pending_remove and decreases N_active."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl

        self._patch_for_headless_run(monkeypatch)

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=False)
        old_n = sim.flock.N_active

        input_ctrl = InputControl(default_config, OrbitCamera())
        input_ctrl.pending_remove = 3           # test deferred remove

        calls = self._make_one_shot_handle_events(input_ctrl, monkeypatch)

        viz.run(input_ctrl)

        assert input_ctrl.pending_remove == 0   # all 3 removed (lines 82-84)
        assert sim.flock.N_active == old_n - 3
        assert calls[0] == 2
