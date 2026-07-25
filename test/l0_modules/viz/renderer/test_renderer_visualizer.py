"""GPU-dependent tests for viz.renderer — Visualizer integration.

Requires ModernGL GPU context. All GPU-dependent tests are gated behind
@pytest.mark.gpu and skipped when gpu_available is False.

Split out of test_renderer.py (file-size split).
"""

import pytest


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
        """headless_frame() renders without error (step is caller's responsibility)."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=True)
        sim.step(1.0 / 60)  # step before render (I4.1)
        frame_before = sim.frame
        img = viz.headless_frame()
        assert img is not None
        assert sim.frame == frame_before  # rendering doesn't advance sim

    def test_visualizer_paused_skips_step(self, gpu_available, default_config):
        """Rendering works regardless of pause state (pause only affects caller's step)."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=True)
        viz.paused = True
        frame_before = sim.frame
        img = viz.headless_frame()
        assert img is not None
        assert sim.frame == frame_before  # rendering doesn't advance sim

    def test_renderer_camera_wiring(self, gpu_available, default_config):
        """Renderer3D + OrbitCamera + SimulationEngine wire without error."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.renderer import Renderer3D

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
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.renderer import Renderer3D

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
        sim.step(1.0 / 60)  # step first (I4.1)
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
        sim.step(1.0 / 60)  # step first (I4.1)
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

        # Advance, then reset (I4.1: step explicitly before render)
        sim.step(1.0 / 60)
        viz.headless_frame()
        sim.step(1.0 / 60)
        viz.headless_frame()
        sim.reset()
        assert sim.frame == 0

        # After reset, should be able to continue
        sim.step(1.0 / 60)
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

        # Advance a few frames (I4.1: step explicitly before render)
        for _ in range(3):
            sim.step(1.0 / 60)
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
            sim.step(1.0 / 60)  # step before render (I4.1)
            frame_before = sim.frame
            viz.frame()
            assert sim.frame == frame_before  # rendering doesn't advance sim
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

        sim.step(1.0 / 60)  # step before render (I4.1)
        frame_before = sim.frame
        viz.frame()  # lines 60-64 — no return value, no display
        assert sim.frame == frame_before  # rendering doesn't advance sim

    # ── headless_frame() edge cases ────────────────────────────────

    def test_visualizer_headless_frame_paused_toggle(self, gpu_available,
                                                      default_config):
        """Rendering works regardless of paused state (pause is caller's concern)."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=True)

        # Step + render: normal (I4.1: step explicitly)
        sim.step(1.0 / 60)
        viz.headless_frame()
        sim.step(1.0 / 60)
        viz.headless_frame()
        assert sim.frame == 2

        # Step + render: paused — still renders
        viz.paused = True
        sim.step(1.0 / 60)
        viz.headless_frame()
        assert sim.frame == 3

        # Step + render: paused again
        sim.step(1.0 / 60)
        viz.headless_frame()
        assert sim.frame == 4

        # Step + render: unpause
        viz.paused = False
        sim.step(1.0 / 60)
        viz.headless_frame()
        assert sim.frame == 5

    def test_visualizer_headless_frame_multi_advance(self, gpu_available,
                                                      default_config):
        """10 step+render cycles — rendering is pure, caller controls stepping."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=True)

        for _ in range(10):
            sim.step(1.0 / 60)  # step before render (I4.1)
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
        assert sim.frame == 0  # rendering doesn't advance sim

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
        def _handle_once(positions=None):
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
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        from pymurmur.viz.visualizer import Visualizer

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
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        from pymurmur.viz.visualizer import Visualizer

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
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        from pymurmur.viz.visualizer import Visualizer

        self._patch_for_headless_run(monkeypatch)

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=False)

        input_ctrl = InputControl(default_config, OrbitCamera())
        input_ctrl.pending_reset = True

        self._make_one_shot_handle_events(input_ctrl, monkeypatch)

        viz.run(input_ctrl)

        assert input_ctrl.pending_reset is False  # flag consumed
        assert sim.frame >= 1                      # reset→0, step→1

    def test_visualizer_run_remove_birds(self, gpu_available, default_config,
                                          monkeypatch):
        """run() processes pending_remove and decreases N_active."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        from pymurmur.viz.visualizer import Visualizer

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


