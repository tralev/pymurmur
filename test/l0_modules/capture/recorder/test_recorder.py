"""Unit tests for capture/recorder.py — Recorder init, on_frame capture + capture_every gating, save_gif.

Split out of test_recorder.py (file-size split).
"""

from pathlib import Path

import pytest


class TestRecorderInit:
    """Recorder initialisation and basic properties."""

    def test_recorder_importable(self):
        """Recorder is importable."""
        from pymurmur.capture.recorder import Recorder
        assert Recorder is not None

    def test_recorder_init_empty_state(self, default_config):
        """Recorder starts with empty frames and metrics_history."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)
        assert rec.frames == []
        assert rec.metrics_history == []
        assert rec._frame_count == 0


class TestRecorderOnFrame:
    """on_frame() captures metrics and optionally FBO frames."""

    def test_on_frame_appends_metrics(self, default_config):
        """on_frame() appends to metrics_history every call."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_every = 1
        cfg.capture_with_viz = False
        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)

        # Run one step manually, then call on_frame
        sim.step(1.0 / 60)
        rec.on_frame(sim)
        assert len(rec.metrics_history) == 1

        sim.step(1.0 / 60)
        rec.on_frame(sim)
        assert len(rec.metrics_history) == 2

    def test_capture_every_n_respected(self, default_config):
        """on_frame() captures metrics every call, frames every N when with_viz."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_every = 3
        cfg.capture_with_viz = False
        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)

        # Call on_frame 10 times — metrics captured every call
        for _ in range(10):
            sim.step(1.0 / 60)
            rec.on_frame(sim)

        assert len(rec.metrics_history) == 10
        assert rec._frame_count == 10

        # with_viz=False → no frames captured regardless of capture_every
        assert len(rec.frames) == 0

    def test_on_frame_no_viz_metrics_only(self, default_config):
        """with_viz=False captures only metrics, no frames."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_every = 1
        cfg.capture_with_viz = False
        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)

        for _ in range(5):
            sim.step(1.0 / 60)
            rec.on_frame(sim)

        assert len(rec.metrics_history) == 5
        assert len(rec.frames) == 0  # no FBO frames

    def test_on_frame_fbo_exception_silent(self, default_config, monkeypatch):
        """FBO capture exception is caught silently (I6.3: RuntimeError for GPU failure)."""
        import builtins

        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_every = 1
        cfg.capture_with_viz = True  # trigger FBO path
        cfg.capture_prewarm = 0  # G6: without this, _capture_frame is never
        # reached (capture_prewarm defaults to 60) and this test is vacuous —
        # the blocked import is never attempted.
        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)

        # Block viz imports so the try block raises
        orig_import = builtins.__import__
        def _block_viz(name, *args, **kwargs):
            if "viz.visualizer" in name:
                raise ImportError("Blocked for test")
            return orig_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", _block_viz)

        sim.step(1.0 / 60)
        rec.on_frame(sim)  # enters FBO block → import fails → ImportError caught

        # Metrics still captured (before FBO block)
        assert len(rec.metrics_history) == 1
        assert len(rec.frames) == 0  # ImportError is a hard bail (no viz
        # module at all — no mpl fallback attempted either), unlike a
        # RuntimeError from an already-imported renderer (see G6 tests below)

    @pytest.mark.gpu
    def test_on_frame_captures_fbo_frames(self, default_config, gpu_available):
        """on_frame() with with_viz=True captures FBO frames via internal renderer.

        This is the only test that verifies the internal FBO capture path
        SUCCEEDS — all other GPU tests manually create external renderers.
        """
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_every = 1
        cfg.capture_with_viz = True
        cfg.capture_prewarm = 0  # P8.7: capture from the first frame
        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)

        # Step and call on_frame — internal FBO path should succeed
        sim.step(1.0 / 60)
        rec.on_frame(sim)

        assert len(rec.metrics_history) == 1
        assert len(rec.frames) == 1  # internal FBO capture succeeded

        # Second call — should reuse cached renderer
        sim.step(1.0 / 60)
        rec.on_frame(sim)
        assert len(rec.frames) == 2  # cached renderer works too

        # Verify frames are real PIL Images
        assert rec.frames[0] is not None
        assert rec.frames[1] is not None


class TestRecorderSaveGif:
    """save_gif() creates valid GIF output."""

    @pytest.mark.gpu
    def test_save_gif_creates_non_empty_file(self, default_config, tmp_path,
                                              gpu_available):
        """save_gif() with captured frames creates a non-empty .gif."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.renderer import Renderer3D

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_with_viz = True
        cfg.capture_every = 1

        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)

        # Manually capture a few FBO frames
        r = Renderer3D(
            width=cfg.window_width, height=cfg.window_height, headless=True
        )
        cam = OrbitCamera(
            target=(cfg.width / 2, cfg.height / 2, cfg.depth / 2),
        )
        for _ in range(3):
            sim.step(1.0 / 60)
            rec.on_frame(sim)  # captures metrics
            r.begin_frame(cam)
            r.draw_birds(sim.flock)
            r.end_frame()
            img = r.capture_frame()
            if img is not None:
                rec.frames.append(img)

        output = tmp_path / "test.gif"
        result = rec.save_gif(path=str(output))
        assert result is not None
        assert output.exists()
        assert output.stat().st_size > 0

        # Verify GIF magic bytes
        with open(output, "rb") as f:
            assert f.read(3) == b"GIF"

    def test_save_gif_defaults_fps_to_config_capture_fps(self, default_config, tmp_path):
        """C3: save_gif(fps=None) uses config.capture_fps, not a hardcoded 20."""
        from PIL import Image

        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 5
        cfg.capture_fps = 10  # non-default, distinguishable from the old hardcoded 20
        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)
        # Distinct colours so PIL's GIF optimizer doesn't merge identical
        # frames (which would sum their durations into a single frame).
        rec.frames = [
            Image.new("RGB", (8, 8), color=c)
            for c in [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        ]

        output = tmp_path / "fps_default.gif"
        result = rec.save_gif(path=str(output))
        assert result is not None

        with Image.open(output) as gif:
            assert gif.info["duration"] == int(1000 / cfg.capture_fps)

    def test_save_gif_empty_frames_returns_none(self, default_config):
        """save_gif() with zero frames returns None gracefully."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)
        assert rec.save_gif() is None

    @pytest.mark.gpu
    def test_save_gif_fps_duration(self, default_config, tmp_path, gpu_available):
        """save_gif() writes correct frame duration for the given fps."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from PIL import Image

        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.renderer import Renderer3D

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_with_viz = True
        cfg.capture_every = 1

        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)

        r = Renderer3D(width=cfg.window_width, height=cfg.window_height, headless=True)
        cam = OrbitCamera(target=(cfg.width / 2, cfg.height / 2, cfg.depth / 2))
        for _ in range(2):
            sim.step(1.0 / 60)
            r.begin_frame(cam)
            r.draw_birds(sim.flock)
            r.end_frame()
            img = r.capture_frame()
            if img is not None:
                rec.frames.append(img)

        output = tmp_path / "fps_test.gif"
        # Save at 10fps → duration should be 100ms per frame
        rec.save_gif(path=str(output), fps=10)

        with Image.open(output) as gif:
            assert gif.n_frames == 2
            # PIL stores duration in ms; 10fps → 100ms
            assert gif.info.get("duration") == 100

    def test_save_gif_pil_unavailable(self, default_config, monkeypatch):
        """save_gif() returns None when PIL import fails."""
        import sys

        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)
        rec.frames = ["mock_frame"]  # non-empty so we pass the early return

        # Block PIL import
        monkeypatch.setitem(sys.modules, "PIL", None)
        assert rec.save_gif() is None

    def test_save_gif_single_frame_path(self, default_config, tmp_path, monkeypatch):
        """save_gif() takes single-frame branch when len(frames) == 1."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)

        # Create a mock frame with .save() method
        saved_calls = []
        class MockFrame:
            def save(self, path, **kwargs):
                saved_calls.append((path, kwargs))
        rec.frames = [MockFrame()]

        output = tmp_path / "single.gif"
        result = rec.save_gif(path=str(output))
        assert result == str(output)
        assert len(saved_calls) == 1
        # Single frame: .save() called without save_all
        assert "save_all" not in saved_calls[0][1]

    def test_save_gif_config_fallback_path(self, default_config, tmp_path):
        """save_gif() falls back to config.capture_output when path=None."""
        from PIL import Image

        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_output = str(tmp_path / "fallback.gif")

        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)
        # Use a real PIL Image so .resize() works
        rec.frames = [Image.new("RGB", (40, 30))]

        result = rec.save_gif()  # path=None → uses cfg.capture_output
        assert result == cfg.capture_output
        assert Path(result).exists()

    @pytest.mark.gpu
    def test_save_gif_lanczos_downscale(self, default_config, tmp_path, gpu_available):
        """save_gif() downscales frames to half resolution via LANCZOS."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from PIL import Image

        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.renderer import Renderer3D

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_with_viz = True
        cfg.capture_every = 1

        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)

        r = Renderer3D(width=cfg.window_width, height=cfg.window_height, headless=True)
        cam = OrbitCamera(target=(cfg.width / 2, cfg.height / 2, cfg.depth / 2))
        sim.step(1.0 / 60)
        r.begin_frame(cam)
        r.draw_birds(sim.flock)
        r.end_frame()
        img = r.capture_frame()
        if img is not None:
            rec.frames.append(img)

        output = tmp_path / "lanczos.gif"
        rec.save_gif(path=str(output))

        with Image.open(output) as gif:
            # Original was config.window_width × config.window_height
            # LANCZOS halves both dimensions
            assert gif.width == cfg.window_width // 2
            assert gif.height == cfg.window_height // 2


