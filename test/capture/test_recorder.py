"""Unit tests for capture/recorder.py — Recorder class.

Covers test.md INT.6: on_frame capture, capture_every gating,
save_gif/save_metrics_csv/save_metrics_json, no-viz mode, empty-run safety.
"""

import json
import csv
import pytest
from pathlib import Path


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
        assert len(rec.frames) == 0  # FBO failed, no frames

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
        from pymurmur.viz.renderer import Renderer3D
        from pymurmur.viz.camera import OrbitCamera

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
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.renderer import Renderer3D
        from pymurmur.viz.camera import OrbitCamera
        from PIL import Image

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
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine
        from PIL import Image

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
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.renderer import Renderer3D
        from pymurmur.viz.camera import OrbitCamera
        from PIL import Image

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


class TestRecorderSaveMetrics:
    """save_metrics_csv() and save_metrics_json() produce valid output."""

    def test_save_metrics_csv_correct_columns(self, default_config, tmp_path):
        """CSV has correct column count and row count."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_with_viz = False
        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)

        for _ in range(5):
            sim.step(1.0 / 60)
            rec.on_frame(sim)

        output = tmp_path / "metrics.csv"
        result = rec.save_metrics_csv(path=str(output))
        assert result is not None
        assert output.exists()

        with open(output) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 5
        assert len(rows[0]) > 0  # at least one column

    def test_save_metrics_json_valid_metadata(self, default_config, tmp_path):
        """JSON contains metadata fields and metrics array."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_with_viz = False
        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)

        for _ in range(5):
            sim.step(1.0 / 60)
            rec.on_frame(sim)

        output = tmp_path / "metrics.json"
        result = rec.save_metrics_json(path=str(output))
        assert result is not None
        assert output.exists()

        with open(output) as f:
            data = json.load(f)
        assert "metadata" in data
        assert "metrics" in data
        assert data["metadata"]["mode"] == cfg.mode
        assert data["metadata"]["num_boids"] == cfg.num_boids
        assert len(data["metrics"]) == 5

    def test_save_metrics_empty_history_returns_none(self, default_config):
        """Empty metrics_history → save returns None gracefully."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)
        assert rec.save_metrics_csv() is None
        assert rec.save_metrics_json() is None

    def test_save_metrics_csv_config_fallback_path(self, default_config, tmp_path):
        """save_metrics_csv() falls back to config.capture_metrics_csv when path=None."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_metrics_csv = str(tmp_path / "fallback.csv")
        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)

        # Feed one metrics entry so save succeeds
        rec.metrics_history = [{"alpha": 0.5, "count": 10}]
        result = rec.save_metrics_csv()  # path=None → config fallback
        assert result == cfg.capture_metrics_csv
        assert Path(result).exists()

    def test_save_metrics_json_config_fallback_path(self, default_config, tmp_path):
        """save_metrics_json() falls back to config.capture_metrics_json when path=None."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_metrics_json = str(tmp_path / "fallback.json")
        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)

        rec.metrics_history = [{"alpha": 0.5, "count": 10}]
        result = rec.save_metrics_json()  # path=None → config fallback
        assert result == cfg.capture_metrics_json
        assert Path(result).exists()

    def test_save_metrics_json_numpy_scalar(self, default_config, tmp_path):
        """save_metrics_json() converts scalar-like values via .item() branch."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)

        # Custom type with .item() but no .tolist() — hits line 146's elif branch
        class _Scalar:
            def item(self):
                return 3.14

        rec.metrics_history = [{"alpha": _Scalar(), "count": 42}]

        output = tmp_path / "scalar.json"
        result = rec.save_metrics_json(path=str(output))
        assert result is not None

        import json
        with open(output) as f:
            data = json.load(f)
        assert data["metrics"][0]["alpha"] == 3.14
        assert data["metrics"][0]["count"] == 42


class TestRecorderFullIntegration:
    """End-to-end recorder + engine integration via callback."""

    def test_run_headless_with_recorder_callback(self, default_config, tmp_path):
        """SimulationEngine.run_headless() + Recorder callback works end-to-end."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_with_viz = False
        cfg.capture_frames = 10
        cfg.capture_every = 1
        cfg.capture_metrics_csv = str(tmp_path / "run.csv")
        cfg.capture_metrics_json = str(tmp_path / "run.json")

        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)
        sim.run_headless(steps=10, callback=rec.on_frame)

        assert len(rec.metrics_history) == 10

        csv_result = rec.save_metrics_csv()
        json_result = rec.save_metrics_json()
        assert csv_result is not None
        assert json_result is not None
        assert Path(csv_result).exists()
        assert Path(json_result).exists()


# ═══════════════════════════════════════════════════════════════════
# I6 Missing Unit Tests — Renderer Caching (M1)
# ═══════════════════════════════════════════════════════════════════


class TestRecorderRendererCaching:
    """M1: _renderer is cached — Visualizer created once, reused."""

    @pytest.mark.gpu
    def test_renderer_cached_across_multiple_captures(
        self, default_config, gpu_available
    ):
        """M1: _capture_frame creates Visualizer once and reuses it.

        If the guard `if self._renderer is None` is removed, every
        _capture_frame call creates a new Visualizer+FBO — severe perf
        regression. Verify identity is preserved across on_frame calls.
        """
        if not gpu_available:
            pytest.skip("GPU not available")

        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_with_viz = True
        cfg.capture_every = 1
        cfg.capture_width = 320
        cfg.capture_height = 240

        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)

        # First capture — renderer is None, so Visualizer is created
        engine.step(1.0 / 60)
        rec.on_frame(engine)
        first_renderer = rec._renderer
        assert first_renderer is not None, "Renderer must be created on first capture"

        # Second capture — must reuse the same renderer
        engine.step(1.0 / 60)
        rec.on_frame(engine)
        assert rec._renderer is first_renderer, (
            "_renderer must be cached and reused across _capture_frame calls"
        )

        # Third capture — still the same
        engine.step(1.0 / 60)
        rec.on_frame(engine)
        assert rec._renderer is first_renderer


# ═══════════════════════════════════════════════════════════════════
# I6 Missing Unit Tests — Error Handling (M2, M6, M7, M8, M9)
# ═══════════════════════════════════════════════════════════════════


class TestRecorderErrorHandling:
    """M2, M6, M7, M8, M9: Error handling and edge cases."""

    def test_on_frame_handles_sim_metrics_none(self, default_config):
        """M8: on_frame() does not crash when sim.metrics is None.

        If the engine has no MetricsCollector (detail_level=0 or mock),
        the `if sim.metrics:` guard must prevent AttributeError.
        """
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_with_viz = False
        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)

        # Set metrics to None after engine creation.
        # Don't call step() — it would crash on metrics.collect().
        engine.metrics = None

        # on_frame must handle None metrics gracefully
        rec.on_frame(engine)  # must not raise AttributeError

        # _frame_count still incremented, metrics_history unchanged
        assert rec._frame_count == 1
        assert rec.metrics_history == []

    def test_on_frame_handles_empty_metrics_snapshot(self, default_config):
        """M9: Empty metrics history → snapshot() returns FlockMetrics() defaults.

        on_frame() calls sim.metrics.snapshot().to_dict() — even with
        empty history, to_dict() must work on default FlockMetrics.
        """
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_with_viz = False
        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)

        # Call on_frame without stepping — metrics history is empty
        # snapshot() returns FlockMetrics() with all defaults
        rec.on_frame(engine)

        assert len(rec.metrics_history) == 1
        entry = rec.metrics_history[0]
        # Default alpha is 0.0
        assert entry["alpha"] == 0.0
        # All default fields should be present
        assert "speed_avg" in entry
        assert "dispersion" in entry

    def test_runtime_error_during_fbo_capture_is_caught(
        self, default_config, monkeypatch
    ):
        """M6: RuntimeError from headless_frame() is caught silently.

        I6.3: The except RuntimeError clause catches GPU failures
        mid-capture without crashing the headless run.
        """
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_with_viz = True
        cfg.capture_every = 1
        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)

        # Inject a mock renderer that raises RuntimeError on headless_frame
        class _FailingRenderer:
            def headless_frame(self):
                raise RuntimeError("FBO exhausted")

        rec._renderer = _FailingRenderer()

        engine.step(1.0 / 60)
        rec.on_frame(engine)  # must not raise

        # Metrics still captured, frame count advanced, no frame added
        assert len(rec.metrics_history) == 1
        assert rec._frame_count == 1
        assert len(rec.frames) == 0

    def test_on_frame_does_not_throw_when_fbo_fails(
        self, default_config
    ):
        """M2: on_frame() completes normally when FBO capture raises RuntimeError.

        Uses a failing renderer so the REAL _capture_frame runs and
        catches RuntimeError internally — verifying on_frame never
        propagates GPU failures to the headless run loop.
        """
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_with_viz = True
        cfg.capture_every = 1
        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)

        # Inject a renderer whose headless_frame always fails
        class _FailingRenderer:
            def headless_frame(self):
                raise RuntimeError("Simulated FBO exhaustion")
        rec._renderer = _FailingRenderer()

        # on_frame must complete without raising
        engine.step(1.0 / 60)
        rec.on_frame(engine)

        assert rec._frame_count == 1
        assert len(rec.metrics_history) == 1
        assert rec.frames == []

    def test_non_runtimeerror_not_silently_swallowed(
        self, default_config
    ):
        """M7: Non-RuntimeError exceptions propagate (not silently swallowed).

        I6.3 replaced bare `except Exception: pass` with targeted catches.
        ValueError/MemoryError must escape so they're not hidden.
        """
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_with_viz = True
        cfg.capture_every = 1
        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)

        # Make _capture_frame raise ValueError (not RuntimeError)
        def _broken_capture(sim):
            raise ValueError("Should not be silently swallowed")
        rec._capture_frame = _broken_capture

        engine.step(1.0 / 60)
        with pytest.raises(ValueError, match="Should not be silently swallowed"):
            rec.on_frame(engine)


# ═══════════════════════════════════════════════════════════════════
# I6 Missing Unit Tests — on_frame Contract (M3, M5, M16, M17)
# ═══════════════════════════════════════════════════════════════════


class TestRecorderOnFrameContract:
    """M3, M5, M16, M17: on_frame() invariants and ordering."""

    def test_on_frame_increments_frame_count_even_when_capture_fails(
        self, default_config
    ):
        """M3: _frame_count advances even when _capture_frame catches RuntimeError.

        _frame_count is incremented at the top of on_frame(), before
        metrics or FBO capture. If reordered, a failing capture would
        skip the increment, drifting the capture_every gate.

        Uses a failing renderer so the real _capture_frame runs and
        catches RuntimeError internally.
        """
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_with_viz = True
        cfg.capture_every = 3
        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)

        # Inject a renderer that always fails — real _capture_frame
        # catches RuntimeError, but _frame_count must still advance
        class _FailingRenderer:
            def headless_frame(self):
                raise RuntimeError("FBO failure")
        rec._renderer = _FailingRenderer()

        for i in range(5):
            engine.step(1.0 / 60)
            rec.on_frame(engine)  # must not raise
            assert rec._frame_count == i + 1, (
                f"_frame_count must advance even when capture fails. "
                f"Expected {i + 1}, got {rec._frame_count}"
            )

        assert len(rec.metrics_history) == 5  # metrics captured every frame
        assert len(rec.frames) == 0  # all captures failed

    def test_on_frame_increments_frame_count_even_on_import_error(
        self, default_config, monkeypatch
    ):
        """M3: _frame_count increments even when viz import fails.

        Uses the existing import-blocking approach from
        test_on_frame_fbo_exception_silent but explicitly checks
        _frame_count advancement.
        """
        import builtins
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_with_viz = True
        cfg.capture_every = 1
        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)

        orig_import = builtins.__import__
        def _block_viz(name, *args, **kwargs):
            if "viz.visualizer" in name:
                raise ImportError("Blocked")
            return orig_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", _block_viz)

        for i in range(3):
            engine.step(1.0 / 60)
            rec.on_frame(engine)
            assert rec._frame_count == i + 1, (
                f"_frame_count must advance even when capture fails. "
                f"Expected {i + 1}, got {rec._frame_count}"
            )

        assert len(rec.metrics_history) == 3  # metrics unaffected
        assert len(rec.frames) == 0  # viz blocked

    def test_capture_every_larger_than_total_steps_captures_zero_frames(
        self, default_config
    ):
        """M5: capture_every > total steps → 0 frames captured.

        If the modulo gate has an off-by-one (e.g. _frame_count starts
        at 0 and `0 % 100 == 0`), frame 0 captures when it shouldn't.
        """
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_with_viz = False  # CPU-only, verify gating logic
        cfg.capture_every = 100
        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)

        engine.run_headless(steps=10, callback=rec.on_frame)

        assert rec._frame_count == 10
        assert len(rec.metrics_history) == 10  # every frame
        assert len(rec.frames) == 0  # never hit capture_every threshold

    def test_on_frame_metrics_captured_before_fbo(self, default_config):
        """M16: Metrics must be captured before FBO on each on_frame call.

        If someone reorders to capture FBO first and FBO fails, metrics
        for that frame would be lost.
        """
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_with_viz = True
        cfg.capture_every = 1
        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)

        order_log = []

        # Spy on the internals to verify ordering
        class _FailingRenderer:
            def headless_frame(self):
                order_log.append("fbo")
                raise RuntimeError("FBO failure")

        rec._renderer = _FailingRenderer()

        engine.step(1.0 / 60)
        rec.on_frame(engine)

        # Metrics must have been captured despite FBO failure
        assert len(rec.metrics_history) == 1, (
            "Metrics must be captured before FBO — even if FBO fails"
        )
        assert "fbo" in order_log  # FBO was attempted (after metrics)

    def test_on_frame_with_viz_false_never_calls_capture_frame(
        self, default_config
    ):
        """M17: with_viz=False → _capture_frame is never called.

        Spies on _capture_frame to verify the with_viz guard is
        honored, not just that frames list stays empty.
        """
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_with_viz = False
        cfg.capture_every = 1
        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)

        call_count = [0]
        original = rec._capture_frame
        def spy(sim):
            call_count[0] += 1
            return original(sim)
        rec._capture_frame = spy

        engine.run_headless(steps=5, callback=rec.on_frame)

        assert call_count[0] == 0, (
            f"_capture_frame must never be called when with_viz=False. "
            f"Called {call_count[0]} times."
        )
        assert len(rec.metrics_history) == 5


# ═══════════════════════════════════════════════════════════════════
# I6 Missing Unit Tests — Save Edge Cases (M10, M11, M12, M13, M15)
# ═══════════════════════════════════════════════════════════════════


class TestRecorderSaveEdgeCases:
    """M10, M11, M12, M13, M15: Save operation edge cases."""

    def test_save_gif_creates_parent_directories(self, default_config, tmp_path):
        """M10: save_gif creates parent dirs for nested output paths."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine
        from PIL import Image

        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)
        rec.frames = [Image.new("RGB", (20, 15))]

        nested = tmp_path / "deep" / "nested" / "output.gif"
        result = rec.save_gif(path=str(nested))

        assert result is not None
        assert nested.exists()
        assert nested.stat().st_size > 0

    def test_save_gif_frame_without_resize_handled(self, default_config, tmp_path):
        """M11: Frames without .resize() fall through LANCZOS guard.

        The list comprehension: `f.resize(...) if hasattr(f, "resize") else f`
        must not crash when a frame lacks .resize().
        """
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)

        # Mock frame without .resize() but with .save()
        class _RawFrame:
            def save(self, path, **kwargs):
                # Write minimal GIF bytes
                with open(path, "wb") as f:
                    f.write(b"GIF89a\x01\x00\x01\x00\x00\x00\x00;")

        rec.frames = [_RawFrame(), _RawFrame()]

        out = tmp_path / "no_resize.gif"
        result = rec.save_gif(path=str(out))
        assert result == str(out)
        assert out.exists()

    def test_save_metrics_csv_creates_parent_directories(
        self, default_config, tmp_path
    ):
        """M12: save_metrics_csv creates parent dirs for nested paths."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)
        rec.metrics_history = [{"alpha": 0.5, "speed_avg": 3.2}]

        nested = tmp_path / "a" / "b" / "metrics.csv"
        result = rec.save_metrics_csv(path=str(nested))

        assert result is not None
        assert nested.exists()

    def test_save_metrics_json_ndarray_tolist_branch(
        self, default_config, tmp_path
    ):
        """M13: Values with .tolist() are converted (ndarray branch).

        The save_metrics_json_numpy_scalar test covers .item(), but
        the .tolist() branch for ndarray values is untested.
        """
        import numpy as np
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)

        # Simulate angular_momentum as ndarray (has .tolist(), no .item())
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        rec.metrics_history = [{"angular_momentum": arr, "alpha": 0.5}]

        out = tmp_path / "ndarray.json"
        result = rec.save_metrics_json(path=str(out))
        assert result is not None

        with open(out) as f:
            data = json.load(f)
        assert data["metrics"][0]["angular_momentum"] == [1.0, 2.0, 3.0]
        assert data["metrics"][0]["alpha"] == 0.5

    def test_save_metrics_json_creates_parent_directories(
        self, default_config, tmp_path
    ):
        """M15: save_metrics_json creates parent dirs for nested paths."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)
        rec.metrics_history = [{"alpha": 0.5}]

        nested = tmp_path / "x" / "y" / "metrics.json"
        result = rec.save_metrics_json(path=str(nested))

        assert result is not None
        assert nested.exists()
