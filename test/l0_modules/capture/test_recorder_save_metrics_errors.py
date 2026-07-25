"""Unit tests for capture/recorder.py — save_metrics_csv/json, full integration, renderer caching, error handling.

Split out of test_recorder.py (file-size split).
"""

import csv
import json
from pathlib import Path

import pytest


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
        cfg.capture_prewarm = 0  # P8.7: capture from the first frame

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
        """G6/M6: RuntimeError from headless_frame() (simulated GPU/GL
        failure) is caught and degrades to the matplotlib fallback (P8.9)
        instead of crashing the headless run.
        """
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_with_viz = True
        cfg.capture_every = 1
        cfg.capture_prewarm = 0  # G6: without this, _capture_frame is never
        # reached (capture_prewarm defaults to 60) and the test is vacuous —
        # it would pass even if the RuntimeError handler were removed.
        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)

        # Inject a mock renderer that raises RuntimeError on headless_frame
        class _FailingRenderer:
            def headless_frame(self):
                raise RuntimeError("FBO exhausted")

        rec._renderer = _FailingRenderer()

        engine.step(1.0 / 60)
        rec.on_frame(engine)  # must not raise

        # _capture_frame was actually reached (sanity — the whole point of
        # this test is exercising it, not the prewarm gate short-circuiting)
        assert rec._frame_count == 1
        assert len(rec.metrics_history) == 1

        # G6: degraded to the mpl fallback and it produced a real frame —
        # not just "didn't crash" but actually recovered capture capability
        assert rec._mpl_fallback_activated is True
        assert rec._mpl_fallback is not None
        assert len(rec.frames) == 1, (
            "mpl fallback should have captured exactly one frame"
        )

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
        cfg.capture_prewarm = 0  # G6: without this, _capture_frame is never
        # reached (capture_prewarm defaults to 60) and this test is vacuous.
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
        # G6: RuntimeError degrades to the mpl fallback (P8.9), which
        # produces a real frame — capture recovers, it doesn't just avoid
        # crashing.  (Default cfg.capture_mpl_fallback=True.)
        assert len(rec.frames) == 1

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
        rec._prewarm = 0  # P8.7: no pre-warm for this test

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


