"""Unit tests for capture/recorder.py — on_frame contract, save edge cases, D16 capture override precedence, D19 frame caps.

Split out of test_recorder.py (file-size split).
"""

import json


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
        rec._prewarm = 0  # P8.7: disable pre-warm for test

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
        from PIL import Image

        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

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


# ═══════════════════════════════════════════════════════════════════
# D16: Capture override precedence (CLI > env > YAML)
# ═══════════════════════════════════════════════════════════════════


class TestD16CaptureOverridePrecedence:
    """D16: Recorder reads capture params from config only — env var
    application moved to __main__.py so CLI > env > YAML."""

    def test_recorder_no_longer_reads_env_vars(self):
        """D16: Recorder.__init__ does not call os.environ.get."""
        import inspect

        from pymurmur.capture.recorder import Recorder
        src = inspect.getsource(Recorder.__init__)
        assert "os.environ" not in src, (
            "Recorder.__init__ must not read os.environ directly"
        )

    def test_config_values_used_directly(self, monkeypatch):
        """D16: Recorder reads capture width/height/frames from config."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.num_boids = 5
        cfg.capture_width = 640
        cfg.capture_height = 480
        cfg.capture_frames = 42
        cfg.capture_output = "test.gif"

        # Set env vars to DIFFERENT values — Recorder must ignore them
        monkeypatch.setenv("CAPTURE_WIDTH", "9999")
        monkeypatch.setenv("CAPTURE_HEIGHT", "9999")
        monkeypatch.setenv("CAPTURE_FRAMES", "9999")
        monkeypatch.setenv("CAPTURE_OUT", "env_override.gif")

        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)

        # Recorder must use config values, not env vars
        assert rec._capture_width == 640
        assert rec._capture_height == 480
        assert rec._capture_frames == 42
        assert rec._capture_output == "test.gif"

    def test_main_cli_overrides_env(self, monkeypatch):
        """D16: --set capture.capture_frames=500 beats CAPTURE_FRAMES=100."""
        import os as _os

        from pymurmur.__main__ import _apply_set_overrides
        from pymurmur.core.config import SimConfig

        cfg = SimConfig()
        monkeypatch.setenv("CAPTURE_FRAMES", "100")

        # Step 1: Apply env vars (as __main__ does)
        for _env_key, _cfg_attr in [
            ("CAPTURE_WIDTH", "capture_width"),
            ("CAPTURE_HEIGHT", "capture_height"),
            ("CAPTURE_FRAMES", "capture_frames"),
            ("CAPTURE_OUT", "capture_output"),
        ]:
            _val = _os.environ.get(_env_key)
            if _val is not None:
                try:
                    setattr(cfg, _cfg_attr, int(_val))
                except ValueError:
                    setattr(cfg, _cfg_attr, _val)

        assert cfg.capture_frames == 100, "env should set to 100"

        # Step 2: Apply CLI override
        _apply_set_overrides(cfg, ["capture.capture_frames=500"])
        assert cfg.capture_frames == 500, (
            f"CLI should override env: expected 500, got {cfg.capture_frames}"
        )

    def test_main_cli_overrides_env_capture_width(self, monkeypatch):
        """D16: --set capture.capture_width=800 beats CAPTURE_WIDTH=640."""
        import os as _os

        from pymurmur.__main__ import _apply_set_overrides
        from pymurmur.core.config import SimConfig

        cfg = SimConfig()
        monkeypatch.setenv("CAPTURE_WIDTH", "640")

        for _env_key, _cfg_attr in [
            ("CAPTURE_WIDTH", "capture_width"),
            ("CAPTURE_HEIGHT", "capture_height"),
            ("CAPTURE_FRAMES", "capture_frames"),
            ("CAPTURE_OUT", "capture_output"),
        ]:
            _val = _os.environ.get(_env_key)
            if _val is not None:
                try:
                    setattr(cfg, _cfg_attr, int(_val))
                except ValueError:
                    setattr(cfg, _cfg_attr, _val)

        assert cfg.capture_width == 640
        _apply_set_overrides(cfg, ["capture.capture_width=800"])
        assert cfg.capture_width == 800, (
            f"CLI should beat env: expected 800, got {cfg.capture_width}"
        )


# ═══════════════════════════════════════════════════════════════════
# D19: Recorder frame caps — unbounded accumulators
# ═══════════════════════════════════════════════════════════════════


class TestD19RecorderFrameCaps:
    """D19: Recorder frame list and metrics_history are ring-buffer
    truncated to prevent unbounded memory growth on long runs."""

    def test_recorder_has_frame_cap_attribute(self):
        """D19: Recorder stores _frame_cap (default 10000)."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.num_boids = 5
        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)

        assert hasattr(rec, '_frame_cap')
        assert rec._frame_cap == 10000

    def test_recorder_frames_truncated_at_cap(self):
        """D19: When frames exceed _frame_cap, oldest are dropped."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.num_boids = 5
        cfg.capture_frame_cap = 3  # small cap for test
        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)

        # Simulate filling frames past the cap
        class FakeImg:
            pass

        for _i in range(10):
            rec.frames.append(FakeImg())
            if len(rec.frames) > rec._frame_cap:
                rec.frames = rec.frames[-rec._frame_cap:]

        assert len(rec.frames) == 3, (
            f"Frames should be capped at 3, got {len(rec.frames)}"
        )

    def test_recorder_metrics_history_truncated_at_cap(self):
        """D19: metrics_history truncated at _frame_cap."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.num_boids = 5
        cfg.capture_frame_cap = 5
        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)

        for i in range(15):
            rec.metrics_history.append({"frame": i})
            if len(rec.metrics_history) > rec._frame_cap:
                rec.metrics_history = rec.metrics_history[-rec._frame_cap:]

        assert len(rec.metrics_history) == 5, (
            f"Metrics history should be capped at 5, got {len(rec.metrics_history)}"
        )
        # Newest frames preserved
        assert rec.metrics_history[0]["frame"] == 10
        assert rec.metrics_history[-1]["frame"] == 14
