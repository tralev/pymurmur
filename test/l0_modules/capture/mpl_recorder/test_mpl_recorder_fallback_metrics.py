"""Unit tests for capture/mpl_recorder.py — warning on activation, _hsv_to_rgb helper, Recorder fallback integration, save_metrics_csv.

Split out of test_mpl_recorder.py (file-size split).
"""

import warnings

import pytest


class TestMPLRecorderWarning:
    """MPLRecorder warns once on first activation."""

    def test_warns_on_creation(self, default_config):
        """Creating MPLRecorder issues a UserWarning about GPU fallback."""
        from pymurmur.capture.mpl_recorder import MPLRecorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 5
        engine = SimulationEngine(cfg)

        # Reset class-level flag for isolated test
        MPLRecorder._WARNED = False

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MPLRecorder(engine, cfg)
            assert len(w) == 1
            assert "Matplotlib fallback" in str(w[0].message)

    def test_warns_only_once(self, default_config):
        """Second MPLRecorder creation does not warn again."""
        from pymurmur.capture.mpl_recorder import MPLRecorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 5
        engine = SimulationEngine(cfg)

        # First creation warns
        MPLRecorder._WARNED = False
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MPLRecorder(engine, cfg)
            assert len(w) == 1

        # Second creation does NOT warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MPLRecorder(engine, cfg)
            assert len(w) == 0


# ── P8.9f: _hsv_to_rgb helper ─────────────────────────────────

class TestHSVToRGB:
    """HSV→RGB conversion helper."""

    def test_hsv_to_rgb_pure_red(self):
        """h=0, s=1, v=1 → RGB red."""
        from pymurmur.capture.mpl_recorder import _hsv_to_rgb
        r, g, b = _hsv_to_rgb(0.0, 1.0, 1.0)
        assert r == pytest.approx(1.0)
        assert g == pytest.approx(0.0)
        assert b == pytest.approx(0.0)

    def test_hsv_to_rgb_pure_green(self):
        """h=1/3, s=1, v=1 → RGB green."""
        from pymurmur.capture.mpl_recorder import _hsv_to_rgb
        r, g, b = _hsv_to_rgb(1.0 / 3.0, 1.0, 1.0)
        assert g == pytest.approx(1.0)

    def test_hsv_to_rgb_black(self):
        """v=0 → black regardless of h, s."""
        from pymurmur.capture.mpl_recorder import _hsv_to_rgb
        r, g, b = _hsv_to_rgb(0.5, 1.0, 0.0)
        assert r == 0.0 and g == 0.0 and b == 0.0


# ── P8.9g: Recorder fallback integration ──────────────────────

class TestRecorderMPLFallback:
    """Existing GPU Recorder falls back to MPLRecorder on GPU failure."""

    def test_recorder_has_fallback_attrs(self, default_config):
        """Recorder initialises with _mpl_fallback_enabled from config."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_mpl_fallback = True
        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)

        assert rec._mpl_fallback_enabled is True
        assert rec._mpl_fallback is None
        assert rec._mpl_fallback_activated is False

    @staticmethod
    def _make_fail_frame(rec):
        """Create a _capture_frame replacement that triggers MPL fallback.

        Mimics the real _capture_frame's try/except RuntimeError handler
        so the exception doesn't propagate through on_frame().
        """
        def fail_frame(sim):
            try:
                raise RuntimeError("GPU not available")
            except RuntimeError:
                if rec._mpl_fallback_enabled:
                    rec._fallback_to_mpl(sim)
        return fail_frame

    def test_recorder_fallback_to_mpl_on_runtimeerror(
        self, default_config, monkeypatch
    ):
        """When GPU capture raises RuntimeError, fallback to MPLRecorder."""
        from pymurmur.capture.mpl_recorder import MPLRecorder
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        # Suppress the fallback warning during test
        MPLRecorder._WARNED = True

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_every = 1
        cfg.capture_prewarm = 0
        cfg.capture_with_viz = True
        cfg.capture_mpl_fallback = True
        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)

        monkeypatch.setattr(rec, "_capture_frame", self._make_fail_frame(rec))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            engine.step(1.0 / 60)
            rec.on_frame(engine)

        # MPL fallback should have been activated
        assert rec._mpl_fallback_activated is True
        assert rec._mpl_fallback is not None

    def test_recorder_fallback_produces_frames(
        self, default_config, monkeypatch
    ):
        """MPL fallback captures real frames into Recorder.frames."""
        from pymurmur.capture.mpl_recorder import MPLRecorder
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        MPLRecorder._WARNED = True

        cfg = default_config
        cfg.num_boids = 20
        cfg.capture_every = 1
        cfg.capture_prewarm = 0
        cfg.capture_with_viz = True
        cfg.capture_mpl_fallback = True
        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)

        monkeypatch.setattr(rec, "_capture_frame", self._make_fail_frame(rec))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            engine.step(1.0 / 60)
            rec.on_frame(engine)
            engine.step(1.0 / 60)
            rec.on_frame(engine)

        # Frames should now exist from MPL fallback
        assert len(rec.frames) == 2

    def test_recorder_no_fallback_when_disabled(
        self, default_config, monkeypatch
    ):
        """When capture_mpl_fallback is False, RuntimeError is silent."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_every = 1
        cfg.capture_prewarm = 0
        cfg.capture_with_viz = True
        cfg.capture_mpl_fallback = False  # fallback OFF
        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)

        monkeypatch.setattr(rec, "_capture_frame", self._make_fail_frame(rec))

        engine.step(1.0 / 60)
        rec.on_frame(engine)

        # Fallback should NOT activate, no frames captured
        assert rec._mpl_fallback_activated is False
        assert rec._mpl_fallback is None
        assert len(rec.frames) == 0

    def test_recorder_fallback_merges_frames(
        self, default_config, monkeypatch
    ):
        """MPLRecorder frames are appended to Recorder.frames."""
        from pymurmur.capture.mpl_recorder import MPLRecorder
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        MPLRecorder._WARNED = True

        cfg = default_config
        cfg.num_boids = 20
        cfg.capture_every = 1
        cfg.capture_prewarm = 0
        cfg.capture_with_viz = True
        cfg.capture_mpl_fallback = True
        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)

        monkeypatch.setattr(rec, "_capture_frame", self._make_fail_frame(rec))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _i in range(5):
                engine.step(1.0 / 60)
                rec.on_frame(engine)

        assert len(rec.frames) == 5

    def test_recorder_fallback_save_gif(
        self, default_config, tmp_path, monkeypatch
    ):
        """save_gif works with MPL-fallback frames."""
        from pymurmur.capture.mpl_recorder import MPLRecorder
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine

        MPLRecorder._WARNED = True

        cfg = default_config
        cfg.num_boids = 20
        cfg.capture_every = 1
        cfg.capture_prewarm = 0
        cfg.capture_with_viz = True
        cfg.capture_mpl_fallback = True
        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)

        monkeypatch.setattr(rec, "_capture_frame", self._make_fail_frame(rec))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _i in range(3):
                engine.step(1.0 / 60)
                rec.on_frame(engine)

        out = tmp_path / "fallback.gif"
        result = rec.save_gif(str(out), fps=10)
        assert result is not None
        assert out.exists()
        assert out.stat().st_size > 0


# ── P8.9h: save_metrics_csv ───────────────────────────────────

class TestMPLRecorderSaveMetrics:
    """save_metrics_csv exports metrics correctly."""

    def test_save_metrics_csv_writes_file(self, default_config, tmp_path):
        """save_metrics_csv writes valid CSV."""
        from pymurmur.capture.mpl_recorder import MPLRecorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_prewarm = 0
        engine = SimulationEngine(cfg)
        rec = MPLRecorder(engine, cfg)

        engine.step(1.0 / 60)
        rec.on_frame(engine)

        out = tmp_path / "metrics.csv"
        result = rec.save_metrics_csv(str(out))
        assert result is not None
        assert out.exists()
        assert out.stat().st_size > 0
