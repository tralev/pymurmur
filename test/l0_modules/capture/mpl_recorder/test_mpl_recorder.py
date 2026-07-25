"""Unit tests for capture/mpl_recorder.py — MPLRecorder init, config fields, on_frame metrics capture/pre-warm/gating, dual-view scatter rendering, save_gif.

Split out of test_mpl_recorder.py (file-size split).
"""

from pathlib import Path


class TestMPLRecorderInit:
    """MPLRecorder creation and basic properties."""

    def test_mpl_recorder_importable(self):
        """MPLRecorder is importable."""
        from pymurmur.capture.mpl_recorder import MPLRecorder
        assert MPLRecorder is not None

    def test_mpl_recorder_init_empty_state(self, default_config):
        """MPLRecorder starts with empty frames and metrics_history."""
        from pymurmur.capture.mpl_recorder import MPLRecorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)
        rec = MPLRecorder(engine, cfg)
        assert rec.frames == []
        assert rec.metrics_history == []
        assert rec._frame_count == 0
        assert rec._every == cfg.capture_every
        assert rec._prewarm == cfg.capture_prewarm
        assert rec._dpi == cfg.capture_mpl_dpi

    def test_mpl_recorder_respects_config_dimensions(self, default_config):
        """MPLRecorder reads width, height, dpi from config."""
        from pymurmur.capture.mpl_recorder import MPLRecorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 5
        cfg.capture_width = 400
        cfg.capture_height = 300
        cfg.capture_mpl_dpi = 100
        engine = SimulationEngine(cfg)
        rec = MPLRecorder(engine, cfg)
        assert rec._width == 400
        assert rec._height == 300
        assert rec._dpi == 100


class TestCaptureConfigMPLFields:
    """CaptureConfig has P8.9 fields + _FIELD_MAP + to_file."""

    def test_capture_mpl_fallback_field(self, default_config):
        """capture_mpl_fallback exists and defaults to True."""
        cfg = default_config
        assert hasattr(cfg, "capture_mpl_fallback")
        assert cfg.capture_mpl_fallback is True

    def test_capture_mpl_dpi_field(self, default_config):
        """capture_mpl_dpi exists and defaults to 72."""
        cfg = default_config
        assert hasattr(cfg, "capture_mpl_dpi")
        assert cfg.capture_mpl_dpi == 72

    def test_capture_mpl_fields_in_to_file(self, default_config, tmp_path):
        """capture_mpl_* fields appear in YAML output."""
        out = tmp_path / "cfg.yaml"
        default_config.to_file(str(out))
        text = out.read_text()
        assert "capture_mpl_fallback" in text
        assert "capture_mpl_dpi" in text


# ── P8.9b: on_frame — metrics + pre-warm + gating ──────────────

class TestMPLRecorderOnFrame:
    """on_frame() captures metrics every call, frames after pre-warm."""

    def test_on_frame_appends_metrics(self, default_config):
        """on_frame() records metrics every call regardless of pre-warm."""
        from pymurmur.capture.mpl_recorder import MPLRecorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_every = 1
        cfg.capture_prewarm = 5
        engine = SimulationEngine(cfg)

        rec = MPLRecorder(engine, cfg)

        # During pre-warm: metrics captured
        for _i in range(3):
            engine.step(1.0 / 60)
            rec.on_frame(engine)
        assert len(rec.metrics_history) == 3
        assert len(rec.frames) == 0  # no frames during pre-warm

    def test_on_frame_captures_after_prewarm(self, default_config):
        """on_frame() starts capturing after pre-warm frames."""
        from pymurmur.capture.mpl_recorder import MPLRecorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_every = 1
        cfg.capture_prewarm = 2
        engine = SimulationEngine(cfg)

        rec = MPLRecorder(engine, cfg)

        # Frame 1: pre-warm
        engine.step(1.0 / 60)
        rec.on_frame(engine)
        assert len(rec.frames) == 0

        # Frame 2: pre-warm
        engine.step(1.0 / 60)
        rec.on_frame(engine)
        assert len(rec.frames) == 0

        # Frame 3: first capture
        engine.step(1.0 / 60)
        rec.on_frame(engine)
        assert len(rec.frames) == 1

    def test_capture_every_n_gating(self, default_config):
        """Frames are captured at capture_every intervals after pre-warm."""
        from pymurmur.capture.mpl_recorder import MPLRecorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_every = 3
        cfg.capture_prewarm = 0
        engine = SimulationEngine(cfg)

        rec = MPLRecorder(engine, cfg)

        frames_captured = []
        for i in range(1, 11):
            engine.step(1.0 / 60)
            prev = len(rec.frames)
            rec.on_frame(engine)
            if len(rec.frames) > prev:
                frames_captured.append(i)

        # Should capture at frames 3, 6, 9
        assert frames_captured == [3, 6, 9]


# ── P8.9c: Dual-view scatter rendering ─────────────────────────

class TestMPLRecorderRender:
    """_render_frame produces dual-view 3D scatter with prey/predator."""

    def test_render_frame_returns_pil_image(self, default_config):
        """_render_frame returns a PIL Image for a non-empty flock."""
        from pymurmur.capture.mpl_recorder import MPLRecorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 20
        engine = SimulationEngine(cfg)
        rec = MPLRecorder(engine, cfg)

        img = rec._render_frame(engine)
        assert img is not None
        from PIL.Image import Image
        assert isinstance(img, Image)

    def test_render_frame_empty_flock_returns_none(self, default_config):
        """_render_frame returns None when N_active == 0."""
        from pymurmur.capture.mpl_recorder import MPLRecorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 20
        engine = SimulationEngine(cfg)
        # Remove all active boids
        engine.flock.active[:] = False
        rec = MPLRecorder(engine, cfg)

        img = rec._render_frame(engine)
        assert img is None

    def test_render_frame_dual_view_width(self, default_config):
        """Dual-view image width is ~2× capture_width for side-by-side subplots."""
        from pymurmur.capture.mpl_recorder import MPLRecorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 15
        cfg.capture_width = 400
        cfg.capture_height = 300
        cfg.capture_mpl_dpi = 72
        engine = SimulationEngine(cfg)
        rec = MPLRecorder(engine, cfg)

        img = rec._render_frame(engine)
        # Dual-view: width should be significantly larger than capture_width
        assert img is not None
        assert img.width >= cfg.capture_width * 1.5

    def test_render_frame_with_predators(self, default_config):
        """Frame renders with predators as red markers."""
        from pymurmur.capture.mpl_recorder import MPLRecorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)
        engine.flock.is_predator[0] = True  # mark one as predator

        rec = MPLRecorder(engine, cfg)
        img = rec._render_frame(engine)
        assert img is not None

    def test_render_frame_exception_survives(self, default_config, monkeypatch):
        """_capture_frame doesn't crash when _render_frame raises."""
        from pymurmur.capture.mpl_recorder import MPLRecorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_every = 1
        cfg.capture_prewarm = 0
        engine = SimulationEngine(cfg)
        rec = MPLRecorder(engine, cfg)

        def boom(*a, **kw):
            raise ValueError("matplotlib exploded")
        monkeypatch.setattr(rec, "_render_frame", boom)

        # Must not raise
        engine.step(1.0 / 60)
        rec.on_frame(engine)
        assert len(rec.frames) == 0

    def test_render_frame_metrics_always_captured(self, default_config):
        """Metrics are captured on every frame even without GPU."""
        from pymurmur.capture.mpl_recorder import MPLRecorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_every = 5  # capture frames rarely
        cfg.capture_prewarm = 10
        engine = SimulationEngine(cfg)
        rec = MPLRecorder(engine, cfg)

        for _i in range(20):
            engine.step(1.0 / 60)
            rec.on_frame(engine)

        assert len(rec.metrics_history) == 20


# ── P8.9d: save_gif ────────────────────────────────────────────

class TestMPLRecorderSaveGif:
    """save_gif writes valid GIFs from captured frames."""

    def test_save_gif_creates_file(self, default_config, tmp_path):
        """save_gif writes a non-empty GIF file."""
        from pymurmur.capture.mpl_recorder import MPLRecorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_every = 1
        cfg.capture_prewarm = 0
        engine = SimulationEngine(cfg)
        rec = MPLRecorder(engine, cfg)

        # Capture 3 frames
        for _i in range(3):
            engine.step(1.0 / 60)
            rec.on_frame(engine)

        assert len(rec.frames) == 3

        out = tmp_path / "test.gif"
        result = rec.save_gif(str(out), fps=10)
        assert result is not None
        assert out.exists()
        assert out.stat().st_size > 0

    def test_save_gif_empty_frames_returns_none(self, default_config):
        """save_gif returns None when no frames were captured."""
        from pymurmur.capture.mpl_recorder import MPLRecorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 5
        engine = SimulationEngine(cfg)
        rec = MPLRecorder(engine, cfg)

        result = rec.save_gif()
        assert result is None

    def test_save_gif_single_frame(self, default_config, tmp_path):
        """save_gif handles a single-frame GIF."""
        from pymurmur.capture.mpl_recorder import MPLRecorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_every = 1
        cfg.capture_prewarm = 0
        engine = SimulationEngine(cfg)
        rec = MPLRecorder(engine, cfg)

        engine.step(1.0 / 60)
        rec.on_frame(engine)
        assert len(rec.frames) == 1

        out = tmp_path / "single.gif"
        result = rec.save_gif(str(out))
        assert result is not None
        assert out.exists()

    def test_save_gif_config_fallback_path(self, default_config, tmp_path):
        """save_gif uses config.capture_output when no path is given."""
        from pymurmur.capture.mpl_recorder import MPLRecorder
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_every = 1
        cfg.capture_prewarm = 0
        cfg.capture_output = str(tmp_path / "config_output.gif")
        engine = SimulationEngine(cfg)
        rec = MPLRecorder(engine, cfg)

        engine.step(1.0 / 60)
        rec.on_frame(engine)

        result = rec.save_gif()
        assert result is not None
        assert Path(cfg.capture_output).exists()


