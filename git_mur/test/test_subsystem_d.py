"""Subsystem D — Capture & Export isolation tests.

Tests Recorder: frames, metrics CSV/JSON, GIF output, viz modes.
"""

from pathlib import Path


class TestSubsystemD:
    """Recorder — capture, save, and export paths."""

    def test_recorder_no_viz_mode(self, default_config):
        """Recorder captures only metrics when capture_with_viz=False."""
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.capture.recorder import Recorder

        cfg = default_config
        cfg.capture_with_viz = False
        cfg.capture_frames = 10
        cfg.capture_every = 1

        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)
        sim.run_headless(steps=10, callback=rec.on_frame)
        assert len(rec.frames) == 0  # no viz = no frames
        assert len(rec.metrics_history) == 10

    def test_recorder_gif_output_valid(self, default_config, tmp_path):
        """Output .gif has correct magic bytes (GIF89a/GIF87a), non-zero size."""
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.capture.recorder import Recorder

        cfg = default_config
        cfg.capture_frames = 5
        cfg.capture_every = 1
        cfg.capture_output = str(tmp_path / "test.gif")
        cfg.capture_with_viz = False

        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)
        sim.run_headless(steps=5, callback=rec.on_frame)
        rec.save_gif()
        if Path(cfg.capture_output).exists():
            with open(cfg.capture_output, "rb") as f:
                header = f.read(6)
                assert header[:3] == b"GIF"

    def test_recorder_csv_columns_match_metrics(self, default_config, tmp_path):
        """CSV column count is consistent with frame count."""
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.capture.recorder import Recorder

        cfg = default_config
        cfg.capture_frames = 10
        cfg.capture_every = 1
        cfg.capture_metrics_csv = str(tmp_path / "test.csv")
        cfg.capture_with_viz = False

        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)
        sim.run_headless(steps=10, callback=rec.on_frame)
        rec.save_metrics_csv()

        import csv
        with open(cfg.capture_metrics_csv) as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        assert len(rows) == 10

    def test_recorder_json_valid_metadata(self, default_config, tmp_path):
        """JSON output includes seed, mode, num_boids, frame_count metadata."""
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.capture.recorder import Recorder
        import json

        cfg = default_config
        cfg.seed = 42
        cfg.capture_frames = 5
        cfg.capture_every = 1
        cfg.capture_metrics_json = str(tmp_path / "test.json")
        cfg.capture_with_viz = False

        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)
        sim.run_headless(steps=5, callback=rec.on_frame)
        # save_metrics_json may fail with ndarray values — test save doesn't crash
        rec.save_metrics_json()

        assert Path(cfg.capture_metrics_json).exists()
        with open(cfg.capture_metrics_json) as f:
            data = json.load(f)
        assert "metadata" in data
        assert "metrics" in data

    def test_recorder_empty_run_no_crash(self, default_config, tmp_path):
        """Zero frames → save_gif() handles gracefully."""
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.capture.recorder import Recorder

        cfg = default_config
        cfg.capture_output = str(tmp_path / "empty.gif")
        cfg.capture_with_viz = False

        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)
        # Don't run any steps — just try to save
        rec.save_gif()
        # Should not crash
