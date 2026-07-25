"""I6 — Integration test for the capture pipeline: full pipeline (dimensions + serialization).

Split out of test_capture_pipeline.py (file-size split).
"""

import csv
import json
from pathlib import Path

import pytest

from pymurmur.simulation.engine import SimulationEngine


class TestFullCapturePipeline:
    """I6.1 + I6.2 + I6.5: Full capture pipeline end-to-end."""

    def test_full_metrics_pipeline_csv_and_json(self, default_config, tmp_path):
        """IT1: Metrics flow step→on_frame→snapshot().to_dict()→save to disk.

        Verifies the complete serialization pipeline:
        1. engine.step() computes metrics via MetricsCollector.collect()
        2. Recorder.on_frame() calls sim.metrics.snapshot().to_dict()
        3. save_metrics_csv() writes valid CSV with correct row/column counts
        4. save_metrics_json() writes valid JSON with correct metadata
        """
        from pymurmur.capture.recorder import Recorder

        cfg = default_config
        cfg.num_boids = 10
        cfg.seed = 42
        cfg.mode = "projection"
        cfg.capture_with_viz = False
        cfg.capture_every = 1
        cfg.capture_frames = 15
        cfg.capture_metrics_csv = str(tmp_path / "pipeline.csv")
        cfg.capture_metrics_json = str(tmp_path / "pipeline.json")

        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)

        # Run full capture pipeline: step + on_frame
        engine.run_headless(steps=15, callback=rec.on_frame)

        # Verify metrics were captured every frame
        assert len(rec.metrics_history) == 15
        assert rec._frame_count == 15

        # Save CSV — verify valid output
        csv_path = rec.save_metrics_csv()
        assert csv_path is not None
        assert Path(csv_path).exists()
        with open(csv_path) as f:
            csv_reader = csv.DictReader(f)
            csv_rows = list(csv_reader)
        assert len(csv_rows) == 15
        # Verify all expected metric keys are present as CSV columns
        expected_keys = {
            "alpha", "theta", "theta_prime", "angular_momentum",
            "dispersion", "speed_avg", "force_avg", "power_avg",
            "local_spacing",
        }
        actual_keys = set(csv_rows[0].keys())
        missing = expected_keys - actual_keys
        assert not missing, f"CSV missing expected metric columns: {missing}"

        # Save JSON — verify metadata and metrics
        json_path = rec.save_metrics_json()
        assert json_path is not None
        assert Path(json_path).exists()
        with open(json_path) as f:
            data = json.load(f)

        assert "metadata" in data
        assert "metrics" in data
        assert data["metadata"]["seed"] == 42
        assert data["metadata"]["mode"] == "projection"
        assert data["metadata"]["num_boids"] == 10
        assert data["metadata"]["frame_count"] == 15
        assert len(data["metrics"]) == 15

        # Verify each metric entry has the same keys
        for i, entry in enumerate(data["metrics"]):
            missing_entry = expected_keys - set(entry.keys())
            assert not missing_entry, (
                f"JSON frame {i} missing keys: {missing_entry}"
            )

    def test_metrics_pipeline_serialization_types_are_json_safe(
        self, default_config, tmp_path
    ):
        """IT1: All metric values in JSON are JSON-safe types (no numpy objects).

        I6.5 mandates ndarray→list, numpy NaN→null, numpy scalar→Python.
        """
        from pymurmur.capture.recorder import Recorder

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_with_viz = False
        cfg.capture_every = 1
        cfg.capture_metrics_json = str(tmp_path / "json_safe.json")

        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)
        engine.run_headless(steps=5, callback=rec.on_frame)
        rec.save_metrics_json()

        with open(cfg.capture_metrics_json) as f:
            data = json.load(f)

        json_safe_types = (str, int, float, bool, list, dict, type(None))
        numpy_types = ("numpy", "ndarray", "float32", "float64", "int32", "int64")

        for i, entry in enumerate(data["metrics"]):
            for key, value in entry.items():
                if isinstance(value, list):
                    # Check list elements
                    for j, elem in enumerate(value):
                        assert isinstance(elem, json_safe_types), (
                            f"Frame {i}, key '{key}', element [{j}]: "
                            f"{type(elem).__name__} is not JSON-safe"
                        )
                else:
                    type_name = type(value).__name__
                    assert isinstance(value, json_safe_types), (
                        f"Frame {i}, key '{key}': "
                        f"{type_name} is not JSON-safe (value={value})"
                    )
                    assert not any(nt in type_name.lower() for nt in numpy_types), (
                        f"Frame {i}, key '{key}': numpy type {type_name} leaked"
                    )

    def test_capture_pipeline_metrics_csv_json_identical_content(
        self, default_config, tmp_path
    ):
        """IT1: CSV and JSON contain the same metric values (cross-format parity).

        After running the same capture, CSV and JSON must agree on values.
        """
        import csv

        from pymurmur.capture.recorder import Recorder

        cfg = default_config
        cfg.num_boids = 10
        cfg.seed = 99
        cfg.capture_with_viz = False
        cfg.capture_every = 1
        cfg.capture_metrics_csv = str(tmp_path / "parity.csv")
        cfg.capture_metrics_json = str(tmp_path / "parity.json")

        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)
        engine.run_headless(steps=5, callback=rec.on_frame)
        rec.save_metrics_csv()
        rec.save_metrics_json()

        with open(cfg.capture_metrics_csv) as f:
            csv_data = list(csv.DictReader(f))
        with open(cfg.capture_metrics_json) as f:
            json_data = json.load(f)["metrics"]

        assert len(csv_data) == len(json_data)
        for i in range(len(csv_data)):
            for key in csv_data[i]:
                csv_val = csv_data[i][key]
                json_val = json_data[i][key]
                # Both None → ok
                if csv_val == "" and json_val is None:
                    continue
                # JSON may have list (angular_momentum), CSV may have string
                if isinstance(json_val, list):
                    # Lists don't round-trip cleanly through CSV, but verify length
                    assert len(json_val) == 3, (
                        f"Frame {i}, key '{key}': expected 3-element list, "
                        f"got {len(json_val)} elements"
                    )
                    continue
                # Numeric comparison (CSV strings vs JSON numbers)
                try:
                    assert float(csv_val) == pytest.approx(float(json_val), rel=1e-5), (
                        f"Frame {i}, key '{key}': CSV={csv_val}, JSON={json_val}"
                    )
                except (ValueError, TypeError):
                    # Non-numeric — compare as strings
                    assert str(csv_val) == str(json_val), (
                        f"Frame {i}, key '{key}': CSV={csv_val}, JSON={json_val}"
                    )

    @pytest.mark.gpu
    def test_capture_dimensions_wired_correctly(
        self, default_config, gpu_available, tmp_path
    ):
        """IT1 (GPU): Captured frames use capture_width/height, not window dims.

        I6.2: Recorder passes capture_width/capture_height to Visualizer.
        The FBO must render at capture dimensions, not window dimensions.
        """
        if not gpu_available:
            pytest.skip("GPU not available")

        from pymurmur.capture.recorder import Recorder

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_with_viz = True
        cfg.capture_every = 1
        # Set capture dims different from window dims
        cfg.window_width = 1600
        cfg.window_height = 1200
        cfg.capture_width = 640
        cfg.capture_height = 480
        cfg.capture_prewarm = 0  # P8.7: capture from the first frame

        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)

        # Run 3 frames with internal FBO capture
        engine.run_headless(steps=3, callback=rec.on_frame)

        # Verify frames were captured
        assert len(rec.frames) >= 1, "Expected at least 1 captured frame"

        # Each captured frame must be at capture dimensions, not window
        for i, frame in enumerate(rec.frames):
            assert frame is not None, f"Frame {i} is None"
            assert frame.width == cfg.capture_width, (
                f"Frame {i} width={frame.width}, expected capture_width={cfg.capture_width} "
                f"(not window_width={cfg.window_width})"
            )
            assert frame.height == cfg.capture_height, (
                f"Frame {i} height={frame.height}, expected capture_height={cfg.capture_height} "
                f"(not window_height={cfg.window_height})"
            )

        # Also verify save_gif produces a valid GIF at half capture dims (LANCZOS)
        out_path = tmp_path / "capture_dims.gif"
        result = rec.save_gif(path=str(out_path), fps=10)
        assert result is not None
        assert out_path.exists()
        assert out_path.stat().st_size > 0

        from PIL import Image
        with Image.open(out_path) as gif:
            # LANCZOS halves dimensions
            assert gif.width == cfg.capture_width // 2, (
                f"GIF width={gif.width}, expected {cfg.capture_width // 2}"
            )
            assert gif.height == cfg.capture_height // 2, (
                f"GIF height={gif.height}, expected {cfg.capture_height // 2}"
            )

    @pytest.mark.gpu
    def test_capture_pipeline_gif_contains_correct_frame_count(
        self, default_config, gpu_available, tmp_path
    ):
        """IT1 (GPU): save_gif() produces GIF with expected number of frames."""
        if not gpu_available:
            pytest.skip("GPU not available")

        from PIL import Image

        from pymurmur.capture.recorder import Recorder

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_with_viz = True
        cfg.capture_every = 2  # capture every 2nd frame
        cfg.capture_width = 320
        cfg.capture_height = 240
        cfg.capture_prewarm = 0  # P8.7: capture from the first frame

        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)

        # Run 10 frames — should capture 5 frames (every 2nd)
        engine.run_headless(steps=10, callback=rec.on_frame)

        assert len(rec.frames) == 5, (
            f"Expected 5 frames (10 steps, every=2), got {len(rec.frames)}"
        )

        out_path = tmp_path / "frame_count.gif"
        rec.save_gif(path=str(out_path), fps=10)

        with Image.open(out_path) as gif:
            assert gif.n_frames == 5

    @pytest.mark.gpu
    def test_capture_pipeline_with_large_capture_dimensions(
        self, default_config, gpu_available, tmp_path
    ):
        """IT1 (GPU): Large capture dimensions (1920×1080) work end-to-end."""
        if not gpu_available:
            pytest.skip("GPU not available")

        from pymurmur.capture.recorder import Recorder

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_with_viz = True
        cfg.capture_every = 1
        cfg.capture_width = 1920
        cfg.capture_height = 1080
        cfg.capture_prewarm = 0  # P8.7: capture from the first frame

        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)

        engine.run_headless(steps=2, callback=rec.on_frame)

        assert len(rec.frames) == 2
        for f in rec.frames:
            assert f.width == 1920
            assert f.height == 1080

        out_path = tmp_path / "large_dims.gif"
        result = rec.save_gif(path=str(out_path))
        assert result is not None
        assert out_path.stat().st_size > 0


