"""I6 Phase — Integration tests for the capture pipeline.

Crosses multiple I6 item boundaries:
  IT1 — Full capture pipeline with dimensions + serialization
    - Capture dimensions wired through Recorder → Visualizer → FBO
    - Metrics serialization: CSV columns, JSON metadata, full round-trip
    - Step → on_frame → snapshot().to_dict() → save pipeline
  IT2 — Buffer growth during capture
    - Metrics buffer grows every frame
    - Frame buffer grows every capture_every (when with_viz)
    - Buffers persist after save (not cleared)
    - Buffer growth under mutations (add/remove mid-capture)
"""

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pymurmur.simulation.engine import SimulationEngine


# ── IT1: Full capture pipeline (dimensions + serialization) ───────

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
        from pymurmur.capture.recorder import Recorder
        import csv

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

        from pymurmur.capture.recorder import Recorder
        from PIL import Image

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_with_viz = True
        cfg.capture_every = 2  # capture every 2nd frame
        cfg.capture_width = 320
        cfg.capture_height = 240

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


# ── IT2: Buffer growth during capture ─────────────────────────────

class TestBufferGrowthDuringCapture:
    """I6.1 + I6.3: Buffer growth patterns during capture."""

    def test_metrics_buffer_grows_every_frame(self, default_config):
        """IT2: metrics_history grows by exactly 1 per on_frame() call."""
        from pymurmur.capture.recorder import Recorder

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_with_viz = False
        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)

        # Track buffer sizes step by step
        sizes = []
        for i in range(10):
            engine.step(1.0 / 60)
            rec.on_frame(engine)
            sizes.append((len(rec.metrics_history), rec._frame_count))
            assert len(rec.metrics_history) == i + 1, (
                f"After frame {i + 1}: expected {i + 1} metrics entries, "
                f"got {len(rec.metrics_history)}"
            )
            assert rec._frame_count == i + 1, (
                f"After frame {i + 1}: expected _frame_count={i + 1}, "
                f"got {rec._frame_count}"
            )

        assert len(rec.metrics_history) == 10
        assert rec._frame_count == 10

    def test_frame_buffer_grows_only_at_capture_every_intervals(
        self, default_config
    ):
        """IT2 (CPU): _capture_frame is called only at capture_every intervals.

        Spy on _capture_frame to verify the Recorder's gating logic
        calls it exactly at frames 3, 6, 9, 12 (every=3). Metrics
        are still captured every frame regardless.
        """
        from pymurmur.capture.recorder import Recorder

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_with_viz = True  # triggers _capture_frame branch
        cfg.capture_every = 3       # every 3rd frame
        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)

        # Spy on _capture_frame so GPU never actually runs.
        # Guard orig_capture call — ModernGL may raise non-RuntimeError
        # exceptions (moderngl.Error) that the Recorder's except clause
        # doesn't catch, so we safely swallow anything here.
        capture_calls = []
        orig_capture = rec._capture_frame
        def spy_capture(sim):
            capture_calls.append(rec._frame_count)
            try:
                return orig_capture(sim)
            except Exception:
                pass  # no GPU — safe, same intent as Recorder's except RuntimeError
        rec._capture_frame = spy_capture

        for i in range(12):
            engine.step(1.0 / 60)
            rec.on_frame(engine)

        assert rec._frame_count == 12
        assert len(rec.metrics_history) == 12  # every frame
        # _capture_frame called at frames 3, 6, 9, 12
        assert capture_calls == [3, 6, 9, 12], (
            f"capture_every=3 should call _capture_frame at frames 3,6,9,12. "
            f"Got: {capture_calls}"
        )

    @pytest.mark.gpu
    def test_frame_buffer_grows_with_capture_every_gating(
        self, default_config, gpu_available
    ):
        """IT2 (GPU): frames list grows only at capture_every intervals."""
        if not gpu_available:
            pytest.skip("GPU not available")

        from pymurmur.capture.recorder import Recorder

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_with_viz = True
        cfg.capture_every = 4  # capture every 4th frame
        cfg.capture_width = 320
        cfg.capture_height = 240

        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)

        # Run 12 frames — should capture at frames 4, 8, 12 = 3 frames
        engine.run_headless(steps=12, callback=rec.on_frame)

        assert rec._frame_count == 12
        assert len(rec.metrics_history) == 12  # every frame
        assert len(rec.frames) == 3, (
            f"Expected 3 frames (12 steps, every=4), got {len(rec.frames)}"
        )

    @pytest.mark.gpu
    def test_buffer_growth_step_by_step_tracking(
        self, default_config, gpu_available
    ):
        """IT2 (GPU): Track buffer sizes after each individual step+on_frame.

        Verifies deterministic growth with no gaps or jumps.
        """
        if not gpu_available:
            pytest.skip("GPU not available")

        from pymurmur.capture.recorder import Recorder

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_with_viz = True
        cfg.capture_every = 3
        cfg.capture_width = 320
        cfg.capture_height = 240

        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)

        growth_log = []
        for i in range(9):
            engine.step(1.0 / 60)
            rec.on_frame(engine)
            growth_log.append({
                "frame": i + 1,
                "_frame_count": rec._frame_count,
                "metrics_count": len(rec.metrics_history),
                "frames_count": len(rec.frames),
            })

        # Verify step-by-step metrics growth (always +1)
        for i, entry in enumerate(growth_log):
            assert entry["metrics_count"] == i + 1, (
                f"Frame {i + 1}: metrics_count={entry['metrics_count']}"
            )

        # Frames only at multiples of capture_every (3)
        for entry in growth_log:
            expected_frames = entry["frame"] // 3
            assert entry["frames_count"] == expected_frames, (
                f"Frame {entry['frame']}: expected {expected_frames} captured "
                f"frames, got {entry['frames_count']}"
            )

    def test_buffers_persist_after_save(self, default_config, tmp_path):
        """IT2: After save_gif/save_metrics, buffers are not cleared.

        The Recorder's internal lists must persist after export so callers
        can inspect them post-capture or save to multiple formats.
        """
        from pymurmur.capture.recorder import Recorder

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_with_viz = False
        cfg.capture_every = 1
        cfg.capture_metrics_csv = str(tmp_path / "persist.csv")
        cfg.capture_metrics_json = str(tmp_path / "persist.json")

        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)
        engine.run_headless(steps=5, callback=rec.on_frame)

        metrics_before = len(rec.metrics_history)
        frames_before = len(rec.frames)

        # Save all formats
        rec.save_metrics_csv()
        rec.save_metrics_json()
        rec.save_gif()

        # Buffers must be unchanged after save
        assert len(rec.metrics_history) == metrics_before, (
            f"metrics_history changed after save: {metrics_before} → "
            f"{len(rec.metrics_history)}"
        )
        assert len(rec.frames) == frames_before, (
            f"frames changed after save: {frames_before} → {len(rec.frames)}"
        )
        assert rec._frame_count == 5, (
            f"_frame_count changed after save: expected 5, got {rec._frame_count}"
        )

    def test_buffer_growth_under_command_mutations(self, default_config):
        """IT2: metrics_history continues growing correctly across mutations.

        After add/remove mid-capture, the metrics buffer must continue
        growing by 1 per frame without gaps or resets.
        """
        from pymurmur.capture.recorder import Recorder

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_with_viz = False
        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)

        # Phase 1: run 5 frames normally
        engine.run_headless(steps=5, callback=rec.on_frame)
        assert len(rec.metrics_history) == 5
        assert rec._frame_count == 5

        # Verify metrics content is valid for the first phase
        assert rec.metrics_history[0]["alpha"] is not None
        assert rec.metrics_history[4]["speed_avg"] is not None

        # Phase 2: enqueue add, run 3 more frames
        engine.enqueue_add(5)
        engine.run_headless(steps=3, callback=rec.on_frame)
        assert len(rec.metrics_history) == 8
        assert rec._frame_count == 8

        # Phase 3: enqueue remove, run 2 more frames
        engine.enqueue_remove(2)
        engine.run_headless(steps=2, callback=rec.on_frame)
        assert len(rec.metrics_history) == 10
        assert rec._frame_count == 10

        # All 10 entries must be present (no gaps from mutation drain)
        assert all(
            isinstance(entry, dict) for entry in rec.metrics_history
        ), "All metrics entries must be dicts"

        # Verify growth was strictly monotonic
        for i in range(1, 10):
            assert len(rec.metrics_history[i - 1]) == len(rec.metrics_history[i]), (
                f"Metrics schema changed between frames {i} and {i + 1}"
            )

    def test_zero_frame_capture_buffer_remains_empty(self, default_config):
        """IT2: With 0 steps, both buffers stay empty — no phantom entries."""
        from pymurmur.capture.recorder import Recorder

        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)

        # Don't call on_frame at all
        assert rec.frames == []
        assert rec.metrics_history == []
        assert rec._frame_count == 0

        # Save operations must return None gracefully
        assert rec.save_gif() is None
        assert rec.save_metrics_csv() is None
        assert rec.save_metrics_json() is None

    @pytest.mark.gpu
    def test_buffer_growth_across_recorder_reuse(
        self, default_config, gpu_available, tmp_path
    ):
        """IT2 (GPU): A single Recorder reused across two headless runs accumulates correctly.

        Both runs append to the same buffers in sequence.
        """
        if not gpu_available:
            pytest.skip("GPU not available")

        from pymurmur.capture.recorder import Recorder

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_with_viz = True
        cfg.capture_every = 1
        cfg.capture_width = 320
        cfg.capture_height = 240

        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)

        # Run 1: 3 frames
        engine.run_headless(steps=3, callback=rec.on_frame)
        assert len(rec.metrics_history) == 3
        assert len(rec.frames) == 3

        # Run 2: 2 more frames — appends to existing buffers
        engine.run_headless(steps=2, callback=rec.on_frame)
        assert len(rec.metrics_history) == 5, (
            f"After reuse: expected 5 metrics, got {len(rec.metrics_history)}"
        )
        assert len(rec.frames) == 5, (
            f"After reuse: expected 5 frames, got {len(rec.frames)}"
        )
        assert rec._frame_count == 5

        # Save GIF with accumulated frames
        out_path = tmp_path / "reuse.gif"
        result = rec.save_gif(path=str(out_path), fps=10)
        assert result is not None

        from PIL import Image
        with Image.open(out_path) as gif:
            assert gif.n_frames == 5
