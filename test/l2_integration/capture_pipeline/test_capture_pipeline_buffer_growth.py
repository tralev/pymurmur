"""I6 — Integration test for the capture pipeline: buffer growth during capture.

Split out of test_capture_pipeline.py (file-size split).
"""


import pytest

from pymurmur.simulation.engine import SimulationEngine


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
        rec._prewarm = 0  # P8.7: no pre-warm for this test

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

        for _i in range(12):
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
        cfg.capture_prewarm = 0  # P8.7: capture from the first frame

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
        cfg.capture_prewarm = 0  # P8.7: capture from the first frame

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
        cfg.capture_prewarm = 0  # P8.7: capture from the first frame

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
