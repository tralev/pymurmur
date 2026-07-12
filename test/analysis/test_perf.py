"""Performance diagnostics tests — Phase 10.3 EMA timing, bottleneck classification.
"""

import time

import numpy as np
import pytest

from pymurmur.analysis.perf import PerfDiagnostics, PerfSnapshot
from pymurmur.core.config import SimConfig
from pymurmur.simulation.engine import SimulationEngine


class TestEMA:
    """EMA smoothing of frame timing."""

    def test_first_frame_sets_ema_directly(self):
        """First tick uses raw value as EMA seed."""
        perf = PerfDiagnostics()
        perf.record_physics(10.0)
        perf.record_render(5.0)
        snap = perf.tick()
        assert snap.physics_ema == pytest.approx(10.0)
        assert snap.render_ema == pytest.approx(5.0)
        assert snap.total_ema == pytest.approx(15.0)

    def test_ema_converges_to_steady_value(self):
        """EMA converges to the input value over many frames."""
        perf = PerfDiagnostics()
        for _ in range(100):
            perf.record_physics(16.0)
            perf.record_render(8.0)
            perf.tick()
        snap = perf.snapshot()
        # After 100 frames, EMA should be close to input
        assert snap.physics_ema == pytest.approx(16.0, abs=0.5)
        assert snap.render_ema == pytest.approx(8.0, abs=0.5)

    def test_ema_changes_slowly(self):
        """EMA changes gradually — one spike doesn't dominate."""
        perf = PerfDiagnostics()
        # Stabilize at 16ms
        for _ in range(50):
            perf.record_physics(16.0)
            perf.record_render(0.0)
            perf.tick()
        before = perf.snapshot()

        # Single spike
        perf.record_physics(80.0)
        perf.record_render(0.0)
        after_spike = perf.tick()

        # EMA should move toward 80 but not jump fully
        alpha = PerfDiagnostics.ALPHA  # 0.08
        expected = alpha * 80.0 + (1 - alpha) * before.physics_ema
        assert after_spike.physics_ema == pytest.approx(expected, abs=0.01)


class TestBottleneck:
    """Bottleneck classification."""

    def test_cpu_bottleneck(self):
        """Physics > 60% of total -> CPU bottleneck."""
        perf = PerfDiagnostics()
        perf.record_physics(20.0)
        perf.record_render(5.0)  # 80% physics
        snap = perf.tick()
        assert snap.bottleneck == "cpu"
        assert snap.cpu_fraction > 0.6

    def test_gpu_bottleneck(self):
        """Render > 60% of total -> GPU bottleneck."""
        perf = PerfDiagnostics()
        perf.record_physics(5.0)
        perf.record_render(20.0)  # 20% physics
        snap = perf.tick()
        assert snap.bottleneck == "gpu"
        assert snap.cpu_fraction < 0.4

    def test_balanced(self):
        """Physics between 40-60% -> balanced."""
        perf = PerfDiagnostics()
        perf.record_physics(10.0)
        perf.record_render(10.0)  # 50% physics
        snap = perf.tick()
        assert snap.bottleneck == "balanced"

    def test_fps_computed_correctly(self):
        """FPS = 1000 / total_ms."""
        perf = PerfDiagnostics()
        perf.record_physics(10.0)
        perf.record_render(6.67)
        snap = perf.tick()
        assert snap.fps == pytest.approx(60.0, abs=1.0)


class TestAdaptiveQuality:
    """Adaptive quality triggers when FPS drops below threshold."""

    def test_no_adaptive_at_good_fps(self):
        """At 60fps, adaptive quality is off."""
        perf = PerfDiagnostics()
        for _ in range(50):
            perf.record_physics(10.0)
            perf.record_render(6.0)  # ~62fps
            perf.tick()
        snap = perf.snapshot()
        assert not snap.reduce_resolution
        assert not snap.reduce_count

    def test_adaptive_triggers_at_low_fps(self):
        """Below 75% of 60fps -> adaptive quality triggers."""
        perf = PerfDiagnostics()
        for _ in range(50):
            perf.record_physics(35.0)
            perf.record_render(20.0)  # ~18fps, 63% CPU
            perf.tick()
        snap = perf.snapshot()
        assert snap.reduce_count  # CPU bottleneck -> reduce count
        assert not snap.reduce_resolution


class TestEngineIntegration:
    """PerfDiagnostics wired into SimulationEngine."""

    def test_engine_records_physics_timing(self):
        """Engine with perf set records physics timing each step."""
        cfg = SimConfig()
        cfg.num_boids = 20
        cfg.mode = "spatial"

        from pymurmur.analysis.perf import PerfDiagnostics
        perf = PerfDiagnostics()
        sim = SimulationEngine(cfg)
        sim.perf = perf

        sim.run_headless(steps=5)
        # tick() advances frame counter; engine only records raw timing
        for _ in range(5):
            perf.tick()

        # After 5 steps + 5 ticks, should have recorded timing
        snap = perf.snapshot()
        assert snap.frame_count == 5
        assert snap.physics_ema > 0

    def test_engine_without_perf_still_works(self):
        """Engine runs fine without perf diagnostics attached."""
        cfg = SimConfig()
        cfg.num_boids = 20
        sim = SimulationEngine(cfg)
        # No perf set — should not crash
        sim.run_headless(steps=5)
        assert sim.frame == 5

    def test_perf_reset_on_engine_reset(self):
        """Engine reset also resets perf state."""
        cfg = SimConfig()
        cfg.num_boids = 20

        from pymurmur.analysis.perf import PerfDiagnostics
        perf = PerfDiagnostics()
        sim = SimulationEngine(cfg)
        sim.perf = perf

        sim.run_headless(steps=10)
        # tick to advance frame counter
        for _ in range(10):
            perf.tick()
        sim.reset()

        snap = perf.snapshot()
        assert snap.frame_count == 0
        assert snap.physics_ema == 0.0


class TestContextManager:
    """Timing context managers."""

    def test_measure_physics_context(self):
        """with perf.measure_physics() records timing."""
        perf = PerfDiagnostics()
        with perf.measure_physics():
            time.sleep(0.01)  # 10ms
        # No tick yet, just raw recording
        snap = perf.tick()
        # physics_ema should be ~10ms (from sleep)
        assert snap.physics_ema > 5.0
        assert snap.physics_ema < 20.0

    def test_measure_render_context(self):
        """with perf.measure_render() records timing."""
        perf = PerfDiagnostics()
        with perf.measure_render():
            time.sleep(0.005)  # 5ms
        snap = perf.tick()
        assert snap.render_ema > 2.0
        assert snap.render_ema < 15.0
