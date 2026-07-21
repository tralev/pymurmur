"""Performance diagnostics tests — Phase 10.3 EMA timing, bottleneck classification.
"""

import time

import pytest

from pymurmur.analysis.perf import PerfDiagnostics
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
        """Render > 60% of total -> GPU bottleneck (legacy); risk_class is "fragment" or "vertex" depending on N."""
        perf = PerfDiagnostics()
        perf.record_physics(5.0)
        perf.record_render(20.0)  # 20% physics
        perf.set_active_count(500)  # low N → fragment
        snap = perf.tick()
        assert snap.bottleneck == "gpu"  # legacy
        assert snap.cpu_fraction < 0.4
        assert snap.risk_class == "fragment"  # S4.10: low N → fragment-bound

    def test_gpu_bottleneck_high_n_vertex(self):
        """S4.10: GPU bottleneck with high N → risk_class is "vertex"."""
        perf = PerfDiagnostics()
        perf.record_physics(5.0)
        perf.record_render(20.0)
        perf.set_active_count(15_000)  # high N → vertex
        snap = perf.tick()
        assert snap.bottleneck == "gpu"  # legacy still says gpu
        assert snap.risk_class == "vertex"  # S4.10

    def test_cpu_bottleneck_risk_class(self):
        """S4.10: CPU bottleneck → risk_class is "cpu" regardless of N."""
        perf = PerfDiagnostics()
        perf.record_physics(20.0)
        perf.record_render(5.0)
        perf.set_active_count(50_000)  # high N doesn't matter for CPU
        snap = perf.tick()
        assert snap.bottleneck == "cpu"
        assert snap.risk_class == "cpu"  # S4.10

    def test_balanced_risk_class_mixed(self):
        """S4.10: Balanced timings → risk_class is "mixed"."""
        perf = PerfDiagnostics()
        perf.record_physics(10.0)
        perf.record_render(10.0)  # 50% physics
        perf.set_active_count(500)
        snap = perf.tick()
        assert snap.bottleneck == "balanced"
        assert snap.risk_class == "mixed"  # S4.10

    def test_risk_class_boundary_at_vertex_threshold(self):
        """S4.10: Exactly at VERTEX_N_THRESHOLD → "vertex"."""
        perf = PerfDiagnostics()
        perf.record_physics(5.0)
        perf.record_render(20.0)
        perf.set_active_count(perf.VERTEX_N_THRESHOLD)  # exactly 10K
        snap = perf.tick()
        assert snap.risk_class == "vertex"

    def test_risk_class_boundary_below_vertex_threshold(self):
        """S4.10: Just below VERTEX_N_THRESHOLD → "fragment"."""
        perf = PerfDiagnostics()
        perf.record_physics(5.0)
        perf.record_render(20.0)
        perf.set_active_count(perf.VERTEX_N_THRESHOLD - 1)  # 9999
        snap = perf.tick()
        assert snap.risk_class == "fragment"

    def test_adaptive_reduce_count_for_cpu_risk(self):
        """S4.10: "cpu" risk → reduce_count flag."""
        perf = PerfDiagnostics()
        perf.set_active_count(100)
        for _ in range(50):
            perf.record_physics(35.0)  # CPU-bound + slow → adaptive
            perf.record_render(20.0)
            perf.tick()
        snap = perf.snapshot()
        assert snap.risk_class == "cpu"
        assert snap.reduce_count  # S4.10: cpu → reduce count
        assert not snap.reduce_resolution

    def test_adaptive_reduce_resolution_for_fragment_risk(self):
        """S4.10: "fragment" risk → reduce_resolution flag."""
        perf = PerfDiagnostics()
        perf.set_active_count(500)  # low N → fragment
        for _ in range(50):
            perf.record_physics(10.0)  # GPU-bound + slow → adaptive
            perf.record_render(40.0)
            perf.tick()
        snap = perf.snapshot()
        assert snap.risk_class == "fragment"
        assert snap.reduce_resolution  # S4.10: fragment → reduce res
        assert not snap.reduce_count

    def test_snapshot_includes_n_active(self):
        """S4.10: PerfSnapshot.n_active reflects the last set_active_count call."""
        perf = PerfDiagnostics()
        perf.set_active_count(1234)
        perf.record_physics(10.0)
        perf.record_render(10.0)
        snap = perf.tick()
        assert snap.n_active == 1234

    def test_risk_class_defaults_to_mixed_without_set_count(self):
        """S4.10: Without set_active_count, balanced timings → "mixed" (N=0 < threshold → would be "fragment", but cpu_frac=0.5 → "mixed")."""
        perf = PerfDiagnostics()
        perf.record_physics(10.0)
        perf.record_render(10.0)
        snap = perf.tick()
        assert snap.risk_class == "mixed"
        assert snap.n_active == 0

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


# ── P8.3+P8.6: Trail quality degradation integration ─────────────

class TestTrailQualityIntegration:
    """P8.3+P8.6: Quality governor degradation disables/enables trails on Renderer3D.

    Verifies the full pipeline: QualityGovernor raises degradation_level,
    Renderer3D.disable_trails() cancels trail rendering, and
    enable_trails() restores it."""

    def test_gov_degrade_triggers_trail_disable(self):
        """P8.3+P8.6: Slow frames → should_degrade() fires → level 1.

        Feeds slow frames for >1.8s, then calls should_degrade()
        each frame to trigger the degradation ladder."""
        from pymurmur.analysis.perf import QualityGovernor

        gov = QualityGovernor(target_fps=60)
        # Feed slow frames (40ms ≈ 25fps, below 0.78×60=46.8 threshold)
        # DEGRADE_WINDOW = 1.8s → need ~45 frames at 40ms to exceed window
        for _ in range(60):
            gov.feed(40.0)
            gov.should_degrade()  # triggers level increment

        assert gov.degradation_level >= 1, (
            f"Expected degradation ≥ 1 after sustained slow frames, "
            f"got {gov.degradation_level}"
        )

    @pytest.mark.gpu
    def test_renderer_disable_trails_makes_trails_none(self, gpu_available):
        """P8.3+P8.6: disable_trails() sets _trails to None."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.renderer import Renderer3D

        r = Renderer3D(width=100, height=100, headless=True,
                       trails_mode="velocity", trails_length=30)
        assert r._trails is not None, "Trails should be active at init"
        r.disable_trails()
        assert r._trails is None, "disable_trails() must set _trails to None"

    @pytest.mark.gpu
    def test_renderer_enable_trails_restores(self, gpu_available):
        """P8.3+P8.6: enable_trails() recreates trail renderer after disable."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.renderer import Renderer3D

        r = Renderer3D(width=100, height=100, headless=True,
                       trails_mode="velocity", trails_length=30)
        r.disable_trails()
        assert r._trails is None
        r.enable_trails("velocity", 30)
        assert r._trails is not None, "enable_trails() must recreate trail renderer"

    @pytest.mark.gpu
    def test_enable_trails_off_mode_noop(self, gpu_available):
        """P8.3+P8.6: enable_trails("off", ...) does nothing (trails stay None)."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.renderer import Renderer3D

        r = Renderer3D(width=100, height=100, headless=True,
                       trails_mode="velocity", trails_length=30)
        r.disable_trails()
        r.enable_trails("off", 0)
        assert r._trails is None, "enable_trails('off') should not recreate trails"

    def test_draw_trails_noop_when_disabled(self, gpu_available):
        """P8.3+P8.6: draw_trails() is no-op when trails disabled."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock
        from pymurmur.viz.renderer import Renderer3D

        r = Renderer3D(width=100, height=100, headless=True,
                       trails_mode="velocity", trails_length=30)
        r.disable_trails()
        cfg = SimConfig(num_boids=5)
        flock = PhysicsFlock(cfg)
        flock.active[:] = True
        # Should not crash — trails are disabled
        r.draw_trails(flock)

    def test_recovery_makes_progress(self):
        """P8.3+P8.6: After degrading, fast frames move toward recovery."""
        from pymurmur.analysis.perf import QualityGovernor

        gov = QualityGovernor(target_fps=60)

        # Degrade to level >= 1
        for _ in range(200):
            gov.feed(40.0)
            gov.should_degrade()
        degraded_at = gov.degradation_level
        assert degraded_at >= 1, f"Should degrade, got {degraded_at}"

        # Feed fast frames + call should_recover repeatedly
        for _ in range(800):
            gov.feed(8.0)
            gov.should_recover()

        # After sustained recovery, level should be lower
        assert gov.degradation_level < degraded_at, (
            f"Should recover below {degraded_at}, got {gov.degradation_level}"
        )
        assert gov.is_healthy, "Governor should be healthy after recovery"
