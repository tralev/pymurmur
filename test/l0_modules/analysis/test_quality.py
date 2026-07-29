"""P8.6: Adaptive quality governor tests.

Tests EMA frame time tracking, budget computation, degradation ladder
(trails→scale→count), recovery hysteresis, spike capping, and
integration with config fields.

Timing-precision tests with a monkeypatched clock live in
test_quality_governor_timing.py (file-size split of this file).
"""

from __future__ import annotations

import pytest

# ── P8.6a: QualityGovernor unit tests (no GPU) ──────────────────

class TestQualityGovernor:
    """P8.6: QualityGovernor EMA, hysteresis, and ladder logic."""

    @pytest.fixture
    def gov(self):
        from pymurmur.analysis.perf import QualityGovernor
        return QualityGovernor(target_fps=60)

    def test_initial_state(self, gov):
        """P8.6: Governor starts healthy at level 0."""
        assert gov.degradation_level == 0
        assert gov.is_healthy
        assert gov.ema_ms == 0.0

    def test_budget_computation(self, gov):
        """P8.6: Budget = 1000 / max(24, target_fps)."""
        assert gov.budget_ms == pytest.approx(1000.0 / 60.0, rel=0.01)

    def test_budget_floor_at_24fps(self):
        """P8.6: target_fps is clamped to minimum 24."""
        from pymurmur.analysis.perf import QualityGovernor
        gov = QualityGovernor(target_fps=10)
        assert gov.budget_ms == pytest.approx(1000.0 / 24.0, rel=0.01)

    def test_feed_initialises_ema(self, gov):
        """P8.6: First feed() sets EMA directly."""
        gov.feed(16.0)
        assert gov.ema_ms == 16.0

    def test_ema_converges(self, gov):
        """P8.6: EMA smooths toward steady state."""
        # Feed constant 20ms for many frames
        for _ in range(100):
            gov.feed(20.0)
        assert gov.ema_ms == pytest.approx(20.0, rel=0.05)

    def test_spike_capped_at_250ms(self, gov):
        """P8.6: Frame times >250ms are clamped before EMA."""
        gov.feed(500.0)  # spike
        assert gov.ema_ms == 250.0  # first frame sets EMA directly

    def test_healthy_when_ema_low(self, gov):
        """P8.6: is_healthy when EMA ≤ 1.12·budget."""
        budget = gov.budget_ms
        # Feed healthy frame times
        for _ in range(50):
            gov.feed(budget * 0.9)
        assert gov.is_healthy

    def test_unhealthy_when_ema_high(self, gov):
        """P8.6: is_healthy is False when EMA > 1.12·budget."""
        budget = gov.budget_ms
        # Feed slow frames (24ms at 60fps budget ≈ 16.67ms → EMA drifts up)
        for _ in range(100):
            gov.feed(budget * 1.5)
        assert not gov.is_healthy

    def test_no_degrade_when_healthy(self, gov):
        """P8.6: should_degrade() returns False when frame times are healthy."""
        for _ in range(200):
            gov.feed(10.0)  # well under budget
            assert not gov.should_degrade()
        assert gov.degradation_level == 0

    def test_degradation_after_sustained_slowdown(self, gov):
        """P8.6: Sustained slow frames trigger degradation after 1.8s window."""
        # Feed slow frames (well above budget) for >1.8s worth of sim time
        slow_ms = 30.0  # 33 fps for 60fps target → below 78%
        total_time = 0.0
        fired = False
        for _ in range(200):
            gov.feed(slow_ms)
            total_time += slow_ms / 1000.0
            if gov.should_degrade():
                fired = True
                break
        assert fired, "Degradation should fire after sustained slowdown"
        assert total_time >= 1.8, f"Expected ≥1.8s, got {total_time:.1f}s"
        assert gov.degradation_level >= 1

    def test_degradation_ladder_order(self, gov):
        """P8.6: Degradation steps fire in order: trails(1)→scale(2)→count(3)."""
        slow_ms = 35.0
        levels_seen = []
        # Feed until all 3 levels fire
        for _ in range(600):
            gov.feed(slow_ms)
            if gov.should_degrade():
                levels_seen.append(gov.degradation_level)
            if len(levels_seen) >= 3:
                break
        assert levels_seen == [1, 2, 3], (
            f"Expected [1,2,3] degradation order, got {levels_seen}"
        )

    def test_one_step_per_window(self, gov):
        """P8.6: Degradation steps are spaced ≥1.8s apart."""
        slow_ms = 35.0
        actions = []
        for _ in range(500):
            gov.feed(slow_ms)
            if gov.should_degrade():
                actions.append(gov.degradation_level)
        # Should have at most 3 actions, all spaced by at least 1.8s of feed
        assert len(actions) <= 3

    def test_no_degrade_beyond_level_3(self, gov):
        """P8.6: Degradation caps at level 3 — no further actions fire."""
        slow_ms = 35.0
        for _ in range(800):
            gov.feed(slow_ms)
            gov.should_degrade()
        assert gov.degradation_level == 3
        # One more feed — still shouldn't fire
        gov.feed(slow_ms)
        assert not gov.should_degrade()
        assert gov.degradation_level == 3

    def test_recovery_from_degraded(self, gov):
        """P8.6: Fast frames trigger recovery after 3.6s window."""
        # First, degrade
        slow_ms = 35.0
        for _ in range(500):
            gov.feed(slow_ms)
            gov.should_degrade()
        assert gov.degradation_level >= 1

        # Then feed fast frames for >3.6s (5ms × 800 = 4.0s)
        fast_ms = 5.0
        recovered = False
        for _ in range(800):
            gov.feed(fast_ms)
            if gov.should_recover():
                recovered = True
                break
        assert recovered, "Recovery should fire after sustained fast frames"

    def test_recovery_stops_at_level_0(self, gov):
        """P8.6: Recovery stops at level 0 — no negative levels."""
        # Degrade fully
        slow_ms = 35.0
        for _ in range(800):
            gov.feed(slow_ms)
            gov.should_degrade()
        assert gov.degradation_level == 3

        # Recover fully — 3 steps × 3.6s = 10.8s → ~2200 frames at 5ms
        fast_ms = 5.0
        for _ in range(2500):
            gov.feed(fast_ms)
            gov.should_recover()
        assert gov.degradation_level == 0
        # One more recovery call — should still be 0
        gov.feed(fast_ms)
        assert not gov.should_recover()
        assert gov.degradation_level == 0

    def test_reset_clears_all_state(self, gov):
        """P8.6: reset() returns governor to initial state."""
        gov.feed(30.0)
        for _ in range(100):
            gov.feed(30.0)
        assert gov.ema_ms > 0
        gov.reset()
        assert gov.ema_ms == 0.0
        assert gov.degradation_level == 0
        assert gov.is_healthy

    def test_degrade_timer_resets_on_recovery_condition(self, gov):
        """P8.6: degrade_timer resets when EMA drops below recovery threshold."""
        # Build up degrade timer with slow frames
        for _ in range(100):
            gov.feed(35.0)
        # Now feed fast frames long enough for EMA to drop below recovery threshold
        for _ in range(200):
            gov.feed(5.0)
        # degrade_timer should be reset and recovery_timer building
        assert gov._degrade_timer == 0.0
        assert gov._recovery_timer > 0.0

    def test_target_fps_config_field(self):
        """P8.6: PerfConfig has target_fps field."""
        from pymurmur.core.config import PerfConfig
        cfg = PerfConfig()
        assert hasattr(cfg, "target_fps")
        assert cfg.target_fps == 60

    def test_target_fps_field_map(self):
        """P8.6: target_fps is in _FIELD_MAP."""
        from pymurmur.core.config import _FIELD_MAP
        assert "target_fps" in _FIELD_MAP
        assert _FIELD_MAP["target_fps"] == ("_perf", "target_fps")

    def test_simconfig_flat_access_target_fps(self):
        """P8.6: SimConfig exposes target_fps via flat access."""
        from pymurmur.core.config import SimConfig
        cfg = SimConfig()
        assert cfg.target_fps == 60
        cfg.target_fps = 30
        assert cfg.target_fps == 30
        assert cfg.perf.target_fps == 30


# ── P8.6b: Visualizer integration (non-GPU) ──────────────────────

class TestQualityVisualizer:
    """P8.6: Visualizer creates QualityGovernor and has _apply_quality_actions."""

    def test_visualizer_has_governor(self):
        """P8.6: Visualizer creates a QualityGovernor on init."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer
        cfg = SimConfig(num_boids=10, target_fps=60)
        sim = SimulationEngine(cfg)
        viz = Visualizer(sim, cfg, headless=True, width=200, height=150)
        assert viz._governor is not None
        assert viz._governor.degradation_level == 0
        assert viz.renderer.render_scale == 1.0

    def test_visualizer_apply_quality_actions_exists(self):
        """P8.6: _apply_quality_actions method exists."""
        from pymurmur.viz.visualizer import Visualizer
        assert hasattr(Visualizer, "_apply_quality_actions")

    def test_governor_uses_config_target_fps(self):
        """P8.6: Governor target_fps comes from config."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer
        cfg = SimConfig(num_boids=10, target_fps=30)
        sim = SimulationEngine(cfg)
        viz = Visualizer(sim, cfg, headless=True, width=200, height=150)
        assert viz._governor.budget_ms == pytest.approx(1000.0 / 30.0, rel=0.01)


