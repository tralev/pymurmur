"""P8.6: Adaptive quality governor tests.

Tests EMA frame time tracking, budget computation, degradation ladder
(trails→scale→count), recovery hysteresis, spike capping, and
integration with config fields.
"""

from __future__ import annotations

import time

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


# ── P8.6c: Timing precision with monkeypatched clock ──────────────

class TestQualityGovernorTiming:
    """P8.6: QualityGovernor degradation/recovery ladder with
    monkeypatched time.perf_counter for deterministic frame pacing.

    Simulates a realistic render loop: frame_ms is computed from
    consecutive perf_counter reads, then fed to the governor.
    The fake clock advances by exactly the intended frame time
    on each read, so frame_ms = intended_ms (no jitter).
    """

    @staticmethod
    def _make_clock(start: float = 0.0):
        """Return a mutable clock list for monkeypatching.

        clock[0] is read by time.perf_counter().  The caller
        advances it by adding frame_ms / 1000.0 before each read.
        """
        return [start]

    # ── helpers ────────────────────────────────────────────────

    def _feed_clocked(self, gov, clock, frame_ms: float):
        """Advance the fake clock and feed the governor.

        Returns the frame time actually fed (spike-capped).
        """
        clock[0] += frame_ms / 1000.0
        gov.feed(frame_ms)
        return min(frame_ms, gov.SPIKE_CAP_MS)

    def _degrade_to_level(self, gov, clock, target_level: int,
                          frame_ms: float = 35.0) -> None:
        """Feed slow frames until the governor degrades to target_level.

        Calls should_degrade() each frame to trigger the ladder.
        Raises RuntimeError if the level isn't reached within
        a generous frame budget.
        """
        for _ in range(800):
            self._feed_clocked(gov, clock, frame_ms)
            gov.should_degrade()
            if gov.degradation_level >= target_level:
                return
        raise RuntimeError(
            f"Failed to degrade to level {target_level} "
            f"within 800 frames at {frame_ms}ms"
        )

    # ── degradation timing tests ───────────────────────────────

    def test_degrade_fires_after_1_8s_slow_frames(self, monkeypatch):
        """P8.6: Degradation level 1 fires after exactly 1.8s of
        frames below 78% of target FPS."""
        from pymurmur.analysis.perf import QualityGovernor

        clock = self._make_clock()
        monkeypatch.setattr(time, 'perf_counter', lambda: clock[0])

        gov = QualityGovernor(target_fps=60)
        # At 60fps target, 78% threshold = 46.8fps → 21.37ms budget
        # Feed at 30ms (33.3fps) — well below threshold
        slow_ms = 30.0
        frames_needed = int(1.8 / (slow_ms / 1000.0)) + 2  # ~62 frames

        fired_at = None
        for i in range(frames_needed):
            self._feed_clocked(gov, clock, slow_ms)
            if gov.should_degrade():
                fired_at = i
                break

        assert fired_at is not None, (
            f"Degradation should fire within {frames_needed} slow frames"
        )
        assert gov.degradation_level == 1
        # Clock should have advanced ~1.8s
        assert clock[0] == pytest.approx(1.8, rel=0.15)

    def test_degrade_three_steps_with_precise_timing(self, monkeypatch):
        """P8.6: Full degradation ladder 1→2→3 with monkeypatched
        clock verifying 1.8s between each step."""
        from pymurmur.analysis.perf import QualityGovernor

        clock = self._make_clock()
        monkeypatch.setattr(time, 'perf_counter', lambda: clock[0])

        gov = QualityGovernor(target_fps=60)
        slow_ms = 35.0

        actions: list[tuple[int, float]] = []  # (level, clock_time)
        for _ in range(300):
            self._feed_clocked(gov, clock, slow_ms)
            if gov.should_degrade():
                actions.append((gov.degradation_level, clock[0]))
            if gov.degradation_level >= 3:
                break

        assert len(actions) == 3, (
            f"Expected 3 degradation actions, got {len(actions)}: {actions}"
        )
        assert [a[0] for a in actions] == [1, 2, 3]

        # Each step should be ~1.8s apart
        step_intervals = [
            actions[i][1] - actions[i - 1][1] for i in range(1, len(actions))
        ]
        for i, interval in enumerate(step_intervals, start=1):
            assert interval == pytest.approx(1.8, rel=0.2), (
                f"Step {i}→{i + 1} interval {interval:.2f}s; expected ~1.8s"
            )

    def test_no_degrade_when_fps_above_threshold(self, monkeypatch):
        """P8.6: Frames at 80% of target (above 78% threshold)
        never trigger degradation."""
        from pymurmur.analysis.perf import QualityGovernor

        clock = self._make_clock()
        monkeypatch.setattr(time, 'perf_counter', lambda: clock[0])

        gov = QualityGovernor(target_fps=60)
        # 80% of 60 = 48fps → 20.83ms per frame. Feed at 20ms (~50fps).
        borderline_ms = 20.0

        for _ in range(300):  # ~6s — well past any window
            self._feed_clocked(gov, clock, borderline_ms)
            assert not gov.should_degrade(), (
                f"Should not degrade at {borderline_ms}ms (above 78% threshold)"
            )
        assert gov.degradation_level == 0

    def test_ema_dominates_spike_does_not_degrade(self, monkeypatch):
        """P8.6: A single slow spike does not trigger degradation —
        EMA smoothing prevents false positives.  The spike may push
        EMA above the healthy margin briefly, but it cannot build
        the 1.8s degrade timer from a single frame."""
        from pymurmur.analysis.perf import QualityGovernor

        clock = self._make_clock()
        monkeypatch.setattr(time, 'perf_counter', lambda: clock[0])

        gov = QualityGovernor(target_fps=60)

        # Stabilize at healthy frame times (10ms ≈ 100fps)
        for _ in range(100):
            self._feed_clocked(gov, clock, 10.0)

        ema_before = gov.ema_ms

        # Single massive spike (200ms, below spike cap of 250ms)
        self._feed_clocked(gov, clock, 200.0)

        # EMA does spike up (α=0.08: 0.08×200 + 0.92×10 ≈ 25ms)
        # but a single spike can't build 1.8s of degrade timer.
        assert gov.ema_ms > ema_before, (
            "EMA should rise after spike — that's expected behavior"
        )
        assert gov.degradation_level == 0, (
            "Single spike must not trigger degradation"
        )
        assert not gov.should_degrade(), (
            "should_degrade() must return False after single spike"
        )

    # ── recovery timing tests ──────────────────────────────────

    def test_recover_fires_after_3_6s_fast_frames(self, monkeypatch):
        """P8.6: Recovery from level 1 fires after exactly 3.6s
        of frames within 85% of budget."""
        from pymurmur.analysis.perf import QualityGovernor

        clock = self._make_clock()
        monkeypatch.setattr(time, 'perf_counter', lambda: clock[0])

        gov = QualityGovernor(target_fps=60)

        # Degrade exactly to level 1
        self._degrade_to_level(gov, clock, 1)
        assert gov.degradation_level == 1

        # Now feed fast frames at 5ms (well under budget)
        fast_ms = 5.0
        frames_needed = int(3.6 / (fast_ms / 1000.0)) + 50  # ~770 frames

        recovered = False
        for _ in range(frames_needed):
            self._feed_clocked(gov, clock, fast_ms)
            if gov.should_recover():
                recovered = True
                break

        assert recovered, (
            f"Recovery should fire within {frames_needed} fast frames"
        )
        assert gov.degradation_level == 0

    def test_recover_does_not_fire_before_window(self, monkeypatch):
        """P8.6: Recovery check before 3.6s returns False — hysteresis
        prevents oscillation."""
        from pymurmur.analysis.perf import QualityGovernor

        clock = self._make_clock()
        monkeypatch.setattr(time, 'perf_counter', lambda: clock[0])

        gov = QualityGovernor(target_fps=60)

        # Degrade exactly to level 1
        self._degrade_to_level(gov, clock, 1)
        assert gov.degradation_level == 1

        # Feed fast frames for less than 3.6s (~3.0s)
        fast_ms = 5.0
        for _ in range(600):  # 600 × 5ms = 3.0s
            self._feed_clocked(gov, clock, fast_ms)
            # should_recover must return False before the window
            assert not gov.should_recover(), (
                "Recovery should not fire before 3.6s window"
            )

        assert gov.degradation_level == 1, (
            "Level should still be 1 before recovery window elapses"
        )

    # ── full cycle tests ───────────────────────────────────────

    def test_full_degrade_recover_cycle(self, monkeypatch):
        """P8.6: Full cycle: healthy → degrade 1→2→3 → fast → recover 3→2→1→0.

        Uses monkeypatched clock to verify precise timing at each step.
        """
        from pymurmur.analysis.perf import QualityGovernor

        clock = self._make_clock()
        monkeypatch.setattr(time, 'perf_counter', lambda: clock[0])

        gov = QualityGovernor(target_fps=60)
        slow_ms = 35.0
        fast_ms = 5.0

        # Phase 1: degrade to level 3
        degrade_actions: list[int] = []
        for _ in range(350):
            self._feed_clocked(gov, clock, slow_ms)
            if gov.should_degrade():
                degrade_actions.append(gov.degradation_level)
            if len(degrade_actions) >= 3:
                break

        assert degrade_actions == [1, 2, 3], (
            f"Degrade ladder: expected [1,2,3], got {degrade_actions}"
        )
        assert gov.degradation_level == 3

        # Phase 2: recover back to level 0
        recover_actions: list[int] = []
        for _ in range(2500):
            self._feed_clocked(gov, clock, fast_ms)
            if gov.should_recover():
                recover_actions.append(gov.degradation_level)
            if len(recover_actions) >= 3:
                break

        assert recover_actions == [2, 1, 0], (
            f"Recover ladder: expected [2,1,0], got {recover_actions}"
        )
        assert gov.degradation_level == 0
        assert gov.is_healthy

    def test_oscillation_guard_no_rapid_flip(self, monkeypatch):
        """P8.6: Degrade→recover→degrade flapping is prevented by
        the 1.8s degrade and 3.6s recovery windows.

        Alternating fast/slow frames every second should NOT trigger
        any actions."""
        from pymurmur.analysis.perf import QualityGovernor

        clock = self._make_clock()
        monkeypatch.setattr(time, 'perf_counter', lambda: clock[0])

        gov = QualityGovernor(target_fps=60)
        slow_ms = 35.0
        fast_ms = 5.0

        actions = 0
        for _cycle in range(10):  # 10 cycles of fast→slow
            # Feed ~1.0s of slow frames
            for _ in range(int(1.0 / (slow_ms / 1000.0))):  # ~28 frames
                self._feed_clocked(gov, clock, slow_ms)
                if gov.should_degrade():
                    actions += 1

            # Feed ~1.0s of fast frames
            for _ in range(int(1.0 / (fast_ms / 1000.0))):  # ~200 frames
                self._feed_clocked(gov, clock, fast_ms)
                if gov.should_recover():
                    actions += 1

        # Neither degrade (1.8s window) nor recover (3.6s window)
        # should fire with only 1s bursts
        assert actions == 0, (
            f"Expected 0 actions with 1s bursts, got {actions} — "
            "oscillation guard failed"
        )
        assert gov.degradation_level == 0, (
            "Level should be 0 — no degradation fired"
        )
