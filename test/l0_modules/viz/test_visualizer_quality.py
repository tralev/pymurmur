"""P8.6 — Visualizer quality-degradation ladder (CPU, mocked renderer).

Drives Visualizer._apply_quality_actions() through the degradation
ladder (1: trails off, 2: render scale −0.15 floor 0.75, 3: N −18%
floor 512) and the recovery path, with Renderer3D and OrbitCamera
mocked out so no GPU is required.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from pymurmur.simulation.engine import SimulationEngine


def _make_viz(monkeypatch, cfg):
    """Visualizer with mocked GL parts; returns (viz, mock_renderer, sim)."""
    mock_renderer = MagicMock()
    mock_renderer.render_scale = 1.0
    monkeypatch.setattr(
        "pymurmur.viz.visualizer.Renderer3D", lambda **kw: mock_renderer)
    monkeypatch.setattr(
        "pymurmur.viz.visualizer.OrbitCamera", lambda **kw: MagicMock())
    from pymurmur.viz.visualizer import Visualizer

    sim = SimulationEngine(cfg)
    viz = Visualizer(sim, cfg, headless=True)
    return viz, mock_renderer, sim


def _make_governor(*, degrade=False, recover=False, level=0):
    """Governor stub with real numeric ladder constants."""
    from pymurmur.analysis.perf import QualityGovernor

    gov = MagicMock()
    gov.should_degrade.return_value = degrade
    gov.should_recover.return_value = recover
    gov.degradation_level = level
    gov.RENDER_SCALE_STEP = QualityGovernor.RENDER_SCALE_STEP
    gov.RENDER_SCALE_FLOOR = QualityGovernor.RENDER_SCALE_FLOOR
    gov.COUNT_STEP_FRACTION = QualityGovernor.COUNT_STEP_FRACTION
    gov.COUNT_FLOOR = QualityGovernor.COUNT_FLOOR
    return gov


class TestDegradationLadder:
    """P8.6: each degradation level triggers exactly its ladder action."""

    def test_level_1_disables_trails(self, default_config, monkeypatch):
        viz, renderer, _ = _make_viz(monkeypatch, default_config)
        viz._governor = _make_governor(degrade=True, level=1)
        viz._apply_quality_actions()
        renderer.disable_trails.assert_called_once()

    def test_level_2_steps_render_scale_down(self, default_config, monkeypatch):
        viz, renderer, _ = _make_viz(monkeypatch, default_config)
        viz._governor = _make_governor(degrade=True, level=2)
        viz._apply_quality_actions()
        assert renderer.render_scale == pytest.approx(0.85)

    def test_level_2_respects_scale_floor(self, default_config, monkeypatch):
        viz, renderer, _ = _make_viz(monkeypatch, default_config)
        renderer.render_scale = 0.80
        viz._governor = _make_governor(degrade=True, level=2)
        viz._apply_quality_actions()
        assert renderer.render_scale == pytest.approx(0.75)

    def test_level_3_enqueues_bird_removal(self, default_config, monkeypatch):
        cfg = default_config
        cfg.num_boids = 600
        viz, _, sim = _make_viz(monkeypatch, cfg)
        viz._governor = _make_governor(degrade=True, level=3)
        viz._apply_quality_actions()
        # target = max(int(600 * 0.82), 512) = 512 → remove 88
        assert sim.commands.pending_remove == 88

    def test_level_3_respects_count_floor(self, default_config, monkeypatch):
        cfg = default_config
        cfg.num_boids = 100  # already below COUNT_FLOOR=512
        viz, _, sim = _make_viz(monkeypatch, cfg)
        viz._governor = _make_governor(degrade=True, level=3)
        viz._apply_quality_actions()
        assert sim.commands.pending_remove == 0

    def test_healthy_is_a_noop(self, default_config, monkeypatch):
        viz, renderer, sim = _make_viz(monkeypatch, default_config)
        viz._governor = _make_governor()
        viz._apply_quality_actions()
        renderer.disable_trails.assert_not_called()
        renderer.enable_trails.assert_not_called()
        assert renderer.render_scale == 1.0
        assert sim.commands.pending_remove == 0


class TestRecovery:
    """P8.6: recovery restores scale first, then trails at full health."""

    def test_full_recovery_restores_scale_and_trails(
            self, default_config, monkeypatch):
        cfg = default_config
        cfg.trails = "velocity"
        viz, renderer, _ = _make_viz(monkeypatch, cfg)
        renderer.render_scale = 0.75
        viz._governor = _make_governor(recover=True, level=0)
        viz._apply_quality_actions()
        assert renderer.render_scale == 1.0
        renderer.enable_trails.assert_called_once_with(
            "velocity", cfg.trail_length)

    def test_full_recovery_keeps_trails_off_when_configured_off(
            self, default_config, monkeypatch):
        cfg = default_config
        cfg.trails = "off"
        viz, renderer, _ = _make_viz(monkeypatch, cfg)
        viz._governor = _make_governor(recover=True, level=0)
        viz._apply_quality_actions()
        renderer.enable_trails.assert_not_called()

    def test_partial_recovery_steps_scale_up(self, default_config, monkeypatch):
        viz, renderer, _ = _make_viz(monkeypatch, default_config)
        renderer.render_scale = 0.75
        viz._governor = _make_governor(recover=True, level=1)
        viz._apply_quality_actions()
        assert renderer.render_scale == pytest.approx(0.90)

    def test_scale_recovery_capped_at_one(self, default_config, monkeypatch):
        viz, renderer, _ = _make_viz(monkeypatch, default_config)
        renderer.render_scale = 0.95
        viz._governor = _make_governor(recover=True, level=1)
        viz._apply_quality_actions()
        assert renderer.render_scale == 1.0


# ── Shared helper for governor integration tests (extracted to avoid
# cross-class coupling between TestGovernorTimingIntegration and
# TestD5AdaptiveQualityGate). ──────────────────────────────────────


def _make_viz_real_gov(default_config, monkeypatch):
    """Visualizer with mocked GL + real QualityGovernor (not stubbed)."""
    mock_renderer = MagicMock()
    mock_renderer.render_scale = 1.0
    monkeypatch.setattr(
        "pymurmur.viz.visualizer.Renderer3D", lambda **kw: mock_renderer,
    )
    monkeypatch.setattr(
        "pymurmur.viz.visualizer.OrbitCamera", lambda **kw: MagicMock(),
    )
    monkeypatch.setattr(
        "pymurmur.viz.visualizer.SliderHUD", lambda cfg: MagicMock(),
    )
    from pymurmur.viz.visualizer import Visualizer

    sim = SimulationEngine(default_config)
    viz = Visualizer(sim, default_config, headless=True)
    return viz, mock_renderer, sim


# ── P8.6: Governor timing integration with monkeypatched clock ────

class TestGovernorTimingIntegration:
    """P8.6: Full visualizer → governor → _apply_quality_actions cycle
    with monkeypatched time.perf_counter for deterministic frame pacing.

    Simulates the render-loop pattern: t0 = perf_counter() at top,
    frame_ms = (perf_counter() − t0) at bottom. The governor is fed
    real QualityGovernor (not mocked), and _apply_quality_actions()
    triggers the actual degradation ladder.
    """

    def _simulate_frame(
        self, viz, clock, frame_ms: float,
    ) -> None:
        """Simulate one render-loop frame: t0 → apply → feed with time delta.

        Matches the Visualizer.run() pattern:
          1. t0 = time.perf_counter()
          2. _apply_quality_actions()  (uses previous frame's governor state)
          3. ... render would happen here ...
          4. frame_ms = (time.perf_counter() − t0) * 1000
          5. governor.feed(frame_ms)

        The clock is advanced by frame_ms/1000 to simulate wall-clock time.
        perf_counter() is monkeypatched to return clock[0] + frame_ms/1000.
        """
        # Step 1+2: apply quality actions (from previous feed)
        viz._apply_quality_actions()
        # Step 4+5: advance clock and feed the governor
        clock[0] += frame_ms / 1000.0
        viz._governor.feed(frame_ms)

    # ── degradation tests ──────────────────────────────────────

    def test_slow_frames_trigger_full_degrade_ladder(
        self, default_config, monkeypatch,
    ):
        """P8.6: Sustained slow frames fed to real governor →
        _apply_quality_actions fires levels 1, 2, 3 in order."""
        cfg = default_config
        cfg.num_boids = 600  # above COUNT_FLOOR for level 3 test
        cfg.perf.adaptive_quality = True
        viz, renderer, sim = _make_viz_real_gov(cfg, monkeypatch)

        clock = [0.0]
        monkeypatch.setattr(time, 'perf_counter', lambda: clock[0])

        slow_ms = 30.0  # 33fps, below 78% of 60fps threshold

        for _ in range(300):
            self._simulate_frame(viz, clock, slow_ms)
            if viz._governor.degradation_level >= 3:
                break

        # Verify integration outcomes through observable side effects
        assert viz._governor.degradation_level >= 1, (
            f"Should have degraded at least to level 1, "
            f"got level {viz._governor.degradation_level}"
        )
        # Level 1 effect: trails disabled
        renderer.disable_trails.assert_called()
        # Level 2 effect: render scale reduced below 1.0
        assert renderer.render_scale < 1.0, (
            f"Render scale {renderer.render_scale} should be < 1.0 after degrade"
        )
        # Level 3 effect: birds enqueued for removal
        if viz._governor.degradation_level >= 3:
            assert sim.commands.pending_remove > 0, (
                "Level 3 should enqueue bird removal"
            )

    def test_fast_frames_after_degrade_trigger_recovery(
        self, default_config, monkeypatch,
    ):
        """P8.6: After degrading, fast frames fed to real governor →
        recovery restores render scale and trails."""
        cfg = default_config
        cfg.num_boids = 600
        cfg.perf.adaptive_quality = True
        cfg.trails = "velocity"
        viz, renderer, sim = _make_viz_real_gov(cfg, monkeypatch)

        clock = [0.0]
        monkeypatch.setattr(time, 'perf_counter', lambda: clock[0])

        # Phase 1: degrade with slow frames
        slow_ms = 35.0
        for _ in range(250):
            self._simulate_frame(viz, clock, slow_ms)

        assert viz._governor.degradation_level >= 1, (
            f"Should have degraded, level={viz._governor.degradation_level}"
        )

        # Phase 2: recover with fast frames
        fast_ms = 5.0
        for _ in range(2500):
            self._simulate_frame(viz, clock, fast_ms)
            if viz._governor.degradation_level == 0:
                break

        assert viz._governor.degradation_level == 0, (
            f"Should have fully recovered, still at level "
            f"{viz._governor.degradation_level}"
        )
        assert renderer.render_scale == 1.0, (
            f"Render scale should be restored to 1.0, got {renderer.render_scale}"
        )

    def test_healthy_frames_never_trigger_degrade(
        self, default_config, monkeypatch,
    ):
        """P8.6: Frames at target FPS never trigger degradation."""
        cfg = default_config
        cfg.perf.adaptive_quality = True
        viz, renderer, sim = _make_viz_real_gov(cfg, monkeypatch)

        clock = [0.0]
        monkeypatch.setattr(time, 'perf_counter', lambda: clock[0])

        # Feed frames at budget (~16.67ms for 60fps target)
        budget_ms = 1000.0 / 60.0
        for _ in range(300):
            self._simulate_frame(viz, clock, budget_ms)
            assert not viz._governor.should_degrade(), (
                "Healthy frames should not trigger degradation"
            )

        assert viz._governor.degradation_level == 0
        renderer.disable_trails.assert_not_called()
        assert renderer.render_scale == 1.0

    def test_governor_ignored_when_adaptive_quality_disabled(
        self, default_config, monkeypatch,
    ):
        """P8.6: When adaptive_quality=False, the run() loop never feeds
        the governor and never calls _apply_quality_actions(). With the
        governor un-fed (level 0), calling _apply_quality_actions() is
        always a no-op regardless of config."""
        cfg = default_config
        cfg.perf.adaptive_quality = False  # disabled
        viz, renderer, sim = _make_viz_real_gov(cfg, monkeypatch)

        clock = [0.0]
        monkeypatch.setattr(time, 'perf_counter', lambda: clock[0])

        # Governor is never fed (as in production when adaptive_quality=False)
        # — it stays at level 0 with zero EMA.
        assert viz._governor.degradation_level == 0
        assert viz._governor.ema_ms == 0.0

        # Calling _apply_quality_actions() on a healthy, un-fed governor
        # must be a no-op.
        viz._apply_quality_actions()

        renderer.disable_trails.assert_not_called()
        renderer.enable_trails.assert_not_called()
        assert renderer.render_scale == 1.0, (
            "Render scale should stay at 1.0 when governor is healthy"
        )
        assert sim.commands.pending_remove == 0, (
            "No birds should be removed when governor is healthy"
        )

    def test_ema_tracks_frame_time_correctly(
        self, default_config, monkeypatch,
    ):
        """P8.6: The governor's EMA converges to the fed frame time
        when using monkeypatched perf_counter."""
        cfg = default_config
        cfg.perf.adaptive_quality = True
        viz, _, _ = _make_viz_real_gov(cfg, monkeypatch)

        clock = [0.0]
        monkeypatch.setattr(time, 'perf_counter', lambda: clock[0])

        # Feed constant 20ms frames
        target_ms = 20.0
        for _ in range(100):
            self._simulate_frame(viz, clock, target_ms)

        # EMA should be close to the target after 100 frames
        assert viz._governor.ema_ms == pytest.approx(target_ms, rel=0.10), (
            f"EMA {viz._governor.ema_ms:.1f}ms should converge to "
            f"{target_ms}ms after 100 frames"
        )


class TestD5AdaptiveQualityGate:
    """D5: Verify adaptive_quality gate controls governor feed + apply.

    NOTE: These tests simulate the gating pattern from Visualizer.run()
    rather than calling run() directly (which requires pygame).  They
    verify the guard conditions, not the actual run() code path.
    """

    def test_adaptive_quality_false_prevents_feed_and_apply(
        self, default_config, monkeypatch,
    ):
        """D5: When adaptive_quality=False, neither governor.feed()
        nor _apply_quality_actions() execute in the run() gating pattern."""
        cfg = default_config
        cfg.perf.adaptive_quality = False
        viz, _, _ = _make_viz_real_gov(cfg, monkeypatch)

        initial_level = viz._governor.degradation_level
        initial_ema = viz._governor.ema_ms

        # Simulate the run() gating pattern
        paused = False
        if not paused and cfg.perf.adaptive_quality:
            viz._apply_quality_actions()  # skipped (adaptive_quality=False)
        if not paused and cfg.perf.adaptive_quality:
            viz._governor.feed(50.0)      # skipped (adaptive_quality=False)

        assert viz._governor.degradation_level == initial_level
        assert viz._governor.ema_ms == initial_ema, (
            "EMA should not change when adaptive_quality=False"
        )

    def test_adaptive_quality_true_allows_feed_and_apply(
        self, default_config, monkeypatch,
    ):
        """D5: When adaptive_quality=True, governor.feed() is called
        and the first feed initialises EMA to the exact frame time."""
        cfg = default_config
        cfg.perf.adaptive_quality = True
        viz, _, _ = _make_viz_real_gov(cfg, monkeypatch)

        paused = False
        if not paused and cfg.perf.adaptive_quality:
            viz._apply_quality_actions()
        if not paused and cfg.perf.adaptive_quality:
            viz._governor.feed(16.67)  # budget frame

        # First feed initialises EMA to raw value (not alpha-blended)
        assert viz._governor.ema_ms == pytest.approx(16.67), (
            f"First feed should initialise EMA to exact frame time, "
            f"got {viz._governor.ema_ms}"
        )

    def test_adaptive_quality_gated_when_paused(
        self, default_config, monkeypatch,
    ):
        """D5: When paused=True, governor.feed() and _apply_quality_actions()
        are skipped even with adaptive_quality=True."""
        cfg = default_config
        cfg.perf.adaptive_quality = True
        viz, _, _ = _make_viz_real_gov(cfg, monkeypatch)

        initial_ema = viz._governor.ema_ms

        paused = True
        if not paused and cfg.perf.adaptive_quality:
            viz._apply_quality_actions()  # skipped (paused)
        if not paused and cfg.perf.adaptive_quality:
            viz._governor.feed(50.0)      # skipped (paused)

        assert viz._governor.ema_ms == initial_ema, (
            "EMA should not change when paused"
        )


class TestThreatMarker:
    """D7/S2.A8: Visualizer._draw_threat_marker draws a red/larger
    non-instanced marker at the predator's live position — an
    invisible predator is undebuggable."""

    def test_no_marker_when_predator_disabled(self, default_config, monkeypatch):
        default_config.predator_enabled = False
        viz, mock_renderer, _ = _make_viz(monkeypatch, default_config)
        viz._draw_threat_marker()
        mock_renderer.draw_layer.assert_not_called()

    def test_marker_drawn_when_predator_enabled(self, default_config, monkeypatch):
        default_config.predator_enabled = True
        viz, mock_renderer, _ = _make_viz(monkeypatch, default_config)
        viz._draw_threat_marker()
        mock_renderer.draw_layer.assert_called_once()
        (pos,), kwargs = mock_renderer.draw_layer.call_args
        assert len(pos) == 3
        assert kwargs.get("scale", 1.0) > 1.0, (
            "Marker must use scale > 1.0 to trigger the shader's "
            "predator_factor red-glow blend"
        )
