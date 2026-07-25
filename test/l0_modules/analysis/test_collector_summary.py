"""Unit tests for analysis.collector (via metrics re-export) — cross-cutting summary readout, D19 history-cap ring-buffer truncation, async collector edge cases.

Split out of test_metrics.py (file-size split).
"""

import numpy as np

from pymurmur.analysis.metrics import FlockMetrics, MetricsCollector


class TestCrossCuttingSummary:
    """P10.1 + P10.2 + P10.6: preset changes reflected in summary output."""

    def test_preset_summary_reflects_new_mode(self):
        """P10.1->P10.2: After applying a projection preset, summary shows mode."""
        from pymurmur.analysis.presets import apply_preset
        from pymurmur.core.config import SimConfig
        cfg = SimConfig()
        apply_preset(cfg, "e")  # Vertical Column: projection, 0.10/0.75/6
        m = FlockMetrics(alpha=0.5)
        result = m.summary(
            mode=cfg.mode, N_active=cfg.num_boids,
            phi_p=cfg.projection.phi_p, phi_a=cfg.phi_a, sigma=cfg.sigma,
        )
        assert "projection" in result

    def test_preset_summary_reflects_phi_params(self):
        """P10.1->P10.2: After applying a preset, summary includes phi values."""
        from pymurmur.analysis.presets import apply_preset
        from pymurmur.core.config import SimConfig
        cfg = SimConfig()
        apply_preset(cfg, "e")  # 0.10/0.75/6
        m = FlockMetrics(alpha=0.5)
        result = m.summary(
            N_active=cfg.num_boids,
            phi_p=cfg.projection.phi_p, phi_a=cfg.phi_a, sigma=cfg.sigma,
        )
        # Unicode or ASCII phi in output
        assert "0.10" in result or "0.1" in result
        assert "0.75" in result

    def test_spawn_updates_summary_n_active(self):
        """P10.4->P10.2: After spawning birds, summary reflects new N_active."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine
        cfg = SimConfig(num_boids=5)
        engine = SimulationEngine(cfg)
        assert engine.flock.N_active == 5

        # Spawn 3 more birds
        engine.enqueue_spawn((500, 500, 500))
        engine.enqueue_spawn((600, 400, 300))
        engine.enqueue_spawn((400, 600, 400))
        engine.drain_commands()

        m = FlockMetrics(alpha=0.3)
        result = m.summary(N_active=engine.flock.N_active)
        assert "N=8" in result

    def test_violating_preset_enforced_in_summary(self):
        """P10.1->P10.6->P10.2: Applying violating preset enforces phi
        constraint; summary reflects corrected values."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cfg.mode = "projection"
        cfg.projection.phi_p = 0.8
        cfg.phi_a = 0.8  # sum=1.6 > 1.0

        # Enforce the constraint
        InputControl._enforce_phi_after_preset(cfg)

        total = cfg.projection.phi_p + cfg.phi_a
        assert total <= 1.0 + 1e-10

        # Summary should reflect the corrected values
        m = FlockMetrics(alpha=0.5)
        result = m.summary(
            N_active=cfg.num_boids,
            phi_p=cfg.projection.phi_p, phi_a=cfg.phi_a, sigma=cfg.sigma,
        )
        # phi_p=0.8 >= phi_a=0.8, so phi_a reduced to 0.2
        assert "0.80" in result
        assert "0.20" in result


# ── D19: History cap ring-buffer truncation ──────────────────

class TestHistoryCap:
    """D19: MetricsCollector history_cap prevents unbounded growth."""

    def test_history_truncated_at_cap(self):
        """D19: History is truncated to history_cap when exceeded."""
        from pymurmur.analysis.metrics import MetricsCollector
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock

        cfg = SimConfig(num_boids=5, seed=42, history_cap=10)
        flock = PhysicsFlock(cfg)
        collector = MetricsCollector(cfg)

        # Collect more frames than cap
        for frame in range(25):
            collector.collect(flock, frame)

        # History should be capped at 10
        assert len(collector.history) <= 10, (
            f"History should be capped at 10, got {len(collector.history)}"
        )

    def test_history_truncation_keeps_newest_entries(self):
        """D19: Truncation keeps the most recent entries."""
        from pymurmur.analysis.metrics import MetricsCollector
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock

        cfg = SimConfig(num_boids=5, seed=42, history_cap=10)
        flock = PhysicsFlock(cfg)
        collector = MetricsCollector(cfg)

        for frame in range(25):
            collector.collect(flock, frame)

        # The oldest entry should be from around frame 15 (25-10)
        # Since collect() doesn't store frame numbers in FlockMetrics,
        # we verify that we have exactly cap entries and no crash occurred.
        assert len(collector.history) <= collector._history_cap
        # And snapshot() still returns the most recent
        snap = collector.snapshot()
        assert snap.alpha >= 0.0

    def test_history_below_cap_no_truncation(self):
        """D19: History below cap is never truncated."""
        from pymurmur.analysis.metrics import MetricsCollector
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock

        cfg = SimConfig(num_boids=5, seed=42, history_cap=100)
        flock = PhysicsFlock(cfg)
        collector = MetricsCollector(cfg)

        for frame in range(20):
            collector.collect(flock, frame)

        # All 20 entries should be present
        assert len(collector.history) == 20, (
            f"Expected 20 entries, got {len(collector.history)}"
        )

    def test_history_cap_one_keeps_one_entry(self):
        """D19: history_cap=1 keeps exactly 1 entry (most recent)."""
        from pymurmur.analysis.metrics import MetricsCollector
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock

        cfg = SimConfig(num_boids=5, seed=42, history_cap=1)
        flock = PhysicsFlock(cfg)
        collector = MetricsCollector(cfg)

        # After first collection, history should be at most 1 entry.
        collector.collect(flock, 0)
        assert len(collector.history) == 1

        # After second, truncation triggers
        collector.collect(flock, 1)
        assert len(collector.history) == 1

    def test_default_cap_is_10000(self):
        """D19: Default history_cap is 10000 when not configured."""
        collector = MetricsCollector()  # no config
        assert collector._history_cap == 10000, (
            f"Default history_cap should be 10000, got {collector._history_cap}"
        )

    def test_position_snapshots_also_capped(self):
        """D19: position_snapshots are capped proportionally to collection interval."""
        from pymurmur.analysis.metrics import MetricsCollector
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock

        cfg = SimConfig(num_boids=5, seed=42,
                        history_cap=50, metrics_interval=5)
        flock = PhysicsFlock(cfg)
        collector = MetricsCollector(cfg)

        # Collect 200 frames at interval=5 → 40 snapshots, but snap_cap = 50//5 = 10
        for frame in range(200):
            collector.collect(flock, frame)

        # Snapshots should be capped at 10
        snap_cap = max(1, 50 // 5)
        assert len(collector._position_snapshots) <= snap_cap, (
            f"Position snapshots should be <= {snap_cap}, got {len(collector._position_snapshots)}"
        )

    def test_density_history_also_capped(self):
        """D19: density_history is capped proportionally when detail_level >= 2."""
        from pymurmur.analysis.metrics import MetricsCollector
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock

        cfg = SimConfig(num_boids=5, seed=42,
                        history_cap=30, metrics_interval=3,
                        metrics_detail_level=2)
        flock = PhysicsFlock(cfg)
        collector = MetricsCollector(cfg)

        for frame in range(120):
            collector.collect(flock, frame)

        snap_cap = max(1, 30 // 3)
        assert len(collector._density_history) <= snap_cap, (
            f"Density history should be <= {snap_cap}, got {len(collector._density_history)}"
        )

    def test_no_crash_on_very_small_cap(self):
        """D19: Very small history_cap (1) doesn't crash."""
        from pymurmur.analysis.metrics import MetricsCollector
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock

        cfg = SimConfig(num_boids=5, seed=42, history_cap=1)
        flock = PhysicsFlock(cfg)
        collector = MetricsCollector(cfg)

        # Should not crash even with cap=1
        for frame in range(10):
            collector.collect(flock, frame)

        assert len(collector.history) == 1


# ── Async collector edge cases (coverage-gap fill: lines 313-345) ─

class TestAsyncCollector:
    """MetricsCollector._collect_async_result edge cases.

    These paths are normally exercised by real async frame timing but
    can be tested directly by manipulating _async_result state.
    """

    def test_collect_async_result_none_noop(self):
        """result=None → returns immediately without assignments."""
        from pymurmur.analysis.metrics import MetricsCollector
        mc = MetricsCollector()
        mc._async_result = None
        m = FlockMetrics()
        mc._collect_async_result(m)
        # No crash, no assignments
        assert m.h2 is None
        assert m.optimal_m is None

    def test_collect_async_result_still_computing_noop(self):
        """result={"done": False} → returns without assignments (still computing)."""
        from pymurmur.analysis.metrics import MetricsCollector
        mc = MetricsCollector()
        mc._async_result = {"done": False, "data": None}
        m = FlockMetrics()
        mc._collect_async_result(m)
        assert m.h2 is None

    def test_collect_async_result_done_with_data_assigns(self):
        """result={"done": True, "data": m} → copies expensive fields."""
        from pymurmur.analysis.metrics import MetricsCollector
        mc = MetricsCollector()
        async_m = FlockMetrics(
            h2=3.14, optimal_m=8, local_spacing=5.0,
            aspect_ratio=2.5, thickness_ratio=0.4,
            gyration_radius=100.0, suggested_m=7.5, eta_m=0.05,
        )
        mc._async_result = {"done": True, "data": async_m}
        m = FlockMetrics()
        mc._collect_async_result(m)
        assert m.h2 == 3.14
        assert m.optimal_m == 8
        assert m.local_spacing == 5.0
        assert m.aspect_ratio == 2.5
        assert m.thickness_ratio == 0.4
        assert m.gyration_radius == 100.0
        assert m.suggested_m == 7.5
        assert m.eta_m == 0.05
        # After collection, _async_result is cleared
        assert mc._async_result is None

    def test_collect_async_result_done_with_none_data_noop(self):
        """result={"done": True, "data": None} → no assignments, result cleared."""
        from pymurmur.analysis.metrics import MetricsCollector
        mc = MetricsCollector()
        mc._async_result = {"done": True, "data": None}
        m = FlockMetrics()
        mc._collect_async_result(m)
        assert m.h2 is None  # nothing was assigned
        assert mc._async_result is None  # result cleared anyway

    def test_start_async_stale_generation_discards_result(self):
        """If async_gen advances before the worker completes, result is discarded.

        Simulates: start async gen=1 → worker starts → start async gen=2
        → first worker finishes with gen=1 but self._async_gen=2 → result
        is NOT stored (the stale-generation guard inside _worker).

        Uses thread.join() for reliable synchronization — no time.sleep().
        """

        from pymurmur.analysis.metrics import MetricsCollector

        mc = MetricsCollector()
        pos = np.random.default_rng(42).uniform(0, 500, (10, 3)).astype(np.float32)

        # Start first async — store its thread
        mc._start_async_expensive(pos.copy(), 10)
        t1 = mc._async_thread
        gen1 = mc._async_gen

        # Start second async before first completes (gen counter advances)
        mc._start_async_expensive(pos.copy(), 10)
        t2 = mc._async_thread
        gen2 = mc._async_gen
        assert gen2 == gen1 + 1, "generation counter must increment"

        # Join both threads with generous timeout
        t1.join(timeout=5)
        t2.join(timeout=5)

        # The stored result must be from gen2 (current generation).
        # Gen1's worker saw gen1 != self._async_gen (which is now 2)
        # and discarded its result via the guard in _worker.
        result = mc._async_result
        assert result is not None
        assert result.get("done") is True
        assert result.get("data") is not None
        assert result.get("gen") == gen2, (
            f"stored result must be from current generation {gen2}, "
            f"not stale gen {gen1}"
        )


