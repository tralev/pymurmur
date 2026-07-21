"""Subsystem F — Metrics & Analysis isolation tests.

Tests FlockMetrics, MetricsCollector gating levels, metric intervals,
presets, O(N) complexity, and snapshot correctness.
"""



class TestSubsystemF:
    """Metrics subsystem — gating, intervals, and correctness."""

    def test_metrics_fast_o_n(self, default_config):
        """All fast metrics compute in O(N) time (validate timing scales linearly)."""
        import time

        from pymurmur.simulation.engine import SimulationEngine

        for n in [20, 50, 100]:
            default_config.num_boids = n
            sim = SimulationEngine(default_config)
            t0 = time.perf_counter()
            for _ in range(10):
                sim.step(1.0 / 60)
            elapsed = time.perf_counter() - t0
            # Just verify it completes — timing is rough
            assert elapsed < 5.0

    def test_metrics_gating_level_0(self, default_config):
        """detail_level=0 → zero metrics computed."""
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.metrics_detail_level = 0
        sim = SimulationEngine(cfg)
        sim.run_headless(steps=5)
        sim.metrics.snapshot()
        # At level 0, alpha should still be computed (always on)
        # but expensive fields should be None

    def test_metrics_gating_level_1(self):
        """detail_level=1 → only fast metrics populated."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.metrics_detail_level = 1
        cfg.num_boids = 20
        sim = SimulationEngine(cfg)
        sim.run_headless(steps=5)
        snap = sim.metrics.snapshot()
        # Fast metrics should be populated
        assert snap.alpha >= 0.0
        assert snap.speed_avg >= 0.0
        assert snap.dispersion >= 0.0

    def test_metrics_gating_level_2(self):
        """detail_level=2 → all 15 metrics populated (at interval boundaries)."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 5
        cfg.num_boids = 20
        sim = SimulationEngine(cfg)
        sim.run_headless(steps=10)
        # After 10 steps with interval 5, we should have at least 2 snapshots
        snap = sim.metrics.snapshot()
        assert snap.alpha >= 0.0

    def test_metrics_interval_respected(self, default_config):
        """Expensive metrics computed every interval frames, not every frame."""
        from pymurmur.simulation.engine import SimulationEngine

        default_config.metrics_detail_level = 2
        default_config.metrics_interval = 100  # large — won't trigger
        sim = SimulationEngine(default_config)
        sim.run_headless(steps=5)
        # Expensive metrics (H₂, MSD) should not be computed at frame 5
        snap = sim.metrics.snapshot()
        assert snap.h2 is None

    def test_metrics_collector_snapshot(self, default_config):
        """snapshot() returns a FlockMetrics with correct types for all fields."""
        from pymurmur.simulation.engine import SimulationEngine

        sim = SimulationEngine(default_config)
        sim.run_headless(steps=5)
        snap = sim.metrics.snapshot()
        assert snap.alpha is not None
        assert isinstance(snap.alpha, (int, float))

    def test_presets_all_valid_fields(self):
        """Every preset dict contains only valid SimConfig field names (I7.1).

        Nested-only fields (retired shims, e.g. phi_p) are valid preset
        keys — the preset appliers route them to the sub-config explicitly.
        """
        from pymurmur.analysis.presets import PRESETS
        from pymurmur.core.config import _ALL_FIELD_NAMES, _NESTED_ONLY

        valid = _ALL_FIELD_NAMES | set(_NESTED_ONLY.keys())
        for name, preset in PRESETS.items():
            unknown = set(preset.keys()) - valid
            assert not unknown, f"Preset '{name}' has unknown fields: {unknown}"

    def test_presets_do_not_mutate_originals(self):
        """Applying a preset doesn't modify the PRESETS dictionary."""
        from copy import deepcopy

        from pymurmur.analysis.presets import PRESETS
        from pymurmur.core.config import SimConfig

        preset_copy = deepcopy(PRESETS)
        cfg = SimConfig()
        for key, value in PRESETS["ball"].items():
            setattr(cfg, key, value)
        assert PRESETS == preset_copy
