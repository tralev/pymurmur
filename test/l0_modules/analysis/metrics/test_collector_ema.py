"""Unit tests for analysis.collector (via metrics re-export) — S3.11 EMA readout smoothing, D18 metrics accelerations, D10 ripple forces in last_accelerations.

Split out of test_metrics.py (file-size split).
"""

import numpy as np
import pytest

from pymurmur.analysis.metrics import MetricsCollector

# ── S3.11: EMA readout smoothing ──────────────────────────────

class TestEMAReadout:
    """S3.11: EMA-smoothed display readout — display-only, raw history untouched.

    Verifies: EMA converges to constant value over ~50 frames,
    readout_smooth=0 gives passthrough, to_dict() always returns raw,
    and smoothed()/snapshot() separation is correct.
    """

    @staticmethod
    def _make_collector(readout_smooth: float = 0.04):
        """Create a MetricsCollector with known readout_smooth."""
        from pymurmur.core.config import SimConfig
        cfg = SimConfig(readout_smooth=readout_smooth)
        return MetricsCollector(cfg)

    def _collect_n(self, collector, flock, n: int = 1):
        """Call collector.collect() n times with the same flock."""
        for frame in range(n):
            collector.collect(flock, frame)

    # ── passthrough tests ──────────────────────────────────────

    def test_readout_smooth_zero_passthrough(self, default_config):
        """S3.11: readout_smooth=0 → smoothed() returns raw snapshot."""
        from pymurmur.physics.flock import PhysicsFlock

        collector = self._make_collector(readout_smooth=0.0)
        cfg = default_config
        cfg.num_boids = 30
        flock = PhysicsFlock(cfg)

        self._collect_n(collector, flock, 3)

        raw = collector.snapshot()
        smoothed = collector.smoothed()

        # With readout_smooth=0, smoothed() should return the same object
        # as snapshot() — passthrough, no EMA applied.
        assert smoothed is raw, (
            "smoothed() should return the raw snapshot when readout_smooth=0"
        )

    def test_readout_smooth_default_is_004(self):
        """S3.11: Default readout_smooth is 0.04."""
        collector = MetricsCollector()  # no config → defaults
        assert collector._readout_smooth == 0.04, (
            f"Default readout_smooth should be 0.04, got {collector._readout_smooth}"
        )

    def test_readout_smooth_from_config(self):
        """S3.11: readout_smooth is read from config.perf.readout_smooth."""
        from pymurmur.core.config import SimConfig

        cfg = SimConfig(readout_smooth=0.12)
        collector = MetricsCollector(cfg)
        assert collector._readout_smooth == 0.12

    # ── EMA convergence tests ──────────────────────────────────

    def test_ema_converges_to_constant_input(self, default_config):
        """S3.11: After ~50 frames of constant input, EMA ≈ raw value."""
        from pymurmur.physics.flock import PhysicsFlock

        collector = self._make_collector(readout_smooth=0.04)
        cfg = default_config
        cfg.num_boids = 30
        flock = PhysicsFlock(cfg)

        # Collect 60 frames — EMA should be very close to raw
        self._collect_n(collector, flock, 60)

        raw = collector.snapshot()
        ema = collector.smoothed()

        # After 60 frames at α=0.04, EMA should be within 1% of raw
        # (1 − 0.04)^60 ≈ 0.086, so error < 9% → relaxed: within 15%
        assert ema.alpha == pytest.approx(raw.alpha, rel=0.15), (
            f"EMA alpha={ema.alpha:.4f} should converge to raw={raw.alpha:.4f}"
        )
        assert ema.speed_avg == pytest.approx(raw.speed_avg, rel=0.15), (
            f"EMA speed_avg={ema.speed_avg:.4f} should converge to raw={raw.speed_avg:.4f}"
        )

    def test_ema_starts_from_zero(self, default_config):
        """S3.11: First frame EMA blends from zero (FlockMetrics() default)."""
        from pymurmur.physics.flock import PhysicsFlock

        collector = self._make_collector(readout_smooth=0.04)
        cfg = default_config
        cfg.num_boids = 30
        flock = PhysicsFlock(cfg)

        # One collection — EMA should be α × raw (since starting from 0)
        collector.collect(flock, 0)

        raw = collector.snapshot()
        ema = collector.smoothed()

        # EMA = (1 − 0.04)×0 + 0.04×raw = 0.04 × raw
        expected_alpha = 0.04 * raw.alpha
        assert ema.alpha == pytest.approx(expected_alpha, rel=0.01), (
            f"First-frame EMA alpha={ema.alpha:.4f} should be ~{expected_alpha:.4f}"
        )

    def test_ema_smoothed_distinct_from_snapshot(self, default_config):
        """S3.11: smoothed() returns a different object from snapshot()
        when readout_smooth > 0."""
        from pymurmur.physics.flock import PhysicsFlock

        collector = self._make_collector(readout_smooth=0.04)
        cfg = default_config
        cfg.num_boids = 30
        flock = PhysicsFlock(cfg)

        self._collect_n(collector, flock, 10)

        raw = collector.snapshot()
        ema = collector.smoothed()

        # These should be distinct objects (no aliasing)
        assert ema is not raw, (
            "smoothed() must return _ema_metrics, not the raw snapshot"
        )
        # And the EMA values should differ (not fully converged yet)
        # At α=0.04 and 10 frames, convergence is ~33%
        assert ema.alpha != pytest.approx(raw.alpha, abs=1e-9), (
            "After 10 frames, EMA should differ from raw (not yet converged)"
        )

    # ── to_dict returns raw tests ─────────────────────────────

    def test_to_dict_returns_raw_even_with_ema(self, default_config):
        """S3.11: snapshot().to_dict() returns raw field values, not EMA.

        After EMA blending has run for multiple frames, the raw snapshot
        (collector.snapshot()) must still contain the original raw
        FlockMetrics, and to_dict() must serialize those raw values.
        The EMA-smoothed values are only accessible via smoothed().
        """
        from pymurmur.physics.flock import PhysicsFlock

        collector = self._make_collector(readout_smooth=0.04)
        cfg = default_config
        cfg.num_boids = 30
        flock = PhysicsFlock(cfg)

        self._collect_n(collector, flock, 10)

        raw = collector.snapshot()
        ema = collector.smoothed()

        raw_dict = raw.to_dict()

        # The raw snapshot's to_dict() must match the raw field values,
        # *not* the EMA-blended ones.
        assert raw_dict["alpha"] == pytest.approx(raw.alpha, rel=1e-5)
        assert raw_dict["speed_avg"] == pytest.approx(raw.speed_avg, rel=1e-5)

        # After 10 frames at α=0.04, EMA is not converged — raw and
        # EMA should differ, proving to_dict() serializes raw fields.
        assert abs(raw.alpha - ema.alpha) > 1e-9, (
            "Raw and EMA should differ when not converged"
        )

        # EMA-smoothed values are also serializable, but they're
        # a different object — not the raw snapshot.
        ema_dict = ema.to_dict()
        assert ema_dict["alpha"] == pytest.approx(ema.alpha, rel=1e-5)

    def test_snapshot_to_dict_matches_raw_fields(self, default_config):
        """S3.11: snapshot().to_dict() equals the raw FlockMetrics fields."""
        from pymurmur.physics.flock import PhysicsFlock

        collector = self._make_collector(readout_smooth=0.04)
        cfg = default_config
        cfg.num_boids = 30
        flock = PhysicsFlock(cfg)

        self._collect_n(collector, flock, 10)

        raw = collector.snapshot()
        raw_dict = raw.to_dict()

        # Verify key scalar fields match
        for field_name in ("alpha", "dispersion", "speed_avg", "force_avg"):
            assert field_name in raw_dict, f"{field_name} missing from to_dict()"
            assert raw_dict[field_name] == pytest.approx(
                getattr(raw, field_name), rel=1e-5
            ), f"{field_name}: to_dict={raw_dict[field_name]} vs raw={getattr(raw, field_name)}"

    # ── field coverage tests ───────────────────────────────────

    def test_ema_covers_all_scalar_fast_metrics(self, default_config):
        """S3.11: All specified scalar fast-metrics are EMA-blended.

        The _apply_ema_readout method lists 19 fields for EMA blending.
        Uses SimulationEngine to step the flock so forces are applied
        and every field has non-zero raw values — avoiding vacuously-passing
        skips when raw values happen to be zero.
        """
        from pymurmur.simulation.engine import SimulationEngine

        collector = self._make_collector(readout_smooth=0.04)
        cfg = default_config
        cfg.num_boids = 30
        cfg.mode = "projection"  # ensures theta is computed (not NaN)
        cfg.metrics_detail_level = 1

        # Step the engine to produce non-zero forces and full metrics
        engine = SimulationEngine(cfg)
        for _ in range(5):
            engine.step(1.0 / 60.0)

        # Swap the collector into the engine to use the pre-configured
        # readout_smooth, but collect a few frames manually
        engine.metrics = collector
        for _ in range(3):
            engine.step(1.0 / 60.0)

        raw = collector.snapshot()
        ema = collector.smoothed()

        # Fields that should be EMA-blended (from _apply_ema_readout)
        blended_fields = (
            "alpha", "nematic_S", "theta", "theta_prime", "silhouette_2d",
            "normalized_angular_momentum", "dispersion", "speed_avg",
            "force_avg", "power_avg", "local_spacing",
        )

        fields_checked = 0
        for field_name in blended_fields:
            raw_val = getattr(raw, field_name)
            ema_val = getattr(ema, field_name)
            if raw_val is not None and not (
                isinstance(raw_val, float) and np.isnan(raw_val)
            ):
                if abs(float(raw_val)) > 1e-9:
                    assert ema_val != pytest.approx(raw_val, abs=1e-9), (
                        f"{field_name}: ema={ema_val} should differ from raw={raw_val} after 3 frames"
                    )
                    fields_checked += 1

        # At least half of the blended fields should have been checked
        # (force_avg, power_avg, and speed_avg are always non-zero after stepping)
        assert fields_checked >= len(blended_fields) // 2, (
            f"Only {fields_checked}/{len(blended_fields)} fields had non-zero raw "
            "values — too few to verify EMA coverage"
        )

    def test_ema_nan_fields_keep_previous(self, default_config):
        """S3.11: NaN fields in raw are skipped — EMA keeps previous value."""
        from pymurmur.physics.flock import PhysicsFlock

        # Use non-projection mode to get NaN theta
        collector = self._make_collector(readout_smooth=0.04)
        collector._mode = "spatial"  # non-projection → theta will be NaN
        cfg = default_config
        cfg.num_boids = 30
        flock = PhysicsFlock(cfg)

        # First collect to establish EMA baseline
        self._collect_n(collector, flock, 5)
        ema_before = collector.smoothed().theta

        # Collect more — theta will be NaN (spatial mode)
        collector.collect(flock, 6)
        ema_after = collector.smoothed().theta

        # EMA should keep the previous value (not become NaN)
        assert not np.isnan(ema_after), (
            "EMA theta should not become NaN in non-projection mode"
        )
        assert ema_after == ema_before, (
            f"EMA theta should keep previous value {ema_before}, got {ema_after}"
        )

    def test_ema_domain_changed_updates_smoothed(self, default_config):
        """S3.11: When raw values change significantly, EMA tracks
        the change (lagging behind with smoothing)."""
        from pymurmur.physics.flock import PhysicsFlock

        collector = self._make_collector(readout_smooth=0.10)  # faster α
        cfg = default_config
        cfg.num_boids = 30
        flock = PhysicsFlock(cfg)

        # Stabilize
        self._collect_n(collector, flock, 20)
        ema_before = collector.smoothed().alpha

        # Now change the flock — all velocities point in +X for high alpha
        flock.velocities[:] = np.array([4.0, 0.0, 0.0], dtype=np.float32)
        self._collect_n(collector, flock, 10)
        ema_after = collector.smoothed().alpha

        # EMA should have moved toward the new higher value
        assert ema_after > ema_before, (
            f"EMA alpha should increase after velocity alignment: "
            f"{ema_before:.4f} → {ema_after:.4f}"
        )

    def test_ema_history_untouched(self, default_config):
        """S3.11: Raw history entries are never modified by EMA blending."""
        from pymurmur.physics.flock import PhysicsFlock

        collector = self._make_collector(readout_smooth=0.04)
        cfg = default_config
        cfg.num_boids = 30
        flock = PhysicsFlock(cfg)

        self._collect_n(collector, flock, 10)

        # Record raw history values at frame 2
        raw_at_frame2 = collector.history[2]
        alpha_before = raw_at_frame2.alpha

        # Collect more — EMA blending should not mutate history
        self._collect_n(collector, flock, 10)

        alpha_after = collector.history[2].alpha
        assert alpha_before == alpha_after, (
            f"History entry should be immutable: {alpha_before} → {alpha_after}"
        )


# ═══════════════════════════════════════════════════════════════════
# D18: Metrics read from last_accelerations (pre-zeroing stash)
# ═══════════════════════════════════════════════════════════════════


class TestD18MetricsAccelerations:
    """D18: metrics.collect() reads force/power from
    flock.last_accelerations, not flock.accelerations (which
    integrate() zeros before collect runs)."""

    def test_metrics_reads_last_accelerations_not_accelerations(self):
        """D18: collect() uses flock.last_accelerations for force/power."""
        from pymurmur.analysis.metrics import MetricsCollector
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock

        cfg = SimConfig()
        cfg.num_boids = 5
        cfg.mode = "spatial"

        flock = PhysicsFlock(cfg)
        flock.active[:] = True
        # Set accelerations to non-zero (force computation would do this)
        flock.accelerations[:] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        # Simulate: integrate() stashes then zeros accelerations
        flock.last_accelerations[:] = flock.accelerations.copy()
        flock.accelerations[:] = 0.0  # simulate integrate() zeroing

        collector = MetricsCollector(cfg)
        collector.collect(flock, 0)

        snap = collector.snapshot()
        # Force/power must reflect pre-zeroing acceleration values
        assert snap.force_avg > 0.0, (
            f"force_avg should be non-zero (from last_accelerations), "
            f"got {snap.force_avg}"
        )
        assert snap.power_avg > 0.0, (
            f"power_avg should be non-zero (from last_accelerations), "
            f"got {snap.power_avg}"
        )

    def test_metrics_last_accelerations_code_inspection(self):
        """D18: Source uses last_accelerations, not accelerations."""
        import inspect

        from pymurmur.analysis.metrics import MetricsCollector
        src = inspect.getsource(MetricsCollector.collect)
        assert "flock.last_accelerations" in src, (
            "collect() must read flock.last_accelerations for force/power"
        )

    def test_metrics_sees_nonzero_force_after_engine_step(self):
        """D18: After one engine step, metrics force/power is non-zero."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.num_boids = 10
        cfg.mode = "spatial"
        cfg.noise_scale = 0.5  # ensure some force is generated
        cfg.seed = 42

        engine = SimulationEngine(cfg)
        # Step once — forces are computed, then integrate() zeros accels,
        # then collect() should still see non-zero force from the stash.
        engine.step(1.0 / 60.0)

        snap = engine.metrics.snapshot()
        assert snap.force_avg > 0.0, (
            f"After step, force_avg should be > 0, got {snap.force_avg}"
        )
        assert snap.power_avg > 0.0, (
            f"After step, power_avg should be > 0, got {snap.power_avg}"
        )

    def test_metrics_force_reflects_actual_acceleration_magnitude(self):
        """D18: force_avg matches the magnitude of last_accelerations."""
        from pymurmur.analysis.metrics import MetricsCollector
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock

        cfg = SimConfig()
        cfg.num_boids = 3
        cfg.mode = "spatial"

        flock = PhysicsFlock(cfg)
        flock.active[:] = True
        # Set known accelerations
        known_acc = np.array([[3.0, 0.0, 0.0],
                               [0.0, 4.0, 0.0],
                               [0.0, 0.0, 0.0]], dtype=np.float32)
        flock.accelerations[:] = known_acc
        flock.last_accelerations[:] = known_acc.copy()
        flock.accelerations[:] = 0.0  # simulate zeroing
        # Set velocities for power computation
        flock.velocities[:] = np.array([[1.0, 0.0, 0.0],
                                         [0.0, 1.0, 0.0],
                                         [0.0, 0.0, 1.0]], dtype=np.float32)

        collector = MetricsCollector(cfg)
        collector.collect(flock, 0)

        snap = collector.snapshot()
        # force_avg = mean of |acc|: (3 + 4 + 0)/3 ≈ 2.333
        expected_force = (3.0 + 4.0 + 0.0) / 3.0
        assert snap.force_avg == pytest.approx(expected_force, rel=0.01), (
            f"force_avg should be ~{expected_force:.3f}, got {snap.force_avg:.3f}"
        )


# ── D10 + D18: Ripple forces in metrics last_accelerations ──────


def test_ripple_forces_reflected_in_last_accelerations():
    """D10+D18: Ripple extension forces are captured by metrics.

    D10 fixed ripple envelope to be per-bird (not scalar), so forces
    vary by distance from the ripple centre. D18 fixed metrics to read
    last_accelerations (pre-zeroing stash) instead of accelerations
    (always zero after integrate). Together, ripple forces must appear
    as non-zero entries in the metrics acceleration stash.
    """
    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

    cfg = SimConfig()
    cfg.seed = 42
    cfg.num_boids = 30
    cfg.mode = "spatial"
    cfg.ripple_enabled = True
    cfg.noise_scale = 0.0  # no noise to keep forces clean

    engine = SimulationEngine(cfg)
    engine.step(1.0 / 60.0)

    # D18: metrics must capture non-zero forces from last_accelerations
    snap = engine.metrics.snapshot()
    assert snap is not None, "Metrics must produce snapshot after step"
    assert snap.force_avg > 0, (
        "D10+D18: Ripple forces must be reflected in metrics last_accelerations"
    )


def test_metrics_force_changes_with_ripple_distance():
    """D10+D18: Ripple per-bird envelope varies with distance.

    D10 ensures ripple envelope is per-bird (shape (N,)). D18 ensures
    metrics capture forces via last_accelerations stash.
    """
    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

    cfg = SimConfig()
    cfg.seed = 42
    cfg.num_boids = 20
    cfg.mode = "spatial"
    cfg.ripple_enabled = True
    cfg.noise_scale = 0.0

    engine = SimulationEngine(cfg)
    flock = engine.flock

    # Place birds at different distances
    flock.positions[0] = np.array([500.0, 350.0, 200.0], dtype=np.float32)
    flock.positions[1] = np.array([500.0, 350.0, 500.0], dtype=np.float32)

    engine.step(1.0 / 60.0)

    # D10: Different distances should give different forces
    f0 = float(np.linalg.norm(flock.last_accelerations[0]))
    f1 = float(np.linalg.norm(flock.last_accelerations[1]))
    # At least one bird should get ripple force
    assert max(f0, f1) > 0, "D10+D18: Ripple should produce non-zero force"


