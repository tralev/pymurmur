"""Unit tests for analysis.opacity (via metrics re-export) — S3.6a marginal-opacity self-regulation validation, B10 public opacity dataset constants, S2.B4 physical-metrics edge cases, G7 fastmath warning.

Split out of test_metrics.py (file-size split).
"""

import numpy as np
import pytest

from pymurmur.analysis.metrics import FlockMetrics, MetricsCollector


@pytest.mark.slow
def test_s36a_projection_flock_self_regulates_to_marginal_opacity():
    """S3.6a: a settled, seeded projection-mode flock's time-averaged 2D
    silhouette (Θ', S3.6) lands in the marginal-opacity band — the
    occlusion-avoidance dynamics keep the flock neither so sparse it
    reads as empty sky nor so dense it reads as a solid disk.

    No new physics — this is a regression test over the existing
    compute_silhouette_2d() metric (S3.6). MARGINAL_OPACITY_MEAN/STD
    (metrics.py) are documented reference constants; the acceptance band
    [0.05, 0.55] is the roadmap's stated S3.6a criterion for N≈150,
    300 frames.
    """
    from pymurmur.analysis.metrics import (
        MARGINAL_OPACITY_MEAN,
        MARGINAL_OPACITY_STD,
    )
    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

    cfg = SimConfig()
    cfg.mode = "projection"
    cfg.num_boids = 150
    cfg.seed = 42
    cfg.metrics_detail_level = 1
    cfg.metrics_interval = 1

    engine = SimulationEngine(cfg)
    settle_frames = 300
    measure_from = 200  # average the last 100 frames, after the flock settles

    silhouettes = []
    for frame in range(settle_frames):
        engine.step(1.0 / 60.0)
        if frame >= measure_from:
            silhouettes.append(engine.metrics.snapshot().silhouette_2d)

    mean_silhouette = float(np.mean(silhouettes))
    assert 0.05 <= mean_silhouette <= 0.55, (
        f"Time-averaged silhouette Θ'={mean_silhouette:.4f} outside the "
        f"marginal-opacity band [0.05, 0.55] (reference: "
        f"mean={MARGINAL_OPACITY_MEAN}, std={MARGINAL_OPACITY_STD})"
    )


# ── S3.6a: Marginal opacity with different seeds ───────────────────

def test_s36a_different_seed_still_in_band():
    """S3.6a: Different seed (seed=123) also settles within [0.05, 0.55]."""
    from pymurmur.analysis.metrics import MARGINAL_OPACITY_MEAN, MARGINAL_OPACITY_STD
    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

    cfg = SimConfig()
    cfg.mode = "projection"
    cfg.num_boids = 150
    cfg.seed = 123
    cfg.metrics_detail_level = 1
    cfg.metrics_interval = 1

    engine = SimulationEngine(cfg)
    settle_frames = 300
    measure_from = 200
    silhouettes = []

    for frame in range(settle_frames):
        engine.step(1.0 / 60.0)
        if frame >= measure_from:
            silhouettes.append(engine.metrics.snapshot().silhouette_2d)

    avg_silhouette = sum(silhouettes) / len(silhouettes)
    assert 0.05 <= avg_silhouette <= 0.55, (
        f"seed=123 silhouette {avg_silhouette:.4f} should be in "
        f"marginal-opacity band [0.05, 0.55] (reference: "
        f"mean={MARGINAL_OPACITY_MEAN}, std={MARGINAL_OPACITY_STD})"
    )


@pytest.mark.slow
def test_b15_marginal_opacity_emerges_without_steric():
    """B15/B2 (Pearce et al. 2014): marginal opacity self-regulation
    emerges from projection (phi_p/phi_a) alone -- steric prevents
    unphysical overlap but does not drive the density regulation. Same
    settle/measure protocol as test_s36a_..._self_regulates_to_
    marginal_opacity, but with steric fully disabled (phantom-particle
    mode, matching the paper's base model). Measured directly before
    writing this assertion: mean silhouette 0.176, well inside the
    S3.6a band -- confirming steric isn't load-bearing for the
    self-regulation dynamics."""
    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

    cfg = SimConfig()
    cfg.mode = "projection"
    cfg.num_boids = 150
    cfg.seed = 42
    cfg.metrics_detail_level = 1
    cfg.metrics_interval = 1
    cfg.steric = 0.0  # phantom-particle mode -- no volume exclusion

    engine = SimulationEngine(cfg)
    settle_frames = 300
    measure_from = 200

    silhouettes = []
    for frame in range(settle_frames):
        engine.step(1.0 / 60.0)
        if frame >= measure_from:
            silhouettes.append(engine.metrics.snapshot().silhouette_2d)

    mean_silhouette = float(np.mean(silhouettes))
    assert 0.05 <= mean_silhouette <= 0.55, (
        f"steric=0.0 (phantom-particle mode): time-averaged silhouette "
        f"Θ'={mean_silhouette:.4f} outside the marginal-opacity band "
        f"[0.05, 0.55] -- marginal opacity should emerge from "
        f"projection alone"
    )


def test_marginal_opacity_constants_accessible():
    """S3.6a: MARGINAL_OPACITY_MEAN and MARGINAL_OPACITY_STD are
    importable and within reasonable ranges."""
    from pymurmur.analysis.metrics import MARGINAL_OPACITY_MEAN, MARGINAL_OPACITY_STD
    assert 0.0 < MARGINAL_OPACITY_MEAN < 1.0, (
        f"MARGINAL_OPACITY_MEAN={MARGINAL_OPACITY_MEAN} should be in (0,1)"
    )
    assert 0.0 < MARGINAL_OPACITY_STD < 1.0, (
        f"MARGINAL_OPACITY_STD={MARGINAL_OPACITY_STD} should be in (0,1)"
    )


def test_b10_public_opacity_constants_accessible_and_in_band():
    """B10: PUBLIC_OPACITY_MEAN/STD (Pearce et al. 2014's second,
    public-domain-image dataset) are importable, in reasonable ranges,
    and the mean falls inside the same S3.6a acceptance band
    [0.05, 0.55] the "own data" MARGINAL_OPACITY_MEAN anchor is
    validated against -- i.e. the existing self-regulation regression
    test already implicitly validates against both empirical anchors,
    not just the first one."""
    from pymurmur.analysis.metrics import PUBLIC_OPACITY_MEAN, PUBLIC_OPACITY_STD
    assert 0.0 < PUBLIC_OPACITY_MEAN < 1.0, (
        f"PUBLIC_OPACITY_MEAN={PUBLIC_OPACITY_MEAN} should be in (0,1)"
    )
    assert 0.0 < PUBLIC_OPACITY_STD < 1.0, (
        f"PUBLIC_OPACITY_STD={PUBLIC_OPACITY_STD} should be in (0,1)"
    )
    assert 0.05 <= PUBLIC_OPACITY_MEAN <= 0.55, (
        f"PUBLIC_OPACITY_MEAN={PUBLIC_OPACITY_MEAN} should fall inside "
        f"the S3.6a acceptance band [0.05, 0.55]"
    )


# ── S2.B4: Physical metrics edge cases ────────────────────────────

def test_physical_metrics_zero_mass():
    """S2.B4: bird_mass_kg=0 → force_real_N=0, power_real_W=0, energy_J=0."""
    from pymurmur.analysis.metrics import _compute_physical_metrics

    speeds = np.array([2.0, 3.0], dtype=np.float32)
    acc_mags = np.array([0.5, 1.0], dtype=np.float32)
    velocities = np.array([[2.0, 0, 0], [0, 3.0, 0]], dtype=np.float32)
    accs = np.array([[0.5, 0, 0], [1.0, 0, 0]], dtype=np.float32)
    dt = 1.0 / 60.0

    m = FlockMetrics()
    _compute_physical_metrics(m, speeds, acc_mags, velocities, accs,
                               0.0, 10.0, 40.0, 4.0, 5.0, dt)
    assert m.force_real_N == 0.0, f"Zero mass → zero force, got {m.force_real_N}"
    assert m.power_real_W == 0.0, f"Zero mass → zero power, got {m.power_real_W}"
    assert m.energy_J == 0.0, f"Zero mass → zero energy, got {m.energy_J}"
    # Speed should still be computed (doesn't depend on mass)
    assert m.speed_real_ms > 0.0


def test_physical_metrics_energy_scales_with_dt():
    """S2.B4: energy_J = power_real_W * dt — doubling dt doubles energy."""
    from pymurmur.analysis.metrics import _compute_physical_metrics

    speeds = np.array([2.0, 3.0], dtype=np.float32)
    acc_mags = np.array([0.5, 1.0], dtype=np.float32)
    velocities = np.array([[2.0, 0, 0], [0, 3.0, 0]], dtype=np.float32)
    accs = np.array([[0.5, 0, 0], [1.0, 0, 0]], dtype=np.float32)
    dt = 1.0 / 60.0

    m1 = FlockMetrics()
    _compute_physical_metrics(m1, speeds, acc_mags, velocities, accs,
                               0.08, 10.0, 40.0, 4.0, 5.0, dt)

    m2 = FlockMetrics()
    _compute_physical_metrics(m2, speeds, acc_mags, velocities, accs,
                               0.08, 10.0, 40.0, 4.0, 5.0, dt * 2.0)

    # Power should be the same (doesn't depend on dt)
    assert m1.power_real_W == pytest.approx(m2.power_real_W, rel=1e-5)
    # Energy should double
    assert m2.energy_J == pytest.approx(m1.energy_J * 2.0, rel=1e-5), (
        f"Doubling dt should double energy: {m1.energy_J} → {m2.energy_J}"
    )


# ── G7: Fastmath × metrics-export warning ──────────────────────

class TestFastmathMetricsWarning:
    """G7: Exporting metrics with perf.fastmath=True raises a RuntimeWarning."""

    def test_fastmath_true_raises_warning_on_first_collect(self):
        """G7: First collect() with fastmath=True emits RuntimeWarning."""
        import warnings

        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock

        cfg = SimConfig(num_boids=10, seed=42)
        cfg.perf.fastmath = True
        flock = PhysicsFlock(cfg)
        collector = MetricsCollector(cfg)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            collector.collect(flock, 0)
            # Should have at least one warning about fastmath
            fastmath_warnings = [x for x in w
                                if "fastmath" in str(x.message).lower()]
            assert len(fastmath_warnings) >= 1, (
                f"Expected a fastmath warning, got {[str(x.message) for x in w]}"
            )
            assert issubclass(fastmath_warnings[0].category, RuntimeWarning)

    def test_fastmath_false_no_warning(self):
        """G7: No warning when fastmath=False (default)."""
        import warnings

        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock

        cfg = SimConfig(num_boids=10, seed=42)
        cfg.perf.fastmath = False
        flock = PhysicsFlock(cfg)
        collector = MetricsCollector(cfg)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            collector.collect(flock, 0)
            fastmath_warnings = [x for x in w
                                if "fastmath" in str(x.message).lower()]
            assert len(fastmath_warnings) == 0, (
                "Unexpected fastmath warning with fastmath=False"
            )

    def test_fastmath_warning_only_once(self):
        """G7: Warning fires only on the first collect(), not every frame."""
        import warnings

        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock

        cfg = SimConfig(num_boids=10, seed=42)
        cfg.perf.fastmath = True
        flock = PhysicsFlock(cfg)
        collector = MetricsCollector(cfg)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for frame in range(5):
                collector.collect(flock, frame)
            fastmath_warnings = [x for x in w
                                if "fastmath" in str(x.message).lower()]
            assert len(fastmath_warnings) == 1, (
                f"Expected exactly 1 fastmath warning, got {len(fastmath_warnings)}"
            )

    def test_fastmath_warning_state_not_shared_across_instances(self):
        """G7: A fresh MetricsCollector instance warns again — the
        one-shot guard (`_warned_fastmath`) is per-instance state, not
        class-level/shared.  This matters because `engine.reset()`
        constructs a brand-new MetricsCollector every time; if the flag
        ever leaked across instances, a second engine (or a reset
        engine) would incorrectly stay silent about fastmath."""
        import warnings

        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock

        cfg = SimConfig(num_boids=10, seed=42)
        cfg.perf.fastmath = True
        flock = PhysicsFlock(cfg)

        for _ in range(2):
            collector = MetricsCollector(cfg)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                collector.collect(flock, 0)
                fastmath_warnings = [x for x in w
                                    if "fastmath" in str(x.message).lower()]
                assert len(fastmath_warnings) == 1, (
                    f"Fresh MetricsCollector instance must warn independently, "
                    f"got {len(fastmath_warnings)} warning(s)"
                )
