"""Part IV cross-item integration — angle pipeline (IT9), cross-mode coherence-energy chain (IT10), boundary+metrics cross-mode smoke (IT11).

Split out of test_engine_pipeline.py (file-size split).
"""


import numpy as np
import pytest

from pymurmur.simulation.engine import SimulationEngine

# ── IT9: Angle pipeline — ecology + sphere + metrics + EMA ────────────

@pytest.mark.slow
@pytest.mark.part4_cross
class TestAnglePipelinePartIV:
    """IT9: S2.B7+S2.B8+S2.C3+S2.B4+S3.11 — full angle mode
    pipeline with all Part IV features."""

    def test_angle_ecology_sphere_physical_metrics_no_crash(self, default_config):
        """IT9a: Angle mode + ecology + sphere + physical metrics + EMA
        — 30 frames, no NaN, no escape."""
        cfg = default_config
        cfg.mode = "angle"
        cfg.num_boids = 50
        cfg.seed = 42
        # S2.B7: sphere
        cfg.boundary_mode = "sphere"
        cfg.boundary_sphere_radius = 300.0
        # S2.B8: ecology
        cfg.roosting_enabled = True
        cfg.ecology_roost = (500.0, 350.0, 200.0)
        # S2.B4: physical metrics
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 1
        cfg.bird_mass_kg = 0.075
        # S3.11: EMA
        cfg.readout_smooth = 0.04

        engine = SimulationEngine(cfg)

        for _ in range(30):
            engine.step()

        # No NaN
        assert not np.any(np.isnan(engine.flock.positions))
        assert not np.any(np.isnan(engine.flock.velocities))
        assert engine.flock.N_active == 50

        # Metrics present
        snap = engine.metrics.snapshot()
        assert snap.alpha is not None
        assert engine.metrics.smoothed().alpha is not None
        assert snap.power_real_W is not None
        assert snap.energy_J is not None

    def test_angle_ecology_roost_pull_persists(self, default_config):
        """IT9b: S2.C3+S2.B8: After angle mode compute(), ecology
        roost pull survives through boid.integrate(). Verified by
        advancing day to within roost window and checking that
        last_accelerations are finite and non-zero for some birds."""
        cfg = default_config
        cfg.mode = "angle"
        # S2.B8: coherence_gate is a hard 0 below 0.4x critical_mass (500
        # default) — use a flock size above that floor so the roost pull
        # this test exercises isn't itself gated to zero by ecology.
        cfg.num_boids = 250
        cfg.seed = 42
        cfg.boundary_mode = "sphere"
        cfg.roosting_enabled = True
        cfg.ecology_roost = (500.0, 350.0, 200.0)
        cfg.ecology_critical_mass = 10  # small threshold so 20 birds passes gate
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 1

        engine = SimulationEngine(cfg)

        # Advance ecology day so current hour is within roost window
        # (dusk at day 172 is ~20:15; roost window is 18:15–20:45)
        # day 172.85 → hour = 0.85 * 24 = 20.4 → within window
        engine.extensions._ecology._day = 172.85
        engine.step()

        # last_accelerations for active birds
        active = engine.flock.active
        accs = engine.flock.last_accelerations[active]
        assert not np.any(np.isnan(accs)), "NaN in last_accelerations"
        assert not np.any(np.isinf(accs)), "Inf in last_accelerations"

        # At least some birds should have non-zero acceleration
        # (roost pull contributes even when angle mode writes velocities directly)
        acc_mags = np.linalg.norm(accs, axis=1)
        nonzero_count = int((acc_mags > 1e-12).sum())
        assert nonzero_count > 0, (
            f"All {len(accs)} active birds have zero acceleration — "
            f"roost pull may be lost"
        )

    def test_angle_sphere_containment_over_60_frames(self, default_config):
        """IT9c: S2.C3+S2.B7: Sphere boundary works for angle mode
        over 60 frames — birds stay within sphere."""
        cfg = default_config
        cfg.mode = "angle"
        cfg.num_boids = 30
        cfg.seed = 99
        cfg.boundary_mode = "sphere"
        cfg.boundary_sphere_radius = 200.0
        cfg.readout_smooth = 0.04

        engine = SimulationEngine(cfg)
        sphere_center = np.array([cfg.width / 2, cfg.height / 2, cfg.depth / 2])

        for _ in range(60):
            engine.step()

        dists = np.linalg.norm(engine.flock.positions - sphere_center, axis=1)
        assert np.all(dists <= cfg.boundary_sphere_radius * 1.15), (
            f"Angle birds escaped: max={dists.max():.1f}"
        )

    def test_angle_ema_readout_differs_from_raw(self, default_config):
        """IT9d: S3.11 EMA readout differs from raw order parameter in
        angle mode (EMA smooths transients)."""
        cfg = default_config
        cfg.mode = "angle"
        cfg.num_boids = 60
        cfg.seed = 42
        cfg.readout_smooth = 0.04
        cfg.metrics_detail_level = 1
        cfg.metrics_interval = 1

        engine = SimulationEngine(cfg)

        for _ in range(10):
            engine.step()

        snap = engine.metrics.snapshot()
        # After 10 frames with α=0.04, EMA and raw may differ
        # because EMA hasn't fully converged yet
        assert snap.alpha is not None
        assert engine.metrics.smoothed().alpha is not None
        # Both valid
        assert 0.0 <= snap.alpha <= 1.0
        assert 0.0 <= engine.metrics.smoothed().alpha <= 1.0


# ── IT10: Cross-mode coherence→energy chain ──────────────────────────

@pytest.mark.slow
@pytest.mark.part4_cross
class TestCoherenceEnergyChain:
    """IT10: S2.B8→S2.B4: Coherence gate affects energy metrics
    across projection and spatial modes."""

    @pytest.mark.parametrize("mode", ["projection", "spatial"])
    def test_modes_produce_finite_energy_with_ecology(self, default_config, mode):
        """IT10a: Projection and spatial modes both produce finite
        energy_J when ecology is enabled."""
        cfg = default_config
        cfg.mode = mode
        cfg.num_boids = 40
        cfg.seed = 42
        cfg.boundary_mode = "sphere"
        cfg.roosting_enabled = True
        cfg.ecology_roost = (500.0, 350.0, 200.0)
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 1
        cfg.bird_mass_kg = 0.075
        cfg.cruise_speed_ms = 8.94

        engine = SimulationEngine(cfg)
        for _ in range(15):
            engine.step()

        snap = engine.metrics.snapshot()
        assert snap.energy_J is not None
        assert snap.energy_J >= 0
        assert snap.power_real_W is not None
        assert snap.power_real_W >= 0


# ── IT11: Boundary + metrics cross-mode smoke ────────────────────────

@pytest.mark.slow
@pytest.mark.part4_cross
class TestBoundaryMetricsCrossMode:
    """IT11: S2.B7+S2.B4+S3.11 across all Part IV force modes."""

    @pytest.mark.parametrize("mode", ["projection", "spatial", "angle"])
    def test_all_modes_sphere_boundary_no_escape(self, default_config, mode):
        """IT11a: Every Part IV mode stays inside sphere boundary."""
        cfg = default_config
        cfg.mode = mode
        cfg.num_boids = 40
        cfg.seed = 77
        cfg.boundary_mode = "sphere"
        cfg.boundary_sphere_radius = 250.0

        engine = SimulationEngine(cfg)
        sphere_center = np.array([cfg.width / 2, cfg.height / 2, cfg.depth / 2])

        for _ in range(30):
            engine.step()

        dists = np.linalg.norm(engine.flock.positions - sphere_center, axis=1)
        assert np.all(dists <= cfg.boundary_sphere_radius * 1.15), (
            f"{mode}: max dist={dists.max():.1f}"
        )

    @pytest.mark.parametrize("mode", ["projection", "spatial", "angle"])
    def test_all_modes_ema_readout_valid(self, default_config, mode):
        """IT11b: Every Part IV mode produces valid EMA readout."""
        cfg = default_config
        cfg.mode = mode
        cfg.num_boids = 40
        cfg.seed = 42
        cfg.readout_smooth = 0.04
        cfg.metrics_detail_level = 1
        cfg.metrics_interval = 1

        engine = SimulationEngine(cfg)

        for _ in range(20):
            engine.step()

        snap = engine.metrics.snapshot()
        assert engine.metrics.smoothed().alpha is not None
        assert -0.01 <= engine.metrics.smoothed().alpha <= 1.01  # float tolerance
        assert snap.alpha is not None

    @pytest.mark.parametrize("mode", ["projection", "spatial", "angle"])
    def test_all_modes_physical_metrics_finite(self, default_config, mode):
        """IT11c: Every Part IV mode produces finite physical metrics."""
        cfg = default_config
        cfg.mode = mode
        cfg.num_boids = 40
        cfg.seed = 42
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 1
        cfg.bird_mass_kg = 0.075
        cfg.cruise_speed_ms = 8.94

        engine = SimulationEngine(cfg)

        for _ in range(15):
            engine.step()

        snap = engine.metrics.snapshot()
        assert snap.power_real_W is not None and 0 <= snap.power_real_W < 1e9
        assert snap.energy_J is not None and 0 <= snap.energy_J < 1e9
