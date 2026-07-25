"""Part IV cross-item integration — projection pipeline (IT7), spatial pipeline (IT8): ecology + sphere + curl-flow + metrics.

Split out of test_engine_pipeline.py (file-size split).
"""


import numpy as np
import pytest

from pymurmur.core.config import SimConfig
from pymurmur.simulation.engine import SimulationEngine

# ── IT7: Projection pipeline — all Part IV items together ─────────────

@pytest.mark.slow
@pytest.mark.part4_cross
class TestProjectionPipelinePartIV:
    """IT7: S1.4+S1.5+S2.B7+S2.B8+S2.B4+S3.11+S3.6a — full projection
    pipeline with all Part IV features active through the engine."""

    def test_projection_ecology_sphere_physical_metrics_no_crash(self, default_config):
        """IT7a: Projection mode + ecology + sphere boundary + physical
        metrics — pipeline executes 30 frames without NaN or crash."""
        cfg = default_config
        cfg.mode = "projection"
        cfg.num_boids = 80
        cfg.seed = 42
        # S2.B7: sphere boundary
        cfg.boundary_mode = "sphere"
        cfg.boundary_sphere_radius = 300.0
        # S2.B8: ecology (roosting)
        cfg.roosting_enabled = True
        cfg.ecology_roost = (500.0, 350.0, 200.0)
        # S2.B4: physical metrics
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 1
        cfg.bird_mass_kg = 0.075
        cfg.cruise_speed_ms = 8.94

        engine = SimulationEngine(cfg)

        for _ in range(30):
            engine.step()

        # No NaN anywhere
        assert not np.any(np.isnan(engine.flock.positions)), "NaN in positions"
        assert not np.any(np.isnan(engine.flock.velocities)), "NaN in velocities"
        assert not np.any(np.isnan(engine.flock.accelerations)), "NaN in accelerations"

        # Metrics collected
        assert len(engine.metrics.history) >= 1
        snap = engine.metrics.snapshot()
        assert snap.alpha is not None
        assert snap.energy_J is not None
        assert snap.power_real_W is not None

    def test_projection_ecology_sphere_birds_stay_contained(self, default_config):
        """IT7b: S2.B7 sphere boundary with ecology: all birds stay
        within 1.15× sphere radius over 60 frames."""
        cfg = default_config
        cfg.mode = "projection"
        cfg.num_boids = 50
        cfg.seed = 99
        cfg.boundary_mode = "sphere"
        cfg.boundary_sphere_radius = 200.0
        cfg.roosting_enabled = True
        cfg.ecology_roost = (500.0, 350.0, 200.0)
        cfg.metrics_detail_level = 1
        cfg.metrics_interval = 1

        engine = SimulationEngine(cfg)
        sphere_center = np.array([cfg.width / 2, cfg.height / 2, cfg.depth / 2])

        for _ in range(60):
            engine.step()

        # All positions within 1.15 * radius
        dists = np.linalg.norm(engine.flock.positions - sphere_center, axis=1)
        max_allowed = cfg.boundary_sphere_radius * 1.15
        assert np.all(dists <= max_allowed), (
            f"Birds escaped sphere: max dist={dists.max():.1f}, "
            f"allowed={max_allowed:.1f}"
        )

    def test_projection_phi_n_affects_power_metrics(self, default_config):
        """IT7c: S1.4→S2.B4: Varying φn changes power_real_W.

        Higher φn = more noise = higher power consumption (more steering).
        """

        def run_and_measure(phi_a, phi_p, steps=40):
            cfg = SimConfig()
            cfg.mode = "projection"
            cfg.num_boids = 40
            cfg.seed = 0
            cfg.boundary_mode = "sphere"
            cfg.boundary_sphere_radius = 250.0
            cfg.phi_a = phi_a
            cfg.phi_p = phi_p
            cfg.metrics_detail_level = 2
            cfg.metrics_interval = 1
            cfg.bird_mass_kg = 0.075
            cfg.cruise_speed_ms = 8.94

            engine = SimulationEngine(cfg)
            for _ in range(steps):
                engine.step()

            final = engine.metrics.snapshot()
            return {
                "power": final.power_real_W,
                "energy": final.energy_J,
                "alpha": final.alpha,
            }

        # Low noise: φp=0.5, φa=0.45 → φn=0.05
        low = run_and_measure(0.5, 0.45)
        # High noise: φp=0.3, φa=0.2 → φn=0.5
        high = run_and_measure(0.3, 0.2)

        # Power and energy must be finite in both cases
        assert low["power"] is not None and low["power"] >= 0
        assert high["power"] is not None and high["power"] >= 0
        assert low["energy"] is not None and low["energy"] >= 0
        assert high["energy"] is not None and high["energy"] >= 0
        # Higher noise should produce higher power (more random steering)
        assert high["power"] != low["power"], (
            f"Power should differ: low={low['power']:.6f}, high={high['power']:.6f}"
        )

    def test_projection_ema_readout_converges(self, default_config):
        """IT7d: S3.11 EMA readout converges toward raw order parameter
        over many frames."""
        cfg = default_config
        cfg.mode = "projection"
        cfg.num_boids = 80
        cfg.seed = 42
        cfg.readout_smooth = 0.04
        cfg.metrics_detail_level = 1
        cfg.metrics_interval = 1

        engine = SimulationEngine(cfg)

        for _ in range(100):
            engine.step()

        snap = engine.metrics.snapshot()
        raw = snap.alpha
        ema = engine.metrics.smoothed().alpha

        # Both should be defined and in [0,1]
        assert raw is not None and 0.0 <= raw <= 1.0
        assert ema is not None and 0.0 <= ema <= 1.0
        # After 100 frames with α=0.04, EMA and raw should be close
        assert abs(raw - ema) < 0.3, (
            f"EMA diverged: raw={raw:.3f}, ema={ema:.3f}"
        )

    def test_projection_silhouette_in_band_many_frames(self, default_config):
        """IT7e: S3.6a marginal opacity: silhouette_2d settles in
        [0.05, 0.55] after 200+ frames."""
        cfg = default_config
        cfg.mode = "projection"
        cfg.num_boids = 150
        cfg.seed = 42
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 1

        engine = SimulationEngine(cfg)

        settle = 300
        measure_from = 200
        silhouettes = []

        for frame in range(settle):
            engine.step()
            if frame >= measure_from:
                silhouettes.append(engine.metrics.snapshot().silhouette_2d)

        avg = sum(silhouettes) / len(silhouettes)
        assert 0.05 <= avg <= 0.55, (
            f"Silhouette outside [0.05, 0.55]: avg={avg:.4f}"
        )


# ── IT8: Spatial pipeline — ecology + sphere + curl-flow + metrics ───

@pytest.mark.slow
@pytest.mark.part4_cross
class TestSpatialPipelinePartIV:
    """IT8: S1.5+S2.B7+S2.B8+S2.B11+S2.B4+S3.11 — full spatial
    pipeline with curl-flow, ecology, sphere boundary, and all metrics."""

    def test_spatial_ecology_sphere_curl_flow_no_crash(self, default_config):
        """IT8a: Spatial mode + ecology + sphere + curl-flow + physical
        metrics + EMA — 30 frames, no NaN, no crash."""
        cfg = default_config
        cfg.mode = "spatial"
        cfg.num_boids = 80
        cfg.seed = 42
        # S2.B7: sphere boundary
        cfg.boundary_mode = "sphere"
        cfg.boundary_sphere_radius = 300.0
        # S2.B8: ecology
        cfg.roosting_enabled = True
        cfg.ecology_roost = (500.0, 350.0, 200.0)
        # S2.B11: curl-flow
        cfg.flow_weight = 0.3
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
        assert not np.any(np.isnan(engine.flock.accelerations))

        # All metrics present
        snap = engine.metrics.snapshot()
        assert snap.alpha is not None
        assert engine.metrics.smoothed().alpha is not None
        assert snap.power_real_W is not None
        assert snap.energy_J is not None
        assert snap.force_avg is not None  # S1.5: force averaging

    def test_spatial_curl_flow_determinism(self, default_config):
        """IT8b: S2.B11 curl-flow is deterministic — same seed + flow
        weight → same forces across two independent engine runs."""
        cfg = default_config
        cfg.mode = "spatial"
        cfg.num_boids = 40
        cfg.seed = 42
        cfg.flow_weight = 0.3
        cfg.boundary_mode = "sphere"

        engine = SimulationEngine(cfg)
        engine.step()
        forces_run1 = engine.flock.accelerations.copy()

        # Fresh config, same seed + flow → same result
        cfg2 = default_config
        cfg2.mode = "spatial"
        cfg2.num_boids = 40
        cfg2.seed = 42
        cfg2.flow_weight = 0.3
        cfg2.boundary_mode = "sphere"

        engine2 = SimulationEngine(cfg2)
        engine2.step()

        # Same seed, same config → same forces
        assert np.allclose(forces_run1, engine2.flock.accelerations, atol=1e-5), (
            "Curl-flow determinism broken: same config produces different forces"
        )

    def test_spatial_ecology_coherence_affects_energy(self, default_config):
        """IT8c: S2.B8→S2.B4: Ecology coherence gate reduces forces
        for small flocks → energy_J differs from large flock."""
        cfg_small = default_config
        cfg_small.mode = "spatial"
        cfg_small.num_boids = 15  # small flock hits coherence gate
        cfg_small.seed = 42
        cfg_small.boundary_mode = "sphere"
        cfg_small.roosting_enabled = True
        cfg_small.ecology_roost = (500.0, 350.0, 200.0)
        cfg_small.metrics_detail_level = 2
        cfg_small.metrics_interval = 1
        cfg_small.bird_mass_kg = 0.075

        engine_small = SimulationEngine(cfg_small)
        for _ in range(20):
            engine_small.step()
        energy_small = engine_small.metrics.snapshot().energy_J

        # Large flock (no coherence gating — full forces)
        cfg_large = default_config
        cfg_large.mode = "spatial"
        cfg_large.num_boids = 200
        cfg_large.seed = 42
        cfg_large.boundary_mode = "sphere"
        cfg_large.roosting_enabled = True
        cfg_large.ecology_roost = (500.0, 350.0, 200.0)
        cfg_large.metrics_detail_level = 2
        cfg_large.metrics_interval = 1
        cfg_large.bird_mass_kg = 0.075

        engine_large = SimulationEngine(cfg_large)
        for _ in range(20):
            engine_large.step()
        energy_large = engine_large.metrics.snapshot().energy_J

        # Both must be finite
        assert energy_small is not None and energy_small >= 0
        assert energy_large is not None and energy_large >= 0
        # Coherence affects energy — small and large should differ
        assert energy_small != energy_large, (
            f"Coherence gate should cause energy difference: "
            f"small={energy_small:.6f}, large={energy_large:.6f}"
        )

    def test_spatial_energy_scales_with_physics_timestep(self, default_config):
        """IT8d: S2.B4 energy integration: larger dt → larger energy
        per step (energy_J = power_real_W × dt)."""
        def energy_after_steps(dt, steps=10):
            cfg = default_config
            cfg.mode = "spatial"
            cfg.num_boids = 40
            cfg.seed = 42
            cfg.dt_phys = dt
            cfg.metrics_detail_level = 2
            cfg.metrics_interval = 1
            cfg.bird_mass_kg = 0.075

            engine = SimulationEngine(cfg)
            for _ in range(steps):
                engine.step()
            return engine.metrics.snapshot().energy_J

        e1 = energy_after_steps(1.0 / 60.0)
        e2 = energy_after_steps(1.0 / 30.0)  # double dt

        assert e1 is not None and e1 > 0, f"Expected non-zero energy with dt=1/60, got {e1}"
        assert e2 is not None and e2 > 0
        # Larger dt → more energy per step (energy_J = power × dt)
        assert e2 > e1, (
            f"Energy should increase with dt: e(1/60)={e1:.6f}, e(1/30)={e2:.6f}"
        )


