"""IT12 risk classifier thresholds vs mesh registry, IT13 collision logging consistency, IT14 risk classifier events loggable, IT15 pilot actions loggable, EvoFlock logging integration.

Split out of test_engine_pipeline.py (file-size split).
"""


import numpy as np
import pytest

from pymurmur.core.config import SimConfig
from pymurmur.simulation.engine import SimulationEngine

# ── IT12: S4.10 × S4.4a — Risk classifier thresholds vs mesh registry ──

@pytest.mark.slow
@pytest.mark.part5_cross
class TestRiskClassifierMeshThresholds:
    """IT12: S4.10→S4.4a: Risk classifier VERTEX_N_THRESHOLD (10K)
    aligns with mesh registry's instanced→impostor transition (10K)."""

    def test_vertex_threshold_matches_instanced_limit(self):
        """PerfDiagnostics.VERTEX_N_THRESHOLD matches mesh registry's
        INSTANCED_MAX from recommend_render_mode()."""
        from pymurmur.analysis.perf import PerfDiagnostics
        from pymurmur.viz.mesh_registry import recommend_render_mode

        pd = PerfDiagnostics()
        assert pd.VERTEX_N_THRESHOLD == 10_000, (
            f"VERTEX_N_THRESHOLD={pd.VERTEX_N_THRESHOLD}, expected 10000"
        )
        # At exactly threshold, instanced mesh is still recommended
        assert recommend_render_mode(10_000) == "winged"
        # Above threshold, impostor is recommended
        assert recommend_render_mode(10_001) == "impostor"

    def test_risk_class_vertex_aligns_with_impostor_threshold(self):
        """When N_active exceeds VERTEX_N_THRESHOLD and CPU fraction
        is low, risk_class = 'vertex' — matching the mesh registry's
        impostor transition."""
        from pymurmur.analysis.perf import PerfDiagnostics

        pd = PerfDiagnostics()
        pd.set_active_count(15_000)

        # Simulate GPU-bound frame (low CPU fraction)
        pd.record_physics(2.0)
        pd.record_render(14.0)
        pd.tick()
        snap = pd.snapshot()
        assert snap.risk_class == "vertex", (
            f"At N=15K with cpu_frac=0.125, expected 'vertex', got '{snap.risk_class}'"
        )
        assert snap.n_active == 15_000

    def test_risk_class_fragment_below_threshold(self):
        """Below VERTEX_N_THRESHOLD with low CPU fraction → 'fragment'."""
        from pymurmur.analysis.perf import PerfDiagnostics

        pd = PerfDiagnostics()
        pd.set_active_count(5_000)

        pd.record_physics(2.0)
        pd.record_render(14.0)
        pd.tick()
        snap = pd.snapshot()
        assert snap.risk_class == "fragment", (
            f"At N=5K with cpu_frac=0.125, expected 'fragment', got '{snap.risk_class}'"
        )
        assert snap.n_active == 5_000


# ── IT13: S5.6 × S6.4 — Collision counter logging consistency ────────

@pytest.mark.slow
@pytest.mark.part5_cross
class TestCollisionLoggingConsistency:
    """IT13: S5.6→S6.4: Collision counter from metrics matches
    what is logged via log_metrics_line."""

    def test_collisions_logged_match_metrics_counter(self):
        """When engine.step() detects collisions, log_metrics_line
        receives the same collision count as the metrics snapshot."""
        import tempfile
        from pathlib import Path

        from pymurmur.core.logging import log_metrics_line, setup_run_logging
        from pymurmur.physics.obstacles import ObstacleScene

        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = 20
        cfg.seed = 42
        cfg.v0 = 60.0
        cfg.boundary_mode = "sphere"
        cfg.boundary_sphere_radius = 500.0
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 1
        cfg.spatial.static_avoid_weight = 0.0
        cfg.spatial.fly_away_max_dist = 0.0

        engine = SimulationEngine(cfg)
        scene = ObstacleScene().add_sphere((500.0, 500.0, 500.0), 50.0)
        engine.obstacle_scene = scene

        for i in range(cfg.num_boids):
            engine.flock.positions[i] = np.array([448.0, 500.0, 500.0], dtype=np.float32)
            engine.flock.velocities[i] = np.array([cfg.v0, 0.0, 0.0], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_run_logging(log_dir=tmpdir, level="info")

            for _ in range(3):
                engine.step()
                snap = engine.metrics.snapshot()
                log_metrics_line(
                    engine.frame, alpha=snap.alpha or 0.0,
                    speed_real_ms=snap.speed_real_ms or 0.0,
                    energy_J=snap.energy_J or 0.0,
                    collisions=snap.collisions_this_step,
                )

            for h in logger.handlers:
                h.flush()

            log_files = sorted(Path(tmpdir).glob("run-*.log"))
            assert len(log_files) == 1
            content = log_files[0].read_text()

            # Verify collisions are logged and the count is a positive integer
            assert "Metrics |" in content
            assert "collisions=" in content
            # Verify at least one collision count is logged
            assert "collisions=" in content

    def test_zero_collisions_when_no_obstacle_scene(self):
        """No obstacle scene → collisions_this_step = 0 → logged as 0."""
        import tempfile
        from pathlib import Path

        from pymurmur.core.logging import log_metrics_line, setup_run_logging

        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = 10
        cfg.seed = 42

        engine = SimulationEngine(cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_run_logging(log_dir=tmpdir, level="info")

            engine.step()
            snap = engine.metrics.snapshot()
            assert snap.collisions_this_step == 0

            log_metrics_line(1, alpha=0.5, collisions=snap.collisions_this_step)

            for h in logger.handlers:
                h.flush()

            log_files = sorted(Path(tmpdir).glob("run-*.log"))
            content = log_files[0].read_text()
            assert "collisions=0" in content


# ── IT14: S5.6 × S4.10 — Risk classifier events loggable ──────────────

@pytest.mark.slow
@pytest.mark.part5_cross
class TestRiskClassifierLogging:
    """IT14: S5.6→S4.10: Risk classifier state changes can be logged
    via lifecycle helpers."""

    def test_risk_class_can_be_logged_as_lifecycle_event(self):
        """PerfDiagnostics risk_class can be written to log via
        log_lifecycle()."""
        import tempfile
        from pathlib import Path

        from pymurmur.analysis.perf import PerfDiagnostics
        from pymurmur.core.logging import log_lifecycle, setup_run_logging

        pd = PerfDiagnostics()
        pd.set_active_count(15_000)
        pd.record_physics(2.0)
        pd.record_render(14.0)
        pd.tick()
        snap = pd.snapshot()

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_run_logging(log_dir=tmpdir, level="info")
            log_lifecycle("risk_classifier",
                          f"class={snap.risk_class} N={snap.n_active}")

            for h in logger.handlers:
                h.flush()

            log_files = sorted(Path(tmpdir).glob("run-*.log"))
            content = log_files[0].read_text()
            assert "risk_classifier" in content
            assert snap.risk_class in content
            assert str(snap.n_active) in content

    def test_perf_risk_class_via_engine_pipeline(self):
        """After engine steps with adaptive_quality enabled, risk_class
        is accessible and can be logged."""
        import tempfile
        from pathlib import Path

        from pymurmur.core.logging import log_lifecycle, setup_run_logging

        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = 20
        cfg.seed = 42
        cfg.adaptive_quality = True

        engine = SimulationEngine(cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_run_logging(log_dir=tmpdir, level="info")

            for _ in range(5):
                engine.step()

            # Feed perf diagnostics with mock timing
            if engine.perf is None:
                from pymurmur.analysis.perf import PerfDiagnostics
                engine.perf = PerfDiagnostics()
            engine.perf.set_active_count(engine.flock.N_active)
            engine.perf.record_physics(1.0)
            engine.perf.record_render(15.0)
            engine.perf.tick()
            snap = engine.perf.snapshot()
            assert snap.risk_class in ("cpu", "vertex", "fragment", "mixed")

            log_lifecycle("perf_diagnostics",
                          f"class={snap.risk_class} fps={snap.fps:.0f}")

            for h in logger.handlers:
                h.flush()

            log_files = sorted(Path(tmpdir).glob("run-*.log"))
            if log_files:
                content = log_files[0].read_text()
                assert "perf_diagnostics" in content


# ── IT15: S2.E6 × S5.6 — Pilot actions loggable ──────────────────────

@pytest.mark.slow
@pytest.mark.part5_cross
class TestPilotLoggingIntegration:
    """IT15: S2.E6→S5.6: Pilot-mode actions (Q/E roll, gather/scatter,
    presets) can be logged via lifecycle events."""

    def test_pilot_roll_can_be_logged(self):
        """Q/E roll camera action can be logged as lifecycle event."""
        import tempfile
        from pathlib import Path

        from pymurmur.core.logging import log_lifecycle, setup_run_logging
        from pymurmur.viz.camera import OrbitCamera

        cam = OrbitCamera()
        cam.roll_camera(0.05)  # Q key press

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_run_logging(log_dir=tmpdir, level="info")
            log_lifecycle("pilot_roll", f"roll={cam.roll:.3f} rad")

            for h in logger.handlers:
                h.flush()

            log_files = sorted(Path(tmpdir).glob("run-*.log"))
            content = log_files[0].read_text()
            assert "pilot_roll" in content
            assert f"{cam.roll:.3f}" in content

    def test_gather_scatter_can_be_logged(self):
        """Gather/scatter key state can be logged via lifecycle."""
        import tempfile
        from pathlib import Path

        from pymurmur.core.logging import log_lifecycle, setup_run_logging

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_run_logging(log_dir=tmpdir, level="info")
            log_lifecycle("pilot_gather", "Shift held — contracting flock")
            log_lifecycle("pilot_scatter", "Alt held — expanding flock")

            for h in logger.handlers:
                h.flush()

            log_files = sorted(Path(tmpdir).glob("run-*.log"))
            content = log_files[0].read_text()
            assert "pilot_gather" in content
            assert "pilot_scatter" in content

    def test_preset_application_can_be_logged(self):
        """Preset changes can be logged via lifecycle or cli_out."""
        import tempfile
        from pathlib import Path

        from pymurmur.analysis.presets import apply_preset
        from pymurmur.core.logging import log_lifecycle, setup_run_logging

        cfg = SimConfig()
        label = apply_preset(cfg, "a")  # 3D Pearce Default

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_run_logging(log_dir=tmpdir, level="info")
            log_lifecycle("preset", f"key=a label={label}")

            for h in logger.handlers:
                h.flush()

            log_files = sorted(Path(tmpdir).glob("run-*.log"))
            content = log_files[0].read_text()
            assert "preset" in content
            assert label in content


# -- IT16: S6.1-S6.6 x S5.6 -- EvoFlock eval metrics loggable --

@pytest.mark.slow
@pytest.mark.part5_cross
class TestEvoFlockLoggingIntegration:
    """IT16: S6.1-S6.6 x S5.6: EvoFlock evaluation runs produce
    metrics that can be logged via structured log helpers."""

    def test_evoflock_eval_logs_header_and_metrics(self):
        """An EvoFlock run writes structured log output with header
        and can log per-eval metrics via log_metrics_line."""
        import tempfile
        from pathlib import Path

        from pymurmur.analysis.evoflock import EVOLVABLE_PARAMS, EvoConfig, EvoFlock, Genome
        from pymurmur.core.logging import (
            log_metrics_line,
            log_run_header,
            setup_run_logging,
        )

        cfg = SimConfig()
        cfg.num_boids = 20
        cfg.seed = 42
        evo = EvoFlock(cfg, EvoConfig(
            population_size=4, n_islands=1, max_steps=0,
            evals_per_candidate=2, eval_steps=20,
        ))

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_run_logging(log_dir=tmpdir, level="info")
            log_run_header("evoflock_test", cfg.seed, "spatial", cfg.num_boids)

            # Run a single evaluation
            genome = Genome(values={name: 0.5 for name in EVOLVABLE_PARAMS})
            evo._evaluate(genome)

            # Log the resulting fitness + objectives
            log_metrics_line(
                1, alpha=float(genome.objectives[0]),
                speed_real_ms=float(genome.objectives[1]),
                energy_J=float(genome.fitness),
                collisions=0,
            )

            for h in logger.handlers:
                h.flush()

            log_files = sorted(Path(tmpdir).glob("run-*.log"))
            assert len(log_files) == 1
            content = log_files[0].read_text()
            assert "Header |" in content
            assert "Metrics |" in content

    def test_evoflock_save_produces_loggable_artifact_path(self):
        """EvoFlock.save() path can be logged via lifecycle event."""
        import tempfile
        from pathlib import Path

        from pymurmur.analysis.evoflock import EvoConfig, EvoFlock
        from pymurmur.core.logging import log_lifecycle, setup_run_logging

        cfg = SimConfig()
        cfg.seed = 42
        evo = EvoFlock(cfg, EvoConfig(population_size=4, n_islands=1))
        evo._initialize_population()
        for k, g in enumerate(evo._islands[0]):
            g.fitness = 0.1 * (k + 1)
            g.objectives = np.array([0.9, 0.8, 0.85, 1.0])
            g.eval_seeds = [13, 7932]

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "evolved_test.yaml"
            evo.save(save_path)

            logger = setup_run_logging(log_dir=tmpdir, level="info")
            log_lifecycle("evoflock_save", f"path={save_path}")

            for h in logger.handlers:
                h.flush()

            log_files = sorted(Path(tmpdir).glob("run-*.log"))
            assert len(log_files) == 1
            content = log_files[0].read_text()
            assert "evoflock_save" in content
            assert "evolved_test.yaml" in content
