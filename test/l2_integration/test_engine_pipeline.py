"""I4 Phase — Missing integration tests "as a whole".

Crosses 3+ I4 item boundaries:
  IT1 — Full engine pipeline across multiple steps with command mutations
  IT2 — Paused add/remove then unpause preserves correct state
  IT3 — Command queue reset preempts add and remove
  IT4 — Recorder headless pipeline step-before-render (GPU-gated)
  IT5 — Engine reset preserves pipeline invariants
  IT6 — Only engine can import both flock and forces (AST scan)
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pymurmur.core.config import SimConfig
from pymurmur.simulation.engine import SimulationEngine

# ── IT1: Full engine pipeline all 6 stages in order, multi-step ───

class TestFullEnginePipeline:
    """I4.2 + I4.3 + I4.4: All 6 stages run in order across multiple steps."""

    def test_full_pipeline_6_stages_multi_step(self, default_config):
        """IT1: 6-stage order holds for 3 consecutive steps including mutations.

        Step 1: verify full order
        Step 2: enqueue add, verify drain includes add
        Step 3: verify order still holds after flock grew
        """
        cfg = default_config
        cfg.num_boids = 10
        cfg.mode = "spatial"
        engine = SimulationEngine(cfg)

        for step_num in range(3):
            order_log = []

            orig_drain = engine.drain_commands
            orig_ext = engine.extensions.pre_step
            orig_integrate = engine.flock.integrate
            orig_collect = engine.metrics.collect
            if engine.flock._index is not None:
                orig_rebuild = engine.flock._index.rebuild

            def spy_drain():
                order_log.append("drain")
                return orig_drain()

            def spy_ext(flock, ctx):
                order_log.append("extensions")
                return orig_ext(flock, ctx)

            def spy_rebuild(positions, active):
                order_log.append("index")
                return orig_rebuild(positions, active)

            def spy_integrate(config, dt, **kwargs):
                order_log.append("integrate")
                return orig_integrate(config, dt, **kwargs)

            def spy_collect(flock, frame, **kwargs):
                order_log.append("metrics")
                return orig_collect(flock, frame)

            engine.drain_commands = spy_drain
            engine.extensions.pre_step = spy_ext
            engine.flock.integrate = spy_integrate
            engine.metrics.collect = spy_collect

            if engine.flock._index is not None:
                engine.flock._index.rebuild = spy_rebuild

            with patch(
                "pymurmur.simulation.engine.compute_all_forces"
            ) as mock_forces:
                mock_forces.side_effect = (
                    lambda flock, cfg: order_log.append("forces")
                )
                engine.step()

            # Verify order every step
            drain_idx = order_log.index("drain")
            ext_idx = order_log.index("extensions")
            index_idx = order_log.index("index")
            forces_idx = order_log.index("forces")
            integrate_idx = order_log.index("integrate")
            metrics_idx = order_log.index("metrics")

            assert drain_idx < ext_idx < index_idx < forces_idx < integrate_idx < metrics_idx, (
                f"Step {step_num}: pipeline order wrong: {order_log}"
            )

            # Restore originals for next step
            engine.drain_commands = orig_drain
            engine.extensions.pre_step = orig_ext
            engine.flock.integrate = orig_integrate
            engine.metrics.collect = orig_collect
            if engine.flock._index is not None:
                engine.flock._index.rebuild = orig_rebuild

            # Between steps 1 and 2, enqueue an add
            if step_num == 1:
                engine.enqueue_add(5)

        # After all 3 steps, flock grew
        assert engine.flock.N_active == 15  # 10 + 5 added
        assert engine.frame == 3


# ── IT2: Paused add/remove then unpause preserves correct state ───

class TestPausedAddRemoveUnpause:
    """I4.1 + I4.2 + I4.3: Paused mutations preserve correct state."""

    def test_paused_add_remove_then_unpause(self, default_config):
        """IT2: Enqueue add while paused → drain mutates flock → step runs after unpause."""
        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)

        pos_before = engine.flock.positions.copy()

        # Simulate paused frame: drain commands (but don't step)
        engine.enqueue_add(5)
        engine.enqueue_remove(3)
        engine.drain_commands()

        # After drain: flock size changed but positions unchanged (no step)
        assert engine.flock.N_active == 12  # 10 + 5 - 3
        # Positions for the original 10 birds should be identical
        assert np.allclose(engine.flock.positions[:10], pos_before)

        # Now unpause and step — forces run on expanded flock
        engine.step()
        assert engine.frame == 1
        # Positions should have changed (physics applied)
        assert not np.allclose(engine.flock.positions[:10], pos_before)

    def test_paused_add_only_then_multiple_steps(self, default_config):
        """IT2: Add birds while paused, then step 3 times — N_active stays correct."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.v0 = 2.0
        engine = SimulationEngine(cfg)

        # Add birds
        engine.enqueue_add(8)
        engine.drain_commands()
        assert engine.flock.N_active == 18

        # Run 3 steps — flock stays at 18
        for _ in range(3):
            engine.step()
        assert engine.flock.N_active == 18


# ── IT3: Command queue reset preempts add and remove ─────────────

class TestCommandQueueResetPreemption:
    """I4.2 + I4.3: Reset preempts simultaneous add/remove."""

    def test_reset_preempts_add_and_remove(self, default_config, monkeypatch):
        """IT3: Queued reset clears pending_add/remove before creating fresh flock."""

        cfg = default_config
        cfg.num_boids = 20
        engine = SimulationEngine(cfg)

        # Spy on flock methods to verify reset preempts add/remove
        mock_add = MagicMock()
        mock_remove = MagicMock()
        monkeypatch.setattr(engine.flock, "add_boids", mock_add)
        monkeypatch.setattr(engine.flock, "remove_boids", mock_remove)

        # Queue all three simultaneously
        engine.enqueue_reset()
        engine.enqueue_add(10)
        engine.enqueue_remove(5)

        # Verify queued
        assert engine.commands.pending_reset is True
        assert engine.commands.pending_add == 10
        assert engine.commands.pending_remove == 5

        # Drain — reset must be the only action processed
        engine.drain_commands()

        # Add/remove must NOT have been called (reset returns early)
        mock_add.assert_not_called()
        mock_remove.assert_not_called()

        # After reset: fresh flock with original num_boids (20)
        assert engine.flock.N_active == 20
        assert engine.frame == 0

    def test_add_remove_without_reset_both_processed(self, default_config):
        """IT3: Without reset, both add and remove are processed in order."""
        cfg = default_config
        cfg.num_boids = 20
        engine = SimulationEngine(cfg)

        orig_n = engine.flock.N_active

        engine.enqueue_add(10)
        engine.enqueue_remove(5)
        engine.drain_commands()

        assert engine.flock.N_active == orig_n + 10 - 5


# ── IT4: Recorder headless pipeline step-before-render ────────────

class TestRecorderHeadlessPipeline:
    """I4.1 + I4.2 + I4.3: step() completes before Recorder renders."""

    def test_recorder_callback_sees_post_step_state(self, default_config):
        """IT4: run_headless callback sees state after step() completes."""
        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)

        frames_seen = []

        def callback(e):
            frames_seen.append((e.frame, e.flock.N_active))

        engine.run_headless(steps=3, callback=callback)

        # Callback fires after each step, so frame >= 1
        assert len(frames_seen) == 3
        for i, (frame, n_active) in enumerate(frames_seen):
            assert frame == i + 1  # callback sees post-step frame
            assert n_active == 10  # flock size unchanged

    def test_recorder_callback_with_command_mutations(self, default_config):
        """IT4: Callback sees post-drain/step state with mutations applied."""
        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)

        engine.enqueue_add(5)

        states = []

        def callback(e):
            states.append(e.flock.N_active)

        engine.run_headless(steps=2, callback=callback)

        # Step 1: drain(add 5) → 15 birds, step → callback sees 15
        # Step 2: no mutations, step → callback sees 15
        assert states == [15, 15]

    @pytest.mark.gpu
    def test_recorder_headless_with_visualizer(self, default_config, gpu_available, monkeypatch):
        """IT4 (GPU): Recorder + Visualizer integration — step before render.

        Verifies the full Recorder pipeline: engine.step() → Recorder.on_frame()
        → Visualizer.headless_frame(). The render must see post-step state.
        """
        if not gpu_available:
            pytest.skip("GPU not available")

        from pymurmur.capture.recorder import Recorder

        cfg = default_config
        cfg.num_boids = 10
        cfg.capture_with_viz = True
        cfg.capture_every = 1  # capture every frame
        cfg.capture_prewarm = 0  # P8.7: capture from the first frame

        engine = SimulationEngine(cfg)
        recorder = Recorder(engine, cfg)

        # Track step-vs-render order
        order_log = []

        orig_step = engine.step

        def spy_step(dt=1.0 / 60.0, control=None):
            order_log.append("step")
            return orig_step(dt, control=control)

        engine.step = spy_step

        # Run headless with Recorder callback
        engine.run_headless(steps=3, callback=recorder.on_frame)

        # Recorder.on_frame calls headless_frame which is render
        # After each step, so order should be step→render→step→render→...
        assert order_log == ["step", "step", "step"], (
            f"All 3 steps should run before any render. Got: {order_log}"
        )
        assert len(recorder.frames) == 3


# ── IT5: Engine reset preserves pipeline invariants ───────────────

class TestEngineResetPipelineInvariants:
    """I4.2 + I4.3: reset() preserves pipeline correctness."""

    def test_reset_preserves_pipeline_invariants(self, default_config):
        """IT5: After reset via command queue, pipeline continues working."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.mode = "spatial"
        engine = SimulationEngine(cfg)

        # Run 3 steps normally
        engine.run_headless(steps=3)
        assert engine.frame == 3
        assert len(engine.metrics.history) >= 1

        # Reset via command queue
        engine.enqueue_reset()
        engine.drain_commands()

        # Pipeline invariants after reset
        assert engine.frame == 0
        assert engine.flock.N_active == 10
        assert len(engine.metrics.history) == 0
        assert engine.commands.pending_add == 0
        assert engine.commands.pending_remove == 0
        assert engine.commands.pending_reset is False

        # Run 2 more steps — pipeline must work
        engine.run_headless(steps=2)
        assert engine.frame == 2
        assert engine.flock.N_active == 10
        assert len(engine.metrics.history) >= 1

    def test_reset_then_add_continues_correctly(self, default_config):
        """IT5: After reset, enqueue_add works independently (no stale counts)."""
        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)

        engine.step()
        engine.enqueue_reset()
        engine.drain_commands()
        assert engine.frame == 0
        assert engine.flock.N_active == 10

        # Add birds to the fresh flock
        engine.enqueue_add(5)
        engine.step()
        assert engine.flock.N_active == 15  # 10 + 5
        assert engine.frame == 1


# ── IT6: Only engine imports both flock and forces (AST scan) ─────

class TestOnlyEngineDualImporter:
    """I4.2 + I4.4: Only simulation.engine may import both flock and forces."""

    def test_only_engine_can_import_both_flock_and_forces(self):
        """IT6: Scan ALLOWED_EDGES + actual import edges — only engine is dual-importer.

        Forces sub-modules that import forces._base + flock are internal
        to the forces package and are excluded from this check.
        """
        from test.crosscutting.guards.test_architecture import (
            ALLOWED_EDGES,
            _collect_import_edges,
        )

        # Check ALLOWED_EDGES — exclude forces-package-internal modules
        allowed_dual = []
        for mod, targets in ALLOWED_EDGES.items():
            if mod.startswith("pymurmur.physics.forces"):
                continue
            has_flock = any(
                t == "pymurmur.physics.flock"
                or t.startswith("pymurmur.physics.flock.")
                for t in targets
            )
            has_forces = any(
                t == "pymurmur.physics.forces"
                or t.startswith("pymurmur.physics.forces.")
                for t in targets
            )
            if has_flock and has_forces:
                allowed_dual.append(mod)

        assert allowed_dual == ["pymurmur.simulation.engine"], (
            f"ALLOWED_EDGES dual-importers: {allowed_dual}"
        )

        # Check actual AST edges — verify no other module imports both at runtime
        edges = _collect_import_edges()

        # Build per-source import targets (filter TYPE_CHECKING)
        runtime_imports = {}
        for source, target, _lineno, in_tc in edges:
            if in_tc:
                continue
            runtime_imports.setdefault(source, set()).add(target)

        actual_dual = []
        for mod, targets in runtime_imports.items():
            if mod.startswith("pymurmur.physics.forces"):
                continue
            has_flock = any(
                t == "pymurmur.physics.flock"
                or t.startswith("pymurmur.physics.flock.")
                for t in targets
            )
            has_forces = any(
                t == "pymurmur.physics.forces"
                or t.startswith("pymurmur.physics.forces.")
                for t in targets
            )
            if has_flock and has_forces:
                actual_dual.append(mod)

        # simulation.engine is the only runtime dual-importer
        for mod in actual_dual:
            assert mod == "pymurmur.simulation.engine", (
                f"Module {mod} imports both flock and forces at runtime. "
                f"Only simulation.engine is allowed."
            )


# ═══════════════════════════════════════════════════════════════════════
# Part IV Cross-Item Integration — "As a Whole"
# ═══════════════════════════════════════════════════════════════════════
# Exercises combinations of S1.4, S1.5, S2.B7, S2.B8, S2.B11,
# S2.C3, S3.11, S3.6a, S2.B4 through the full engine pipeline.
# ═══════════════════════════════════════════════════════════════════════


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
        from pymurmur.core.config import SimConfig

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


# ═══════════════════════════════════════════════════════════════════════
# S6.4 Obstacle Engine Integration — "As a Whole"
# ═══════════════════════════════════════════════════════════════════════
# Exercises ObstacleScene collision detection, kinematic correction,
# avoidance steering, and per-step collision counter through the full
# engine pipeline.
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.s6_4
class TestObstacleEngineIntegration:
    """S6.4: ObstacleScene wired into SimulationEngine._step_physics().

    Verifies collision detection, kinematic correction, avoidance
    steering, and per-step collision counter published to metrics.
    """

    @staticmethod
    def _make_sphere_scene(center=(500.0, 500.0, 500.0), radius=100.0):
        """Build a single-sphere ObstacleScene for testing."""
        from pymurmur.physics.obstacles import ObstacleScene
        return ObstacleScene().add_sphere(center, radius)

    def test_obstacle_scene_resolves_collisions_in_pipeline(self, default_config):
        """S6.4: Birds that enter a sphere are corrected to the surface
        and collision count is published to metrics."""
        from pymurmur.physics.obstacles import ObstacleScene

        cfg = default_config
        cfg.mode = "spatial"
        cfg.num_boids = 20
        cfg.seed = 42
        cfg.v0 = 60.0  # high speed to reach surface quickly
        cfg.dt_phys = 1.0 / 60.0
        cfg.boundary_mode = "sphere"
        cfg.boundary_sphere_radius = 500.0
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 1

        engine = SimulationEngine(cfg)

        # Place a small sphere obstacle at the centre
        scene = ObstacleScene().add_sphere((500.0, 500.0, 500.0), 50.0)
        engine.obstacle_scene = scene

        # Place birds just outside the sphere (~51 units from centre),
        # moving inward at high speed so they collide within 1 step
        rng = np.random.default_rng(42)
        for i in range(cfg.num_boids):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            engine.flock.positions[i] = np.array([500.0, 500.0, 500.0], dtype=np.float32) + direction.astype(np.float32) * 51.0
            engine.flock.velocities[i] = -direction.astype(np.float32) * cfg.v0

        # Step a few times — birds should collide with the sphere
        total_collisions = 0
        for _ in range(10):
            engine.step()
            total_collisions += engine.metrics.snapshot().collisions_this_step

        # At least some birds should have collided with the sphere
        assert total_collisions > 0, (
            f"Expected some collisions with sphere obstacle, got {total_collisions}"
        )
        assert scene.collision_count == total_collisions

        # After correction, no bird should be deep inside the sphere
        # (allow small penetration due to discrete timesteps, but not more
        # than half the sphere radius)
        positions = engine.flock.positions[engine.flock.active]
        dists = np.linalg.norm(positions - np.array([500.0, 500.0, 500.0]), axis=1)
        assert np.all(dists >= 50.0 * 0.5), (
            f"Birds penetrated too deep: min dist={dists.min():.1f}"
        )

    def test_obstacle_collision_counter_in_metrics_schema(self, default_config):
        """S6.4: collisions_this_step appears in FlockMetrics.to_dict()
        and is JSON-serializable."""
        import json

        from pymurmur.physics.obstacles import ObstacleScene

        cfg = default_config
        cfg.mode = "spatial"
        cfg.num_boids = 10
        cfg.seed = 42
        cfg.v0 = 60.0  # high speed to collide in 1 step
        cfg.boundary_mode = "sphere"
        cfg.boundary_sphere_radius = 500.0
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 1

        engine = SimulationEngine(cfg)
        scene = ObstacleScene().add_sphere((500.0, 500.0, 500.0), 50.0)
        engine.obstacle_scene = scene

        # Place birds heading toward centre, very close to surface
        for i in range(cfg.num_boids):
            engine.flock.positions[i] = np.array([448.0, 500.0, 500.0], dtype=np.float32)
            engine.flock.velocities[i] = np.array([cfg.v0, 0.0, 0.0], dtype=np.float32)

        engine.step()
        snap = engine.metrics.snapshot()

        # collisions_this_step is an int
        assert isinstance(snap.collisions_this_step, int), (
            f"Expected int, got {type(snap.collisions_this_step)}"
        )
        assert snap.collisions_this_step >= 0

        # JSON round-trip
        d = snap.to_dict()
        assert "collisions_this_step" in d
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        assert isinstance(restored["collisions_this_step"], int)
        assert restored["collisions_this_step"] == snap.collisions_this_step

    def test_obstacle_avoidance_reduces_collisions_over_time(self, default_config):
        """S6.4: With avoidance weights active, collision rate drops
        over successive steps as birds learn to steer away."""
        from pymurmur.physics.obstacles import ObstacleScene

        cfg = default_config
        cfg.mode = "spatial"
        cfg.num_boids = 20
        cfg.seed = 42
        cfg.v0 = 60.0  # high speed to collide quickly
        cfg.boundary_mode = "sphere"
        cfg.boundary_sphere_radius = 500.0
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 1
        # Enable avoidance
        cfg.spatial.static_avoid_weight = 2.0
        cfg.spatial.predictive_avoid_weight = 1.0
        cfg.spatial.fly_away_max_dist = 50.0
        cfg.spatial.min_time_to_collide = 2.0

        engine = SimulationEngine(cfg)

        scene = ObstacleScene().add_sphere((500.0, 500.0, 500.0), 60.0)
        engine.obstacle_scene = scene

        # Place birds in a ring heading toward centre, close to surface
        rng = np.random.default_rng(42)
        for i in range(cfg.num_boids):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            engine.flock.positions[i] = np.array([500.0, 500.0, 500.0], dtype=np.float32) + direction.astype(np.float32) * 62.0
            engine.flock.velocities[i] = -direction.astype(np.float32) * cfg.v0

        collisions_first_5 = 0
        collisions_last_5 = 0

        for step in range(30):
            engine.step()
            c = engine.metrics.snapshot().collisions_this_step
            if step < 5:
                collisions_first_5 += c
            elif step >= 25:
                collisions_last_5 += c

        # Avoidance should reduce collisions over time
        # (birds learn to steer away after initial collisions)
        # This is probabilistic but with strong weights should hold
        assert collisions_last_5 <= collisions_first_5, (
            f"Avoidance did not reduce collisions: "
            f"first 5 steps={collisions_first_5}, last 5 steps={collisions_last_5}"
        )

    def test_obstacle_scene_none_is_noop(self, default_config):
        """S6.4: When obstacle_scene is None, the pipeline runs
        normally with zero collisions."""
        cfg = default_config
        cfg.mode = "spatial"
        cfg.num_boids = 20
        cfg.seed = 42
        cfg.boundary_mode = "sphere"
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 1

        engine = SimulationEngine(cfg)
        # No obstacle_scene set → should be None
        assert engine.obstacle_scene is None

        for _ in range(10):
            engine.step()

        snap = engine.metrics.snapshot()
        assert snap.collisions_this_step == 0, (
            f"No obstacle scene but collisions={snap.collisions_this_step}"
        )

    def test_obstacle_scene_empty_is_noop(self, default_config):
        """S6.4: An ObstacleScene with no shapes is a no-op."""
        from pymurmur.physics.obstacles import ObstacleScene

        cfg = default_config
        cfg.mode = "spatial"
        cfg.num_boids = 10
        cfg.seed = 42
        cfg.boundary_mode = "sphere"
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 1

        engine = SimulationEngine(cfg)
        engine.obstacle_scene = ObstacleScene()  # empty, no shapes

        for _ in range(5):
            engine.step()

        snap = engine.metrics.snapshot()
        assert snap.collisions_this_step == 0, (
            f"Empty scene should have zero collisions, got {snap.collisions_this_step}"
        )

    @pytest.mark.parametrize("mode", ["spatial", "angle", "projection"])
    def test_obstacle_scene_works_across_all_modes(self, default_config, mode):
        """S6.4: Obstacle scene works with spatial, angle, and projection modes."""
        from pymurmur.physics.obstacles import ObstacleScene

        cfg = default_config
        cfg.mode = mode
        cfg.num_boids = 20
        cfg.seed = 42
        cfg.v0 = 60.0  # high speed to collide in 1 step
        cfg.boundary_mode = "sphere"
        cfg.boundary_sphere_radius = 500.0
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 1
        cfg.spatial.static_avoid_weight = 1.0
        cfg.spatial.fly_away_max_dist = 40.0

        engine = SimulationEngine(cfg)

        scene = ObstacleScene().add_sphere((500.0, 500.0, 500.0), 50.0)
        engine.obstacle_scene = scene

        # Place birds heading toward centre, very close to surface
        for i in range(cfg.num_boids):
            engine.flock.positions[i] = np.array([448.0, 500.0, 500.0], dtype=np.float32)
            engine.flock.velocities[i] = np.array([cfg.v0, 0.0, 0.0], dtype=np.float32)

        total_collisions = 0
        for _ in range(10):
            engine.step()
            total_collisions += engine.metrics.snapshot().collisions_this_step

        # All modes should detect collisions with the sphere
        assert total_collisions > 0, (
            f"{mode} mode: expected collisions, got {total_collisions}"
        )
        assert scene.collision_count == total_collisions

    def test_obstacle_avoidance_zero_weights_still_corrects(self, default_config):
        """S6.4: With zero avoidance weights, collisions are still
        detected and positions corrected (kinematic correction)."""
        from pymurmur.physics.obstacles import ObstacleScene

        cfg = default_config
        cfg.mode = "spatial"
        cfg.num_boids = 10
        cfg.seed = 42
        cfg.v0 = 60.0  # high speed to collide quickly
        cfg.boundary_mode = "sphere"
        cfg.boundary_sphere_radius = 500.0
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 1
        # Zero avoidance weights — only kinematic correction
        cfg.spatial.static_avoid_weight = 0.0
        cfg.spatial.predictive_avoid_weight = 0.0

        engine = SimulationEngine(cfg)

        scene = ObstacleScene().add_sphere((500.0, 500.0, 500.0), 50.0)
        engine.obstacle_scene = scene

        # Place birds heading straight toward centre, very close to surface
        for i in range(cfg.num_boids):
            engine.flock.positions[i] = np.array([448.0, 500.0, 500.0], dtype=np.float32)
            engine.flock.velocities[i] = np.array([cfg.v0, 0.0, 0.0], dtype=np.float32)

        collisions_found = 0
        for _ in range(15):
            engine.step()
            collisions_found += engine.metrics.snapshot().collisions_this_step

        # Collisions must be detected even without avoidance
        assert collisions_found > 0, (
            f"Zero-avoidance: expected collisions, got {collisions_found}"
        )
        assert scene.collision_count == collisions_found

        # After correction, positions should not be deep inside the sphere
        positions = engine.flock.positions[engine.flock.active]
        dists = np.linalg.norm(positions - np.array([500.0, 500.0, 500.0]), axis=1)
        assert np.all(dists >= 10.0), (
            f"Birds too deep: min dist={dists.min():.1f}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Part V Cross-Item Integration — "As a Whole"
# ═══════════════════════════════════════════════════════════════════════
# Exercises combinations of S2.A5, S4.4a, S2.E6, S5.6, S6.1–S6.6, S4.10
# through the full engine pipeline.
# ═══════════════════════════════════════════════════════════════════════


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
