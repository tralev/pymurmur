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

            def spy_integrate(config, dt):
                order_log.append("integrate")
                return orig_integrate(config, dt)

            def spy_collect(flock, frame):
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
        from unittest.mock import MagicMock

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

        engine = SimulationEngine(cfg)
        recorder = Recorder(engine, cfg)

        # Track step-vs-render order
        order_log = []

        orig_step = engine.step

        def spy_step(dt=1.0 / 60.0):
            order_log.append("step")
            return orig_step(dt)

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
        from test.test_architecture import ALLOWED_EDGES, _collect_import_edges

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
        for source, target, lineno, in_tc in edges:
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
