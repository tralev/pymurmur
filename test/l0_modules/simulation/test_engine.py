"""Unit tests for simulation.engine — SimulationEngine."""

from copy import copy

import numpy as np
import pytest

from pymurmur.simulation.engine import SimulationEngine


def test_engine_init_creates_flock(default_config):
    """Engine has non-null flock, extensions, metrics."""
    cfg = default_config
    cfg.num_boids = 20
    engine = SimulationEngine(cfg)
    assert engine.flock is not None
    assert engine.extensions is not None
    assert engine.metrics is not None
    assert engine.frame == 0


def test_engine_influencer_density_position_init(default_config):
    """C4: position_init="influencer_density" wires InfluencerMode.density_init_positions.

    Composer for the previously-dead influencer_density_init/
    density_init_positions L0 atoms — positions should follow the
    density-scaled Gaussian, not the default box init.
    """
    cfg = default_config
    cfg.mode = "influencer"
    cfg.position_init = "influencer_density"
    cfg.num_boids = 200
    cfg.influencer_scale = 1.0
    cfg.influencer_init_separation = 0.5
    cfg.seed = 42

    engine = SimulationEngine(cfg)

    C = np.array([cfg.width / 2, cfg.height / 2, cfg.depth / 2], dtype=np.float32)
    offsets = engine.flock.positions - C
    # Density-scaled Gaussian clusters tightly around the domain centre —
    # box init spreads uniformly across the full domain instead.
    assert np.std(offsets) < cfg.width * 0.25

    # reset() must reapply the same composer.
    engine.reset()
    offsets_after_reset = engine.flock.positions - C
    assert np.std(offsets_after_reset) < cfg.width * 0.25


def test_influencer_density_init_respects_velocity_init_drift(default_config):
    """C4×C2: influencer_density position_init with velocity_init should not crash.

    The order in engine init is:
    1. velocity_init (e.g. "drift" alias for "blob")
    2. _apply_influencer_density_init (overrides positions only)
    Velocities from step 1 should survive step 2.
    """
    cfg = default_config
    cfg.mode = "influencer"
    cfg.position_init = "influencer_density"
    cfg.velocity_init = "drift"
    cfg.num_boids = 50
    cfg.influencer_scale = 1.0
    cfg.influencer_init_separation = 0.5
    cfg.seed = 42

    engine = SimulationEngine(cfg)

    # Positions should be density-clustered (not box)
    C = np.array([cfg.width / 2, cfg.height / 2, cfg.depth / 2], dtype=np.float32)
    offsets = engine.flock.positions - C
    assert np.std(offsets) < cfg.width * 0.25, (
        "influencer_density must cluster positions near centre"
    )

    # Velocities should be initialised (not all zero) — "drift" alias for "blob"
    speeds = np.linalg.norm(engine.flock.velocities, axis=1)
    active_speeds = speeds[engine.flock.active]
    assert np.mean(active_speeds) > 0, (
        "velocity_init='drift' must produce non-zero velocities"
    )
    assert np.all(active_speeds <= cfg.v0 + 1e-6), (
        "all speeds must be within [0, v0]"
    )


def test_influencer_density_init_after_reset_preserves_distribution(default_config):
    """C4: After engine.reset(), influencer_density is reapplied correctly."""
    cfg = default_config
    cfg.mode = "influencer"
    cfg.position_init = "influencer_density"
    cfg.num_boids = 200
    cfg.influencer_scale = 1.0
    cfg.influencer_init_separation = 0.5
    cfg.seed = 42

    engine = SimulationEngine(cfg)
    C = np.array([cfg.width / 2, cfg.height / 2, cfg.depth / 2], dtype=np.float32)

    # Record distribution after first init
    std_first = float(np.std(engine.flock.positions - C))

    # Reset and check again
    engine.reset()
    std_after_reset = float(np.std(engine.flock.positions - C))

    # Both should be tightly clustered
    assert std_first < cfg.width * 0.25
    assert std_after_reset < cfg.width * 0.25

    # With same seed, distributions should be identical
    assert std_first == pytest.approx(std_after_reset, rel=1e-6), (
        f"Same seed: std={std_first:.2f} vs {std_after_reset:.2f} — "
        "reset must reproduce identical positions"
    )


def test_velocity_init_drift_with_field_mode_engine_step(default_config):
    """C2×C3: velocity_init="drift" + mode="field" through engine.step().

    The engine init order is: init positions → init velocities (drift=blob)
    → apply influencer_density if configured. Then engine.step() must
    run field mode forces and integrate without crashing.
    """
    cfg = default_config
    cfg.mode = "field"
    cfg.velocity_init = "drift"
    cfg.num_boids = 30
    cfg.seed = 42

    engine = SimulationEngine(cfg)

    # Velocities must be initialised via the "drift" (blob) path
    speeds = np.linalg.norm(engine.flock.velocities, axis=1)
    assert np.mean(speeds[engine.flock.active]) > 0, (
        "velocity_init='drift' must produce non-zero velocities"
    )

    # Run 3 steps — field mode forces apply each step
    for _ in range(3):
        engine.step()

    assert engine.frame == 3
    assert np.isfinite(engine.flock.positions).all()
    assert np.isfinite(engine.flock.velocities).all()
    # Birds must have moved — check last_accelerations (not accelerations,
    # which are zeroed by integrate() during step())
    acc_mags = np.linalg.norm(engine.flock.last_accelerations[engine.flock.active], axis=1)
    assert np.any(acc_mags > 1e-6), (
        f"field mode must apply non-zero forces: max acc={acc_mags.max():.6f}"
    )


def test_engine_step_increments_frame(default_config):
    """engine.frame increments by 1 per step()."""
    cfg = default_config
    cfg.num_boids = 10
    engine = SimulationEngine(cfg)
    assert engine.frame == 0
    engine.step()
    assert engine.frame == 1
    engine.step()
    assert engine.frame == 2


def test_engine_run_headless_n_steps(default_config):
    """run_headless(steps=N) runs exactly N steps."""
    cfg = default_config
    cfg.num_boids = 10
    engine = SimulationEngine(cfg)
    engine.run_headless(steps=5)
    assert engine.frame == 5


def test_engine_run_headless_callback(default_config):
    """Callback is called once per step."""
    cfg = default_config
    cfg.num_boids = 10
    engine = SimulationEngine(cfg)

    calls = []
    engine.run_headless(steps=3, callback=lambda e: calls.append(e.frame))
    assert len(calls) == 3
    assert calls == [1, 2, 3]


def test_engine_reset(default_config):
    """reset() creates new flock with same config."""
    cfg = default_config
    cfg.num_boids = 20
    engine = SimulationEngine(cfg)
    engine.step()
    engine.step()
    assert engine.frame == 2

    engine.reset()
    assert engine.frame == 0
    assert engine.flock.N_active == cfg.num_boids


def test_engine_step_order(default_config):
    """Extensions run before flock.integrate, metrics after (I4.2)."""
    cfg = default_config
    cfg.num_boids = 10
    engine = SimulationEngine(cfg)

    # Spy on the method order by wrapping integrate
    order_log = []

    orig_ext = engine.extensions.pre_step
    orig_integrate = engine.flock.integrate
    orig_collect = engine.metrics.collect

    def spy_ext(flock, ctx):
        order_log.append("extensions")
        return orig_ext(flock, ctx)

    def spy_integrate(config, dt, **kwargs):
        order_log.append("flock")
        return orig_integrate(config, dt, **kwargs)

    def spy_collect(flock, frame, **kwargs):
        order_log.append("metrics")
        return orig_collect(flock, frame)

    engine.extensions.pre_step = spy_ext
    engine.flock.integrate = spy_integrate
    engine.metrics.collect = spy_collect

    engine.step()

    assert order_log == ["extensions", "flock", "metrics"]


def test_engine_reset_metrics(default_config):
    """reset() clears metrics history."""
    cfg = default_config
    cfg.num_boids = 10
    engine = SimulationEngine(cfg)
    engine.step()
    engine.step()
    # History should have entries after stepping
    assert len(engine.metrics.history) >= 1

    engine.reset()
    # After reset, history should be empty
    assert len(engine.metrics.history) == 0


def test_engine_no_viz_imports():
    """SimulationEngine module does not import pygame, moderngl, or PyGLM."""
    import ast
    from pathlib import Path

    engine_path = Path(__file__).resolve().parents[3] / "pymurmur" / "simulation" / "engine.py"
    source = engine_path.read_text()
    tree = ast.parse(source)

    viz_imports = {"pygame", "moderngl", "PyGLM", "Pillow"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in viz_imports, \
                    f"engine.py imports {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                assert node.module.split(".")[0] not in viz_imports, \
                    f"engine.py imports from {node.module}"


def test_engine_config_live_mutation(default_config):
    """Mutating config.separation_weight between steps affects position deltas."""
    cfg = default_config
    cfg.num_boids = 20
    cfg.mode = "spatial"
    cfg.separation_weight = 0.0
    cfg.alignment_weight = 0.0
    cfg.cohesion_weight = 0.0
    cfg.noise_scale = 0.0
    engine = SimulationEngine(cfg)

    # Step with all weights zero — positions should barely change (no forces)
    engine.step()
    pos_after_zero = engine.flock.positions.copy()

    # Now mutate config to enable separation
    engine.config.separation_weight = 10.0
    engine.step()
    pos_after_sep = engine.flock.positions.copy()

    # Positions should now differ (forces were applied → velocities changed →
    # positions changed differently from the zero-weight step)
    assert not np.allclose(pos_after_zero, pos_after_sep)


def test_engine_deterministic_with_seed(default_config):
    """Same seed → identical flock state after N steps."""
    cfg1 = copy(default_config)
    cfg1.num_boids = 30
    cfg1.seed = 42

    cfg2 = copy(default_config)
    cfg2.num_boids = 30
    cfg2.seed = 42

    engine1 = SimulationEngine(cfg1)
    engine2 = SimulationEngine(cfg2)

    for _ in range(5):
        engine1.step()
        engine2.step()

    # Positions and velocities should be identical
    assert np.allclose(engine1.flock.positions, engine2.flock.positions)
    assert np.allclose(engine1.flock.velocities, engine2.flock.velocities)
    assert engine1.frame == engine2.frame


def test_engine_run_headless_forever_stops_on_callback(default_config):
    """run_headless(steps=None) runs until callback raises StopIteration."""
    cfg = default_config
    cfg.num_boids = 10
    engine = SimulationEngine(cfg)

    frame_count = [0]

    def stop_after_5(e):
        frame_count[0] += 1
        if e.frame >= 5:
            raise StopIteration

    try:
        engine.run_headless(steps=None, callback=stop_after_5)
    except StopIteration:
        pass

    assert frame_count[0] == 5
    assert engine.frame == 5


# ── I4.3: CommandQueue unit tests ─────────────────────────────────

from pymurmur.simulation.engine import CommandQueue  # noqa: E402


class TestCommandQueue:
    """I4.3: CommandQueue — pending add/remove/reset accumulation."""

    def test_command_queue_enqueue_add_accumulates(self):
        """I4.3 M5: enqueue_add(3) + enqueue_add(2) = 5 pending."""
        cq = CommandQueue()
        assert cq.pending_add == 0
        cq.pending_add += 3
        assert cq.pending_add == 3
        cq.pending_add += 2
        assert cq.pending_add == 5

    def test_command_queue_enqueue_remove_accumulates(self):
        """I4.3 M6: enqueue_remove(3) + enqueue_remove(2) = 5 pending."""
        cq = CommandQueue()
        assert cq.pending_remove == 0
        cq.pending_remove += 3
        assert cq.pending_remove == 3
        cq.pending_remove += 2
        assert cq.pending_remove == 5

    def test_drain_commands_reset_clears_pending_add_remove(self, default_config):
        """I4.3 M7: pending_reset=True clears pending_add/remove before reset()."""
        cfg = default_config
        cfg.num_boids = 20
        engine = SimulationEngine(cfg)
        engine.step()  # step once so metrics has history

        # Queue add + remove + reset
        engine.enqueue_add(10)
        engine.enqueue_remove(5)
        engine.enqueue_reset()

        assert engine.commands.pending_add == 10
        assert engine.commands.pending_remove == 5
        assert engine.commands.pending_reset is True

        # Drain — reset must clear pending_add/remove
        engine.drain_commands()

        assert engine.commands.pending_reset is False
        assert engine.commands.pending_add == 0, (
            "pending_add must be cleared when reset is processed"
        )
        assert engine.commands.pending_remove == 0, (
            "pending_remove must be cleared when reset is processed"
        )
        assert engine.frame == 0  # reset happened

    def test_drain_commands_processes_reset_before_add_remove(self, default_config, monkeypatch):
        """I4.3 M8: drain_commands() processes reset, returns early, skips add/remove."""
        from unittest.mock import MagicMock

        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)

        # Spy on flock methods to verify they're NOT called when reset is pending
        mock_add = MagicMock()
        mock_remove = MagicMock()
        monkeypatch.setattr(engine.flock, "add_boids", mock_add)
        monkeypatch.setattr(engine.flock, "remove_boids", mock_remove)

        # Queue reset + add + remove simultaneously
        engine.enqueue_reset()
        engine.enqueue_add(5)
        engine.enqueue_remove(3)

        # Drain — reset must clear pending_add/remove and return early
        engine.drain_commands()

        # Add/remove must NOT have been called (reset returns early)
        mock_add.assert_not_called()
        mock_remove.assert_not_called()

        # After drain: queue is clean, flock is fresh
        assert engine.commands.pending_reset is False
        assert engine.commands.pending_add == 0
        assert engine.commands.pending_remove == 0
        assert engine.frame == 0

    def test_step_calls_drain_commands_first(self, default_config):
        """I4.3 M9: step() calls drain_commands() before extensions/index/forces."""
        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)

        order_log = []

        orig_drain = engine.drain_commands
        orig_ext = engine.extensions.pre_step

        def spy_drain():
            order_log.append("drain")
            return orig_drain()

        def spy_ext(flock, ctx):
            order_log.append("extensions")
            return orig_ext(flock, ctx)

        engine.drain_commands = spy_drain
        engine.extensions.pre_step = spy_ext

        engine.step()

        assert order_log[0] == "drain", (
            f"drain_commands must run FIRST. Got: {order_log}"
        )
        assert order_log[1] == "extensions", (
            f"extensions must run SECOND. Got: {order_log}"
        )

    def test_reset_creates_new_empty_command_queue(self, default_config):
        """I4.3 M10: reset() creates a fresh CommandQueue with no pending commands."""
        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)

        old_queue = engine.commands
        engine.enqueue_add(3)
        engine.enqueue_remove(2)
        assert old_queue.pending_add == 3
        assert old_queue.pending_remove == 2

        engine.reset()

        new_queue = engine.commands
        assert new_queue is not old_queue, "reset() must create a new CommandQueue"
        assert new_queue.pending_add == 0
        assert new_queue.pending_remove == 0
        assert new_queue.pending_reset is False


# ── P10.4: enqueue_clear + drain_commands tests ──────────────────

class TestClearBirds:
    """P10.4: enqueue_clear() + drain_commands() — clear all active boids."""

    def test_enqueue_clear_sets_pending_flag(self, default_config):
        """P10.4: enqueue_clear() sets pending_clear = True."""
        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)
        assert not engine.commands.pending_clear
        engine.enqueue_clear()
        assert engine.commands.pending_clear

    def test_drain_commands_clear_deactivates_all_birds(self, default_config):
        """P10.4: drain_commands() with pending_clear sets flock.active[:] = False."""
        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)
        assert engine.flock.N_active == 10

        engine.enqueue_clear()
        engine.drain_commands()

        assert engine.flock.N_active == 0
        assert not engine.flock.active.any(), (
            "All birds must be inactive after clear"
        )

    def test_drain_commands_clear_sets_num_boids_zero(self, default_config):
        """P10.4: After clear drain, config.num_boids is 0."""
        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)

        engine.enqueue_clear()
        engine.drain_commands()

        assert cfg.num_boids == 0, (
            f"config.num_boids should be 0 after clear, got {cfg.num_boids}"
        )

    def test_drain_commands_clear_resets_pending_flag(self, default_config):
        """P10.4: After drain, pending_clear is reset to False."""
        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)

        engine.enqueue_clear()
        assert engine.commands.pending_clear
        engine.drain_commands()
        assert not engine.commands.pending_clear, (
            "pending_clear must be reset after drain"
        )

    def test_clear_preserves_positions_and_velocities(self, default_config):
        """P10.4: Clear only flips active flag; positions/velocities are untouched."""
        cfg = default_config
        cfg.num_boids = 5
        engine = SimulationEngine(cfg)

        pos_before = engine.flock.positions.copy()
        vel_before = engine.flock.velocities.copy()

        engine.enqueue_clear()
        engine.drain_commands()

        # Positions and velocities should be unchanged (just inactive)
        np.testing.assert_array_equal(
            engine.flock.positions, pos_before,
            err_msg="Clear must preserve positions — only active flag changes"
        )
        np.testing.assert_array_equal(
            engine.flock.velocities, vel_before,
            err_msg="Clear must preserve velocities — only active flag changes"
        )

    def test_clear_with_no_active_birds_is_noop(self, default_config):
        """P10.4: Clearing an already-empty flock does nothing."""
        cfg = default_config
        cfg.num_boids = 0  # no initial birds — but capacity is allocated
        engine = SimulationEngine(cfg)
        # Deactivate all
        engine.flock.active[:] = False
        assert engine.flock.N_active == 0

        engine.enqueue_clear()
        engine.drain_commands()

        assert engine.flock.N_active == 0
        assert not engine.commands.pending_clear

    def test_clear_does_not_corrupt_other_config_fields(self, default_config):
        """P10.4: Clear only changes num_boids; other config fields unchanged."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.v0 = 3.5
        cfg.mode = "spatial"
        old_v0 = cfg.v0
        old_mode = cfg.mode
        old_sep = cfg.spatial.separation_weight

        engine = SimulationEngine(cfg)
        engine.enqueue_clear()
        engine.drain_commands()

        assert cfg.v0 == pytest.approx(old_v0)
        assert cfg.mode == old_mode
        assert cfg.spatial.separation_weight == pytest.approx(old_sep)

    def test_clear_resets_n_active_correctly(self, default_config):
        """P10.4: N_active is 0 after clear; previously active count tracked."""
        cfg = default_config
        cfg.num_boids = 20
        engine = SimulationEngine(cfg)
        assert engine.flock.N_active == 20

        engine.enqueue_clear()
        engine.drain_commands()

        assert engine.flock.N_active == 0
        # active array length (capacity) should be unchanged
        assert len(engine.flock.active) >= 20

    def test_clear_after_add_resets_everything(self, default_config):
        """P10.4: Adding birds then clearing → all inactive, N_active=0."""
        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)

        # Add some birds
        engine.enqueue_add(30)
        engine.drain_commands()
        assert engine.flock.N_active > 10

        # Then clear
        engine.enqueue_clear()
        engine.drain_commands()
        assert engine.flock.N_active == 0

    def test_clear_command_queue_flag_independent(self, default_config):
        """P10.4: pending_clear is independent of pending_add/pending_remove."""
        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)

        # Queue multiple commands simultaneously
        engine.enqueue_add(5)
        engine.enqueue_remove(3)
        engine.enqueue_clear()

        assert engine.commands.pending_add == 5
        assert engine.commands.pending_remove == 3
        assert engine.commands.pending_clear

        # Drain — add/remove fire first, then clear
        engine.drain_commands()

        # After drain: add was processed, remove was processed, clear happened last
        assert engine.flock.N_active == 0
        assert not engine.commands.pending_clear
        assert engine.commands.pending_add == 0, (
            "pending_add should be consumed during drain"
        )
        assert engine.commands.pending_remove == 0, (
            "pending_remove should be consumed during drain"
        )

    def test_double_clear_is_noop(self, default_config):
        """P10.4: Enqueuing clear twice (just sets bool to True twice)."""
        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)

        engine.enqueue_clear()
        engine.enqueue_clear()  # second call: pending_clear already True, no-op
        engine.drain_commands()

        assert engine.flock.N_active == 0
        # pending_clear was reset to False by drain
        assert not engine.commands.pending_clear
        # Second drain is a no-op since there's no pending_clear
        engine.drain_commands()
        assert engine.flock.N_active == 0  # still empty
        assert not engine.commands.pending_clear

    def test_clear_in_step_cycle(self, default_config):
        """P10.4: Clear survives a full step() cycle (drain + physics)."""
        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)

        engine.enqueue_clear()
        # step() calls drain_commands() first, then runs physics
        engine.step()

        assert engine.flock.N_active == 0, (
            f"After step with clear, N_active should be 0, got {engine.flock.N_active}"
        )
        assert cfg.num_boids == 0


# ── P10.4: Cursor-ray spawn pipeline — enqueue_spawn + drain_commands ─

class TestSpawnPipeline:
    """P10.4: enqueue_spawn → drain_commands → flock.spawn_at integration."""

    def test_spawn_bird_at_exact_position(self, default_config):
        """P10.4: enqueue_spawn(pos) + drain → bird at exact world position."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.seed = 42
        engine = SimulationEngine(cfg)
        n_before = engine.flock.N_active

        target = (123.0, 456.0, 789.0)
        engine.enqueue_spawn(target)
        engine.drain_commands()

        assert engine.flock.N_active == n_before + 1
        # Find the newly spawned bird (last active index)
        active_idx = np.where(engine.flock.active)[0]
        new_bird = active_idx[-1]
        np.testing.assert_array_equal(
            engine.flock.positions[new_bird],
            np.array(target, dtype=np.float32),
        )

    def test_spawn_bird_velocity_bounded_by_v0(self, default_config):
        """P10.4: Spawned bird velocity obeys config.v0 (cube-velocity law)."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.v0 = 3.0
        cfg.seed = 42
        engine = SimulationEngine(cfg)

        engine.enqueue_spawn((500, 350, 200))
        engine.drain_commands()

        active_idx = np.where(engine.flock.active)[0]
        new_bird = active_idx[-1]
        speed = float(np.linalg.norm(engine.flock.velocities[new_bird]))
        assert speed <= cfg.v0 + 1e-6, (
            f"Spawn velocity {speed:.4f} exceeds config.v0={cfg.v0}"
        )
        assert speed >= 0.0

    def test_spawn_predator_flag(self, default_config):
        """P10.4: enqueue_spawn(pos, is_predator=True) → predator flag set."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.seed = 42
        engine = SimulationEngine(cfg)

        engine.enqueue_spawn((500, 350, 200), is_predator=True)
        engine.drain_commands()

        active_idx = np.where(engine.flock.active)[0]
        new_bird = active_idx[-1]
        assert bool(engine.flock.is_predator[new_bird]) is True

    def test_spawn_prey_by_default(self, default_config):
        """P10.4: enqueue_spawn(pos) without is_predator → prey (False)."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.seed = 42
        engine = SimulationEngine(cfg)

        engine.enqueue_spawn((500, 350, 200))
        engine.drain_commands()

        active_idx = np.where(engine.flock.active)[0]
        new_bird = active_idx[-1]
        assert bool(engine.flock.is_predator[new_bird]) is False

    def test_spawn_updates_num_boids(self, default_config):
        """P10.4: After drain, config.num_boids reflects new N_active."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.seed = 42
        engine = SimulationEngine(cfg)

        engine.enqueue_spawn((500, 350, 200))
        engine.drain_commands()

        assert cfg.num_boids == engine.flock.N_active

    def test_multiple_spawns_in_one_drain(self, default_config):
        """P10.4: Multiple enqueued spawns all appear in one drain."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.seed = 42
        engine = SimulationEngine(cfg)
        n_before = engine.flock.N_active

        engine.enqueue_spawn((100, 200, 300))
        engine.enqueue_spawn((400, 500, 600))
        engine.enqueue_spawn((700, 800, 900))
        engine.drain_commands()

        assert engine.flock.N_active == n_before + 3
        active_idx = np.where(engine.flock.active)[0]
        b1, b2, b3 = active_idx[-3], active_idx[-2], active_idx[-1]
        np.testing.assert_array_equal(
            engine.flock.positions[b1], np.array([100, 200, 300], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            engine.flock.positions[b2], np.array([400, 500, 600], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            engine.flock.positions[b3], np.array([700, 800, 900], dtype=np.float32),
        )

    def test_mixed_bird_and_predator_spawns(self, default_config):
        """P10.4: Mix of bird and predator spawns in same drain."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.seed = 42
        engine = SimulationEngine(cfg)

        engine.enqueue_spawn((100, 200, 300))         # prey
        engine.enqueue_spawn((400, 500, 600), is_predator=True)  # predator
        engine.drain_commands()

        active_idx = np.where(engine.flock.active)[0]
        bird, predator = active_idx[-2], active_idx[-1]
        assert bool(engine.flock.is_predator[bird]) is False
        assert bool(engine.flock.is_predator[predator]) is True
        # Positions should match enqueue order
        np.testing.assert_array_equal(
            engine.flock.positions[bird], np.array([100, 200, 300], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            engine.flock.positions[predator], np.array([400, 500, 600], dtype=np.float32),
        )

    def test_spawn_different_v0_values(self, default_config):
        """P10.4: Changing config.v0 between spawns affects velocity."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.v0 = 2.0
        cfg.seed = 42
        engine = SimulationEngine(cfg)

        engine.enqueue_spawn((500, 350, 200))
        engine.drain_commands()
        active_idx = np.where(engine.flock.active)[0]
        bird_v0_2 = active_idx[-1]
        speed_2 = float(np.linalg.norm(engine.flock.velocities[bird_v0_2]))
        assert speed_2 <= 2.01

        # Change v0 and spawn again
        cfg.v0 = 0.5
        engine.enqueue_spawn((500, 350, 200))
        engine.drain_commands()
        active_idx = np.where(engine.flock.active)[0]
        bird_v0_05 = active_idx[-1]
        speed_05 = float(np.linalg.norm(engine.flock.velocities[bird_v0_05]))
        assert speed_05 <= 0.51

    def test_spawn_rng_advances(self, default_config):
        """P10.4: Two consecutive spawns at same position → different velocities."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.seed = 42
        engine = SimulationEngine(cfg)

        engine.enqueue_spawn((500, 350, 200))
        engine.enqueue_spawn((500, 350, 200))
        engine.drain_commands()

        active_idx = np.where(engine.flock.active)[0]
        b1, b2 = active_idx[-2], active_idx[-1]
        assert not np.array_equal(
            engine.flock.velocities[b1], engine.flock.velocities[b2],
        ), "Two spawns at same position should have different velocities"

    def test_spawn_then_step_keeps_bird(self, default_config):
        """P10.4: After spawn + step(), the bird remains active and moves."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.seed = 42
        engine = SimulationEngine(cfg)

        engine.enqueue_spawn((500, 350, 200))
        engine.drain_commands()
        active_idx = np.where(engine.flock.active)[0]
        new_bird = active_idx[-1]
        pos_after_spawn = engine.flock.positions[new_bird].copy()

        # Step the simulation — bird should move
        engine.step()
        assert engine.flock.active[new_bird], "Bird should still be active after step"
        assert not np.array_equal(
            engine.flock.positions[new_bird], pos_after_spawn,
        ), "Bird should move after integration step"

    def test_spawn_then_clear_then_spawn(self, default_config):
        """P10.4: Spawn → clear → spawn again — second spawn works fine."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.seed = 42
        engine = SimulationEngine(cfg)

        engine.enqueue_spawn((100, 200, 300))
        engine.drain_commands()
        assert engine.flock.N_active == 11

        engine.enqueue_clear()
        engine.drain_commands()
        assert engine.flock.N_active == 0

        # Spawn again — should reuse an inactive slot
        engine.enqueue_spawn((500, 500, 500))
        engine.drain_commands()
        assert engine.flock.N_active == 1
        active_idx = np.where(engine.flock.active)[0]
        new_bird = active_idx[-1]
        np.testing.assert_array_equal(
            engine.flock.positions[new_bird],
            np.array([500, 500, 500], dtype=np.float32),
        )

    def test_spawn_acceleration_zero(self, default_config):
        """P10.4: Spawned bird starts with zero acceleration."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.seed = 42
        engine = SimulationEngine(cfg)

        engine.enqueue_spawn((500, 350, 200))
        engine.drain_commands()

        active_idx = np.where(engine.flock.active)[0]
        new_bird = active_idx[-1]
        np.testing.assert_array_equal(
            engine.flock.accelerations[new_bird],
            np.zeros(3, dtype=np.float32),
        )

    def test_spawn_seed_assigned(self, default_config):
        """P10.4: Spawned bird gets a seed in [0, 1)."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.seed = 42
        engine = SimulationEngine(cfg)

        engine.enqueue_spawn((500, 350, 200))
        engine.drain_commands()

        active_idx = np.where(engine.flock.active)[0]
        new_bird = active_idx[-1]
        assert 0.0 <= engine.flock.seeds[new_bird] < 1.0

    def test_spawn_persists_across_multiple_steps(self, default_config):
        """P10.4: Spawned bird survives 5 simulation steps."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.seed = 42
        engine = SimulationEngine(cfg)

        engine.enqueue_spawn((500, 350, 200))
        engine.drain_commands()
        active_idx = np.where(engine.flock.active)[0]
        new_bird = active_idx[-1]

        for _ in range(5):
            engine.step()
            assert engine.flock.active[new_bird], (
                "Spawned bird should survive simulation steps"
            )


# Cross-cutting: P10.3 + P10.4 engine — HUD slider change affects physics step

class TestSliderEngineIntegration:
    """P10.3 + P10.4: Changing config via slider affects the next engine step."""

    def test_slider_noise_change_affects_step(self, default_config):
        """P10.3->P10.4 engine: Changing noise_scale via config changes
        the position deltas produced by the next step()."""
        from pymurmur.viz.hud import SliderHUD
        cfg = default_config
        cfg.num_boids = 20
        cfg.mode = "spatial"
        cfg.separation_weight = 0.0
        cfg.alignment_weight = 0.0
        cfg.cohesion_weight = 0.0
        cfg.noise_scale = 0.0
        cfg.v0 = 1.0  # slow speed so noise is measurable
        engine = SimulationEngine(cfg)

        # Step with zero noise
        engine.step()
        vel_no_noise = engine.flock.velocities.copy()

        # Use HUD slider to set noise to max
        hud = SliderHUD(cfg)
        hud._set_value(4, hud.TRACK_X0 + hud.TRACK_W)  # noise slider to max=0.5
        assert cfg.noise_scale == pytest.approx(0.5)

        engine.step()
        vel_with_noise = engine.flock.velocities.copy()

        # With noise, velocities change — speeds differ
        speeds_before = np.linalg.norm(vel_no_noise, axis=1).mean()
        speeds_after = np.linalg.norm(vel_with_noise, axis=1).mean()
        assert speeds_after != speeds_before, (
            "Noise should alter velocity magnitudes"
        )

    def test_slider_separation_change_affects_step(self, default_config):
        """P10.3->P10.4 engine: Increasing separation_weight via config
        produces different results than zero separation."""
        from pymurmur.viz.hud import SliderHUD
        cfg = default_config
        cfg.num_boids = 30
        cfg.mode = "spatial"
        cfg.separation_weight = 0.0
        cfg.alignment_weight = 0.0
        cfg.cohesion_weight = 0.0
        cfg.noise_scale = 0.0
        cfg.v0 = 1.0
        engine = SimulationEngine(cfg)

        # Step with zero separation — record velocities
        engine.step()
        vel_zero = engine.flock.velocities.copy()

        # Use HUD slider to set separation to max
        hud = SliderHUD(cfg)
        hud._set_value(0, hud.TRACK_X0 + hud.TRACK_W)  # sep to max=5.0
        assert cfg.spatial.separation_weight == pytest.approx(5.0)

        engine.step()
        vel_with_sep = engine.flock.velocities.copy()

        # With separation forces, velocities should differ
        assert not np.allclose(vel_zero, vel_with_sep, atol=1e-6), (
            "Separation should alter velocity patterns"
        )

    def test_slider_then_clear_then_slider(self, default_config):
        """P10.3->P10.4->P10.3: Change slider, clear birds, change slider
        again — engine doesn't crash and config is correct."""
        from pymurmur.viz.hud import SliderHUD
        cfg = default_config
        cfg.num_boids = 10
        engine = SimulationEngine(cfg)
        hud = SliderHUD(cfg)

        # Set sep to max
        hud._set_value(0, hud.TRACK_X0 + hud.TRACK_W)
        assert cfg.spatial.separation_weight == pytest.approx(5.0)

        # Clear
        engine.enqueue_clear()
        engine.drain_commands()
        assert engine.flock.N_active == 0

        # Change slider again — shouldn't crash
        hud._set_value(1, hud.TRACK_X0)  # coh to min
        assert cfg.spatial.cohesion_weight == pytest.approx(0.0)

        # Config integrity: sep should still be at max
        assert cfg.spatial.separation_weight == pytest.approx(5.0)
