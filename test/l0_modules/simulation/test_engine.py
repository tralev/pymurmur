"""Unit tests for simulation.engine — SimulationEngine init/step/reset/determinism, I4.3 CommandQueue unit tests.

Split out of test_engine.py (file-size split).
"""

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


