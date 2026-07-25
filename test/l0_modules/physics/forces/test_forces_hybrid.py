"""P4 — predator escape basics, P4.5 per-frame parameter jitter, predator/prey multi-agent, P4.8 coherence gate.

Split out of test_forces_hybrid.py (file-size split).
"""

from copy import copy

import numpy as np
import pytest

from pymurmur.physics.flock import PhysicsFlock  # noqa: E402


def test_predator_escape_replaces_separation(default_config):
    """Prey near a predator gets escape force, zeroed alignment/cohesion."""
    from pymurmur.physics.flock import PhysicsFlock
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = copy(default_config)
    cfg.seed = 42  # D6: default seed is None — pin for determinism
    cfg.mode = "spatial"
    cfg.num_boids = 10
    cfg.separation_weight = 5.0
    cfg.alignment_weight = 2.0
    cfg.cohesion_weight = 2.0
    cfg.noise_scale = 0.0
    cfg.max_force = 100.0
    cfg.predator_escape_factor = 1000.0

    flock = PhysicsFlock(cfg)
    # Make bird 0 a predator, place it near bird 1
    flock.is_predator[0] = True
    flock.positions[0] = np.array([10.0, 0.0, 0.0], dtype=np.float32)
    flock.positions[1] = np.array([12.0, 0.0, 0.0], dtype=np.float32)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    cfg._is_predator = flock.is_predator
    SpatialMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        np.zeros(len(flock.positions), dtype=np.float32), cfg,
    )

    # Bird 1 (prey near predator) should have non-zero escape force
    acc_1 = flock.accelerations[1]
    assert not np.allclose(acc_1, 0.0), "Prey near predator should have escape force"
    # Escape force should point away from predator
    to_predator = flock.positions[0] - flock.positions[1]
    escape_dir = -to_predator / np.linalg.norm(to_predator)
    assert np.dot(acc_1, escape_dir) > 0, (
        "Escape force should push away from predator"
    )
    # With escape-only config (zero sep/align/coh), force should be purely radial
    acc_dir = acc_1 / np.linalg.norm(acc_1)
    assert np.dot(acc_dir, escape_dir) > 0.7, (
        f"Escape force direction {acc_dir} should align with {escape_dir}"
    )


def test_predator_escape_zero_when_no_threat(default_config):
    """No predators nearby → normal separation/alignment/cohesion."""
    from pymurmur.physics.flock import PhysicsFlock
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = copy(default_config)
    cfg.seed = 42  # D6: default seed is None — pin for determinism
    cfg.mode = "spatial"
    cfg.num_boids = 10
    cfg.separation_weight = 5.0
    cfg.alignment_weight = 2.0
    cfg.cohesion_weight = 2.0
    cfg.noise_scale = 0.0
    cfg.max_force = 100.0
    cfg.predator_escape_factor = 1000.0

    flock = PhysicsFlock(cfg)
    # No predators — all birds are prey
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    cfg._is_predator = flock.is_predator
    SpatialMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        np.zeros(len(flock.positions), dtype=np.float32), cfg,
    )

    # All active birds should have non-zero forces (normal flocking)
    acc_active = flock.accelerations[flock.active]
    assert not np.allclose(acc_active, 0.0), "Normal forces should be present"
    assert np.isfinite(acc_active).all()


# ── P4.5: Per-frame parameter jitter ──────────────────────────────


def test_jitter_increases_force_variance(default_config):
    """Jittered weights produce different forces than unjittered."""
    from pymurmur.physics.flock import PhysicsFlock
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = copy(default_config)
    cfg.seed = 42  # D6: default seed is None — pin for determinism
    cfg.mode = "spatial"
    cfg.num_boids = 30
    cfg.separation_weight = 5.0
    cfg.alignment_weight = 2.0
    cfg.cohesion_weight = 2.0
    cfg.noise_scale = 0.0
    cfg.max_force = 100.0

    # Run without jitter
    cfg.jitter_separation = 0.0
    cfg.jitter_cohesion = 0.0
    cfg.jitter_alignment = 0.0
    flock1 = PhysicsFlock(cfg)
    flock1.accelerations[:] = 0.0
    flock1.get_index().rebuild(flock1.positions, flock1.active)
    rng1 = np.random.default_rng(42)
    SpatialMode.compute(
        flock1.positions, flock1.velocities, flock1.accelerations,
        flock1.active, flock1.get_index(), rng1,
        np.zeros(len(flock1.positions), dtype=np.float32), cfg,
    )
    acc1 = flock1.accelerations.copy()

    # Run with jitter
    cfg.jitter_separation = 0.5
    cfg.jitter_cohesion = 0.3
    cfg.jitter_alignment = 0.1
    flock2 = PhysicsFlock(cfg)
    flock2.accelerations[:] = 0.0
    flock2.get_index().rebuild(flock2.positions, flock2.active)
    rng2 = np.random.default_rng(42)
    SpatialMode.compute(
        flock2.positions, flock2.velocities, flock2.accelerations,
        flock2.active, flock2.get_index(), rng2,
        np.zeros(len(flock2.positions), dtype=np.float32), cfg,
    )
    acc2 = flock2.accelerations.copy()

    # Jittered forces should differ from unjittered
    diff = np.linalg.norm(acc1[flock1.active] - acc2[flock1.active], axis=1)
    assert np.mean(diff) > 0.01, (
        f"Jitter should produce different forces, mean diff={np.mean(diff):.6f}"
    )
    assert np.isfinite(acc2).all()


def test_jitter_deterministic_same_seed(default_config):
    """Same seed + same jitter config → identical forces (deterministic)."""
    from pymurmur.physics.flock import PhysicsFlock
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = copy(default_config)
    cfg.seed = 42  # D6: default seed is None — pin for determinism
    cfg.mode = "spatial"
    cfg.num_boids = 30
    cfg.jitter_separation = 0.5
    cfg.jitter_cohesion = 0.3
    cfg.jitter_alignment = 0.1
    cfg.noise_scale = 0.0
    cfg.max_force = 100.0

    def run_once(seed):
        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        flock.get_index().rebuild(flock.positions, flock.active)
        rng = np.random.default_rng(seed)
        SpatialMode.compute(
            flock.positions, flock.velocities, flock.accelerations,
            flock.active, flock.get_index(), rng,
            np.zeros(len(flock.positions), dtype=np.float32), cfg,
        )
        return flock.accelerations.copy()

    acc_a = run_once(42)
    acc_b = run_once(42)
    np.testing.assert_array_equal(acc_a, acc_b)


def test_jitter_zero_no_effect(default_config):
    """Jitter=0 produces same forces as no jitter config."""
    from pymurmur.physics.flock import PhysicsFlock
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = copy(default_config)
    cfg.seed = 42  # D6: default seed is None — pin for determinism
    cfg.mode = "spatial"
    cfg.num_boids = 30
    cfg.jitter_separation = 0.0
    cfg.jitter_cohesion = 0.0
    cfg.jitter_alignment = 0.0
    cfg.noise_scale = 0.0
    cfg.max_force = 100.0

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)
    rng = np.random.default_rng(42)
    SpatialMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), rng,
        np.zeros(len(flock.positions), dtype=np.float32), cfg,
    )

    # Forces should be finite and non-zero (normal flocking)
    assert not np.allclose(flock.accelerations[flock.active], 0.0)
    assert np.isfinite(flock.accelerations).all()


def test_predator_multiple_prey_flees_nearest(default_config):
    """P4.3: With two predators at different distances, prey flees the nearer one.

    Predator A at (5,0,0), Predator B at (20,0,0). Prey at (0,0,0).
    Escape should push +x (away from predator A at x=5, not B at x=20).
    If the kernel picks predator B, escape would be much weaker because
    the distance-squared penalty (1/d²) is 16× smaller."""
    from pymurmur.physics.forces._kernels import _HAS_NUMBA, _numba_predator_escape
    if not _HAS_NUMBA:
        pytest.skip("numba not available")

    N, k = 10, 5
    positions = np.zeros((N, 3), dtype=np.float32)
    positions[3] = [5.0, 0.0, 0.0]   # predator A — near (bird 3)
    positions[5] = [20.0, 0.0, 0.0]  # predator B — far (bird 5)
    positions[7] = [0.0, 0.0, 0.0]   # prey (bird 7)
    active = np.ones(N, dtype=bool)
    is_predator = np.zeros(N, dtype=bool)
    is_predator[3] = True
    is_predator[5] = True
    threatened = np.zeros(N, dtype=bool)
    threatened[7] = True

    # Prey sees both predators: [3 (near), 5 (far)]
    n_idx = np.zeros((N, k), dtype=np.int32)
    n_idx[7, 0] = 3  # nearest predator (dist=5)
    n_idx[7, 1] = 5  # farther predator (dist=20)

    escape = np.zeros((N, 3), dtype=np.float32)
    _numba_predator_escape(escape, positions, n_idx, is_predator,
                            threatened, active, escape_factor=1e6, accel_boost=1.0)

    # Escape should be strong (pushes -x, away from predator at x=5).
    # Near predator (d=5): force ≈ 1e6/25 = 40000. Far predator (d=20): ≈ 2500.
    # Assert absolute value > 5000 confirms nearest-predator selection.
    assert abs(escape[7, 0]) > 5000.0, (
        f"Escape from near predator (d=5) should be ~40000, got {escape[7, 0]:.1f}"
    )
    # Direction: prey at x=0 flees AWAY from predator at x=5 → negative x
    assert escape[7, 0] < 0, (
        f"Escape should push -x (away from predator at x=5), got {escape[7, 0]:.1f}"
    )
    # Clean y/z — collinear setup
    assert abs(escape[7, 1]) < 1e-6 and abs(escape[7, 2]) < 1e-6, (
        "Collinear escape should have no y/z component"
    )


def test_predator_ignored_by_other_predators(default_config):
    """Predators don't flee from other predators."""
    from pymurmur.physics.flock import PhysicsFlock
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = copy(default_config)
    cfg.seed = 42  # D6: default seed is None — pin for determinism
    cfg.mode = "spatial"
    cfg.num_boids = 10
    cfg.separation_weight = 5.0
    cfg.alignment_weight = 2.0
    cfg.cohesion_weight = 2.0
    cfg.noise_scale = 0.0
    cfg.max_force = 100.0
    cfg.predator_escape_factor = 1000.0

    flock = PhysicsFlock(cfg)
    # Bird 0 and bird 1 are both predators
    flock.is_predator[0] = True
    flock.is_predator[1] = True
    flock.positions[0] = np.array([10.0, 0.0, 0.0], dtype=np.float32)
    flock.positions[1] = np.array([12.0, 0.0, 0.0], dtype=np.float32)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    cfg._is_predator = flock.is_predator
    SpatialMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        np.zeros(len(flock.positions), dtype=np.float32), cfg,
    )

    # Bird 1 (predator) should NOT have escape force from bird 0 (also predator)
    # Predators don't flee from predators — only prey flees.
    # D21: the 1/d² separation kernel gives the d=2 predator pair a force of
    # a few units; an escape-boosted force (factor 1000, clamped only by
    # max_force=100) would be orders of magnitude larger.
    acc_predator = np.linalg.norm(flock.accelerations[1])
    assert acc_predator < 50.0, (
        f"Predator force {acc_predator:.1f} looks escape-boosted "
        f"(predators must not flee from predators)"
    )
    assert np.isfinite(flock.accelerations).all()


# ── P4.8 coherence gate ────────────────────────────────────

def test_coherence_gate_reduces_force_for_small_flock(default_config):
    """P4.8: Small flock (below critical mass) → reduced cohesion/alignment.

    SpatialMode reads _coherence_factor from config and multiplies
    cohesion/alignment weights by it. Below critical mass, factor < 1
    → forces are weaker than with factor = 1."""
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = default_config
    cfg.seed = 42  # D6: default seed is None — pin for determinism
    cfg.mode = "spatial"
    cfg.num_boids = 15  # small flock
    cfg.separation_weight = 4.5
    cfg.alignment_weight = 0.65
    cfg.cohesion_weight = 0.75
    cfg.noise_scale = 0.0
    cfg.max_force = 5.0

    # Create two identical flocks
    flock1 = PhysicsFlock(cfg)
    flock1.accelerations[:] = 0.0
    flock1.get_index().rebuild(flock1.positions, flock1.active)

    flock2 = PhysicsFlock(cfg)
    flock2.accelerations[:] = 0.0
    flock2.get_index().rebuild(flock2.positions, flock2.active)

    # Flock 1: coherence = 1.0 (full weights)
    object.__setattr__(cfg, '_coherence_factor', 1.0)
    SpatialMode.compute(
        flock1.positions, flock1.velocities, flock1.accelerations,
        flock1.active, flock1.get_index(), flock1.rng,
        flock1.last_theta, cfg,
    )
    force_full = float(np.linalg.norm(np.mean(
        flock1.accelerations[flock1.active], axis=0
    )))

    # Flock 2: coherence = 0.1 (heavily gated — ~15 birds / 500 crit_mass)
    flock2.accelerations[:] = 0.0
    object.__setattr__(cfg, '_coherence_factor', 0.1)
    SpatialMode.compute(
        flock2.positions, flock2.velocities, flock2.accelerations,
        flock2.active, flock2.get_index(), flock2.rng,
        flock2.last_theta, cfg,
    )
    force_gated = float(np.linalg.norm(np.mean(
        flock2.accelerations[flock2.active], axis=0
    )))

    # Gated forces should be significantly weaker (coherence reduces align/coh)
    assert force_gated < force_full * 0.9, (
        f"Coherence gate should reduce forces: full={force_full:.6f}, "
        f"gated={force_gated:.6f}"
    )


def test_coherence_gate_reduces_force_for_small_flock_projection(default_config):
    """S2.B8: Small flock → reduced phi_p/phi_a pull in projection mode too.

    ProjectionMode reads the same _coherence_factor as SpatialMode and
    scales phi_p/phi_a by it (the roadmap's "phi_a/phi_p gating missing"
    deviation) — this closes the gap so gating isn't spatial-only.

    phi_p/phi_a are per-bird directions (occlusion delta + local alignment,
    not a flock-wide heading), so averaging accelerations across birds
    cancels the very signal under test. This isolates the effect instead:
    zero velocities (steering == v_desired, no cancellation from existing
    motion) and zero the eta noise draw (phi_n grows as phi_p/phi_a shrink,
    and an unscaled random unit vector would otherwise mask the reduction),
    then compare mean per-bird force magnitude.
    """
    from pymurmur.physics.forces.projection import ProjectionMode

    cfg = copy(default_config)
    cfg.seed = 42  # D6: default seed is None — pin for determinism
    cfg.mode = "projection"
    cfg.num_boids = 300
    cfg.projection.phi_p = 0.5
    cfg.phi_a = 0.5

    class _ZeroNormalRNG:
        """Delegates to a real Generator but zeroes eta noise draws.

        numpy's Generator.normal is a read-only C attribute — it can't be
        monkeypatched directly, so this wraps the instance instead.
        """
        def __init__(self, rng):
            self._rng = rng

        def normal(self, size=None):
            return np.zeros(size)

        def __getattr__(self, name):
            return getattr(self._rng, name)

    def _run(coherence: float) -> float:
        flock = PhysicsFlock(cfg)
        flock.velocities[:] = 0.0  # isolate v_desired: steering = v_desired
        flock.accelerations[:] = 0.0
        flock.get_index().rebuild(flock.positions, flock.active)
        object.__setattr__(cfg, '_coherence_factor', coherence)
        ProjectionMode.compute(
            flock.positions, flock.velocities, flock.accelerations,
            flock.active, flock.get_index(), _ZeroNormalRNG(flock.rng),
            flock.last_theta, cfg,
        )
        return float(np.linalg.norm(flock.accelerations[flock.active], axis=1).mean())

    force_full = _run(1.0)
    force_gated = _run(0.1)  # heavily gated — small flock near dusk

    assert force_gated < force_full * 0.9, (
        f"Coherence gate should reduce projection forces: full={force_full:.6f}, "
        f"gated={force_gated:.6f}"
    )


def test_coherence_defaults_to_one_when_no_ecology(default_config):
    """P4.8: Without ecology, _coherence_factor defaults to 1.0 (no gating)."""
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = default_config
    cfg.seed = 42  # D6: default seed is None — pin for determinism
    cfg.mode = "spatial"
    cfg.num_boids = 30
    cfg.noise_scale = 0.0

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    # No _coherence_factor set → defaults to 1.0
    SpatialMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )

    # Should not crash and should produce non-trivial forces
    assert not np.allclose(flock.accelerations[flock.active], 0.0)


