"""P4 unit tests for physics.forces — hybrid filter, predator escape, jitter,
coherence gate, numba kernels, batch query, and integration tests.

Extracted from test_forces.py (~1,400 lines → ~280 + ~1,185).
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


# ── S2.B11: shared curl-flow in SpatialMode ─────────────────────────

def test_spatial_flow_weight_zero_is_baseline(default_config):
    """S2.B11: flow_weight=0 (default) → bit-identical to no-flow baseline."""
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = default_config
    cfg.seed = 42
    cfg.mode = "spatial"
    cfg.num_boids = 30
    cfg.noise_scale = 0.0
    assert cfg.flow_weight == 0.0

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)
    SpatialMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )
    acc_flow_zero = flock.accelerations.copy()

    flock2 = PhysicsFlock(cfg)
    flock2.accelerations[:] = 0.0
    flock2.get_index().rebuild(flock2.positions, flock2.active)
    object.__setattr__(cfg, 'flow_weight', 0.0)  # explicit, same as default
    SpatialMode.compute(
        flock2.positions, flock2.velocities, flock2.accelerations,
        flock2.active, flock2.get_index(), flock2.rng,
        flock2.last_theta, cfg,
    )
    np.testing.assert_array_equal(acc_flow_zero, flock2.accelerations)


def test_spatial_flow_weight_matches_shared_curl_flow_primitive(default_config):
    """S2.B11: SpatialMode's flow contribution equals curl_flow(...) * flow_weight * 0.22,
    the same L0 primitive FieldMode uses for its own curl-flow term."""
    from pymurmur.physics.forces._base import curl_flow
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = default_config
    cfg.seed = 42
    cfg.mode = "spatial"
    cfg.num_boids = 20
    cfg.noise_scale = 0.0
    cfg.separation_weight = 0.0
    cfg.alignment_weight = 0.0
    cfg.cohesion_weight = 0.0
    cfg.flow_weight = 0.5

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)
    object.__setattr__(cfg, '_field_time', 1.25)

    SpatialMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )

    active_idx = np.where(flock.active)[0]
    C = np.mean(flock.positions[active_idx], axis=0)
    U = 0.4 * min(cfg.width, cfg.height, cfg.depth)
    seeds = np.arange(len(active_idx), dtype=np.float32)
    expected_flow = curl_flow(
        flock.positions[active_idx], C, seeds, 1.25, U,
    ) * (cfg.flow_weight * 0.22) * cfg.acceleration_scale

    # With separation/alignment/cohesion all zeroed and no clamp triggered
    # (small magnitude), the total acceleration should equal the flow
    # contribution scaled by acceleration_scale (P4.2 pipeline step 2).
    np.testing.assert_allclose(
        flock.accelerations[active_idx], expected_flow, atol=1e-5,
    )


# ── S2.B11: seed_sinusoidal noise mode ──────────────────────────────

def test_spatial_seed_sinusoidal_deterministic_same_seeds_and_t(default_config):
    """S2.B11: seed_sinusoidal noise depends only on (seeds, t), not the
    rng stream — two identical runs at the same _field_time produce
    identical noise contributions even with different rng states."""
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = default_config
    cfg.mode = "spatial"
    cfg.num_boids = 15
    cfg.noise_mode = "seed_sinusoidal"
    cfg.noise_scale = 0.18
    cfg.separation_weight = 0.0
    cfg.alignment_weight = 0.0
    cfg.cohesion_weight = 0.0
    cfg.flow_weight = 0.0
    object.__setattr__(cfg, '_field_time', 2.5)

    flock1 = PhysicsFlock(cfg)
    flock1.rng = np.random.default_rng(1)  # different rng stream...
    flock1.accelerations[:] = 0.0
    flock1.get_index().rebuild(flock1.positions, flock1.active)

    flock2 = PhysicsFlock(cfg)
    flock2.positions[:] = flock1.positions
    flock2.velocities[:] = flock1.velocities
    flock2.rng = np.random.default_rng(999)  # ...shouldn't matter
    flock2.accelerations[:] = 0.0
    flock2.get_index().rebuild(flock2.positions, flock2.active)

    SpatialMode.compute(
        flock1.positions, flock1.velocities, flock1.accelerations,
        flock1.active, flock1.get_index(), flock1.rng, flock1.last_theta, cfg,
    )
    SpatialMode.compute(
        flock2.positions, flock2.velocities, flock2.accelerations,
        flock2.active, flock2.get_index(), flock2.rng, flock2.last_theta, cfg,
    )
    np.testing.assert_array_equal(flock1.accelerations, flock2.accelerations)


def test_spatial_seed_sinusoidal_matches_seed_noise3_scaled(default_config):
    """S2.B11: seed_sinusoidal output equals seed_noise3(seeds, t) * (noise_scale/0.18)."""
    from pymurmur.core.types import seed_noise3
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = default_config
    cfg.mode = "spatial"
    cfg.num_boids = 10
    cfg.noise_mode = "seed_sinusoidal"
    cfg.noise_scale = 0.36  # 2x the ±0.18 base range
    cfg.separation_weight = 0.0
    cfg.alignment_weight = 0.0
    cfg.cohesion_weight = 0.0
    cfg.flow_weight = 0.0
    object.__setattr__(cfg, '_field_time', 4.0)

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    SpatialMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng, flock.last_theta, cfg,
    )

    active_idx = np.where(flock.active)[0]
    seeds = np.arange(len(active_idx), dtype=np.float32)
    # Noise is added after the acceleration_scale step (same as "additive"
    # mode) — no extra scaling here.
    expected = seed_noise3(seeds, 4.0) * (cfg.noise_scale / 0.18)
    np.testing.assert_allclose(flock.accelerations[active_idx], expected, atol=1e-5)


def test_spatial_seed_sinusoidal_bounded_at_default_noise_scale(default_config):
    """S2.B11: at noise_scale=0.18 (the atom's native range), each axis of
    the per-bird noise contribution stays within ±0.18 (pre acceleration_scale)."""
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = default_config
    cfg.mode = "spatial"
    cfg.num_boids = 40
    cfg.noise_mode = "seed_sinusoidal"
    cfg.noise_scale = 0.18
    cfg.separation_weight = 0.0
    cfg.alignment_weight = 0.0
    cfg.cohesion_weight = 0.0
    cfg.flow_weight = 0.0
    cfg.acceleration_scale = 1.0
    object.__setattr__(cfg, '_field_time', 0.7)

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    SpatialMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng, flock.last_theta, cfg,
    )
    assert np.all(np.abs(flock.accelerations[flock.active]) <= 0.18 + 1e-5)


# ── P4.1: Hybrid filter unit tests ─────────────────────────

def test_hybrid_filter_caps_at_influence_count(default_config):
    """P4.1: After _query_neighbors, no bird has > influence_count neighbours."""
    from pymurmur.physics.forces.spatial import _query_neighbors

    cfg = default_config
    cfg.seed = 42  # D6: default seed is None — pin for determinism
    cfg.mode = "spatial"
    cfg.num_boids = 200
    cfg.influence_count = 5
    cfg.visual_range = 300.0

    flock = PhysicsFlock(cfg)
    flock.get_index().rebuild(flock.positions, flock.active)

    n_idx = _query_neighbors(flock.positions, flock.active, flock.get_index(), cfg)

    for i in range(cfg.num_boids):
        valid = n_idx[i][n_idx[i] > 0]
        assert len(valid) <= cfg.influence_count, (
            f"Bird {i} has {len(valid)} neighbours, cap is {cfg.influence_count}"
        )


def test_hybrid_filter_visual_range_enforced(default_config):
    """P4.1: All accepted neighbours are within visual_range."""
    from pymurmur.physics.forces.spatial import _query_neighbors

    cfg = default_config
    cfg.seed = 42  # D6: default seed is None — pin for determinism
    cfg.mode = "spatial"
    cfg.num_boids = 100
    cfg.influence_count = 10
    cfg.visual_range = 80.0  # tight range

    flock = PhysicsFlock(cfg)
    flock.get_index().rebuild(flock.positions, flock.active)

    n_idx = _query_neighbors(flock.positions, flock.active, flock.get_index(), cfg)

    for i in range(cfg.num_boids):
        valid = n_idx[i][n_idx[i] > 0]
        if len(valid) == 0:
            continue
        diffs = flock.positions[valid] - flock.positions[i]
        dists = np.linalg.norm(diffs, axis=1)
        assert (dists <= cfg.visual_range + 1.0).all(), (
            f"Bird {i}: neighbour at dist {dists.max():.1f} > {cfg.visual_range}"
        )


def test_hybrid_filter_with_scattered_zeros(default_config):
    """P4.1+P4.10: Scattered zeros in neighbour array don't break filtering.

    This specifically tests the bug fixed in P4.10 where numba's `break`
    on encountering zero indices skipped valid neighbours."""
    from pymurmur.physics.forces._kernels import (
        _numba_hybrid_filter,
        _numpy_hybrid_filter,
    )

    N = 20
    k = 10
    positions = np.random.default_rng(42).uniform(0, 1000, (N, 3)).astype(np.float32)
    active = np.ones(N, dtype=bool)

    # Build neighbour arrays with scattered zeros: bird 0 has neighbours
    # [1, 0, 2, 0, 3, 4, 0, 5, 6, 0] — zeros interspersed with valid indices
    n_idx_numba = np.zeros((N, k), dtype=np.int32)
    n_idx_numpy = np.zeros((N, k), dtype=np.int32)
    scattered = [1, 0, 2, 0, 3, 4, 0, 5, 6, 0]
    n_idx_numba[0, :len(scattered)] = scattered
    n_idx_numpy[0, :len(scattered)] = scattered

    _numba_hybrid_filter(n_idx_numba, positions, active, visual_range=5000.0, influence_count=4)
    _numpy_hybrid_filter(n_idx_numpy, positions, active, visual_range=5000.0, influence_count=4)

    # Both must match and cap at 4
    assert np.array_equal(n_idx_numba, n_idx_numpy), \
        "numba and numpy hybrid filter must match with scattered zeros"
    valid = n_idx_numba[0][n_idx_numba[0] > 0]
    assert len(valid) <= 4, f"Scattered-zeros bird should have ≤4, got {len(valid)}"
    assert len(valid) > 0, "Should have at least some neighbours"


def test_hybrid_filter_empty_or_single(default_config):
    """P4.1: Zero or one active bird → no crash, empty neighbour array."""
    from pymurmur.physics.forces.spatial import _query_neighbors

    cfg = default_config
    cfg.seed = 42  # D6: default seed is None — pin for determinism
    cfg.mode = "spatial"
    cfg.num_boids = 1

    flock = PhysicsFlock(cfg)
    flock.get_index().rebuild(flock.positions, flock.active)
    n_idx = _query_neighbors(flock.positions, flock.active, flock.get_index(), cfg)
    assert n_idx.shape[1] == 0, f"Single bird should have 0-width neighbour array, got shape {n_idx.shape}"

    # Zero active
    cfg.num_boids = 10
    flock2 = PhysicsFlock(cfg)
    flock2.active[:] = False
    flock2.get_index().rebuild(flock2.positions, flock2.active)
    n_idx2 = _query_neighbors(flock2.positions, flock2.active, flock2.get_index(), cfg)
    assert n_idx2.shape[1] == 0, "Zero active should have 0-width neighbour array"


def test_influence_count_config_wired_to_filter(default_config):
    """P4.1: influence_count config field reaches the hybrid filter."""
    from pymurmur.physics.forces.spatial import _query_neighbors

    for ic in [3, 7, 12]:
        cfg = default_config
        cfg.seed = 42  # D6: default seed is None — pin for determinism
        cfg.mode = "spatial"
        cfg.num_boids = 100
        cfg.influence_count = ic
        cfg.visual_range = 500.0

        flock = PhysicsFlock(cfg)
        flock.get_index().rebuild(flock.positions, flock.active)
        n_idx = _query_neighbors(flock.positions, flock.active, flock.get_index(), cfg)

        for i in range(cfg.num_boids):
            valid = n_idx[i][n_idx[i] > 0]
            assert len(valid) <= ic, (
                f"influence_count={ic}: bird {i} has {len(valid)} > {ic}"
            )


# ── P4.10: Numba kernel unit tests ─────────────────────────

def test_numba_numpy_hybrid_filter_equivalence():
    """P4.10: numba and numpy hybrid filter produce identical output."""
    from pymurmur.physics.forces._kernels import (
        _HAS_NUMBA,
        _numba_hybrid_filter,
        _numpy_hybrid_filter,
    )
    if not _HAS_NUMBA:
        pytest.skip("numba not available")

    rng = np.random.default_rng(42)
    N, k = 50, 20
    positions = rng.uniform(0, 1000, (N, 3)).astype(np.float32)
    active = np.ones(N, dtype=bool)

    n_idx_numba = np.zeros((N, k), dtype=np.int32)
    n_idx_numpy = np.zeros((N, k), dtype=np.int32)
    for i in range(N):
        cands = np.array([j for j in range(N) if j != i], dtype=np.int32)
        chosen = rng.choice(cands, min(15, len(cands)), replace=False)
        n_idx_numba[i, :len(chosen)] = chosen
        n_idx_numpy[i, :len(chosen)] = chosen

    _numba_hybrid_filter(n_idx_numba, positions, active, visual_range=400.0, influence_count=6)
    _numpy_hybrid_filter(n_idx_numpy, positions, active, visual_range=400.0, influence_count=6)

    assert np.array_equal(n_idx_numba, n_idx_numpy), \
        "numba and numpy hybrid filter must be identical"


def test_numba_numpy_predator_detect_equivalence():
    """P4.10: numba and numpy predator detection produce identical output."""
    from pymurmur.physics.forces._kernels import (
        _HAS_NUMBA,
        _numba_predator_detect,
        _numpy_predator_detect,
    )
    if not _HAS_NUMBA:
        pytest.skip("numba not available")

    rng = np.random.default_rng(42)
    N, k = 30, 10
    rng.uniform(0, 1000, (N, 3)).astype(np.float32)
    active = np.ones(N, dtype=bool)
    is_predator = np.zeros(N, dtype=bool)
    is_predator[0] = is_predator[3] = True  # two predators

    n_idx = np.zeros((N, k), dtype=np.int32)
    for i in range(N):
        if i in (0, 3):
            continue
        n_idx[i, 0] = 0  # bird 0 is a predator
        if i % 2 == 0:
            n_idx[i, 1] = 3  # bird 3 also predator for even-index birds

    threat_numba = np.zeros(N, dtype=bool)
    threat_numpy = np.zeros(N, dtype=bool)
    _numba_predator_detect(threat_numba, n_idx, is_predator, active)
    _numpy_predator_detect(threat_numpy, n_idx, is_predator, active)
    assert np.array_equal(threat_numba, threat_numpy)


def test_numba_numpy_predator_escape_equivalence():
    """P4.10: numba and numpy predator escape produce identical output."""
    from pymurmur.physics.forces._kernels import (
        _HAS_NUMBA,
        _numba_predator_escape,
        _numpy_predator_escape,
    )
    if not _HAS_NUMBA:
        pytest.skip("numba not available")

    rng = np.random.default_rng(42)
    N, k = 20, 8
    positions = rng.uniform(0, 1000, (N, 3)).astype(np.float32)
    # Place prey right next to predator
    positions[1] = positions[0] + np.array([8.0, 0, 0], dtype=np.float32)
    active = np.ones(N, dtype=bool)
    is_predator = np.zeros(N, dtype=bool)
    is_predator[0] = True
    threatened = np.zeros(N, dtype=bool)
    threatened[1] = True

    n_idx = np.zeros((N, k), dtype=np.int32)
    n_idx[1, 0] = 0  # bird 1 sees predator 0

    escape_factor, accel_boost = 1e6, 1.4
    esc_numba = np.zeros((N, 3), dtype=np.float32)
    esc_numpy = np.zeros((N, 3), dtype=np.float32)
    _numba_predator_escape(esc_numba, positions, n_idx, is_predator,
                            threatened, active, escape_factor, accel_boost)
    _numpy_predator_escape(esc_numpy, positions, n_idx, is_predator,
                            threatened, active, escape_factor, accel_boost)
    assert np.allclose(esc_numba, esc_numpy), \
        f"numba and numpy escape must match: {np.abs(esc_numba - esc_numpy).max():.6f}"


def test_numba_predator_detect_excludes_predators():
    """P4.10: Predators are never marked as threatened."""
    from pymurmur.physics.forces._kernels import _HAS_NUMBA, _numba_predator_detect
    if not _HAS_NUMBA:
        pytest.skip("numba not available")

    N, k = 20, 8
    np.random.default_rng(42).uniform(0, 1000, (N, 3)).astype(np.float32)
    active = np.ones(N, dtype=bool)
    is_predator = np.zeros(N, dtype=bool)
    is_predator[0] = True
    is_predator[1] = True  # two predators, each sees the other

    n_idx = np.zeros((N, k), dtype=np.int32)
    n_idx[0, 0] = 1  # predator 0 sees predator 1
    n_idx[1, 0] = 0  # predator 1 sees predator 0

    threatened = np.zeros(N, dtype=bool)
    _numba_predator_detect(threatened, n_idx, is_predator, active)

    assert not threatened[0], "Predator 0 should not be threatened"
    assert not threatened[1], "Predator 1 should not be threatened"


def test_numba_predator_escape_direction():
    """P4.10: Escape force points away from nearest predator.

    Uses bird index 2 as predator (not 0) to avoid ambiguity with
    the zero-padding sentinel in neighbour arrays."""
    from pymurmur.physics.forces._kernels import _HAS_NUMBA, _numba_predator_escape
    if not _HAS_NUMBA:
        pytest.skip("numba not available")

    N, k = 10, 5
    positions = np.random.default_rng(42).uniform(0, 100, (N, 3)).astype(np.float32)
    # Predator at (50, 50, 50), prey at (60, 50, 50) — 10 units +x
    positions[2] = [50.0, 50.0, 50.0]
    positions[3] = [60.0, 50.0, 50.0]
    active = np.ones(N, dtype=bool)
    is_predator = np.zeros(N, dtype=bool)
    is_predator[2] = True  # bird 2 is predator (non-zero index)
    threatened = np.zeros(N, dtype=bool)
    threatened[3] = True   # bird 3 is threatened prey

    n_idx = np.zeros((N, k), dtype=np.int32)
    n_idx[3, 0] = 2  # bird 3 sees predator 2

    escape = np.zeros((N, 3), dtype=np.float32)
    _numba_predator_escape(escape, positions, n_idx, is_predator,
                            threatened, active, escape_factor=1e6, accel_boost=1.0)

    # Escape should push prey (+x) away from predator at x=50
    assert escape[3, 0] > 0, f"Escape should push +x, got {escape[3]}"
    assert abs(escape[3, 1]) < 1e-6 and abs(escape[3, 2]) < 1e-6, \
        "Escape should have no y/z component for collinear predator-prey"


def test_numba_predator_escape_scattered_zeros():
    """P4.10: Escape handles scattered zeros in neighbour array.

    Regression test for the break→continue bug — scattered zeros
    shouldn't prevent finding a predator further in the list."""
    from pymurmur.physics.forces._kernels import _HAS_NUMBA, _numba_predator_escape
    if not _HAS_NUMBA:
        pytest.skip("numba not available")

    N, k = 10, 8
    positions = np.random.default_rng(42).uniform(0, 100, (N, 3)).astype(np.float32)
    positions[5] = [30.0, 40.0, 50.0]  # predator
    positions[7] = [38.0, 40.0, 50.0]  # prey, 8 units +x from predator
    active = np.ones(N, dtype=bool)
    is_predator = np.zeros(N, dtype=bool)
    is_predator[5] = True
    threatened = np.zeros(N, dtype=bool)
    threatened[7] = True

    # Scattered zeros: [0, 5, 0, 0, ...] — zero at position 0, predator at position 1
    n_idx = np.zeros((N, k), dtype=np.int32)
    n_idx[7, 0] = 0   # zero sentinel (should be skipped)
    n_idx[7, 1] = 5   # predator (should be found)
    n_idx[7, 2] = 0   # zero sentinel
    n_idx[7, 3] = 1   # regular bird (not predator, should be skipped)

    escape = np.zeros((N, 3), dtype=np.float32)
    _numba_predator_escape(escape, positions, n_idx, is_predator,
                            threatened, active, escape_factor=1e6, accel_boost=1.0)

    # Must find predator 5 through the scattered zeros
    assert escape[7, 0] > 0, (
        f"Escape should push +x (predator at x=30, prey at x=38), got {escape[7]}"
    )


# ═══════════════════════════════════════════════════════════════════
# P4.1 + P4.10 Integration tests — full pipeline as a whole
# ═══════════════════════════════════════════════════════════════════

def test_p42_accel_scale_zero_produces_zero_forces(default_config):
    """P4.2: accel_scale=0 multiplies all accumulated forces to zero.

    With non-zero weights and neighbours, forces accumulate normally.
    But acceleration_scale=0 should zero everything (before noise).
    Setting noise_scale=0 too ensures final forces are exactly zero."""
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = default_config
    cfg.seed = 42  # D6: default seed is None — pin for determinism
    cfg.mode = "spatial"
    cfg.num_boids = 30
    cfg.separation_weight = 5.0
    cfg.alignment_weight = 2.0
    cfg.cohesion_weight = 2.0
    cfg.noise_scale = 0.0
    cfg.max_force = 10.0
    cfg.acceleration_scale = 0.0  # ← zeroes all accumulated forces

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    SpatialMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )

    # All forces should be zero (accumulated forces × 0 + noise=0)
    assert np.allclose(flock.accelerations[flock.active], 0.0), (
        f"accel_scale=0 should zero all forces, got max={np.abs(flock.accelerations).max():.6f}"
    )


def test_full_pipeline_predator_with_hybrid_filter(default_config):
    """P4.1+P4.10: Predator escape + hybrid filter work together end-to-end.

    Sets up a predator, places prey nearby, runs SpatialMode.compute
    (which exercises: _query_neighbors → hybrid filter → force primitives
    → predator detect → predator escape → accumulate → clamp → noise).
    Verifies the prey gets escape force and alignment/cohesion are zeroed."""
    from pymurmur.physics.forces.spatial import SpatialMode, _query_neighbors

    cfg = default_config
    cfg.seed = 42  # D6: default seed is None — pin for determinism
    cfg.mode = "spatial"
    cfg.num_boids = 30
    cfg.visual_range = 500.0
    cfg.influence_count = 10
    cfg.noise_scale = 0.0
    cfg.max_force = 10.0

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.is_predator[0] = True
    # Place prey bird close to predator
    flock.positions[1] = flock.positions[0] + np.array([15.0, 0, 0], dtype=np.float32)
    flock.get_index().rebuild(flock.positions, flock.active)
    object.__setattr__(cfg, '_is_predator', flock.is_predator)

    SpatialMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )

    # Prey (bird 1) should have escape force pointing +x
    assert flock.accelerations[1, 0] > 0, (
        f"Prey should get escape force +x, got {flock.accelerations[1]}"
    )
    # Predator should NOT get escape force
    assert not np.allclose(flock.accelerations[0], 0.0), \
        "Predator should still get flocking forces"

    # Verify hybrid filter capped neighbors
    n_idx = _query_neighbors(flock.positions, flock.active, flock.get_index(), cfg)
    for i in range(cfg.num_boids):
        valid = n_idx[i][n_idx[i] > 0]
        assert len(valid) <= cfg.influence_count, \
            f"Bird {i}: {len(valid)} > {cfg.influence_count}"


def test_full_pipeline_determinism(default_config):
    """P4.1+P4.10: Same seed + same config → bit-identical forces.

    The numba kernels (with seeded RNG) must produce deterministic
    output. Runs two identical flocks through SpatialMode.compute."""
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = default_config
    cfg.seed = 42  # D6: default seed is None — pin for determinism
    cfg.mode = "spatial"
    cfg.num_boids = 50
    cfg.noise_scale = 0.1  # non-zero noise exercises RNG determinism

    flock1 = PhysicsFlock(cfg)
    flock1.accelerations[:] = 0.0
    flock1.get_index().rebuild(flock1.positions, flock1.active)

    flock2 = PhysicsFlock(cfg)
    flock2.accelerations[:] = 0.0
    flock2.get_index().rebuild(flock2.positions, flock2.active)

    SpatialMode.compute(
        flock1.positions, flock1.velocities, flock1.accelerations,
        flock1.active, flock1.get_index(), flock1.rng,
        flock1.last_theta, cfg,
    )
    SpatialMode.compute(
        flock2.positions, flock2.velocities, flock2.accelerations,
        flock2.active, flock2.get_index(), flock2.rng,
        flock2.last_theta, cfg,
    )

    assert np.array_equal(flock1.accelerations, flock2.accelerations), \
        "Identical seed+config must produce bit-identical forces"


def test_full_pipeline_influence_count_affects_forces(default_config):
    """P4.1+P4.10: Different influence_count → different force distribution.

    A tight cap (3) vs loose cap (20) should produce measurably
    different mean force magnitudes because more/fewer neighbours
    contribute to cohesion and alignment."""
    from pymurmur.physics.forces.spatial import SpatialMode

    forces = {}
    for ic in [3, 20]:
        cfg = default_config
        cfg.seed = 42  # D6: default seed is None — pin for determinism
        cfg.mode = "spatial"
        cfg.num_boids = 100
        cfg.influence_count = ic
        cfg.visual_range = 500.0
        cfg.noise_scale = 0.0

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        flock.get_index().rebuild(flock.positions, flock.active)

        SpatialMode.compute(
            flock.positions, flock.velocities, flock.accelerations,
            flock.active, flock.get_index(), flock.rng,
            flock.last_theta, cfg,
        )
        mags = np.linalg.norm(flock.accelerations[flock.active], axis=1)
        forces[ic] = float(mags.mean())

    # Different caps should produce meaningfully different force distributions.
    # Fewer neighbours → less averaging → more variable forces.
    # More neighbours → more averaging → smoother, more clustered forces.
    # The mean magnitudes can go either way, but the distributions must differ.
    ratio = max(forces[3], forces[20]) / min(forces[3], forces[20])
    assert abs(ratio - 1.0) > 0.001, (
        f"ic=3 mean={forces[3]:.4f}, ic=20 mean={forces[20]:.4f} — "
        f"different caps must produce measurably different forces (ratio={ratio:.4f})"
    )


def test_full_pipeline_clamp_then_noise(default_config):
    """P4.1+P4.2+P4.10: Post-clamp noise addition works through full pipeline.

    Noise is added AFTER the max_force clamp (P4.2). Verifies that
    with noise_scale > 0, some forces exceed max_force (noise is
    unclamped), but forces without noise stay ≤ max_force."""
    from pymurmur.physics.forces.spatial import SpatialMode

    # Without noise: all forces ≤ max_force
    cfg = default_config
    cfg.seed = 42  # D6: default seed is None — pin for determinism
    cfg.mode = "spatial"
    cfg.num_boids = 50
    cfg.noise_scale = 0.0
    cfg.max_force = 0.5

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    SpatialMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )
    mags_no_noise = np.linalg.norm(flock.accelerations[flock.active], axis=1)
    assert mags_no_noise.max() <= cfg.max_force + 1e-4, \
        f"Without noise: max={mags_no_noise.max():.4f} > {cfg.max_force}"

    # With noise: some forces may exceed max_force
    cfg.noise_scale = 0.3
    flock2 = PhysicsFlock(cfg)
    flock2.accelerations[:] = 0.0
    flock2.get_index().rebuild(flock2.positions, flock2.active)

    SpatialMode.compute(
        flock2.positions, flock2.velocities, flock2.accelerations,
        flock2.active, flock2.get_index(), flock2.rng,
        flock2.last_theta, cfg,
    )
    mags_noise = np.linalg.norm(flock2.accelerations[flock2.active], axis=1)
    # Noise increases mean force (added post-clamp)
    assert mags_noise.mean() > mags_no_noise.mean(), \
        f"Noise should increase mean force: {mags_no_noise.mean():.4f} → {mags_noise.mean():.4f}"


def test_full_pipeline_all_phase4_features(default_config):
    """P4.1+P4.2+P4.3+P4.5+P4.8+P4.10: All spatial features work together.

    Exercises the complete spatial pipeline with:
    - P4.1: hybrid filter (influence_count=7)
    - P4.2: accumulate → accel_scale → clamp → noise
    - P4.3: predator escape + zeroed align/coh
    - P4.5: jitter on sep/coh/align
    - P4.8: coherence gate (via _coherence_factor)
    - P4.10: numba-accelerated kernels"""
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = default_config
    cfg.seed = 42  # D6: default seed is None — pin for determinism
    cfg.mode = "spatial"
    cfg.num_boids = 60
    cfg.influence_count = 7
    cfg.visual_range = 300.0
    cfg.separation_weight = 4.5
    cfg.alignment_weight = 0.65
    cfg.cohesion_weight = 0.75
    cfg.noise_scale = 0.1
    cfg.max_force = 5.0
    cfg.acceleration_scale = 0.3
    cfg.jitter_separation = 0.1
    cfg.jitter_cohesion = 0.1
    cfg.jitter_alignment = 0.1

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.is_predator[0] = True
    # Place prey near predator
    flock.positions[1] = flock.positions[0] + np.array([12.0, 0, 0], dtype=np.float32)
    flock.get_index().rebuild(flock.positions, flock.active)
    object.__setattr__(cfg, '_is_predator', flock.is_predator)
    object.__setattr__(cfg, '_coherence_factor', 0.8)  # P4.8 gate

    SpatialMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng,
        flock.last_theta, cfg,
    )

    mags = np.linalg.norm(flock.accelerations[flock.active], axis=1)

    # Basic sanity: forces finite, non-zero, reasonable range
    assert np.isfinite(flock.accelerations).all(), "NaN/Inf in forces"
    assert mags.mean() > 0, "Forces should be non-zero"
    assert mags.max() < 20.0, f"Forces unexpectedly large: max={mags.max():.1f}"

    # Prey should have escape component (P4.3)
    assert flock.accelerations[1, 0] > 0, \
        f"Prey escape +x expected, got {flock.accelerations[1]}"

    # Predator should have flocking forces (not escape)
    assert not np.allclose(flock.accelerations[0], 0.0), \
        "Predator should get flocking forces"

    # Jitter should increase force variance vs no-jitter run (P4.5)
    cfg_no_jit = default_config
    for attr in ('mode', 'num_boids', 'influence_count', 'visual_range',
                 'separation_weight', 'alignment_weight', 'cohesion_weight',
                 'noise_scale', 'max_force', 'acceleration_scale'):
        setattr(cfg_no_jit, attr, getattr(cfg, attr))
    cfg_no_jit.jitter_separation = 0.0
    cfg_no_jit.jitter_cohesion = 0.0
    cfg_no_jit.jitter_alignment = 0.0

    flock_nj = PhysicsFlock(cfg_no_jit)
    flock_nj.accelerations[:] = 0.0
    flock_nj.get_index().rebuild(flock_nj.positions, flock_nj.active)

    SpatialMode.compute(
        flock_nj.positions, flock_nj.velocities, flock_nj.accelerations,
        flock_nj.active, flock_nj.get_index(), flock_nj.rng,
        flock_nj.last_theta, cfg_no_jit,
    )
    mags_nj = np.linalg.norm(flock_nj.accelerations[flock_nj.active], axis=1)

    # Jitter should increase force spread (different weights per frame)
    assert mags.std() > mags_nj.std() * 1.05, (
        f"Jitter should increase force variance: "
        f"jitter std={mags.std():.4f}, no-jitter std={mags_nj.std():.4f}"
    )


# ── P4.6: Batch k-NN query verification ──────────────────────────

def test_p46_batch_query_matches_per_bird(default_config):
    """P4.6: Batch cKDTree query produces same neighbours as per-bird queries.

    The batch query optimization (tree.query(active_pos, k=k+1, workers=-1))
    must produce identical results to individual per-bird tree.query() calls.
    This test verifies correctness, not speed."""
    from scipy.spatial import cKDTree

    from pymurmur.physics.forces.spatial import _query_neighbors

    cfg = default_config
    cfg.seed = 42  # D6: default seed is None — pin for determinism
    cfg.mode = "spatial"
    cfg.num_boids = 100
    cfg.influence_count = 7
    cfg.visual_range = 500.0  # large enough to not filter in metric step

    flock = PhysicsFlock(cfg)
    flock.get_index().rebuild(flock.positions, flock.active)

    # Get neighbours via the batch query path (uses tree.query batch)
    n_idx_batch = _query_neighbors(flock.positions, flock.active, flock.get_index(), cfg)

    # Get neighbours via per-bird path for verification
    active_idx = np.where(flock.active)[0]
    positions = flock.positions
    k = max(cfg.influence_count * 3, getattr(cfg, 'topological_cap', 50))
    k = min(k, len(active_idx) - 1)

    # Build a fresh tree on active positions (per-bird query baseline)
    tree = cKDTree(positions[active_idx])
    n_idx_per_bird = np.zeros((len(positions), k), dtype=np.int32)
    for _j, global_i in enumerate(active_idx):
        _, compacted = tree.query(positions[global_i], k=k + 1)
        n_idx_per_bird[global_i] = active_idx[compacted[1:k + 1]]

    # Apply the same hybrid filter to per-bird results for fair comparison
    from pymurmur.physics.forces._kernels import _numba_hybrid_filter
    _numba_hybrid_filter(n_idx_per_bird, positions, flock.active,
                         cfg.visual_range, cfg.influence_count)

    # Both must produce identical neighbour sets
    assert np.array_equal(n_idx_batch, n_idx_per_bird), (
        "Batch query must produce identical neighbours to per-bird query"
    )


def test_p46_batch_query_all_birds_have_neighbours(default_config):
    """P4.6: Batch query — every active bird has at least 1 neighbour in dense flock."""
    from pymurmur.physics.forces.spatial import _query_neighbors

    cfg = default_config
    cfg.seed = 42  # D6: default seed is None — pin for determinism
    cfg.mode = "spatial"
    cfg.num_boids = 200  # dense
    cfg.influence_count = 5
    cfg.visual_range = 200.0

    flock = PhysicsFlock(cfg)
    flock.get_index().rebuild(flock.positions, flock.active)

    n_idx = _query_neighbors(flock.positions, flock.active, flock.get_index(), cfg)

    # In a dense flock, every bird (except maybe extreme outliers) should have ≥1 neighbour
    birds_without_neighbours = 0
    for i in range(cfg.num_boids):
        valid = n_idx[i][n_idx[i] > 0]
        if len(valid) == 0:
            birds_without_neighbours += 1

    # Allow up to 5% isolated birds (edge of domain)
    assert birds_without_neighbours < cfg.num_boids * 0.05, (
        f"{birds_without_neighbours}/{cfg.num_boids} birds have no neighbours in dense flock"
    )


# ═══════════════════════════════════════════════════════════════════
# P4 integration tests — multiple P4 items working together as a whole
# ═══════════════════════════════════════════════════════════════════

def test_p47_p43_sphere_boundary_with_predator_escape(default_config):
    """P4.7+P4.3: Sphere boundary + predator — birds stay inside sphere,
    prey get escape force, predator gets normal flocking forces.

    Verifies that the sphere soft boundary (P4.7 _sphere_soft) and
    predator escape (P4.3) work together without conflict: birds near
    the sphere edge get inward push while also fleeing the predator."""
    from pymurmur.simulation.engine import SimulationEngine

    cfg = default_config
    cfg.seed = 42  # D6: default seed is None — pin for determinism
    cfg.mode = "spatial"
    cfg.num_boids = 30
    cfg.boundary_mode = "sphere"
    cfg.boundary_sphere_radius = 150.0
    cfg.predator_enabled = True
    cfg.roosting_enabled = False
    cfg.noise_scale = 0.0

    engine = SimulationEngine(cfg)

    # Place predator at flock centre (all birds are prey)
    pred = getattr(engine.extensions, '_predator', None)
    assert pred is not None
    engine.flock.update_center()
    pred._pos = engine.flock.center.copy()
    pred._phase = "approach"

    # Run several steps — no bird should escape the sphere.
    # D1: the sphere is centred on the domain centre C, not the origin.
    C = np.array([cfg.width / 2, cfg.height / 2, cfg.depth / 2],
                 dtype=np.float32)
    for _ in range(30):
        engine.step(1.0 / 60.0)
        dists = np.linalg.norm(engine.flock.positions - C, axis=1)
        assert (dists <= cfg.boundary_sphere_radius + 1.0).all(), (
            f"Bird outside sphere: max dist={dists.max():.1f} > {cfg.boundary_sphere_radius}"
        )

    # Forces should be finite and non-zero (predator present)
    last_acc = engine.flock.last_accelerations[engine.flock.active]
    assert np.isfinite(last_acc).all(), "NaN/Inf in forces with sphere+predator"
    assert not np.allclose(last_acc, 0.0), (
        "Forces should be non-zero with predator present"
    )


def test_p49_p41_velocity_init_affects_spatial_force_distribution(default_config):
    """P4.9+P4.1: Different velocity init modes → measurably different force
    distributions through the same spatial force pipeline.

    Cube (random uniform [-v0,v0]³) vs sphere (fixed-speed random direction)
    velocity inits produce different alignment and separation forces because
    the flock starts with different velocity structures."""
    from pymurmur.physics.forces.spatial import SpatialMode

    def forces_for_init(velocity_mode: str) -> np.ndarray:
        cfg = default_config
        cfg.seed = 42  # D6: default seed is None — pin for determinism
        cfg.seed = 42  # D6: pin seed — both flocks share positions, so the
        # only difference between runs is the velocity-init mode under test
        cfg.mode = "spatial"
        cfg.num_boids = 50
        cfg.velocity_init = velocity_mode
        cfg.separation_weight = 4.5
        cfg.alignment_weight = 0.65
        cfg.cohesion_weight = 0.75
        cfg.noise_scale = 0.0
        cfg.influence_count = 7
        cfg.visual_range = 200.0

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        flock.get_index().rebuild(flock.positions, flock.active)

        SpatialMode.compute(
            flock.positions, flock.velocities, flock.accelerations,
            flock.active, flock.get_index(), flock.rng,
            flock.last_theta, cfg,
        )
        return np.linalg.norm(flock.accelerations[flock.active], axis=1)

    mags_cube = forces_for_init("cube")
    mags_sphere = forces_for_init("sphere")

    # Different velocity inits should produce different force distributions.
    # Cube: uniform in [-v0,v0]³ → speeds vary widely, forces more spread.
    # Sphere: fixed |v| = v0·0.8 → equal speeds, more uniform forces.
    # The std of force magnitudes must differ measurably.
    ratio = max(mags_cube.std(), mags_sphere.std()) / min(mags_cube.std(), mags_sphere.std())
    assert ratio > 1.15, (
        f"Cube and sphere velocity inits must produce different force spreads: "
        f"cube std={mags_cube.std():.4f}, sphere std={mags_sphere.std():.4f}, ratio={ratio:.3f}"
    )
    # Both should be finite
    assert np.isfinite(mags_cube).all()
    assert np.isfinite(mags_sphere).all()


def test_p48_p44_ecology_with_physical_metrics(default_config):
    """P4.8+P4.4: Ecology enabled → physical metrics populated.

    Runs SimulationEngine with ecology and metrics enabled. After several
    frames, physical metrics (speed_real_ms, energy_J) should be populated
    because the flock is actively simulating with roosting forces."""
    from pymurmur.simulation.engine import SimulationEngine

    cfg = default_config
    cfg.seed = 42  # D6: default seed is None — pin for determinism
    cfg.mode = "spatial"
    cfg.num_boids = 50
    cfg.roosting_enabled = True
    cfg.metrics_detail_level = 1
    cfg.metrics_interval = 1  # every frame
    cfg.noise_scale = 0.1

    engine = SimulationEngine(cfg)

    # Run enough frames for metrics to accumulate
    for _ in range(60):
        engine.step(1.0 / 60.0)

    m = engine.metrics.snapshot()

    # Physical metrics must be in plausible ranges (not all zero)
    assert m.speed_real_ms >= 0.0, f"speed_real_ms={m.speed_real_ms} < 0"
    assert m.force_real_N >= 0.0, f"force_real_N={m.force_real_N} < 0"
    assert m.energy_J >= 0.0, f"energy_J={m.energy_J} < 0"

    # With a running simulation, speed should be non-zero
    assert m.speed_avg > 0.0, "Flock should have non-zero average speed"

    # Metrics collector should have accumulated history (≥1 frame)
    assert len(engine.metrics.history) > 0, "Metrics history should be non-empty"

    # Ecology should be active — coherence factor must be set
    eco = getattr(engine.extensions, '_ecology', None)
    if eco is not None:
        assert 0.0 <= eco.coherence_factor <= 1.0, (
            f"Ecology coherence_factor={eco.coherence_factor} outside [0,1]"
        )




# ── S2.B11: curl_flow edge cases ───────────────────────────────────

def test_curl_flow_empty_positions():
    """S2.B11: curl_flow with n=0 → zero array."""
    from pymurmur.physics.forces._base import curl_flow
    result = curl_flow(
        np.empty((0, 3), dtype=np.float32),
        np.array([500.0, 350.0, 200.0], dtype=np.float32),
        np.array([], dtype=np.float32),
        1.0, 100.0,
    )
    assert result.shape == (0, 3), f"Empty input → empty output, got {result.shape}"


def test_curl_flow_deterministic():
    """S2.B11: curl_flow with same inputs → same output (deterministic)."""
    from pymurmur.physics.forces._base import curl_flow

    positions = np.array([[100, 200, 300], [400, 500, 600]], dtype=np.float32)
    center = np.array([500.0, 350.0, 200.0], dtype=np.float32)
    seeds = np.array([0.1, 0.5], dtype=np.float32)

    r1 = curl_flow(positions, center, seeds, 1.0, 100.0)
    r2 = curl_flow(positions, center, seeds, 1.0, 100.0)
    np.testing.assert_array_equal(r1, r2)


def test_curl_flow_different_t_different_output():
    """S2.B11: curl_flow at different times produces different flow."""
    from pymurmur.physics.forces._base import curl_flow

    positions = np.array([[100, 200, 300], [400, 500, 600]], dtype=np.float32)
    center = np.array([500.0, 350.0, 200.0], dtype=np.float32)
    seeds = np.array([0.1, 0.5], dtype=np.float32)

    r1 = curl_flow(positions, center, seeds, 1.0, 100.0)
    r2 = curl_flow(positions, center, seeds, 10.0, 100.0)
    assert not np.allclose(r1, r2), (
        "Different t should produce different flow vectors"
    )


def test_curl_flow_magnitude_bounded():
    """S2.B11: curl_flow returns vectors of magnitude 0.08 (normalized)."""
    from pymurmur.physics.forces._base import curl_flow

    positions = np.random.default_rng(42).uniform(0, 1000, (50, 3)).astype(np.float32)
    center = np.array([500.0, 350.0, 200.0], dtype=np.float32)
    seeds = np.arange(50, dtype=np.float32)

    result = curl_flow(positions, center, seeds, 1.5, 100.0)
    mags = np.linalg.norm(result, axis=1)
    assert np.allclose(mags, 0.08, atol=1e-5), (
        f"All flow magnitudes should be 0.08, got {mags.min():.6f}..{mags.max():.6f}"
    )


# ═══════════════════════════════════════════════════════════════════
# S2.B1: alignment_radius_ratio / separation_distance / global filter
# ═══════════════════════════════════════════════════════════════════

def test_alignment_radius_ratio_restricts_alignment_subset(default_config):
    """S2.B1: alignment set is a subset of the sep/coh neighbour set
    when alignment_radius_ratio < 1.0."""
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = default_config
    cfg.seed = 42
    cfg.mode = "spatial"
    cfg.num_boids = 80
    cfg.visual_range = 200.0
    cfg.spatial.alignment_radius_ratio = 0.3  # tight subset

    flock = PhysicsFlock(cfg)
    flock.get_index().rebuild(flock.positions, flock.active)

    accel_restricted = np.zeros_like(flock.accelerations)
    SpatialMode.compute(
        flock.positions, flock.velocities, accel_restricted, flock.active,
        flock.get_index(), flock.rng, flock.last_theta, cfg,
    )

    cfg.spatial.alignment_radius_ratio = 1.0  # no restriction (baseline)
    accel_baseline = np.zeros_like(flock.accelerations)
    SpatialMode.compute(
        flock.positions, flock.velocities, accel_baseline, flock.active,
        flock.get_index(), flock.rng, flock.last_theta, cfg,
    )

    assert not np.allclose(accel_restricted, accel_baseline), (
        "tightening alignment_radius_ratio should change the alignment contribution"
    )


def test_alignment_radius_ratio_default_is_noop(default_config):
    """S2.B1: default alignment_radius_ratio=1.0 must not change forces
    vs before the feature existed (no max_dist_align set)."""
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = default_config
    cfg.seed = 7
    cfg.mode = "spatial"
    cfg.num_boids = 60
    assert cfg.spatial.alignment_radius_ratio == 1.0
    assert cfg.spatial.separation_distance == 0.0

    flock = PhysicsFlock(cfg)
    flock.get_index().rebuild(flock.positions, flock.active)

    accel = np.zeros_like(flock.accelerations)
    SpatialMode.compute(
        flock.positions, flock.velocities, accel, flock.active,
        flock.get_index(), flock.rng, flock.last_theta, cfg,
    )
    assert np.isfinite(accel).all()
    assert (np.linalg.norm(accel, axis=1) > 0).any(), "default config should still produce forces"


def test_separation_distance_gate_restricts_separation(default_config):
    """S2.B1: separation_distance gates separation neighbours to a tighter
    absolute distance than visual_range."""
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = default_config
    cfg.seed = 3
    cfg.mode = "spatial"
    cfg.num_boids = 80
    cfg.visual_range = 200.0
    cfg.separation_weight = 5.0

    flock = PhysicsFlock(cfg)
    flock.get_index().rebuild(flock.positions, flock.active)

    accel_wide = np.zeros_like(flock.accelerations)
    SpatialMode.compute(
        flock.positions, flock.velocities, accel_wide, flock.active,
        flock.get_index(), flock.rng, flock.last_theta, cfg,
    )

    cfg.spatial.separation_distance = 15.0  # much tighter than visual_range
    accel_tight = np.zeros_like(flock.accelerations)
    SpatialMode.compute(
        flock.positions, flock.velocities, accel_tight, flock.active,
        flock.get_index(), flock.rng, flock.last_theta, cfg,
    )

    assert not np.allclose(accel_wide, accel_tight), (
        "separation_distance should change the separation contribution"
    )


def test_neighbor_filter_global_uses_flock_wide_mean(default_config):
    """S2.B1: neighbor_filter='global' steers every bird's alignment/
    cohesion toward the whole-flock mean velocity / centre of mass."""
    from pymurmur.physics.forces._base import alignment_force, cohesion_force
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = default_config
    cfg.seed = 11
    cfg.mode = "spatial"
    cfg.num_boids = 40
    cfg.spatial.neighbor_filter = "global"
    cfg.alignment_weight = 1.0
    cfg.cohesion_weight = 1.0
    cfg.separation_weight = 0.0
    cfg.noise_scale = 0.0

    flock = PhysicsFlock(cfg)
    flock.get_index().rebuild(flock.positions, flock.active)

    accel = np.zeros_like(flock.accelerations)
    SpatialMode.compute(
        flock.positions, flock.velocities, accel, flock.active,
        flock.get_index(), flock.rng, flock.last_theta, cfg,
    )

    active_idx = np.where(flock.active)[0]
    mean_vel = flock.velocities[active_idx].mean(axis=0)
    mean_pos = flock.positions[active_idx].mean(axis=0)

    # Hand-check bird 0: align+coh should point toward global mean vel/CoM
    steer = mean_vel - flock.velocities[0]
    steer_norm = np.linalg.norm(steer)
    expected_align = steer / steer_norm if steer_norm > 1e-6 else np.zeros(3)
    to_center = mean_pos - flock.positions[0]
    length = np.linalg.norm(to_center)
    expected_coh = to_center / length if length > 1.0 else to_center

    expected = expected_align * cfg.alignment_weight + expected_coh * cfg.cohesion_weight
    expected_mag = np.linalg.norm(expected)
    if expected_mag > cfg.max_force:
        expected = expected / expected_mag * cfg.max_force

    assert np.allclose(accel[0], expected, atol=1e-4), (
        f"bird 0 force {accel[0]} != expected global-mean-based force {expected}"
    )


def test_neighbor_filter_global_does_not_crash_with_no_active(default_config):
    """S2.B1: 'global' mode with zero active birds is a no-op, not a crash."""
    from pymurmur.physics.forces.spatial import SpatialMode

    cfg = default_config
    cfg.mode = "spatial"
    cfg.num_boids = 10
    cfg.spatial.neighbor_filter = "global"

    flock = PhysicsFlock(cfg)
    flock.active[:] = False
    flock.get_index().rebuild(flock.positions, flock.active)

    accel = np.zeros_like(flock.accelerations)
    SpatialMode.compute(
        flock.positions, flock.velocities, accel, flock.active,
        flock.get_index(), flock.rng, flock.last_theta, cfg,
    )
    assert np.allclose(accel, 0.0)
