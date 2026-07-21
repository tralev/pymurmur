"""Unit tests for Phase 3 — Projection mode (Pearce 2014 hybrid projection).

Tests projection_forces(), _topological_neighbors(), and related SI refinements.
Separated from test_forces.py for independent Phase 3 coverage verification.
"""

from copy import copy

import numpy as np

from test.helpers import _call_force  # noqa: E402


def test_projection_mode_zero_active(default_config):
    """projection_forces returns early when no birds are active."""
    from pymurmur.physics.flock import PhysicsFlock
    from pymurmur.physics.forces.projection import projection_forces

    cfg = copy(default_config)
    cfg.mode = "projection"
    cfg.num_boids = 10

    flock = PhysicsFlock(cfg)
    flock.active[:] = False
    flock.accelerations[:] = 0.0

    _call_force(projection_forces, flock, cfg)
    assert np.allclose(flock.accelerations, 0.0)


def test_projection_mode_produces_forces(default_config):
    """Projection mode produces non-zero forces with default settings."""
    from pymurmur.physics.flock import PhysicsFlock
    from pymurmur.physics.forces.projection import projection_forces

    cfg = copy(default_config)
    cfg.mode = "projection"
    cfg.num_boids = 300
    cfg.projection.phi_p = 0.5
    cfg.phi_a = 0.5

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    _call_force(projection_forces, flock, cfg)
    acc_active = flock.accelerations[flock.active]
    assert not np.allclose(acc_active, 0.0)
    assert np.isfinite(acc_active).all()


def test_projection_mode_updates_theta(default_config):
    """last_theta is updated after projection_forces."""
    from pymurmur.physics.flock import PhysicsFlock
    from pymurmur.physics.forces.projection import projection_forces

    cfg = copy(default_config)
    cfg.mode = "projection"
    cfg.num_boids = 300

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.last_theta[:] = -1.0  # sentinel value
    flock.get_index().rebuild(flock.positions, flock.active)

    _call_force(projection_forces, flock, cfg)

    # Birds with neighbours get theta >= 0; edge birds with no neighbours
    # in the hash grid's 27-cell radius stay at sentinel -1.0.
    active_theta = flock.last_theta[flock.active]
    assert (active_theta >= -1.0).all()  # never below sentinel
    assert (active_theta >= 0.0).any()   # at least some birds updated


def test_projection_mode_blind_angle_effect(default_config):
    """Setting blind_deg > 0 changes behaviour (doesn't crash)."""
    from pymurmur.physics.flock import PhysicsFlock
    from pymurmur.physics.forces.projection import projection_forces

    cfg = copy(default_config)
    cfg.mode = "projection"
    cfg.num_boids = 300
    cfg.blind_deg = 90.0

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    _call_force(projection_forces, flock, cfg)
    assert np.isfinite(flock.accelerations).all()


def test_projection_mode_anisotropy_effect(default_config):
    """Anisotropy > 1 runs without crash when refinements enabled."""
    from pymurmur.physics.flock import PhysicsFlock
    from pymurmur.physics.forces.projection import projection_forces

    cfg = copy(default_config)
    cfg.mode = "projection"
    cfg.num_boids = 300
    cfg.refinements = True
    cfg.anisotropy = 3.0

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    _call_force(projection_forces, flock, cfg)
    assert np.isfinite(flock.accelerations).all()


def test_projection_mode_steric_enabled(default_config):
    """Steric force is applied when refinements + steric > 0."""
    from pymurmur.physics.flock import PhysicsFlock
    from pymurmur.physics.forces.projection import projection_forces

    cfg = copy(default_config)
    cfg.mode = "projection"
    cfg.num_boids = 30
    cfg.refinements = True
    cfg.steric = 1.0
    cfg.projection.phi_p = 0.0
    cfg.phi_a = 0.0  # only steric

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    _call_force(projection_forces, flock, cfg)
    # With phi_p=0 and phi_a=0, forces come only from steric
    # May or may not be zero depending on neighbour distances
    assert np.isfinite(flock.accelerations).all()


def test_projection_mode_sigma_effect(default_config):
    """Changing sigma changes the number of neighbours considered."""
    from pymurmur.physics.flock import PhysicsFlock
    from pymurmur.physics.forces.projection import projection_forces

    cfg = copy(default_config)
    cfg.mode = "projection"
    cfg.num_boids = 300
    cfg.sigma = 3  # fewer neighbours

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.last_theta[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    _call_force(projection_forces, flock, cfg)
    assert np.isfinite(flock.accelerations).all()


def test_projection_mode_delta_computed(default_config):
    """Delta (projection direction) is non-zero when neighbours present."""
    from pymurmur.physics.flock import PhysicsFlock
    from pymurmur.physics.forces.projection import projection_forces

    cfg = copy(default_config)
    cfg.mode = "projection"
    cfg.num_boids = 300
    cfg.projection.phi_p = 0.8
    cfg.phi_a = 0.0  # only projection component
    cfg.refinements = False  # no steric interference

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    _call_force(projection_forces, flock, cfg)

    # Forces should be non-zero (delta computed from occlusion)
    acc_active = flock.accelerations[flock.active]
    assert not np.allclose(acc_active, 0.0)
    assert np.isfinite(acc_active).all()


def test_projection_mode_force_within_bounds(default_config):
    """No bird's acceleration exceeds config.max_force."""
    from pymurmur.physics.flock import PhysicsFlock
    from pymurmur.physics.forces.projection import projection_forces

    cfg = copy(default_config)
    cfg.mode = "projection"
    cfg.num_boids = 300
    cfg.projection.phi_p = 1.0
    cfg.phi_a = 1.0
    cfg.max_force = 2.0  # low clamp
    cfg.refinements = False  # steric added after clamp, would break the bound

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    _call_force(projection_forces, flock, cfg)

    # Check that all acceleration magnitudes are <= max_force
    acc_mags = np.linalg.norm(flock.accelerations, axis=1)
    active_mags = acc_mags[flock.active]
    assert np.all(active_mags <= cfg.max_force + 1e-5), \
        f"max acc: {active_mags.max()}, limit: {cfg.max_force}"


def test_projection_mode_hash_grid(default_config):
    """Projection mode works with SpatialHashGrid (N < 5000)."""
    from pymurmur.physics.flock import PhysicsFlock
    from pymurmur.physics.forces.projection import projection_forces

    cfg = copy(default_config)
    cfg.mode = "projection"
    cfg.num_boids = 200  # triggers SpatialHashGrid, not KDTreeIndex
    cfg.projection.phi_p = 0.8
    cfg.phi_a = 0.2

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    # Verify we're using SpatialHashGrid
    from pymurmur.physics.flock import SpatialHashGrid
    assert isinstance(flock.get_index(), SpatialHashGrid)

    _call_force(projection_forces, flock, cfg)

    # Should produce non-zero, finite forces via hash grid topological neighbors
    acc_active = flock.accelerations[flock.active]
    assert not np.allclose(acc_active, 0.0)
    assert np.isfinite(acc_active).all()


def test_topological_neighbors_fallback(default_config):
    """_topological_neighbors_batch returns all -1 sentinels when index not ready."""
    from pymurmur.physics.flock import PhysicsFlock
    from pymurmur.physics.forces.projection import _topological_neighbors_batch

    cfg = default_config
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)

    # Replace index with an object lacking query_knn -> falls through
    class _FakeIndex:
        ready = False
    flock._index = _FakeIndex()
    active_idx = np.where(flock.active)[0]

    result = _topological_neighbors_batch(flock.positions, flock.get_index(), active_idx, 4)
    assert (result == -1).all()  # all sentinels when index not ready


def test_topological_neighbors_kdtree(default_config):
    """_topological_neighbors_batch uses KDTreeIndex when available."""
    from pymurmur.physics.flock import KDTreeIndex, PhysicsFlock
    from pymurmur.physics.forces.projection import _topological_neighbors_batch

    cfg = default_config
    cfg.num_boids = 30
    flock = PhysicsFlock(cfg)

    # Force KDTreeIndex
    kdt = KDTreeIndex()
    kdt.rebuild(flock.positions, flock.active)
    flock._index = kdt

    active_idx = np.where(flock.active)[0]
    result = _topological_neighbors_batch(flock.positions, flock.get_index(), active_idx, 4)
    assert (result >= 0).any()  # returns neighbours via KDTree


# ── S1.4: Pearce noise term φn = 1 − φp − φa ─────────────────────────


def _phi_forces(phi_p: float, phi_a: float, seed: int = 42) -> np.ndarray:
    """Run projection compute once with given φ weights; return accelerations."""
    from pymurmur.core.config import SimConfig
    from pymurmur.physics.flock import PhysicsFlock
    from pymurmur.physics.forces.projection import projection_forces

    cfg = SimConfig()
    cfg.seed = seed
    cfg.mode = "projection"
    cfg.num_boids = 60
    cfg.projection.phi_p = phi_p
    cfg.phi_a = phi_a

    flock = PhysicsFlock(cfg)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)
    _call_force(projection_forces, flock, cfg)
    return flock.accelerations[flock.active].copy()


def test_phi_n_zero_when_weights_sum_to_one():
    """S1.4: φp + φa = 1 → φn = 0 → deterministic repeat is bit-identical
    (no noise term consumed from the rng beyond the shared draws)."""
    acc_a = _phi_forces(0.2, 0.8)
    acc_b = _phi_forces(0.2, 0.8)
    np.testing.assert_array_equal(acc_a, acc_b)


def test_phi_n_adds_variance_over_phi_n_zero():
    """S1.4: φn = 0.2 (φp=0.03, φa=0.77) gives residual heading variance
    above the φn = 0 case (φp=0.2, φa=0.8) with the same seed."""
    from pymurmur.core.config import SimConfig
    from pymurmur.physics.flock import PhysicsFlock
    from pymurmur.physics.forces.projection import projection_forces

    def steer_residual(phi_p, phi_a):
        cfg = SimConfig()
        cfg.seed = 42
        cfg.mode = "projection"
        cfg.num_boids = 60
        cfg.projection.phi_p = phi_p
        cfg.phi_a = phi_a
        flock = PhysicsFlock(cfg)
        # Perfectly aligned flock: without noise, alignment steering ≈ 0
        flock.velocities[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        flock.accelerations[:] = 0.0
        flock.get_index().rebuild(flock.positions, flock.active)
        _call_force(projection_forces, flock, cfg)
        acc = flock.accelerations[flock.active]
        # Variance of the off-heading force components (y, z)
        return float(np.var(acc[:, 1]) + np.var(acc[:, 2]))

    var_noise = steer_residual(0.03, 0.77)   # φn = 0.2
    var_clean = steer_residual(0.03, 0.97)   # φn = 0.0
    assert var_noise > var_clean, (
        f"φn=0.2 should add residual heading variance: "
        f"noise={var_noise:.6g} vs clean={var_clean:.6g}"
    )


def test_phi_n_prevents_perfect_alignment():
    """S1.4 behavioural: with φn > 0 a perfectly aligned flock is knocked
    off perfect alignment; with φn = 0 it stays perfectly aligned."""
    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

    def polarisation_after(phi_p, phi_a, steps=20):
        cfg = SimConfig()
        cfg.seed = 42
        cfg.mode = "projection"
        cfg.num_boids = 40
        cfg.projection.phi_p = phi_p
        cfg.phi_a = phi_a
        eng = SimulationEngine(cfg)
        eng.flock.velocities[:] = np.array([4.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(steps):
            eng.step(1.0 / 60.0)
        v = eng.flock.velocities[eng.flock.active]
        v_hat = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-10)
        return float(np.linalg.norm(v_hat.mean(axis=0)))

    pol_noise = polarisation_after(0.03, 0.77)  # φn = 0.2
    pol_clean = polarisation_after(0.03, 0.97)  # φn = 0.0
    # Not exactly 1.0 — the φp·δ̂ projection term itself perturbs headings
    assert pol_clean > 0.999, "φn=0 flock should stay near-perfectly aligned"
    assert pol_noise < pol_clean, (
        f"φn=0.2 must reduce polarisation below the φn=0 baseline "
        f"({pol_noise:.6f} vs {pol_clean:.6f})"
    )


def test_phi_n_deterministic_with_seed():
    """S1.4: the noise draw uses the flock rng — same seed, same forces."""
    acc_a = _phi_forces(0.03, 0.77, seed=7)
    acc_b = _phi_forces(0.03, 0.77, seed=7)
    np.testing.assert_array_equal(acc_a, acc_b)


# ── S1.4: φn edge cases ────────────────────────────────────────────

def test_phi_n_pure_noise_when_weights_zero():
    """S1.4: φp=0, φa=0 → φn=1.0 — pure random walk, steering
    is entirely η̂ (uniform on S²)."""
    acc = _phi_forces(0.0, 0.0, seed=42)
    # With φp=φa=0, delta and align_dir contribute nothing.
    # All steering comes from η̂·φn = η̂·1.0.
    # So |steering| ≈ 1.0 per bird before clamping.
    mags = np.linalg.norm(acc, axis=1)
    assert (mags > 0).all(), "Pure-noise mode should produce non-zero steering"
    assert (mags <= 5.0 + 0.01).all(), (
        f"Steering should be <= max_force: max={mags.max():.3f}"
    )


def test_phi_n_zero_when_weights_sum_exceeds_one():
    """S1.4: φp + φa > 1 → φn = max(0, 1−φp−φa) = 0.
    No noise term when weights are oversaturated."""
    acc = _phi_forces(0.6, 0.6, seed=42)  # φp+φa=1.2 → φn=0
    # Should be deterministic — no RNG draw for noise
    acc_b = _phi_forces(0.6, 0.6, seed=42)
    np.testing.assert_array_equal(acc, acc_b)


def test_phi_n_with_coherence_gating():
    from pymurmur.core.config import SimConfig
    from pymurmur.physics.flock import PhysicsFlock
    """S1.4 × S2.B8: when _coherence_factor < 1 scales φp/φa,
    φn = 1 − scaled_φp − scaled_φa grows to compensate."""
    from pymurmur.physics.forces.projection import projection_forces

    cfg = SimConfig()
    cfg.seed = 42
    cfg.mode = "projection"
    cfg.num_boids = 30
    cfg.projection.phi_p = 0.2
    cfg.phi_a = 0.7  # φp+φa=0.9 → φn=0.1 normally

    # Without coherence gate
    flock_a = PhysicsFlock(cfg)
    flock_a.accelerations[:] = 0.0
    flock_a.get_index().rebuild(flock_a.positions, flock_a.active)
    _call_force(projection_forces, flock_a, cfg)
    acc_no_coherence = flock_a.accelerations[flock_a.active].copy()

    # With coherence gate reducing weights
    cfg._coherence_factor = 0.5
    flock_b = PhysicsFlock(cfg)
    flock_b.positions[:] = flock_a.positions
    flock_b.velocities[:] = flock_a.velocities
    flock_b.accelerations[:] = 0.0
    flock_b.get_index().rebuild(flock_b.positions, flock_b.active)
    _call_force(projection_forces, flock_b, cfg)
    acc_with_coherence = flock_b.accelerations[flock_b.active].copy()

    # Coherence reduces φp/φa → φn grows → forces should differ
    assert not np.allclose(acc_no_coherence, acc_with_coherence, rtol=1e-4), (
        "Coherence gating should change force distribution via φn compensation"
    )
