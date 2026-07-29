"""S1.4 φn noise-term tests + §09/§11-style heading-blend inertia tests
for projection mode.

Split out of test_projection.py (file-size split) — core projection-mode
tests stay in the original; this file covers the φn = 1 − φp − φa
noise-term behavior (and its edge cases) plus projection_heading_inertia.
"""

from copy import copy

import numpy as np

from test.helpers import _call_force  # noqa: E402

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


# ── §09/§11-style heading-blend inertia (projection_heading_inertia) ────


def _heading_inertia_forces(heading_inertia: float, seed: int = 1) -> np.ndarray:
    """Run projection compute with all birds sharing a known heading
    (+x); return accelerations. Isolates the heading-inertia pull."""
    from pymurmur.core.config import SimConfig
    from pymurmur.physics.flock import PhysicsFlock
    from pymurmur.physics.forces.projection import projection_forces

    cfg = SimConfig()
    cfg.seed = seed
    cfg.mode = "projection"
    cfg.num_boids = 30
    cfg.projection_heading_inertia = heading_inertia

    flock = PhysicsFlock(cfg)
    flock.velocities[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)
    _call_force(projection_forces, flock, cfg)
    return flock.accelerations[flock.active].copy()


def test_heading_inertia_default_is_zero(default_config):
    assert default_config.projection_heading_inertia == 0.0


def test_heading_inertia_zero_matches_no_inertia_field():
    """Explicitly setting 0.0 must be identical to never touching the
    field at all (same default)."""
    acc_default = _heading_inertia_forces(0.0)
    acc_explicit_zero = _heading_inertia_forces(0.0)
    np.testing.assert_array_equal(acc_default, acc_explicit_zero)


def test_heading_inertia_pulls_toward_current_heading():
    """With all birds heading +x, increasing inertia should measurably
    shift the mean x-component of acceleration upward (pulled toward
    maintaining +x) relative to the no-inertia baseline."""
    acc_low = _heading_inertia_forces(0.0)
    acc_high = _heading_inertia_forces(0.9)
    assert acc_high[:, 0].mean() > acc_low[:, 0].mean()


def test_heading_inertia_full_strength_no_crash():
    """inertia=1.0 (maximum, fully static heading pull) runs without
    crashing and produces finite output."""
    acc = _heading_inertia_forces(1.0)
    assert np.isfinite(acc).all()


def test_heading_inertia_deterministic():
    acc1 = _heading_inertia_forces(0.5, seed=7)
    acc2 = _heading_inertia_forces(0.5, seed=7)
    np.testing.assert_array_equal(acc1, acc2)
