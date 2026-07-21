"""T4.4 metamorphic invariances — the matrix roadmap1.md's T4.4 spec asked
for beyond the nematic rotation/sign-flip pair already covered in
test_metrics.py: alpha rotation-invariance, dispersion/gyration
translation-invariance, permutation invariance, and a [0,1]-bounds sweep.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymurmur.analysis.metrics import (
    MetricsCollector,
    compute_gyration,
    compute_nematic_order,
    compute_silhouette_2d,
    compute_theta_prime,
)
from pymurmur.core.config import SimConfig
from pymurmur.physics.flock import PhysicsFlock


def _random_rotation(rng: np.random.RandomState) -> np.ndarray:
    """A uniformly-random rotation matrix via QR decomposition of a
    Gaussian matrix, sign-corrected to det = +1 (proper rotation)."""
    a = rng.randn(3, 3)
    q, r = np.linalg.qr(a)
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q.astype(np.float32)


def _flock_from(positions: np.ndarray, velocities: np.ndarray) -> PhysicsFlock:
    n = len(positions)
    cfg = SimConfig()
    cfg.num_boids = n
    flock = PhysicsFlock(cfg)
    flock.positions = positions.astype(np.float32)
    flock.velocities = velocities.astype(np.float32)
    flock.active = np.ones(n, dtype=bool)
    flock.last_accelerations = np.zeros((n, 3), dtype=np.float32)
    return flock


def _alpha_and_dispersion(positions: np.ndarray, velocities: np.ndarray) -> tuple[float, float]:
    collector = MetricsCollector()
    collector.collect(_flock_from(positions, velocities), 0)
    snap = collector.snapshot()
    return snap.alpha, snap.dispersion


# ── Rotation invariance (α, nematic S) ──────────────────────────────

def test_alpha_rotation_invariant_SO3():
    """α = |Σv̂|/N is unchanged under a random SO(3) rotation of velocities."""
    rng = np.random.RandomState(11)
    for trial in range(30):
        N = rng.randint(10, 100)
        positions = rng.uniform(0, 500, (N, 3)).astype(np.float32)
        velocities = rng.randn(N, 3).astype(np.float32)
        R = _random_rotation(rng)

        alpha_before, _ = _alpha_and_dispersion(positions, velocities)
        alpha_after, _ = _alpha_and_dispersion(positions, (R @ velocities.T).T)

        assert alpha_before == pytest.approx(alpha_after, abs=1e-4), (
            f"trial {trial}: alpha not rotation-invariant "
            f"({alpha_before} vs {alpha_after})"
        )


def test_nematic_S_rotation_invariant_SO3_random_matrices():
    """compute_nematic_order is SO(3)-invariant across 30 random rotations
    (test_metrics.py already covers one fixed composed rotation)."""
    rng = np.random.RandomState(12)
    for trial in range(30):
        N = rng.randint(10, 200)
        dirs = rng.randn(N, 3).astype(np.float32)
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        dirs /= norms
        R = _random_rotation(rng)

        S_before = compute_nematic_order(dirs)
        S_after = compute_nematic_order((R @ dirs.T).T.astype(np.float32))

        assert S_before == pytest.approx(S_after, abs=1e-3), (
            f"trial {trial}: nematic S not rotation-invariant "
            f"({S_before} vs {S_after})"
        )


# ── Translation invariance (dispersion, gyration) ───────────────────

def test_dispersion_translation_invariant():
    """dispersion = <|r - r_com|> is unchanged by translating every position."""
    rng = np.random.RandomState(21)
    for trial in range(30):
        N = rng.randint(10, 100)
        positions = rng.uniform(0, 500, (N, 3)).astype(np.float32)
        velocities = rng.randn(N, 3).astype(np.float32)
        shift = rng.uniform(-1000, 1000, 3).astype(np.float32)

        _, disp_before = _alpha_and_dispersion(positions, velocities)
        _, disp_after = _alpha_and_dispersion(positions + shift, velocities)

        assert disp_before == pytest.approx(disp_after, rel=1e-4, abs=1e-4), (
            f"trial {trial}: dispersion not translation-invariant "
            f"({disp_before} vs {disp_after})"
        )


def test_gyration_translation_invariant():
    """compute_gyration (median-centroid, trimmed RMS) is translation-invariant."""
    rng = np.random.RandomState(22)
    for trial in range(30):
        N = rng.randint(10, 100)
        positions = rng.uniform(0, 500, (N, 3)).astype(np.float32)
        shift = rng.uniform(-1000, 1000, 3).astype(np.float32)

        rg_before = compute_gyration(positions)
        rg_after = compute_gyration(positions + shift)

        assert rg_before == pytest.approx(rg_after, rel=1e-4, abs=1e-4), (
            f"trial {trial}: gyration not translation-invariant "
            f"({rg_before} vs {rg_after})"
        )


# ── Permutation invariance ───────────────────────────────────────────

def test_metrics_permutation_invariant():
    """Shuffling bird order changes none of alpha/nematic_S/dispersion/gyration."""
    rng = np.random.RandomState(31)
    for trial in range(30):
        N = rng.randint(10, 100)
        positions = rng.uniform(0, 500, (N, 3)).astype(np.float32)
        velocities = rng.randn(N, 3).astype(np.float32)
        perm = rng.permutation(N)

        alpha_before, disp_before = _alpha_and_dispersion(positions, velocities)
        alpha_after, disp_after = _alpha_and_dispersion(positions[perm], velocities[perm])
        rg_before = compute_gyration(positions)
        rg_after = compute_gyration(positions[perm])

        assert alpha_before == pytest.approx(alpha_after, abs=1e-5), trial
        assert disp_before == pytest.approx(disp_after, rel=1e-5, abs=1e-5), trial
        assert rg_before == pytest.approx(rg_after, rel=1e-5, abs=1e-5), trial


# ── [0,1]-bounds sweep ────────────────────────────────────────────────

def test_bounds_sweep_metrics_in_0_1():
    """alpha, nematic_S, theta_prime, silhouette_2d all stay within [0, 1]
    across randomized flock configurations (positions, velocities, N)."""
    rng = np.random.RandomState(41)
    for trial in range(30):
        N = rng.randint(5, 300)
        positions = rng.uniform(0, rng.uniform(50, 2000), (N, 3)).astype(np.float32)
        velocities = rng.randn(N, 3).astype(np.float32) * rng.uniform(0.1, 20)

        alpha, _ = _alpha_and_dispersion(positions, velocities)
        norms = np.linalg.norm(velocities, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        S = compute_nematic_order(velocities / norms)
        theta_prime = compute_theta_prime(positions)
        silhouette = compute_silhouette_2d(positions)

        assert 0.0 <= alpha <= 1.0 + 1e-6, f"trial {trial}: alpha={alpha}"
        assert 0.0 <= S <= 1.0 + 1e-6, f"trial {trial}: nematic_S={S}"
        assert 0.0 <= theta_prime <= 1.0 + 1e-6, f"trial {trial}: theta_prime={theta_prime}"
        assert 0.0 <= silhouette <= 1.0 + 1e-6, f"trial {trial}: silhouette_2d={silhouette}"
