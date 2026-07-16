"""Influencer mode tests — Phase 8.3

Lissajous target follow, no neighbour queries, rank-based influence.
"""

import numpy as np
import pytest

from pymurmur.core.config import SimConfig
from pymurmur.physics.forces.influencer import influencer_forces
from pymurmur.physics.flock import PhysicsFlock


from test.helpers import _call_force  # noqa: E402


class TestInfluencerMode:
    """JerBoon cosmic influencer — Lissajous target, no neighbours."""

    # ── Core behaviour ──────────────────────────────────────────

    def test_produces_nonzero_forces(self):
        """Influencer produces non-zero accelerations on all birds."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 50
        cfg.influencer_substeps = 3

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        _call_force(influencer_forces, flock, cfg)

        acc_mags = np.linalg.norm(flock.accelerations[flock.active], axis=1)
        assert np.all(acc_mags > 1e-6), (
            f"Not all birds felt force: {np.sum(acc_mags > 1e-6)}/{len(acc_mags)}"
        )

    def test_birds_pulled_toward_target(self):
        """All birds pulled toward the same Lissajous target (consistent directions)."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 30
        cfg.influencer_substeps = 1

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0

        np.random.seed(42)
        _call_force(influencer_forces, flock, cfg)

        # All birds pulled toward same target → acc directions should be consistent
        active_idx = np.where(flock.active)[0]
        dirs = flock.accelerations[active_idx]
        norms = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-10
        dirs = dirs / norms

        # Pairwise dot products should be positive (all point roughly same way)
        pairwise = np.dot(dirs, dirs.T)
        assert np.all(pairwise > -0.1), (
            f"Bird directions inconsistent — some oppose each other: "
            f"min dot = {pairwise.min():.4f}"
        )

    def test_closer_birds_have_more_influence(self):
        """Birds closer to CoM feel stronger pull toward target."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 100
        cfg.influencer_substeps = 1
        cfg.influencer_rank_exponent = 2.0

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0

        np.random.seed(42)
        _call_force(influencer_forces, flock, cfg)

        active_pos = flock.positions[flock.active]
        com = np.mean(active_pos, axis=0)
        dists = np.linalg.norm(active_pos - com, axis=1)

        # Top 30% closest vs bottom 30% farthest
        n = len(dists)
        close_idx = np.argsort(dists)[: n // 3]
        far_idx = np.argsort(dists)[-n // 3:]

        acc_mags = np.linalg.norm(flock.accelerations[flock.active], axis=1)
        close_avg = np.mean(acc_mags[close_idx])
        far_avg = np.mean(acc_mags[far_idx])

        assert close_avg > far_avg, (
            f"Closer birds should feel more force: close={close_avg:.4f}, far={far_avg:.4f}"
        )

    def test_no_neighbour_queries(self):
        """Influencer mode never queries the spatial index — purely per-bird."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 50

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0

        # Mock the index to verify it's never queried
        class SpyIndex:
            def __init__(self): self.ready = True
            def query_knn(self, *a, **kw): raise RuntimeError("Should not be called")
            def query_radius(self, *a, **kw): raise RuntimeError("Should not be called")
            def rebuild(self, *a, **kw): pass

        flock._index = SpyIndex()
        _call_force(influencer_forces, flock, cfg)  # should not crash

        acc_mags = np.linalg.norm(flock.accelerations[flock.active], axis=1)
        assert np.all(acc_mags > 1e-6)

    # ── Edge cases ──────────────────────────────────────────────

    def test_zero_active(self):
        """Empty flock produces no change."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 10
        flock = PhysicsFlock(cfg)
        flock.active[:] = False

        old_acc = flock.accelerations.copy()
        _call_force(influencer_forces, flock, cfg)
        assert np.allclose(flock.accelerations, old_acc)

    def test_single_bird(self):
        """Single bird: rank=0 → influence=1.0, pulled to target."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 1
        cfg.influencer_substeps = 1

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0

        np.random.seed(42)
        _call_force(influencer_forces, flock, cfg)

        acc_mag = np.linalg.norm(flock.accelerations[0])
        assert acc_mag > 0

    def test_force_clamped_to_max(self):
        """No acceleration exceeds config.max_force."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 50
        cfg.influencer_substeps = 10
        cfg.max_force = 2.0

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        _call_force(influencer_forces, flock, cfg)

        acc_mags = np.linalg.norm(flock.accelerations[flock.active], axis=1)
        assert np.all(acc_mags <= cfg.max_force + 1e-6), (
            f"Max force exceeded: {acc_mags.max():.3f} > {cfg.max_force}"
        )

    def test_substeps_multiply_force(self):
        """More substeps → proportionally larger total force."""
        cfg1 = SimConfig()
        cfg1.mode = "influencer"
        cfg1.num_boids = 30

        cfg2 = SimConfig()
        cfg2.mode = "influencer"
        cfg2.num_boids = 30

        np.random.seed(42)
        flock1 = PhysicsFlock(cfg1)
        flock1.accelerations[:] = 0.0
        cfg1.influencer_substeps = 1
        _call_force(influencer_forces, flock1, cfg1)
        mag1 = np.linalg.norm(flock1.accelerations[flock1.active], axis=1).mean()

        np.random.seed(42)
        flock2 = PhysicsFlock(cfg2)
        flock2.accelerations[:] = 0.0
        cfg2.influencer_substeps = 3
        _call_force(influencer_forces, flock2, cfg2)
        mag2 = np.linalg.norm(flock2.accelerations[flock2.active], axis=1).mean()

        # Each substep adds force — 3 substeps should give >2x the force
        ratio = mag2 / mag1
        assert 1.5 < ratio < 4.5, f"Substep scaling: mag2/mag1 = {ratio:.2f}"

    def test_rank_exponent_zero_equal_influence(self):
        """rank_exp=0 → all birds get equal influence regardless of distance."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 50
        cfg.influencer_substeps = 1
        cfg.influencer_rank_exponent = 0.0  # (1-rank)^0 = 1 for all

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        np.random.seed(42)
        _call_force(influencer_forces, flock, cfg)

        acc_mags = np.linalg.norm(flock.accelerations[flock.active], axis=1)
        # All birds should have similar force magnitudes (same influence)
        std_dev = np.std(acc_mags)
        avg = np.mean(acc_mags)
        assert std_dev < avg * 0.1, (
            f"Force magnitudes vary too much: std={std_dev:.6f}, avg={avg:.6f}"
        )

    def test_inactive_birds_unchanged(self):
        """Inactive birds get zero force while active ones are pulled."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 30
        cfg.influencer_substeps = 2

        flock = PhysicsFlock(cfg)
        flock.active[10:20] = False
        flock.accelerations[:] = 0.0
        old_acc_inactive = flock.accelerations[~flock.active].copy()

        _call_force(influencer_forces, flock, cfg)

        assert np.allclose(flock.accelerations[~flock.active], old_acc_inactive)
        acc_mags = np.linalg.norm(flock.accelerations[flock.active], axis=1)
        assert np.any(acc_mags > 1e-6)

    def test_substeps_zero(self):
        """substeps=0 → no force applied, no crash."""
        cfg = SimConfig()
        cfg.mode = "influencer"
        cfg.num_boids = 30
        cfg.influencer_substeps = 0

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        _call_force(influencer_forces, flock, cfg)

        assert np.allclose(flock.accelerations, 0.0)
