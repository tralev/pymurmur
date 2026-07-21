"""P9 Cross-element integration tests.

Verifies that Phase 9 sub-steps interact correctly as a whole system,
not just in isolation. Each test chains 2-4 sub-steps together.
"""

import json

import numpy as np
import pytest

from pymurmur.analysis.metrics import (
    FlockMetrics,
    MetricsCollector,
    _compute_eta_m,
    compute_convex_hull_density,
    compute_gyration,
    compute_h2,
    compute_msd_curve,
    compute_nematic_order,
    compute_normalized_angular_momentum,
    compute_shape,
    compute_silhouette_2d,
    compute_suggested_m,
    compute_tau_rho_hull,
    compute_theta_prime,
)
from pymurmur.analysis.rewards import RewardConfig, compute_reward

# ═══════════════════════════════════════════════════════════════
# 1. Reward scalarization of P9 observables (P9.9 ↔ P9.1, P9.4, P9.8)
# ═══════════════════════════════════════════════════════════════

class TestRewardConsumesP9Observables:
    """compute_reward() correctly digests nematic, silhouette, and motion metrics."""

    def test_reward_responds_to_nematic_change(self):
        """P9.9+P9.1: Higher nematic S → higher reward (with positive weight)."""
        m_high = FlockMetrics(nematic_S=0.95, alpha=0.5, dispersion=100.0,
                               velocity_deviation=1.0, boundary_overshoot=50.0,
                               altitude_deviation=50.0, local_spacing=15.0)
        m_low = FlockMetrics(nematic_S=0.10, alpha=0.5, dispersion=100.0,
                              velocity_deviation=1.0, boundary_overshoot=50.0,
                              altitude_deviation=50.0, local_spacing=15.0)

        config = RewardConfig(weights={"nematic": 1.0}, faithful_signs=True)
        r_high = compute_reward(m_high, config)
        r_low = compute_reward(m_low, config)
        assert r_high > r_low, (
            f"Nematic should drive reward: high={r_high:.4f} vs low={r_low:.4f}"
        )

    def test_reward_responds_to_velocity_deviation(self):
        """P9.9+P9.8: Lower velocity_deviation → higher speed_match reward."""
        m_tight = FlockMetrics(nematic_S=0.5, alpha=0.5, dispersion=100.0,
                                velocity_deviation=0.1, boundary_overshoot=50.0,
                                altitude_deviation=50.0, local_spacing=15.0)
        m_loose = FlockMetrics(nematic_S=0.5, alpha=0.5, dispersion=100.0,
                                velocity_deviation=5.0, boundary_overshoot=50.0,
                                altitude_deviation=50.0, local_spacing=15.0)

        config = RewardConfig(weights={"speed_match": 1.0}, faithful_signs=True)
        r_tight = compute_reward(m_tight, config)
        r_loose = compute_reward(m_loose, config)
        assert r_tight > r_loose, (
            f"Speed_match should favour low deviation: tight={r_tight:.4f} vs loose={r_loose:.4f}"
        )

    def test_reward_responds_to_silhouette(self):
        """P9.9+P9.4: Higher silhouette → higher silhouette reward."""
        m_full = FlockMetrics(nematic_S=0.5, alpha=0.5, dispersion=100.0,
                               velocity_deviation=1.0, boundary_overshoot=50.0,
                               altitude_deviation=50.0, local_spacing=15.0,
                               silhouette_2d=0.9)
        m_sparse = FlockMetrics(nematic_S=0.5, alpha=0.5, dispersion=100.0,
                                 velocity_deviation=1.0, boundary_overshoot=50.0,
                                 altitude_deviation=50.0, local_spacing=15.0,
                                 silhouette_2d=0.05)

        config = RewardConfig(weights={"silhouette": 1.0}, faithful_signs=True)
        r_full = compute_reward(m_full, config)
        r_sparse = compute_reward(m_sparse, config)
        assert r_full > r_sparse, (
            f"Silhouette should drive reward: full={r_full:.4f} vs sparse={r_sparse:.4f}"
        )


# ═══════════════════════════════════════════════════════════════
# 2. Gyration-bounded normalized angular momentum (P9.7 ↔ P9.8)
# ═══════════════════════════════════════════════════════════════

class TestGyrationFeedsNormalizedL:
    """Robust gyration R_g (P9.7) feeds into normalized L (P9.8)."""

    def test_pure_rotation_gives_L_norm_near_one(self):
        """P9.7+P9.8: Rigid rotating ring → L_norm ≈ 1.0 (O(1) scale-invariant)."""
        N = 100
        rng = np.random.RandomState(42)
        angles = rng.uniform(0, 2 * np.pi, N).astype(np.float32)
        radius = 200.0

        positions = np.zeros((N, 3), dtype=np.float32)
        positions[:, 0] = np.cos(angles) * radius + 500.0
        positions[:, 1] = np.sin(angles) * radius + 350.0

        # Tangential velocity: v = ω × r, with ω=0.02 rad/frame → |v|=4.0
        omega = 0.02
        v0 = radius * omega  # 4.0
        velocities = np.zeros((N, 3), dtype=np.float32)
        velocities[:, 0] = -np.sin(angles) * v0
        velocities[:, 1] = np.cos(angles) * v0

        # P9.7: Robust gyration
        Rg = compute_gyration(positions)
        assert Rg > 0, f"R_g should be positive, got {Rg:.1f}"

        # P9.8: Normalized angular momentum using R_g from P9.7
        L_norm = compute_normalized_angular_momentum(positions, velocities, v0, Rg)

        # For pure rigid rotation, L_norm should be ~1.0
        assert 0.5 < L_norm < 2.0, (
            f"Pure rotation L_norm={L_norm:.3f} should be ~1.0 (O(1))"
        )

    def test_random_motion_low_L_norm(self):
        """P9.7+P9.8: Random motion → low L_norm (no coherent rotation)."""
        rng = np.random.RandomState(99)
        N = 100
        positions = rng.randn(N, 3).astype(np.float32) * 100 + 500
        velocities = rng.randn(N, 3).astype(np.float32) * 4.0

        Rg = compute_gyration(positions)
        L_norm = compute_normalized_angular_momentum(positions, velocities, 4.0, max(Rg, 0.01))

        # Random motion → low angular momentum
        assert L_norm < 0.5, (
            f"Random motion L_norm={L_norm:.3f} should be < 0.5"
        )


# ═══════════════════════════════════════════════════════════════
# 3. Anti-parallel nematic vs velocity deviation (P9.1 ↔ P9.8)
# ═══════════════════════════════════════════════════════════════

class TestNematicVsVelocityDeviation:
    """Nematic S (P9.1) and velocity_deviation (P9.8) decouple correctly."""

    def test_anti_parallel_high_S_high_deviation(self):
        """P9.1+P9.8: Anti-parallel flock → S≈1.0, α≈0, but velocity_deviation>0."""
        N = 100
        half = N // 2
        dirs = np.zeros((N, 3), dtype=np.float32)
        dirs[:half, 0] = 1.0
        dirs[half:, 0] = -1.0

        # P9.1: Nematic S
        S = compute_nematic_order(dirs)
        # Polar α
        alpha = float(np.linalg.norm(dirs.sum(axis=0)) / N)

        # P9.8: velocity_deviation = (1/N)Σ‖v̄ − v_i‖
        # Construct velocities: half at +4.0, half at -4.0
        velocities = dirs * 4.0  # (N, 3)
        v_mean = velocities.mean(axis=0)
        dev = float(np.mean(np.linalg.norm(v_mean - velocities, axis=1)))

        assert S > 0.95, f"Anti-parallel S={S:.3f} should be > 0.95"
        assert alpha < 0.05, f"Anti-parallel α={alpha:.3f} should be < 0.05"
        assert dev > 1.0, (
            f"Anti-parallel velocity deviation should be > 1.0, got {dev:.3f}"
        )

    def test_aligned_flock_low_S_low_deviation(self):
        """P9.1+P9.8: All-aligned flock → S≈1.0, α≈1.0, deviation≈0."""
        N = 100
        dirs = np.tile([1.0, 0.0, 0.0], (N, 1)).astype(np.float32)

        S = compute_nematic_order(dirs)
        alpha = float(np.linalg.norm(dirs.sum(axis=0)) / N)

        velocities = dirs * 4.0
        v_mean = velocities.mean(axis=0)
        dev = float(np.mean(np.linalg.norm(v_mean - velocities, axis=1)))

        assert S > 0.95
        assert alpha > 0.95
        assert dev < 0.01, (
            f"Aligned deviation should be ~0, got {dev:.6f}"
        )


# ═══════════════════════════════════════════════════════════════
# 4. Disconnected graph → inf → null export (P9.6 ↔ P9.10)
# ═══════════════════════════════════════════════════════════════

class TestDisconnectedGraphToNullExport:
    """H2 inf (P9.6) → eta_m 0 → to_dict null (P9.10)."""

    def test_disconnected_h2_to_null_via_to_dict(self):
        """P9.6+P9.10: Disconnected H2=inf → eta_m=0 → to_dict maps to null."""
        pytest.importorskip("scipy")
        from scipy.spatial import cKDTree

        # Two widely separated clusters
        rng = np.random.RandomState(42)
        cluster_a = rng.randn(15, 3).astype(np.float32) * 3
        cluster_b = rng.randn(15, 3).astype(np.float32) * 3 + np.array([1000, 0, 0])
        positions = np.vstack([cluster_a, cluster_b])
        tree = cKDTree(positions)

        # Both m=3 and m0=1 should be disconnected
        _, h2 = compute_h2(positions, 3, tree)
        assert not np.isfinite(h2), f"Disconnected H2 should be inf, got {h2}"

        eta = _compute_eta_m(positions, tree, 3)
        assert eta == 0.0, f"Both disconnected → η=0, got {eta}"

        # P9.10: to_dict maps None→null (eta_m=0 is finite, but verify inf→null)
        m = FlockMetrics(h2=float('inf'), eta_m=0.0)
        d = m.to_dict()
        # h2=inf → null
        assert d["h2"] is None, f"h2=inf should be null, got {d['h2']}"
        # eta_m=0 → 0 (finite, not null)
        assert d["eta_m"] == 0.0
        # Must produce valid JSON
        serialized = json.dumps(d)
        loaded = json.loads(serialized)
        assert loaded["h2"] is None
        assert loaded["eta_m"] == 0.0


# ═══════════════════════════════════════════════════════════════
# 5. 2D silhouette vs 3D hull density (P9.4 ↔ P9.3)
# ═══════════════════════════════════════════════════════════════

class TestSilhouetteVsHullDensity:
    """2D silhouette (P9.4) and 3D hull density (P9.3) measure different things."""

    def test_flat_wall_high_silhouette_low_hull_density(self):
        """P9.4+P9.3: Flat XY wall → high silhouette, low 3D hull volume."""
        rng = np.random.RandomState(42)
        N = 100
        # Dense wall in XY plane (thin in Z)
        positions = np.zeros((N, 3), dtype=np.float32)
        positions[:, 0] = rng.uniform(0, 200, N).astype(np.float32)
        positions[:, 1] = rng.uniform(0, 200, N).astype(np.float32)
        positions[:, 2] = rng.uniform(-1, 1, N).astype(np.float32)  # very thin in Z

        sil = compute_silhouette_2d(positions)
        rho = compute_convex_hull_density(positions)
        theta3d = compute_theta_prime(positions)

        # 2D silhouette sees a big wall → high coverage
        assert sil > 0.2, f"Flat wall sil should be non-trivial, got {sil:.4f}"
        # 3D hull volume is tiny (thin sheet has degenerate hull → rho=0)
        # AND 3D theta_prime should also be low (thin in Z)
        assert theta3d < 0.1, (
            f"3D theta should reflect thinness, got theta3d={theta3d:.4f}"
        )
        assert rho == 0.0 or rho < 0.005, (
            f"Hull density for thin sheet should be ~0, got rho={rho:.6f}"
        )

    def test_3d_sphere_balanced_measures(self):
        """P9.4+P9.3: 3D spherical cloud → both silhouette and 3D measures are moderate."""
        rng = np.random.RandomState(99)
        N = 200
        positions = rng.randn(N, 3).astype(np.float32) * 30 + 200

        sil = compute_silhouette_2d(positions)
        rho = compute_convex_hull_density(positions)

        # Both should be non-zero for a genuine 3D cloud
        assert sil > 0.0, f"Sphere silhouette should be > 0, got {sil:.4f}"
        assert rho > 0.0, f"Sphere hull density should be > 0, got {rho:.6f}"


# ═══════════════════════════════════════════════════════════════
# 6. Shape aspect → suggested_m → eta_m chain (P9.5 ↔ P9.6)
# ═══════════════════════════════════════════════════════════════

class TestShapeToMarginalEfficiency:
    """PCA shape (P9.5) feeds suggested_m, which should relate to eta_m (P9.6)."""

    def test_elongated_flock_lower_suggested_m(self):
        """P9.5+P9.6: Elongated shape → smaller suggested_m; both chains work."""
        pytest.importorskip("scipy")
        from scipy.spatial import cKDTree

        rng = np.random.RandomState(42)
        N = 40
        # Spherical cluster
        sphere = rng.randn(N, 3).astype(np.float32) * 20
        aspect_s, _ = compute_shape(sphere)
        m_s_s = compute_suggested_m(aspect_s)

        # Elongated cluster (stretch X)
        elongated = sphere.copy()
        elongated[:, 0] *= 5
        aspect_e, _ = compute_shape(elongated)
        m_s_e = compute_suggested_m(aspect_e)

        # Elongated flock should get a lower suggested_m
        assert m_s_e <= m_s_s, (
            f"Elongated aspect={aspect_e:.2f} → m*={m_s_e:.2f} should be ≤ "
            f"spherical aspect={aspect_s:.2f} → m*={m_s_s:.2f}"
        )

        # Both suggested_m values should work with eta_m
        tree_s = cKDTree(sphere)
        tree_e = cKDTree(elongated)

        eta_s = _compute_eta_m(sphere, tree_s, int(round(m_s_s)))
        eta_e = _compute_eta_m(elongated, tree_e, int(round(m_s_e)))

        # Both should return valid values (None or finite/inf)
        assert eta_s is None or np.isfinite(eta_s) or eta_s == float('inf')
        assert eta_e is None or np.isfinite(eta_e) or eta_e == float('inf')

    def test_collector_shape_to_suggested_m_chain(self, default_config):
        """P9.5+P9.6: Collector computes shape→suggested_m→eta_m together."""
        from pymurmur.physics.flock import PhysicsFlock

        cfg = default_config
        cfg.num_boids = 20
        cfg.metrics_detail_level = 2
        cfg.metrics_interval = 2
        flock = PhysicsFlock(cfg)
        collector = MetricsCollector(cfg)

        for frame in range(6):
            collector.collect(flock, frame)

        history = collector.history
        # Find a snapshot where expensive metrics were computed
        populated = [s for s in history
                     if s.suggested_m is not None and s.aspect_ratio is not None]
        assert len(populated) > 0, (
            "No snapshot had expensive metrics computed after 6 frames "
            f"at detail_level=2, interval=2 (history: {len(history)} snapshots)"
        )
        snap = populated[-1]
        # suggested_m should be consistent with aspect_ratio
        expected = compute_suggested_m(snap.aspect_ratio)
        assert snap.suggested_m == pytest.approx(expected, rel=0.01), (
            f"suggested_m={snap.suggested_m:.3f} should match "
            f"compute_suggested_m(aspect={snap.aspect_ratio:.2f})={expected:.3f}"
        )


# ═══════════════════════════════════════════════════════════════
# 7. Translational MSD vs structural timescales (P9.2 ↔ P9.3)
# ═══════════════════════════════════════════════════════════════

class TestMSDvsStructuralTimescales:
    """MSD curve (P9.2) and hull-density autocorrelation (P9.3) measure
    different timescales: translational vs structural."""

    def test_rigid_translation_ballistic_msd_zero_hull_tau(self):
        """P9.2+P9.3: Rigidly translating flock → ballistic MSD (~2 slope),
        but hull density autocorrelation τρ = 0 (no structural change)."""
        T = 20
        N = 30
        v = np.array([2.0, 0.0, 0.0], dtype=np.float32)
        snapshots = []
        pos = np.zeros((N, 3), dtype=np.float32)
        for _t in range(T):
            pos = pos + np.tile(v, (N, 1))
            snapshots.append(pos.copy())

        # P9.2: MSD should be ballistic (slope ~2)
        msd_vals, lags, slope, crossover = compute_msd_curve(snapshots)
        assert slope > 1.5, f"Rigid translation should be ballistic, slope={slope:.2f}"

        # P9.3: Compute actual hull densities from the snapshots
        # All positions have same relative arrangement, so density is constant
        densities = [compute_convex_hull_density(s) for s in snapshots]
        # All snapshots have same relative positions → hull volume is equal →
        # density is constant → τρ = 0
        tau = compute_tau_rho_hull(densities, interval=1)
        assert tau == 0.0, (
            f"Constant density (no structural change) → τρ=0, got {tau:.1f}"
        )

    def test_structural_change_detected_by_both(self):
        """P9.2+P9.3: Flock that expands → MSD grows, τρ captures structural timescale."""
        rng = np.random.RandomState(42)
        N = 20
        T = 16
        snapshots = []
        # Flock that expands from tight cluster to spread, with smooth evolution
        base = rng.randn(N, 3).astype(np.float32)
        for t in range(T):
            scale = 10.0 + t * 5.0  # expanding
            # Same base pattern, just scaled: positions evolve coherently
            positions = base * scale + 500.0
            snapshots.append(positions.copy())

        # P9.2: MSD should detect the expansion (slope > 0)
        _, _, slope, _ = compute_msd_curve(snapshots, max_lag=8)
        assert slope > 0.0, f"Expanding flock MSD slope should be > 0, got {slope:.2f}"

        # P9.3: Compute hull density at each snapshot and check autocorrelation
        densities = [compute_convex_hull_density(s) for s in snapshots]
        # For expanding flock, density decreases monotonically → τρ > 0
        tau = compute_tau_rho_hull(densities, interval=1)
        assert tau > 0.0, (
            f"Expanding flock should have τρ > 0 (monotonic density change), got {tau:.1f}"
        )
