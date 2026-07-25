"""EvoFlock objective-function and per-step collector tests.

Split out of test_evoflock.py (file-size split), mirroring the
production evoflock.py -> evoflock_objectives.py split: covers
_linear_ramp/_trapezoid (scoring helpers) and _ObjectiveCollector
(per-boid-step objective sampling).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pymurmur.analysis.evoflock import EvoConfig, EvoFlock
from pymurmur.analysis.evoflock_objectives import (
    _linear_ramp,
    _ObjectiveCollector,
    _trapezoid,
)
from pymurmur.core.config import SimConfig


class TestObjectiveFunctions:
    """Scoring helpers — linear ramp and P11.3 trapezoid."""

    def test_linear_ramp_in_range(self):
        """x in [lo, hi] -> score = 1.0."""
        assert _linear_ramp(3.0, 2.0, 4.0, 1.0, 8.0) == 1.0
        assert _linear_ramp(2.0, 2.0, 4.0, 1.0, 8.0) == 1.0

    def test_linear_ramp_below_lo(self):
        """x < lo -> linear ramp from floor to 0."""
        score = _linear_ramp(1.5, 2.0, 4.0, 1.0, 8.0)
        assert 0.0 < score < 1.0

    def test_linear_ramp_above_hi(self):
        """x > hi -> linear ramp to 0 at ceiling."""
        score = _linear_ramp(6.0, 2.0, 4.0, 1.0, 8.0)
        assert 0.0 < score < 1.0

    def test_linear_ramp_at_floor(self):
        """x == floor -> score = 0."""
        assert _linear_ramp(1.0, 2.0, 4.0, 1.0, 8.0) == pytest.approx(0.0)

    def test_linear_ramp_at_ceiling(self):
        """x == ceiling -> score = 0."""
        assert _linear_ramp(8.0, 2.0, 4.0, 1.0, 8.0) == pytest.approx(0.0)

    def test_trapezoid_pinned_values(self):
        """P11.3: separation trapezoid pinned at d/body ∈ {1.9→0, 2.5→1, 4→1, 5→0}."""
        x = np.array([1.9, 2.5, 4.0, 5.0])
        scores = _trapezoid(x, 2.0, 2.5, 4.0, 5.0)
        assert scores == pytest.approx([0.0, 1.0, 1.0, 0.0])

    def test_trapezoid_ramp_midpoints(self):
        """Ramps are linear: midpoints score 0.5."""
        assert _trapezoid(2.25, 2.0, 2.5, 4.0, 5.0) == pytest.approx(0.5)
        assert _trapezoid(4.5, 2.0, 2.5, 4.0, 5.0) == pytest.approx(0.5)

    def test_speed_band_pinned_values(self):
        """P11.3: speed band [19,21] m/s with ramps [18,22]."""
        x = np.array([18.0, 19.0, 20.0, 21.0, 22.0])
        scores = _trapezoid(x, 18.0, 19.0, 21.0, 22.0)
        assert scores == pytest.approx([0.0, 1.0, 1.0, 1.0, 0.0])

    def test_curvature_score_formula(self):
        """P11.3: curvature score = clamp(0.8 + (κ/0.1)·0.2, 0.8, 1.0).
        κ=0 → 0.8, κ=0.05 → 0.9, κ=0.15 → clipped to 1.0."""
        # Test through _compute_objectives with synthesised kappa values
        cfg = SimConfig()
        evo = EvoFlock(cfg, EvoConfig())
        col = _ObjectiveCollector(cfg)
        col.n_steps = 10
        col.collision_free_steps = 10

        # κ = 0 → curv = 0.8 + 0*0.2 = 0.8
        col.kappas.append(np.zeros(5))
        _, _, curv_flat, _ = evo._compute_objectives(col)
        assert curv_flat == pytest.approx(0.8)

        # κ = 0.05 → curv = 0.8 + 0.5*0.2 = 0.9
        col.kappas.clear()
        col.kappas.append(np.full(5, 0.05))
        _, _, curv_mid, _ = evo._compute_objectives(col)
        assert curv_mid == pytest.approx(0.9, abs=1e-6)

        # κ = 0.15 → curv = 0.8 + 1.5*0.2 = 1.1 → clipped to 1.0
        col.kappas.clear()
        col.kappas.append(np.full(5, 0.15))
        _, _, curv_high, _ = evo._compute_objectives(col)
        assert curv_high == pytest.approx(1.0)





def _fake_engine(positions, velocities, accelerations):
    """Minimal engine stand-in for _ObjectiveCollector tests."""
    n = len(positions)
    flock = SimpleNamespace(
        active=np.ones(n, dtype=bool),
        positions=np.asarray(positions, dtype=np.float32),
        prev_positions=np.asarray(positions, dtype=np.float32).copy(),
        velocities=np.asarray(velocities, dtype=np.float32),
        last_accelerations=np.asarray(accelerations, dtype=np.float32),
    )
    return SimpleNamespace(flock=flock)


class TestObjectiveCollector:
    """P11.3: per-boid-step objective sampling."""

    def _config(self, boid_size=0.5, v0=20.0, cruise=20.0):
        cfg = SimConfig()
        cfg.boid_size = boid_size  # body diameter = 1.0
        cfg.v0 = v0
        cfg.cruise_speed_ms = cruise
        return cfg

    def test_nn_distance_in_body_diameters(self):
        """NN distances are recorded per boid-step in body diameters."""
        cfg = self._config()
        col = _ObjectiveCollector(cfg)
        pos = [[0, 0, 0], [3, 0, 0], [7, 0, 0]]
        vel = np.full((3, 3), [20.0, 0, 0])
        col(_fake_engine(pos, vel, np.zeros((3, 3))))
        assert col.n_steps == 1
        # body diameter 1.0 → ratios equal raw NN distances [3, 3, 4]
        assert col.nn_ratios[0] == pytest.approx([3.0, 3.0, 4.0])

    def test_speed_real_conversion(self):
        """speed_real = |v| · cruise_speed_ms / v0."""
        cfg = self._config(v0=4.0, cruise=8.0)  # ×2 conversion
        col = _ObjectiveCollector(cfg)
        vel = [[4.0, 0, 0], [0, 3.0, 0]]
        col(_fake_engine([[0, 0, 0], [50, 0, 0]], vel, np.zeros((2, 3))))
        assert col.speeds_real[0] == pytest.approx([8.0, 6.0])

    def test_helix_curvature_matches_analytic(self):
        """P11.3: κ = |v×a|/|v|³ on a helix matches R/(R²+b²) within 2%."""
        R, b = 5.0, 2.0  # helix (R cos t, R sin t, b t), ω = 1
        kappa_analytic = R / (R * R + b * b)
        cfg = self._config()
        col = _ObjectiveCollector(cfg)
        ts = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
        pos = np.stack([R * np.cos(ts), R * np.sin(ts), b * ts], axis=1)
        vel = np.stack([-R * np.sin(ts), R * np.cos(ts), np.full_like(ts, b)], axis=1)
        acc = np.stack([-R * np.cos(ts), -R * np.sin(ts), np.zeros_like(ts)], axis=1)
        col(_fake_engine(pos, vel, acc))
        kappa_measured = float(np.mean(np.concatenate(col.kappas)))
        assert kappa_measured == pytest.approx(kappa_analytic, rel=0.02)

    def test_compute_objectives_empty_collector(self):
        """No samples → sep 0, speed 0, curvature floor, obstacle perfect."""
        cfg = SimConfig()
        evo = EvoFlock(cfg, EvoConfig())
        col = _ObjectiveCollector(cfg)
        col.n_steps = 1
        col.collision_free_steps = 1
        sep, speed, curv, obst = evo._compute_objectives(col)
        assert sep == 0.0
        assert speed == 0.0
        assert curv == pytest.approx(0.8)
        assert obst == pytest.approx(1.0)

    def test_obstacle_score_penalises_collisions(self):
        """(f_cf)^500 crushes even a 1% collision rate."""
        cfg = SimConfig()
        evo = EvoFlock(cfg, EvoConfig())
        col = _ObjectiveCollector(cfg)
        col.n_steps = 1000
        col.collision_free_steps = 990  # f_cf = 0.99
        *_, obst = evo._compute_objectives(col)
        assert obst < 0.01  # 0.99^500 ≈ 0.0066

        col.collision_free_steps = 1000
        *_, obst = evo._compute_objectives(col)
        assert obst == pytest.approx(1.0)

    def test_collector_counts_collisions_with_scene(self):
        """P11.4: a bird crossing an obstacle surface is counted and corrected."""
        from pymurmur.physics.obstacles import ObstacleScene

        cfg = self._config()
        scene = ObstacleScene().add_sphere([0.0, 0.0, 0.0], 5.0)
        col = _ObjectiveCollector(cfg, scene=scene)

        engine = _fake_engine(
            [[3.0, 0, 0], [20.0, 0, 0]],   # bird 0 inside the sphere
            np.full((2, 3), [20.0, 0, 0]),
            np.zeros((2, 3)),
        )
        engine.flock.prev_positions = np.array(
            [[7.0, 0, 0], [20.0, 0, 0]], dtype=np.float32,
        )  # bird 0 was outside → sign flip
        col(engine)
        assert col.n_steps == 1
        assert col.collision_free_steps == 0
        assert scene.collision_count == 1
        # Kinematic correction pushed bird 0 back to the surface
        assert scene.sdf(engine.flock.positions[:1])[0] >= -1e-3

    def test_settled_uses_last_half(self):
        """P11.3: The settled() helper in _compute_objectives uses only
        the last 50% of collected data chunks."""
        cfg = SimConfig()
        evo = EvoFlock(cfg, EvoConfig())
        body_dia = 1.0
        col = _ObjectiveCollector(cfg)
        col._body_diameter = body_dia
        col.n_steps = 10
        col.collision_free_steps = 10

        # Collect 10 steps: first 5 have NN ratio 10 (far apart → score=0),
        # last 5 have NN ratio 3 (optimal → score=1)
        for _ in range(5):
            col.nn_ratios.append(np.full(5, 10.0))
        for _ in range(5):
            col.nn_ratios.append(np.full(5, 3.0))

        sep, _, _, _ = evo._compute_objectives(col)
        # If all 10 steps used: mean trapezoid would include 0s from first 5
        # If only last 5 used: mean trapezoid = all 1.0s → sep = 1.0
        # Since settled uses last half (len=10, start=5), sep should be 1.0
        assert sep == pytest.approx(1.0), (
            f"Settled sep should be 1.0 (last-half only), got {sep}"
        )

    def test_compute_objectives_with_data(self):
        """P11.3: _compute_objectives produces valid scores from a
        populated collector with nn_ratios, speeds_real, and kappas."""
        cfg = SimConfig()
        cfg.boid_size = 0.5  # body diameter = 1.0
        cfg.v0 = 20.0
        cfg.cruise_speed_ms = 20.0
        evo = EvoFlock(cfg, EvoConfig())
        col = _ObjectiveCollector(cfg)
        col.n_steps = 4
        col.collision_free_steps = 4

        # Good separation: NN dist ~3 body diameters → trapezoid(3,2,2.5,4,5) = 1.0
        col.nn_ratios.append(np.array([3.0, 3.0, 3.0]))
        col.nn_ratios.append(np.array([3.0, 3.0, 3.0]))
        # Good speed: 20 m/s → trapezoid(20,18,19,21,22) = 1.0
        col.speeds_real.append(np.array([20.0, 20.0, 20.0]))
        col.speeds_real.append(np.array([20.0, 20.0, 20.0]))
        # Low curvature: κ ≈ 0 → curvature score ≈ 0.8
        col.kappas.append(np.array([0.0, 0.0, 0.0]))
        col.kappas.append(np.array([0.0, 0.0, 0.0]))

        sep, speed, curv, obst = evo._compute_objectives(col)
        assert 0.0 <= sep <= 1.0, f"sep={sep} should be in [0,1]"
        assert 0.0 <= speed <= 1.0, f"speed={speed} should be in [0,1]"
        assert 0.8 <= curv <= 1.0, f"curv={curv} should be in [0.8,1.0]"
        assert obst == pytest.approx(1.0), f"obst={obst} should be 1.0 for 100% collision-free"


