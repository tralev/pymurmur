"""Per-step objective sampling + objective-function helpers for EvoFlock.

Extracted from evoflock.py (file-size split) — self-contained: the
per-step collector, obstacle-scene YAML loader, and small scoring
helpers used by EvoFlock._compute_objectives/_update_pareto.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..physics.obstacles import ObstacleScene
    from .evoflock import Genome


# ── P11.3/P11.4: Per-step objective sampling ─────────────────────

class _ObjectiveCollector:
    """run_headless callback — samples per-boid-step objective data.

    Records nearest-neighbour distances (in body diameters), real speeds
    (m/s), and curvature κ = |v×a|/|v|³ each step. With an ObstacleScene
    attached, counts collision-free steps, applies kinematic correction,
    and feeds P11.5 avoidance steering into the next step (deferred, the
    same pattern as P12.1 external control).
    """

    def __init__(self, config, scene: "ObstacleScene | None" = None) -> None:
        self._body_diameter = max(float(config.boid_size) * 2.0, 1e-9)
        v0 = max(float(config.v0), 1e-9)
        self._speed_to_ms = float(config.cruise_speed_ms) / v0
        self._scene = scene
        self._static_w = float(config.static_avoid_weight)
        self._predictive_w = float(config.predictive_avoid_weight)
        self._fly_away = float(config.fly_away_max_dist)
        self._min_ttc = float(config.min_time_to_collide)
        self.nn_ratios: list[np.ndarray] = []
        self.speeds_real: list[np.ndarray] = []
        self.kappas: list[np.ndarray] = []
        self.n_steps: int = 0
        self.collision_free_steps: int = 0

    def __call__(self, engine) -> None:
        from scipy.spatial import cKDTree

        flock = engine.flock
        act_idx = np.where(flock.active)[0]
        self.n_steps += 1
        if len(act_idx) == 0:
            self.collision_free_steps += 1
            return

        collided_any = False
        if self._scene is not None and self._scene.n_shapes:
            corrected, collided = self._scene.resolve(
                flock.prev_positions[act_idx], flock.positions[act_idx],
            )
            if collided.any():
                collided_any = True
            flock.positions[act_idx] = corrected
            # P11.5: avoidance steering — applied to v, felt next step
            avoid = self._scene.avoidance_accel(
                flock.positions[act_idx], flock.velocities[act_idx],
                static_weight=self._static_w * 1e-3,
                predictive_weight=self._predictive_w * 1e-3,
                fly_away_max_dist=self._fly_away,
                min_time_to_collide=self._min_ttc,
            )
            flock.velocities[act_idx] += avoid
        if not collided_any:
            self.collision_free_steps += 1

        pos = flock.positions[act_idx]
        vel = flock.velocities[act_idx]
        acc = flock.last_accelerations[act_idx]

        if len(pos) >= 2:
            tree = cKDTree(pos)
            d, _ = tree.query(pos, k=2)
            self.nn_ratios.append(
                (d[:, 1] / self._body_diameter).astype(np.float64)
            )

        speeds = np.linalg.norm(vel, axis=1)
        self.speeds_real.append((speeds * self._speed_to_ms).astype(np.float64))

        moving = speeds > 1e-6
        if moving.any():
            cross = np.cross(vel[moving], acc[moving])
            kappa = np.linalg.norm(cross, axis=1) / speeds[moving] ** 3
            self.kappas.append(kappa.astype(np.float64))


# ── P11.6: Config helpers ─────────────────────────────────────────

def load_obstacle_scene(path: str | Path) -> "ObstacleScene | None":
    """Read the `obstacles:` section of an evaluation YAML into an
    ObstacleScene (P11.4/P11.6). Returns None when the config has no
    obstacles (e.g. conf/evo_open.yaml)."""
    import yaml

    from ..physics.obstacles import ObstacleScene

    with open(path) as f:
        data = yaml.safe_load(f) or {}
    spec = data.get("obstacles")
    if not spec:
        return None
    return ObstacleScene.from_spec(spec)


# ── Objective function helpers ────────────────────────────────────

def _trapezoid(
    x: np.ndarray | float, a: float, b: float, c: float, d: float,
) -> np.ndarray:
    """P11.3 trapezoid membership: 0 below a, ramp a→b, plateau b→c,
    ramp c→d, 0 above. Vectorized."""
    x = np.asarray(x, dtype=np.float64)
    up = np.clip((x - a) / max(b - a, 1e-12), 0.0, 1.0)
    down = np.clip((d - x) / max(d - c, 1e-12), 0.0, 1.0)
    return np.minimum(up, down)


def _linear_ramp(
    x: float, lo: float, hi: float, floor: float, ceiling: float,
) -> float:
    """Linear ramp scoring: 1.0 in [lo, hi], ramps to 0 at floor/ceiling."""
    if lo <= x <= hi:
        return 1.0
    if x < lo:
        return max(0.0, (x - floor) / max(lo - floor, 0.01))
    return max(0.0, (ceiling - x) / max(ceiling - hi, 0.01))


def _pareto_front(
    genomes: "list[Genome]", epsilon: float,
) -> "list[Genome]":
    """Extract non-dominated individuals (Pareto front).

    Uses epsilon-dominance: x dominates y if x_i >= y_i for all i
    AND x_i > y_i + epsilon for at least one i.
    """
    if not genomes:
        return []

    objs = np.array([g.objectives for g in genomes], dtype=np.float64)
    n = len(genomes)
    dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            oi, oj = objs[i], objs[j]
            if np.all(oi >= oj) and np.any(oi > oj + epsilon):
                dominated[j] = True

    return [g for g, d in zip(genomes, dominated) if not d]
