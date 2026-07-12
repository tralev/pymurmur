"""Evolutionary parameter tuning — SSGA for automatic flock optimisation (Phase 11).

SSGA with 3-way tournament, negative selection, island model (4 islands,
0.05 migration), hypervolume scalarization (epsilon=0.01).
Population 300, 30,000 steps.
"""

from __future__ import annotations

import copy
import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.config import SimConfig


# ── Evolvable parameter space ────────────────────────────────────

EVOLVABLE_PARAMS: dict[str, tuple[float, float]] = {
    "separation_weight": (0.5, 10.0),
    "alignment_weight": (0.1, 5.0),
    "cohesion_weight": (0.1, 5.0),
    "noise_scale": (0.0, 2.0),
    "max_force": (0.05, 0.5),
    "phi_p": (0.01, 0.2),
    "phi_a": (0.1, 2.0),
    "steric": (0.0, 2.0),
    "predictive_avoid_weight": (0.0, 100.0),
    "static_avoid_weight": (0.0, 100.0),
}

OBJECTIVE_NAMES: tuple[str, ...] = (
    "separation", "speed", "curvature", "obstacle_avoidance",
)


# ── EvoConfig ─────────────────────────────────────────────────────

@dataclass
class EvoConfig:
    """GA hyperparameters."""

    population_size: int = 300
    max_steps: int = 30000
    n_islands: int = 4
    migration_rate: float = 0.05
    tournament_size: int = 3
    eval_steps: int = 500
    eval_parallel: int = 1
    epsilon: float = 0.01       # hypervolume floor
    mutation_rate: float = 0.1  # fraction of genes mutated per child
    mutation_sigma: float = 0.1 # std dev of Gaussian mutation (relative to range)


@dataclass
class Genome:
    """One individual in the GA population."""

    values: dict[str, float]  # parameter name → value in [0, 1]
    fitness: float = float("-inf")
    objectives: np.ndarray = field(
        default_factory=lambda: np.zeros(len(OBJECTIVE_NAMES), dtype=np.float64),
    )

    def to_config_params(self) -> dict[str, float]:
        """Scale from [0, 1] to parameter range."""
        params: dict[str, float] = {}
        for name, (lo, hi) in EVOLVABLE_PARAMS.items():
            params[name] = lo + self.values.get(name, 0.5) * (hi - lo)
        return params


# ── EvoFlock ──────────────────────────────────────────────────────

class EvoFlock:
    """SSGA with tournament selection, island model, hypervolume scalarization.

    Usage:
        evo = EvoFlock(base_config)
        best = evo.run(n_runs=1)
        # best genome is saved to output/evolved.yaml
    """

    def __init__(self, base_config: SimConfig, ga_config: EvoConfig | None = None) -> None:
        self._base = base_config
        self._ga = ga_config or EvoConfig()
        self._rng = np.random.default_rng(base_config.seed)
        self._islands: list[list[Genome]] = []
        self._pareto_front: list[Genome] = []

    # ── Run ───────────────────────────────────────────────────

    def run(self, n_runs: int = 1) -> dict[str, float]:
        """Run GA for n_runs independent trials.

        Returns the best genome's decoded parameters.
        """
        best_overall: Genome | None = None
        best_fitness = float("-inf")

        for _ in range(n_runs):
            self._initialize_population()
            self._run_generation_loop()
            best = self._best_genome()
            if best is not None and best.fitness > best_fitness:
                best_fitness = best.fitness
                best_overall = best

        if best_overall is not None:
            return best_overall.to_config_params()
        return {}

    # ── Population management ─────────────────────────────────

    def _initialize_population(self) -> None:
        """Create random population split across islands."""
        n_per_island = self._ga.population_size // self._ga.n_islands
        self._islands = []
        for _ in range(self._ga.n_islands):
            island: list[Genome] = []
            for _ in range(n_per_island):
                values = {
                    name: float(self._rng.uniform(0, 1))
                    for name in EVOLVABLE_PARAMS
                }
                island.append(Genome(values=values))
            self._islands.append(island)

    def _run_generation_loop(self) -> None:
        """Main SSGA loop: evaluate, select, mutate, replace."""
        n_islands = self._ga.n_islands
        tournament = self._ga.tournament_size

        for step in range(self._ga.max_steps):
            # Island migration every migration_rate fraction of steps
            if step > 0 and step % max(1, int(1.0 / self._ga.migration_rate)) == 0:
                self._migrate()

            for island_idx in range(n_islands):
                island = self._islands[island_idx]

                # Select parent via tournament
                parent = self._tournament_select(island, tournament)
                if parent is None:
                    continue

                # Create child via mutation
                child = self._mutate(parent)

                # Evaluate child
                self._evaluate(child)

                # Negative selection: replace worst in island
                worst_idx = min(
                    range(len(island)),
                    key=lambda i: island[i].fitness,
                )
                if child.fitness > island[worst_idx].fitness:
                    island[worst_idx] = child

            # Update Pareto front every 100 steps (O(n^2) per update)
            if step % 100 == 0:
                self._update_pareto()

    # ── Genetic operators ─────────────────────────────────────

    def _tournament_select(self, island: list[Genome], k: int) -> Genome | None:
        """Select best of k random individuals."""
        if len(island) == 0:
            return None
        indices = self._rng.choice(len(island), size=min(k, len(island)), replace=False)
        best_idx = max(indices, key=lambda i: island[i].fitness)
        return island[best_idx]

    def _mutate(self, parent: Genome) -> Genome:
        """Gaussian mutation on a fraction of genes."""
        child_values = copy.deepcopy(parent.values)
        for name in EVOLVABLE_PARAMS:
            if self._rng.random() < self._ga.mutation_rate:
                delta = self._rng.normal(0, self._ga.mutation_sigma)
                child_values[name] = float(np.clip(child_values[name] + delta, 0.0, 1.0))
        return Genome(values=child_values)

    def _migrate(self) -> None:
        """Swap worst individual from one island with one from its neighbour.
        Rotates which island pair is used each call."""
        n = self._ga.n_islands
        if not hasattr(self, '_migrate_idx'):
            self._migrate_idx = 0  # type: ignore[attr-defined]
        self._migrate_idx = (self._migrate_idx + 1) % n  # type: ignore[attr-defined]
        i = self._migrate_idx  # type: ignore[attr-defined]
        j = (i + 1) % n
        if not self._islands[i] or not self._islands[j]:
            return
        worst_i = min(range(len(self._islands[i])), key=lambda x: self._islands[i][x].fitness)
        dst_idx = self._rng.integers(len(self._islands[j]))
        self._islands[i][worst_i], self._islands[j][dst_idx] = (
            self._islands[j][dst_idx],
            self._islands[i][worst_i],
        )

    # ── Evaluation ────────────────────────────────────────────

    def _evaluate(self, genome: Genome) -> None:
        """Run headless simulation and compute multi-objective fitness."""
        from ..simulation.engine import SimulationEngine

        cfg = copy.copy(self._base)
        cfg.mode = "spatial"
        cfg.num_boids = self._base.num_boids if self._base.num_boids > 0 else 50
        cfg.metrics_detail_level = 2   # need expensive metrics (local_spacing, shape)
        cfg.metrics_interval = max(1, self._ga.eval_steps // 10)  # collect frequently
        cfg.seed = self._rng.integers(0, 2**31)

        # Apply genome parameters
        params = genome.to_config_params()
        for name, value in params.items():
            setattr(cfg, name, value)

        sim = SimulationEngine(cfg)
        sim.run_headless(steps=self._ga.eval_steps)

        # Compute objectives from metrics history
        history = sim.metrics.history
        if not history:
            genome.fitness = 0.0
            return

        # Only use settled portion (last 50%)
        settle = len(history) // 2
        if settle == 0:
            settle = 1
        settled = history[settle:]

        scores = self._compute_objectives(settled)
        genome.objectives = np.array(scores, dtype=np.float64)

        # Hypervolume scalarization
        genome.fitness = float(np.prod(np.maximum(scores, self._ga.epsilon)))

    def _compute_objectives(
        self, settled: list
    ) -> tuple[float, float, float, float]:
        """Compute separation, speed, curvature, and obstacle-avoidance scores
        in [0, 1].
        """
        import numpy as np

        # Separation score: trapezoidal [2, 4] body diameters
        spacings = [s.local_spacing for s in settled if s.local_spacing > 0]
        if spacings:
            avg_spacing = float(np.mean(spacings))
            body_diameter = self._base.boid_size * 2  # body size * 2 = diameter
            spacing_ratio = avg_spacing / max(body_diameter, 0.01)
            # Trapezoidal: optimal at 2-4 body diameters
            sep_score = _linear_ramp(spacing_ratio, 2.0, 4.0, 1.0, 8.0)
        else:
            sep_score = 0.0

        # Speed score: piecewise-linear target [19, 21] m/s (scaled)
        speeds = [s.speed_avg for s in settled]
        if speeds:
            avg_speed = float(np.mean(speeds))
            speed_score = _linear_ramp(avg_speed, 3.0, 5.0, 1.0, 8.0)
        else:
            speed_score = 0.0

        # Curvature: approximated by dispersion/alpha ratio
        alphas = [s.alpha for s in settled]
        dispersions = [s.dispersion for s in settled]
        if alphas and dispersions:
            avg_alpha = float(np.mean(alphas))
            avg_disp = float(np.mean(dispersions))
            curv = avg_disp / max(avg_alpha, 0.01)
            curv_score = np.clip(0.8 + (curv / 100.0) * 0.2, 0.8, 1.0)
        else:
            curv_score = 0.5

        # Obstacle avoidance: (f_cf)^500 — hard threshold at ~99.99% collision-free.
        # Stretch goal: requires obstacle SDFs and per-step collision tracking.
        # Without obstacle infrastructure, f_cf = 1.0 (all steps collision-free),
        # giving a perfect score that doesn't influence selection.
        collision_free = getattr(self._base, "collision_free_steps", None)
        if collision_free is not None:
            total = getattr(self._base, "total_eval_steps", self._ga.eval_steps)
            f_cf = max(0.0, collision_free / max(total, 1))
        else:
            f_cf = 1.0  # no obstacles — perfect score
        obstacle_score = float(f_cf ** 500)

        return sep_score, speed_score, curv_score, obstacle_score

    # ── Pareto front ──────────────────────────────────────────

    def _update_pareto(self) -> None:
        """Update Pareto front from all islands."""
        all_genomes: list[Genome] = []
        for island in self._islands:
            all_genomes.extend(island)
        self._pareto_front = _pareto_front(all_genomes, self._ga.epsilon)

    def _best_genome(self) -> Genome | None:
        """Best genome by hypervolume fitness across all islands."""
        best: Genome | None = None
        for island in self._islands:
            for g in island:
                if best is None or g.fitness > best.fitness:
                    best = g
        return best


# ── Objective function helpers ────────────────────────────────────

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
    genomes: list[Genome], epsilon: float,
) -> list[Genome]:
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
