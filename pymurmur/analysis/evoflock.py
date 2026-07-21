"""Evolutionary parameter tuning — SSGA for automatic flock optimisation (Phase 11).

SSGA with island model (4 islands, 0.05 migration) and hypervolume
scalarization (epsilon=0.01). Population 300, 30,000 steps.

P11.1: SSGA fidelity — per update: select 3 at random → evaluate all 3
(fitness cache keyed on genome) → delete the worst of 3 (negative
selection) → uniform crossover of the best two (each gene from a random
parent) + per-gene Gaussian mutation → child fills the freed slot.
P11.2: Worst-of-4 evaluation — evals_per_candidate sims per candidate
with fixed per-sim seeds, min-reduction, deterministic order.
P11.3: Objectives — separation trapezoid over body diameters on
nearest-neighbour distance per boid-step, speed band [19, 21] m/s on
speed_real (ramps [18, 22]), curvature κ = |v×a|/|v|³ with
score = clamp(0.8 + (κ_avg/0.1)·0.2, 0.8, 1.0), hypervolume
F = Π max(o_k, ε).
P11.4: SDF obstacle scene — per-step collision counter feeds (f_cf)^500
as the obstacle-avoidance objective.
P11.5: Expanded gene set (21 evolvable parameters, σ decoded to integer,
fixed k=7 topological neighbours during evaluation).
P11.6: Persistence — best genome + Pareto front + per-run seeds +
objective scores to output/evolved.yaml.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..core.config import SimConfig
    from ..physics.obstacles import ObstacleScene


# ── Evolvable parameter space ────────────────────────────────────

EVOLVABLE_PARAMS: dict[str, tuple[float, float]] = {
    # Core 10 (pre-P11.5)
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
    # P11.5: expanded gene set
    "w_fwd": (0.0, 2.0),                 # forward force toward v0
    "max_dist_sep": (1.0, 100.0),        # per-interaction perception distances
    "max_dist_align": (1.0, 100.0),
    "max_dist_coh": (1.0, 100.0),
    "angle_sep": (-1.0, 1.0),            # perception cones (cos α)
    "angle_align": (-1.0, 1.0),
    "angle_coh": (-1.0, 1.0),
    "fly_away_max_dist": (0.0, 100.0),   # static obstacle fly-away trigger
    "min_time_to_collide": (0.0, 10.0),  # predictive avoidance look-ahead
    "sigma": (1.0, 10.0),                # integer gene (topological count)
    "speed_min_factor": (0.0, 1.0),      # flock.speed_min_factor
}

# P11.5: genes decoded to integers
INTEGER_PARAMS: frozenset[str] = frozenset({"sigma"})

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
    evals_per_candidate: int = 4  # P11.2: worst-of-4
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
    eval_seeds: list[int] = field(default_factory=list)  # P11.2: recorded per-sim seeds

    def to_config_params(self) -> dict[str, float]:
        """Scale from [0, 1] to parameter range. σ decodes to integer (P11.5)."""
        params: dict[str, float] = {}
        for name, (lo, hi) in EVOLVABLE_PARAMS.items():
            val = lo + self.values.get(name, 0.5) * (hi - lo)
            if name in INTEGER_PARAMS:
                val = int(round(val))
            params[name] = val
        return params


# ── EvoFlock ──────────────────────────────────────────────────────

class EvoFlock:
    """SSGA with island model, uniform crossover, hypervolume scalarization.

    Usage:
        evo = EvoFlock(base_config, scene=obstacle_scene)
        best = evo.run(n_runs=1)
        evo.save("output/evolved.yaml")   # P11.6 persistence
    """

    def __init__(
        self,
        base_config: SimConfig,
        ga_config: EvoConfig | None = None,
        scene: "ObstacleScene | None" = None,
    ) -> None:
        self._base = base_config
        self._ga = ga_config or EvoConfig()
        self._scene = scene  # P11.4: optional SDF obstacle scene
        self._rng = np.random.default_rng(base_config.seed)
        self._seed_base: int = (
            int(base_config.seed) if base_config.seed is not None else 0
        )
        self._islands: list[list[Genome]] = []
        self._pareto_front: list[Genome] = []
        self._save_path: str | Path | None = None  # D4: periodic checkpoint path
        # P11.1: fitness cache — genome key → (fitness, objectives, seeds)
        self._fitness_cache: dict[tuple, tuple[float, np.ndarray, list[int]]] = {}

    # ── Run ───────────────────────────────────────────────────

    def run(
        self, n_runs: int = 1, save_path: str | Path | None = None,
    ) -> dict[str, float]:
        """Run GA for n_runs independent trials.

        Returns the best genome's decoded parameters. When save_path is
        given, persists the artifact after the final run AND periodic
        checkpoint every 1000 steps (P11.6 + D4).
        """
        self._save_path = save_path
        best_overall: Genome | None = None
        best_fitness = float("-inf")

        for _ in range(n_runs):
            self._initialize_population()
            self._run_generation_loop()
            best = self._best_genome()
            if best is not None and best.fitness > best_fitness:
                best_fitness = best.fitness
                best_overall = best

        if best_overall is None:
            return {}
        if save_path is not None:
            self.save(save_path)
        return best_overall.to_config_params()

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
        """Main SSGA loop — one P11.1 update per island per step."""
        for step in range(self._ga.max_steps):
            # Island migration every migration_rate fraction of steps
            if step > 0 and step % max(1, int(1.0 / self._ga.migration_rate)) == 0:
                self._migrate()

            for island in self._islands:
                self._ssga_update(island)

            # Update Pareto front every 100 steps (O(n^2) per update)
            if step % 100 == 0:
                self._update_pareto()

            # D4: Periodic checkpoint — save best genome every 1000 steps
            # so progress survives crashes and long runs.
            if step > 0 and step % 1000 == 0 and self._save_path:
                self.save(self._save_path)

    def _ssga_update(self, island: list[Genome]) -> None:
        """P11.1: one SSGA update on an island.

        Select 3 at random → evaluate all 3 (founders evaluated on first
        selection, cache-backed) → delete the worst of 3 → uniform
        crossover of the best two + Gaussian mutation → child fills the
        freed slot.
        """
        if len(island) < 3:
            return
        picks = self._rng.choice(len(island), size=3, replace=False)
        for i in picks:
            self._ensure_evaluated(island[i])
        ranked = sorted(picks, key=lambda i: island[i].fitness, reverse=True)
        child = self._mutate(self._crossover(island[ranked[0]], island[ranked[1]]))
        self._ensure_evaluated(child)
        island[ranked[2]] = child  # worst-of-3 gone, child in the freed slot

    # ── Genetic operators ─────────────────────────────────────

    def _tournament_select(self, island: list[Genome], k: int) -> Genome | None:
        """Select best of k random individuals."""
        if len(island) == 0:
            return None
        indices = self._rng.choice(len(island), size=min(k, len(island)), replace=False)
        best_idx = max(indices, key=lambda i: island[i].fitness)
        return island[best_idx]

    def _crossover(self, a: Genome, b: Genome) -> Genome:
        """P11.1: uniform crossover — each gene from a random parent."""
        values = {
            name: float(
                a.values.get(name, 0.5)
                if self._rng.random() < 0.5
                else b.values.get(name, 0.5)
            )
            for name in EVOLVABLE_PARAMS
        }
        return Genome(values=values)

    def _mutate(self, parent: Genome) -> Genome:
        """Gaussian mutation on a fraction of genes."""
        child_values = copy.deepcopy(parent.values)
        for name in EVOLVABLE_PARAMS:
            if self._rng.random() < self._ga.mutation_rate:
                delta = self._rng.normal(0, self._ga.mutation_sigma)
                child_values[name] = float(
                    np.clip(child_values.get(name, 0.5) + delta, 0.0, 1.0)
                )
        return Genome(values=child_values)

    def _migrate(self) -> None:
        """Swap worst individual from one island with one from its neighbour.
        Rotates which island pair is used each call."""
        n = self._ga.n_islands
        if not hasattr(self, '_migrate_idx'):
            self._migrate_idx = 0
        self._migrate_idx = (self._migrate_idx + 1) % n
        i = self._migrate_idx
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

    def _genome_key(self, genome: Genome) -> tuple:
        """P11.1: hashable cache key from gene values."""
        return tuple(
            round(float(genome.values.get(name, 0.5)), 12)
            for name in sorted(EVOLVABLE_PARAMS)
        )

    def _ensure_evaluated(self, genome: Genome) -> None:
        """P11.1: evaluate through the fitness cache — identical genomes
        are never re-simulated."""
        if np.isfinite(genome.fitness):
            return
        key = self._genome_key(genome)
        cached = self._fitness_cache.get(key)
        if cached is not None:
            genome.fitness = cached[0]
            genome.objectives = cached[1].copy()
            genome.eval_seeds = list(cached[2])
            return
        self._evaluate(genome)
        self._fitness_cache[key] = (
            genome.fitness, genome.objectives.copy(), list(genome.eval_seeds),
        )

    def _evaluate(self, genome: Genome) -> None:
        """P11.2: worst-of-N evaluation — min-reduction over sims with
        fixed per-sim seeds, deterministic order."""
        n = max(1, int(self._ga.evals_per_candidate))
        seeds = [self._seed_base + 7919 * k + 13 for k in range(n)]
        genome.eval_seeds = list(seeds)

        if self._ga.eval_parallel > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self._ga.eval_parallel) as pool:
                results = list(
                    pool.map(lambda s: self._evaluate_single(genome, s), seeds)
                )
        else:
            results = [self._evaluate_single(genome, seed) for seed in seeds]

        fits = np.array([r[0] for r in results], dtype=np.float64)
        k_min = int(np.argmin(fits))  # min-reduction: worst sim decides
        genome.fitness = float(fits[k_min])
        genome.objectives = np.asarray(results[k_min][1], dtype=np.float64)

    def _evaluate_single(
        self, genome: Genome, seed: int,
    ) -> tuple[float, np.ndarray]:
        """Run one headless simulation and score the genome on it."""
        from ..simulation.engine import SimulationEngine

        cfg = copy.copy(self._base)
        cfg.mode = "spatial"
        cfg.num_boids = self._base.num_boids if self._base.num_boids > 0 else 50
        cfg.seed = int(seed)
        cfg.influence_count = 7  # P11.5: fixed k topological neighbours

        # Apply genome parameters
        for name, value in genome.to_config_params().items():
            if name == "phi_p":  # nested-only (flat shim retired)
                cfg.projection.phi_p = value
            else:
                setattr(cfg, name, value)

        sim = SimulationEngine(cfg)
        collector = _ObjectiveCollector(cfg, scene=self._scene)
        sim.run_headless(steps=self._ga.eval_steps, callback=collector)

        if collector.n_steps == 0:
            return 0.0, np.zeros(len(OBJECTIVE_NAMES), dtype=np.float64)

        scores = self._compute_objectives(collector)
        # Hypervolume scalarization: F = Π max(o_k, ε)
        fitness = float(np.prod(np.maximum(scores, self._ga.epsilon)))
        return fitness, np.asarray(scores, dtype=np.float64)

    def _compute_objectives(
        self, collector: "_ObjectiveCollector",
    ) -> tuple[float, float, float, float]:
        """P11.3: separation, speed, curvature, obstacle avoidance in [0, 1].

        Uses the settled portion (last 50% of sampled steps).
        """
        def settled(chunks: list[np.ndarray]) -> np.ndarray:
            if not chunks:
                return np.empty(0, dtype=np.float64)
            start = len(chunks) // 2 if len(chunks) > 1 else 0
            return np.concatenate(chunks[start:])

        # Separation: trapezoid over body diameters on per-boid-step NN distance
        nn = settled(collector.nn_ratios)
        sep_score = float(np.mean(_trapezoid(nn, 2.0, 2.5, 4.0, 5.0))) if nn.size else 0.0

        # Speed: band [19, 21] m/s on speed_real, ramps [18, 22]
        speeds = settled(collector.speeds_real)
        speed_score = (
            float(np.mean(_trapezoid(speeds, 18.0, 19.0, 21.0, 22.0)))
            if speeds.size else 0.0
        )

        # Curvature: κ = |v×a|/|v|³, score = clamp(0.8 + (κ_avg/0.1)·0.2, 0.8, 1.0)
        kappas = settled(collector.kappas)
        if kappas.size:
            kappa_avg = float(np.mean(kappas))
            curv_score = float(np.clip(0.8 + (kappa_avg / 0.1) * 0.2, 0.8, 1.0))
        else:
            curv_score = 0.8  # κ undefined (stationary) → floor score

        # Obstacle avoidance: (f_cf)^500 — hard threshold near 100% collision-free
        f_cf = collector.collision_free_steps / max(collector.n_steps, 1)
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

    # ── P11.6: Persistence ────────────────────────────────────

    def save(self, path: str | Path = "output/evolved.yaml") -> Path:
        """Persist best genome + Pareto front + per-run seeds + objective
        scores as the evolved.yaml artifact (P11.6, schema guarded by P0.16)."""
        import yaml  # type: ignore[import-untyped]

        best = self._best_genome()
        if best is None:
            raise ValueError("No population to save — run() first")
        self._update_pareto()

        def decoded(g: Genome) -> dict:
            return {
                name: (int(v) if name in INTEGER_PARAMS else float(v))
                for name, v in g.to_config_params().items()
            }

        def objectives(g: Genome) -> dict:
            return {
                name: float(o) for name, o in zip(OBJECTIVE_NAMES, g.objectives)
            }

        data = {
            "evolved_params": decoded(best),
            "fitness": float(best.fitness),
            "objective_scores": objectives(best),
            "eval_seeds": [int(s) for s in best.eval_seeds],
            "pareto_front": [
                {
                    "params": decoded(g),
                    "objectives": objectives(g),
                    "fitness": float(g.fitness),
                }
                for g in self._pareto_front
            ],
        }

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            yaml.safe_dump(data, f, sort_keys=True)
        return out


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
