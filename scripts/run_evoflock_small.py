"""Small EvoFlock training run to verify end-to-end GA convergence.

Population 20, 200 steps, 2 islands, 25 boids, 40 eval steps.
Tracks best fitness per generation and reports final evolved params.
"""
import time

from pymurmur.analysis.evoflock import EVOLVABLE_PARAMS, EvoConfig, EvoFlock
from pymurmur.core.config import SimConfig

cfg = SimConfig()
cfg.num_boids = 150          # dense enough for separation scoring
cfg.width = 100.0             # tight volume: ~67 boids per dimension
cfg.height = 100.0
cfg.depth = 100.0
cfg.boid_size = 3.0           # body diameter = 6, separation target = 12-24 units
cfg.v0 = 10.0                 # slower speed for denser packing

# With 150 birds in 100³ and boid hitting v0=10,
# the birds form tight flocks where local_spacing ≈ 15-20 units,
# which maps to spacing_ratio ≈ 2.5-3.3 body diameters → score ≈ 0.5-1.0

ga_cfg = EvoConfig(
    population_size=20,
    max_steps=200,
    n_islands=2,
    eval_steps=100,              # longer eval: give flock time to settle
    tournament_size=3,
    mutation_rate=0.3,
    mutation_sigma=0.15,
)

print("=" * 60)
print("EvoFlock Training Run — Convergence Verification")
print("=" * 60)
print(f"  Population: {ga_cfg.population_size}")
print(f"  Steps:      {ga_cfg.max_steps}")
print(f"  Islands:    {ga_cfg.n_islands}")
print(f"  Birds:      {cfg.num_boids}")
print(f"  Eval steps: {ga_cfg.eval_steps}")
print()

evo = EvoFlock(cfg, ga_cfg)
evo._initialize_population()

# Train step-by-step, tracking best fitness
best_history: list[float] = []
start = time.perf_counter()

for step in range(ga_cfg.max_steps):
    # Island migration
    if step > 0 and step % max(1, int(1.0 / ga_cfg.migration_rate)) == 0:
        evo._migrate()

    for island_idx in range(ga_cfg.n_islands):
        island = evo._islands[island_idx]
        parent = evo._tournament_select(island, ga_cfg.tournament_size)
        if parent is None:
            continue
        child = evo._mutate(parent)
        evo._evaluate(child)
        worst_idx = min(range(len(island)), key=lambda i: island[i].fitness)
        if child.fitness > island[worst_idx].fitness:
            island[worst_idx] = child

    if step % 100 == 0:
        evo._update_pareto()

    # Record best fitness
    best = evo._best_genome()
    if best is not None:
        best_history.append(best.fitness)

    if step % 20 == 0:
        pareto_size = len(evo._pareto_front)
        bf = best.fitness if best else float("-inf")
        print(f"  Step {step:4d} | best_fitness={bf:.6e} | pareto={pareto_size}")

elapsed = time.perf_counter() - start

# Final report
best = evo._best_genome()
print()
print("=" * 60)
print("Final Results")
print("=" * 60)
print(f"  Elapsed:      {elapsed:.1f}s")
print(f"  Best fitness: {best.fitness:.6e}" if best else "  No best genome")
print(f"  Pareto front: {len(evo._pareto_front)}")

if best is not None:
    print(f"  Objectives:   sep={best.objectives[0]:.4f}  "
          f"speed={best.objectives[1]:.4f}  "
          f"curv={best.objectives[2]:.4f}  "
          f"obs_avoid={best.objectives[3]:.4f}")
    print()
    print("  Evolved Parameters:")
    params = best.to_config_params()
    for name in EVOLVABLE_PARAMS:
        print(f"    {name:<30s} = {params[name]:.4f}")

# Verify convergence: fitness should have improved
if len(best_history) >= 2:
    initial = best_history[0]
    final = best_history[-1]
    improvement = final - initial
    print()
    print(f"  Initial fitness: {initial:.6e}")
    print(f"  Final fitness:   {final:.6e}")
    print(f"  Improvement:     {improvement:.6e} ({'+' if improvement > 0 else ''}{improvement/abs(initial)*100:.1f}%)")
    if improvement > 0:
        print("  CONVERGENCE: ✅ Fitness improved over training")
    else:
        print("  CONVERGENCE: ⚠️ No fitness improvement (may need more steps)")
