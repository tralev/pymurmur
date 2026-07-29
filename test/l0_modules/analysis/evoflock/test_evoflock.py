"""EvoFlock tests — Phase 11 SSGA, uniform crossover, worst-of-4
evaluation, objectives, obstacle integration, persistence.

Split out of this file (file-size split): genome-level operations
(Constants, Genome, Pareto, SSGAUpdate, WorstOfFour) live in
test_evoflock_genome_ops.py; this file keeps TestEvoFlock (the
GA-integration class).
"""

from __future__ import annotations

import numpy as np
import pytest

from pymurmur.analysis.evoflock import (
    EVOLVABLE_PARAMS,
    OBJECTIVE_NAMES,
    EvoConfig,
    EvoFlock,
    Genome,
)
from pymurmur.core.config import SimConfig


def _uniform_genome(value: float = 0.5) -> Genome:
    return Genome(values={name: value for name in EVOLVABLE_PARAMS})


def _stub_single_eval(evo: EvoFlock, log: list | None = None, fitness_fn=None):
    """Replace _evaluate_single with a cheap deterministic stub."""
    def fake_single(genome, seed):
        if log is not None:
            log.append(seed)
        fit = (
            fitness_fn(genome) if fitness_fn is not None
            else float(sum(genome.values.values()))
        )
        return fit, np.full(len(OBJECTIVE_NAMES), 0.5)
    evo._evaluate_single = fake_single


class TestEvoFlock:
    """Integration tests for the GA."""

    def test_initialize_creates_population(self):
        """_initialize_population creates correct number of genomes."""
        cfg = SimConfig()
        cfg.num_boids = 20
        evo = EvoFlock(cfg, EvoConfig(population_size=40, n_islands=2))
        evo._initialize_population()
        total = sum(len(island) for island in evo._islands)
        assert total == 40
        assert len(evo._islands) == 2

    def test_tournament_select_returns_genome(self):
        """Tournament selection returns a genome from the island."""
        cfg = SimConfig()
        evo = EvoFlock(cfg, EvoConfig(population_size=10))
        island = [_uniform_genome() for _ in range(5)]
        island[2].fitness = 100.0  # make one clearly better
        selected = evo._tournament_select(island, 3)
        assert selected is not None
        assert isinstance(selected, Genome)

    def test_mutation_produces_child(self):
        """Mutation creates a child different from parent."""
        cfg = SimConfig()
        evo = EvoFlock(cfg, EvoConfig(mutation_rate=1.0, mutation_sigma=0.2))
        parent = _uniform_genome()
        child = evo._mutate(parent)
        # With mutation_rate=1.0, at least some genes should differ
        differ = any(
            child.values[name] != parent.values[name]
            for name in EVOLVABLE_PARAMS
        )
        assert differ, "Mutation should change at least one gene"

    @pytest.mark.slow
    def test_evaluate_produces_fitness(self):
        """_evaluate runs simulations and computes fitness."""
        cfg = SimConfig()
        cfg.num_boids = 20
        cfg.seed = 11
        evo = EvoFlock(cfg, EvoConfig(eval_steps=50, evals_per_candidate=2))
        genome = _uniform_genome()
        evo._evaluate(genome)
        assert genome.fitness >= 0.0
        assert np.isfinite(genome.fitness)
        assert len(genome.objectives) == 4
        assert len(genome.eval_seeds) == 2

    @pytest.mark.slow
    def test_run_minimal_ga(self):
        """Full GA run with small population completes."""
        cfg = SimConfig()
        cfg.num_boids = 15
        cfg.seed = 3
        ga_cfg = EvoConfig(
            population_size=6,
            max_steps=2,
            n_islands=2,
            eval_steps=30,
            evals_per_candidate=1,
            mutation_rate=0.3,
        )
        evo = EvoFlock(cfg, ga_cfg)
        result = evo.run(n_runs=1)
        assert isinstance(result, dict)
        # Should return evolved parameters
        assert "separation_weight" in result

    def test_run_zero_runs_returns_empty(self):
        """run(n_runs=0) returns empty dict."""
        cfg = SimConfig()
        evo = EvoFlock(cfg, EvoConfig())
        result = evo.run(n_runs=0)
        assert result == {}

    def test_migrate_rotates_islands(self):
        """_migrate rotates which island pair swaps each call.

        First call: i=1,j=2. Second call: i=2,j=3.
        """
        cfg = SimConfig()
        evo = EvoFlock(cfg, EvoConfig(population_size=20, n_islands=4))
        evo._initialize_population()

        # Give each island a recognizable fitness signature
        for idx, island in enumerate(evo._islands):
            for g in island:
                g.fitness = float(idx)  # island 0=0.0, 1=1.0, 2=2.0, 3=3.0

        # First migration: _migrate_idx 0→1, swaps islands 1↔2
        evo._migrate()
        f1 = {g.fitness for g in evo._islands[1]}
        f2 = {g.fitness for g in evo._islands[2]}
        assert len(f1) > 1, f"Island 1 should be mixed after 1↔2 swap, got {f1}"
        assert len(f2) > 1, f"Island 2 should be mixed after 1↔2 swap, got {f2}"

        # Second migration: _migrate_idx 1→2, swaps islands 2↔3
        evo._migrate()
        f2_after = {g.fitness for g in evo._islands[2]}
        f3_after = {g.fitness for g in evo._islands[3]}
        assert len(f2_after) > 1, f"Island 2 should be mixed after 2↔3 swap, got {f2_after}"
        assert len(f3_after) > 1, f"Island 3 should be mixed after 2↔3 swap, got {f3_after}"

    def test_migrate_handles_empty_island(self):
        """_migrate with an empty island returns early without error."""
        cfg = SimConfig()
        evo = EvoFlock(cfg, EvoConfig(n_islands=4))
        evo._islands = [[], [], [], []]  # all empty
        # Should not raise
        evo._migrate()

    def test_best_genome_returns_max_fitness(self):
        """_best_genome returns the genome with highest fitness."""
        cfg = SimConfig()
        evo = EvoFlock(cfg, EvoConfig(population_size=10, n_islands=2))
        evo._initialize_population()
        # Set one genome to very high fitness
        evo._islands[1][2].fitness = 999.0
        best = evo._best_genome()
        assert best is not None
        assert best.fitness == 999.0

    def test_best_genome_empty_islands(self):
        """_best_genome returns None when all islands are empty."""
        cfg = SimConfig()
        evo = EvoFlock(cfg, EvoConfig())
        evo._islands = [[], []]
        assert evo._best_genome() is None

    def test_tournament_select_empty_island(self):
        """_tournament_select returns None for empty island."""
        cfg = SimConfig()
        evo = EvoFlock(cfg, EvoConfig())
        assert evo._tournament_select([], 3) is None

    def test_mutation_rate_zero_no_change(self):
        """With mutation_rate=0, child is identical to parent."""
        cfg = SimConfig()
        evo = EvoFlock(cfg, EvoConfig(mutation_rate=0.0))
        parent = _uniform_genome(0.3)
        child = evo._mutate(parent)
        for name in EVOLVABLE_PARAMS:
            assert child.values[name] == parent.values[name], \
                f"{name}: child={child.values[name]} != parent={parent.values[name]}"

    def test_evaluate_no_steps_zero_fitness(self):
        """A sim that never invokes the callback yields fitness 0.0."""
        from unittest.mock import MagicMock, patch

        cfg = SimConfig()
        cfg.num_boids = 20
        evo = EvoFlock(cfg, EvoConfig(eval_steps=10))
        genome = _uniform_genome()

        fake_sim = MagicMock()  # run_headless does nothing → collector empty

        # SimulationEngine is imported locally in _evaluate_single() via
        # `from ..simulation.engine import SimulationEngine`,
        # so patch the actual source module.
        with patch("pymurmur.simulation.engine.SimulationEngine", return_value=fake_sim):
            evo._evaluate(genome)

        assert genome.fitness == 0.0

    def test_evaluate_single_enforces_eval_protocol(self):
        """P11.5: eval sims run in spatial mode with fixed k=7 neighbours."""
        from unittest.mock import MagicMock, patch

        cfg = SimConfig()
        cfg.num_boids = 20
        cfg.mode = "projection"
        evo = EvoFlock(cfg, EvoConfig(eval_steps=10))
        captured = {}

        def fake_engine(inner_cfg):
            captured["cfg"] = inner_cfg
            return MagicMock()

        with patch("pymurmur.simulation.engine.SimulationEngine", side_effect=fake_engine):
            evo._evaluate_single(_uniform_genome(), seed=99)

        assert captured["cfg"].mode == "spatial"
        assert captured["cfg"].influence_count == 7
        assert captured["cfg"].seed == 99
        assert cfg.mode == "projection", "Base config must not be mutated"

    def test_genome_default_fitness(self):
        """New Genome has fitness=-inf, zero objectives, no seeds."""
        g = _uniform_genome()
        assert g.fitness == float("-inf")
        assert len(g.objectives) == len(OBJECTIVE_NAMES)
        assert np.all(g.objectives == 0.0)
        assert g.eval_seeds == []

    def test_tournament_select_picks_highest_fitness(self):
        """P11.1: _tournament_select with k=5 must pick the genome
        with the highest fitness, not just any genome."""
        cfg = SimConfig()
        evo = EvoFlock(cfg, EvoConfig(population_size=10))
        island = [_uniform_genome() for _ in range(10)]
        # Give each genome a unique fitness
        for i, g in enumerate(island):
            g.fitness = float(i)
        # With tournament_size=10 (entire island), must pick index 9
        selected = evo._tournament_select(island, 10)
        assert selected is not None
        assert selected.fitness == 9.0, (
            f"Tournament select must pick highest, got {selected.fitness}"
        )

    def test_initialize_creates_distinct_genomes(self):
        """P11.1: Random initialization creates diverse genomes —
        not all genomes are identical."""
        cfg = SimConfig()
        evo = EvoFlock(cfg, EvoConfig(population_size=20, n_islands=1))
        evo._initialize_population()
        genomes = evo._islands[0]
        # Compare all pairs — at least one pair must differ
        found_different = False
        for i, a in enumerate(genomes):
            for j, b in enumerate(genomes):
                if i >= j:
                    continue
                if a.values != b.values:
                    found_different = True
                    break
            if found_different:
                break
        assert found_different, (
            "All 20 genomes are identical — random init not working"
        )


