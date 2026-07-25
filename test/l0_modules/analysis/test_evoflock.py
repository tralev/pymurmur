"""EvoFlock tests — Phase 11 SSGA, uniform crossover, worst-of-4
evaluation, objectives, obstacle integration, persistence.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymurmur.analysis.evoflock import (
    EVOLVABLE_PARAMS,
    INTEGER_PARAMS,
    OBJECTIVE_NAMES,
    EvoConfig,
    EvoFlock,
    Genome,
    _pareto_front,
)
from pymurmur.core.config import SimConfig

CORE_PARAMS = {
    "separation_weight", "alignment_weight", "cohesion_weight",
    "noise_scale", "max_force", "phi_p", "phi_a", "steric",
    "predictive_avoid_weight", "static_avoid_weight",
}

P11_5_PARAMS = {
    "w_fwd", "max_dist_sep", "max_dist_align", "max_dist_coh",
    "angle_sep", "angle_align", "angle_coh",
    "fly_away_max_dist", "min_time_to_collide", "sigma", "speed_min_factor",
}


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


class TestConstants:
    """EVOLVABLE_PARAMS, OBJECTIVE_NAMES, EvoConfig defaults."""

    def test_evolvable_params_count(self):
        """P11.5: 10 core + 11 expanded genes = 21 parameters."""
        assert len(EVOLVABLE_PARAMS) == 21

    def test_evolvable_params_valid_ranges(self):
        """Every parameter has lo < hi and both are finite."""
        for name, (lo, hi) in EVOLVABLE_PARAMS.items():
            assert isinstance(name, str)
            assert lo < hi, f"{name}: lo={lo} not < hi={hi}"
            assert np.isfinite(lo) and np.isfinite(hi), f"{name}: non-finite range"

    def test_evolvable_params_all_known(self):
        """EVOLVABLE_PARAMS is exactly the core set plus the P11.5 set."""
        assert set(EVOLVABLE_PARAMS.keys()) == CORE_PARAMS | P11_5_PARAMS

    def test_integer_params(self):
        """P11.5: sigma is the (only) integer-decoded gene."""
        assert INTEGER_PARAMS == frozenset({"sigma"})

    def test_objective_names_count(self):
        """OBJECTIVE_NAMES has exactly 4 objectives (including obstacle avoidance)."""
        assert len(OBJECTIVE_NAMES) == 4
        assert set(OBJECTIVE_NAMES) == {"separation", "speed", "curvature", "obstacle_avoidance"}

    def test_evoconfig_defaults(self):
        """EvoConfig() has documented default values."""
        ec = EvoConfig()
        assert ec.population_size == 300
        assert ec.max_steps == 30000
        assert ec.n_islands == 4
        assert ec.migration_rate == 0.05
        assert ec.tournament_size == 3
        assert ec.eval_steps == 500
        assert ec.evals_per_candidate == 4  # P11.2
        assert ec.epsilon == 0.01
        assert ec.mutation_rate == 0.1
        assert ec.mutation_sigma == 0.1



class TestGenome:
    """Genome encoding and decoding."""

    def test_genome_decodes_to_range(self):
        """to_config_params() maps [0,1] to actual parameter range."""
        params = _uniform_genome(0.5).to_config_params()
        for name, (lo, hi) in EVOLVABLE_PARAMS.items():
            assert lo <= params[name] <= hi, f"{name}: {params[name]} not in [{lo}, {hi}]"

    def test_genome_boundary_values(self):
        """Values at 0 and 1 map to min and max."""
        params = _uniform_genome(0.0).to_config_params()
        for name, (lo, _) in EVOLVABLE_PARAMS.items():
            assert params[name] == pytest.approx(lo)

        params = _uniform_genome(1.0).to_config_params()
        for name, (_, hi) in EVOLVABLE_PARAMS.items():
            assert params[name] == pytest.approx(hi)

    def test_sigma_integer_after_decode(self):
        """P11.5: σ decodes to an integer for any gene value."""
        for v in (0.0, 0.33, 0.5, 0.77, 1.0):
            sigma = _uniform_genome(v).to_config_params()["sigma"]
            assert isinstance(sigma, int), f"sigma at {v} decoded to {type(sigma)}"
            lo, hi = EVOLVABLE_PARAMS["sigma"]
            assert lo <= sigma <= hi

    def test_decode_produces_exactly_21_params(self):
        """P11.5: to_config_params() returns exactly 21 key-value pairs."""
        params = _uniform_genome(0.5).to_config_params()
        assert len(params) == 21
        assert set(params.keys()) == set(EVOLVABLE_PARAMS.keys())

    def test_sigma_decode_boundary_values(self):
        """P11.5: σ rounds to integer at gene boundaries.
        σ range [1,10]: 1 + gene·9 → round.
        gene 0.0→1, gene 0.499→5, gene 0.5→6, gene 1.0→10."""
        lo, hi = EVOLVABLE_PARAMS["sigma"]
        assert lo == 1.0 and hi == 10.0, f"sigma range is [{lo}, {hi}]"

        def sigma_at(gene_val):
            g = _uniform_genome(gene_val)
            return g.to_config_params()["sigma"]

        assert sigma_at(0.0) == 1
        # 1 + 0.499 * 9 = 5.491 → round(5.491) = 5
        assert sigma_at(0.499) == 5, (
            f"sigma at 0.499 should be 5, got {sigma_at(0.499)}"
        )
        # 1 + 0.5 * 9 = 5.5 → round(5.5) = 6 (banker's rounding to even)
        assert sigma_at(0.5) == 6, (
            f"sigma at 0.5 should be 6, got {sigma_at(0.5)}"
        )
        assert sigma_at(1.0) == 10


class TestPareto:
    """Pareto front computation."""

    def test_single_genome_is_front(self):
        """One genome is always non-dominated."""
        g = Genome(values={}, objectives=np.array([1.0, 0.5, 0.8]))
        front = _pareto_front([g], 0.01)
        assert len(front) == 1

    def test_dominated_removed(self):
        """Pareto-dominated genome is excluded."""
        g1 = Genome(values={}, objectives=np.array([1.0, 1.0, 1.0]))
        g2 = Genome(values={}, objectives=np.array([0.5, 0.5, 0.5]))
        front = _pareto_front([g1, g2], 0.01)
        assert len(front) == 1, f"Expected 1 non-dominated, got {len(front)}"

    def test_nondominated_both_kept(self):
        """Incomparable genomes both kept."""
        g1 = Genome(values={}, objectives=np.array([1.0, 0.0, 0.0]))
        g2 = Genome(values={}, objectives=np.array([0.0, 1.0, 0.0]))
        front = _pareto_front([g1, g2], 0.01)
        assert len(front) == 2

    def test_pareto_front_empty_list(self):
        """_pareto_front([]) returns empty list."""
        assert _pareto_front([], 0.01) == []


class TestSSGAUpdate:
    """P11.1: SSGA fidelity — worst-of-3 negative selection, uniform
    crossover, fitness cache."""

    def _evo(self, seed: int = 7) -> EvoFlock:
        cfg = SimConfig()
        cfg.seed = seed
        return EvoFlock(cfg, EvoConfig(population_size=12, n_islands=1))

    def test_worst_of_three_gone(self):
        """The worst of the 3 selected genomes is deleted; child fills the slot."""
        evo = self._evo()
        _stub_single_eval(evo)
        # Island of exactly 3 → the triple is always all of them
        low, mid, high = _uniform_genome(0.1), _uniform_genome(0.5), _uniform_genome(0.9)
        island = [low, mid, high]
        evo._ssga_update(island)
        assert len(island) == 3
        assert low not in island, "Worst-of-3 must be deleted"
        assert mid in island and high in island

    def test_all_three_finite_fitness(self):
        """All selected genomes (and the child) end with finite fitness."""
        evo = self._evo()
        _stub_single_eval(evo)
        island = [_uniform_genome(v) for v in (0.2, 0.5, 0.8)]
        evo._ssga_update(island)
        assert all(np.isfinite(g.fitness) for g in island)

    def test_crossover_mixes_parent_genes(self):
        """Uniform crossover with disjoint-value parents mixes genes from both."""
        evo = self._evo()
        a = _uniform_genome(0.0)
        b = _uniform_genome(1.0)
        child = evo._crossover(a, b)
        vals = set(child.values.values())
        assert vals == {0.0, 1.0}, f"Child should mix both parents, got {vals}"

    def test_child_in_island_mixes_parents(self):
        """After an update, the inserted child carries genes from both survivors."""
        evo = self._evo()
        _stub_single_eval(evo)
        a, b, worst = _uniform_genome(1.0), _uniform_genome(0.9), _uniform_genome(0.0)
        island = [a, b, worst]
        # Disable mutation so gene provenance is exact
        evo._ga.mutation_rate = 0.0
        evo._ssga_update(island)
        child = next(g for g in island if g is not a and g is not b)
        assert set(child.values.values()) <= {1.0, 0.9}
        assert len(set(child.values.values())) == 2, "Child should mix both parents"

    def test_cache_prevents_resimulation(self):
        """P11.1: identical genomes are never re-simulated (fitness cache)."""
        evo = self._evo()
        calls: list[int] = []
        _stub_single_eval(evo, log=calls)

        g1 = _uniform_genome(0.3)
        g2 = _uniform_genome(0.3)  # identical gene values
        evo._ensure_evaluated(g1)
        n_after_first = len(calls)
        assert n_after_first > 0
        evo._ensure_evaluated(g2)
        assert len(calls) == n_after_first, "Cache must prevent re-simulation"
        assert g2.fitness == g1.fitness
        assert list(g2.eval_seeds) == list(g1.eval_seeds)

    def test_small_island_no_update(self):
        """Islands with fewer than 3 members are left untouched."""
        evo = self._evo()
        island = [_uniform_genome(0.5), _uniform_genome(0.6)]
        before = list(island)
        evo._ssga_update(island)
        assert island == before

    def test_ensure_evaluated_skips_already_finite(self):
        """P11.1: _ensure_evaluated on a genome with finite fitness
        is a no-op — it does not re-simulate."""
        evo = self._evo()
        calls: list[int] = []
        _stub_single_eval(evo, log=calls)
        g = _uniform_genome(0.4)
        evo._ensure_evaluated(g)
        n_first = len(calls)
        assert n_first > 0
        # Second call — fitness already finite, must not re-evaluate
        evo._ensure_evaluated(g)
        assert len(calls) == n_first, "Finite fitness should prevent re-evaluation"


class TestWorstOfFour:
    """P11.2: worst-of-4 evaluation with fixed per-sim seeds."""

    def test_min_reduction(self):
        """Per-sim fitnesses [0.9, 0.8, 0.95, 0.7] → candidate fitness 0.7."""
        cfg = SimConfig()
        cfg.seed = 0
        evo = EvoFlock(cfg, EvoConfig(evals_per_candidate=4))
        sim_fits = iter([0.9, 0.8, 0.95, 0.7])

        def fake_single(genome, seed):
            fit = next(sim_fits)
            return fit, np.full(len(OBJECTIVE_NAMES), fit)

        evo._evaluate_single = fake_single
        g = _uniform_genome()
        evo._evaluate(g)
        assert g.fitness == pytest.approx(0.7)
        assert g.objectives == pytest.approx(np.full(4, 0.7))

    def test_seeds_fixed_and_recorded(self):
        """Per-sim seeds are deterministic, recorded, and shared by all candidates."""
        cfg = SimConfig()
        cfg.seed = 42
        evo = EvoFlock(cfg, EvoConfig(evals_per_candidate=4))
        _stub_single_eval(evo)

        g1, g2 = _uniform_genome(0.2), _uniform_genome(0.8)
        evo._evaluate(g1)
        evo._evaluate(g2)
        assert len(g1.eval_seeds) == 4
        assert g1.eval_seeds == g2.eval_seeds, "All candidates share the fixed seeds"
        assert len(set(g1.eval_seeds)) == 4, "Per-sim seeds must differ"

    def test_deterministic_order(self):
        """Sims run in deterministic seed order matching the recorded seeds."""
        cfg = SimConfig()
        cfg.seed = 5
        evo = EvoFlock(cfg, EvoConfig(evals_per_candidate=4))
        order: list[int] = []
        _stub_single_eval(evo, log=order)
        g = _uniform_genome()
        evo._evaluate(g)
        assert order == g.eval_seeds, "Sims must run in recorded seed order"

    def test_single_eval_no_min_reduction(self):
        """P11.2: With evals_per_candidate=1, fitness is simply
        that single eval's result (no min-reduction needed)."""
        cfg = SimConfig()
        evo = EvoFlock(cfg, EvoConfig(evals_per_candidate=1))
        _stub_single_eval(evo)
        g = _uniform_genome(0.5)
        evo._evaluate(g)
        assert g.fitness > 0.0
        assert len(g.eval_seeds) == 1
        assert len(g.objectives) == 4

    def test_seed_formula_exact(self):
        """P11.2: eval seeds follow the formula base_seed + 7919*k + 13."""
        cfg = SimConfig()
        cfg.seed = 100
        evo = EvoFlock(cfg, EvoConfig(evals_per_candidate=3))
        _stub_single_eval(evo)
        g = _uniform_genome()
        evo._evaluate(g)
        expected = [100 + 13, 100 + 7919 + 13, 100 + 2 * 7919 + 13]
        assert g.eval_seeds == expected, (
            f"Expected {expected}, got {g.eval_seeds}"
        )


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


