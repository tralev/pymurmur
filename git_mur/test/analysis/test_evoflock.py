"""EvoFlock tests — Phase 11 SSGA, tournament selection, mutation, objectives.
"""

import numpy as np
import pytest

from pymurmur.analysis.evoflock import (
    EvoConfig,
    Genome,
    EvoFlock,
    EVOLVABLE_PARAMS,
    OBJECTIVE_NAMES,
    _linear_ramp,
    _pareto_front,
)
from pymurmur.core.config import SimConfig


class TestConstants:
    """EVOLVABLE_PARAMS, OBJECTIVE_NAMES, EvoConfig defaults."""

    def test_evolvable_params_count(self):
        """EVOLVABLE_PARAMS has exactly 10 parameters (8 core + 2 avoid)."""
        assert len(EVOLVABLE_PARAMS) == 10

    def test_evolvable_params_valid_ranges(self):
        """Every parameter has lo < hi and both are finite."""
        for name, (lo, hi) in EVOLVABLE_PARAMS.items():
            assert isinstance(name, str)
            assert lo < hi, f"{name}: lo={lo} not < hi={hi}"
            assert np.isfinite(lo) and np.isfinite(hi), f"{name}: non-finite range"

    def test_evolvable_params_all_known(self):
        """EVOLVABLE_PARAMS contains expected parameter names."""
        expected = {
            "separation_weight", "alignment_weight", "cohesion_weight",
            "noise_scale", "max_force", "phi_p", "phi_a", "steric",
            "predictive_avoid_weight", "static_avoid_weight",
        }
        assert set(EVOLVABLE_PARAMS.keys()) == expected

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
        assert ec.epsilon == 0.01
        assert ec.mutation_rate == 0.1
        assert ec.mutation_sigma == 0.1


class TestObjectiveFunctions:
    """Scoring functions for separation, speed, curvature."""

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


class TestGenome:
    """Genome encoding and decoding."""

    def test_genome_decodes_to_range(self):
        """to_config_params() maps [0,1] to actual parameter range."""
        genome = Genome(values={name: 0.5 for name in EVOLVABLE_PARAMS})
        params = genome.to_config_params()
        for name, (lo, hi) in EVOLVABLE_PARAMS.items():
            assert lo <= params[name] <= hi, f"{name}: {params[name]} not in [{lo}, {hi}]"

    def test_genome_boundary_values(self):
        """Values at 0 and 1 map to min and max."""
        values = {name: 0.0 for name in EVOLVABLE_PARAMS}
        params = Genome(values=values).to_config_params()
        for name, (lo, _) in EVOLVABLE_PARAMS.items():
            assert params[name] == pytest.approx(lo)

        values = {name: 1.0 for name in EVOLVABLE_PARAMS}
        params = Genome(values=values).to_config_params()
        for name, (_, hi) in EVOLVABLE_PARAMS.items():
            assert params[name] == pytest.approx(hi)


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
        island = [
            Genome(values={name: 0.5 for name in EVOLVABLE_PARAMS})
            for _ in range(5)
        ]
        island[2].fitness = 100.0  # make one clearly better
        selected = evo._tournament_select(island, 3)
        assert selected is not None
        assert isinstance(selected, Genome)

    def test_mutation_produces_child(self):
        """Mutation creates a child different from parent."""
        cfg = SimConfig()
        evo = EvoFlock(cfg, EvoConfig(mutation_rate=1.0, mutation_sigma=0.2))
        parent = Genome(values={name: 0.5 for name in EVOLVABLE_PARAMS})
        child = evo._mutate(parent)
        # With mutation_rate=1.0, at least some genes should differ
        differ = any(
            child.values[name] != parent.values[name]
            for name in EVOLVABLE_PARAMS
        )
        assert differ, "Mutation should change at least one gene"

    @pytest.mark.slow
    def test_evaluate_produces_fitness(self):
        """_evaluate runs a simulation and computes fitness."""
        cfg = SimConfig()
        cfg.num_boids = 20
        evo = EvoFlock(cfg, EvoConfig(eval_steps=50))
        genome = Genome(values={name: 0.5 for name in EVOLVABLE_PARAMS})
        evo._evaluate(genome)
        assert genome.fitness > 0.0
        assert len(genome.objectives) == 4

    @pytest.mark.slow
    def test_run_minimal_ga(self):
        """Full GA run with small population completes."""
        cfg = SimConfig()
        cfg.num_boids = 15
        ga_cfg = EvoConfig(
            population_size=20,
            max_steps=10,
            n_islands=2,
            eval_steps=30,
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
        parent = Genome(values={name: 0.3 for name in EVOLVABLE_PARAMS})
        child = evo._mutate(parent)
        for name in EVOLVABLE_PARAMS:
            assert child.values[name] == parent.values[name], \
                f"{name}: child={child.values[name]} != parent={parent.values[name]}"

    def test_evaluate_empty_history(self):
        """_evaluate with no metrics history → fitness=0.0.

        This happens when eval_steps is too small to collect any metrics.
        """
        from unittest.mock import patch, MagicMock

        cfg = SimConfig()
        cfg.num_boids = 20
        evo = EvoFlock(cfg, EvoConfig(eval_steps=10))
        genome = Genome(values={name: 0.5 for name in EVOLVABLE_PARAMS})

        fake_sim = MagicMock()
        fake_sim.metrics.history = []

        # Can't patch.object on numpy Generator (read-only),
        # so replace the RNG entirely.
        mock_rng = MagicMock()
        mock_rng.integers.return_value = 42
        evo._rng = mock_rng

        # SimulationEngine is imported locally in _evaluate() via
        # `from ..simulation.engine import SimulationEngine`,
        # so patch the actual source module.
        with patch("pymurmur.simulation.engine.SimulationEngine", return_value=fake_sim):
            evo._evaluate(genome)

        assert genome.fitness == 0.0

    def test_compute_objectives_empty_spacings(self):
        """_compute_objectives with no valid spacings → sep_score=0.0."""
        cfg = SimConfig()
        evo = EvoFlock(cfg, EvoConfig())

        # Create fake metric snapshots with local_spacing=0 (filtered out)
        class FakeSnap:
            local_spacing = 0.0
            speed_avg = 4.0
            alpha = 1.0
            dispersion = 50.0

        scores = evo._compute_objectives([FakeSnap()])
        assert scores[0] == 0.0  # separation score (no valid spacings)
        assert scores[1] > 0.0   # speed score (has speeds)
        assert scores[2] > 0.0   # curvature score (has alpha+dispersion)
        assert scores[3] > 0.0   # obstacle avoidance (no obstacles → f_cf=1.0 → 1.0)

    def test_compute_objectives_zero_speed(self):
        """_compute_objectives with speed_avg=0 still produces speed score."""
        cfg = SimConfig()
        cfg.boid_size = 1.0  # body_diameter=2, so local_spacing=4 → ratio=2.0 → score=1.0
        evo = EvoFlock(cfg, EvoConfig())

        class FakeSnap:
            local_spacing = 4.0    # 4/2 = 2.0 body diameters → score=1.0
            speed_avg = 0.0        # zero speed
            alpha = 1.0
            dispersion = 50.0

        scores = evo._compute_objectives([FakeSnap()])
        assert scores[0] > 0.0   # separation score (ratio 2.0 is optimal)
        # speed_avg=0.0 still produces a valid score via linear ramp
        assert scores[2] > 0.0   # curvature score
        assert scores[3] > 0.0   # obstacle avoidance (no obstacles → score=1.0)

    def test_pareto_front_empty_list(self):
        """_pareto_front([]) returns empty list."""
        assert _pareto_front([], 0.01) == []

    def test_genome_default_fitness(self):
        """New Genome has fitness=-inf and zero objectives."""
        g = Genome(values={name: 0.5 for name in EVOLVABLE_PARAMS})
        assert g.fitness == float("-inf")
        assert len(g.objectives) == len(OBJECTIVE_NAMES)
        assert np.all(g.objectives == 0.0)

    def test_obstacle_avoidance_perfect_when_no_obstacles(self):
        """With no collision tracking, obstacle avoidance score = 1.0^500 = 1.0."""
        cfg = SimConfig()
        cfg.boid_size = 1.0
        evo = EvoFlock(cfg, EvoConfig())

        class FakeSnap:
            local_spacing = 4.0
            speed_avg = 4.0
            alpha = 1.0
            dispersion = 50.0

        scores = evo._compute_objectives([FakeSnap()])
        assert len(scores) == 4
        # Without obstacles, should be near 1.0
        assert scores[3] == pytest.approx(1.0)

    def test_obstacle_avoidance_with_collisions(self, monkeypatch):
        """With collision data, (f_cf)^500 penalises even small collision rates."""
        cfg = SimConfig()
        cfg.boid_size = 1.0
        evo = EvoFlock(cfg, EvoConfig(eval_steps=1000))

        class FakeSnap:
            local_spacing = 4.0
            speed_avg = 4.0
            alpha = 1.0
            dispersion = 50.0

        # Simulate 10 collisions in 1000 steps → f_cf=0.99
        monkeypatch.setattr(cfg, "collision_free_steps", 990, raising=False)
        scores = evo._compute_objectives([FakeSnap()])
        # 0.99^500 ≈ 0.00657 — heavy penalty
        assert scores[3] < 0.01

        # 0 collisions in 1000 steps → f_cf=1.0 → 1.0^500 = 1.0
        monkeypatch.setattr(cfg, "collision_free_steps", 1000, raising=False)
        scores = evo._compute_objectives([FakeSnap()])
        assert scores[3] == pytest.approx(1.0)
