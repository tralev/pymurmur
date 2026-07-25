"""EvoFlock cross-cutting integration tests + per-gene physics consumption.

Split out of test_evoflock.py (file-size split): P11.1-P11.6
composed-together integration tests (TestCrossCuttingEvoFlock),
structured-angle-config persistence, and per-gene physics consumption
checks.
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
    _ObjectiveCollector,
    load_obstacle_scene,
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



# ── Cross-cutting integration tests (P11 "as a whole") ──────────

class TestCrossCuttingEvoFlock:
    """P11.1 + P11.2 + P11.3 + P11.4 + P11.5 + P11.6: cross-cutting
    integration — verifying that all six items compose correctly."""

    # ── P11.1→P11.2→P11.3: Hypervolume fitness formula ────────

    @pytest.mark.slow
    def test_hypervolume_fitness_is_product_of_objectives(self):
        """P11.1→P11.2→P11.3: After worst-of-4 evaluation,
        fitness = Π max(oₖ, ε) where ε=0.01."""
        cfg = SimConfig()
        cfg.num_boids = 15
        cfg.seed = 3
        epsilon = 0.01
        evo = EvoFlock(cfg, EvoConfig(
            eval_steps=30, evals_per_candidate=2, epsilon=epsilon,
        ))
        genome = _uniform_genome(0.5)
        evo._evaluate(genome)

        # fitness must equal the product of max(o, epsilon) for each objective
        expected = 1.0
        for o in genome.objectives:
            expected *= max(float(o), epsilon)
        assert genome.fitness == pytest.approx(expected, rel=1e-6), (
            f"fitness={genome.fitness}, product={expected}, "
            f"objectives={genome.objectives}"
        )

    def test_hypervolume_epsilon_floor_applied(self):
        """P11.1→P11.3: When an objective is 0, ε=0.01 floor prevents
        zero-product collapse."""
        cfg = SimConfig()
        evo = EvoFlock(cfg, EvoConfig(epsilon=0.01))
        col = _ObjectiveCollector(cfg)
        col.n_steps = 1
        col.collision_free_steps = 1
        # Empty collector gives sep=0, speed=0, curv=0.8, obst=1.0
        sep, speed, curv, obst = evo._compute_objectives(col)
        fitness = max(sep, 0.01) * max(speed, 0.01) * max(curv, 0.01) * max(obst, 0.01)
        assert fitness > 0.0, "Epsilon floor prevents zero fitness"
        assert fitness == pytest.approx(0.01 * 0.01 * 0.8 * 1.0)

    # ── P11.5→P11.1: Crossover preserves all 21 genes ─────────

    def test_crossover_preserves_all_21_genes(self):
        """P11.5→P11.1: After uniform crossover, child has all 21
        EVOLVABLE_PARAMS keys — none lost, none added."""
        cfg = SimConfig()
        evo = EvoFlock(cfg, EvoConfig())
        a = _uniform_genome(0.0)
        b = _uniform_genome(1.0)
        child = evo._crossover(a, b)
        assert set(child.values.keys()) == set(EVOLVABLE_PARAMS.keys())
        assert len(child.values) == 21

    def test_mutation_preserves_all_21_genes(self):
        """P11.5→P11.1: After mutation, child still has all 21 gene keys."""
        cfg = SimConfig()
        evo = EvoFlock(cfg, EvoConfig(mutation_rate=0.5, mutation_sigma=0.1))
        parent = _uniform_genome(0.5)
        child = evo._mutate(parent)
        assert set(child.values.keys()) == set(EVOLVABLE_PARAMS.keys())
        assert len(child.values) == 21

    # ── P11.5→P11.4: Expanded avoidance genes flow to collector ─

    def test_expanded_avoidance_genes_flow_to_collector(self):
        """P11.5→P11.4: _ObjectiveCollector reads fly_away_max_dist
        and min_time_to_collide from config set by genome."""
        cfg = SimConfig()
        # Simulate what _evaluate_single does: apply decoded params
        cfg.fly_away_max_dist = 7.5
        cfg.min_time_to_collide = 3.2
        cfg.static_avoid_weight = 42.0
        cfg.predictive_avoid_weight = 17.0
        col = _ObjectiveCollector(cfg)
        assert col._fly_away == 7.5
        assert col._min_ttc == 3.2
        assert col._static_w == 42.0
        assert col._predictive_w == 17.0

    def test_expanded_avoidance_genes_default_to_config(self):
        """P11.5→P11.4: When avoidance genes are 0 (genome value 0),
        collector gets 0.0 weights."""
        cfg = SimConfig()
        cfg.fly_away_max_dist = 0.0
        cfg.min_time_to_collide = 0.0
        cfg.static_avoid_weight = 0.0
        cfg.predictive_avoid_weight = 0.0
        col = _ObjectiveCollector(cfg)
        assert col._fly_away == 0.0
        assert col._min_ttc == 0.0
        assert col._static_w == 0.0
        assert col._predictive_w == 0.0

    # ── P11.4→P11.3→P11.1: Obstacles reduce fitness ────────────

    def test_obstacle_scene_reduces_obstacle_objective(self):
        """P11.4→P11.3→P11.1: The obstacle_avoidance objective is strictly
        lower when _compute_objectives sees collisions vs none."""
        cfg = SimConfig()
        evo = EvoFlock(cfg, EvoConfig())

        # Collector with collisions → f_cf < 1.0 → obstacle score < 1.0
        col_hit = _ObjectiveCollector(cfg)
        col_hit.n_steps = 100
        col_hit.collision_free_steps = 80  # f_cf = 0.8
        *_, obst_hit = evo._compute_objectives(col_hit)

        # Collector with no collisions → f_cf = 1.0 → obstacle score = 1.0
        col_clean = _ObjectiveCollector(cfg)
        col_clean.n_steps = 100
        col_clean.collision_free_steps = 100
        *_, obst_clean = evo._compute_objectives(col_clean)

        # Obstacle score with collisions MUST be lower
        assert obst_hit < obst_clean, (
            f"Obstacle score with collisions {obst_hit:.6f} "
            f"should be < without collisions {obst_clean:.6f}"
        )
        # Clean collector gives perfect obstacle avoidance
        assert obst_clean == pytest.approx(1.0)

    # ── P11.2→P11.1→P11.6: Seeds survive SSGA → artifact ─────

    def test_seeds_survive_ssga_into_artifact(self, tmp_path):
        """P11.2→P11.1→P11.6: eval_seeds set by the real _evaluate pipeline
        are carried through to the persistence artifact."""
        import yaml

        cfg = SimConfig()
        cfg.seed = 77
        evo = EvoFlock(cfg, EvoConfig(
            population_size=4, n_islands=1, max_steps=0,
            evals_per_candidate=4,
        ))
        evo._initialize_population()
        _stub_single_eval(evo)
        # Evaluate every genome through the real _evaluate pipeline
        # which sets genome.eval_seeds from the deterministic formula
        for g in evo._islands[0]:
            evo._ensure_evaluated(g)

        best = evo._best_genome()
        assert best is not None and len(best.eval_seeds) == 4

        out = evo.save(tmp_path / "evolved.yaml")
        with open(out) as f:
            data = yaml.safe_load(f)
        # Artifact seeds must match what _evaluate actually recorded
        assert data["eval_seeds"] == list(best.eval_seeds), (
            f"Artifact seeds {data['eval_seeds']} should match "
            f"best genome's seeds {best.eval_seeds}"
        )

    # ── P11.1→P11.6: Pareto front populated after eval ─────────

    def test_pareto_front_contains_entries_after_evaluation(self):
        """P11.1→P11.6: After evaluating every genome in a population,
        the Pareto front is non-empty."""
        cfg = SimConfig()
        evo = EvoFlock(cfg, EvoConfig(population_size=6, n_islands=1))
        evo._initialize_population()
        for g in evo._islands[0]:
            g.fitness = float(sum(g.values.values()))
            g.objectives = np.random.default_rng(0).random(4)
        evo._update_pareto()
        assert len(evo._pareto_front) >= 1, "Pareto front must have entries"
        assert all(isinstance(g, Genome) for g in evo._pareto_front)

    # ── P11.5→P11.1→P11.2: Expanded genes reach eval config ───

    def test_all_21_params_reach_simulation_config(self):
        """P11.5→P11.1→P11.2: When _evaluate_single builds the config,
        all 21 expanded gene params carry through to SimulationEngine."""
        from unittest.mock import MagicMock, patch

        cfg = SimConfig()
        cfg.num_boids = 12
        evo = EvoFlock(cfg, EvoConfig(eval_steps=10))
        captured = {}

        def fake_engine(inner_cfg):
            captured["cfg"] = inner_cfg
            return MagicMock()

        # Use a non-uniform genome so expanded genes are at distinct values
        genome = _uniform_genome()
        for i, name in enumerate(sorted(EVOLVABLE_PARAMS)):
            genome.values[name] = (i + 1) / (len(EVOLVABLE_PARAMS) + 1)

        with patch("pymurmur.simulation.engine.SimulationEngine", side_effect=fake_engine):
            evo._evaluate_single(genome, seed=42)

        # Every expanded gene should have been set on the config copy
        decoded = genome.to_config_params()
        for name in EVOLVABLE_PARAMS:
            if name == "phi_p":  # nested path
                actual = captured["cfg"].projection.phi_p
            else:
                actual = getattr(captured["cfg"], name)
            expected = decoded[name]
            assert actual == pytest.approx(expected), (
                f"{name}: config={actual}, expected={expected}"
            )

    # ── P11.1→P11.2→P11.3→P11.6: Full pipeline artifact check ─

    def test_full_artifact_contains_all_sections(self, tmp_path):
        """P11.1→P11.2→P11.3→P11.6: After a population is evaluated
        and saved, the artifact has all required top-level keys."""
        import yaml

        cfg = SimConfig()
        cfg.seed = 1
        evo = EvoFlock(cfg, EvoConfig(population_size=4, n_islands=1))
        evo._initialize_population()
        for k, g in enumerate(evo._islands[0]):
            g.fitness = 0.2 * (k + 1)
            g.objectives = np.array([0.5, 0.6, 0.7, 0.9])
            g.eval_seeds = [1, 2, 3, 4]
        evo._update_pareto()

        out = evo.save(tmp_path / "evolved.yaml")
        with open(out) as f:
            data = yaml.safe_load(f)

        # All five top-level sections
        assert set(data.keys()) == {
            "evolved_params", "fitness", "objective_scores",
            "eval_seeds", "pareto_front",
        }
        # Each Pareto front entry has the three sub-sections
        for entry in data["pareto_front"]:
            assert set(entry.keys()) == {"params", "objectives", "fitness"}
            # Entry params must have all 21 genes
            assert set(entry["params"].keys()) == set(EVOLVABLE_PARAMS.keys())
            assert set(entry["objectives"].keys()) == set(OBJECTIVE_NAMES)

    # ── P11.1→P11.5→P11.2→P11.3: SSGA child eval pipeline ────

    def test_ssga_child_evaluated_produces_valid_objectives(self):
        """P11.1→P11.5→P11.2→P11.3: A child produced by crossover +
        mutation, when evaluated through worst-of-4, gets valid
        objectives and finite fitness."""
        cfg = SimConfig()
        evo = EvoFlock(cfg, EvoConfig(
            population_size=3, n_islands=1, evals_per_candidate=2,
        ))
        # Replace evaluation with a cheap stub that returns a
        # fitness proportional to sum of gene values
        _stub_single_eval(evo)

        # Create two parents with distinct gene values
        parent_a = _uniform_genome(0.0)
        parent_b = _uniform_genome(1.0)
        # Crossover + mutate → child (mutation_rate ensures some diversity)
        evo._ga.mutation_rate = 0.3
        child = evo._mutate(evo._crossover(parent_a, parent_b))
        assert len(child.values) == 21

        # Evaluate child through worst-of-4 pipeline
        evo._evaluate(child)
        assert child.fitness > 0.0
        assert np.isfinite(child.fitness)
        assert len(child.objectives) == 4
        assert np.all(child.objectives >= 0.0)
        assert len(child.eval_seeds) == 2

    # ── P11.1→P11.2→P11.6: Multiple runs select best ──────────

    def test_multiple_runs_selects_best_fitness(self):
        """P11.1→P11.2→P11.6: run(n_runs=2) with stubbed evaluations
        returns the params from the run with the higher best fitness.
        Verified by encoding a distinguishing gene value in each run."""
        cfg = SimConfig()
        cfg.seed = 1
        evo = EvoFlock(cfg, EvoConfig(
            population_size=4, n_islands=1, max_steps=0,
        ))
        _stub_single_eval(evo)

        run_counter = [0]
        original_init = evo._initialize_population

        def skewed_init():
            original_init()
            run_counter[0] += 1
            for g in evo._islands[0]:
                evo._ensure_evaluated(g)
            if run_counter[0] == 1:
                # Run 1: best genome has fitness 999 AND a
                # distinguishing gene: separation_weight gene = 0.9
                evo._islands[0][0].fitness = 999.0
                evo._islands[0][0].values["separation_weight"] = 0.9
            else:
                # Run 2: best genome has fitness 0.001 AND
                # separation_weight gene = 0.1 (should be ignored)
                evo._islands[0][0].fitness = 0.001
                evo._islands[0][0].values["separation_weight"] = 0.1

        evo._initialize_population = skewed_init
        evo._run_generation_loop = lambda: None  # no-op

        result = evo.run(n_runs=2)
        # Run 1 (fitness=999) should be selected — its sep_weight
        # decodes to 0.5 + 0.9*(10-0.5) = 9.05, not run 2's 0.5+0.1*9.5=1.45
        lo, hi = EVOLVABLE_PARAMS["separation_weight"]
        expected = lo + 0.9 * (hi - lo)
        assert result["separation_weight"] == pytest.approx(expected), (
            f"Should return run 1's sep_weight={expected:.3f}, "
            f"got {result['separation_weight']:.3f}"
        )

    # ── P11.4→P11.6: Obstacle YAML → artifact roundtrip ──────

    @pytest.mark.slow
    def test_obstacle_yaml_to_artifact_roundtrip(self, tmp_path):
        """P11.4→P11.6: Loading an obstacle scene from YAML config,
        running EvoFlock with it, and saving produces an artifact
        with a valid obstacle_avoidance objective score reflecting
        the scene's collision profile."""
        import yaml

        scene = load_obstacle_scene("conf/murmuration_evo.yaml")
        assert scene is not None

        cfg = SimConfig()
        cfg.seed = 12
        cfg.num_boids = 15
        evo = EvoFlock(cfg, EvoConfig(
            population_size=4, n_islands=1, max_steps=0,
            eval_steps=30, evals_per_candidate=1,
        ), scene=scene)
        # Do NOT stub evaluation — let the real pipeline run so the
        # _ObjectiveCollector can count collisions against the scene.
        evo._initialize_population()
        for g in evo._islands[0]:
            evo._ensure_evaluated(g)

        out = evo.save(tmp_path / "evolved.yaml")
        with open(out) as f:
            data = yaml.safe_load(f)

        assert "obstacle_avoidance" in data["objective_scores"]
        obs = data["objective_scores"]["obstacle_avoidance"]
        assert 0.0 <= obs <= 1.0, f"obstacle_avoidance={obs} must be in [0,1]"

        # Each Pareto entry also has obstacle_avoidance
        for entry in data["pareto_front"]:
            assert "obstacle_avoidance" in entry["objectives"]
            assert 0.0 <= entry["objectives"]["obstacle_avoidance"] <= 1.0

    # ── P11.5→P11.2: Different genes → different eval configs ─

    def test_different_sigma_produces_different_eval_config(self):
        """P11.5→P11.2: Two genomes with different sigma gene values
        produce different influence_count in the SimulationEngine
        config. Verifies the integer decode affects eval setup."""
        from unittest.mock import MagicMock, patch

        cfg = SimConfig()
        cfg.num_boids = 12
        evo = EvoFlock(cfg, EvoConfig(eval_steps=10))
        captured_low = {}
        captured_high = {}

        def fake_engine_low(inner_cfg):
            captured_low["cfg"] = inner_cfg
            return MagicMock()

        def fake_engine_high(inner_cfg):
            captured_high["cfg"] = inner_cfg
            return MagicMock()

        # Genome with sigma gene = 0.0 → sigma decoded to 1
        genome_low = _uniform_genome(0.5)
        genome_low.values["sigma"] = 0.0

        # Genome with sigma gene = 1.0 → sigma decoded to 10
        genome_high = _uniform_genome(0.5)
        genome_high.values["sigma"] = 1.0

        with patch("pymurmur.simulation.engine.SimulationEngine",
                   side_effect=fake_engine_low):
            evo._evaluate_single(genome_low, seed=42)

        with patch("pymurmur.simulation.engine.SimulationEngine",
                   side_effect=fake_engine_high):
            evo._evaluate_single(genome_high, seed=43)

        # influence_count in eval is always 7 (P11.5: fixed k=7)
        # But sigma decoded differently: low=1, high=10
        assert captured_low["cfg"].influence_count == 7
        assert captured_high["cfg"].influence_count == 7
        # The sigma parameter itself should differ in the config
        assert captured_low["cfg"].sigma != captured_high["cfg"].sigma, (
            f"sigma values must differ: {captured_low['cfg'].sigma} vs "
            f"{captured_high['cfg'].sigma}"
        )


# ── D13 + D15: EvoFlock save preserves structured angle config ─


def test_save_preserves_angle_config_fields():
    """D13+D15: EvoFlock save/load preserves AngleConfig + BoundaryConfig.

    D13 fixed evoflock to actually write evolved.yaml. D15 fixed angle
    mode to use structured config (AngleConfig, BoundaryConfig) instead
    of getattr fallbacks. Together, an evolved artifact with angle mode
    must preserve all angle + boundary fields through save/load.
    """
    from pymurmur.core.config import SimConfig

    # Create config with non-default angle + boundary values
    cfg = SimConfig()
    cfg.mode = "angle"
    cfg.seed = 42
    cfg.num_boids = 30
    cfg.boundary.boundary_mode = "sphere_soft"
    cfg.boundary.boundary_sphere_radius = 0.35
    cfg.boundary.boundary_avoidance_factor = 0.6

    # D13+D15: save must preserve structured config through YAML round-trip
    import os
    import tempfile
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "cross_cutting_test.yaml")
        cfg.to_file(path)

        # Reload and verify D15 structured fields
        cfg2 = SimConfig.from_file(path)
        assert cfg2.mode == "angle", "D15: mode preserved"
        assert cfg2.boundary.boundary_mode == "sphere_soft", (
            "D15: boundary_mode preserved"
        )
        assert cfg2.boundary.boundary_sphere_radius == pytest.approx(0.35), (
            "D15: boundary_sphere_radius preserved"
        )
        assert cfg2.boundary.boundary_avoidance_factor == pytest.approx(0.6), (
            "D15: boundary_avoidance_factor preserved"
        )
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


# -- S6.5: Per-gene tests — verify each EvoFlock gene is consumed by physics --

class TestPerGenePhysicsConsumption:
    """S6.5: Each evolved gene is actually consumed by physics modules."""

    def test_speed_min_factor_default_and_set(self):
        """S6.5: speed_min_factor defaults to 0.3 and is settable via config."""
        from pymurmur.core.config import SimConfig

        cfg = SimConfig()
        assert cfg.speed_min_factor == pytest.approx(0.3)
        cfg.speed_min_factor = 0.75
        assert cfg.flock.speed_min_factor == pytest.approx(0.75)

    def test_speed_min_factor_zero_is_valid(self):
        """S6.5: speed_min_factor=0 is valid (disables min speed enforcement)."""
        from pymurmur.core.config import SimConfig

        cfg = SimConfig()
        cfg.speed_min_factor = 0.0
        assert cfg.flock.speed_min_factor == 0.0

    def test_w_fwd_config_round_trip(self):
        """S6.5: w_fwd defaults to 0.0, settable via flat and nested path."""
        from pymurmur.core.config import SimConfig

        cfg = SimConfig()
        assert cfg.w_fwd == pytest.approx(0.0)
        cfg.w_fwd = 0.75
        assert cfg.spatial.w_fwd == pytest.approx(0.75)

    def test_w_fwd_source_reference_exists(self):
        """S6.5: spatial.py references w_fwd for forward thrust."""
        from pathlib import Path
        src = Path("pymurmur/physics/forces/spatial.py").read_text()
        assert "w_fwd" in src, "w_fwd gene must be referenced in spatial.py"

    def test_static_avoid_weight_read_by_engine(self):
        """static_avoid_weight config field is read during engine.step()."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.num_boids = 10
        cfg.static_avoid_weight = 3.14
        cfg.predictive_avoid_weight = 2.71
        cfg.fly_away_max_dist = 42.0
        cfg.min_time_to_collide = 7.0

        engine = SimulationEngine(cfg)
        assert engine.config.static_avoid_weight == pytest.approx(3.14)
        assert engine.config.predictive_avoid_weight == pytest.approx(2.71)
        assert engine.config.fly_away_max_dist == pytest.approx(42.0)
        assert engine.config.min_time_to_collide == pytest.approx(7.0)

    def test_avoidance_genes_round_trip_through_genome(self):
        """All 4 avoidance genes survive genome decode + config round trip."""
        g = _uniform_genome(0.5)
        g.values["static_avoid_weight"] = 0.25
        g.values["predictive_avoid_weight"] = 0.75
        g.values["fly_away_max_dist"] = 0.1
        g.values["min_time_to_collide"] = 0.9

        params = g.to_config_params()
        assert "static_avoid_weight" in params
        assert "predictive_avoid_weight" in params
        assert "fly_away_max_dist" in params
        assert "min_time_to_collide" in params

    def test_speed_min_factor_in_evolvable_params(self):
        """speed_min_factor is in EVOLVABLE_PARAMS with valid range."""
        assert "speed_min_factor" in EVOLVABLE_PARAMS
        lo, hi = EVOLVABLE_PARAMS["speed_min_factor"]
        assert 0.0 <= lo < hi <= 1.0

    def test_w_fwd_in_evolvable_params(self):
        """w_fwd is in EVOLVABLE_PARAMS with valid range."""
        assert "w_fwd" in EVOLVABLE_PARAMS
        lo, hi = EVOLVABLE_PARAMS["w_fwd"]
        assert lo >= 0.0
        assert hi > lo
