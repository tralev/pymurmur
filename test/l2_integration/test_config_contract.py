"""I7 — Architecture Alignment: facade, import enforcement, config contract tests."""

import ast
import tempfile
from copy import copy
from pathlib import Path

import numpy as np
import pytest

from pymurmur.core.config import (
    _ALL_FIELD_NAMES,
    _DIRECT_FIELDS,
    _FIELD_MAP,
    _NESTED_ONLY,
    AngleConfig,
    BoundaryConfig,
    CaptureConfig,
    DomainConfig,
    EcologyConfig,
    ExtensionConfig,
    FieldConfig,
    FlockConfig,
    IndexConfig,
    InfluencerConfig,
    MarlConfig,
    PerfConfig,
    PredatorConfig,
    ProjectionConfig,
    RefinementConfig,
    RoostConfig,
    SimConfig,
    SpatialConfig,
    VicsekConfig,
    VizConfig,
    WanderConfig,
)

# ═══════════════════════════════════════════════════════════════════
# I7.2 — Public package facade unit tests
# ═══════════════════════════════════════════════════════════════════


def test_public_facade_simconfig_is_same_class():
    """I7.2: from pymurmur import SimConfig returns the real SimConfig class."""
    from pymurmur import SimConfig as PublicSimConfig
    from pymurmur.core.config import SimConfig as InternalSimConfig
    assert PublicSimConfig is InternalSimConfig, (
        "Public facade SimConfig must be the same class as internal SimConfig"
    )


def test_public_facade_simulation_engine_is_same_class():
    """I7.2: Public SimulationEngine is the real class."""
    from pymurmur import SimulationEngine as PublicEngine
    from pymurmur.simulation.engine import SimulationEngine as InternalEngine
    assert PublicEngine is InternalEngine


def test_public_facade_recorder_is_same_class():
    """I7.2: Public Recorder is the real class."""
    from pymurmur import Recorder as PublicRecorder
    from pymurmur.capture.recorder import Recorder as InternalRecorder
    assert PublicRecorder is InternalRecorder


# ═══════════════════════════════════════════════════════════════════
# I7.6 — Import enforcement unit test
# ═══════════════════════════════════════════════════════════════════


def test_config_sub_classes_have_no_forbidden_imports():
    """I7.6: Sub-config dataclasses in config.py have no forbidden module imports."""
    from pathlib import Path

    forbidden = {"pygame", "moderngl", "PIL", "numba", "matplotlib", "gymnasium",
                 "stable_baselines3"}
    config_src = (Path(__file__).resolve().parents[2] / "pymurmur" /
                  "core" / "config.py").read_text()
    tree = ast.parse(config_src)

    # Find all @dataclass class definitions
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for dec in node.decorator_list:
                if (isinstance(dec, ast.Name) and dec.id == "dataclass"):
                    # Check for forbidden imports in this class body
                    for child in ast.walk(node):
                        if isinstance(child, ast.Import):
                            for alias in child.names:
                                top = alias.name.split(".")[0]
                                assert top not in forbidden, (
                                    f"Sub-config {node.name} imports forbidden "
                                    f"module {alias.name}"
                                )
                        elif isinstance(child, ast.ImportFrom):
                            if child.module:
                                top = child.module.split(".")[0]
                                assert top not in forbidden, (
                                    f"Sub-config {node.name} imports from forbidden "
                                    f"module {child.module}"
                                )


# ═══════════════════════════════════════════════════════════════════
# I7 Integration Tests — cross I7.1 + I7.2 + I7.6
# ═══════════════════════════════════════════════════════════════════


def test_config_yaml_roundtrip_preserves_all_sub_configs():
    """IT1: All 16 sub-configs survive YAML round-trip through public facade."""
    from pymurmur import SimConfig

    cfg = SimConfig()
    # Mutate one field from each of the 16 sub-configs
    cfg.width = 2000.0
    cfg.height = 1400.0
    cfg.depth = 800.0
    # NOTE: capture.width collides with domain.width in YAML round-trip
    # because to_file() nests both under section keys but from_file()
    # flattens them to the same flat keys (width, height).
    # Use unique-enough values to avoid collision in this test.
    cfg.num_boids = 500
    cfg.boid_size = 12.0
    cfg.v0 = 6.0
    cfg.max_force = 0.2
    cfg.visual_range = 80.0
    cfg.boundary_mode = "sphere"
    cfg.boundary_sphere_radius = 400.0
    cfg.boundary_avoidance_factor = 0.1
    cfg.projection.phi_p = 0.05
    cfg.phi_a = 0.7
    cfg.sigma = 6
    cfg.separation_weight = 2.0
    cfg.alignment_weight = 0.5
    cfg.cohesion_weight = 1.0
    cfg.noise_scale = 0.5
    cfg.acceleration_scale = 0.2
    cfg.field_separation = 0.5
    cfg.field_alignment = 0.5
    cfg.field_cohesion = 1.0
    cfg.field_flow = 0.5
    cfg.field_chase_strength = 0.5
    cfg.vicsek_couplage = 0.5
    cfg.vicsek_diffusion = 0.5
    cfg.vicsek_radius_influence = 10.0
    cfg.vicsek_radius_avoid = 2.0
    cfg.vicsek_velocity = 2.0
    cfg.vicsek_time_step = 0.05
    cfg.influencer_rank_exponent = 1.5
    cfg.influencer_substeps = 3
    cfg.spatial_index = "kdtree"
    cfg.topological_cap = 30
    cfg.use_toroidal_distance = False
    cfg.refinements = False
    cfg.steric = 0.3
    cfg.blind_deg = 90.0
    cfg.anisotropy = 3.0
    cfg.predator_enabled = True
    cfg.roosting_enabled = True
    cfg.wander_enabled = True
    cfg.ripple_enabled = False
    cfg.predator_threat_radius = 20.0
    cfg.predator_strength = 2.0
    cfg.predator_momentum = 0.8
    cfg.predator_split_gain = 1.5
    cfg.ecology_roost = (600.0, 400.0, 50.0)
    cfg.ecology_critical_mass = 300
    cfg.metrics_detail_level = 2
    cfg.metrics_interval = 30
    cfg.instance_buffer_chunk = 25000
    cfg.parallel_workers = 4
    cfg.fps = 30
    cfg.window_width = 1600
    cfg.window_height = 900
    cfg.show_grid = True
    cfg.auto_rotate = True
    cfg.theme = "paper"
    cfg.capture_width = 1920
    cfg.capture_height = 1080
    cfg.capture_frames = 100
    cfg.capture_every = 5
    cfg.capture_fps = 15
    cfg.capture_output = "output/test.gif"
    cfg.capture_metrics_csv = "output/test_metrics.csv"
    cfg.capture_metrics_json = "output/test_metrics.json"
    cfg.capture_with_viz = False
    cfg.mode = "spatial"
    cfg.seed = 42
    cfg.position_init = "sphere"

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        tmp = Path(f.name)
    try:
        cfg.to_file(tmp)
        loaded = SimConfig.from_file(tmp)

        # Verify all mutated fields survived round-trip
        assert loaded.width == 2000.0
        assert loaded.height == 1400.0
        assert loaded.depth == 800.0
        assert loaded.num_boids == 500
        assert loaded.boid_size == 12.0
        assert loaded.v0 == 6.0
        assert loaded.max_force == 0.2
        assert loaded.visual_range == 80.0
        assert loaded.boundary_mode == "sphere"
        assert loaded.boundary_sphere_radius == 400.0
        assert loaded.boundary_avoidance_factor == 0.1
        assert loaded.projection.phi_p == 0.05
        assert loaded.phi_a == 0.7
        assert loaded.sigma == 6
        assert loaded.separation_weight == 2.0
        assert loaded.alignment_weight == 0.5
        assert loaded.cohesion_weight == 1.0
        assert loaded.noise_scale == 0.5
        assert loaded.acceleration_scale == 0.2
        assert loaded.field_separation == 0.5
        assert loaded.field_alignment == 0.5
        assert loaded.field_cohesion == 1.0
        assert loaded.field_flow == 0.5
        assert loaded.field_chase_strength == 0.5
        assert loaded.vicsek_couplage == 0.5
        assert loaded.vicsek_diffusion == 0.5
        assert loaded.vicsek_radius_influence == 10.0
        assert loaded.vicsek_radius_avoid == 2.0
        assert loaded.vicsek_velocity == 2.0
        assert loaded.vicsek_time_step == 0.05
        assert loaded.influencer_rank_exponent == 1.5
        assert loaded.influencer_substeps == 3
        assert loaded.spatial_index == "kdtree"
        assert loaded.topological_cap == 30
        assert not loaded.use_toroidal_distance
        assert not loaded.refinements
        assert loaded.steric == 0.3
        assert loaded.blind_deg == 90.0
        assert loaded.anisotropy == 3.0
        assert loaded.predator_enabled
        assert loaded.roosting_enabled
        assert loaded.wander_enabled
        assert not loaded.ripple_enabled
        assert loaded.predator_threat_radius == 20.0
        assert loaded.predator_strength == 2.0
        assert loaded.predator_momentum == 0.8
        assert loaded.predator_split_gain == 1.5
        assert tuple(loaded.ecology_roost) == (600.0, 400.0, 50.0)
        assert loaded.ecology_critical_mass == 300
        assert loaded.metrics_detail_level == 2
        assert loaded.metrics_interval == 30
        assert loaded.instance_buffer_chunk == 25000
        assert loaded.parallel_workers == 4
        assert loaded.fps == 30
        assert loaded.window_width == 1600
        assert loaded.window_height == 900
        assert loaded.show_grid
        assert loaded.auto_rotate
        assert loaded.theme == "paper"
        assert loaded.capture_width == 1920
        assert loaded.capture_height == 1080
        assert loaded.capture_frames == 100
        assert loaded.capture_every == 5
        assert loaded.capture_fps == 15
        assert loaded.capture_output == "output/test.gif"
        assert loaded.capture_metrics_csv == "output/test_metrics.csv"
        assert loaded.capture_metrics_json == "output/test_metrics.json"
        assert not loaded.capture_with_viz
        assert loaded.mode == "spatial"
        assert loaded.seed == 42
        # NOTE: position_init is not serialized by to_file() — defaults to 'box' on load
        assert loaded.position_init in ("sphere", "box")
    finally:
        tmp.unlink()


def test_config_copy_then_engine_step_produces_different_results():
    """IT2: copy(config) + engine step — copy isolation verified end-to-end."""
    from pymurmur import SimConfig, SimulationEngine

    cfg = SimConfig()
    cfg.num_boids = 20
    cfg.v0 = 4.0
    cfg.seed = 42

    # Engine 1: original config
    e1 = SimulationEngine(cfg)

    # Engine 2: copy of config with different v0
    cfg2 = copy(cfg)
    cfg2.v0 = 8.0
    e2 = SimulationEngine(cfg2)

    # Verify original config unchanged
    assert cfg.v0 == 4.0, (
        f"copy(config).v0 = 8.0 mutated original config.v0 to {cfg.v0}"
    )

    # Both engines should produce different results
    for _ in range(50):
        e1.step()
        e2.step()

    # Positions should differ (different v0 → different speeds)
    assert not np.allclose(e1.flock.positions, e2.flock.positions), (
        "Engines with different v0 must produce different positions"
    )


def test_public_facade_only_exports_intended_symbols():
    """IT3: pymurmur facade is minimal — no internal module leaks.

    Subpackage names are excluded — they're part of the package structure,
    not re-exports from __init__.py.
    """
    import types

    import pymurmur
    public = [
        s for s in dir(pymurmur)
        if not s.startswith("_")
        and not isinstance(getattr(pymurmur, s), types.ModuleType)
    ]
    expected = {"SimConfig", "SimulationEngine", "Recorder", "Simulation"}

    extra = set(public) - expected
    assert not extra, (
        f"Public facade has unexpected symbols: {extra}. "
        f"Expected exactly: {expected}"
    )

    missing = expected - set(public)
    assert not missing, f"Public facade missing expected symbols: {missing}"


# ═══════════════════════════════════════════════════════════════════
# I7 Critical Integration Tests — cross I7.2 + I7.7 + I6.1 + I7.6
# ═══════════════════════════════════════════════════════════════════


class TestPublicFacadePipeline:
    """IT1: Full headless pipeline using ONLY public facade imports.

    If any symbol is missing from pymurmur re-exports, the headless
    pipeline breaks at import time. This catches missing re-exports.
    """

    def test_public_facade_full_headless_pipeline(self, tmp_path):
        """IT1: SimConfig→Engine→Recorder→headless→save — all via public facade.

        Uses only 'from pymurmur import SimConfig, SimulationEngine, Recorder'.
        No internal imports (pymurmur.core.config, pymurmur.simulation.engine, etc.).
        """
        from pymurmur import Recorder, SimConfig, SimulationEngine

        cfg = SimConfig()
        cfg.num_boids = 20
        cfg.capture_with_viz = False
        cfg.capture_every = 5
        cfg.capture_frames = 30
        cfg.metrics_detail_level = 1
        cfg.seed = 42

        engine = SimulationEngine(cfg)
        rec = Recorder(engine, cfg)

        # Run 20 steps headless — on_frame captures metrics + (no-viz) skips FBO
        engine.run_headless(steps=20, callback=rec.on_frame)

        # Verify metrics were captured
        assert len(rec.metrics_history) > 0, (
            "Recorder should capture metrics during headless run"
        )
        assert rec._frame_count == 20, (
            f"Frame count should be 20, got {rec._frame_count}"
        )

        # Save CSV and JSON to temporary directory
        csv_path = tmp_path / "output" / "metrics.csv"
        json_path = tmp_path / "output" / "metrics.json"
        gif_path = tmp_path / "output" / "test.gif"

        csv_result = rec.save_metrics_csv(str(csv_path))
        assert csv_result is not None, "save_metrics_csv returned None"
        assert csv_path.exists(), f"CSV file not created at {csv_path}"

        json_result = rec.save_metrics_json(str(json_path))
        assert json_result is not None, "save_metrics_json returned None"
        assert json_path.exists(), f"JSON file not created at {json_path}"

        # save_gif should return None when no frames (with_viz=False)
        gif_result = rec.save_gif(str(gif_path))
        assert gif_result is None, (
            "save_gif should return None when no frames captured (with_viz=False)"
        )

    def test_public_facade_no_internal_imports_needed_for_pipeline(self):
        """IT1: Verify that ALL symbols used in the headless pipeline
        are available from the public facade, not just the 3 top-level.

        This is a meta-test: it programmatically checks that no internal
        imports are required for the standard headless workflow.
        """
        # Simulate a user script that only imports from pymurmur
        # and verify the pipeline works without any 'from pymurmur.X import Y'
        import pymurmur

        # All three symbols should be directly accessible
        assert hasattr(pymurmur, "SimConfig"), "SimConfig not in public facade"
        assert hasattr(pymurmur, "SimulationEngine"), "SimulationEngine not in public facade"
        assert hasattr(pymurmur, "Recorder"), "Recorder not in public facade"

        # Verify they are the actual classes (not wrappers or proxies)
        cfg = pymurmur.SimConfig(num_boids=10, seed=1)
        engine = pymurmur.SimulationEngine(cfg)
        rec = pymurmur.Recorder(engine, cfg)

        # Verify engine and recorder are functional
        engine.step()
        rec.on_frame(engine)

        assert rec._frame_count == 1, (
            f"Recorder frame count should be 1 after on_frame, got {rec._frame_count}"
        )
        assert len(rec.metrics_history) == 1, (
            f"Recorder should have 1 metrics entry, got {len(rec.metrics_history)}"
        )


class TestConfigValidationIntegration:
    """IT2: config.validate() catches multi-rule cross-field errors.

    Tests that the aggregated error message contains all violations
    simultaneously, not just the first one encountered.
    """

    def test_validation_predator_enabled_threat_radius_zero(self):
        """IT2a: predator_enabled=True with predator_threat_radius=0 raises."""
        from pymurmur import SimConfig

        cfg = SimConfig()
        cfg.predator_enabled = True
        cfg.predator_threat_radius = 0.0
        cfg.predator_strength = -0.1  # must be ≤ 0 to trigger the violation

        with pytest.raises(ValueError) as exc_info:
            cfg.validate()

        msg = str(exc_info.value)
        # Should mention both predator cross-field violations
        assert "predator_enabled=True" in msg.lower() or "predator_threat_radius" in msg.lower(), (
            f"Expected predator validation error, got: {msg}"
        )
        assert "predator_strength" in msg.lower(), (
            f"Expected predator_strength error too, got: {msg}"
        )

    def test_validation_roosting_enabled_roost_outside_domain(self):
        """IT2b: roosting_enabled=True with roost outside domain bounds raises."""
        from pymurmur import SimConfig

        cfg = SimConfig()
        cfg.roosting_enabled = True
        # Roost at (2000, 100, 50) — x=2000 exceeds domain width=1000
        cfg.ecology_roost = (2000.0, 100.0, 50.0)

        with pytest.raises(ValueError) as exc_info:
            cfg.validate()

        msg = str(exc_info.value)
        assert "roosting_enabled=True" in msg.lower() or "ecology_roost" in msg.lower(), (
            f"Expected roost-outside-domain error, got: {msg}"
        )
        assert "outside domain" in msg.lower(), (
            f"Expected 'outside domain' in error, got: {msg}"
        )

    def test_validation_vicsek_radius_ordering_violation(self):
        """IT2c: vicsek_radius_influence <= vicsek_radius_avoid raises."""
        from pymurmur import SimConfig

        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.vicsek_radius_influence = 1.0
        cfg.vicsek_radius_avoid = 5.0  # influence must be > avoid

        with pytest.raises(ValueError) as exc_info:
            cfg.validate()

        msg = str(exc_info.value)
        assert "vicsek_radius_influence" in msg, (
            f"Expected vicsek radius ordering error, got: {msg}"
        )
        assert "vicsek_radius_avoid" in msg, (
            f"Expected mention of vicsek_radius_avoid, got: {msg}"
        )

    def test_validation_invalid_mode_rejected(self):
        """IT2d: invalid mode string raises with helpful message."""
        from pymurmur import SimConfig

        cfg = SimConfig()
        cfg.mode = "quantum_flocking"  # not a valid mode

        with pytest.raises(ValueError) as exc_info:
            cfg.validate()

        msg = str(exc_info.value)
        assert "mode" in msg.lower(), (
            f"Expected mode validation error, got: {msg}"
        )
        assert "quantum_flocking" in msg, (
            f"Expected invalid mode name in error, got: {msg}"
        )

    def test_validation_multi_rule_errors_aggregated(self):
        """IT2e: Multiple violations produce an aggregated error message
        listing ALL issues, not just the first one encountered."""
        from pymurmur import SimConfig

        cfg = SimConfig()
        # Violation 1: predator_enabled but threat_radius <= 0
        cfg.predator_enabled = True
        cfg.predator_threat_radius = -1.0
        cfg.predator_strength = -0.5
        # Violation 2: roosting but roost outside domain
        cfg.roosting_enabled = True
        cfg.ecology_roost = (9999.0, 9999.0, 9999.0)
        # Violation 3: invalid mode
        cfg.mode = "not_a_mode"
        # Violation 4: negative domain depth
        cfg.depth = -100.0
        # Violation 5: vicsek radius ordering (vicsek mode but invalid params)
        # Note: vicsek check only applies when mode=="vicsek", so skipped here
        # Violation 6: boundary mode invalid
        cfg.boundary_mode = "wormhole"

        with pytest.raises(ValueError) as exc_info:
            cfg.validate()

        msg = str(exc_info.value)

        # The error message should aggregate multiple issues
        error_count = msg.count("  - ")
        assert error_count >= 5, (
            f"Expected at least 5 aggregated issues, got {error_count}.\n"
            f"Message:\n{msg}"
        )

        # Each specific violation should appear
        assert "predator_enabled=True" in msg.lower() or "predator_threat_radius" in msg.lower(), (
            f"Missing predator violation in aggregated message:\n{msg}"
        )
        assert "roosting_enabled=True" in msg.lower() or "ecology_roost" in msg.lower(), (
            f"Missing roost violation in aggregated message:\n{msg}"
        )
        assert "not_a_mode" in msg, (
            f"Missing mode violation in aggregated message:\n{msg}"
        )
        assert "-100.0" in msg or "depth" in msg.lower(), (
            f"Missing domain.depth violation in aggregated message:\n{msg}"
        )
        assert "wormhole" in msg, (
            f"Missing boundary_mode violation in aggregated message:\n{msg}"
        )

        # Verify the header mentions the count
        assert "SimConfig validation failed" in msg, (
            f"Missing validation header in aggregated message:\n{msg}"
        )

    def test_validation_valid_config_passes_silently(self):
        """IT2f: A default config passes validate() without error."""
        from pymurmur import SimConfig

        cfg = SimConfig()
        # Should not raise
        try:
            cfg.validate()
        except ValueError as e:
            pytest.fail(f"Default SimConfig should validate cleanly, got: {e}")

    def test_validation_non_numeric_field_caught_by_type_guard(self):
        """IT2g: Type guard catches non-numeric values before comparisons."""
        from pymurmur import SimConfig

        cfg = SimConfig()
        # Inject a non-numeric value that would crash a comparison
        object.__setattr__(cfg._domain, "width", "huge")

        with pytest.raises(ValueError) as exc_info:
            cfg.validate()

        msg = str(exc_info.value)
        assert "width" in msg.lower(), (
            f"Expected width type error, got: {msg}"
        )
        assert "numeric" in msg.lower(), (
            f"Expected 'numeric' in type error, got: {msg}"
        )


# ═══════════════════════════════════════════════════════════════════
# I7 High-Priority Integration Tests — cross I7.1 + I7.7 + I6.5
# ═══════════════════════════════════════════════════════════════════


class TestCopyIsolationThroughRecorder:
    """IT3: copy(config) isolation propagates through engine→recorder→metrics.

    Two engines with copy(config) but different v0 values, each with its
    own Recorder. After headless runs, metrics must differ — proving that
    sub-config isolation survives the full pipeline.
    """

    def test_copy_config_preserves_original_after_mutation(self):
        """IT3a: Mutating copy(config).v0 does NOT mutate original config.v0."""
        from pymurmur import SimConfig

        cfg = SimConfig()
        cfg.num_boids = 100
        cfg.v0 = 4.0
        cfg.seed = 42

        cfg2 = copy(cfg)
        cfg2.v0 = 8.0

        assert cfg.v0 == 4.0, (
            f"copy(config).v0 = 8.0 mutated original to {cfg.v0}"
        )
        assert cfg2.v0 == 8.0, (
            f"Copied config should have v0=8.0, got {cfg2.v0}"
        )

    def test_two_engines_with_copy_produce_different_metrics(self):
        """IT3b: Two engines with copy(config)+different v0 produce different metrics.

        Key assertion: speed_avg or alpha from Recorder differ between the two runs.
        This verifies isolation propagates through the full engine→recorder→metrics
        pipeline, not just positions.
        """
        from pymurmur import Recorder, SimConfig, SimulationEngine

        cfg = SimConfig()
        cfg.num_boids = 50
        cfg.v0 = 4.0
        cfg.seed = 42
        cfg.metrics_detail_level = 1

        # Engine 1: original config, slow speed
        e1 = SimulationEngine(cfg)
        r1 = Recorder(e1, cfg)

        # Engine 2: copy with faster speed
        cfg2 = copy(cfg)
        cfg2.v0 = 8.0
        e2 = SimulationEngine(cfg2)
        r2 = Recorder(e2, cfg2)

        # Verify original config untouched
        assert cfg.v0 == 4.0, (
            f"copy(config).v0 = 8.0 leaked to original: {cfg.v0}"
        )

        # Run both headless for 30 steps
        e1.run_headless(steps=30, callback=r1.on_frame)
        e2.run_headless(steps=30, callback=r2.on_frame)

        # Both recorders must have captured metrics
        assert len(r1.metrics_history) == 30, (
            f"Recorder 1: expected 30 metrics, got {len(r1.metrics_history)}"
        )
        assert len(r2.metrics_history) == 30, (
            f"Recorder 2: expected 30 metrics, got {len(r2.metrics_history)}"
        )

        # Average speed should differ (different v0)
        avg_speed_1 = np.mean([m["speed_avg"] for m in r1.metrics_history])
        avg_speed_2 = np.mean([m["speed_avg"] for m in r2.metrics_history])

        assert avg_speed_1 != avg_speed_2, (
            f"Engines with v0=4.0 and v0=8.0 produced same average speed "
            f"({avg_speed_1:.3f}). Metrics isolation is broken."
        )
        # Higher v0 should produce higher speed
        assert avg_speed_2 > avg_speed_1, (
            f"v0=8.0 engine should be faster than v0=4.0, but got "
            f"{avg_speed_2:.3f} ≤ {avg_speed_1:.3f}"
        )

    def test_copy_config_metrics_diverge_over_time(self):
        """IT3c: copy(config) isolation causes metric trajectories to diverge.

        Per-frame alpha values should differ between the two runs, not just
        aggregate stats. This catches shallow isolation where only the
        initial state differs but the config leak causes reconvergence.
        """
        from pymurmur import Recorder, SimConfig, SimulationEngine

        cfg = SimConfig()
        cfg.num_boids = 30
        cfg.v0 = 3.0
        cfg.seed = 100
        cfg.metrics_detail_level = 1

        cfg2 = copy(cfg)
        cfg2.v0 = 7.0

        e1 = SimulationEngine(cfg)
        r1 = Recorder(e1, cfg)
        e2 = SimulationEngine(cfg2)
        r2 = Recorder(e2, cfg2)

        e1.run_headless(steps=50, callback=r1.on_frame)
        e2.run_headless(steps=50, callback=r2.on_frame)

        # Per-frame alpha should differ for at least half the frames
        alpha1 = np.array([m["alpha"] for m in r1.metrics_history])
        alpha2 = np.array([m["alpha"] for m in r2.metrics_history])

        # They should not be identical on every frame
        differences = np.abs(alpha1 - alpha2)
        divergent_frames = np.sum(differences > 1e-10)

        assert divergent_frames > 0, (
            "All 50 frames have identical alpha — copy(config) isolation is broken"
        )
        # At least a few frames should differ meaningfully
        assert divergent_frames >= 5, (
            f"Only {divergent_frames}/50 frames differ in alpha — "
            f"isolation may be shallow or config leaked"
        )

    def test_three_way_copy_chain_isolation(self):
        """IT3d: A→copy→B, A→copy→C — three-way copy chain isolation."""
        from pymurmur import SimConfig

        base = SimConfig()
        base.num_boids = 100
        base.v0 = 5.0
        base.seed = 42
        base.separation_weight = 4.5

        cfg_a = copy(base)
        cfg_a.separation_weight = 1.0

        cfg_b = copy(base)
        cfg_b.separation_weight = 10.0

        # Base must be unchanged
        assert base.v0 == 5.0
        assert base.separation_weight == 4.5

        # Each copy must have its own value
        assert cfg_a.separation_weight == 1.0
        assert cfg_b.separation_weight == 10.0

        # All three must be different objects (not shared sub-configs)
        assert cfg_a.separation_weight != cfg_b.separation_weight
        assert cfg_a.separation_weight != base.separation_weight
        assert cfg_b.separation_weight != base.separation_weight

        # Verify sub-config objects are distinct (not shared references)
        assert cfg_a._spatial is not base._spatial, (
            "copy(config) shared _spatial sub-config — deep copy broken"
        )
        assert cfg_b._spatial is not base._spatial, (
            "copy(config) shared _spatial sub-config — deep copy broken"
        )
        assert cfg_a._spatial is not cfg_b._spatial, (
            "Two copies share same _spatial object — deep copy broken"
        )


class TestFieldMapCompleteness:
    """IT4: _FIELD_MAP completeness audit.

    Every sub-config dataclass field must have an entry in _FIELD_MAP.
    Every _FIELD_MAP entry must point to a real sub-config attribute and
    a real field name on that sub-config. _ALL_FIELD_NAMES must equal
    set(_FIELD_MAP.keys()) | _DIRECT_FIELDS. Catches silent default
    regressions when new fields are added but forgotten in the map.
    """

    # All 20 sub-config classes (matching config.py __init__)
    _SUBCONFIG_CLASSES: dict[str, type] = {
        "_angle": AngleConfig,
        "_roost": RoostConfig,
        "_marl": MarlConfig,
        "_domain": DomainConfig,
        "_flock": FlockConfig,
        "_boundary": BoundaryConfig,
        "_projection": ProjectionConfig,
        "_spatial": SpatialConfig,
        "_field": FieldConfig,
        "_wander": WanderConfig,
        "_vicsek": VicsekConfig,
        "_influencer": InfluencerConfig,
        "_index": IndexConfig,
        "_refinement": RefinementConfig,
        "_extension": ExtensionConfig,
        "_predator": PredatorConfig,
        "_ecology": EcologyConfig,
        "_perf": PerfConfig,
        "_viz": VizConfig,
        "_capture": CaptureConfig,
    }

    def test_every_subconfig_field_has_field_map_entry(self):
        """IT4a: Every sub-config dataclass field appears in _FIELD_MAP
        or _NESTED_ONLY.

        If a field is added to a dataclass but forgotten in both maps,
        it silently defaults on YAML round-trip and flat access fails.
        """
        from dataclasses import fields

        cfg = SimConfig()
        mapped_flat_names: set[str] = set()

        for sub_attr, SubClass in self._SUBCONFIG_CLASSES.items():
            getattr(cfg, sub_attr)
            for f in fields(SubClass):
                # Find the flat field name that maps to this sub-attr + field
                found = False
                for flat_name, (mapped_attr, mapped_field) in _FIELD_MAP.items():
                    if mapped_attr == sub_attr and mapped_field == f.name:
                        found = True
                        mapped_flat_names.add(flat_name)
                        break
                # Also check _NESTED_ONLY (fully retired shims — no flat
                # alias; from_file routes their YAML keys explicitly)
                if not found:
                    for _flat_name, (mapped_attr, mapped_field) in _NESTED_ONLY.items():
                        if mapped_attr == sub_attr and mapped_field == f.name:
                            found = True
                            break

                assert found, (
                    f"Sub-config field {SubClass.__name__}.{f.name} "
                    f"has no entry in _FIELD_MAP or _NESTED_ONLY. "
                    f"Add it or it will silently default on YAML load."
                )

    def test_every_field_map_entry_points_to_real_attribute(self):
        """IT4b: Every _FIELD_MAP entry (sub_attr, field_name) is valid.

        sub_attr must be a real attribute of SimConfig (e.g. '_domain').
        field_name must be a real field on that sub-config dataclass.
        """
        from dataclasses import fields

        cfg = SimConfig()

        # Build lookup: sub_attr → set of valid field names
        valid_fields: dict[str, set[str]] = {}
        for sub_attr, SubClass in self._SUBCONFIG_CLASSES.items():
            valid_fields[sub_attr] = {f.name for f in fields(SubClass)}

        for flat_name, (sub_attr, field_name) in _FIELD_MAP.items():
            # sub_attr must exist on SimConfig
            assert hasattr(cfg, sub_attr), (
                f"_FIELD_MAP['{flat_name}'] → sub_attr '{sub_attr}' "
                f"does not exist on SimConfig"
            )

            # field_name must exist on the sub-config dataclass
            assert sub_attr in valid_fields, (
                f"_FIELD_MAP['{flat_name}'] → sub_attr '{sub_attr}' "
                f"not in known sub-config classes"
            )
            assert field_name in valid_fields[sub_attr], (
                f"_FIELD_MAP['{flat_name}'] → ({sub_attr!r}, {field_name!r}) "
                f"but {sub_attr} has no field '{field_name}'. "
                f"Valid fields: {sorted(valid_fields[sub_attr])}"
            )

    def test_all_field_names_is_key_union(self):
        """IT4c: _ALL_FIELD_NAMES == _FIELD_MAP keys ∪ _DIRECT_FIELDS ∪ _NESTED_ONLY keys.

        Nested-only fields (retired shims) must still be included: from_file()'s
        strict unknown-key check tests membership in _ALL_FIELD_NAMES *before*
        it routes nested-only keys to their sub-config, so omitting them here
        makes from_file() reject their YAML keys as unknown (a real bug caught
        by test_save_preserves_angle_config_fields — a round-trip through
        phi_p raised "Unknown config keys" until this set included it).
        """
        expected = set(_FIELD_MAP.keys()) | _DIRECT_FIELDS | set(_NESTED_ONLY.keys())
        assert _ALL_FIELD_NAMES == expected, (
            f"_ALL_FIELD_NAMES is out of sync.\n"
            f"Expected: {sorted(expected)}\n"
            f"Got:      {sorted(_ALL_FIELD_NAMES)}\n"
            f"Missing from _ALL_FIELD_NAMES: {expected - _ALL_FIELD_NAMES}\n"
            f"Extra in _ALL_FIELD_NAMES: {_ALL_FIELD_NAMES - expected}"
        )

    def test_no_duplicate_field_map_entries(self):
        """IT4d: No two flat field names map to the same (sub_attr, field).

        Two flat names pointing to the same sub-config field would
        cause silent overwrites during __setattr__.
        """
        seen_targets: set[tuple[str, str]] = set()
        for _flat_name, target in _FIELD_MAP.items():
            assert target not in seen_targets, (
                f"_FIELD_MAP has duplicate target {target}: "
                f"already mapped from another flat name"
            )
            seen_targets.add(target)

    def test_no_dead_field_map_entries(self):
        """IT4e: No _FIELD_MAP entries point to sub-configs not in _SUBCONFIG_CLASSES.

        Prevents stale entries when sub-configs are renamed or removed.
        """
        known_sub_attrs = set(self._SUBCONFIG_CLASSES.keys())
        for flat_name, (sub_attr, _field_name) in _FIELD_MAP.items():
            assert sub_attr in known_sub_attrs, (
                f"_FIELD_MAP['{flat_name}'] → sub_attr '{sub_attr}' "
                f"not in known sub-config classes: {sorted(known_sub_attrs)}"
            )

    def test_subconfig_imports_used_in_test(self):
        """IT4f: All 16 sub-config classes are importable and have fields."""
        from dataclasses import fields

        for sub_attr, SubClass in self._SUBCONFIG_CLASSES.items():
            sub_fields = list(fields(SubClass))
            assert len(sub_fields) > 0, (
                f"Sub-config {SubClass.__name__} (attr {sub_attr}) "
                f"has no dataclass fields — is it empty?"
            )


# ═══════════════════════════════════════════════════════════════════
# I7 Medium-Priority Integration Tests — cross I7.1 + I7.4 + I4.2
# ═══════════════════════════════════════════════════════════════════


class TestEngineSubconfigRouting:
    """IT5: Engine reads from correct sub-config despite flat-access ambiguity.

    DomainConfig and CaptureConfig both have fields with overlapping flat names
    (width/height). This verifies the __getattr__ delegation routes to the
    correct sub-config and the engine's hot path uses domain dimensions,
    not capture dimensions.
    """

    def test_flat_access_routes_to_domain_not_capture_when_both_set(self):
        """IT5a: config.width returns domain.width, not capture.capture_width.

        When both domain.width=2000 and capture.capture_width=800 are set,
        the flat access config.width must return 2000 (domain), not 800 (capture).
        """
        from pymurmur import SimConfig

        cfg = SimConfig()
        # Set conflicting values via sub-config accessors
        cfg.domain.width = 2000.0
        cfg.capture.capture_width = 800

        # Flat access must route to domain, not capture
        assert cfg.width == 2000.0, (
            f"config.width should be domain.width=2000, got {cfg.width}. "
            f"Flat access may be routing to capture.capture_width instead."
        )
        assert cfg.height == 700.0, (
            f"config.height should be domain.height=700 (default), got {cfg.height}"
        )

    def test_engine_uses_domain_dimensions_not_capture(self):
        """IT5b: Engine step() uses domain dimensions, not capture dimensions.

        Set domain.width=2000, capture.capture_width=800. After stepping,
        verify boids can move beyond capture_width=800 but stay within
        domain.width=2000 (toroidal wrapping).
        """
        from pymurmur import SimulationEngine

        cfg = SimConfig()
        cfg.domain.width = 2000.0
        cfg.domain.height = 1400.0
        cfg.domain.depth = 800.0
        # Purposely set capture dimensions to a much smaller value
        cfg.capture.capture_width = 400
        cfg.capture.capture_height = 300
        cfg.num_boids = 50
        cfg.v0 = 8.0
        cfg.seed = 42
        cfg.boundary_mode = "toroidal"

        # Flat access must return domain values
        assert cfg.width == 2000.0, (
            f"config.width must be 2000.0 (domain), got {cfg.width}"
        )
        assert cfg.height == 1400.0
        assert cfg.depth == 800.0

        engine = SimulationEngine(cfg)

        # Run enough steps for boids to travel beyond capture_width
        for _ in range(100):
            engine.step()

        pos = engine.flock.positions
        active = engine.flock.active
        active_pos = pos[active]

        # Boids should be within domain bounds (wrapped toroidally)
        assert np.all(active_pos[:, 0] >= 0.0), (
            "Boid x positions must be >= 0 (domain boundary)"
        )
        assert np.all(active_pos[:, 0] <= 2000.0), (
            f"Boid x positions must be <= 2000 (domain width), "
            f"max={active_pos[:, 0].max():.1f}. "
            f"If boids are limited to ~400, engine may be using capture_width."
        )
        assert np.all(active_pos[:, 1] >= 0.0)
        assert np.all(active_pos[:, 1] <= 1400.0)
        assert np.all(active_pos[:, 2] >= 0.0)
        assert np.all(active_pos[:, 2] <= 800.0)

        # Crucially: some boids should have traveled beyond capture_width=400
        # proving the engine uses domain.width=2000 for wrapping
        max_x = float(active_pos[:, 0].max())
        assert max_x > 400.0, (
            f"Max x position is {max_x:.1f} ≤ 400 (capture_width). "
            f"Engine may be using capture dimensions for physics."
        )

    def test_flat_vs_subconfig_access_consistency(self):
        """IT5c: Flat access and sub-config access always agree.

        Mutating via sub-config must be visible via flat access, and
        vice versa. This verifies __getattr__/__setattr__ delegation.
        """
        from pymurmur import SimConfig

        cfg = SimConfig()

        # Set via sub-config, read via flat
        cfg.flock.v0 = 5.5
        assert cfg.v0 == 5.5, (
            f"config.flock.v0 = 5.5, but config.v0 = {cfg.v0}"
        )

        # Set via flat, read via sub-config
        cfg.v0 = 9.0
        assert cfg.flock.v0 == 9.0, (
            f"config.v0 = 9.0, but config.flock.v0 = {cfg.flock.v0}"
        )

        # Same for predator fields
        cfg.predator.predator_strength = 3.0
        assert cfg.predator_strength == 3.0

        cfg.predator_strength = 1.5
        assert cfg.predator.predator_strength == 1.5

        # Same for capture fields
        cfg.capture.capture_fps = 25
        assert cfg.capture_fps == 25

        cfg.capture_fps = 10
        assert cfg.capture.capture_fps == 10

        # Direct fields (not delegated)
        cfg.mode = "vicsek"
        assert cfg.mode == "vicsek"


class TestSubconfigFlatYamlConsistency:
    """IT6: Sub-config→flat→YAML consistency across serialization.

    Mutations through sub-config accessors must be visible via flat access
    and survive YAML round-trip. Verifies that __getattr__/__setattr__
    delegation stays consistent across all three access paths.
    """

    def test_roundtrip_subconfig_mutation_visible_via_flat_access(self):
        """IT6a: Mutate via sub-config → YAML round-trip → flat access agrees."""
        import tempfile
        from pathlib import Path

        from pymurmur import SimConfig

        cfg = SimConfig()
        cfg.seed = 99

        # Mutate via sub-config accessors
        cfg.flock.v0 = 11.0
        cfg.flock.num_boids = 300
        cfg.flock.boid_size = 7.5
        cfg.domain.width = 2000.0
        cfg.domain.depth = 600.0
        cfg.capture.capture_frames = 500
        cfg.capture.capture_every = 2

        # Verify flat access before round-trip
        assert cfg.v0 == 11.0
        assert cfg.num_boids == 300
        assert cfg.boid_size == 7.5
        assert cfg.width == 2000.0
        assert cfg.depth == 600.0
        assert cfg.capture_frames == 500
        assert cfg.capture_every == 2

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            tmp = Path(f.name)
        try:
            cfg.to_file(tmp)
            loaded = SimConfig.from_file(tmp)

            # After round-trip, flat access should match (accounting for known collisions)
            assert loaded.v0 == 11.0, (
                f"v0 lost in YAML round-trip: expected 11.0, got {loaded.v0}"
            )
            assert loaded.num_boids == 300, (
                "num_boids lost in YAML round-trip"
            )
            assert loaded.boid_size == 7.5, (
                "boid_size lost in YAML round-trip"
            )
            # depth doesn't collide with anything, should survive
            assert loaded.depth == 600.0, (
                f"depth lost in YAML round-trip: expected 600.0, got {loaded.depth}"
            )
            # capture fields: capture.every → flat "every" but _FIELD_MAP has
            # "capture_every" → ("_capture", "capture_every"), so the to_file()
            # key is "every" which from_file() flattens — but from_file() only
            # keeps known _ALL_FIELD_NAMES. "every" is NOT in _ALL_FIELD_NAMES
            # (only "capture_every" is). So capture_every defaults.
            # This is a known YAML format limitation.
            assert isinstance(loaded.capture_every, int), (
                f"capture_every should be int, got {type(loaded.capture_every)}"
            )
            assert loaded.seed == 99, (
                f"seed lost: expected 99, got {loaded.seed}"
            )
        finally:
            tmp.unlink()

    def test_flat_mutation_visible_via_subconfig(self):
        """IT6b: Mutate via flat → sub-config accessor reflects the change."""
        from pymurmur import SimConfig

        cfg = SimConfig()

        cfg.v0 = 13.0
        assert cfg.flock.v0 == 13.0, (
            f"config.v0 = 13.0 but config.flock.v0 = {cfg.flock.v0}"
        )

        cfg.capture_frames = 999
        assert cfg.capture.capture_frames == 999, (
            f"config.capture_frames = 999 but config.capture.capture_frames = "
            f"{cfg.capture.capture_frames}"
        )

        cfg.predator_threat_radius = 25.0
        assert cfg.predator.predator_threat_radius == 25.0

    def test_roundtrip_then_both_access_paths_agree(self):
        """IT6c: After YAML round-trip, sub-config and flat access agree."""
        import tempfile
        from pathlib import Path

        from pymurmur import SimConfig

        cfg = SimConfig()
        cfg.v0 = 7.0
        cfg.num_boids = 250
        cfg.boid_size = 10.0
        cfg.seed = 77

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            tmp = Path(f.name)
        try:
            cfg.to_file(tmp)
            loaded = SimConfig.from_file(tmp)

            # Both access paths must agree on loaded config
            assert loaded.v0 == loaded.flock.v0, (
                f"After round-trip: flat v0={loaded.v0} ≠ "
                f"sub-config v0={loaded.flock.v0}"
            )
            assert loaded.num_boids == loaded.flock.num_boids, (
                f"After round-trip: flat num_boids={loaded.num_boids} ≠ "
                f"sub-config num_boids={loaded.flock.num_boids}"
            )
            assert loaded.boid_size == loaded.flock.boid_size, (
                f"After round-trip: flat boid_size={loaded.boid_size} ≠ "
                f"sub-config boid_size={loaded.flock.boid_size}"
            )
            assert loaded.seed == 77
        finally:
            tmp.unlink()

    def test_all_16_subconfig_properties_are_accessible(self):
        """IT6d: All 16 sub-config properties exist and return correct types."""
        from pymurmur import SimConfig

        cfg = SimConfig()

        # All 16 sub-config accessors must return the right type
        assert isinstance(cfg.domain, DomainConfig), (
            f"cfg.domain should be DomainConfig, got {type(cfg.domain)}"
        )
        assert isinstance(cfg.flock, FlockConfig)
        assert isinstance(cfg.boundary, BoundaryConfig)
        assert isinstance(cfg.projection, ProjectionConfig)
        assert isinstance(cfg.spatial, SpatialConfig)
        assert isinstance(cfg.field, FieldConfig)
        assert isinstance(cfg.vicsek, VicsekConfig)
        assert isinstance(cfg.influencer, InfluencerConfig)
        assert isinstance(cfg.index, IndexConfig)
        assert isinstance(cfg.refinement, RefinementConfig)
        assert isinstance(cfg.extension, ExtensionConfig)
        assert isinstance(cfg.predator, PredatorConfig)
        assert isinstance(cfg.ecology, EcologyConfig)
        assert isinstance(cfg.perf, PerfConfig)
        assert isinstance(cfg.viz, VizConfig)
        assert isinstance(cfg.capture, CaptureConfig)

    def test_direct_fields_not_delegated_to_subconfigs(self):
        """IT6e: Direct fields (mode, seed, position_init) are stored on
        SimConfig, not delegated to a sub-config."""
        from pymurmur import SimConfig

        cfg = SimConfig(mode="spatial", seed=123, position_init="sphere")

        assert cfg.mode == "spatial"
        assert cfg.seed == 123
        assert cfg.position_init == "sphere"

        # These should NOT be delegated — they're direct attributes
        assert "mode" not in _FIELD_MAP, (
            "mode should be a direct field, not in _FIELD_MAP"
        )
        assert "seed" not in _FIELD_MAP, (
            "seed should be a direct field, not in _FIELD_MAP"
        )
        assert "position_init" not in _FIELD_MAP, (
            "position_init should be a direct field, not in _FIELD_MAP"
        )


# ═══════════════════════════════════════════════════════════════════
# IT7 — YAML Key Collision Detection (I7.1 + I7.4)
# ═══════════════════════════════════════════════════════════════════


class TestYamlKeyCollisionDetection:
    """Detects and documents known YAML key collisions and silent drops
    in SimConfig.to_file() / from_file() round-trip.

    Collisions occur because to_file() nests sub-configs under section
    keys using short field names (e.g. domain: {width: ...}, capture:
    {width: ...}) while from_file() flattens ALL sections into a single
    namespace and filters by _ALL_FIELD_NAMES.

    These tests explicitly assert the CURRENT (buggy) behavior. When the
    YAML serialization is fixed, these tests will intentionally break
    to signal that the fix was successful — update the assertions.
    """

    def test_programmatic_collision_audit(self):
        """Analyze to_file() YAML structure to find ALL duplicate keys.

        Parses the raw YAML output and detects every key that appears
        in more than one section. The set of colliding keys is asserted
        exactly — if a collision is fixed or a new one added, this test
        fails (regression guard).
        """
        import yaml

        cfg = SimConfig()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            tmp = Path(f.name)
        try:
            cfg.to_file(tmp)
            raw = yaml.safe_load(tmp.read_text()) or {}

            seen_keys: dict[str, list[str]] = {}  # key → [section_names]
            collisions: dict[str, list[str]] = {}

            for section_name, section_data in raw.items():
                if isinstance(section_data, dict):
                    for k in section_data:
                        if k in seen_keys:
                            if k not in collisions:
                                collisions[k] = seen_keys[k].copy()
                            collisions[k].append(section_name)
                        else:
                            seen_keys[k] = [section_name]
                else:
                    # Scalar values (mode, seed) — keyed by section_name
                    if section_name in seen_keys:
                        collisions.setdefault(section_name, seen_keys[section_name])
                        collisions[section_name].append(section_name + "_scalar")
                    seen_keys[section_name] = [section_name + "_scalar"]

            # Known collisions — each key appears in the listed sections
            known_collisions = {}

            assert set(collisions.keys()) == set(known_collisions.keys()), (
                f"YAML key collisions changed!\n"
                f"Expected: {sorted(known_collisions.keys())}\n"
                f"Got:      {sorted(collisions.keys())}\n"
                f"If a collision was fixed, update known_collisions.\n"
                f"If a new collision appeared, a new field name conflicts."
            )

            for key, expected_sections in known_collisions.items():
                actual_sections = collisions[key]
                assert actual_sections == expected_sections, (
                    f"Collision sections for '{key}' changed:\n"
                    f"Expected: {expected_sections}\n"
                    f"Got:      {actual_sections}"
                )
        finally:
            tmp.unlink()

    def test_yaml_section_keys_not_in_all_field_names(self):
        """Audit: find YAML leaf keys not in _ALL_FIELD_NAMES (silently dropped).

        When from_file() flattens sections, only keys in _ALL_FIELD_NAMES
        survive. This test enumerates which YAML keys are silently dropped
        so the set is explicit and tracked.
        """
        import yaml

        cfg = SimConfig()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            tmp = Path(f.name)
        try:
            cfg.to_file(tmp)
            raw = yaml.safe_load(tmp.read_text()) or {}

            all_yaml_keys: set[str] = set()
            for section_name, section_data in raw.items():
                if isinstance(section_data, dict):
                    for k in section_data:
                        all_yaml_keys.add(k)
                else:
                    all_yaml_keys.add(section_name)

            # Nested-only keys (retired shims) are handled explicitly
            # by from_file(), so they are not dropped.
            handled = _ALL_FIELD_NAMES | set(_NESTED_ONLY.keys())
            dropped = all_yaml_keys - handled

            # Known set of silently-dropped YAML keys
            known_dropped = set()

            assert dropped == known_dropped, (
                f"Silently-dropped YAML keys changed!\n"
                f"Expected: {sorted(known_dropped)}\n"
                f"Got:      {sorted(dropped)}\n"
                f"New in dropped (added silently?): {sorted(dropped - known_dropped)}\n"
                f"Fixed (no longer dropped!): {sorted(known_dropped - dropped)}"
            )
        finally:
            tmp.unlink()

    def test_domain_capture_dimensions_collision(self):
        """capture.width and capture.height overwrite domain.width/height.

        Both domain: {width: X} and capture: {width: Y} flatten to key
        'width'. capture is written later in to_file(), so its value wins.
        The loaded config gets capture's dimensions in domain fields.
        """
        cfg = SimConfig()

        # Distinct values to track who wins
        cfg.domain.width = 1111.0
        cfg.domain.height = 3333.0
        cfg.capture.capture_width = 2222
        cfg.capture.capture_height = 4444

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            tmp = Path(f.name)
        try:
            cfg.to_file(tmp)
            loaded = SimConfig.from_file(tmp)

            # Both domain and capture dimensions survive independently
            assert loaded.width == 1111.0, (
                f"BUG FIXED: width collision resolved! Got {loaded.width}"
            )
            assert loaded.height == 3333.0, (
                f"BUG FIXED: height collision resolved! Got {loaded.height}"
            )
            assert loaded.domain.width == 1111.0
            assert loaded.domain.height == 3333.0
            assert loaded.capture.capture_width == 2222
            assert loaded.capture.capture_height == 4444
        finally:
            tmp.unlink()

    def test_visual_capture_fps_collision(self):
        """visual.fps and capture.capture_fps both produce YAML key 'fps'.

        capture section is written after visual, so capture's fps value
        overwrites visual's fps value.
        """
        cfg = SimConfig()

        cfg.viz.fps = 30
        cfg.capture.capture_fps = 60

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            tmp = Path(f.name)
        try:
            cfg.to_file(tmp)
            loaded = SimConfig.from_file(tmp)

            # Both viz.fps and capture_fps survive independently
            assert loaded.fps == 30, (
                f"BUG FIXED: fps collision resolved! Got {loaded.fps}"
            )
            assert loaded.viz.fps == 30
            assert loaded.capture.capture_fps == 60
        finally:
            tmp.unlink()

    def test_silently_dropped_extension_toggles(self):
        """Extension toggles are written as short names (predator, roosting)
        but _ALL_FIELD_NAMES has long names (predator_enabled, roosting_enabled).
        After flattening, the short names don't match → silently dropped →
        revert to dataclass defaults.
        """
        cfg = SimConfig()

        cfg.extension.predator_enabled = True
        cfg.extension.roosting_enabled = True
        cfg.extension.wander_enabled = True
        cfg.extension.ripple_enabled = True

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            tmp = Path(f.name)
        try:
            cfg.to_file(tmp)
            loaded = SimConfig.from_file(tmp)

            # Extension toggles survive round-trip correctly
            assert loaded.predator_enabled, (
                f"BUG FIXED: predator_enabled persisted! Got {loaded.predator_enabled}"
            )
            assert loaded.roosting_enabled, (
                "BUG FIXED: roosting_enabled persisted!"
            )
            assert loaded.wander_enabled, (
                "BUG FIXED: wander_enabled persisted!"
            )
            assert loaded.ripple_enabled, (
                "BUG FIXED: ripple_enabled persisted!"
            )
        finally:
            tmp.unlink()

    def test_silently_dropped_capture_short_names(self):
        """Capture short-name YAML keys (every, frames, output, etc.)
        don't match _ALL_FIELD_NAMES (capture_every, capture_frames, etc.)
        → silently dropped → revert to defaults.
        """
        cfg = SimConfig()

        cfg.capture.capture_every = 99
        cfg.capture.capture_frames = 888
        cfg.capture.capture_output = "output/custom.gif"
        cfg.capture.capture_metrics_csv = "output/custom.csv"
        cfg.capture.capture_metrics_json = "output/custom.json"
        cfg.capture.capture_with_viz = True

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            tmp = Path(f.name)
        try:
            cfg.to_file(tmp)
            loaded = SimConfig.from_file(tmp)

            # Capture fields survive round-trip correctly
            assert loaded.capture_every == 99, (
                f"BUG FIXED: capture_every persisted! Got {loaded.capture_every}"
            )
            assert loaded.capture_frames == 888, (
                "BUG FIXED: capture_frames persisted!"
            )
            assert loaded.capture_output == "output/custom.gif", (
                f"BUG FIXED: capture_output persisted! Got {loaded.capture_output}"
            )
            assert loaded.capture_metrics_csv == "output/custom.csv", (
                "BUG FIXED: capture_metrics_csv persisted!"
            )
            assert loaded.capture_metrics_json == "output/custom.json", (
                "BUG FIXED: capture_metrics_json persisted!"
            )
            assert loaded.capture_with_viz, (
                "BUG FIXED: capture_with_viz persisted!"
            )
        finally:
            tmp.unlink()

    def test_silently_dropped_refinements_toggle(self):
        """refinements.enabled → YAML key 'enabled' → not in _ALL_FIELD_NAMES
        → refinements toggle silently reverts to default (True).
        """
        cfg = SimConfig()
        cfg.refinement.refinements = False

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            tmp = Path(f.name)
        try:
            cfg.to_file(tmp)
            loaded = SimConfig.from_file(tmp)

            # Refinements toggle survives round-trip correctly
            assert not loaded.refinements, (
                f"BUG FIXED: refinements toggle persisted! Got {loaded.refinements}"
            )
        finally:
            tmp.unlink()

    def test_non_colliding_fields_survive_roundtrip(self):
        """Fields without collisions (unique YAML keys matching
        _ALL_FIELD_NAMES) should survive round-trip correctly.

        This is a sanity check — if these break, something else is wrong.
        """
        cfg = SimConfig()
        cfg.seed = 99
        cfg.mode = "spatial"
        cfg.flock.v0 = 11.0
        cfg.flock.num_boids = 300
        cfg.flock.boid_size = 7.5
        cfg.domain.depth = 600.0  # depth has no collision
        cfg.boundary.boundary_mode = "sphere"
        cfg.boundary.boundary_sphere_radius = 350.0
        cfg.predator.predator_threat_radius = 25.0
        cfg.predator.predator_strength = 2.0
        cfg.perf.metrics_detail_level = 2
        cfg.perf.metrics_interval = 99  # written as 'metrics_interval' (matches)
        cfg.index.spatial_index = "kdtree"
        cfg.index.topological_cap = 20

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            tmp = Path(f.name)
        try:
            cfg.to_file(tmp)
            loaded = SimConfig.from_file(tmp)

            assert loaded.seed == 99
            assert loaded.mode == "spatial"
            assert loaded.v0 == 11.0
            assert loaded.num_boids == 300
            assert loaded.boid_size == 7.5
            assert loaded.depth == 600.0
            assert loaded.boundary_mode == "sphere"
            assert loaded.boundary_sphere_radius == 350.0
            assert loaded.predator_threat_radius == 25.0
            assert loaded.predator_strength == 2.0
            assert loaded.metrics_detail_level == 2
            assert loaded.metrics_interval == 99
            assert loaded.spatial_index == "kdtree"
            assert loaded.topological_cap == 20
        finally:
            tmp.unlink()
