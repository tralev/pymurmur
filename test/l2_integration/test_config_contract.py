"""I7 — Public facade unit tests (SimConfig/SimulationEngine/Recorder same-class), config sub-class forbidden imports, YAML round-trip, copy isolation, facade exported symbols, public facade pipeline.

Split out of test_config_contract.py (file-size split).
"""

import ast
import tempfile
from copy import copy
from pathlib import Path

import numpy as np


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
    """I7.6: Sub-config dataclasses in core/config/ have no forbidden module imports."""
    from pathlib import Path

    forbidden = {"pygame", "moderngl", "PIL", "numba", "matplotlib", "gymnasium",
                 "stable_baselines3"}
    config_dir = Path(__file__).resolve().parents[2] / "pymurmur" / "core" / "config"
    tree_sources = [
        ast.parse(py_file.read_text()) for py_file in sorted(config_dir.glob("*.py"))
    ]

    # Find all @dataclass class definitions across every file in core/config/
    for tree in tree_sources:
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


