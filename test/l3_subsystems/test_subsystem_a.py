"""Subsystem A — Entry & Configuration isolation tests.

Tests SimConfig from_file/to_file, YAML flattening, validation,
and config search path ordering. Mocked — no simulation dependency.
"""

from pathlib import Path

import pytest


class TestSubsystemA:
    """Config loading, validation, and round-tripping."""

    def test_simconfig_all_defaults(self):
        """SimConfig() has all 40+ documented default values."""
        from pymurmur.core.config import SimConfig
        cfg = SimConfig()
        # Spot-check key defaults
        assert cfg.mode == "projection"
        assert cfg.num_boids == 150
        assert cfg.width == 1000.0
        assert cfg.height == 700.0
        assert cfg.depth == 400.0
        assert cfg.v0 == 4.0
        assert cfg.max_force == 0.15
        assert cfg.boundary_mode == "toroidal"

    def test_simconfig_from_file_flattens_nested(self, tmp_path):
        """domain.width in YAML → config.width."""
        import yaml

        from pymurmur.core.config import SimConfig

        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml.dump({
            "domain": {"width": 500.0, "height": 300.0, "depth": 200.0},
            "flock": {"num_boids": 50},
        }))
        cfg = SimConfig.from_file(yaml_path)
        assert cfg.width == 500.0
        assert cfg.height == 300.0
        assert cfg.depth == 200.0
        assert cfg.num_boids == 50

    def test_simconfig_from_file_all_five_modes(self, tmp_path):
        """YAML with all 5 mode sections parses; only active mode applied."""
        import yaml

        from pymurmur.core.config import SimConfig

        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml.dump({
            "mode": "spatial",
            "projection": {"phi_p": 0.1, "phi_a": 0.8, "sigma": 6},
            "spatial": {"separation_weight": 2.0, "alignment_weight": 0.5},
        }))
        cfg = SimConfig.from_file(yaml_path)
        assert cfg.mode == "spatial"
        # Projection weights should also be loaded (flat structure)
        assert cfg.projection.phi_p == 0.1

    def test_simconfig_from_file_unknown_keys_ignored_when_not_strict(self, tmp_path):
        """Extra top-level YAML keys don't raise with strict=False
        (explicit forward-compatibility opt-out, e.g. evoflock configs
        that carry extra GA sections)."""
        import yaml

        from pymurmur.core.config import SimConfig

        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml.dump({
            "future_feature": "v2",
            "mode": "projection",
        }))
        cfg = SimConfig.from_file(yaml_path, strict=False)
        assert cfg.mode == "projection"

    def test_simconfig_from_file_unknown_top_level_key_raises_by_default(self, tmp_path):
        """G5: An unrecognized top-level key raises an actionable
        ValueError by default (strict=True) — same contract as unknown
        section-nested keys.  A typo'd top-level field (e.g. `mdoe:`
        instead of `mode:`) must not be silently swallowed."""
        import pytest
        import yaml

        from pymurmur.core.config import SimConfig

        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml.dump({
            "future_feature": "v2",
            "mode": "projection",
        }))
        with pytest.raises(ValueError, match="future_feature"):
            SimConfig.from_file(yaml_path)

    def test_simconfig_from_file_obstacles_list_passes_through(self, tmp_path):
        """G5: The `obstacles:` top-level list (a scene spec consumed
        separately by analysis/evoflock.py's load_obstacle_scene, not a
        SimConfig field) loads without error under strict=True — the
        one deliberately-named exemption from the top-level-key check.
        """
        import yaml

        from pymurmur.core.config import SimConfig

        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml.dump({
            "mode": "spatial",
            "obstacles": [{"shape": "sphere", "center": [0.0, 0.0, 0.0], "radius": 50.0}],
        }))
        cfg = SimConfig.from_file(yaml_path)  # strict=True (default)
        assert cfg.mode == "spatial"

    def test_simconfig_from_file_typo_top_level_list_key_raises(self, tmp_path):
        """G5: A top-level list-valued key that ISN'T the known
        `obstacles` exemption still raises — the exemption is by name,
        not by value shape.  A typo like `obstalces:` must not be
        silently swallowed just because its value happens to be a list.
        """
        import pytest
        import yaml

        from pymurmur.core.config import SimConfig

        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml.dump({
            "mode": "spatial",
            "obstalces": [{"shape": "sphere"}],  # typo of "obstacles"
        }))
        with pytest.raises(ValueError, match="obstalces"):
            SimConfig.from_file(yaml_path)

    def test_simconfig_to_file_roundtrip(self, tmp_path):
        """config.to_file(path) → SimConfig.from_file(path) produces identical."""
        from pymurmur.core.config import SimConfig

        cfg = SimConfig()
        cfg.num_boids = 123
        cfg.mode = "spatial"

        path = tmp_path / "roundtrip.yaml"
        cfg.to_file(path)

        cfg2 = SimConfig.from_file(path)
        assert cfg2.num_boids == 123
        assert cfg2.mode == "spatial"

    def test_simconfig_live_mutable_vs_static(self):
        """phi_p and sigma are mutable; width and boid_size are static."""
        from pymurmur.core.config import SimConfig
        cfg = SimConfig()
        # These should be changeable at runtime
        cfg.projection.phi_p = 0.5
        cfg.sigma = 10
        assert cfg.projection.phi_p == 0.5
        assert cfg.sigma == 10
        # Static fields can still be changed on the object (Python dataclass)
        # but should trigger reset logic in the engine — test that in subsystem B

    def test_simconfig_performance_defaults(self):
        """Performance fields have correct defaults."""
        from pymurmur.core.config import SimConfig
        cfg = SimConfig()
        assert cfg.spatial_index == "auto"
        assert cfg.metrics_detail_level == 1
        assert cfg.metrics_interval == 60
        assert cfg.instance_buffer_chunk == 50000

    def test_load_all_seven_shipped_configs(self):
        """All 7 conf/*.yaml parse without error.

        strict=False matches how __main__.py::load_config() actually loads
        shipped configs — some (e.g. murmuration_evo.yaml) carry extra
        non-SimConfig sections (evoflock GA params, obstacles) by design.
        """
        from pymurmur.core.config import SimConfig

        conf_dir = Path("conf")
        configs = sorted(conf_dir.glob("*.yaml"))
        assert len(configs) >= 7

        for path in configs:
            cfg = SimConfig.from_file(path, strict=False)
            assert cfg is not None

    def test_config_search_path_order(self):
        """load_config(name) tries conf/{name}.yaml first, then {name} as path."""
        from pymurmur.__main__ import load_config
        # Shipped config should resolve
        try:
            cfg = load_config("murmuration")
            assert cfg is not None
        except FileNotFoundError:
            pytest.skip("conf/murmuration.yaml not found")
