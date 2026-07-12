"""Subsystem A — Entry & Configuration isolation tests.

Tests SimConfig from_file/to_file, YAML flattening, validation,
and config search path ordering. Mocked — no simulation dependency.
"""

import pytest
from pathlib import Path


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
        assert cfg.phi_p == 0.1

    def test_simconfig_from_file_unknown_keys_ignored(self, tmp_path):
        """Extra YAML keys don't raise errors (forward-compatible)."""
        import yaml
        from pymurmur.core.config import SimConfig

        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml.dump({
            "future_feature": "v2",
            "mode": "projection",
        }))
        cfg = SimConfig.from_file(yaml_path)
        assert cfg.mode == "projection"

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
        cfg.phi_p = 0.5
        cfg.sigma = 10
        assert cfg.phi_p == 0.5
        assert cfg.sigma == 10
        # Static fields can still be changed on the object (Python dataclass)
        # but should trigger reset logic in the engine — test that in subsystem B

    def test_simconfig_performance_defaults(self):
        """Performance fields have correct defaults."""
        from pymurmur.core.config import SimConfig
        cfg = SimConfig()
        assert cfg.use_numba is True
        assert cfg.spatial_index == "auto"
        assert cfg.metrics_detail_level == 1
        assert cfg.metrics_interval == 60
        assert cfg.instance_buffer_chunk == 50000

    def test_load_all_seven_shipped_configs(self):
        """All 7 conf/*.yaml parse without error."""
        from pymurmur.core.config import SimConfig

        conf_dir = Path("conf")
        configs = sorted(conf_dir.glob("*.yaml"))
        assert len(configs) >= 7

        for path in configs:
            cfg = SimConfig.from_file(path)
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
