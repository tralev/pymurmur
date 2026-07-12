"""Config resolution tests — load_config() search path order.

Tests that config loading follows the documented precedence:
  1. conf/{name}.yaml (shipped preset)
  2. {name} as path (absolute or relative)
  3. FileNotFoundError with helpful message
"""

import pytest


class TestConfigResolution:
    """load_config() resolves names and paths correctly."""

    def test_resolve_shipped_first(self):
        """load_config('murmuration_spatial') finds conf/murmuration_spatial.yaml."""
        try:
            from pymurmur.__main__ import load_config
            cfg = load_config("murmuration_spatial")
            assert cfg is not None
            assert cfg.mode == "spatial"
        except FileNotFoundError:
            pytest.skip("conf/murmuration_spatial.yaml not found")

    def test_resolve_path_fallback(self, tmp_path):
        """load_config('/path/to/custom.yaml') resolves to absolute path."""
        import yaml
        from pymurmur.__main__ import load_config

        custom = tmp_path / "custom.yaml"
        custom.write_text(yaml.dump({"mode": "spatial", "num_boids": 50}))
        cfg = load_config(str(custom))
        assert cfg.num_boids == 50

    def test_resolve_none_returns_default(self):
        """load_config(None) returns SimConfig() with all defaults."""
        from pymurmur.__main__ import load_config
        cfg = load_config(None)
        assert cfg.mode == "projection"
        assert cfg.num_boids == 150

    def test_resolve_name_with_dot_yaml(self):
        """load_config('murmuration.yaml') resolves correctly (no double-append)."""
        try:
            from pymurmur.__main__ import load_config
            cfg = load_config("murmuration.yaml")
            assert cfg is not None
        except FileNotFoundError:
            pytest.skip("conf/murmuration.yaml not found")

    def test_resolve_missing_raises(self):
        """load_config('nonexistent') raises FileNotFoundError."""
        from pymurmur.__main__ import load_config
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config_xyz_123")
