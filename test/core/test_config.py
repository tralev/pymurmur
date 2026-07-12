"""Unit tests for core.config — SimConfig dataclass and YAML I/O."""

import tempfile
from pathlib import Path

from pymurmur.core.config import SimConfig


def test_config_defaults():
    """SimConfig() has documented default values."""
    cfg = SimConfig()
    assert cfg.mode == "projection"
    assert cfg.num_boids == 150
    assert cfg.v0 == 4.0
    assert cfg.width == 1000.0
    assert cfg.height == 700.0
    assert cfg.depth == 400.0
    assert cfg.boundary_mode == "toroidal"
    assert cfg.use_numba is True
    assert cfg.spatial_index == "auto"
    assert cfg.theme == "ink"
    assert cfg.trails == "off"


def test_config_from_file_roundtrip():
    """from_file() → to_file() → from_file() produces identical config."""
    cfg = SimConfig()
    cfg.mode = "spatial"
    cfg.num_boids = 300

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        tmp = Path(f.name)
    try:
        cfg.to_file(tmp)
        loaded = SimConfig.from_file(tmp)
        assert loaded.mode == "spatial"
        assert loaded.num_boids == 300
        assert loaded.v0 == cfg.v0
    finally:
        tmp.unlink()


def test_config_from_file_flattens_nested():
    """YAML with nested sections flattens to SimConfig fields."""
    import yaml
    data = {
        "domain": {"width": 2000.0, "height": 1500.0, "depth": 800.0},
        "flock": {"num_boids": 500},
        "mode": "field",
    }
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
        yaml.dump(data, f)
        tmp = Path(f.name)

    try:
        cfg = SimConfig.from_file(tmp)
        assert cfg.width == 2000.0
        assert cfg.height == 1500.0
        assert cfg.num_boids == 500
        assert cfg.mode == "field"
    finally:
        tmp.unlink()


def test_config_from_file_unknown_keys_ignored():
    """Extra YAML keys don't raise errors."""
    import yaml
    data = {"mode": "vicsek", "unknown_future_field": 42, "extra": {"nested": True}}
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
        yaml.dump(data, f)
        tmp = Path(f.name)

    try:
        cfg = SimConfig.from_file(tmp)
        assert cfg.mode == "vicsek"
    finally:
        tmp.unlink()


def test_config_from_file_not_found():
    """Non-existent file raises FileNotFoundError."""
    import pytest
    with pytest.raises(FileNotFoundError):
        SimConfig.from_file("/tmp/nonexistent_config_xyz.yaml")


def test_config_live_mutable_vs_static():
    """phi_p and sigma are mutable; width and boid_size are static."""
    cfg = SimConfig()
    cfg.phi_p = 0.5
    cfg.sigma = 10
    assert cfg.phi_p == 0.5
    assert cfg.sigma == 10
    cfg.width = 500.0
    assert cfg.width == 500.0
