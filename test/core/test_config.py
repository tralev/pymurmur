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
    assert cfg.spatial_index == "auto"
    assert cfg.theme == "ink"


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


# ── I7.1: Composed config tests ─────────────────────────────────

def test_copy_config_deep_copies_sub_configs():
    """copy.copy(config) produces independent sub-configs."""
    from copy import copy
    cfg = SimConfig()
    cfg2 = copy(cfg)

    # All 16 sub-config objects must be different (use public accessors)
    accessors = ["domain", "flock", "boundary", "projection", "spatial",
                 "field", "vicsek", "influencer", "index", "refinement",
                 "extension", "predator", "ecology", "perf", "viz", "capture"]
    for name in accessors:
        orig = getattr(cfg, name)
        copied = getattr(cfg2, name)
        assert copied is not orig, f"copy must deep-copy sub-config cfg.{name}"

    # Values must be equal
    assert cfg2.v0 == cfg.v0
    assert cfg2.domain.width == cfg.domain.width

    # Mutating copy must NOT affect original
    cfg2.v0 = 8.0
    assert cfg.v0 == 4.0, f"copy(config).v0 = 8.0 mutated original config.v0 to {cfg.v0}"
    cfg2.width = 500.0
    assert cfg.width == 1000.0, f"copy(config).width = 500 mutated original to {cfg.width}"


def test_sub_config_accessor_properties_all_present():
    """All 16 sub-config accessor properties exist on SimConfig."""
    cfg = SimConfig()
    expected = [
        "domain", "flock", "boundary", "projection", "spatial",
        "field", "vicsek", "influencer", "index", "refinement",
        "extension", "predator", "ecology", "perf", "viz", "capture",
    ]
    for name in expected:
        assert hasattr(cfg, name), f"SimConfig missing sub-config accessor: config.{name}"
        sub = getattr(cfg, name)
        assert sub is not None, f"config.{name} is None"


def test_field_map_routes_all_fields_correctly():
    """Every _FIELD_MAP entry correctly routes flat access to sub-config."""
    from pymurmur.core.config import _FIELD_MAP, _DIRECT_FIELDS
    cfg = SimConfig()

    for flat_name, (sub_attr, field_name) in _FIELD_MAP.items():
        sub_cfg = getattr(cfg, sub_attr)
        expected = getattr(sub_cfg, field_name)
        actual = getattr(cfg, flat_name)
        assert actual == expected, (
            f"_FIELD_MAP routing broken: config.{flat_name} = {actual!r}, "
            f"but config.{sub_attr}.{field_name} = {expected!r}"
        )

    # Direct fields should also be accessible
    for name in _DIRECT_FIELDS:
        assert hasattr(cfg, name), f"Direct field '{name}' not accessible"


def test_setattr_routes_to_sub_config_not_simconfig():
    """config.phi_p = 0.05 routes to _projection.phi_p, not SimConfig.__dict__."""
    cfg = SimConfig()
    cfg.phi_p = 0.05
    cfg.width = 2000.0
    cfg.theme = "paper"
    cfg.predator_enabled = True

    # Values must be in sub-configs, not on SimConfig itself
    assert cfg._projection.phi_p == 0.05
    assert cfg._domain.width == 2000.0
    assert cfg._viz.theme == "paper"
    assert cfg._extension.predator_enabled == True

    # Verify they're NOT directly on SimConfig (no accidental __dict__ storage)
    assert "phi_p" not in cfg.__dict__, "phi_p should not be stored on SimConfig"
    assert "theme" not in cfg.__dict__, "theme should not be stored on SimConfig"


def test_config_equality_compares_all_fields():
    """__eq__ compares all _ALL_FIELD_NAMES for equality."""
    cfg1 = SimConfig()
    cfg2 = SimConfig()
    assert cfg1 == cfg2, "Identical default configs must be equal"

    cfg2.v0 = 8.0
    assert cfg1 != cfg2, "Different v0 must make configs unequal"

    cfg3 = SimConfig(width=2000.0)
    assert cfg3 != cfg1, "Different width must make configs unequal"

    cfg4 = SimConfig(mode="spatial")
    assert cfg4 != cfg1, "Different mode must make configs unequal"


# ── P2.1: Sub-config independently instantiable ───────────────────

def test_sub_config_domain_standalone():
    """P2.1: DomainConfig can be instantiated and used independently."""
    from pymurmur.core.config import DomainConfig
    d = DomainConfig()
    assert d.width == 1000.0
    assert d.height == 700.0
    assert d.depth == 400.0
    # Custom construction
    d2 = DomainConfig(width=500.0, height=300.0, depth=200.0)
    assert d2.width == 500.0
    assert d2.height == 300.0
    assert d2.depth == 200.0


def test_sub_config_flock_standalone():
    """P2.1: FlockConfig can be instantiated and used independently."""
    from pymurmur.core.config import FlockConfig
    f = FlockConfig()
    assert f.num_boids == 150
    assert f.v0 == 4.0
    f2 = FlockConfig(num_boids=50, v0=6.0)
    assert f2.num_boids == 50
    assert f2.v0 == 6.0


def test_sub_config_boundary_standalone():
    """P2.1: BoundaryConfig can be instantiated and used independently."""
    from pymurmur.core.config import BoundaryConfig
    b = BoundaryConfig()
    assert b.boundary_mode == "toroidal"


def test_sub_config_extension_standalone():
    """P2.1: ExtensionConfig can be instantiated and used independently."""
    from pymurmur.core.config import ExtensionConfig
    e = ExtensionConfig()
    assert e.predator_enabled is False
    assert e.roosting_enabled is False
    assert e.wander_enabled is False
    assert e.ripple_enabled is False


def test_sub_config_viz_standalone():
    """P2.1: VizConfig can be instantiated and used independently."""
    from pymurmur.core.config import VizConfig
    v = VizConfig()
    assert v.theme == "ink"
    assert v.window_width == 1200
    assert v.window_height == 800


def test_all_16_sub_configs_are_dataclasses():
    """P2.1: All 16 sub-config types are @dataclass and independently
    instantiable via their own constructor (no parent SimConfig needed)."""
    from dataclasses import is_dataclass
    from pymurmur.core.config import (
        DomainConfig, FlockConfig, BoundaryConfig,
        ProjectionConfig, SpatialConfig, FieldConfig, VicsekConfig,
        InfluencerConfig, IndexConfig, RefinementConfig,
        ExtensionConfig, PredatorConfig, EcologyConfig, PerfConfig,
        VizConfig, CaptureConfig,
    )

    classes = [
        DomainConfig, FlockConfig, BoundaryConfig,
        ProjectionConfig, SpatialConfig, FieldConfig, VicsekConfig,
        InfluencerConfig, IndexConfig, RefinementConfig,
        ExtensionConfig, PredatorConfig, EcologyConfig, PerfConfig,
        VizConfig, CaptureConfig,
    ]

    for cls in classes:
        assert is_dataclass(cls), f"{cls.__name__} must be a dataclass"

        # Each must be independently instantiable (no required parent)
        instance = cls()
        assert instance is not None, f"{cls.__name__}() returned None"
