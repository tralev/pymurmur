"""Config file validation — all 7 shipped conf/*.yaml files must be valid.

Tests that every config preset in conf/ parses correctly, has the
required top-level sections, and has consistent values.
"""

from pathlib import Path

CONF_DIR = Path("conf")
ALL_CONFIGS = sorted(CONF_DIR.glob("*.yaml"))


def _load_config(path: Path):
    """Parse a YAML config file and return the raw dict."""
    import yaml
    return yaml.safe_load(path.read_text()) or {}


class TestConfigFileValidation:
    """All shipped config files are valid and complete."""

    def test_all_7_configs_parse(self):
        """All conf/*.yaml files parse without error."""
        assert len(ALL_CONFIGS) >= 7, f"Expected ≥ 7 configs, found {len(ALL_CONFIGS)}"
        for path in ALL_CONFIGS:
            data = _load_config(path)
            assert isinstance(data, dict), f"{path.name}: not a dict"

    def test_config_required_fields_present(self):
        """Each config has domain, flock, mode, boundary."""
        for path in ALL_CONFIGS:
            data = _load_config(path)
            for field in ["domain", "flock", "mode", "boundary_mode"]:
                assert field in data, f"{path.name}: missing '{field}'"

    def test_config_performance_fields_present(self):
        """All 7 configs have performance.use_numba and performance.spatial_index."""
        for path in ALL_CONFIGS:
            data = _load_config(path)
            perf = data.get("performance", {})
            assert "use_numba" in perf, f"{path.name}: perforce.use_numba missing"
            assert "spatial_index" in perf, f"{path.name}: perforce.spatial_index missing"

    def test_config_metrics_fields_present(self):
        """All 7 configs have metrics.detail_level and metrics.interval."""
        for path in ALL_CONFIGS:
            data = _load_config(path)
            metrics = data.get("metrics", {})
            assert "detail_level" in metrics, f"{path.name}: metrics.detail_level missing"
            assert "interval" in metrics, f"{path.name}: metrics.interval missing"

    def test_config_modes_valid(self):
        """Config mode is one of the 5 valid values."""
        valid = {"projection", "spatial", "field", "vicsek", "influencer"}
        for path in ALL_CONFIGS:
            data = _load_config(path)
            mode = data.get("mode", "")
            assert mode in valid, f"{path.name}: mode='{mode}' not in {valid}"

    def test_config_boundary_valid(self):
        """Config boundary is one of the valid values."""
        valid = {"toroidal", "open", "margin", "sphere"}
        for path in ALL_CONFIGS:
            data = _load_config(path)
            boundary = data.get("boundary_mode", "")
            assert boundary in valid, f"{path.name}: boundary='{boundary}' not in {valid}"

    def test_spatial_config_has_predator(self):
        """murmuration_spatial.yaml has extensions.predator: true."""
        path = CONF_DIR / "murmuration_spatial.yaml"
        if path.exists():
            data = _load_config(path)
            ext = data.get("extensions", {})
            assert ext.get("predator") is True, "spatial config should have predator enabled"

    def test_field_config_has_wander(self):
        """murmuration_field.yaml has extensions.wander: true."""
        path = CONF_DIR / "murmuration_field.yaml"
        if path.exists():
            data = _load_config(path)
            ext = data.get("extensions", {})
            assert ext.get("wander") is True, "field config should have wander enabled"

    def test_300k_config_numba_enabled(self):
        """murmuration_300k.yaml has performance.use_numba: true."""
        path = CONF_DIR / "murmuration_300k.yaml"
        if path.exists():
            data = _load_config(path)
            perf = data.get("performance", {})
            assert perf.get("use_numba") is True, "300K config should have numba enabled"

    def test_300k_config_kdtree(self):
        """murmuration_300k.yaml has performance.spatial_index: kdtree."""
        path = CONF_DIR / "murmuration_300k.yaml"
        if path.exists():
            data = _load_config(path)
            perf = data.get("performance", {})
            assert perf.get("spatial_index") == "kdtree", (
                "300K config should use kdtree spatial index"
            )
