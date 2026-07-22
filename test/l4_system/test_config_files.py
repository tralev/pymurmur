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
        """All 7 configs have performance.spatial_index."""
        for path in ALL_CONFIGS:
            data = _load_config(path)
            perf = data.get("performance", {})
            assert "spatial_index" in perf, f"{path.name}: perforce.spatial_index missing"

    def test_config_metrics_fields_present(self):
        """All 7 configs have metrics.detail_level and metrics.interval."""
        for path in ALL_CONFIGS:
            data = _load_config(path)
            metrics = data.get("metrics", {})
            assert "detail_level" in metrics, f"{path.name}: metrics.detail_level missing"
            assert "interval" in metrics, f"{path.name}: metrics.interval missing"

    def test_config_modes_valid(self):
        """Config mode is a registered ForceMode (S2.C8: was a stale
        hardcoded 5-mode set that predated angle/marl registration)."""
        from pymurmur.physics.forces._mode import MODE_REGISTRY
        valid = set(MODE_REGISTRY.keys())
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
        """murmuration_spatial.yaml has extensions.predator_enabled: true."""
        path = CONF_DIR / "murmuration_spatial.yaml"
        if path.exists():
            data = _load_config(path)
            ext = data.get("extensions", {})
            assert ext.get("predator_enabled") is True, "spatial config should have predator enabled"

    def test_field_config_has_wander(self):
        """murmuration_field.yaml has extensions.wander_enabled: true."""
        path = CONF_DIR / "murmuration_field.yaml"
        if path.exists():
            data = _load_config(path)
            ext = data.get("extensions", {})
            assert ext.get("wander_enabled") is True, "field config should have wander enabled"

    def test_300k_config_kdtree(self):
        """murmuration_300k.yaml has performance.spatial_index: kdtree."""
        path = CONF_DIR / "murmuration_300k.yaml"
        if path.exists():
            data = _load_config(path)
            perf = data.get("performance", {})
            assert perf.get("spatial_index") == "kdtree", (
                "300K config should use kdtree spatial index"
            )

    def test_vicsek_config_sentinel_values(self):
        """S2.D4: murmuration_vicsek.yaml carries the source-parity vector.

        n_preys=100, n_predators=1, R_inf=5, R_avoid=1, R_pred=5,
        v=v_pred=1, dt=1, D=0.8, eta=0.8, w_afraid=3, detect_ratio=1.5,
        predator_noise_ratio=0.2, domain 40^3.
        """
        path = CONF_DIR / "murmuration_vicsek.yaml"
        assert path.exists(), "murmuration_vicsek.yaml must exist"
        data = _load_config(path)

        domain = data["domain"]
        assert domain["width"] == domain["height"] == domain["depth"] == 40.0

        flock = data["flock"]
        assert flock["num_boids"] == 101, "100 prey + 1 predator"
        assert flock["visual_range"] == 5.0  # R_inf

        v = data["vicsek"]
        assert v["couplage"] == 0.8          # eta
        assert v["diffusion"] == 0.8         # D
        assert v["time_step"] == 1.0         # dt
        assert v["velocity"] == 1.0          # v
        assert v["radius_influence"] == 5.0  # R_inf
        assert v["radius_avoid"] == 1.0      # R_avoid
        assert v["radius_predators"] == 5.0  # R_pred
        assert v["n_predators"] == 1
        assert v["velocity_predator"] == 1.0  # v_pred
        assert v["predator_noise_ratio"] == 0.2
        assert v["detect_ratio"] == 1.5
        assert v["weight_afraid"] == 3.0
