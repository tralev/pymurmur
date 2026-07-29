"""Config file validation — all shipped conf/*.yaml files must be valid.

Tests that every config preset in conf/ parses correctly, has the
required top-level sections, and has consistent values.
"""

from pathlib import Path

CONF_DIR = Path("conf")
ALL_CONFIGS = sorted(CONF_DIR.glob("*.yaml"))
# Narrow kernel-coverage fixtures — real/loadable/validated, but kept
# out of the top-level conf/*.yaml glob (and so out of --list-configs)
# since they're technical coverage fixtures, not exploration-worthy
# presets. See conf/kernels/kernel_sum.yaml for why they exist.
KERNEL_CONF_DIR = CONF_DIR / "kernels"
ALL_KERNEL_CONFIGS = sorted(KERNEL_CONF_DIR.glob("*.yaml"))


def _load_config(path: Path):
    """Parse a YAML config file and return the raw dict."""
    import yaml
    return yaml.safe_load(path.read_text()) or {}


class TestConfigFileValidation:
    """All shipped config files are valid and complete."""

    def test_all_configs_parse(self):
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
        """All configs have performance.spatial_index."""
        for path in ALL_CONFIGS:
            data = _load_config(path)
            perf = data.get("performance", {})
            assert "spatial_index" in perf, f"{path.name}: perforce.spatial_index missing"

    def test_config_metrics_fields_present(self):
        """All configs have metrics.detail_level and metrics.interval."""
        for path in ALL_CONFIGS:
            data = _load_config(path)
            metrics = data.get("metrics", {})
            assert "detail_level" in metrics, f"{path.name}: metrics.detail_level missing"
            assert "interval" in metrics, f"{path.name}: metrics.interval missing"

    def test_config_modes_valid(self):
        """Config mode is a registered ForceMode (S2.C8: was a stale
        hardcoded 5-mode set that predated angle/marl registration)."""
        from pymurmur.physics.plugins.force_mode import MODE_REGISTRY
        valid = set(MODE_REGISTRY.keys())
        for path in ALL_CONFIGS:
            data = _load_config(path)
            mode = data.get("mode", "")
            assert mode in valid, f"{path.name}: mode='{mode}' not in {valid}"

    def test_config_boundary_valid(self):
        """Config boundary is one of the valid values."""
        valid = {"toroidal", "open", "margin", "sphere", "sphere_soft"}
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

    def test_priority_stack_has_example_coverage(self):
        """Some shipped preset has extensions.priority_stack_enabled: true.

        priority_stack_enabled is real ExtensionConfig field
        (pymurmur/physics/priority_stack.py's tier1/2/3 force-budget
        cascade) but engine-level branching, not a class-based
        Extension — structurally invisible to
        test_every_extension_has_example_coverage's live
        EXTENSION_REGISTRY scan. No registry exists for this single
        flag, so it's a direct assertion (mirrors
        test_spatial_config_has_predator) rather than a registry-driven
        loop.
        """
        covered = any(
            _load_config(path).get("extensions", {}).get("priority_stack_enabled") is True
            for path in ALL_CONFIGS
        )
        assert covered, (
            "No conf/*.yaml has extensions.priority_stack_enabled: true — "
            "see conf/murmuration_obstacles.yaml."
        )

    def test_obstacles_demonstrated_outside_evoflock(self):
        """Some preset other than murmuration_evo.yaml has a non-empty
        obstacles: scene, run through the normal (non-GA) CLI path.

        Before conf/murmuration_obstacles.yaml, obstacles: only appeared
        in murmuration_evo.yaml's headless GA training scene — and
        __main__.py never wired obstacles: into the engine at all for
        the normal CLI path (only EvoFlock's own runner did), so the
        section was silently inert for every other preset regardless.
        """
        covered = any(
            path.name != "murmuration_evo.yaml" and bool(_load_config(path).get("obstacles"))
            for path in ALL_CONFIGS
        )
        assert covered, (
            "No conf/*.yaml other than murmuration_evo.yaml has a non-empty "
            "obstacles: scene — see conf/murmuration_obstacles.yaml."
        )

    def test_every_extension_has_example_coverage(self):
        """Every registered extension is enabled=true in at least one
        shipped preset — no more silent gaps like the one this test
        closes: SpeedNoise, NeighborAdaptiveSpeed, DynamicVisionRange,
        and BoidStateMachine had zero preset coverage anywhere (the
        same 4 extensions that were also missing from arch.md §7's
        table until that was fixed separately) until
        conf/murmuration_showcase.yaml was added specifically to
        demonstrate them. Derived from the live EXTENSION_REGISTRY, not
        a hardcoded list, so a newly-registered extension with no
        example preset fails this test immediately instead of drifting
        silently."""
        from pymurmur.physics.extensions.extension_registry import EXTENSION_REGISTRY

        all_extensions_data = [_load_config(path) for path in ALL_CONFIGS]
        uncovered = []
        for cls, config_attr, _cleanup_attr in EXTENSION_REGISTRY:
            covered = any(
                data.get("extensions", {}).get(config_attr) is True
                for data in all_extensions_data
            )
            if not covered:
                uncovered.append(f"{cls.__name__} ({config_attr})")

        assert not uncovered, (
            f"Extensions with zero preset coverage (enabled=true in no "
            f"conf/*.yaml): {uncovered}. Add extensions.{{config_attr}}: "
            f"true to at least one preset, e.g. conf/murmuration_showcase.yaml."
        )

    def test_all_top_level_configs_strict_load_and_validate(self):
        """Every conf/*.yaml loads via the real strict SimConfig.from_file()
        loader and passes validate() — not just a raw yaml.safe_load dict
        check (that's what test_all_configs_parse does, and it's why
        conf/murmuration_field.yaml was able to ship with 16 dead field:
        keys — leftover from an earlier, richer field-force design that
        was simplified out of FieldConfig — and crash on the normal CLI
        path undetected until this test was added).

        Three presets are narrowly skipped by name, each a pre-existing,
        out-of-scope bug flagged (not fixed) here:
          - murmuration_influencer.yaml / murmuration_field.yaml both ship
            visual_range=0.0 (neither mode queries neighbors, so it's a
            documented no-op value) but validate() rejects visual_range<=0
            unconditionally for every mode. Matches
            scripts/generate_examples.py's workaround.
          - murmuration_300k.yaml ships viz.fps=0 ("uncapped, measure real
            throughput" per its own comment) but validate() rejects
            viz.fps<=0 unconditionally.
        murmuration_evo.yaml is excluded structurally, not as a bug: it
        carries EvoFlock-only sections (evoflock/objectives/
        parameters_to_optimize) that collide with real SimConfig field
        names under strict parsing and is loaded via
        SimConfig.from_file(path, strict=False) plus EvoFlock-specific
        extraction elsewhere (__main__.py, test_evoflock_*.py) — never
        through this plain strict+validate path.
        """
        from pymurmur.core.config import SimConfig

        known_validation_bugs = {
            "murmuration_influencer.yaml",
            "murmuration_field.yaml",
            "murmuration_300k.yaml",
        }
        structurally_excluded = {"murmuration_evo.yaml"}
        for path in ALL_CONFIGS:
            if path.name in known_validation_bugs or path.name in structurally_excluded:
                continue
            cfg = SimConfig.from_file(str(path))
            cfg.validate()

    def test_kernel_configs_parse_and_validate(self):
        """Every conf/kernels/*.yaml loads and validates via SimConfig."""
        from pymurmur.core.config import SimConfig

        assert len(ALL_KERNEL_CONFIGS) >= 9, (
            f"Expected >= 9 kernel-coverage configs, found {len(ALL_KERNEL_CONFIGS)}"
        )
        for path in ALL_KERNEL_CONFIGS:
            cfg = SimConfig.from_file(str(path))
            cfg.validate()

    def test_every_kernel_has_example_coverage(self):
        """Every registered separation/alignment/cohesion kernel appears
        as a spatial.{kernel}_kernel value in at least one shipped
        preset (conf/*.yaml or conf/kernels/*.yaml) — mirrors
        test_every_extension_has_example_coverage's closure of the
        analogous gap. Before conf/kernels/ was added, 8 of 11
        separation kernels, 2 of 4 alignment kernels, and 1 of 3
        cohesion kernels had zero preset coverage — only named in
        conf/examples/murmuration_nested.yaml's comments. Derived from
        the live kernel registries, not a hardcoded list."""
        from pymurmur.physics.plugins.kernel_registry import (
            ALIGNMENT_KERNEL_REGISTRY,
            COHESION_KERNEL_REGISTRY,
            SEPARATION_KERNEL_REGISTRY,
        )

        all_data = [_load_config(path) for path in ALL_CONFIGS + ALL_KERNEL_CONFIGS]

        def covered(field_name: str, value: str) -> bool:
            return any(
                data.get("spatial", {}).get(field_name) == value
                for data in all_data
            )

        uncovered = []
        for registry, field_name in (
            (SEPARATION_KERNEL_REGISTRY, "separation_kernel"),
            (ALIGNMENT_KERNEL_REGISTRY, "alignment_kernel"),
            (COHESION_KERNEL_REGISTRY, "cohesion_kernel"),
        ):
            for name in registry:
                if not covered(field_name, name):
                    uncovered.append(f"{field_name}={name}")

        assert not uncovered, (
            f"Kernels with zero preset coverage: {uncovered}. Add a "
            f"spatial.{{field}}: {{name}} preset, e.g. under conf/kernels/."
        )

    def test_every_noise_mode_has_example_coverage(self):
        """Every valid spatial.noise_mode value appears as a literal
        value in at least one shipped preset. No live registry exists
        for noise_mode (it's a bare str field with an inline comment
        listing the valid set, config_sections.py's SpatialConfig,
        unlike kernels/extensions) so the valid set is hardcoded here —
        mirrors test_config_boundary_valid's existing precedent for the
        same reason. Before this, only "velocity"
        (murmuration_boids.yaml) and the implicit "additive" default had
        any real preset behind them; maxwellian/none/seed_sinusoidal had
        zero coverage anywhere."""
        valid_noise_modes = {"additive", "maxwellian", "none", "seed_sinusoidal", "velocity"}
        all_data = [_load_config(path) for path in ALL_CONFIGS + ALL_KERNEL_CONFIGS]

        uncovered = [
            mode for mode in valid_noise_modes
            if not any(data.get("spatial", {}).get("noise_mode") == mode for data in all_data)
        ]
        assert not uncovered, (
            f"noise_mode values with zero preset coverage: {uncovered}. Add a "
            f"spatial.noise_mode: {{value}} preset, e.g. under conf/kernels/."
        )

    def test_every_velocity_init_has_example_coverage(self):
        """Every valid velocity_init value appears as a literal value in
        at least one shipped preset. Unlike noise_mode,
        _VALID_VELOCITY_INITS is a real SimConfig class attribute, so
        it's imported live rather than hardcoded. Before this, only the
        implicit "sphere" default had any real preset behind it —
        blob/drift/cube/speed_uniform/tangential/fixed had zero coverage
        anywhere (only named in conf/examples/murmuration_nested.yaml's
        comments)."""
        from pymurmur.core.config import SimConfig

        all_data = [_load_config(path) for path in ALL_CONFIGS + ALL_KERNEL_CONFIGS]

        uncovered = [
            mode for mode in SimConfig._VALID_VELOCITY_INITS
            if not any(data.get("velocity_init") == mode for data in all_data)
        ]
        assert not uncovered, (
            f"velocity_init values with zero preset coverage: {uncovered}. Add a "
            f"top-level velocity_init: {{value}} preset, e.g. under conf/kernels/."
        )

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

    def test_marl_config_sentinel_values(self):
        """S7.1: murmuration_marl.yaml carries the source-parity vector.

        action_scale=0.1, velocity_cap=0.1, rule_weight=0.01,
        separation_radius=0.2, episode_steps=500, num_boids=200,
        seed=42, boundary=open, dual_view=true.
        """
        path = CONF_DIR / "murmuration_marl.yaml"
        assert path.exists(), "murmuration_marl.yaml must exist"
        data = _load_config(path)

        assert data["mode"] == "marl"
        assert data["boundary_mode"] == "open"
        assert data["seed"] == 42

        flock = data["flock"]
        assert flock["num_boids"] == 200

        m = data["marl"]
        assert m["action_scale"] == 0.1
        assert m["velocity_cap"] == 0.1
        assert m["rule_weight"] == 0.01
        assert m["separation_radius"] == 0.2
        assert m["episode_steps"] == 500

        assert data["visual"]["dual_view"] is True
