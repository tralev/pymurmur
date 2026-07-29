"""Config modular-axis coverage guards, part 1 — extensions, kernels,
noise_mode, velocity_init, neighbor_filter, position_init. Every
registered/enumerated value for a given axis must appear as literal
text in at least one shipped conf/*.yaml preset.

Split out of test_config_files.py (file-size split) — that file keeps
the basic structural validation (required fields, mode/boundary
validity, strict-load, sentinel value checks). Part 2
(test_config_axis_coverage_visual.py) holds the visual/predator/
speed-law/bird_mesh/deep-knob axes — this file itself was re-split
after round 4's additions pushed it back over the 600-line guideline.
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
# Narrow neighbor_filter-coverage fixtures — same rationale/precedent as
# ALL_KERNEL_CONFIGS. See conf/filters/filter_metric.yaml.
FILTER_CONF_DIR = CONF_DIR / "filters"
ALL_FILTER_CONFIGS = sorted(FILTER_CONF_DIR.glob("*.yaml"))
# Narrow speed-law-coverage fixtures — same rationale/precedent as
# ALL_KERNEL_CONFIGS/ALL_FILTER_CONFIGS. See
# conf/speed_laws/speed_law_quadratic.yaml.
SPEED_LAW_CONF_DIR = CONF_DIR / "speed_laws"
ALL_SPEED_LAW_CONFIGS = sorted(SPEED_LAW_CONF_DIR.glob("*.yaml"))


def _load_config(path: Path):
    """Parse a YAML config file and return the raw dict."""
    import yaml
    return yaml.safe_load(path.read_text()) or {}


def _find_any_section_value(data: dict, key: str):
    """Look for `key` at top level or nested one level into any dict
    section, trying both `key` and `{section}_{key}` (the same
    de-prefix/full-prefix ambiguity config_io.py's loader itself
    tolerates — e.g. a `predator:` section may spell it `mode:` or
    `predator_mode:`, and direct fields like position_init/velocity_init
    may be nested under any section, e.g. field_silk_sheet.yaml's
    `flock.position_init`). Returns the first match found, or None.

    Only needed for coverage-scanning tests that read raw YAML text
    directly — SimConfig.from_file() resolves this ambiguity for real
    loading, but these tests intentionally avoid a full SimConfig load
    per preset for speed across dozens of files.
    """
    if key in data:
        return data[key]
    for section_name, section_data in data.items():
        if not isinstance(section_data, dict):
            continue
        if key in section_data:
            return section_data[key]
        prefixed = f"{section_name}_{key}"
        if prefixed in section_data:
            return section_data[prefixed]
    return None


class TestConfigAxisCoverage:
    """Every modular-axis value is exercised by at least one shipped preset."""

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

    def test_filter_configs_parse_and_validate(self):
        """Every conf/filters/*.yaml loads and validates via SimConfig.

        Mirrors test_kernel_configs_parse_and_validate for the sibling
        conf/filters/ family (neighbor_filter coverage).
        """
        from pymurmur.core.config import SimConfig

        assert len(ALL_FILTER_CONFIGS) >= 4, (
            f"Expected >= 4 filter-coverage configs, found {len(ALL_FILTER_CONFIGS)}"
        )
        for path in ALL_FILTER_CONFIGS:
            cfg = SimConfig.from_file(str(path))
            cfg.validate()

    def test_every_neighbor_filter_has_example_coverage(self):
        """Every valid spatial.neighbor_filter value appears as a literal
        value in at least one shipped preset. No live registry exists
        for neighbor_filter (bare str field, inline comment in
        config_sections.py's SpatialConfig) so the valid set is
        hardcoded here — mirrors test_every_noise_mode_has_example_coverage's
        precedent. Before conf/filters/ was added, only "hybrid"
        (murmuration_starlings.yaml) had any preset coverage —
        metric/topological/global/none had zero."""
        valid_neighbor_filters = {"hybrid", "metric", "topological", "global", "none"}
        all_data = [
            _load_config(path)
            for path in ALL_CONFIGS + ALL_KERNEL_CONFIGS + ALL_FILTER_CONFIGS
        ]

        uncovered = [
            mode for mode in valid_neighbor_filters
            if not any(data.get("spatial", {}).get("neighbor_filter") == mode for data in all_data)
        ]
        assert not uncovered, (
            f"neighbor_filter values with zero preset coverage: {uncovered}. Add a "
            f"spatial.neighbor_filter: {{value}} preset, e.g. under conf/filters/."
        )

    def test_every_position_init_has_example_coverage(self):
        """Every valid position_init value appears as a literal value in
        at least one shipped preset, EXCEPT "influencer_density" which
        is checked separately: engine.py's _apply_influencer_density_init
        auto-triggers it whenever mode=="influencer" and
        influencer.density_scaled_init is set (murmuration_influencer.yaml
        does exactly this) — so it's genuinely exercised despite never
        appearing as literal position_init: text anywhere. Scanning for
        the literal string would report a false gap.

        _VALID_POSITION_INITS is a real SimConfig class attribute,
        imported live (mirrors velocity_init's guard). Before this, only
        "gaussian" (field_silk_sheet.yaml) had explicit coverage —
        box/random/sphere/grid/sphere_shell/blob had zero."""
        from pymurmur.core.config import SimConfig

        all_data = [
            _load_config(path)
            for path in ALL_CONFIGS + ALL_KERNEL_CONFIGS + ALL_FILTER_CONFIGS
        ]

        influencer_density_covered = any(
            data.get("mode") == "influencer"
            and data.get("influencer", {}).get("density_scaled_init") is True
            for data in all_data
        )

        uncovered = []
        for mode in SimConfig._VALID_POSITION_INITS:
            if mode == "influencer_density":
                if not influencer_density_covered:
                    uncovered.append(f"{mode} (checked via auto-trigger, not literal text)")
                continue
            if not any(_find_any_section_value(data, "position_init") == mode for data in all_data):
                uncovered.append(mode)

        assert not uncovered, (
            f"position_init values with zero preset coverage: {uncovered}. Add a "
            f"top-level position_init: {{value}} preset, e.g. under conf/kernels/."
        )

