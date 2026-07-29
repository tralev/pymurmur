"""Config modular-axis coverage guards — every registered/enumerated
value for a given axis (kernels, extensions, noise_mode, velocity_init,
position_init, neighbor_filter, theme, trails, predator_mode, speed_mode,
angle_speed_mode, neighbor_adaptive_speed.mode, bird_mesh, ...) must
appear as literal text in at least one shipped conf/*.yaml preset.

Split out of test_config_files.py (file-size split) — that file keeps
the basic structural validation (required fields, mode/boundary
validity, strict-load, sentinel value checks) and this one holds the
"*_has_example_coverage" family that grew across three rounds of
DeepSeek-flagged conf/examples coverage work.
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

    def test_every_theme_has_example_coverage(self):
        """Every valid visual theme appears as a literal value in at
        least one shipped preset. _VALID_THEMES is a real SimConfig
        class attribute, imported live. Before conf/filters/ was added,
        ink/inverse/paper were covered but graphite/heading had zero."""
        from pymurmur.core.config import SimConfig

        all_data = [
            _load_config(path)
            for path in ALL_CONFIGS + ALL_KERNEL_CONFIGS + ALL_FILTER_CONFIGS
        ]

        uncovered = [
            theme for theme in SimConfig._VALID_THEMES
            if not any(_find_any_section_value(data, "theme") == theme for data in all_data)
        ]
        assert not uncovered, (
            f"theme values with zero preset coverage: {uncovered}. Add a "
            f"visual.theme: {{value}} preset, e.g. under conf/filters/."
        )

    def test_every_trail_mode_has_example_coverage(self):
        """Every valid trails value appears as a literal value in at
        least one shipped preset. No live registry exists for trails
        (bare str field, inline comment in config_sections.py's VizConfig)
        so the valid set is hardcoded here. Before conf/filters/ was
        added, velocity/ring/accumulation were covered but "lines" had
        zero ("off" is the implicit default, not required to be explicit)."""
        valid_trails = {"off", "velocity", "ring", "accumulation", "lines"}
        all_data = [
            _load_config(path)
            for path in ALL_CONFIGS + ALL_KERNEL_CONFIGS + ALL_FILTER_CONFIGS
        ]

        uncovered = [
            trail for trail in valid_trails - {"off"}
            if not any(_find_any_section_value(data, "trails") == trail for data in all_data)
        ]
        assert not uncovered, (
            f"trails values with zero preset coverage: {uncovered}. Add a "
            f"visual.trails: {{value}} preset, e.g. under conf/filters/."
        )

    def test_every_predator_mode_has_example_coverage(self):
        """Every valid predator_mode value appears as a literal value in
        at least one shipped preset, EXCEPT "off" — paired with
        extensions.predator_enabled: true (required for the predator:
        section to matter at all), predator_mode: off is a degenerate,
        self-contradictory combination not worth manufacturing a preset
        for. _VALID_PREDATOR_MODES is a real SimConfig class attribute,
        imported live. Before this, only "orbit"
        (field_predator_ripple.yaml) had explicit coverage — "autonomous"
        is the default everywhere but was never explicit, and "cursor"
        had zero coverage anywhere."""
        from pymurmur.core.config import SimConfig

        all_data = [
            _load_config(path)
            for path in ALL_CONFIGS + ALL_KERNEL_CONFIGS + ALL_FILTER_CONFIGS
        ]

        def predator_mode_of(data: dict):
            # Deliberately NOT _find_any_section_value("mode") — top-level
            # "mode" is the force-mode field (spatial/projection/...);
            # scanning generically would false-match every preset's mode
            # instead of looking inside predator: specifically.
            predator_section = data.get("predator", {})
            if not isinstance(predator_section, dict):
                return None
            return predator_section.get("mode", predator_section.get("predator_mode"))

        uncovered = [
            mode for mode in SimConfig._VALID_PREDATOR_MODES - {"off"}
            if not any(predator_mode_of(data) == mode for data in all_data)
        ]
        assert not uncovered, (
            f"predator_mode values with zero preset coverage: {uncovered}. Add a "
            f"predator.mode: {{value}} preset."
        )

    def test_speed_law_configs_parse_and_validate(self):
        """Every conf/speed_laws/*.yaml loads and validates via SimConfig.

        Mirrors test_kernel_configs_parse_and_validate for the sibling
        conf/speed_laws/ family (angle_speed_mode +
        neighbor_adaptive_speed.mode coverage).
        """
        from pymurmur.core.config import SimConfig

        assert len(ALL_SPEED_LAW_CONFIGS) >= 2, (
            f"Expected >= 2 speed-law configs, found {len(ALL_SPEED_LAW_CONFIGS)}"
        )
        for path in ALL_SPEED_LAW_CONFIGS:
            cfg = SimConfig.from_file(str(path))
            cfg.validate()

    def test_every_speed_mode_has_example_coverage(self):
        """Every registered spatial.speed_mode value appears as a literal
        value in at least one shipped preset. SPEED_MODEL_REGISTRY is a
        real, live registry (pymurmur/physics/plugins/speed_model.py) —
        imported directly, not hardcoded, mirroring
        test_every_kernel_has_example_coverage's precedent. "clamp" and
        "band" are both registered (aliases of the same BandSpeedModel)
        but neither literal spelling appeared in any preset before this
        — SpatialConfig.speed_mode's implicit "clamp" default was never
        made explicit anywhere. "fixed"/"ceiling" were already covered
        (murmuration_starlings.yaml/murmuration_boids.yaml); "none" had
        zero coverage.

        Checked via a section-scoped lookup (spatial.speed_mode), NOT
        _find_any_section_value — angle.yaml's `angle: {speed_mode: ...}`
        also spells its OWN unrelated field (angle_speed_mode) as bare
        "speed_mode" (config_sections.py's own comment: named
        angle_speed_mode specifically because "speed_mode" is already
        SpatialConfig's flat name) — a generic scan would conflate the
        two different config fields that happen to share a YAML spelling.
        """
        from pymurmur.physics.plugins.speed_model import SPEED_MODEL_REGISTRY

        all_data = [
            _load_config(path)
            for path in ALL_CONFIGS + ALL_KERNEL_CONFIGS + ALL_FILTER_CONFIGS
            + ALL_SPEED_LAW_CONFIGS
        ]

        def spatial_speed_mode_of(data: dict):
            spatial_section = data.get("spatial", {})
            if not isinstance(spatial_section, dict):
                return None
            return spatial_section.get("speed_mode")

        uncovered = [
            mode for mode in SPEED_MODEL_REGISTRY
            if not any(spatial_speed_mode_of(data) == mode for data in all_data)
        ]
        assert not uncovered, (
            f"spatial.speed_mode values with zero preset coverage: {uncovered}. "
            f"Add a spatial.speed_mode: {{value}} preset, e.g. under conf/kernels/."
        )

    def test_every_angle_speed_mode_has_example_coverage(self):
        """Every valid angle_speed_mode value appears as a literal value
        (spelled either "speed_mode" or "angle_speed_mode") inside some
        preset's angle: section. _VALID_ANGLE_SPEED_MODES is a real
        SimConfig class attribute, imported live. Before
        conf/speed_laws/ was added, only "linear"
        (murmuration_angle.yaml's default) had explicit coverage —
        quadratic/softened had zero.

        Section-scoped (not _find_any_section_value) for the same
        "speed_mode" spelling-collision reason as
        test_every_speed_mode_has_example_coverage.
        """
        from pymurmur.core.config import SimConfig

        all_data = [
            _load_config(path)
            for path in ALL_CONFIGS + ALL_KERNEL_CONFIGS + ALL_FILTER_CONFIGS
            + ALL_SPEED_LAW_CONFIGS
        ]

        def angle_speed_mode_of(data: dict):
            angle_section = data.get("angle", {})
            if not isinstance(angle_section, dict):
                return None
            return angle_section.get("speed_mode", angle_section.get("angle_speed_mode"))

        uncovered = [
            mode for mode in SimConfig._VALID_ANGLE_SPEED_MODES
            if not any(angle_speed_mode_of(data) == mode for data in all_data)
        ]
        assert not uncovered, (
            f"angle_speed_mode values with zero preset coverage: {uncovered}. "
            f"Add an angle.speed_mode: {{value}} preset, e.g. under conf/speed_laws/."
        )

    def test_every_neighbor_adaptive_speed_mode_has_example_coverage(self):
        """Every valid neighbor_adaptive_speed.mode value appears as a
        literal value (spelled either "mode" or
        "neighbor_adaptive_speed_mode") inside some preset's
        neighbor_adaptive_speed: section. No live registry/class
        attribute exists for this field (unlike angle_speed_mode) so the
        valid set is hardcoded here — mirrors noise_mode's guard
        precedent. Before conf/speed_laws/ was added, only the implicit
        "linear" default was ever exercised (murmuration_showcase.yaml
        enables the extension but never sets mode explicitly) —
        quadratic/softened had zero coverage.

        Section-scoped (not _find_any_section_value) — a bare "mode" key
        would collide with the top-level force-mode field otherwise,
        same reasoning as predator_mode's guard.
        """
        valid_modes = {"linear", "quadratic", "softened"}
        all_data = [
            _load_config(path)
            for path in ALL_CONFIGS + ALL_KERNEL_CONFIGS + ALL_FILTER_CONFIGS
            + ALL_SPEED_LAW_CONFIGS
        ]

        def nas_mode_of(data: dict):
            section = data.get("neighbor_adaptive_speed", {})
            if not isinstance(section, dict):
                return None
            return section.get("mode", section.get("neighbor_adaptive_speed_mode"))

        uncovered = [
            mode for mode in valid_modes
            if not any(nas_mode_of(data) == mode for data in all_data)
        ]
        assert not uncovered, (
            f"neighbor_adaptive_speed.mode values with zero preset coverage: "
            f"{uncovered}. Add a neighbor_adaptive_speed.mode: {{value}} preset, "
            f"e.g. under conf/speed_laws/."
        )

    def test_every_bird_mesh_has_example_coverage(self):
        """Every valid visual.bird_mesh value appears as a literal value
        in at least one shipped preset. _VALID_MESH_NAMES is a real
        SimConfig class attribute (9 values), imported live. Before
        this, only "winged" (murmuration_starlings.yaml) had explicit
        coverage — the other 8 (auto/sphere/tetra/impostor/ellipsoid/
        cone/arrow/points) had zero, including "auto" which is the
        implicit default everywhere.
        """
        from pymurmur.core.config import SimConfig

        all_data = [
            _load_config(path)
            for path in ALL_CONFIGS + ALL_KERNEL_CONFIGS + ALL_FILTER_CONFIGS
            + ALL_SPEED_LAW_CONFIGS
        ]

        uncovered = [
            mesh for mesh in SimConfig._VALID_MESH_NAMES
            if not any(_find_any_section_value(data, "bird_mesh") == mesh for data in all_data)
        ]
        assert not uncovered, (
            f"bird_mesh values with zero preset coverage: {uncovered}. Add a "
            f"visual.bird_mesh: {{value}} preset, e.g. under conf/kernels/."
        )
