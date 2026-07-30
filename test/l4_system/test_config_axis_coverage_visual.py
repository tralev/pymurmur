"""Config modular-axis coverage guards, part 2 — visual/behavioral axes
(theme, trails, predator_mode), speed-law axes (spatial.speed_mode,
angle_speed_mode, neighbor_adaptive_speed.mode), bird_mesh, and the
smaller round-4 "deep knob" guards (spatial_index, steric_visible_only,
disabled_terms, heading_inertia).

Split out of test_config_axis_coverage.py (file-size split) — that file
itself was already split out of test_config_files.py in round 3 and
grew back over the 600-line guideline after round 4's additions. Part 1
keeps the extension/kernel/noise_mode/velocity_init/neighbor_filter/
position_init family.
"""

from pathlib import Path

CONF_DIR = Path("conf")
ALL_CONFIGS = sorted(CONF_DIR.glob("*.yaml"))
KERNEL_CONF_DIR = CONF_DIR / "kernels"
ALL_KERNEL_CONFIGS = sorted(KERNEL_CONF_DIR.glob("*.yaml"))
FILTER_CONF_DIR = CONF_DIR / "filters"
ALL_FILTER_CONFIGS = sorted(FILTER_CONF_DIR.glob("*.yaml"))
SPEED_LAW_CONF_DIR = CONF_DIR / "speed_laws"
ALL_SPEED_LAW_CONFIGS = sorted(SPEED_LAW_CONF_DIR.glob("*.yaml"))
# Sibling of speed_laws/ but a genuinely different config field
# (spatial.speed_mode, not angle_speed_mode) -- deliberately its own
# directory rather than mixed into speed_laws/, since config_sections.py
# itself notes angle mode's field is named angle_speed_mode specifically
# to avoid colliding with this one (see
# test_every_speed_mode_has_example_coverage's docstring).
SPEED_MODEL_CONF_DIR = CONF_DIR / "speed_models"
ALL_SPEED_MODEL_CONFIGS = sorted(SPEED_MODEL_CONF_DIR.glob("*.yaml"))


def _load_config(path: Path):
    """Parse a YAML config file and return the raw dict."""
    import yaml
    return yaml.safe_load(path.read_text()) or {}


def _find_any_section_value(data: dict, key: str):
    """Look for `key` at top level or nested one level into any dict
    section, trying both `key` and `{section}_{key}` (the same
    de-prefix/full-prefix ambiguity config_io.py's loader itself
    tolerates). Returns the first match found, or None. See
    test_config_axis_coverage.py's copy of this helper for the full
    rationale.
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


class TestConfigAxisCoverageVisual:
    """Every modular-axis value is exercised by at least one shipped preset."""

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

    def test_speed_model_configs_parse_and_validate(self):
        """Every conf/speed_models/*.yaml loads and validates via SimConfig.

        Mirrors test_speed_law_configs_parse_and_validate for the
        spatial.speed_mode axis (added alongside noise_modulated/
        velocity_adaptive completing SPEED_MODEL_REGISTRY's 6-strategy
        taxonomy).
        """
        from pymurmur.core.config import SimConfig

        assert len(ALL_SPEED_MODEL_CONFIGS) >= 2, (
            f"Expected >= 2 speed-model configs, found {len(ALL_SPEED_MODEL_CONFIGS)}"
        )
        for path in ALL_SPEED_MODEL_CONFIGS:
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
            + ALL_SPEED_LAW_CONFIGS + ALL_SPEED_MODEL_CONFIGS
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

    def test_every_spatial_index_has_example_coverage(self):
        """Every valid performance.spatial_index value appears as a
        literal value in at least one shipped preset.
        _VALID_INDEX_TYPES is a real SimConfig class attribute, imported
        live. Unlike every other guard in this file, all 4 values
        (auto/kdtree/hash_grid/none) already had real preset coverage
        before this test was added — every preset sets spatial_index
        (test_config_performance_fields_present enforces presence, just
        not value diversity). This guard's only job is preventing future
        drift if a 5th index strategy is ever registered without a
        preset to demonstrate it."""
        from pymurmur.core.config import SimConfig

        all_data = [
            _load_config(path)
            for path in ALL_CONFIGS + ALL_KERNEL_CONFIGS + ALL_FILTER_CONFIGS
            + ALL_SPEED_LAW_CONFIGS
        ]

        uncovered = [
            index_type for index_type in SimConfig._VALID_INDEX_TYPES
            if not any(
                data.get("performance", {}).get("spatial_index") == index_type
                for data in all_data
            )
        ]
        assert not uncovered, (
            f"spatial_index values with zero preset coverage: {uncovered}. Add a "
            f"performance.spatial_index: {{value}} preset."
        )

    def test_steric_visible_only_has_example_coverage(self):
        """Some shipped preset has refinements.steric_visible_only: true.

        Real RefinementConfig field (occlusion-gated steric repulsion —
        restricts the steric force to occlusion-visible neighbors only)
        with zero preset coverage before conf/filters/filter_metric.yaml.
        No registry exists for this single boolean, so it's a direct
        assertion mirroring test_priority_stack_has_example_coverage.
        """
        covered = any(
            _load_config(path).get("refinements", {}).get("steric_visible_only") is True
            for path in ALL_CONFIGS + ALL_KERNEL_CONFIGS + ALL_FILTER_CONFIGS
            + ALL_SPEED_LAW_CONFIGS
        )
        assert covered, (
            "No preset has refinements.steric_visible_only: true — "
            "see conf/filters/filter_metric.yaml."
        )

    def test_disabled_terms_has_example_coverage(self):
        """Some shipped preset has a non-empty field.disabled_terms list.

        Real FieldConfig field (pymurmur/physics/forces/field.py's
        FIELD_TERMS table lists 11 named sub-terms disabled_terms can
        skip at runtime) with zero preset coverage before
        conf/field_lava_lamp.yaml — it only ever appeared as an empty
        `[]` example value in the non-loadable
        conf/examples/murmuration_nested.yaml reference comment, which
        doesn't actually demonstrate the feature."""
        covered = any(
            bool(_load_config(path).get("field", {}).get("disabled_terms"))
            for path in ALL_CONFIGS
        )
        assert covered, (
            "No preset has a non-empty field.disabled_terms list — "
            "see conf/field_lava_lamp.yaml."
        )

    def test_heading_inertia_has_example_coverage(self):
        """Some preset sets projection.heading_inertia nonzero, and some
        preset sets vicsek.heading_inertia nonzero.

        Two independent real fields (ProjectionConfig.
        projection_heading_inertia, VicsekConfig.vicsek_heading_inertia,
        config_sections.py:55,241) that both defaulted to 0.0 with zero
        preset ever setting either explicitly, before
        conf/murmuration.yaml / conf/murmuration_vicsek.yaml. No live
        registry for either (plain floats), so this is a direct
        nonzero-value assertion rather than an enumerated-set scan.
        """
        all_data = [_load_config(path) for path in ALL_CONFIGS]

        projection_covered = any(
            (data.get("projection", {}).get("heading_inertia") or 0) != 0
            for data in all_data
        )
        vicsek_covered = any(
            (data.get("vicsek", {}).get("heading_inertia") or 0) != 0
            for data in all_data
        )
        assert projection_covered, (
            "No preset has projection.heading_inertia != 0 — see conf/murmuration.yaml."
        )
        assert vicsek_covered, (
            "No preset has vicsek.heading_inertia != 0 — see conf/murmuration_vicsek.yaml."
        )
