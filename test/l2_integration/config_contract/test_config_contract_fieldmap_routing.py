"""I7 — Field-map completeness, engine sub-config routing.

Split out of test_config_contract.py (file-size split).
"""


import numpy as np

from pymurmur.core.config import (
    _ALL_FIELD_NAMES,
    _DIRECT_FIELDS,
    _FIELD_MAP,
    _NESTED_ONLY,
    AngleConfig,
    BoundaryConfig,
    CaptureConfig,
    DomainConfig,
    EcologyConfig,
    ExtensionConfig,
    FieldConfig,
    FlockConfig,
    IndexConfig,
    InfluencerConfig,
    MarlConfig,
    PerfConfig,
    PredatorConfig,
    ProjectionConfig,
    RefinementConfig,
    RoostConfig,
    SimConfig,
    SpatialConfig,
    SpeedNoiseConfig,
    VicsekConfig,
    VizConfig,
    WanderConfig,
)


class TestFieldMapCompleteness:
    """IT4: _FIELD_MAP completeness audit.

    Every sub-config dataclass field must have an entry in _FIELD_MAP.
    Every _FIELD_MAP entry must point to a real sub-config attribute and
    a real field name on that sub-config. _ALL_FIELD_NAMES must equal
    set(_FIELD_MAP.keys()) | _DIRECT_FIELDS. Catches silent default
    regressions when new fields are added but forgotten in the map.
    """

    # All 20 sub-config classes (matching config.py __init__)
    _SUBCONFIG_CLASSES: dict[str, type] = {
        "_angle": AngleConfig,
        "_roost": RoostConfig,
        "_marl": MarlConfig,
        "_domain": DomainConfig,
        "_flock": FlockConfig,
        "_boundary": BoundaryConfig,
        "_projection": ProjectionConfig,
        "_spatial": SpatialConfig,
        "_field": FieldConfig,
        "_wander": WanderConfig,
        "_speed_noise": SpeedNoiseConfig,
        "_vicsek": VicsekConfig,
        "_influencer": InfluencerConfig,
        "_index": IndexConfig,
        "_refinement": RefinementConfig,
        "_extension": ExtensionConfig,
        "_predator": PredatorConfig,
        "_ecology": EcologyConfig,
        "_perf": PerfConfig,
        "_viz": VizConfig,
        "_capture": CaptureConfig,
    }

    def test_every_subconfig_field_has_field_map_entry(self):
        """IT4a: Every sub-config dataclass field appears in _FIELD_MAP
        or _NESTED_ONLY.

        If a field is added to a dataclass but forgotten in both maps,
        it silently defaults on YAML round-trip and flat access fails.
        """
        from dataclasses import fields

        cfg = SimConfig()
        mapped_flat_names: set[str] = set()

        for sub_attr, SubClass in self._SUBCONFIG_CLASSES.items():
            getattr(cfg, sub_attr)
            for f in fields(SubClass):
                # Find the flat field name that maps to this sub-attr + field
                found = False
                for flat_name, (mapped_attr, mapped_field) in _FIELD_MAP.items():
                    if mapped_attr == sub_attr and mapped_field == f.name:
                        found = True
                        mapped_flat_names.add(flat_name)
                        break
                # Also check _NESTED_ONLY (fully retired shims — no flat
                # alias; from_file routes their YAML keys explicitly)
                if not found:
                    for _flat_name, (mapped_attr, mapped_field) in _NESTED_ONLY.items():
                        if mapped_attr == sub_attr and mapped_field == f.name:
                            found = True
                            break

                assert found, (
                    f"Sub-config field {SubClass.__name__}.{f.name} "
                    f"has no entry in _FIELD_MAP or _NESTED_ONLY. "
                    f"Add it or it will silently default on YAML load."
                )

    def test_every_field_map_entry_points_to_real_attribute(self):
        """IT4b: Every _FIELD_MAP entry (sub_attr, field_name) is valid.

        sub_attr must be a real attribute of SimConfig (e.g. '_domain').
        field_name must be a real field on that sub-config dataclass.
        """
        from dataclasses import fields

        cfg = SimConfig()

        # Build lookup: sub_attr → set of valid field names
        valid_fields: dict[str, set[str]] = {}
        for sub_attr, SubClass in self._SUBCONFIG_CLASSES.items():
            valid_fields[sub_attr] = {f.name for f in fields(SubClass)}

        for flat_name, (sub_attr, field_name) in _FIELD_MAP.items():
            # sub_attr must exist on SimConfig
            assert hasattr(cfg, sub_attr), (
                f"_FIELD_MAP['{flat_name}'] → sub_attr '{sub_attr}' "
                f"does not exist on SimConfig"
            )

            # field_name must exist on the sub-config dataclass
            assert sub_attr in valid_fields, (
                f"_FIELD_MAP['{flat_name}'] → sub_attr '{sub_attr}' "
                f"not in known sub-config classes"
            )
            assert field_name in valid_fields[sub_attr], (
                f"_FIELD_MAP['{flat_name}'] → ({sub_attr!r}, {field_name!r}) "
                f"but {sub_attr} has no field '{field_name}'. "
                f"Valid fields: {sorted(valid_fields[sub_attr])}"
            )

    def test_all_field_names_is_key_union(self):
        """IT4c: _ALL_FIELD_NAMES == _FIELD_MAP keys ∪ _DIRECT_FIELDS ∪ _NESTED_ONLY keys.

        Nested-only fields (retired shims) must still be included: from_file()'s
        strict unknown-key check tests membership in _ALL_FIELD_NAMES *before*
        it routes nested-only keys to their sub-config, so omitting them here
        makes from_file() reject their YAML keys as unknown (a real bug caught
        by test_save_preserves_angle_config_fields — a round-trip through
        phi_p raised "Unknown config keys" until this set included it).
        """
        expected = set(_FIELD_MAP.keys()) | _DIRECT_FIELDS | set(_NESTED_ONLY.keys())
        assert _ALL_FIELD_NAMES == expected, (
            f"_ALL_FIELD_NAMES is out of sync.\n"
            f"Expected: {sorted(expected)}\n"
            f"Got:      {sorted(_ALL_FIELD_NAMES)}\n"
            f"Missing from _ALL_FIELD_NAMES: {expected - _ALL_FIELD_NAMES}\n"
            f"Extra in _ALL_FIELD_NAMES: {_ALL_FIELD_NAMES - expected}"
        )

    def test_no_duplicate_field_map_entries(self):
        """IT4d: No two flat field names map to the same (sub_attr, field).

        Two flat names pointing to the same sub-config field would
        cause silent overwrites during __setattr__.
        """
        seen_targets: set[tuple[str, str]] = set()
        for _flat_name, target in _FIELD_MAP.items():
            assert target not in seen_targets, (
                f"_FIELD_MAP has duplicate target {target}: "
                f"already mapped from another flat name"
            )
            seen_targets.add(target)

    def test_no_dead_field_map_entries(self):
        """IT4e: No _FIELD_MAP entries point to sub-configs not in _SUBCONFIG_CLASSES.

        Prevents stale entries when sub-configs are renamed or removed.
        """
        known_sub_attrs = set(self._SUBCONFIG_CLASSES.keys())
        for flat_name, (sub_attr, _field_name) in _FIELD_MAP.items():
            assert sub_attr in known_sub_attrs, (
                f"_FIELD_MAP['{flat_name}'] → sub_attr '{sub_attr}' "
                f"not in known sub-config classes: {sorted(known_sub_attrs)}"
            )

    def test_subconfig_imports_used_in_test(self):
        """IT4f: All 16 sub-config classes are importable and have fields."""
        from dataclasses import fields

        for sub_attr, SubClass in self._SUBCONFIG_CLASSES.items():
            sub_fields = list(fields(SubClass))
            assert len(sub_fields) > 0, (
                f"Sub-config {SubClass.__name__} (attr {sub_attr}) "
                f"has no dataclass fields — is it empty?"
            )


# ═══════════════════════════════════════════════════════════════════
# I7 Medium-Priority Integration Tests — cross I7.1 + I7.4 + I4.2
# ═══════════════════════════════════════════════════════════════════


class TestEngineSubconfigRouting:
    """IT5: Engine reads from correct sub-config despite flat-access ambiguity.

    DomainConfig and CaptureConfig both have fields with overlapping flat names
    (width/height). This verifies the __getattr__ delegation routes to the
    correct sub-config and the engine's hot path uses domain dimensions,
    not capture dimensions.
    """

    def test_flat_access_routes_to_domain_not_capture_when_both_set(self):
        """IT5a: config.width returns domain.width, not capture.capture_width.

        When both domain.width=2000 and capture.capture_width=800 are set,
        the flat access config.width must return 2000 (domain), not 800 (capture).
        """
        from pymurmur import SimConfig

        cfg = SimConfig()
        # Set conflicting values via sub-config accessors
        cfg.domain.width = 2000.0
        cfg.capture.capture_width = 800

        # Flat access must route to domain, not capture
        assert cfg.width == 2000.0, (
            f"config.width should be domain.width=2000, got {cfg.width}. "
            f"Flat access may be routing to capture.capture_width instead."
        )
        assert cfg.height == 700.0, (
            f"config.height should be domain.height=700 (default), got {cfg.height}"
        )

    def test_engine_uses_domain_dimensions_not_capture(self):
        """IT5b: Engine step() uses domain dimensions, not capture dimensions.

        Set domain.width=2000, capture.capture_width=800. After stepping,
        verify boids can move beyond capture_width=800 but stay within
        domain.width=2000 (toroidal wrapping).
        """
        from pymurmur import SimulationEngine

        cfg = SimConfig()
        cfg.domain.width = 2000.0
        cfg.domain.height = 1400.0
        cfg.domain.depth = 800.0
        # Purposely set capture dimensions to a much smaller value
        cfg.capture.capture_width = 400
        cfg.capture.capture_height = 300
        cfg.num_boids = 50
        cfg.v0 = 8.0
        cfg.seed = 42
        cfg.boundary_mode = "toroidal"

        # Flat access must return domain values
        assert cfg.width == 2000.0, (
            f"config.width must be 2000.0 (domain), got {cfg.width}"
        )
        assert cfg.height == 1400.0
        assert cfg.depth == 800.0

        engine = SimulationEngine(cfg)

        # Run enough steps for boids to travel beyond capture_width
        for _ in range(100):
            engine.step()

        pos = engine.flock.positions
        active = engine.flock.active
        active_pos = pos[active]

        # Boids should be within domain bounds (wrapped toroidally)
        assert np.all(active_pos[:, 0] >= 0.0), (
            "Boid x positions must be >= 0 (domain boundary)"
        )
        assert np.all(active_pos[:, 0] <= 2000.0), (
            f"Boid x positions must be <= 2000 (domain width), "
            f"max={active_pos[:, 0].max():.1f}. "
            f"If boids are limited to ~400, engine may be using capture_width."
        )
        assert np.all(active_pos[:, 1] >= 0.0)
        assert np.all(active_pos[:, 1] <= 1400.0)
        assert np.all(active_pos[:, 2] >= 0.0)
        assert np.all(active_pos[:, 2] <= 800.0)

        # Crucially: some boids should have traveled beyond capture_width=400
        # proving the engine uses domain.width=2000 for wrapping
        max_x = float(active_pos[:, 0].max())
        assert max_x > 400.0, (
            f"Max x position is {max_x:.1f} ≤ 400 (capture_width). "
            f"Engine may be using capture dimensions for physics."
        )

    def test_flat_vs_subconfig_access_consistency(self):
        """IT5c: Flat access and sub-config access always agree.

        Mutating via sub-config must be visible via flat access, and
        vice versa. This verifies __getattr__/__setattr__ delegation.
        """
        from pymurmur import SimConfig

        cfg = SimConfig()

        # Set via sub-config, read via flat
        cfg.flock.v0 = 5.5
        assert cfg.v0 == 5.5, (
            f"config.flock.v0 = 5.5, but config.v0 = {cfg.v0}"
        )

        # Set via flat, read via sub-config
        cfg.v0 = 9.0
        assert cfg.flock.v0 == 9.0, (
            f"config.v0 = 9.0, but config.flock.v0 = {cfg.flock.v0}"
        )

        # Same for predator fields
        cfg.predator.predator_strength = 3.0
        assert cfg.predator_strength == 3.0

        cfg.predator_strength = 1.5
        assert cfg.predator.predator_strength == 1.5

        # Same for capture fields
        cfg.capture.capture_fps = 25
        assert cfg.capture_fps == 25

        cfg.capture_fps = 10
        assert cfg.capture.capture_fps == 10

        # Direct fields (not delegated)
        cfg.mode = "vicsek"
        assert cfg.mode == "vicsek"


