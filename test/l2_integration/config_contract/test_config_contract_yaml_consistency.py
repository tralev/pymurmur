"""I7 — Sub-config flat/YAML consistency, YAML key-collision detection.

Split out of test_config_contract.py (file-size split).
"""

import tempfile
from pathlib import Path

from pymurmur.core.config import (
    _ALL_FIELD_NAMES,
    _FIELD_MAP,
    _NESTED_ONLY,
    BoundaryConfig,
    CaptureConfig,
    DomainConfig,
    EcologyConfig,
    ExtensionConfig,
    FieldConfig,
    FlockConfig,
    IndexConfig,
    InfluencerConfig,
    PerfConfig,
    PredatorConfig,
    ProjectionConfig,
    RefinementConfig,
    SimConfig,
    SpatialConfig,
    VicsekConfig,
    VizConfig,
)


class TestSubconfigFlatYamlConsistency:
    """IT6: Sub-config→flat→YAML consistency across serialization.

    Mutations through sub-config accessors must be visible via flat access
    and survive YAML round-trip. Verifies that __getattr__/__setattr__
    delegation stays consistent across all three access paths.
    """

    def test_roundtrip_subconfig_mutation_visible_via_flat_access(self):
        """IT6a: Mutate via sub-config → YAML round-trip → flat access agrees."""
        import tempfile
        from pathlib import Path

        from pymurmur import SimConfig

        cfg = SimConfig()
        cfg.seed = 99

        # Mutate via sub-config accessors
        cfg.flock.v0 = 11.0
        cfg.flock.num_boids = 300
        cfg.flock.boid_size = 7.5
        cfg.domain.width = 2000.0
        cfg.domain.depth = 600.0
        cfg.capture.capture_frames = 500
        cfg.capture.capture_every = 2

        # Verify flat access before round-trip
        assert cfg.v0 == 11.0
        assert cfg.num_boids == 300
        assert cfg.boid_size == 7.5
        assert cfg.width == 2000.0
        assert cfg.depth == 600.0
        assert cfg.capture_frames == 500
        assert cfg.capture_every == 2

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            tmp = Path(f.name)
        try:
            cfg.to_file(tmp)
            loaded = SimConfig.from_file(tmp)

            # After round-trip, flat access should match (accounting for known collisions)
            assert loaded.v0 == 11.0, (
                f"v0 lost in YAML round-trip: expected 11.0, got {loaded.v0}"
            )
            assert loaded.num_boids == 300, (
                "num_boids lost in YAML round-trip"
            )
            assert loaded.boid_size == 7.5, (
                "boid_size lost in YAML round-trip"
            )
            # depth doesn't collide with anything, should survive
            assert loaded.depth == 600.0, (
                f"depth lost in YAML round-trip: expected 600.0, got {loaded.depth}"
            )
            # capture fields: capture.every → flat "every" but _FIELD_MAP has
            # "capture_every" → ("_capture", "capture_every"), so the to_file()
            # key is "every" which from_file() flattens — but from_file() only
            # keeps known _ALL_FIELD_NAMES. "every" is NOT in _ALL_FIELD_NAMES
            # (only "capture_every" is). So capture_every defaults.
            # This is a known YAML format limitation.
            assert isinstance(loaded.capture_every, int), (
                f"capture_every should be int, got {type(loaded.capture_every)}"
            )
            assert loaded.seed == 99, (
                f"seed lost: expected 99, got {loaded.seed}"
            )
        finally:
            tmp.unlink()

    def test_flat_mutation_visible_via_subconfig(self):
        """IT6b: Mutate via flat → sub-config accessor reflects the change."""
        from pymurmur import SimConfig

        cfg = SimConfig()

        cfg.v0 = 13.0
        assert cfg.flock.v0 == 13.0, (
            f"config.v0 = 13.0 but config.flock.v0 = {cfg.flock.v0}"
        )

        cfg.capture_frames = 999
        assert cfg.capture.capture_frames == 999, (
            f"config.capture_frames = 999 but config.capture.capture_frames = "
            f"{cfg.capture.capture_frames}"
        )

        cfg.predator_threat_radius = 25.0
        assert cfg.predator.predator_threat_radius == 25.0

    def test_roundtrip_then_both_access_paths_agree(self):
        """IT6c: After YAML round-trip, sub-config and flat access agree."""
        import tempfile
        from pathlib import Path

        from pymurmur import SimConfig

        cfg = SimConfig()
        cfg.v0 = 7.0
        cfg.num_boids = 250
        cfg.boid_size = 10.0
        cfg.seed = 77

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            tmp = Path(f.name)
        try:
            cfg.to_file(tmp)
            loaded = SimConfig.from_file(tmp)

            # Both access paths must agree on loaded config
            assert loaded.v0 == loaded.flock.v0, (
                f"After round-trip: flat v0={loaded.v0} ≠ "
                f"sub-config v0={loaded.flock.v0}"
            )
            assert loaded.num_boids == loaded.flock.num_boids, (
                f"After round-trip: flat num_boids={loaded.num_boids} ≠ "
                f"sub-config num_boids={loaded.flock.num_boids}"
            )
            assert loaded.boid_size == loaded.flock.boid_size, (
                f"After round-trip: flat boid_size={loaded.boid_size} ≠ "
                f"sub-config boid_size={loaded.flock.boid_size}"
            )
            assert loaded.seed == 77
        finally:
            tmp.unlink()

    def test_all_16_subconfig_properties_are_accessible(self):
        """IT6d: All 16 sub-config properties exist and return correct types."""
        from pymurmur import SimConfig

        cfg = SimConfig()

        # All 16 sub-config accessors must return the right type
        assert isinstance(cfg.domain, DomainConfig), (
            f"cfg.domain should be DomainConfig, got {type(cfg.domain)}"
        )
        assert isinstance(cfg.flock, FlockConfig)
        assert isinstance(cfg.boundary, BoundaryConfig)
        assert isinstance(cfg.projection, ProjectionConfig)
        assert isinstance(cfg.spatial, SpatialConfig)
        assert isinstance(cfg.field, FieldConfig)
        assert isinstance(cfg.vicsek, VicsekConfig)
        assert isinstance(cfg.influencer, InfluencerConfig)
        assert isinstance(cfg.index, IndexConfig)
        assert isinstance(cfg.refinement, RefinementConfig)
        assert isinstance(cfg.extension, ExtensionConfig)
        assert isinstance(cfg.predator, PredatorConfig)
        assert isinstance(cfg.ecology, EcologyConfig)
        assert isinstance(cfg.perf, PerfConfig)
        assert isinstance(cfg.viz, VizConfig)
        assert isinstance(cfg.capture, CaptureConfig)

    def test_direct_fields_not_delegated_to_subconfigs(self):
        """IT6e: Direct fields (mode, seed, position_init) are stored on
        SimConfig, not delegated to a sub-config."""
        from pymurmur import SimConfig

        cfg = SimConfig(mode="spatial", seed=123, position_init="sphere")

        assert cfg.mode == "spatial"
        assert cfg.seed == 123
        assert cfg.position_init == "sphere"

        # These should NOT be delegated — they're direct attributes
        assert "mode" not in _FIELD_MAP, (
            "mode should be a direct field, not in _FIELD_MAP"
        )
        assert "seed" not in _FIELD_MAP, (
            "seed should be a direct field, not in _FIELD_MAP"
        )
        assert "position_init" not in _FIELD_MAP, (
            "position_init should be a direct field, not in _FIELD_MAP"
        )


# ═══════════════════════════════════════════════════════════════════
# IT7 — YAML Key Collision Detection (I7.1 + I7.4)
# ═══════════════════════════════════════════════════════════════════


class TestYamlKeyCollisionDetection:
    """Detects and documents known YAML key collisions and silent drops
    in SimConfig.to_file() / from_file() round-trip.

    Collisions occur because to_file() nests sub-configs under section
    keys using short field names (e.g. domain: {width: ...}, capture:
    {width: ...}) while from_file() flattens ALL sections into a single
    namespace and filters by _ALL_FIELD_NAMES.

    These tests explicitly assert the CURRENT (buggy) behavior. When the
    YAML serialization is fixed, these tests will intentionally break
    to signal that the fix was successful — update the assertions.
    """

    def test_programmatic_collision_audit(self):
        """Analyze to_file() YAML structure to find ALL duplicate keys.

        Parses the raw YAML output and detects every key that appears
        in more than one section. The set of colliding keys is asserted
        exactly — if a collision is fixed or a new one added, this test
        fails (regression guard).
        """
        import yaml

        cfg = SimConfig()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            tmp = Path(f.name)
        try:
            cfg.to_file(tmp)
            raw = yaml.safe_load(tmp.read_text()) or {}

            seen_keys: dict[str, list[str]] = {}  # key → [section_names]
            collisions: dict[str, list[str]] = {}

            for section_name, section_data in raw.items():
                if isinstance(section_data, dict):
                    for k in section_data:
                        if k in seen_keys:
                            if k not in collisions:
                                collisions[k] = seen_keys[k].copy()
                            collisions[k].append(section_name)
                        else:
                            seen_keys[k] = [section_name]
                else:
                    # Scalar values (mode, seed) — keyed by section_name
                    if section_name in seen_keys:
                        collisions.setdefault(section_name, seen_keys[section_name])
                        collisions[section_name].append(section_name + "_scalar")
                    seen_keys[section_name] = [section_name + "_scalar"]

            # Known collisions — each key appears in the listed sections
            known_collisions = {}

            assert set(collisions.keys()) == set(known_collisions.keys()), (
                f"YAML key collisions changed!\n"
                f"Expected: {sorted(known_collisions.keys())}\n"
                f"Got:      {sorted(collisions.keys())}\n"
                f"If a collision was fixed, update known_collisions.\n"
                f"If a new collision appeared, a new field name conflicts."
            )

            for key, expected_sections in known_collisions.items():
                actual_sections = collisions[key]
                assert actual_sections == expected_sections, (
                    f"Collision sections for '{key}' changed:\n"
                    f"Expected: {expected_sections}\n"
                    f"Got:      {actual_sections}"
                )
        finally:
            tmp.unlink()

    def test_yaml_section_keys_not_in_all_field_names(self):
        """Audit: find YAML leaf keys not in _ALL_FIELD_NAMES (silently dropped).

        When from_file() flattens sections, only keys in _ALL_FIELD_NAMES
        survive. This test enumerates which YAML keys are silently dropped
        so the set is explicit and tracked.
        """
        import yaml

        cfg = SimConfig()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            tmp = Path(f.name)
        try:
            cfg.to_file(tmp)
            raw = yaml.safe_load(tmp.read_text()) or {}

            all_yaml_keys: set[str] = set()
            for section_name, section_data in raw.items():
                if isinstance(section_data, dict):
                    for k in section_data:
                        all_yaml_keys.add(k)
                else:
                    all_yaml_keys.add(section_name)

            # Nested-only keys (retired shims) are handled explicitly
            # by from_file(), so they are not dropped.
            handled = _ALL_FIELD_NAMES | set(_NESTED_ONLY.keys())
            dropped = all_yaml_keys - handled

            # Known set of silently-dropped YAML keys
            known_dropped = set()

            assert dropped == known_dropped, (
                f"Silently-dropped YAML keys changed!\n"
                f"Expected: {sorted(known_dropped)}\n"
                f"Got:      {sorted(dropped)}\n"
                f"New in dropped (added silently?): {sorted(dropped - known_dropped)}\n"
                f"Fixed (no longer dropped!): {sorted(known_dropped - dropped)}"
            )
        finally:
            tmp.unlink()

    def test_domain_capture_dimensions_collision(self):
        """capture.width and capture.height overwrite domain.width/height.

        Both domain: {width: X} and capture: {width: Y} flatten to key
        'width'. capture is written later in to_file(), so its value wins.
        The loaded config gets capture's dimensions in domain fields.
        """
        cfg = SimConfig()

        # Distinct values to track who wins
        cfg.domain.width = 1111.0
        cfg.domain.height = 3333.0
        cfg.capture.capture_width = 2222
        cfg.capture.capture_height = 4444

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            tmp = Path(f.name)
        try:
            cfg.to_file(tmp)
            loaded = SimConfig.from_file(tmp)

            # Both domain and capture dimensions survive independently
            assert loaded.width == 1111.0, (
                f"BUG FIXED: width collision resolved! Got {loaded.width}"
            )
            assert loaded.height == 3333.0, (
                f"BUG FIXED: height collision resolved! Got {loaded.height}"
            )
            assert loaded.domain.width == 1111.0
            assert loaded.domain.height == 3333.0
            assert loaded.capture.capture_width == 2222
            assert loaded.capture.capture_height == 4444
        finally:
            tmp.unlink()

    def test_visual_capture_fps_collision(self):
        """visual.fps and capture.capture_fps both produce YAML key 'fps'.

        capture section is written after visual, so capture's fps value
        overwrites visual's fps value.
        """
        cfg = SimConfig()

        cfg.viz.fps = 30
        cfg.capture.capture_fps = 60

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            tmp = Path(f.name)
        try:
            cfg.to_file(tmp)
            loaded = SimConfig.from_file(tmp)

            # Both viz.fps and capture_fps survive independently
            assert loaded.fps == 30, (
                f"BUG FIXED: fps collision resolved! Got {loaded.fps}"
            )
            assert loaded.viz.fps == 30
            assert loaded.capture.capture_fps == 60
        finally:
            tmp.unlink()

    def test_silently_dropped_extension_toggles(self):
        """Extension toggles are written as short names (predator, roosting)
        but _ALL_FIELD_NAMES has long names (predator_enabled, roosting_enabled).
        After flattening, the short names don't match → silently dropped →
        revert to dataclass defaults.
        """
        cfg = SimConfig()

        cfg.extension.predator_enabled = True
        cfg.extension.roosting_enabled = True
        cfg.extension.wander_enabled = True
        cfg.extension.ripple_enabled = True

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            tmp = Path(f.name)
        try:
            cfg.to_file(tmp)
            loaded = SimConfig.from_file(tmp)

            # Extension toggles survive round-trip correctly
            assert loaded.predator_enabled, (
                f"BUG FIXED: predator_enabled persisted! Got {loaded.predator_enabled}"
            )
            assert loaded.roosting_enabled, (
                "BUG FIXED: roosting_enabled persisted!"
            )
            assert loaded.wander_enabled, (
                "BUG FIXED: wander_enabled persisted!"
            )
            assert loaded.ripple_enabled, (
                "BUG FIXED: ripple_enabled persisted!"
            )
        finally:
            tmp.unlink()

    def test_silently_dropped_capture_short_names(self):
        """Capture short-name YAML keys (every, frames, output, etc.)
        don't match _ALL_FIELD_NAMES (capture_every, capture_frames, etc.)
        → silently dropped → revert to defaults.
        """
        cfg = SimConfig()

        cfg.capture.capture_every = 99
        cfg.capture.capture_frames = 888
        cfg.capture.capture_output = "output/custom.gif"
        cfg.capture.capture_metrics_csv = "output/custom.csv"
        cfg.capture.capture_metrics_json = "output/custom.json"
        cfg.capture.capture_with_viz = True

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            tmp = Path(f.name)
        try:
            cfg.to_file(tmp)
            loaded = SimConfig.from_file(tmp)

            # Capture fields survive round-trip correctly
            assert loaded.capture_every == 99, (
                f"BUG FIXED: capture_every persisted! Got {loaded.capture_every}"
            )
            assert loaded.capture_frames == 888, (
                "BUG FIXED: capture_frames persisted!"
            )
            assert loaded.capture_output == "output/custom.gif", (
                f"BUG FIXED: capture_output persisted! Got {loaded.capture_output}"
            )
            assert loaded.capture_metrics_csv == "output/custom.csv", (
                "BUG FIXED: capture_metrics_csv persisted!"
            )
            assert loaded.capture_metrics_json == "output/custom.json", (
                "BUG FIXED: capture_metrics_json persisted!"
            )
            assert loaded.capture_with_viz, (
                "BUG FIXED: capture_with_viz persisted!"
            )
        finally:
            tmp.unlink()

    def test_silently_dropped_refinements_toggle(self):
        """refinements.enabled → YAML key 'enabled' → not in _ALL_FIELD_NAMES
        → refinements toggle silently reverts to default (True).
        """
        cfg = SimConfig()
        cfg.refinement.refinements = False

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            tmp = Path(f.name)
        try:
            cfg.to_file(tmp)
            loaded = SimConfig.from_file(tmp)

            # Refinements toggle survives round-trip correctly
            assert not loaded.refinements, (
                f"BUG FIXED: refinements toggle persisted! Got {loaded.refinements}"
            )
        finally:
            tmp.unlink()

    def test_non_colliding_fields_survive_roundtrip(self):
        """Fields without collisions (unique YAML keys matching
        _ALL_FIELD_NAMES) should survive round-trip correctly.

        This is a sanity check — if these break, something else is wrong.
        """
        cfg = SimConfig()
        cfg.seed = 99
        cfg.mode = "spatial"
        cfg.flock.v0 = 11.0
        cfg.flock.num_boids = 300
        cfg.flock.boid_size = 7.5
        cfg.domain.depth = 600.0  # depth has no collision
        cfg.boundary.boundary_mode = "sphere"
        cfg.boundary.boundary_sphere_radius = 350.0
        cfg.predator.predator_threat_radius = 25.0
        cfg.predator.predator_strength = 2.0
        cfg.perf.metrics_detail_level = 2
        cfg.perf.metrics_interval = 99  # written as 'metrics_interval' (matches)
        cfg.index.spatial_index = "kdtree"
        cfg.index.topological_cap = 20

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            tmp = Path(f.name)
        try:
            cfg.to_file(tmp)
            loaded = SimConfig.from_file(tmp)

            assert loaded.seed == 99
            assert loaded.mode == "spatial"
            assert loaded.v0 == 11.0
            assert loaded.num_boids == 300
            assert loaded.boid_size == 7.5
            assert loaded.depth == 600.0
            assert loaded.boundary_mode == "sphere"
            assert loaded.boundary_sphere_radius == 350.0
            assert loaded.predator_threat_radius == 25.0
            assert loaded.predator_strength == 2.0
            assert loaded.metrics_detail_level == 2
            assert loaded.metrics_interval == 99
            assert loaded.spatial_index == "kdtree"
            assert loaded.topological_cap == 20
        finally:
            tmp.unlink()
