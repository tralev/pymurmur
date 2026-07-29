"""IT7 — YAML Key Collision Detection (I7.1 + I7.4).

Split out of test_config_contract_yaml_consistency.py (file-size split) —
IT6 sub-config flat/YAML consistency tests stay in the original; this
file covers YAML key-collision detection and silent-drop auditing.
"""

import tempfile
from pathlib import Path

from pymurmur.core.config import _ALL_FIELD_NAMES, _NESTED_ONLY, SimConfig


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
