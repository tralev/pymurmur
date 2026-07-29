"""Unit tests for physics.extensions.extension_registry — EXTENSION_REGISTRY
and the @register_extension decorator, mirroring test_boundary_registry.py's
shape (registry-membership assertions for a modularity-pass registry).

Modularity pass 6: formalises ExtensionManager's 8 pluggable extensions
behind a registry, replacing hardcoded if/elif chains in __init__/pre_step.
These tests verify the registry itself — the extension attributes and
lifecycle behavior are covered separately in test_extensions.py /
test_extensions_lifecycle.py.
"""

from __future__ import annotations

from pymurmur.physics.extensions.boid_state_machine import BoidStateMachine
from pymurmur.physics.extensions.dynamic_vision_range import DynamicVisionRange
from pymurmur.physics.extensions.ecology import Ecology
from pymurmur.physics.extensions.extension_registry import (
    EXTENSION_REGISTRY,
    register_extension,
)
from pymurmur.physics.extensions.neighbor_adaptive_speed import NeighborAdaptiveSpeed
from pymurmur.physics.extensions.predator import Predator
from pymurmur.physics.extensions.ripple import Ripple
from pymurmur.physics.extensions.speed_noise import SpeedNoise
from pymurmur.physics.extensions.wander import Wander

# Expected (cls, config_attr, cleanup_attr) tuples, exactly as declared by
# each extension module's own @register_extension(...) call.
_EXPECTED_ENTRIES = {
    (BoidStateMachine, "boid_state_machine_enabled", "boid_state_speed_mult"),
    (DynamicVisionRange, "dynamic_vision_range_enabled", None),
    (Ecology, "roosting_enabled", None),
    (NeighborAdaptiveSpeed, "neighbor_adaptive_speed_enabled", "neighbor_adaptive_speed_mult"),
    (Predator, "predator_enabled", "predator_priority_accel"),
    (Ripple, "ripple_enabled", None),
    (SpeedNoise, "speed_noise_enabled", "speed_noise_mult"),
    (Wander, "wander_enabled", None),
}


class TestExtensionRegistry:
    def test_exactly_eight_entries(self):
        assert len(EXTENSION_REGISTRY) == 8

    def test_entries_match_expected_set(self):
        assert set(EXTENSION_REGISTRY) == _EXPECTED_ENTRIES

    def test_no_duplicate_config_attrs(self):
        attrs = [attr for _cls, attr, _cleanup in EXTENSION_REGISTRY]
        assert len(attrs) == len(set(attrs)), (
            f"Duplicate config_attr entries: {attrs}"
        )

    def test_ecology_registered_before_predator(self):
        """Ecology.pre_step() ordering relies on Ecology being importable
        (and thus registered) before Predator — confirmed by list position,
        not just import success."""
        attrs = [attr for _cls, attr, _cleanup in EXTENSION_REGISTRY]
        assert attrs.index("roosting_enabled") < attrs.index("predator_enabled")

    def test_register_extension_appends_to_registry(self):
        """Decorator side effect: applying @register_extension to a new
        class appends a new tuple without disturbing existing entries."""
        before = list(EXTENSION_REGISTRY)

        class _DummyExtension:
            pass

        register_extension("dummy_enabled", None)(_DummyExtension)
        try:
            assert EXTENSION_REGISTRY[-1] == (_DummyExtension, "dummy_enabled", None)
            assert EXTENSION_REGISTRY[:-1] == before
        finally:
            EXTENSION_REGISTRY.pop()  # don't leak test state into other tests
