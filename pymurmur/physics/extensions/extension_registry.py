"""EXTENSION_REGISTRY — pluggable extension discovery.

Modularity pass 6: replaces the hardcoded if/elif chains in
ExtensionManager.__init__() and pre_step() with a registry so adding
a new extension no longer requires modifying ExtensionManager at all.

Each extension registers a metadata tuple:
  (cls, config_flag_attr, needs_flock_cleanup_attr)

- cls: the Extension subclass
- config_flag_attr: e.g. "predator_enabled" — the SimConfig boolean
- needs_flock_cleanup_attr: e.g. "predator_priority_accel" — the
  flock attribute to reset on teardown, or None

Priority is implicit in registration order (first-registered runs
first). Ecology is registered first so it always runs before
Predator, which preserves the existing ordering contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...core.config import SimConfig
    from ...physics.extensions._base import Extension

# Each entry: (ExtensionClass, config_attr_name, flock_cleanup_attr_name_or_None)
EXTENSION_REGISTRY: list[tuple[type[Any], str, str | None]] = []


def register_extension(
    config_attr: str,
    flock_cleanup_attr: str | None = None,
) -> Any:
    """Decorator to register an Extension subclass.

    Usage::

        @register_extension("predator_enabled", "predator_priority_accel")
        class Predator(Extension):
            ...

    Args:
        config_attr: name of the boolean on SimConfig, e.g. "predator_enabled"
        flock_cleanup_attr: name of the flock attribute to reset on
            teardown, or None if no cleanup needed
    """

    def decorator(cls: type):
        EXTENSION_REGISTRY.append((cls, config_attr, flock_cleanup_attr))
        return cls

    return decorator
