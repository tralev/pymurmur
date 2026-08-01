"""ExtensionManager — assembles enabled extensions and applies them in pre_step().

Modularity pass 6: uses EXTENSION_REGISTRY for pluggable discovery instead
of hardcoded per-extension if/elif chains. Adding a new extension now only
requires a new file in extensions/ with @register_extension — no editing
of ExtensionManager needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._base import StepContext  # noqa: F401  # used in type hints with annotations future
from .boid_state_machine import BoidStateMachine  # noqa: F401  # registers via @register_extension
from .dynamic_vision_range import (
    DynamicVisionRange,  # noqa: F401  # registers via @register_extension
)
from .ecology import Ecology
from .extension_registry import EXTENSION_REGISTRY
from .neighbor_adaptive_speed import (
    NeighborAdaptiveSpeed,  # noqa: F401  # registers via @register_extension
)
from .predator import Predator
from .ripple import Ripple  # noqa: F401  # registers via @register_extension
from .speed_noise import SpeedNoise  # noqa: F401  # registers via @register_extension
from .wander import Wander  # noqa: F401  # registers via @register_extension

if TYPE_CHECKING:
    from ...core.config import SimConfig
    from ..flock import PhysicsFlock


# ── Extension instance factory ────────────────────────────────────
# Each registered entry maps (cls, config_attr, cleanup_attr) to a
# class reference.  Some extensions need config in their constructor
# (Ecology, Predator), others don't (Wander, Ripple, SpeedNoise, etc.)
# — the factory handles this transparently.

_EXTENSION_CTOR_NEEDS_CONFIG: set[type[Any]] = {Ecology, Predator}


def _make_extension(
    cls: type[Any], config: SimConfig
) -> Any | None:
    """Instantiate an extension if its class needs config, else return None
    to signal \"instantiate with no-arg constructor later\"."""
    if cls in _EXTENSION_CTOR_NEEDS_CONFIG:
        return cls(config)
    return cls()


class ExtensionManager:
    """Manages pluggable behavioural extensions.

    Instantiate once; call pre_step() before each simulation step.
    Extensions are lazily created on first enable and dropped on disable
    — T/K toggles take effect immediately without a reset (I5.3).

    Ecology runs first (advances day, sets predator presence), then
    Predator is conditionally applied based on predator_active.
    Extensions are ordered by registration order — Ecology is registered
    first (in ecology.py) so it always runs before Predator.
    """

    def __init__(self, config: SimConfig) -> None:
        # Build initial active set from whatever's enabled in config
        self._active: dict[str, Any | None] = {}
        for _cls, _attr, _cleanup in EXTENSION_REGISTRY:
            enabled = getattr(config, _attr, False)
            self._active[_attr] = (
                _make_extension(_cls, config) if enabled else None
            )

    def pre_step(self, flock: PhysicsFlock, ctx: StepContext) -> None:
        """Run all enabled extensions before force computation.

        Ecology runs first to advance the day and update predator_active.
        Predator only runs when predator is present (or ecology not enabled).
        All other extensions run after.
        """
        cfg = ctx.config

        # ── Lazy lifecycle: check config flags each frame (I5.3) ──
        for cls, attr, cleanup_attr in EXTENSION_REGISTRY:
            enabled = getattr(cfg, attr, False)
            instance = self._active.get(attr)

            if enabled and instance is None:
                instance = _make_extension(cls, cfg)
                self._active[attr] = instance
            elif not enabled and instance is not None:
                self._active[attr] = None
                # Handle flock-level cleanup on teardown
                if cleanup_attr is not None:
                    setattr(flock, cleanup_attr, None)
                # Special-case cleanup for extensions that reset config
                # attributes rather than flock attributes
                if attr == "predator_enabled":
                    flock.predator_priority_accel = None
                elif attr == "speed_noise_enabled":
                    flock.speed_noise_mult = None
                elif attr == "neighbor_adaptive_speed_enabled":
                    flock.neighbor_adaptive_speed_mult = None
                elif attr == "dynamic_vision_range_enabled":
                    cfg._dynamic_visual_range_mult = 1.0
                elif attr == "boid_state_machine_enabled":
                    flock.boid_state_speed_mult = None
                    flock.boid_state[:] = 0

        # ── Ecology runs first ──
        eco = self._active.get("roosting_enabled")
        pred = self._active.get("predator_enabled")

        if eco is not None:
            eco.apply(flock, ctx)
            flock.coherence_factor = eco.coherence_factor
        else:
            flock.coherence_factor = 1.0

        # ── Predator runs conditionally ──
        if pred is not None:
            if eco is None or eco.predator_active:
                pred.apply(flock, ctx)

        # ── Remaining extensions (all except ecology + predator, and
        #     only those with a non-None active instance) ──
        _SKIP = frozenset({"roosting_enabled", "predator_enabled"})
        for _cls, attr, _cleanup_attr in EXTENSION_REGISTRY:
            if attr in _SKIP:
                continue
            ext = self._active.get(attr)
            if ext is not None:
                ext.apply(flock, ctx)

    @property
    def count(self) -> int:
        return sum(1 for v in self._active.values() if v is not None)

    # ── Backward-compatible named accessors ────────────────────────
    # Pre-registry ExtensionManager exposed each extension as a plain
    # instance attribute (self._predator, self._ecology, etc.) — several
    # tests and one integration path read these directly. Read-only
    # properties over self._active preserve that surface exactly
    # (nothing external ever assigns to these, only reads).

    @property
    def _predator(self):
        return self._active.get("predator_enabled")

    @property
    def _ecology(self):
        return self._active.get("roosting_enabled")

    @property
    def _wander(self):
        return self._active.get("wander_enabled")

    @property
    def _ripple(self):
        return self._active.get("ripple_enabled")

    @property
    def _speed_noise(self):
        return self._active.get("speed_noise_enabled")

    @property
    def _neighbor_adaptive_speed(self):
        return self._active.get("neighbor_adaptive_speed_enabled")

    @property
    def _dynamic_vision_range(self):
        return self._active.get("dynamic_vision_range_enabled")

    @property
    def _boid_state_machine(self):
        return self._active.get("boid_state_machine_enabled")

    @property
    def predator_position(self):
        """D7/S2.A8: threat marker position (np.ndarray) for rendering,
        or None when no predator extension is active."""
        pred = self._active.get("predator_enabled")
        return pred.position if pred is not None else None
