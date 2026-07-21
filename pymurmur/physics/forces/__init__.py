"""Force dispatch and shared primitives.

Level 1 — composes Level 0 primitives into 5 pluggable force modes.

P2.2: Dispatch uses MODE_REGISTRY (populated by @register decorators)
instead of a hardcoded _DISPATCH dict.  Each mode module registers its
ForceMode subclass at import time.

All mode classes share the same compute() signature:
  (positions, velocities, accelerations, active, index, rng, last_theta, config) -> None
"""
from __future__ import annotations

from typing import TYPE_CHECKING

# Import mode modules to trigger @register decorators (populates MODE_REGISTRY)
from . import (
    angle,  # noqa: F401
    field,  # noqa: F401
    influencer,  # noqa: F401
    marl,  # noqa: F401  # P12.1: MARL mode
    projection,  # noqa: F401
    spatial,  # noqa: F401
    vicsek,  # noqa: F401
)
from ._base import (  # noqa: F401  # re-exports
    ForceTerm,
    alignment_force,
    cohesion_force,
    composeForces,
    noise_force,
    separation_force,
)
from ._mode import MODE_REGISTRY, ForceFn, ForceMode, register  # noqa: F401 — public API
from .angle import angle_forces  # noqa: F401  # re-export
from .field import field_forces  # noqa: F401  # re-export
from .influencer import influencer_forces  # noqa: F401  # re-export
from .marl import marl_forces  # noqa: F401  # P12.1 re-export

# Backward-compatible exports — tests import these names directly
from .projection import projection_forces  # noqa: F401  # re-export
from .spatial import spatial_forces  # noqa: F401  # re-export
from .vicsek import vicsek_forces  # noqa: F401  # re-export

if TYPE_CHECKING:
    from ...core.config import SimConfig
    from ..flock import PhysicsFlock


def compute_all_forces(flock: PhysicsFlock, config: SimConfig) -> None:
    """Route to the active force mode based on config.mode.

    Unpacks flock fields into array-based args so each force mode
    is testable without constructing a PhysicsFlock (I3.7).

    P2.2: Dispatches via MODE_REGISTRY populated by @register decorators.
    P4.3: Wires flock.is_predator into config so force modes can do
          species-based predator-prey dynamics.
    """
    mode_cls = MODE_REGISTRY.get(config.mode)
    if mode_cls is None:
        raise ValueError(
            f"Unknown force mode: '{config.mode}'. "
            f"Valid modes: {list(MODE_REGISTRY.keys())}"
        )
    # P4.3: Expose species column for predator-prey force dynamics
    object.__setattr__(config, '_is_predator', flock.is_predator)
    # P4.8: Bridge flock.coherence_factor → config (transient —
    # set by ecology extension on the flock, consumed by spatial mode).
    # Not persisted to YAML; this is a private runtime bridge only.
    object.__setattr__(config, '_coherence_factor', flock.coherence_factor)
    mode_cls.compute(
        flock.positions,
        flock.velocities,
        flock.accelerations,
        flock.active,
        flock.get_index(),
        flock.rng,
        flock.last_theta,
        config,
    )


def mode_needs_index(mode: str) -> bool:
    """Return True if the named force mode requires a spatial index.

    P2.2: Reads from the ForceMode class's `needs_index` attribute
    instead of a hardcoded boolean on each module-level function.
    """
    cls = MODE_REGISTRY.get(mode)
    return cls.needs_index if cls is not None else False
