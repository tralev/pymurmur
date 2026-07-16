"""Force dispatch and shared primitives.

Level 1 — composes Level 0 primitives into 5 pluggable force modes.

P2.2: Dispatch uses MODE_REGISTRY (populated by @register decorators)
instead of a hardcoded _DISPATCH dict.  Each mode module registers its
ForceMode subclass at import time.

All mode classes share the same compute() signature:
  (positions, velocities, accelerations, active, index, rng, last_theta, config) -> None
"""
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from ._mode import MODE_REGISTRY, ForceFn, ForceMode, register  # noqa: F401 — public API

from ._base import (
    alignment_force,
    cohesion_force,
    noise_force,
    separation_force,
    ForceTerm,
    composeForces,
)

# Import mode modules to trigger @register decorators (populates MODE_REGISTRY)
from . import projection  # noqa: F401
from . import spatial     # noqa: F401
from . import field       # noqa: F401
from . import vicsek      # noqa: F401
from . import influencer  # noqa: F401

# Backward-compatible exports — tests import these names directly
from .projection import projection_forces
from .spatial import spatial_forces
from .field import field_forces
from .vicsek import vicsek_forces
from .influencer import influencer_forces

if TYPE_CHECKING:
    from ..flock import PhysicsFlock
    from ...core.config import SimConfig
    from ...core.types import SpatialIndex


def compute_all_forces(flock: PhysicsFlock, config: SimConfig) -> None:
    """Route to the active force mode based on config.mode.

    Unpacks flock fields into array-based args so each force mode
    is testable without constructing a PhysicsFlock (I3.7).

    P2.2: Dispatches via MODE_REGISTRY populated by @register decorators.
    """
    mode_cls = MODE_REGISTRY.get(config.mode)
    if mode_cls is None:
        raise ValueError(
            f"Unknown force mode: '{config.mode}'. "
            f"Valid modes: {list(MODE_REGISTRY.keys())}"
        )
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
