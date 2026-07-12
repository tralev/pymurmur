"""Force dispatch and shared primitives.

Level 1 — composes Level 0 primitives into 5 pluggable force modes.
All mode functions share the same signature: (flock, config) -> None.
"""
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from ._base import (
    alignment_force,
    cohesion_force,
    noise_force,
    separation_force,
)
from .spatial import spatial_forces
from .projection import projection_forces
from .field import field_forces
from .vicsek import vicsek_forces
from .influencer import influencer_forces

if TYPE_CHECKING:
    from ..flock import PhysicsFlock
    from ...core.config import SimConfig

_DISPATCH = {
    "projection": projection_forces,
    "spatial": spatial_forces,
    "field": field_forces,
    "vicsek": vicsek_forces,
    "influencer": influencer_forces,
}


def compute_all_forces(flock: PhysicsFlock, config: SimConfig) -> None:
    """Route to the active force mode based on config.mode."""
    mode_fn = _DISPATCH.get(config.mode)
    if mode_fn is None:
        raise ValueError(
            f"Unknown force mode: '{config.mode}'. "
            f"Valid modes: {list(_DISPATCH.keys())}"
        )
    mode_fn(flock, config)
