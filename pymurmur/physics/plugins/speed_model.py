"""SpeedModel ABC, SPEED_MODEL_REGISTRY, and @register decorator.

Modularity pass 4: extracts the 4-tier if/elif speed-enforcement chain
from boid.py::integrate() (band/clamp, fixed, ceiling, none) behind a
registry, mirroring physics/forces/_mode.py's ForceMode/MODE_REGISTRY
pattern and physics/boundary/_mode.py's BoundaryMode pattern.

Comparison.md's Speed Models taxonomy identifies 6 strategies across the
surveyed implementations; this codebase implements 4 of them. This pass
registers those 4 and stops there — noise_modulated (§05, §20) and
velocity_adaptive (§11) are genuinely new feature work, not extraction.

Each registered strategy returns the raw_vel array (or None) so the
caller (boid.py::integrate) can still apply inertia blending, which
needs the pre-clamp velocity independent of the chosen speed model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass  # numpy is already imported at module level

SPEED_MODEL_REGISTRY: dict[str, type["SpeedModel"]] = {}


class SpeedModel(ABC):
    """Protocol for speed-enforcement strategies.

    Each subclass overrides apply() with the shared speed-clamp signature.
    Returns the raw (pre-clamp) velocity copy if the caller needs it for
    inertia blending, or None if not needed.

    Usage::

        @register("band")
        class BandSpeedModel(SpeedModel):
            @staticmethod
            def apply(velocities, active, caps, min_speed, speeds):
                ...  # clamp to [min_speed, caps]
    """

    @staticmethod
    @abstractmethod
    def apply(
        velocities: "np.ndarray",
        active: "np.ndarray",
        caps: "np.ndarray",
        min_speed: "np.ndarray",
        speeds: "np.ndarray",
    ) -> "np.ndarray | None":
        """Enforce speed policy on active birds, mutating velocities in place.

        Args:
            velocities: (N, 3) float32 — mutated in place
            active: (N,) bool
            caps: (N,) float32 — per-bird maximum speed
            min_speed: (N,) float32 — per-bird minimum speed
            speeds: (N, 1) float32 — precomputed ||velocities|| per row

        Returns:
            Copy of pre-clamp velocities (for inertia) or None.
        """
        ...


def register(name: str):
    """Decorator to register a SpeedModel subclass in SPEED_MODEL_REGISTRY.

    Usage::

        @register("band")
        class BandSpeedModel(SpeedModel):
            ...
    """

    def decorator(cls: type[SpeedModel]) -> type[SpeedModel]:
        SPEED_MODEL_REGISTRY[name] = cls
        return cls

    return decorator


# ── Registered strategies ─────────────────────────────────────────

@register("band")
@register("clamp")  # "clamp" aliases "band" (config vocabulary compat)
class BandSpeedModel(SpeedModel):
    """Clamp to [min_speed, caps] — the default band strategy.

    Too-fast birds are scaled down to their cap; too-slow birds are
    boosted to their floor.  This is the standard variable-speed model
    used by 14 of 22 surveyed implementations.
    """

    @staticmethod
    def apply(velocities, active, caps, min_speed, speeds):
        too_fast = (speeds.ravel() > caps).ravel() & active
        too_slow = (speeds.ravel() < min_speed).ravel() & active
        if too_fast.any():
            velocities[too_fast] = (
                velocities[too_fast] / speeds[too_fast]
            ) * caps[too_fast, np.newaxis]
        if too_slow.any():
            velocities[too_slow] = (
                velocities[too_slow] / (speeds[too_slow] + 1e-10)
            ) * min_speed[too_slow, np.newaxis]
        return None  # caller must snapshot raw_vel before calling if inertia > 0


@register("fixed")
class FixedSpeedModel(SpeedModel):
    """Exact renormalisation to caps — constant-speed enforcement.

    Zero-velocity birds get deterministic direction (1, 0, 0) to avoid
    NaN.  Used by VicsekMode and InfluencerMode (speed_mode = "fixed").

    Matches §04, §06, §09, §10a, §10b from the comparison taxonomy.
    """

    @staticmethod
    def apply(velocities, active, caps, min_speed, speeds):
        safe_speeds = speeds + 1e-10
        dirs = velocities / safe_speeds
        zero_mask = (speeds.ravel() < 1e-6) & active
        if zero_mask.any():
            dirs[zero_mask.ravel(), 0] = 1.0
            dirs[zero_mask.ravel(), 1] = 0.0
            dirs[zero_mask.ravel(), 2] = 0.0
        velocities[active] = dirs[active] * caps[active, np.newaxis]
        return None  # fixed speed has no meaningful raw_vel for inertia


@register("ceiling")
class CeilingSpeedModel(SpeedModel):
    """Cap only — speeds above caps are scaled down; slow speeds pass through.

    No lower bound — boids can drift to arbitrarily slow speeds.
    Matches §02's direct-truncation pattern in the comparison taxonomy.
    """

    @staticmethod
    def apply(velocities, active, caps, min_speed, speeds):
        too_fast = (speeds.ravel() > caps).ravel() & active
        if too_fast.any():
            velocities[too_fast] = (
                velocities[too_fast] / speeds[too_fast]
            ) * caps[too_fast, np.newaxis]
        return None


@register("none")
class NoneSpeedModel(SpeedModel):
    """No speed enforcement — velocities pass through unchanged.

    Used by MarlMode (speed_mode = "none"), where the RL policy owns
    velocity control end-to-end.
    """

    @staticmethod
    def apply(velocities, active, caps, min_speed, speeds):
        return None
