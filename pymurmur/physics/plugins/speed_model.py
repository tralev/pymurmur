"""SpeedModel ABC, SPEED_MODEL_REGISTRY, and @register decorator.

Modularity pass 4: extracts the 4-tier if/elif speed-enforcement chain
from boid.py::integrate() (band/clamp, fixed, ceiling, none) behind a
registry, mirroring physics/forces/_mode.py's ForceMode/MODE_REGISTRY
pattern and physics/boundary/_mode.py's BoundaryMode pattern.

Comparison.md's Speed Models taxonomy identifies 6 strategies across the
surveyed implementations; pass 4 registered 4 of them. This pass adds
the remaining two — noise_modulated (§05, §20) and velocity_adaptive
(§11) — completing the taxonomy. Both are genuinely new feature work,
not pure extraction: neither fits the original 5-arg apply() signature
(noise_modulated needs each bird's position to sample a noise field;
velocity_adaptive needs an RNG for its randomized bonus and dt for its
smooth-approach lerp), so apply() gained three new keyword params
(positions, rng, dt) — all optional, defaulting to None/0.0, so the
four pre-existing strategies are unaffected and don't need to reference
them.

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
            def apply(velocities, active, caps, min_speed, speeds,
                      positions=None, rng=None, dt=0.0):
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
        positions: "np.ndarray | None" = None,
        rng: "np.random.Generator | None" = None,
        dt: float = 0.0,
    ) -> "np.ndarray | None":
        """Enforce speed policy on active birds, mutating velocities in place.

        Args:
            velocities: (N, 3) float32 — mutated in place
            active: (N,) bool
            caps: (N,) float32 — per-bird maximum speed
            min_speed: (N,) float32 — per-bird minimum speed
            speeds: (N, 1) float32 — precomputed ||velocities|| per row
            positions: (N, 3) float32 or None — only consulted by
                position-dependent strategies (noise_modulated)
            rng: seeded generator or None — only consulted by strategies
                with a stochastic element (velocity_adaptive)
            dt: seconds this frame — only consulted by strategies with a
                time-dependent smoothing rate (velocity_adaptive)

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
    def apply(velocities, active, caps, min_speed, speeds,
              positions=None, rng=None, dt=0.0):
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
    def apply(velocities, active, caps, min_speed, speeds,
              positions=None, rng=None, dt=0.0):
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
    def apply(velocities, active, caps, min_speed, speeds,
              positions=None, rng=None, dt=0.0):
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
    def apply(velocities, active, caps, min_speed, speeds,
              positions=None, rng=None, dt=0.0):
        return None


@register("noise_modulated")
class NoiseModulatedSpeedModel(SpeedModel):
    """Speed varies continuously with a deterministic 3D noise field
    sampled at each bird's position — organic slow/fast zones with no
    behavioural logic. Matches §05 (3D Simplex)/§20 (3D value noise)
    from the comparison taxonomy; not literal simplex noise (no such
    dependency anywhere in this codebase), but the same "value noise"
    family §20 already is — built from the same deterministic sinusoidal
    field construction _base.py's curl_flow uses for its own pseudo-noise.

    speed_cap = caps * lerp(0.5, 2.0, noise_01^3), where noise_01 is the
    field's raw [-1, 1] output rescaled to [0, 1] before the lerp (so
    "noise^3" biases toward the extremes exactly as the taxonomy's
    formula specifies). The floor (min_speed) is untouched — only the
    cap is modulated, so slow zones still respect the mode's normal
    minimum rather than letting birds stall.
    """

    _FREQ = 0.01  # spatial frequency — cycles per world unit

    @staticmethod
    def apply(velocities, active, caps, min_speed, speeds,
              positions=None, rng=None, dt=0.0):
        if positions is None:
            # No position available — degrade to the band strategy
            # rather than silently no-op (a caller that forgets to pass
            # positions should still get *a* speed policy).
            return BandSpeedModel.apply(
                velocities, active, caps, min_speed, speeds,
                positions, rng, dt,
            )

        f = NoiseModulatedSpeedModel._FREQ
        q = positions * f
        noise = (
            np.sin(q[:, 0] * 1.3 + np.cos(q[:, 1] * 1.7 + 0.5))
            * np.cos(q[:, 2] * 1.1 + np.sin(q[:, 0] * 0.9 + 1.7))
        ).astype(np.float32)
        noise_01 = (np.clip(noise, -1.0, 1.0) + 1.0) * 0.5
        mult = 0.5 + 1.5 * (noise_01 ** 3)
        noisy_caps = caps * mult

        too_fast = (speeds.ravel() > noisy_caps).ravel() & active
        too_slow = (speeds.ravel() < min_speed).ravel() & active
        if too_fast.any():
            velocities[too_fast] = (
                velocities[too_fast] / speeds[too_fast]
            ) * noisy_caps[too_fast, np.newaxis]
        if too_slow.any():
            velocities[too_slow] = (
                velocities[too_slow] / (speeds[too_slow] + 1e-10)
            ) * min_speed[too_slow, np.newaxis]
        return None


@register("velocity_adaptive")
class VelocityAdaptiveSpeedModel(SpeedModel):
    """Speed smoothly approaches a randomized-bonus target via
    exponential lerp rather than a hard clamp. Matches §11's
    lerp-toward-goal mechanism from the comparison taxonomy, simplified:
    §11's full form keys the target speed on an external per-bird
    behavioural state (normal vs. emergency acceleration rate) that this
    signature has no channel for — implementing that would need a
    per-bird state array threaded in from outside, which is exactly the
    kind of "steering decoupling" work tracked separately, not something
    to bolt onto a speed-enforcement strategy. This implements the
    randomized-bonus + smooth-approach part on its own, at a fixed lerp
    rate, re-rolling the bonus every frame rather than the taxonomy's
    periodic re-roll (no per-bird persistent state channel to remember
    "next reroll time" either).

    goal = caps * bonus, bonus ~ U(0.85, 1.15)
    v <- lerp(v, direction(v) * goal, clamp(LERP_RATE * dt, 0, 1))
    """

    LERP_RATE = 3.0  # 1/seconds — how fast speed re-converges to goal
    BONUS_MIN = 0.85
    BONUS_MAX = 1.15

    @staticmethod
    def apply(velocities, active, caps, min_speed, speeds,
              positions=None, rng=None, dt=0.0):
        n = len(velocities)
        if rng is not None:
            bonus = rng.uniform(
                VelocityAdaptiveSpeedModel.BONUS_MIN,
                VelocityAdaptiveSpeedModel.BONUS_MAX,
                n,
            ).astype(np.float32)
        else:
            bonus = np.ones(n, dtype=np.float32)
        goal = caps * bonus

        safe_speeds = speeds + 1e-10
        dirs = velocities / safe_speeds
        zero_mask = (speeds.ravel() < 1e-6) & active
        if zero_mask.any():
            dirs[zero_mask.ravel(), 0] = 1.0
            dirs[zero_mask.ravel(), 1] = 0.0
            dirs[zero_mask.ravel(), 2] = 0.0
        target = dirs * goal[:, np.newaxis]

        alpha = float(np.clip(VelocityAdaptiveSpeedModel.LERP_RATE * dt, 0.0, 1.0))
        velocities[active] = (
            velocities[active] * (1.0 - alpha) + target[active] * alpha
        )
        return None
