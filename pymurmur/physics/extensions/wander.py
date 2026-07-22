"""Wander extension — bounded attractor motion for the flock centre.

P3.1: boundedUnitTravel — 13-term composite trig + pulse normalisation.
‖path(t)‖ ≤ 1 guaranteed.  heading(t) = normalize(path(t+0.75) − path(t)).

O(1) per frame.  Publishes wander_center and wander_heading to PhysicsFlock
for consumption by field mode's drift alignment (P3.6).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._base import Extension

if TYPE_CHECKING:
    from ..flock import PhysicsFlock
    from ._base import StepContext


# ── Bounded unit path (P3.1) ──────────────────────────────────────

def bounded_unit_path(t: float) -> np.ndarray:
    """Compute ‖path‖ ≤ 1 guaranteed bounded-unit-travel position.

    Formula from roadmap P3.1: 13-term composite trig with pulse
    normalisation.  Safe for vectorised calls (t can be an array).
    """
    raw_x = (
        np.sin(t * 0.47 + np.sin(t * 0.19) * 1.15) * 0.82
        + np.sin(t * 1.07 + 1.4) * 0.38
        + np.cos(t * 0.23 + 2.1) * 0.22
    )
    raw_y = (
        np.cos(t * 0.43 + 0.6 + np.sin(t * 0.13) * 0.9) * 0.78
        + np.sin(t * 0.91 + 2.8) * 0.42
        + np.cos(t * 0.29 + 0.4) * 0.24
    )
    raw_z = (
        np.sin(t * 0.39 + 1.1 + np.cos(t * 0.17) * 1.05) * 0.80
        + np.cos(t * 0.97 + 0.2) * 0.40
        + np.sin(t * 0.21 + 2.6) * 0.22
    )
    pulse = 0.72 + 0.28 * (0.5 + 0.5 * np.sin(t * 0.41 + np.cos(t * 0.17)))

    # Stack into (N,3) array — works for both scalar and array t
    raw = np.column_stack([np.atleast_1d(raw_x),
                           np.atleast_1d(raw_y),
                           np.atleast_1d(raw_z)]).astype(np.float32)
    pulse_arr = np.atleast_1d(pulse).astype(np.float32)

    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    scale = np.where(norms > 1.0, 1.0 / norms, 1.0)
    result = raw * scale * pulse_arr[:, np.newaxis]

    # Return scalar result when t was scalar
    if np.ndim(t) == 0:
        return result[0]
    return result


def wander_heading(t: float) -> np.ndarray:
    """heading(t) = normalize(path(t + 0.75) − path(t)).

    Returns a unit vector indicating the direction of motion.
    """
    p0 = bounded_unit_path(t)
    p1 = bounded_unit_path(t + 0.75)
    diff = p1 - p0
    norm = np.linalg.norm(diff)
    if norm > 1e-10:
        return (diff / norm).astype(np.float32)
    return np.array([1.0, 0.0, 0.0], dtype=np.float32)


# ── Wander extension ──────────────────────────────────────────────

class Wander(Extension):
    """Bounded flock-centre wander using boundedUnitTravel.

    Computes wander_center(t) = C + path(t·speed)·radius·U each frame
    and publishes to PhysicsFlock.wander_center / wander_heading.

    For non-field modes, also applies a gentle global pull toward the
    wander centre to maintain backward compatibility.
    """

    def __init__(self) -> None:
        self._t: float = 0.0

    def apply(self, flock: PhysicsFlock, ctx: StepContext) -> None:
        """Compute wander centre and heading, publish to flock.

        For non-field modes, applies a vectorised pull on all active
        birds toward the wander centre (backward-compat fallback).
        """
        cfg = ctx.config
        active = flock.active
        n_active = active.sum()

        # ── Advance internal clock (always, even with no birds) ──
        self._t += ctx.dt

        # ── Wander speed and radius from config ──
        speed = cfg.wander.wander_attractor_speed
        radius = cfg.wander.wander_attractor_radius

        # ── Compute wander centre ──
        # Use flock's smoothed centre C as the anchor point.
        # Note: U (unit_scale) is NOT applied to wander — radius is
        # already in domain-space units.  U is reserved for field-mode
        # blob anchors (P3.2).
        if flock.center is not None:
            C = flock.center
        elif n_active > 0:
            C = np.mean(flock.positions[active], axis=0)
        else:
            # Fallback: domain centre when no birds and no EMA centre
            C = np.array(
                [cfg.width / 2, cfg.height / 2, cfg.depth / 2],
                dtype=np.float32,
            )

        path_t = bounded_unit_path(self._t * speed)
        wander_center = C + path_t * radius

        # ── Compute heading ──
        heading = wander_heading(self._t * speed)

        # ── Publish to flock for external inspection/tests ──
        flock.wander_center = wander_center.astype(np.float32)
        flock.wander_heading = heading.astype(np.float32)

        # S2.A3/P3.6: also publish onto cfg — FieldMode.compute() is a
        # stateless module function that only ever receives (arrays,
        # config), never `flock`, so this is the only bridge that
        # actually reaches field.py's drift-alignment and leader-target
        # terms (both previously always saw None here despite this
        # method's flock.wander_heading assignment above — that attribute
        # was published but never consumed by anything).
        cfg._wander_heading = heading.astype(np.float32)

        # ── Backward-compat: apply pull for non-field modes ──
        if n_active > 0 and cfg.mode != "field":
            self._apply_pull(flock, wander_center, active)

    @staticmethod
    def _apply_pull(
        flock: PhysicsFlock,
        target: np.ndarray,
        active: np.ndarray,
    ) -> None:
        """Vectorised gentle pull toward wander centre (non-field fallback).

        Uses integer indexing to modify accelerations in-place —
        boolean-index chaining (e.g. accelerations[active][mask])
        would operate on a copy and leave the original unchanged.
        """
        to_target = target - flock.positions[active]
        dists = np.linalg.norm(to_target, axis=1)
        mask = dists > 1e-6
        if not mask.any():
            return
        direction = np.zeros_like(to_target)
        direction[mask] = to_target[mask] / dists[mask, np.newaxis]
        # Gentle pull: 0.0005 * dist clamped to 0.1 max
        pull = np.minimum(dists[mask] * 0.0005, 0.1)

        # Integer-index into original array to mutate in-place
        active_idx = np.where(active)[0]
        masked_idx = active_idx[mask]
        flock.accelerations[masked_idx] += direction[mask] * pull[:, np.newaxis]
