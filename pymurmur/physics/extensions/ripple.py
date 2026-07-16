"""Ripple extension — 3-train density pulse envelopes (P3.7).

Three trains at staggered offsets {0, 9.33, 18.67}s, each with a 28 s
cycle, smoothstep envelope, moving Lissajous origins, radial + twist
forces, and ripple_envelope_sum exported for fold-noise coupling (P3.6).

O(1) per frame.  Vectorised — no per-bird Python loops.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from ._base import Extension

if TYPE_CHECKING:
    from ..flock import PhysicsFlock
    from ._base import StepContext

# ── Smoothstep helper ──────────────────────────────────────────────

def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    """Hermite smoothstep: t²(3−2t) clamped to [0,1]."""
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0)
    return (t ** 2 * (3.0 - 2.0 * t)).astype(np.float32)


# ── Ripple extension (P3.7) ────────────────────────────────────────

class Ripple(Extension):
    """3-train enveloped travelling pulses (P3.7).

    Three trains at offsets {0, 9.33, 18.67}s, each with a 28 s cycle.
    Per train:
        τ = (t − offset) mod 28
        env(τ) = smoothstep(0.6, 1.7, τ) · (1 − smoothstep(6.2, 8.8, τ))
        radius(τ) = (0.16 + τ·0.16)·U
        width(τ) = (0.11 + τ·0.012)·U
        origin = C + Lissajous path at phase τ+offset
        amount = exp(−((r−radius)/width)²) · env
        F_radial = (p−origin)/r · amount
        F_twist = heading × F_radial
        F = (F_radial + F_twist·0.28) · flow · (0.13 + waveGain·0.04)

    Exports ripple_envelope_sum = Σ_trains amount for fold-noise coupling.
    """

    _CYCLE = 28.0
    _OFFSETS = np.array([0.0, 9.33, 18.67], dtype=np.float32)

    def __init__(self) -> None:
        self._t: float = 0.0

    def apply(self, flock: PhysicsFlock, ctx: StepContext) -> None:
        """Apply 3-train ripple pulses to the flock."""
        config = ctx.config
        n_active = flock.active.sum()
        if n_active == 0:
            config._ripple_envelope_sum = 0.0
            return

        self._t += ctx.dt

        # ── Unit scale ──
        unit_scale = getattr(config, 'field_unit_scale', None)
        U = float(unit_scale) if unit_scale is not None else (
            0.4 * min(config.width, config.height, config.depth)
        )

        flow = config.field_flow
        wave_gain = getattr(config, 'field_wave_gain', 0.5)

        # ── Flock centre ──
        C = np.mean(flock.positions[flock.active], axis=0)
        active_idx = np.where(flock.active)[0]
        positions = flock.positions[active_idx]

        # Compute heading from flock mean velocity for twist
        mean_vel = np.mean(flock.velocities[flock.active], axis=0)
        heading = mean_vel / max(np.linalg.norm(mean_vel), 1e-6)
        heading = heading.astype(np.float32)

        # ── Accumulators ──
        ripple_envelope_sum = np.zeros(n_active, dtype=np.float32)

        for offset in self._OFFSETS:
            train_t = self._t - offset
            if train_t < 0.0:
                continue

            tau = train_t % self._CYCLE

            # ── Smoothstep envelope ──
            env_rise = _smoothstep(0.6, 1.7, tau)
            env_fall = 1.0 - _smoothstep(6.2, 8.8, tau)
            env = env_rise * env_fall  # scalar

            if env < 1e-6:
                continue

            # ── Radius and width ──
            radius = (0.16 + tau * 0.16) * U
            width = (0.11 + tau * 0.012) * U

            # ── Moving Lissajous origin ──
            origin_phase = offset  # per-train phase shift
            origin = C + np.array([
                np.sin(self._t * 0.17 + origin_phase) * 0.46,
                np.cos(self._t * 0.13 + origin_phase * 1.7) * 0.25,
                np.cos(self._t * 0.19 + origin_phase * 0.6) * 0.42,
            ], dtype=np.float32) * U

            # ── Radial distance, amount, force ──
            to_origin = positions - origin
            r = np.linalg.norm(to_origin, axis=1)
            safe_r = np.maximum(r, 1e-6)
            r_hat = to_origin / safe_r[:, np.newaxis]

            delta = (r - radius) / max(width, 1e-6)
            amount = np.exp(-delta * delta) * env

            # Radial force (outward from origin)
            F_radial = r_hat * amount[:, np.newaxis]

            # Twist force (perpendicular to radial, in heading plane)
            twist = np.cross(heading.reshape(1, 3), F_radial)
            F_total = (F_radial + twist * 0.28) * flow * (0.13 + wave_gain * 0.04)

            flock.accelerations[active_idx] += F_total.astype(np.float32)

            # Accumulate envelope for fold-noise coupling
            ripple_envelope_sum += amount.astype(np.float32)

        # ── Export envelope SUM for field mode fold noise (P3.6) ──
        # Spec: ripple_envelope_sum = Σ_trains amount
        config._ripple_envelope_sum = float(np.sum(ripple_envelope_sum)) if n_active > 0 else 0.0
