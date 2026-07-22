"""P12.1 — MARL force mode: deferred global rules under external control.

The engine order for this mode:
  1. Apply external control: v += a_ext * action_scale * v_cap (component-clipped to ±v_cap)
  2. Move: p += v * dt
  3. Rules prep the *next* step: v += rule_weight * (F_sep(d < sep_radius*U) + (v̄−v) + (CoM−p))

where rule_weight=0.01, global neighbourhood (all active birds, no radius limit
on align/cohere), and v_cap = marl.velocity_cap * U with U = min(W,H,D)/6.

This two-step lag means positions at step k depend only on rules from step k-1
(and the control action at step k) — essential for the MARL observation model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._mode import ForceFn, ForceMode, register

if TYPE_CHECKING:
    from ...core.config import SimConfig
    from ...core.types import SpatialIndex


# MARL hyperparameter defaults are now in MarlConfig (core/config.py).


@register("marl")
class MarlMode(ForceMode):
    """MARL force mode: external control + deferred global rules.

    Reads the external action from config._marl_action (set by the gym
    wrapper before each step), applies it to velocities, then computes
    deferred separation/alignment/cohesion for the *next* step.
    """

    needs_index = False  # no index — global neighbourhood
    # D2: mode clamps velocities to its own [0.3*v_cap, v_cap] band using
    # v_cap (unit-scale U-relative), not v0 — the generic integrate() clamp
    # uses v0 as its reference and would double-clamp against a mismatched
    # scale, so the mode must own its speed policy entirely.
    speed_mode = "none"

    @staticmethod
    def compute(
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
        active: np.ndarray,
        index: SpatialIndex | None,
        rng: np.random.Generator,
        last_theta: np.ndarray,
        config: SimConfig,
    ) -> None:
        act_idx = np.where(active)[0]
        n = len(act_idx)
        if n == 0:
            return

        W, H, D = config.width, config.height, config.depth
        U = min(W, H, D) / 6.0
        v_cap = config.marl.marl_velocity_cap * U
        rule_weight = config.marl.marl_rule_weight
        sep_radius = config.marl.marl_separation_radius * U
        action_scale = config.marl.marl_action_scale

        # ── Step 1: Apply external control action ──────────────────
        a_ext = getattr(config, "_marl_action", None)
        if a_ext is not None and hasattr(a_ext, "shape"):
            a_ext = np.asarray(a_ext, dtype=np.float32)
            if a_ext.shape[0] == n:
                # Component-clip to ±v_cap
                delta = np.clip(a_ext * action_scale * v_cap, -v_cap, v_cap)
                velocities[act_idx] += delta

        # ── Step 2: Rules prep the *next* step ─────────────────────
        pos = positions[act_idx]
        vel = velocities[act_idx]

        # --- Separation: 1/d² repulsion within sep_radius ---
        # Global neighbourhood for separation (kd-tree not required —
        # flocks are small in MARL mode, N ≈ 20–50).
        f_sep = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            delta = pos - pos[i]
            # Toroidal wrapping
            for dim, domain in enumerate([W, H, D]):
                col = delta[:, dim]
                col[col < -domain / 2] += domain
                col[col > domain / 2] -= domain
                delta[:, dim] = col
            dists = np.linalg.norm(delta, axis=1)
            close = (dists < sep_radius) & (dists > 1e-6)
            if close.any():
                direction = delta[close] / dists[close, np.newaxis]
                repulsion = direction / (dists[close, np.newaxis] ** 2 + 1e-8)
                # Cap each repulsion vector at 1.0 to prevent explosions
                repulsion = np.clip(repulsion, -1.0, 1.0)
                f_sep[i] = np.sum(repulsion, axis=0)

        # --- Alignment: global mean velocity ---
        v_mean = np.mean(vel, axis=0)
        f_align = v_mean - vel  # (n, 3): difference from mean

        # --- Cohesion: toward global CoM ---
        com = np.mean(pos, axis=0)
        f_coh = com - pos  # (n, 3): toward centre

        # Apply deferred rules
        rule_accel = rule_weight * (f_sep + f_align + f_coh)
        velocities[act_idx] += rule_accel

        # Clamp speed: [0.3 * v_cap, v_cap]
        speeds = np.linalg.norm(velocities[act_idx], axis=1)
        min_speed = 0.3 * v_cap
        scale = np.ones(n, dtype=np.float32)
        high = speeds > v_cap
        if high.any():
            scale[high] = v_cap / speeds[high]
        low = speeds < min_speed
        if low.any():
            scale[low] = min_speed / np.maximum(speeds[low], 1e-8)
        velocities[act_idx] *= scale[:, np.newaxis]


# Backward-compatible function alias
marl_forces: ForceFn = MarlMode.compute  # type: ignore[assignment]
marl_forces.needs_index = MarlMode.needs_index
