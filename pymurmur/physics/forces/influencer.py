"""Phase 7 — Influencer parity (MurmuratR).

Persistent tick-driven Lissajous target, move-then-steer at unit speed,
rank-by-target-distance influence, density-scaled Gaussian init,
per-frame distance diagnostics, desktop pilotable-flock mode.

P2.2: Wrapped in InfluencerMode(ForceMode) with @register("influencer").
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from ._mode import ForceFn, ForceMode, register

if TYPE_CHECKING:
    from ...core.config import SimConfig
    from ...core.types import SpatialIndex


# ── P7.1: Spec Lissajous target formula ─────────────────────────

def _lissajous_target(
    t: float,
    C: np.ndarray,   # domain centre [3,]
    s: float,         # spatial scale factor
    freq_p: tuple[float, float, float] = (97.0, 29.0, 41.0),
    freq_s: tuple[float, float, float] = (217.0, 13.0, 7.0),
    amp_p: tuple[float, float, float] = (200.0, 200.0, 100.0),
    amp_s: tuple[float, float, float] = (30.0, 30.0, 27.0),
    phase: tuple[float, float, float] = (0.0, 53.0, 61.0),
    vert_offset: float = 40.0,
) -> np.ndarray:
    """P7.1/S2.E1: Compute 3D Lissajous target at persistent tick t.

    Verbatim spec (default coefficients):
        T_raw(t) = (sin(t/97)*200 + cos(t/217)*30,
                    cos((t+53)/29)*200 + sin((47-t)/13)*30,
                    cos((t+61)/41)*100 + sin((t+13)/7)*27 + 40)

    freq_p/amp_p/phase parametrize the primary (orbit) term per axis;
    freq_s/amp_s parametrize the secondary (flutter) term. The secondary
    term's additive constants (47 on y, 13 on z) and x's lack of a phase
    shift are structural to the verbatim formula, not exposed as config —
    only the frequencies/amplitudes/primary-phase and vertical offset are.

    Scale and offset so (0,0,vert_offset) maps to (C_x, C_y, vert_offset*s)
    above centre.
    """
    x = math.sin(t / freq_p[0]) * amp_p[0] + math.cos(t / freq_s[0]) * amp_s[0]
    y = (
        math.cos((t + phase[1]) / freq_p[1]) * amp_p[1]
        + math.sin((47.0 - t) / freq_s[1]) * amp_s[1]
    )
    z = (
        math.cos((t + phase[2]) / freq_p[2]) * amp_p[2]
        + math.sin((t + 13.0) / freq_s[2]) * amp_s[2]
        + vert_offset
    )

    T_raw = np.array([x, y, z], dtype=np.float32)
    # Shift: (0,0,vert_offset) in raw maps to centre at z + vert_offset*s
    offset = np.array([0.0, 0.0, vert_offset], dtype=np.float32)
    return C + (T_raw - offset) * s + np.array([0.0, 0.0, vert_offset * s], dtype=np.float32)


# ── P7.4: Density-scaled Gaussian init ──────────────────────────

def influencer_density_init(
    n: int,
    width: float,
    height: float,
    depth: float,
    scale: float,
    separation: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """P7.4/S2.E4: Density-scaled Gaussian positions for influencer mode.

    Math:
        σ = N^(1/3) · separation · s
        positions = rnorm(N,3) · σ + C + U(0, 10s)³

    S2.E4: the U(0, 10s)³ offset is a single SHARED draw applied to every
    bird (not per-bird jitter) — it shifts the whole cloud, it doesn't
    fuzz individual positions (that's already sigma's job).

    Returns (N, 3) float32 positions clustered around domain centre.
    """
    C = np.array([width / 2.0, height / 2.0, depth / 2.0], dtype=np.float32)
    sigma = (n ** (1.0 / 3.0)) * separation * scale
    positions = rng.normal(0.0, sigma, (n, 3)).astype(np.float32)
    positions += C
    # S2.E4: shared offset — one draw for the whole cloud, not per-bird.
    shared_offset = rng.uniform(0.0, 10.0 * scale, (3,)).astype(np.float32)
    positions += shared_offset
    return positions


# ── P7.6: Pilot target (user-controlled) ────────────────────────

class PilotTarget:
    """P7.6: User-steered attractor for desktop pilotable-flock mode.

    The pilot target is a WASD-controlled point in space.  Set .position
    and .heading directly from keyboard input; the influencer reads them
    when pilot_mode is active.

    Shell radius drives the enclosing shell force.  Expand/contract with
    scatter/gather toggles (Shift/Alt in P10 UX).
    """

    __slots__ = ("position", "heading", "shell_radius", "active")

    def __init__(
        self,
        position: np.ndarray | None = None,
        heading: np.ndarray | None = None,
    ) -> None:
        self.position: np.ndarray = (
            position.copy()
            if position is not None
            else np.zeros(3, dtype=np.float32)
        )
        self.heading: np.ndarray = (
            heading.copy()
            if heading is not None
            else np.array([0.0, 0.0, 0.0], dtype=np.float32)
        )
        self.shell_radius: float = 50.0
        self.active: bool = False

    def update_shell(
        self,
        dt: float,
        scatter: bool = False,
        gather: bool = False,
    ) -> None:
        """P7.6: Dynamically expand/contract shell radius.

        shell_radius = clamp(0.42, 2.2, radius + (scatter−gather)·dt·1.35)

        Args:
            dt: timestep
            scatter: True to expand (Shift)
            gather: True to contract (Alt)
        """
        delta = 0.0
        if scatter:
            delta += 1.0
        if gather:
            delta -= 1.0
        self.shell_radius = np.clip(
            self.shell_radius + delta * dt * 1.35,
            0.42,
            2.2,
        )


# ── InfluencerMode ──────────────────────────────────────────────

@register("influencer")
class InfluencerMode(ForceMode):
    """P7: Persistent tick-driven Lissajous target with move-then-steer.

    Class-level metadata:
        needs_index = False  — no neighbour queries
        owns_positions = True  — integrate() is called with move=False (D11);
            true per-substep move-then-steer happens inside compute() itself
        speed_mode = "fixed"   — constant-speed enforcement

    The mode steers bird directions toward a 3D Lissajous target
    with rank-based or distance-based influence weights (P7.3).
    A persistent tick counter ensures deterministic, repeatable
    trajectories (P7.1).  Per-frame distance diagnostics are
    stored on the config for metrics collection (P7.5).
    """

    needs_index = False
    owns_positions = True     # P7.2: integrate() should skip position update
    speed_mode = "fixed"      # P7.2: constant |v| = v0

    _pilot: PilotTarget | None = None     # P7.6: shared pilot state

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
        """P7.1–P7.5: Lissajous target follow with move-then-steer.

        Owns position updates (owns_positions=True): each substep moves
        birds along their current velocity, then steers toward the target.
        integrate() is called with move=False for this mode (D11), so the
        per-substep move here is the only position advance. Boundary
        enforcement still happens in integrate().
        """
        active_idx = np.where(active)[0]
        n_active = len(active_idx)
        if n_active == 0:
            return

        substeps = config.influencer_substeps
        rank_exp = config.influencer_rank_exponent
        scale = config.influencer_scale
        # S2.E3: use_rank_override forces rank-based influence regardless
        # of the influence_mode selector.
        influence_mode = "rank" if config.influencer_use_rank_override else config.influencer_influence_mode
        near_dist_sq = config.influencer_near_dist_sq
        influence_min = config.influencer_influence_min
        influence_max = config.influencer_influence_max
        tick_rate = config.influencer_tick_rate
        v0 = config.v0

        # Persistent tick (like _field_time for field mode)
        if not hasattr(config, '_influencer_tick'):
            config._influencer_tick = 0.0

        C = np.array(
            [config.width / 2.0, config.height / 2.0, config.depth / 2.0],
            dtype=np.float32,
        )
        s_val = scale * min(
            config.width / 460.0,
            config.height / 460.0,
            config.depth / 254.0,
        )
        traj_kwargs = dict(
            freq_p=config.influencer_target_freq_primary,
            freq_s=config.influencer_target_freq_secondary,
            amp_p=config.influencer_target_amp_primary,
            amp_s=config.influencer_target_amp_secondary,
            phase=config.influencer_target_phase_offsets,
            vert_offset=config.influencer_target_vert_offset,
        )

        pos = positions[active_idx]
        vel = velocities[active_idx]

        # P7.6: Pilot mode — use user-controlled target if active
        pilot = InfluencerMode._pilot
        pilot_active = pilot is not None and pilot.active

        # Hoist fibonacci_sphere import (used in both pilot and Lissajous branches)
        from ...core.types import fibonacci_sphere

        # Initialize target before loop (needed for substeps=0 diagnostics)
        if pilot_active and pilot is not None:
            target = pilot.position.astype(np.float32)
        else:
            target = _lissajous_target(config._influencer_tick, C, s_val, **traj_kwargs)

        dt_sub = 1.0 / 60.0  # per-substep dt (matches pilot steering dt)

        for _ in range(substeps):
            t = config._influencer_tick
            config._influencer_tick += tick_rate

            # D11: move-then-steer — advance positions along the current
            # velocity before steering (this mode owns positions).
            pos += vel * dt_sub

            # P7.1: Compute target position (update each substep)
            if pilot_active and pilot is not None:
                target = pilot.position.astype(np.float32)
                heading = pilot.heading.astype(np.float32)
                shell_r = pilot.shell_radius
            else:
                target = _lissajous_target(t, C, s_val, **traj_kwargs)
                heading = np.zeros(3, dtype=np.float32)
                shell_r = 0.0

            if pilot_active and pilot is not None:
                # P7.6: Full pilot formula — F = heading_force + core_follow + shell_pull
                # heading_force = pilot_heading * alignment * 0.12
                h_dir = heading / (np.linalg.norm(heading) + 1e-10)
                F_heading = h_dir * 0.12  # alignment fixed at 1.0 for pilot

                # core_follow = (pilot_pos - p_i) * cohesion * 0.22  [unbounded]
                to_pilot = target - pos
                F_core = to_pilot * 0.22  # cohesion fixed at 1.0 for pilot

                # shell_pull = (pilot_pos - p_i) / d * (d - shell_radius) * 0.42
                d_to_pilot = np.linalg.norm(to_pilot, axis=1, keepdims=True)
                shell_mask = (d_to_pilot > shell_r).flatten()
                F_shell = np.zeros_like(to_pilot)
                if shell_mask.any():
                    t_hat_shell = to_pilot[shell_mask] / (d_to_pilot[shell_mask] + 1e-10)
                    F_shell[shell_mask] = (
                        t_hat_shell
                        * (d_to_pilot[shell_mask] - shell_r)
                        * 0.42
                    )

                # Accumulate and convert to velocity steering
                F_pilot = F_heading + F_core + F_shell
                vel += F_pilot * (1.0 / 60.0)  # per-substep dt

                # Re-normalise to constant speed
                v_norms = np.linalg.norm(vel, axis=1, keepdims=True)
                zero_mask = (v_norms < 1e-10).flatten()
                vel[~zero_mask] = vel[~zero_mask] / (v_norms[~zero_mask] + 1e-10) * v0
                if zero_mask.any():
                    nz = int(np.sum(zero_mask))
                    vel[zero_mask] = fibonacci_sphere(nz)[:nz] * v0
            else:
                # P7.2: Move-then-steer — direction-based, not acceleration
                v_norms = np.linalg.norm(vel, axis=1, keepdims=True)
                zero_mask = (v_norms < 1e-10).flatten()
                d_old = vel / (v_norms + 1e-10)

                # Zero-velocity birds get a random direction (then steered)
                if zero_mask.any():
                    nz = int(np.sum(zero_mask))
                    rand_dirs = fibonacci_sphere(nz)
                    perm = rng.permutation(nz)
                    d_old[zero_mask] = rand_dirs[perm]

                # P7.3: Influence weights — rank or distance-based
                to_target = target - pos
                dists_to_target = np.linalg.norm(to_target, axis=1)

                if influence_mode == "rank":
                    ranks = np.argsort(dists_to_target).argsort().astype(np.float32)
                    ranks = ranks / max(n_active - 1, 1)
                    influence = (1.0 - ranks * 0.8) ** rank_exp
                else:
                    raw = near_dist_sq * (s_val ** 2) / (dists_to_target ** 2 + 1e-10)
                    influence = np.clip(raw, influence_min, influence_max)

                # Blend: d̂_new = normalize(d̂_old·(1−inf) + t̂·inf)
                t_hat = to_target / (dists_to_target[:, np.newaxis] + 1e-10)
                d_new = (
                    d_old * (1.0 - influence[:, np.newaxis])
                    + t_hat * influence[:, np.newaxis]
                )
                d_new_norms = np.linalg.norm(d_new, axis=1, keepdims=True)
                d_new = d_new / (d_new_norms + 1e-10)

                vel[:] = d_new * v0

        # P7.5: Distance diagnostics
        final_dists = np.linalg.norm(pos - target, axis=1)
        config._target_dist_min = float(final_dists.min())
        config._target_dist_max = float(final_dists.max())
        # S2.E5/D7: stash the final substep's target for the marker
        # renderer — an unrendered target is as undebuggable as an
        # unrendered threat (see Visualizer._draw_threat_marker).
        config._influencer_target_pos = target.copy()

        # Write back positions + velocities; accelerations zeroed (influencer
        # doesn't use the acceleration pipeline — it steers directly)
        positions[active_idx] = pos
        velocities[active_idx] = vel
        accelerations[active_idx] = 0.0

    @staticmethod
    def density_init_positions(
        n: int,
        width: float,
        height: float,
        depth: float,
        config: SimConfig,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """P7.4: Density-scaled Gaussian position init.

        σ = N^(1/3) · separation · s
        positions = rnorm(N,3) · σ + C + U(0, 10s)³
        """
        return influencer_density_init(
            n=n,
            width=width,
            height=height,
            depth=depth,
            scale=config.influencer_scale,
            separation=config.influencer_init_separation,
            rng=rng,
        )

    @classmethod
    def set_pilot(cls, pilot: PilotTarget | None) -> None:
        """P7.6: Set or clear the shared pilot target.

        When active, all InfluencerMode instances steer toward the
        pilot position instead of the Lissajous target.
        """
        cls._pilot = pilot


# Backward compatibility alias — tests import influencer_forces directly
influencer_forces: ForceFn = InfluencerMode.compute  # type: ignore[assignment]
influencer_forces.needs_index = False
