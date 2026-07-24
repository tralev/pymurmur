"""crs48 field/blob mode — vectorised per-bird terms, O(N), no neighbour queries.

P3.1: boundedUnitTravel wander path (in extensions/wander.py).
P3.2: 5 Lissajous blob anchors + per-bird cyclic phase weights for
      target blending.
P3.3: Leader/chaser groups — 7 seed groups, ~16% leaders,
      golden-angle stratified shells, chase_target blending.
P3.4: Shell force + inner cavity — per-bird R_blob oscillating shell,
      inner floor push-out.
S2.A5: Target-pull — direct pull toward T, distinct from the shell's
       oscillating equilibrium-radius behaviour.
P3.5: Slot repulsion — quadratic kernel ((r_slot−d)/r_slot)² at
      offsets ±{1,7,31}, mod-wrapped around the active-bird index ring.
P3.6: Remaining 6 terms — tangential orbital, buoyancy, curl flow,
      fold noise, viscous drag, drift alignment to wander_heading.
C3: field_noise — deterministic per-bird jitter via seed_noise3.
P3.12: Floating boundary — 1.45·R_blob dynamic soft boundary.

S2.A5: All of the above are registered as named ForceTerm entries in
FIELD_TERMS and composed via composeForces() (physics/forces/_base.py)
— disabled_terms toggles them at runtime by name.

P2.2: Wrapped in FieldMode(ForceMode) with @register("field").

File-size split: the blob-anchor/leader-chaser setup functions live in
field_anchors.py, the 14 per-bird _compute_* force-term implementations
in field_terms.py. This file keeps FieldTermContext, the thin _term_*
adapters, FIELD_TERMS, and the FieldMode class itself.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._base import ForceTerm, composeForces
from ._mode import ForceFn, ForceMode, register
from .field_anchors import (
    _compute_anchors,
    _compute_leader_chaser,
    _compute_phases,
    _compute_targets,
    _group_anchor,  # noqa: F401  # re-export (tests import directly)
    _hash01,  # noqa: F401  # re-export (tests import directly)
)
from .field_terms import (
    _compute_buoyancy,
    _compute_curl_flow,
    _compute_drift_alignment,
    _compute_field_noise,
    _compute_floating_boundary,
    _compute_fold_noise,
    _compute_grid_sep_normalized,  # noqa: F401  # re-export (tests import directly)
    _compute_shell_force,
    _compute_slot_repulsion,
    _compute_tangential,
    _compute_target_pull,
    _compute_viscous_drag,
)

if TYPE_CHECKING:
    from ...core.config import SimConfig
    from ...core.types import SpatialIndex


# ── S2.A5: Field-mode term composition contract ──────────────────────

@dataclass
class FieldTermContext:
    """S2.A5: Shared per-frame context passed to every field-mode ForceTerm.

    Built once per FieldMode.compute() call; consumed by each entry in
    FIELD_TERMS. Most terms read the active-compacted p_active/v_active/
    seeds slices; slot_repulsion reads the full-width positions/active
    directly since it pairs birds by array offset, not a per-bird query.
    """

    config: SimConfig
    positions: np.ndarray       # (N, 3) full-width
    active: np.ndarray          # (N,) full-width bool
    n_active: int
    p_active: np.ndarray        # (n_active, 3)
    v_active: np.ndarray        # (n_active, 3)
    seeds: np.ndarray           # (n_active,)
    t: float
    C: np.ndarray                # (3,) flock centroid
    U: float
    targets: np.ndarray          # (n_active, 3) — from leader/chaser (P3.2/P3.3)
    coh_eff: np.ndarray | float  # per-bird or scalar (P3.8 blackening)
    sep_eff: np.ndarray | float
    chase: float
    align: float
    flow: float
    flow_pull: float
    v0: float
    ripple_env: float | np.ndarray


def _scatter(active: np.ndarray, n: int, values: np.ndarray) -> np.ndarray:
    """Place (n_active, 3) per-bird values into a zeroed (n, 3) array
    at the active rows — the shared shape every ForceTerm.fn returns."""
    out = np.zeros((n, 3), dtype=np.float32)
    out[active] = values
    return out


def _term_shell(fx: FieldTermContext) -> np.ndarray:
    shell_influence = fx.config.field.field_shell_influence
    vals = _compute_shell_force(
        fx.p_active, fx.targets, fx.seeds, fx.t, fx.U,
        fx.coh_eff, fx.chase, fx.sep_eff, shell_influence,
        shell_radius_base=fx.config.field.field_shell_radius_base,
        inner_radius_factor=fx.config.field.field_inner_radius_factor,
    )
    return _scatter(fx.active, len(fx.positions), vals)


def _term_target_pull(fx: FieldTermContext) -> np.ndarray:
    vals = _compute_target_pull(
        fx.p_active, fx.targets, fx.U, fx.coh_eff, fx.config.field_target_pull,
    )
    return _scatter(fx.active, len(fx.positions), vals)


def _term_slot_repulsion(fx: FieldTermContext) -> np.ndarray:
    # Already full-width — pairs birds by array offset, not active-compacted.
    return _compute_slot_repulsion(
        fx.positions, fx.active, fx.n_active, fx.U, fx.config.field_separation, fx.chase,
    )


def _term_tangential(fx: FieldTermContext) -> np.ndarray:
    tangent_pull = fx.config.field.field_tangent_pull
    vals = _compute_tangential(
        fx.p_active, fx.targets, fx.seeds, fx.t, fx.align, fx.chase, tangent_pull,
    )
    return _scatter(fx.active, len(fx.positions), vals)


def _term_buoyancy(fx: FieldTermContext) -> np.ndarray:
    vals = _compute_buoyancy(fx.p_active, fx.targets, fx.seeds, fx.t, fx.U, fx.flow)
    return _scatter(fx.active, len(fx.positions), vals)


def _term_curl_flow(fx: FieldTermContext) -> np.ndarray:
    vals = _compute_curl_flow(fx.p_active, fx.C, fx.seeds, fx.t, fx.U, fx.flow, fx.flow_pull)
    return _scatter(fx.active, len(fx.positions), vals)


def _term_fold_noise(fx: FieldTermContext) -> np.ndarray:
    vals = _compute_fold_noise(
        fx.p_active, fx.C, fx.seeds, fx.t, fx.U, fx.flow, fx.flow_pull, fx.ripple_env,
    )
    return _scatter(fx.active, len(fx.positions), vals)


def _term_field_noise(fx: FieldTermContext) -> np.ndarray:
    vals = _compute_field_noise(fx.seeds, fx.t, fx.config.field_noise)
    return _scatter(fx.active, len(fx.positions), vals)


def _term_viscous_drag(fx: FieldTermContext) -> np.ndarray:
    vals = _compute_viscous_drag(fx.v_active, fx.chase, fx.flow)
    return _scatter(fx.active, len(fx.positions), vals)


def _term_drift_alignment(fx: FieldTermContext) -> np.ndarray:
    drift_pull = fx.config.field.field_drift_pull
    wander_heading = getattr(fx.config, '_wander_heading', None)
    # C3: field_drift_direction — static fallback when Wander is disabled.
    # Default (0,0,0) means "unset", so it never engages unless a preset
    # configures a real direction.
    if wander_heading is None:
        static_dir = np.asarray(fx.config.field.field_drift_direction, dtype=np.float32)
        static_norm = np.linalg.norm(static_dir)
        if static_norm > 1e-6:
            wander_heading = static_dir / static_norm
    vals = _compute_drift_alignment(fx.v_active, wander_heading, fx.v0, fx.align, drift_pull)
    return _scatter(fx.active, len(fx.positions), vals)


def _term_floating_boundary(fx: FieldTermContext) -> np.ndarray:
    phases = _compute_phases(fx.seeds, fx.t)
    R_blobs = (
        fx.config.field.field_shell_radius_base
        + np.sin(fx.seeds * 41.0 + fx.t * 0.29) * 0.08
        + np.sin(phases * 2.0 * np.pi + fx.t * 0.17) * 0.05
    ) * fx.U
    mu = fx.config.boundary_avoidance_factor
    vals = _compute_floating_boundary(fx.p_active, fx.C, R_blobs, fx.U, mu)
    return _scatter(fx.active, len(fx.positions), vals)


# S2.A5: ordered, named term table — disabled_terms names below must match
# these exactly. Each ForceTerm.gain stays at the default 1.0 since every
# term's own formula already bakes in its config-driven scaling; gain is
# an extra multiplier hook for future A/B comparison, not required to
# reproduce current behaviour.
FIELD_TERMS: list[ForceTerm] = [
    ForceTerm("shell", fn=_term_shell),
    ForceTerm("target_pull", fn=_term_target_pull),
    ForceTerm("slot_repulsion", fn=_term_slot_repulsion),
    ForceTerm("tangential", fn=_term_tangential),
    ForceTerm("buoyancy", fn=_term_buoyancy),
    ForceTerm("curl_flow", fn=_term_curl_flow),
    ForceTerm("fold_noise", fn=_term_fold_noise),
    ForceTerm("noise", fn=_term_field_noise),
    ForceTerm("viscous_drag", fn=_term_viscous_drag),
    ForceTerm("drift_alignment", fn=_term_drift_alignment),
    ForceTerm("floating_boundary", fn=_term_floating_boundary),
]

_FIELD_TERM_NAMES = frozenset(term.name for term in FIELD_TERMS)


# ── FieldMode ──────────────────────────────────────────────────────

@register("field")
class FieldMode(ForceMode):
    """crs48 field/blob anchor mode — O(N), fully vectorised.

    Composes P3.1–P3.12 terms via the S2.A5 ForceTerm/composeForces
    contract (FIELD_TERMS, above):
    P3.2: Blob anchors + phase weights → T_legacy
    P3.3: Leader/chaser → blended targets
    P3.4: Shell force + inner cavity
    S2.A5: Target-pull (direct pull toward T, distinct from shell)
    P3.5: Slot repulsion (quadratic kernel, mod-wrapped ±{1,7,31})
    P3.6: Tangential, buoyancy, curl flow, fold noise, drag, drift
    C3: field_noise — deterministic per-bird jitter
    P3.12: Floating boundary (1.45·R_blob)
    """

    needs_index = False

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
        """Compute field/blob anchor forces — O(N), fully vectorised."""
        n_active = active.sum()
        if n_active == 0:
            return

        # ── Time, centre, unit scale ──
        t = getattr(config, '_field_time', 0.0)
        C = np.mean(positions[active], axis=0)

        unit_scale = config.field.field_unit_scale
        U = float(unit_scale) if unit_scale is not None else (
            0.4 * min(config.width, config.height, config.depth)
        )

        # ── Config shorthand ──
        coh = config.field_cohesion
        align = config.field_alignment
        sep = config.field_separation
        flow = config.field_flow
        chase = config.field.field_chase_strength
        v0 = config.v0

        # S2.A5/C3: disabled_terms — skip named sub-terms at runtime.
        # Unknown names (typos, renamed/removed terms) warn rather than
        # silently no-op.
        skip = frozenset(config.field.disabled_terms) if config.field.disabled_terms else frozenset()
        unknown = skip - _FIELD_TERM_NAMES
        if unknown:
            warnings.warn(
                f"config.field.disabled_terms contains unknown term name(s): "
                f"{sorted(unknown)}. Known terms: {sorted(_FIELD_TERM_NAMES)}",
                stacklevel=2,
            )

        # ── Seeds ──
        seeds = np.arange(n_active, dtype=np.float32)

        # ── P3.2: Blob anchors + T_legacy ──
        anchors = _compute_anchors(t, C, U)
        T_legacy = _compute_targets(seeds, t, anchors)

        # ── P3.3/S2.A3: Leader/chaser → blended targets ──
        wander_heading = getattr(config, '_wander_heading', None)
        targets = _compute_leader_chaser(
            seeds, t, T_legacy, anchors, U, chase, sep,
            num_groups=config.field.field_num_groups,
            leader_fraction=config.field.field_leader_fraction,
            C=C, wander_heading=wander_heading,
        )

        # Active-sliced views
        p_active = positions[active]
        v_active = velocities[active]

        # ── P3.8: Blackening — modulate separation/cohesion for threatened birds ──
        threat_present = getattr(config, '_threat_present', False)
        if threat_present:
            threat_black = getattr(config, '_threat_blackening', None)
            threat_active_idx = getattr(config, '_threat_active', None)
            if threat_black is not None and threat_active_idx is not None and len(threat_active_idx) > 0:
                # Create per-bird effective cohesion and separation arrays
                coh_eff = np.full(n_active, coh, dtype=np.float32)
                sep_eff = np.full(n_active, sep, dtype=np.float32)
                # Vectorised reverse-lookup: global index → active-sliced position
                active_idx = np.where(active)[0]
                active_pos = np.full(positions.shape[0], -1, dtype=np.int32)
                active_pos[active_idx] = np.arange(n_active, dtype=np.int32)
                # Only modulate birds that are both active AND threatened
                valid = active_pos[threat_active_idx] >= 0
                if valid.any():
                    pos = active_pos[threat_active_idx[valid]]
                    black = threat_black[threat_active_idx[valid]].astype(np.float32)
                    # P3.8: sep_eff = sep · (2 − black), coh_eff = coh · black
                    sep_eff[pos] = sep * (2.0 - black)
                    coh_eff[pos] = coh * black
            else:
                coh_eff = coh
                sep_eff = sep
        else:
            coh_eff = coh
            sep_eff = sep

        # ── D10: per-bird ripple envelope — index by active when it's an array ──
        ripple_env = getattr(config, '_ripple_envelope_sum', 1.0)
        if isinstance(ripple_env, np.ndarray) and ripple_env.ndim == 1:
            ripple_env = ripple_env[active]

        # ── S2.A5: build the shared context and compose all enabled terms ──
        fx = FieldTermContext(
            config=config,
            positions=positions,
            active=active,
            n_active=n_active,
            p_active=p_active,
            v_active=v_active,
            seeds=seeds,
            t=t,
            C=C,
            U=U,
            targets=targets,
            coh_eff=coh_eff,
            sep_eff=sep_eff,
            chase=chase,
            align=align,
            flow=flow,
            flow_pull=config.field.field_flow_pull,
            v0=v0,
            ripple_env=ripple_env,
        )
        active_terms = [term for term in FIELD_TERMS if term.name not in skip]
        accelerations += composeForces(fx, active_terms, n=len(positions))

        # ── Clamp ──
        acc_mags = np.linalg.norm(accelerations, axis=1)
        too_strong = (acc_mags > config.max_force) & active
        if too_strong.any():
            accelerations[too_strong] = (
                accelerations[too_strong]
                / acc_mags[too_strong, np.newaxis]
                * config.max_force
            )


# Backward compatibility alias — tests import field_forces directly
field_forces: ForceFn = FieldMode.compute  # type: ignore[assignment]
field_forces.needs_index = False
