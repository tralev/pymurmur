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
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ...core.types import seed_noise3
from ._base import ForceTerm, composeForces, curl_flow
from ._mode import ForceFn, ForceMode, register

if TYPE_CHECKING:
    from ...core.config import SimConfig
    from ...core.types import SpatialIndex


# ── P3.2: 5 Lissajous blob anchors ─────────────────────────────────

def _compute_anchors(
    t: float, C: np.ndarray, U: float
) -> np.ndarray:
    """Compute 5 Lissajous blob anchors B₀–B₄ at time t.

    Each anchor is a 3D point offset from centre C by a sinusoidal path
    scaled by unit scale U.  Returns shape (5, 3).
    """
    return C + np.array([
        [
            np.sin(t * 0.19) * 0.74,
            np.sin(t * 0.31 + 0.8) * 0.48,
            np.cos(t * 0.23) * 0.62,
        ],
        [
            np.cos(t * 0.17 + 1.6) * 0.68,
            np.sin(t * 0.37 + 2.1) * 0.54,
            np.sin(t * 0.29 + 0.4) * 0.72,
        ],
        [
            np.sin(t * 0.27 + 2.7) * 0.58,
            np.cos(t * 0.21 + 1.2) * 0.42,
            np.cos(t * 0.33 + 2.5) * 0.68,
        ],
        [
            np.cos(t * 0.24 + 3.4) * 0.70,
            np.sin(t * 0.33 + 0.6) * 0.50,
            np.sin(t * 0.18 + 1.4) * 0.58,
        ],
        [
            np.sin(t * 0.14 + 4.4) * 0.48,
            np.sin(t * 0.47 + 2.3) * 0.62,
            np.cos(t * 0.26 + 4.0) * 0.70,
        ],
    ], dtype=np.float32) * U


# ── P3.2: Cyclic phase weights ─────────────────────────────────────

def _compute_targets(
    seeds: np.ndarray,  # shape (n_active,)
    t: float,
    anchors: np.ndarray,  # shape (5, 3)
) -> np.ndarray:
    """Compute per-bird blended targets T_legacy via cyclic phase weights.

    φ_i = fract(seed_i · 3.71 + t · 0.022 + sin(seed_i · 19 + t · 0.11) · 0.09)
    c_k ∈ {0, 0.2, 0.4, 0.6, 0.8}
    w_k = max(0, 1 − wrap_dist(φ_i, c_k) · 7.5)²
    T_legacy_i = (Σ_k B_k · w_{i,k}) / Σ_k w_{i,k}

    Returns T_legacy: shape (n_active, 3).
    """
    n = len(seeds)

    raw_phi = (
        seeds * 3.71 + t * 0.022
        + np.sin(seeds * 19.0 + t * 0.11) * 0.09
    )
    phi = (raw_phi - np.floor(raw_phi)).reshape(n, 1).astype(np.float32)

    c_k = np.array([0.0, 0.2, 0.4, 0.6, 0.8], dtype=np.float32)

    dist = np.abs(phi - c_k[np.newaxis, :])
    wrap_dist = np.minimum(dist, 1.0 - dist)

    w = np.maximum(0.0, 1.0 - wrap_dist * 7.5) ** 2  # float64 for numerical stability

    w_sum = w.sum(axis=1, keepdims=True)
    w_sum[w_sum == 0.0] = 1.0
    w_norm = w / w_sum

    return np.dot(w_norm, anchors).astype(np.float32)  # np.dot avoids Apple Silicon BLAS bug


# ── P3.2: Per-bird phase helper ────────────────────────────────────

def _compute_phases(seeds: np.ndarray, t: float) -> np.ndarray:
    """Compute per-bird cyclic phase φ_i ∈ [0,1) from seeds and time.

    Returns shape (n_active,) float32.
    """
    raw_phi = (
        seeds * 3.71 + t * 0.022
        + np.sin(seeds * 19.0 + t * 0.11) * 0.09
    )
    return (raw_phi - np.floor(raw_phi)).astype(np.float32)


# ── Hash helper for deterministic per-bird values ──────────────────

def _hash01(x: np.ndarray) -> np.ndarray:
    """fract(sin(x·12.9898)·43758.5453) — deterministic hash to [0,1)."""
    return (np.sin(x * 12.9898) * 43758.5453) % 1.0


# ── S2.A3: dedicated per-group anchor(t, gs) formula ───────────────

def _group_anchor(t: np.ndarray, gs: np.ndarray, C: np.ndarray, U: float) -> np.ndarray:
    """anchor(t, gs) — S2.A3's dedicated leader/chaser anchor, distinct
    from S2.A2's 5 fixed Lissajous blob anchors (_compute_anchors).

    t and gs may be per-bird (n,) arrays (vectorised — every bird can
    evaluate this at its own lagged time and its own/a neighbouring
    group's phase without a Python loop).

    Returns (n, 3) float32.
    """
    phase = gs * 2.0 * np.pi
    return C + np.column_stack([
        np.cos(phase + t * 0.21) * 0.50 + np.sin(t * 0.13 + phase * 2.3) * 0.16,
        np.sin(phase * 1.7 + t * 0.19) * 0.34 + np.cos(t * 0.11 + phase) * 0.12,
        np.sin(phase + t * 0.16) * 0.46 + np.cos(t * 0.23 + phase * 1.4) * 0.14,
    ]).astype(np.float32) * U


# ── P3.3: Leader/chaser groups ─────────────────────────────────────

def _compute_leader_chaser(
    seeds: np.ndarray,
    t: float,
    T_legacy: np.ndarray,
    anchors: np.ndarray,
    U: float,
    chase_strength: float,
    sep: float,
    num_groups: int = 7,
    leader_fraction: float = 0.16,
    C: np.ndarray | None = None,
    wander_heading: np.ndarray | None = None,
) -> np.ndarray:
    """Compute blended targets T with leader/chaser dynamics (P3.3/S2.A3).

    Returns (n_active, 3) float32 — the final targets after
    leader/chaser blending: T = lerp(T_legacy, chase_target, chase_strength).

    chase_strength=0 returns T_legacy unchanged (backward compat with P3.2).

    S2.A3: uses the dedicated anchor(t, gs) formula (_group_anchor),
    not S2.A2's 5 fixed blob anchors; blends a primary anchor at the
    bird's own group with a secondary anchor at the next group over
    (sec_mix); every bird evaluates its own per-bird lagged time
    (no per-group averaging); leaders steer toward wander_heading
    rather than an approximated group-phase direction.
    """
    n = len(seeds)
    if chase_strength <= 0.0 or n < 2:
        return T_legacy.astype(np.float32)

    cs = chase_strength  # shorthand
    ng = max(1, int(num_groups))
    if C is None:
        C = np.zeros(3, dtype=np.float32)

    # ── seed groups (C3: field_num_groups) ──
    group_seed = np.floor(seeds * ng) / ng       # shape (n,) ∈ {0, 1/ng, …, (ng-1)/ng}
    gs = group_seed                                # shorthand
    group_phase = gs * 2.0 * np.pi                 # per-group base phase

    # ── Per-bird lag ──
    lag = _hash01(seeds + 9.17) * (1.1 + cs * 2.4)  # shape (n,)

    # ── Leader classification (C3: field_leader_fraction, ~16% default) ──
    is_leader = _hash01(seeds + 5.91) >= (1.0 - leader_fraction)  # shape (n,) bool

    # ── Slot rank within each group ──
    # Assign each bird a stable rank within its group via seed sorting.
    # Uniquify seeds per group by adding tiny group offset.
    sort_keys = seeds + gs * 1e-4
    group_ids = (gs * ng).astype(np.int32)          # 0..ng-1
    slot = np.zeros(n, dtype=np.float32)
    for gid in range(ng):
        mask = group_ids == gid
        if mask.sum() == 0:
            continue
        order = np.argsort(sort_keys[mask])
        slot[mask] = np.arange(mask.sum(), dtype=np.float32)[np.argsort(order)]

    # ── Golden-angle stratified shells ──
    ga = 2.39996323
    y = 1.0 - 2.0 * ((slot + 0.5) * 0.618034 + gs * 0.13) % 1.0
    ring = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = slot * ga + group_phase
    shell = ((slot + 1.0) * 0.754877 % 1.0) ** (1.0 / 3.0)
    radius = (0.16 + shell * 0.34) * (0.68 + cs * 0.34) * (0.92 + sep * 0.045) * U
    breath = 1.0 + np.sin(t * 0.13 + gs * 12.0) * 0.035

    offset = np.column_stack([
        np.cos(theta) * ring,
        y,
        np.sin(theta) * ring,
    ]).astype(np.float32) * (radius * breath)[:, np.newaxis]

    # ── S2.A3: per-bird lagged primary + secondary anchors ──
    lagged_t = np.clip(t - lag, 0.0, None)
    primary = _group_anchor(lagged_t, gs, C, U)
    secondary_gs = (gs + 1.0 / ng) % 1.0
    secondary = _group_anchor(lagged_t, secondary_gs, C, U)
    sec_mix = (_hash01(seeds + 3.33) * 0.5)[:, np.newaxis]
    anchor_primary = primary * (1.0 - sec_mix) + secondary * sec_mix

    # ── Chase target: blended anchor + stratified shell offset ──
    chase_target = anchor_primary + offset

    # ── Leaders: override target with a wander-heading steering point ──
    if is_leader.any():
        lead_dist = (0.18 + _hash01(seeds[is_leader] + 7.1) * 0.18) * U
        if wander_heading is not None:
            heading = wander_heading.reshape(1, 3).astype(np.float32)
        else:
            # No Wander extension active — degenerate to the blend
            # centre C (no directional bias to steer toward).
            heading = np.zeros((1, 3), dtype=np.float32)
        chase_target[is_leader] = C + heading * lead_dist[:, np.newaxis]

    # ── Blend: T = lerp(T_legacy, chase_target, chase_strength) ──
    return (T_legacy * (1.0 - cs) + chase_target * cs).astype(np.float32)


# ── P3.4: Shell force + inner cavity ───────────────────────────────

def _compute_shell_force(
    positions_active: np.ndarray,
    targets: np.ndarray,
    seeds: np.ndarray,
    t: float,
    U: float,
    cohesion: np.ndarray | float,
    chase_strength: float,
    sep: np.ndarray | float,
    shell_influence: float,
    shell_radius_base: float = 0.32,
    inner_radius_factor: float = 0.28,
) -> np.ndarray:
    """Compute per-bird shell force + inner cavity push-out (P3.4).

    Returns (n_active, 3) float32 force to add to accelerations.

    R_blob,i = (shell_radius_base + 0.08·sin(seed·41 + t·0.29) + sin(φ·2π + t·0.17)·0.05)·U
    F_shell = −d̂·(d − R_blob)·coh_i·1.35·(1−chase)·shell_influence
    Inner cavity: if d < inner, push out.

    P3.8: cohesion and sep can be per-bird arrays for blackening modulation.
    """
    n = len(seeds)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # Coerce scalars to per-bird arrays
    coh_arr = np.broadcast_to(np.asarray(cohesion, dtype=np.float32), (n,))
    sep_arr = np.broadcast_to(np.asarray(sep, dtype=np.float32), (n,))

    phases = _compute_phases(seeds, t)
    cs = chase_strength

    # Per-bird blob radius (oscillating) — C3: field_shell_radius_base
    R_blob = (
        shell_radius_base
        + np.sin(seeds * 41.0 + t * 0.29) * 0.08
        + np.sin(phases * 2.0 * np.pi + t * 0.17) * 0.05
    ) * U

    to_target = positions_active - targets
    d = np.linalg.norm(to_target, axis=1)
    safe_d = np.maximum(d, 1e-6)
    d_hat = to_target / safe_d[:, np.newaxis]

    # ── Shell force: pull toward / push away from target at R_blob ──
    shell_mag = (d - R_blob) * coh_arr * 1.35 * (1.0 - cs) * shell_influence
    F_shell = -d_hat * shell_mag[:, np.newaxis]

    # ── Inner cavity: push out when inside the inner floor ──
    # C3: field_inner_radius_factor
    inner = R_blob * (inner_radius_factor + (1.0 - cs) * 0.18 + sep_arr * 0.012)
    inside = d < inner
    if inside.any():
        F_expand = np.zeros_like(F_shell)
        F_expand[inside] = d_hat[inside] * (inner[inside] - d[inside])[:, np.newaxis] * sep_arr[inside, np.newaxis] * 1.4
        F_shell += F_expand

    return F_shell.astype(np.float32)


# ── S2.A5: Target-pull term ──────────────────────────────────────────

def _compute_target_pull(
    positions_active: np.ndarray,
    targets: np.ndarray,
    U: float,
    cohesion: np.ndarray | float,
    target_pull: float,
) -> np.ndarray:
    """Compute direct target-pull force (S2.A5).

    F_target_pull = (T−p)/U · coh · target_pull

    A direct, non-oscillating pull toward each bird's per-bird target —
    distinct from the shell force's oscillating equilibrium-radius
    behaviour (P3.4), which pulls toward a moving R_blob shell rather
    than straight at T. Resolves the previously-dead `field_target_pull`
    config field (Part III C3 deferred it here pending this exact formula).

    P3.8: cohesion can be a per-bird array for blackening modulation
    (same convention as _compute_shell_force).

    Returns (n_active, 3) float32.
    """
    n = len(positions_active)
    if n == 0 or target_pull <= 0.0:
        return np.zeros((n, 3), dtype=np.float32)

    coh_arr = np.broadcast_to(np.asarray(cohesion, dtype=np.float32), (n,))
    return (
        (targets - positions_active) / max(U, 1e-6)
        * coh_arr[:, np.newaxis] * target_pull
    ).astype(np.float32)


# ── P3.5: Slot repulsion (quadratic kernel) ────────────────────────

def _compute_slot_repulsion(
    positions: np.ndarray,
    active: np.ndarray,
    n_active: int,
    U: float,
    separation: float,
    chase_strength: float,
) -> np.ndarray:
    """Compute slot repulsion with quadratic kernel (P3.5).

    Returns (N_capacity, 3) float32 force added IN-PLACE style.
    Callers should add the returned force to accelerations.

    Kernel: ((r_slot − d) / r_slot)² when d < r_slot, zero otherwise.
    r_slot = (0.07 + separation·0.02)·U at offsets ±{1, 7, 31}.
    gain = separation · (0.14 + chase_strength·0.05).

    S2.A5: pairs are mod-wrapped around the active-bird index ring —
    bird i pairs with bird (i+offset) mod n_active for every i, so birds
    near the start and end of the active-index ordering interact too
    (previously the last `offset` birds had no partner at all for that
    offset, an artefact of the index ordering rather than physical
    distance).
    """
    N = positions.shape[0]
    F = np.zeros((N, 3), dtype=np.float32)
    if separation <= 0.0 or n_active < 2:
        return F

    r_slot = (0.07 + separation * 0.02) * U
    gain = separation * (0.14 + chase_strength * 0.05)
    active_idx = np.where(active)[0]

    for offset in [1, 7, 31]:
        if offset >= n_active:
            continue
        src = active_idx
        dst = active_idx[(np.arange(n_active) + offset) % n_active]
        diffs = positions[dst] - positions[src]  # vector FROM src TO dst
        d = np.linalg.norm(diffs, axis=1)
        within = d < r_slot
        if not within.any():
            continue
        # Quadratic kernel: ((r_slot−d)/r_slot)²
        kernel = ((r_slot - d[within]) / r_slot) ** 2
        away_subset = diffs[within] / (d[within, np.newaxis] + 1e-6)
        force = np.zeros_like(diffs)
        force[within] = away_subset * (kernel * gain)[:, np.newaxis]
        # Action-reaction: push apart
        F[src[within]] -= force[within]
        F[dst[within]] += force[within]

    return F


# ── P3.6: Tangential orbital ───────────────────────────────────────

def _compute_tangential(
    positions_active: np.ndarray,
    targets: np.ndarray,
    seeds: np.ndarray,
    t: float,
    alignment: float,
    chase_strength: float,
    tangent_pull: float,
) -> np.ndarray:
    """Compute tangential orbital force about the blob axes (P3.6).

    axis_i = normalize(sin(t·0.13+seed·7), 0.72+sin(t·0.19+seed·3)·0.28,
                       cos(t·0.17+seed·5))
    F_tan = normalize(axis × (p−T)) · align · 0.035 · (1−chase) · tangent_pull

    Returns (n_active, 3) float32.
    """
    n = len(seeds)
    if n == 0 or tangent_pull <= 0.0:
        return np.zeros((n, 3), dtype=np.float32)

    axis = np.column_stack([
        np.sin(t * 0.13 + seeds * 7.0),
        0.72 + np.sin(t * 0.19 + seeds * 3.0) * 0.28,
        np.cos(t * 0.17 + seeds * 5.0),
    ]).astype(np.float32)

    to_target = positions_active - targets
    cross = np.cross(axis, to_target)
    cross_norm = np.linalg.norm(cross, axis=1, keepdims=True)
    cross_norm = np.maximum(cross_norm, 1e-6)
    F_tan = cross / cross_norm * alignment * 0.035 * (1.0 - chase_strength) * tangent_pull

    return F_tan.astype(np.float32)


# ── P3.6: Buoyancy (z-up) ──────────────────────────────────────────

def _compute_buoyancy(
    positions_active: np.ndarray,
    targets: np.ndarray,
    seeds: np.ndarray,
    t: float,
    U: float,
    flow: float,
) -> np.ndarray:
    """Compute buoyancy force in the z-up direction (P3.6).

    d = ||p − T||
    F_z += (sin(d·8/U − t·1.1 + seed·17)·0.09 + (T_z−p_z)/U·0.24) · (0.75 + flow·0.25)

    Returns (n_active, 3) float32 with only z-component non-zero.
    """
    n = len(seeds)
    if n == 0:
        return np.zeros((n, 3), dtype=np.float32)

    to_target = targets - positions_active
    d = np.linalg.norm(positions_active - targets, axis=1)

    buoyancy_mag = (
        np.sin(d * 8.0 / U - t * 1.1 + seeds * 17.0) * 0.09
        + to_target[:, 2] / U * 0.24
    ) * (0.75 + flow * 0.25)

    F = np.zeros((n, 3), dtype=np.float32)
    F[:, 2] = buoyancy_mag
    return F


# ── P3.6: Curl flow ────────────────────────────────────────────────

def _compute_curl_flow(
    positions_active: np.ndarray,
    C: np.ndarray,
    seeds: np.ndarray,
    t: float,
    U: float,
    flow: float,
    flow_pull: float,
) -> np.ndarray:
    """Compute normalized curl flow force (P3.6).

    S2.B11: delegates to the shared L0 primitive (_base.py::curl_flow,
    also consumed by SpatialMode's flow_weight) and applies field mode's
    own flow*flow_pull gain on top.

    F_flow = curl_flow(...) · flow · flow_pull

    Returns (n_active, 3) float32.
    """
    n = len(seeds)
    if n == 0 or flow_pull <= 0.0:
        return np.zeros((n, 3), dtype=np.float32)

    return (curl_flow(positions_active, C, seeds, t, U) * flow * flow_pull).astype(np.float32)


# ── P3.6: Fold noise ───────────────────────────────────────────────

def _compute_fold_noise(
    positions_active: np.ndarray,
    C: np.ndarray,
    seeds: np.ndarray,
    t: float,
    U: float,
    flow: float,
    flow_pull: float,
    ripple_envelope_sum: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Compute fold noise force (P3.6).

    q = (p−C)/U
    fold = (sin(q_y·3.7 + t·0.73 + seed) + cos(q_z·2.9 − t·0.51),
            sin(q_z·3.1 − t·0.67 + seed) − cos(q_x·2.4 + t·0.43),
            sin(q_x·3.3 + t·0.59 + seed) + cos(q_y·2.6 − t·0.47))
    F_fold = fold · flow · flow_pull · ripple_envelope_sum

    D10: ripple_envelope_sum may be a per-bird (n_active,) array; it is
    broadcast column-wise against the (n_active, 3) fold vector.

    Returns (n_active, 3) float32.
    """
    n = len(seeds)
    if n == 0 or flow_pull <= 0.0:
        return np.zeros((n, 3), dtype=np.float32)

    if isinstance(ripple_envelope_sum, np.ndarray) and ripple_envelope_sum.ndim == 1:
        ripple_envelope_sum = ripple_envelope_sum[:, np.newaxis]

    q = (positions_active - C) / max(U, 1e-6)
    fold = np.column_stack([
        np.sin(q[:, 1] * 3.7 + t * 0.73 + seeds)
        + np.cos(q[:, 2] * 2.9 - t * 0.51),
        np.sin(q[:, 2] * 3.1 - t * 0.67 + seeds)
        - np.cos(q[:, 0] * 2.4 + t * 0.43),
        np.sin(q[:, 0] * 3.3 + t * 0.59 + seeds)
        + np.cos(q[:, 1] * 2.6 - t * 0.47),
    ]).astype(np.float32)

    return (fold * flow * flow_pull * ripple_envelope_sum).astype(np.float32)


# ── P3.6: Viscous drag ─────────────────────────────────────────────

def _compute_viscous_drag(
    velocities_active: np.ndarray,
    chase_strength: float,
    flow: float,
) -> np.ndarray:
    """Compute viscous drag force (P3.6).

    F_drag = −v · chase_strength · (0.08 + flow·0.02)

    Returns (n_active, 3) float32.
    """
    return (-velocities_active * chase_strength * (0.08 + flow * 0.02)).astype(np.float32)


# ── P3.6: Drift alignment to wander_heading ────────────────────────

def _compute_drift_alignment(
    velocities_active: np.ndarray,
    wander_heading: np.ndarray | None,
    v0: float,
    alignment: float,
    drift_pull: float,
) -> np.ndarray:
    """Compute drift alignment force toward wander heading (P3.6).

    F_drift = (wander_heading·v0 − v) · alignment · drift_pull

    If wander_heading is None, returns zero force.

    Returns (n_active, 3) float32.
    """
    if wander_heading is None or drift_pull <= 0.0:
        return np.zeros_like(velocities_active)
    target_vel = wander_heading.reshape(1, 3).astype(np.float32) * v0
    return ((target_vel - velocities_active) * alignment * drift_pull).astype(np.float32)


# ── C3: Deterministic per-bird noise (field_noise) ──────────────────

def _compute_field_noise(
    seeds: np.ndarray,
    t: float,
    noise: float,
) -> np.ndarray:
    """Compute deterministic per-bird noise jitter (C3: field_noise).

    Uses the seed_noise3 L0 atom (bounded ±0.18/axis) scaled to
    ±noise/axis so `field_noise` reads as a direct force amplitude.

    Returns (n_active, 3) float32.
    """
    n = len(seeds)
    if n == 0 or noise <= 0.0:
        return np.zeros((n, 3), dtype=np.float32)
    return (seed_noise3(seeds, t) * (noise / 0.18)).astype(np.float32)


# ── P3.11: Grid-mode separation normalization ──────────────────────

def _compute_grid_sep_normalized(
    positions_active: np.ndarray,
    separation: float,
    neighbour_count: int,
) -> float:
    """Return separation factor normalized by neighbour count (P3.11).

    F_sep_grid = −(separation / max(1, neighbour_count)) · Σ r̂/d²

    Returns the normalization factor to multiply the raw separation force.
    """
    return separation / max(1, neighbour_count)


# ── P3.12: Floating boundary ───────────────────────────────────────

def _compute_floating_boundary(
    positions_active: np.ndarray,
    C: np.ndarray,
    R_blobs: np.ndarray,
    U: float,
    mu: float = 0.05,
) -> np.ndarray:
    """Compute floating boundary containment force (P3.12).

    R_boundary = 1.45 · max_i(R_blob,i)  — floats with the blob radius.
    If ||p − C|| > R_boundary:
        F = −μ · r̂ / max(‖p−C‖ − R_boundary, 0.05·R_boundary)

    S2.A5 assessment: the roadmap's literal spec form is a linear force,
    `F_mag = (d − 1.45U)·1.6` (increasing with overshoot). Blessed the
    current asymptotic form instead — deliberate, not an oversight:
    - The alternative spec reading this replaced, `μ·r̂/(R_boundary−d)`,
      is singular and sign-flips exactly at the boundary crossing
      (denominator → 0 then negative), giving a discontinuous force.
    - The current form — overshoot in the denominator, strongest right
      at the boundary and *weakening* further out — is the same
      asymptotic pattern already used by `boid.py::_sphere_soft_asymptotic`
      (`Δv = −μ·r̂/max(R−r, 0.05R)`, strongest approaching the boundary
      from inside). Mirroring an established, tested pattern rather than
      introducing a second, differently-shaped containment law.
    - A literal increasing-with-distance linear force would need its own
      re-derivation of `mu`'s scale (0.05 tuned for the asymptotic form)
      to avoid a golden-changing magnitude jump for no physical benefit.

    Returns (n_active, 3) float32.
    """
    n = R_blobs.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)

    R_boundary = 1.45 * float(np.max(R_blobs))
    if R_boundary <= 0.0:
        return np.zeros((n, 3), dtype=np.float32)

    to_centre = positions_active - C
    dist = np.linalg.norm(to_centre, axis=1)
    outside = dist > R_boundary
    if not outside.any():
        return np.zeros((n, 3), dtype=np.float32)

    F = np.zeros((n, 3), dtype=np.float32)
    d_out = dist[outside]
    r_hat = to_centre[outside] / d_out[:, np.newaxis]
    overshoot = d_out - R_boundary
    denominator = np.maximum(overshoot, 0.05 * R_boundary)
    F[outside] = -mu * r_hat / denominator[:, np.newaxis]

    return F.astype(np.float32)


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
