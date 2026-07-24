"""crs48 field mode — the 14 per-bird force-term implementations.

Extracted from field.py (file-size split). P3.4-P3.12: shell force,
target pull, slot repulsion, tangential orbital, buoyancy, curl flow,
fold noise, viscous drag, drift alignment, field noise, floating
boundary. Each _compute_* function here is wrapped by a thin
_term_* adapter (still in field.py) that unpacks FieldTermContext.
"""

from __future__ import annotations

import numpy as np

from ...core.types import seed_noise3
from ._base import curl_flow
from .field_anchors import _compute_phases

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
