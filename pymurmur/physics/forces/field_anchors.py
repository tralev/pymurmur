"""crs48 field mode — blob anchors, target blending, leader/chaser groups.

Extracted from field.py (file-size split). P3.2/P3.3: Lissajous blob
anchors, per-bird cyclic phase weights, and the leader/chaser group
targeting used as input to the per-term forces in field_terms.py.
Pure numpy -- no pymurmur cross-module dependencies.
"""
from __future__ import annotations

import numpy as np


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
