"""P5 — Angle force mode: axis-angle heading steering with Rodrigues rotation.

Unified neighbour modes (flee/align+coh/coh-only), adaptive speed,
cardinal-axis edge avoidance, per-frame heading jitter, incremental
spatial grid, and body-unit scale invariance.

P5.1–P5.7 — L1 assembly: depends on core/types.py (rotate_about,
safe_normalize, min_image) and physics/flock (read-only position query).

Modularity pass 10: the per-bird steering loop (jitter -> neighbour-mode
selection -> edge avoidance -> adaptive speed -> Rodrigues rotation) is
fully vectorised — no Python per-bird loop remains. This is what
neighbor_selection.py's own docstring flagged as future work ("would
require restructuring that loop, which this pure-extraction pass
deliberately avoids — see the modularity-pass-2 plan"): unlike that
pass and Modularity pass 9 (composeForces), this one is NOT a
behaviour-preserving pure extraction. The RNG draw order changes (all
stationary-heading fallbacks and all jitter draws are now each drawn in
one batched call, instead of interleaved per-bird), so results differ
from before at the bit level even for the same seed — golden_angle.npz
and golden_angle_sphere.npz were regenerated for this change. All
*deterministic* math (neighbour-mode thresholds, edge-avoidance
geometry, adaptive speed law, Rodrigues rotation formula) is unchanged,
verified against a per-bird reference implementation in
test_angle_vectorization.py.

Known pre-existing quirk, preserved as-is (not introduced or fixed by
this pass): neighbour-index sentinels use `> 0`, not `>= 0`, so a
global bird index of exactly 0 can never be treated as anyone's
neighbour. Out of scope here — flagged, not fixed, to keep this a
loop-restructuring change only, not a behaviour fix bundled in with it.

Modularity pass 11: this mode's adaptive deficit-based speed law
(new_speed, computed in Stage 3) was silently discarded by
flock.integrate()'s speed_mode="fixed" post-processing — see
stash_target_speed()'s docstring (physics/forces/_base.py) for the
full explanation. Fixed by stashing new_speed there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...core.types import min_image, safe_normalize
from ..plugins.force_mode import ForceMode, register
from ._base import stash_target_speed

if TYPE_CHECKING:
    from ...core.config import SimConfig
    from ...core.types import SpatialIndex


def _rotate_about_batch(
    v: np.ndarray, k: np.ndarray, angle: np.ndarray,
) -> np.ndarray:
    """Rodrigues rotation with a PER-ROW axis and PER-ROW angle.

    core.types.rotate_about only supports a single shared (3,) axis and
    a scalar/shared angle applied to all rows (its (N,3) branch does
    np.dot(v, k), which is a matrix product, not a row-wise dot — it
    was never meant for a genuinely per-row axis). Angle mode's own
    steering needs both to vary per bird (each bird rotates toward its
    own target, by its own gated turn angle), so this is a local,
    fully-general variant rather than a change to the shared L0 atom.

    Args:
        v: (N, 3) vectors to rotate
        k: (N, 3) per-row rotation axes (assumed already unit length)
        angle: (N,) per-row rotation angles in radians

    Returns:
        (N, 3) rotated vectors
    """
    cos_a = np.cos(angle)[:, np.newaxis]
    sin_a = np.sin(angle)[:, np.newaxis]
    dot_vk = np.sum(v * k, axis=1, keepdims=True)
    return v * cos_a + np.cross(k, v) * sin_a + k * dot_vk * (1 - cos_a)


@register("angle")
class AngleMode(ForceMode):
    """Angle-based boids steering — Rodrigues heading rotation.

    Each bird stores its heading as its velocity direction. Per frame:
      1. Jitter heading by ±jitter_deg° about a random axis
      2. Compute target direction from neighbours (flee/align+coh/coh-only)
      3. Steer toward target via Rodrigues rotation, capped at turnRate·dt
      4. Apply dead-zone: no turn if angular error < turn_threshold°
      5. Edge handling: steer away from domain boundaries within margin
      6. Adaptive speed: faster when isolated, slower in dense groups

    P5.6: Per-bird _last_cell tracked for incremental spatial grid
    updates — only re-files birds that cross cell boundaries.

    Modularity pass 10: all of the above is vectorised across the
    active set — no per-bird Python loop.
    """

    needs_index = True
    speed_mode = "fixed"  # D2: sets velocities[i] = heading * adaptive per-bird speed directly

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
        """Angle-mode force computation — mutates accelerations and velocities."""
        n_active = active.sum()
        if n_active == 0:
            return

        # ── Config knobs ──
        b = config.boid_size
        turn_rate = np.radians(config.angle.turn_rate)
        max_turn_rate = np.radians(config.angle.max_turn_rate)
        turn_threshold = np.radians(config.angle.turn_threshold)
        jitter_deg = config.angle.jitter_deg
        margin = config.boundary.boundary_margin
        base_speed = config.angle.base_speed
        sep_r = config.angle.sep_radius_bodies * b
        align_r = config.angle.align_radius_bodies * b
        range_r = config.angle.range_radius_bodies * b
        n_neighbors = config.angle.angle_neighbors
        speed_mode = config.angle.angle_speed_mode
        deficit_cap = float(n_neighbors * n_neighbors)  # S2.C3: (n_neighbors)² cap
        border_mode = config.boundary.boundary_mode
        width = config.width
        height = config.height
        depth = config.depth
        dt = 1.0 / max(config.fps, 1)

        # S2.B8: Coherence gate — reduce steering responsiveness for small
        # flocks at dusk.  Only gates alignment/cohesion steering; flee and
        # edge avoidance always use full turn_rate (safety-critical).
        # Consistent with SpatialMode/ProjectionMode which gate alignment/
        # cohesion weights but leave separation at full strength.
        coherence = getattr(config, '_coherence_factor', 1.0)

        # Angle mode steers via direct velocity assignment (Rodrigues rotation)
        # rather than acceleration-based physics. Do NOT zero accelerations here —
        # extensions (ecology roost pull, etc.) may have written forces during
        # pre_step that should survive through integrate().  boid.integrate()
        # resets accelerations at the end of each frame anyway.

        # Use spatial index for neighbour queries
        if index is None or not index.ready:
            return

        active_idx = np.where(active)[0]

        # ── P5.6: Incremental spatial grid update ──
        # D14: _last_cell is per-index (not class-level) so two engines
        # with different N don't corrupt each other's cell tracking.
        n_total = len(active)
        last_cell = getattr(index, '_angle_last_cell', None)
        if last_cell is None or last_cell.shape[0] < n_total:
            last_cell = np.full((n_total, 3), -1, dtype=np.int32)
            index._angle_last_cell = last_cell  # type: ignore[attr-defined]

        from ...physics.flock import SpatialHashGrid
        if isinstance(index, SpatialHashGrid):
            index.incremental_rebuild(positions, active, last_cell)

        # ── Batch k-NN query ──
        k = min(n_neighbors + 1, len(active_idx))
        active_pos = positions[active_idx]

        tree = getattr(index, 'tree', None)
        if tree is None:
            from scipy.spatial import cKDTree
            tree = cKDTree(active_pos)

        _, compacted = tree.query(active_pos, k=k, workers=-1)

        if compacted.ndim == 1:
            compacted = compacted.reshape(-1, 1)

        # nbr_idx: (n_active, k-1) — skip self (column 0). 0 is the
        # padding sentinel (pre-existing quirk — see module docstring).
        nbr_idx = np.zeros((n_active, k - 1), dtype=np.int32)
        for j in range(n_active):
            row = compacted[j, 1:k]
            nbr_idx[j, :len(row)] = active_idx[row]

        box = np.array([width, height, depth], dtype=np.float32)
        toroidal = border_mode == "toroidal"

        # ══════════════════════════════════════════════════════════
        # Stage 1: resolve heading (velocity direction, or a random
        # fallback for stationary birds — one batched rng draw for all
        # stationary birds, not interleaved per-bird as before).
        # ══════════════════════════════════════════════════════════
        velocities_a = velocities[active_idx]
        positions_a = positions[active_idx]
        speeds = np.linalg.norm(velocities_a, axis=1)
        stationary = speeds < 1e-6

        hdg = np.zeros((n_active, 3), dtype=np.float32)
        moving = ~stationary
        hdg[moving] = velocities_a[moving] / speeds[moving, np.newaxis]
        n_stationary = int(stationary.sum())
        if n_stationary > 0:
            rand_dirs = rng.uniform(-1, 1, (n_stationary, 3)).astype(np.float32)
            hdg[stationary] = safe_normalize(rand_dirs)

        # ── P5.5: Heading jitter — one batched rng draw for all active
        # birds when enabled, not interleaved per-bird as before. ──
        if jitter_deg > 0:
            jitter_rad = np.radians(
                rng.uniform(-jitter_deg, jitter_deg, n_active)
            ).astype(np.float32)
            jitter_axis_raw = rng.uniform(-1, 1, (n_active, 3)).astype(np.float32)
            jitter_axis = safe_normalize(jitter_axis_raw)
            hdg = _rotate_about_batch(hdg, jitter_axis, jitter_rad)

        # ══════════════════════════════════════════════════════════
        # Stage 2: P5.2 unified neighbour modes (flee / align+coh / coh-only)
        # ══════════════════════════════════════════════════════════
        valid_nbr = nbr_idx > 0  # (n_active, k-1) — bug-compatible sentinel
        has_any_nbr = valid_nbr.any(axis=1)
        n_valid_nbr = valid_nbr.sum(axis=1)

        safe_nbr_idx = np.where(valid_nbr, nbr_idx, 0)
        nbr_pos = positions[safe_nbr_idx]   # (n_active, k-1, 3)
        nbr_vel = velocities[safe_nbr_idx]  # (n_active, k-1, 3)

        diffs = nbr_pos - positions_a[:, np.newaxis, :]
        if toroidal:
            diffs = min_image(diffs, box)
        dists = np.linalg.norm(diffs, axis=2)
        dists_masked = np.where(valid_nbr, dists, np.inf)

        if dists_masked.shape[1] == 0:
            # k-1 == 0 (a single active bird has no k-NN columns at all —
            # the cKDTree query itself returns no neighbours to skip).
            # .min(axis=1)/.argmin(axis=1) can't reduce an empty axis;
            # the original per-bird loop's `if len(nbrs) > 0:` guard
            # never entered the neighbour block at all in this case,
            # which this reproduces via has_any_nbr staying all-False.
            nearest_dist = np.full(n_active, np.inf, dtype=np.float32)
            nearest_glob_idx = np.zeros(n_active, dtype=np.int32)
        else:
            nearest_dist = dists_masked.min(axis=1)
            nearest_col = dists_masked.argmin(axis=1)
            nearest_glob_idx = nbr_idx[np.arange(n_active), nearest_col]

        is_fleeing = has_any_nbr & (nearest_dist < sep_r)
        is_align = has_any_nbr & (nearest_dist >= sep_r) & (nearest_dist < align_r)
        is_cohere = has_any_nbr & (nearest_dist >= align_r) & (nearest_dist < range_r)
        has_nbr_target = is_fleeing | is_align | is_cohere

        # --- Flee: steer directly away from nearest neighbour ---
        to_nbr = positions[nearest_glob_idx] - positions_a
        if toroidal:
            to_nbr = min_image(to_nbr, box)
        flee_target = safe_normalize(-to_nbr)

        # --- Shared centroid (used by both align+coh and coh-only) ---
        nbr_count_safe = np.maximum(n_valid_nbr, 1)[:, np.newaxis].astype(np.float32)
        centroid = (
            np.sum(nbr_pos * valid_nbr[:, :, np.newaxis], axis=1) / nbr_count_safe
        )
        if toroidal:
            to_centroid = min_image(centroid - positions_a, box)
            centroid = positions_a + to_centroid
        c_hat = safe_normalize(centroid - positions_a)
        cohere_target = safe_normalize(centroid - positions_a)

        # --- Align+coh: normalize(centroid direction + mean heading) ---
        vn_speed = np.linalg.norm(nbr_vel, axis=2)
        vn_valid = valid_nbr & (vn_speed > 1e-6)
        vn_unit = np.zeros_like(nbr_vel)
        vn_speed_safe = np.where(vn_speed > 1e-6, vn_speed, 1.0)
        vn_unit = np.where(
            vn_valid[:, :, np.newaxis], nbr_vel / vn_speed_safe[:, :, np.newaxis], 0.0,
        )
        m_hat_raw = vn_unit.sum(axis=1)
        m_hat = safe_normalize(m_hat_raw)
        align_target = safe_normalize(c_hat + m_hat)

        target = np.zeros((n_active, 3), dtype=np.float32)
        target[is_fleeing] = flee_target[is_fleeing]
        target[is_align] = align_target[is_align]
        target[is_cohere] = cohere_target[is_cohere]

        # ══════════════════════════════════════════════════════════
        # Stage 3: P5.3/S2.C3 adaptive speed law
        # ══════════════════════════════════════════════════════════
        deficit = n_neighbors - n_valid_nbr
        deficit_f = deficit.astype(np.float32)
        has_deficit = deficit > 0
        if speed_mode == "quadratic":
            bonus = np.minimum(deficit_cap, deficit_f * deficit_f)
        elif speed_mode == "softened":
            bonus = np.minimum(deficit_cap, deficit_f * deficit_f / 2.0)
        else:  # "linear" (default, P5.3 original)
            bonus = deficit_f * 5.0
        new_speed = np.where(has_deficit, base_speed + bonus, base_speed).astype(np.float32)

        # ══════════════════════════════════════════════════════════
        # Stage 4: P5.4 edge handling
        # ══════════════════════════════════════════════════════════
        edge_target = np.zeros((n_active, 3), dtype=np.float32)
        has_edge_target = np.zeros(n_active, dtype=bool)
        turn_rate_now = np.full(n_active, turn_rate, dtype=np.float32)

        if border_mode == "sphere":
            radius = config.boundary.boundary_sphere_radius
            dist_from_center = np.linalg.norm(positions_a, axis=1)
            near_edge = dist_from_center > (radius - margin)
            has_edge_target = near_edge
            edge_target[near_edge] = safe_normalize(-positions_a[near_edge])
            edge_factor = np.where(
                near_edge, 1.0 - (radius - dist_from_center) / margin, 0.0,
            ).astype(np.float32)
            turn_rate_now = turn_rate + edge_factor * (max_turn_rate - turn_rate)

        elif border_mode == "margin":
            # Cube margin: steer toward the nearest face's inward normal.
            face_candidates = np.full((n_active, 6), np.inf, dtype=np.float32)
            face_candidates[:, 0] = np.where(positions_a[:, 0] < margin, positions_a[:, 0], np.inf)
            face_candidates[:, 1] = np.where(positions_a[:, 0] > width - margin, width - positions_a[:, 0], np.inf)
            face_candidates[:, 2] = np.where(positions_a[:, 1] < margin, positions_a[:, 1], np.inf)
            face_candidates[:, 3] = np.where(positions_a[:, 1] > height - margin, height - positions_a[:, 1], np.inf)
            face_candidates[:, 4] = np.where(positions_a[:, 2] < margin, positions_a[:, 2], np.inf)
            face_candidates[:, 5] = np.where(positions_a[:, 2] > depth - margin, depth - positions_a[:, 2], np.inf)

            face_normals = np.array([
                [1.0, 0, 0], [-1.0, 0, 0],
                [0, 1.0, 0], [0, -1.0, 0],
                [0, 0, 1.0], [0, 0, -1.0],
            ], dtype=np.float32)

            face_dist = face_candidates.min(axis=1)
            face_arg = face_candidates.argmin(axis=1)
            has_edge_target = face_dist < margin
            edge_target = face_normals[face_arg]
            edge_factor = np.where(
                has_edge_target, 1.0 - face_dist / margin, 0.0,
            ).astype(np.float32)
            turn_rate_now = turn_rate + edge_factor * (max_turn_rate - turn_rate)

        # ── Combine target with edge avoidance ──
        both = has_nbr_target & has_edge_target
        edge_only = ~has_nbr_target & has_edge_target
        target[both] = safe_normalize(target[both] + edge_target[both])
        target[edge_only] = edge_target[edge_only]
        has_target = has_nbr_target | has_edge_target

        # ══════════════════════════════════════════════════════════
        # Stage 5: P5.1 steering core (Rodrigues rotation)
        # ══════════════════════════════════════════════════════════
        cos_phi = np.clip(np.sum(hdg * target, axis=1), -1.0, 1.0)
        phi = np.arccos(cos_phi)
        should_turn = has_target & (phi > turn_threshold)

        # S2.B8: Coherence gate — reduce turn_rate for small flocks at
        # dusk, but only for alignment/cohesion steering. Flee and
        # edge-only avoidance use full turn_rate (safety-critical, like
        # separation in spatial/projection modes).
        gate = (~is_fleeing) & (~edge_only) & (coherence < 1.0)
        gated_turn = np.where(gate, turn_rate_now * coherence, turn_rate_now).astype(np.float32)

        axis = np.cross(hdg, target)
        axis_norm = np.linalg.norm(axis, axis=1)
        degenerate = axis_norm < 1e-10

        # Parallel/anti-parallel fallback: cross(hdg, +X), or +Y if that's
        # also degenerate (hdg parallel to the X axis).
        x_axis = np.zeros((n_active, 3), dtype=np.float32)
        x_axis[:, 0] = 1.0
        fallback_axis_raw = np.cross(hdg, x_axis)
        fallback_norm = np.linalg.norm(fallback_axis_raw, axis=1)
        fallback_degenerate = fallback_norm < 1e-10
        y_axis = np.zeros((n_active, 3), dtype=np.float32)
        y_axis[:, 1] = 1.0
        fallback_axis = np.where(
            fallback_degenerate[:, np.newaxis], y_axis, safe_normalize(fallback_axis_raw),
        )
        axis_safe = np.where(
            degenerate[:, np.newaxis], fallback_axis, safe_normalize(axis),
        )

        turn_angle = np.minimum(phi, gated_turn * dt)
        new_hdg = _rotate_about_batch(hdg, axis_safe, turn_angle)
        hdg = np.where(should_turn[:, np.newaxis], new_hdg, hdg)

        # ── Apply new heading as velocity ──
        velocities[active_idx] = hdg * new_speed[:, np.newaxis]
        stash_target_speed(config, len(positions), active_idx, new_speed)


# Backward compatibility alias
angle_forces = AngleMode.compute
angle_forces.needs_index = True  # type: ignore[attr-defined]
