"""P5 — Angle force mode: axis-angle heading steering with Rodrigues rotation.

Unified neighbour modes (flee/align+coh/coh-only), adaptive speed,
cardinal-axis edge avoidance, per-frame heading jitter, incremental
spatial grid, and body-unit scale invariance.

P5.1–P5.7 — L1 assembly: depends on core/types.py (rotate_about,
safe_normalize, min_image) and physics/flock (read-only position query).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...core.types import min_image, rotate_about, safe_normalize
from ._mode import ForceMode, register

if TYPE_CHECKING:
    from ...core.config import SimConfig
    from ...core.types import SpatialIndex


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

        # neighbour_idx: (n_active, k-1) — skip self (column 0)
        nbr_idx = np.zeros((len(active_idx), k - 1), dtype=np.int32)
        for j in range(len(active_idx)):
            row = compacted[j, 1:k]
            nbr_idx[j, :len(row)] = active_idx[row]

        # ── Per-bird steering loop ──
        for j, global_i in enumerate(active_idx):
            pi = positions[global_i]
            vi = velocities[global_i]
            speed = np.linalg.norm(vi)

            # Use heading = velocity direction (or random if stationary)
            if speed < 1e-6:
                hdg = safe_normalize(rng.uniform(-1, 1, 3).astype(np.float32))
            else:
                hdg = vi / speed

            # ── P5.5: Heading jitter ──
            if jitter_deg > 0:
                jitter_rad = np.radians(rng.uniform(-jitter_deg, jitter_deg))
                jitter_axis = safe_normalize(
                    rng.uniform(-1, 1, 3).astype(np.float32)
                )
                hdg = rotate_about(hdg, jitter_axis, jitter_rad)

            # ── P5.2: Unified neighbour modes ──
            nbrs = nbr_idx[j]
            nbrs = nbrs[nbrs > 0]  # filter zero sentinels

            target: np.ndarray | None = None
            is_fleeing: bool = False

            if len(nbrs) > 0:
                # Find nearest neighbour distance
                nbr_pos = positions[nbrs]
                diffs = nbr_pos - pi
                if border_mode == "toroidal":
                    box = np.array([width, height, depth], dtype=np.float32)
                    diffs = min_image(diffs, box)
                dists = np.linalg.norm(diffs, axis=1)
                nearest_dist = float(dists.min())
                nearest_idx = int(nbrs[np.argmin(dists)])

                is_fleeing = nearest_dist < sep_r

                if is_fleeing:
                    # Flee: steer directly away from nearest neighbour
                    to_nbr = positions[nearest_idx] - pi
                    if border_mode == "toroidal":
                        to_nbr = min_image(
                            to_nbr.reshape(1, 3),
                            np.array([width, height, depth], dtype=np.float32),
                        ).ravel()
                    target = safe_normalize(-to_nbr)

                elif nearest_dist < align_r:
                    # Align + cohere: toward normalize(centroid + mean heading)
                    centroid = nbr_pos.mean(axis=0)
                    if border_mode == "toroidal":
                        to_centroid = centroid - pi
                        to_centroid = min_image(
                            to_centroid.reshape(1, 3),
                            np.array([width, height, depth], dtype=np.float32),
                        ).ravel()
                        centroid = pi + to_centroid
                    c_hat = safe_normalize(centroid - pi)

                    m_hat = np.zeros(3, dtype=np.float32)
                    for nbr_i in nbrs:
                        vn = velocities[nbr_i]
                        vn_speed = np.linalg.norm(vn)
                        if vn_speed > 1e-6:
                            m_hat += vn / vn_speed
                    m_hat = safe_normalize(m_hat)

                    target = safe_normalize(c_hat + m_hat)

                elif nearest_dist < range_r:
                    # Cohere only: steer toward centroid
                    centroid = nbr_pos.mean(axis=0)
                    if border_mode == "toroidal":
                        to_centroid = centroid - pi
                        to_centroid = min_image(
                            to_centroid.reshape(1, 3),
                            np.array([width, height, depth], dtype=np.float32),
                        ).ravel()
                        centroid = pi + to_centroid
                    target = safe_normalize(centroid - pi)

            # ── P5.3/S2.C3: Adaptive speed law ──
            n_nbrs = len(nbrs)
            deficit = n_neighbors - n_nbrs
            if deficit > 0:
                if speed_mode == "quadratic":
                    new_speed = base_speed + min(deficit_cap, deficit * deficit)
                elif speed_mode == "softened":
                    new_speed = base_speed + min(deficit_cap, deficit * deficit / 2.0)
                else:  # "linear" (default, P5.3 original)
                    new_speed = base_speed + deficit * 5.0
            else:
                new_speed = base_speed

            # ── P5.4: Edge handling ──
            edge_target: np.ndarray | None = None
            if border_mode in ("margin", "sphere"):
                if border_mode == "sphere":
                    radius = config.boundary.boundary_sphere_radius
                    dist_from_center = np.linalg.norm(pi)
                    if dist_from_center > radius - margin:
                        edge_target = safe_normalize(-pi)  # toward centre
                        # Ramp turn rate toward max
                        edge_factor = float(
                            1.0 - (radius - dist_from_center) / margin
                        )
                        turn_rate_now = turn_rate + edge_factor * (
                            max_turn_rate - turn_rate
                        )
                    else:
                        turn_rate_now = turn_rate
                else:
                    # Cube margin: steer toward nearest face inward normal
                    turn_rate_now = turn_rate
                    face_dist = float("inf")
                    face_normal = np.zeros(3, dtype=np.float32)

                    if pi[0] < margin:
                        d = pi[0]
                        if d < face_dist:
                            face_dist = d
                            face_normal = np.array([1.0, 0, 0], dtype=np.float32)
                    if pi[0] > width - margin:
                        d = width - pi[0]
                        if d < face_dist:
                            face_dist = d
                            face_normal = np.array([-1.0, 0, 0], dtype=np.float32)
                    if pi[1] < margin:
                        d = pi[1]
                        if d < face_dist:
                            face_dist = d
                            face_normal = np.array([0, 1.0, 0], dtype=np.float32)
                    if pi[1] > height - margin:
                        d = height - pi[1]
                        if d < face_dist:
                            face_dist = d
                            face_normal = np.array([0, -1.0, 0], dtype=np.float32)
                    if pi[2] < margin:
                        d = pi[2]
                        if d < face_dist:
                            face_dist = d
                            face_normal = np.array([0, 0, 1.0], dtype=np.float32)
                    if pi[2] > depth - margin:
                        d = depth - pi[2]
                        if d < face_dist:
                            face_dist = d
                            face_normal = np.array([0, 0, -1.0], dtype=np.float32)

                    if face_dist < margin:
                        edge_target = face_normal
                        edge_factor = float(1.0 - face_dist / margin)
                        turn_rate_now = turn_rate + edge_factor * (
                            max_turn_rate - turn_rate
                        )
            else:
                turn_rate_now = turn_rate

            # ── Combine target with edge avoidance ──
            if edge_target is not None:
                if target is not None:
                    target = safe_normalize(target + edge_target)
                else:
                    target = edge_target

            # ── P5.1: Steering core (Rodrigues rotation) ──
            if target is not None:
                cos_phi = np.clip(np.dot(hdg, target), -1.0, 1.0)
                phi = np.arccos(cos_phi)

                # Dead zone: no turn if angular error < threshold
                if phi > turn_threshold:
                    # S2.B8: Coherence gate — reduce turn_rate for small
                    # flocks at dusk, but only for alignment/cohesion
                    # steering.  Flee and edge-only avoidance use full
                    # turn_rate (safety-critical, like separation in
                    # spatial/projection modes).
                    edge_only = (edge_target is not None
                                 and target is edge_target)
                    gated_turn = turn_rate_now
                    if not is_fleeing and not edge_only and coherence < 1.0:
                        gated_turn *= coherence

                    # Rotation axis: hdg × target ("axis" — the name k
                    # is taken by the int kNN count in this scope)
                    axis = np.cross(hdg, target)
                    axis_norm = np.linalg.norm(axis)
                    if axis_norm < 1e-10:
                        # Parallel/anti-parallel — pick arbitrary perpendicular
                        axis = safe_normalize(
                            np.cross(hdg, np.array([1.0, 0, 0], dtype=np.float32))
                        )
                        if np.linalg.norm(axis) < 1e-10:
                            axis = np.array([0, 1.0, 0], dtype=np.float32)
                    else:
                        axis = axis / axis_norm

                    turn_angle = min(phi, gated_turn * dt)
                    hdg = rotate_about(hdg, axis, turn_angle)

            # ── Apply new heading as velocity ──
            velocities[global_i] = hdg * new_speed


# Backward compatibility alias
angle_forces = AngleMode.compute
angle_forces.needs_index = True  # type: ignore[attr-defined]
