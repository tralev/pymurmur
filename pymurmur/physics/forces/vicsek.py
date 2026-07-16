"""Vicsek 1995 mode — constant-speed angle coupling with memory term.

Phase 1.8 (P1.8): Memory term + tangent-plane noise.
  - u_noisy = normalize(u_old + sqrt(2*D*dt) * n_perp)
    where n_perp = g - (g*u_old)*u_old,  g ~ N(0, I3)
  - u_new = normalize(eta * u_target + (1-eta) * u_noisy)
  - D and dt both active — noise magnitude scales with diffusion.

Uses batched cKDTree.query_ball_tree + sparse matvec for neighbour averaging.

P2.2: Wrapped in VicsekMode(ForceMode) with @register("vicsek").
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from ._mode import ForceFn, ForceMode, register

if TYPE_CHECKING:
    from ...core.config import SimConfig
    from ...core.types import SpatialIndex


@register("vicsek")
class VicsekMode(ForceMode):
    """Vicsek 1995 constant-speed angle coupling with memory term (P1.8)."""

    needs_index = True

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
        """Compute Vicsek angle-coupling forces with memory term (P1.8).

        Phase 1.8 update:
          For each bird i:
            1. Compute u_target from neighbour average (as before).
            2. Apply tangent-plane noise to u_old:
               g_i ~ N(0, I3)
               n_perp = g_i - (g_i*u_old_i)*u_old_i        [project to tangent plane]
               u_noisy = normalize(u_old_i + sqrt(2*D*dt) * n_perp)
            3. Blend with target:
               u_new = normalize(eta * u_target + (1-eta) * u_noisy)
            4. Set velocity = u_new * v0 (constant speed).

        The memory term means noise accumulates over time rather than
        being independent each frame — D now controls the diffusion rate
        rather than just the fresh-noise magnitude.
        """
        active_idx = np.where(active)[0]
        n_active = len(active_idx)
        if n_active == 0:
            return

        from scipy.spatial import cKDTree

        eta = config.vicsek_couplage
        D = config.vicsek_diffusion
        dt = config.vicsek_time_step
        radius = config.vicsek_radius_influence
        v0 = config.vicsek_velocity

        active_pos = positions[active_idx]

        # Current directions (old, pre-update)
        old_dirs = velocities[active_idx].copy()
        old_norms = np.linalg.norm(old_dirs, axis=1)
        # Normalise old directions (handle zero-speed edge case)
        valid_old = old_norms > 1e-6
        old_dirs[valid_old] = old_dirs[valid_old] / old_norms[valid_old, np.newaxis]
        # Birds with zero speed get a random direction as their "old" state
        if not valid_old.all():
            zero_mask = ~valid_old
            n_zero = zero_mask.sum()
            random_dirs = rng.normal(size=(n_zero, 3)).astype(np.float32)
            rnd_norms = np.linalg.norm(random_dirs, axis=1, keepdims=True) + 1e-10
            old_dirs[zero_mask] = random_dirs / rnd_norms

        if n_active < 2:
            # Single bird: pure memory + noise, no neighbours
            noise_scale = np.sqrt(2.0 * D * dt)
            g = rng.normal(size=(n_active, 3)).astype(np.float32)
            # Tangent-plane projection (RAW, not normalised — P1.8 spec)
            g_dot_u = np.sum(g * old_dirs, axis=1, keepdims=True)
            n_perp = g - g_dot_u * old_dirs

            noisy_dirs = old_dirs + noise_scale * n_perp
            noisy_norms = np.linalg.norm(noisy_dirs, axis=1, keepdims=True) + 1e-10
            directions = noisy_dirs / noisy_norms
            velocities[active_idx] = directions * v0
            return

        # Use shared spatial index for neighbour queries
        tree = getattr(index, 'tree', None) if index is not None else None
        if tree is None:
            tree = cKDTree(active_pos)

        # Batched radius query — all points at once (single tree traversal)
        all_nbrs = tree.query_ball_tree(tree, radius)

        neighbour_dirs = np.zeros((n_active, 3), dtype=np.float32)

        # Pre-compute validity mask: zero-speed birds have no meaningful direction
        vel_norms = np.linalg.norm(velocities[active_idx], axis=1)
        valid_mask = vel_norms > 1e-6

        # Normalize directions of valid birds for vectorized neighbour averaging
        unit_dirs = np.zeros((n_active, 3), dtype=np.float32)
        if valid_mask.any():
            unit_dirs[valid_mask] = (
                velocities[active_idx][valid_mask]
                / vel_norms[valid_mask, np.newaxis]
            )

        # Build sparse adjacency: bird i connected to neighbour j (including self
        # if valid — matching old query_ball_point behaviour that includes the
        # query point). Filter zero-speed neighbours via valid_mask.
        rows: list[int] = []
        cols: list[int] = []
        for i, nbrs in enumerate(all_nbrs):
            for j in nbrs:
                if valid_mask[j]:
                    rows.append(i)
                    cols.append(j)

        if rows:
            from scipy.sparse import coo_matrix

            adj = coo_matrix(
                (np.ones(len(rows), dtype=np.float32), (rows, cols)),
                shape=(n_active, n_active),
            ).tocsr()
            # Row sums = valid neighbour counts per bird (self included if valid)
            nbr_counts = np.array(adj.sum(axis=1)).flatten()
            # Sparse matvec: sum of neighbour unit directions per bird
            sums = adj @ unit_dirs
            # Require > 1 valid neighbour (self + at least one other, matching
            # old code's valid.sum() > 1 guard).
            mask = nbr_counts > 1
            neighbour_dirs[mask] = (sums[mask] / nbr_counts[mask, np.newaxis]).astype(np.float32)

        # --- Phase 1.8: Memory term with tangent-plane noise ---
        # Step 1: Generate Gaussian noise
        g = rng.normal(size=(n_active, 3)).astype(np.float32)

        # Step 2: Project to tangent plane of old direction
        #   n_perp = g - (g*u_old)*u_old  (RAW projection, not normalised — P1.8 spec)
        g_dot_u = np.sum(g * old_dirs, axis=1, keepdims=True)
        n_perp = g - g_dot_u * old_dirs

        # Step 3: Scale by diffusion coefficient
        #   sqrt(2*D*dt)
        noise_scale = np.sqrt(2.0 * D * dt)

        # Step 4: u_noisy = normalize(u_old + sqrt(2*D*dt) * n_perp)
        noisy_dirs = old_dirs + noise_scale * n_perp
        noisy_norms = np.linalg.norm(noisy_dirs, axis=1, keepdims=True) + 1e-10
        noisy_dirs = noisy_dirs / noisy_norms

        # Step 6: Blend with neighbour average
        #   Birds with neighbours: u_new = normalize(eta*u_target + (1-eta)*u_noisy)
        #   Birds without neighbours: u_new = u_noisy (no target to blend with)
        has_neighbours = np.linalg.norm(neighbour_dirs, axis=1) > 1e-6
        directions = noisy_dirs.copy()  # default: pure memory+noise

        if has_neighbours.any():
            nd = neighbour_dirs[has_neighbours]
            nd_norms = np.linalg.norm(nd, axis=1, keepdims=True) + 1e-10
            nd = nd / nd_norms
            blended = eta * nd + (1.0 - eta) * noisy_dirs[has_neighbours]
            blended_norms = np.linalg.norm(blended, axis=1, keepdims=True) + 1e-10
            directions[has_neighbours] = blended / blended_norms

        # Step 7: Set constant-speed velocity
        velocities[active_idx] = directions * v0


# Backward compatibility alias — tests import vicsek_forces directly
vicsek_forces: ForceFn = VicsekMode.compute  # type: ignore[assignment]
vicsek_forces.needs_index = True
