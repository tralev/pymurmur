"""Vicsek 1995 mode — constant-speed angle coupling with memory term.

Phase 1.8 (P1.8): Memory term + tangent-plane noise.
  - u_noisy = normalize(u_old + sqrt(2*D*dt) * n_perp)
    where n_perp = g - (g*u_old)*u_old,  g ~ N(0, I3)
  - u_new = normalize(eta * u_target + (1-eta) * u_noisy)
  - D and dt both active — noise magnitude scales with diffusion.

Phase 6 (P6.1–P6.3): Predator-prey species dynamics.
  P6.1: Fear-weighted alignment blending for prey near predators.
  P6.2: Predator hunting strategy with nearest-prey pursuit.
  P6.3: Asymmetric position collisions (same-type symmetric,
         prey-predator asymmetric, toroidal seam-crossing).

Uses batched cKDTree.query_ball_tree + sparse matvec for neighbour averaging.

P2.2: Wrapped in VicsekMode(ForceMode) with @register("vicsek").
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._mode import ForceFn, ForceMode, register

if TYPE_CHECKING:
    from ...core.config import SimConfig
    from ...core.types import SpatialIndex


@register("vicsek")
class VicsekMode(ForceMode):
    """Vicsek 1995 constant-speed angle coupling with memory term (P1.8)
    and predator-prey species dynamics (P6.1–P6.2)."""

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
        """Compute Vicsek angle-coupling forces with memory term and species.

        Phase 1.8 update + Phase 6 species dynamics:
          P6.1: Prey near predators blend alignment with flee direction.
          P6.2: Predators hunt nearest prey; random walk if none.
          All-predators flock → early-out (skip all interaction).
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

        # ── P6: Species detection ─────────────────────────────
        is_predator = getattr(config, '_is_predator', None)
        if is_predator is not None:
            is_pred = is_predator[active_idx]
            n_pred = int(is_pred.sum())
            n_prey = n_active - n_pred
            # All-predator flock → skip all interaction (P6.2 early-out)
            if n_prey == 0:
                return
        else:
            is_pred = np.zeros(n_active, dtype=bool)
            n_pred = 0
            n_prey = n_active

        active_pos = positions[active_idx]

        # Current directions (old, pre-update)
        old_dirs = velocities[active_idx].copy()
        old_norms = np.linalg.norm(old_dirs, axis=1)
        valid_old = old_norms > 1e-6
        old_dirs[valid_old] = old_dirs[valid_old] / old_norms[valid_old, np.newaxis]
        if not valid_old.all():
            zero_mask = ~valid_old
            n_zero = zero_mask.sum()
            random_dirs = rng.normal(size=(n_zero, 3)).astype(np.float32)
            rnd_norms = np.linalg.norm(random_dirs, axis=1, keepdims=True) + 1e-10
            old_dirs[zero_mask] = random_dirs / rnd_norms

        # ── Single-bird case ──────────────────────────────────
        if n_active < 2:
            noise_scale = np.sqrt(2.0 * D * dt)
            g = rng.normal(size=(n_active, 3)).astype(np.float32)
            g_dot_u = np.sum(g * old_dirs, axis=1, keepdims=True)
            n_perp = g - g_dot_u * old_dirs
            noisy_dirs = old_dirs + noise_scale * n_perp
            noisy_norms = np.linalg.norm(noisy_dirs, axis=1, keepdims=True) + 1e-10
            directions = noisy_dirs / noisy_norms
            velocities[active_idx] = directions * v0
            return

        # ── Neighbour alignment (standard Vicsek) ─────────────
        tree = getattr(index, 'tree', None) if index is not None else None
        if tree is None:
            tree = cKDTree(active_pos)

        all_nbrs = tree.query_ball_tree(tree, radius)

        neighbour_dirs = np.zeros((n_active, 3), dtype=np.float32)

        vel_norms = np.linalg.norm(velocities[active_idx], axis=1)
        valid_mask = vel_norms > 1e-6

        unit_dirs = np.zeros((n_active, 3), dtype=np.float32)
        if valid_mask.any():
            unit_dirs[valid_mask] = (
                velocities[active_idx][valid_mask]
                / vel_norms[valid_mask, np.newaxis]
            )

        rows: list[int] = []
        cols: list[int] = []
        for i, nbrs in enumerate(all_nbrs):
            for j in nbrs:
                if valid_mask[j]:
                    rows.append(i)
                    cols.append(j)

        nbr_counts = np.zeros(n_active, dtype=np.float32)
        if rows:
            from scipy.sparse import coo_matrix

            adj = coo_matrix(
                (np.ones(len(rows), dtype=np.float32), (rows, cols)),
                shape=(n_active, n_active),
            ).tocsr()
            nbr_counts = np.array(adj.sum(axis=1)).flatten()
            sums = adj @ unit_dirs
            mask = nbr_counts > 1
            neighbour_dirs[mask] = (sums[mask] / nbr_counts[mask, np.newaxis]).astype(np.float32)

        # ── Phase 1.8: Memory term with tangent-plane noise ───
        g = rng.normal(size=(n_active, 3)).astype(np.float32)
        g_dot_u = np.sum(g * old_dirs, axis=1, keepdims=True)
        n_perp = g - g_dot_u * old_dirs
        noise_scale = np.sqrt(2.0 * D * dt)
        noisy_dirs = old_dirs + noise_scale * n_perp
        noisy_norms = np.linalg.norm(noisy_dirs, axis=1, keepdims=True) + 1e-10
        noisy_dirs = noisy_dirs / noisy_norms

        # ── Blend neighbour average with memory ───────────────
        has_neighbours = np.linalg.norm(neighbour_dirs, axis=1) > 1e-6
        directions = noisy_dirs.copy()

        if has_neighbours.any():
            # Normalise neighbour directions (compressed to birds with neighbours)
            nd = neighbour_dirs[has_neighbours]
            nd_norms = np.linalg.norm(nd, axis=1, keepdims=True) + 1e-10
            nd = nd / nd_norms
            hn_idx = np.where(has_neighbours)[0]  # global→compressed map

            # Standard Vicsek blend for all birds with neighbours
            blended = eta * nd + (1.0 - eta) * noisy_dirs[has_neighbours]
            blended_norms = np.linalg.norm(blended, axis=1, keepdims=True) + 1e-10
            directions[has_neighbours] = blended / blended_norms

            # P6.1: Override prey near predators with fear-weighted blend
            if n_pred > 0:
                prey_with_nbrs = has_neighbours & ~is_pred
                if prey_with_nbrs.any():
                    # Build compressed index lookup for fast mapping
                    hn_to_compressed: dict[int, int] = {}
                    for ci, gi in enumerate(hn_idx):
                        hn_to_compressed[int(gi)] = ci
                    _apply_fear_blending(
                        active_pos, directions, nd, has_neighbours,
                        hn_idx, hn_to_compressed,
                        is_pred, prey_with_nbrs, eta, noisy_dirs,
                        config, rng,
                    )

        # P6.1: Solo prey near predators still flee (no neighbours required)
        if n_pred > 0:
            prey_without_nbrs = ~has_neighbours & ~is_pred
            if prey_without_nbrs.any():
                _apply_solo_fear(
                    active_pos, directions, is_pred,
                    prey_without_nbrs, config, rng,
                )

        # P6.2: Predator hunting — always runs, regardless of neighbours
        if n_pred > 0:
            pred_mask = is_pred
            if pred_mask.any():
                _apply_predator_hunting(
                    active_pos, directions, pred_mask,
                    is_pred, config, rng,
                )

        # ── Finalise velocities ───────────────────────────────
        speeds = np.full(n_active, v0, dtype=np.float32)
        if n_pred > 0:
            v_pred = config.vicsek_velocity_predator
            speeds[is_pred] = v_pred

        velocities[active_idx] = directions * speeds[:, np.newaxis]


def _apply_fear_blending(
    positions: np.ndarray,
    directions: np.ndarray,
    nd: np.ndarray,       # pre-normalised neighbour directions [compressed]
    has_neighbours: np.ndarray,
    hn_idx: np.ndarray,   # global indices of birds with neighbours
    hn_to_compressed: dict[int, int],  # global → compressed index
    is_pred: np.ndarray,
    prey_mask: np.ndarray,
    eta: float,
    noisy_dirs: np.ndarray,
    config: SimConfig,
    rng: np.random.Generator,
) -> None:
    """P6.1: Blend alignment with flee direction for afraid prey.

    fear = clamp((R_pred - d_mean_pred) / R_pred, 0, 1)
    u_flee = normalize(mean(prey_pos - pred_pos))
    u_combined = normalize((1-fear)*u_align + fear*u_flee * weight_afraid)
    """
    R_pred = config.vicsek_radius_predators
    w_afraid = config.vicsek_weight_afraid
    width = config.width
    height = config.height
    depth = config.depth
    pred_idx = np.where(is_pred)[0]
    prey_indices = np.where(prey_mask)[0]

    if len(pred_idx) == 0:
        return

    pred_pos = positions[pred_idx]

    for pi in prey_indices:
        prey_pos = positions[pi]
        # Min-image distances to all predators
        diffs = pred_pos - prey_pos
        for dim_idx, domain in enumerate([width, height, depth]):
            half = domain / 2.0
            col_diffs = diffs[:, dim_idx]
            col_diffs[col_diffs > half] -= domain
            col_diffs[col_diffs < -half] += domain
        dists = np.linalg.norm(diffs, axis=1)
        near_mask = dists < R_pred
        if not near_mask.any():
            continue

        near_dists = dists[near_mask]
        fear = float(np.clip((R_pred - near_dists.mean()) / R_pred, 0.0, 1.0))
        if fear <= 0.0:
            continue

        # Flee direction: normalize(mean(prey_pos - pred_pos))
        near_diffs = diffs[near_mask]
        flee_dir = -near_diffs.mean(axis=0)
        flee_norm = np.linalg.norm(flee_dir)
        if flee_norm < 1e-10:
            flee_dir = rng.normal(size=3).astype(np.float32)
            flee_norm = np.linalg.norm(flee_dir) + 1e-10
        flee_dir = flee_dir / flee_norm

        # Get neighbour direction for this bird via compressed index lookup
        h_compressed = hn_to_compressed.get(int(pi))
        if h_compressed is None:
            continue
        u_align = nd[h_compressed]
        u_noisy = noisy_dirs[pi]

        # Blend: u_combined = normalize((1-fear)*eta*u_align + w_afraid*fear*flee + (1-eta)*u_noisy)
        blended = (1.0 - fear) * eta * u_align + w_afraid * fear * flee_dir + (1.0 - eta) * u_noisy
        b_norm = np.linalg.norm(blended)
        if b_norm > 1e-10:
            directions[pi] = blended / b_norm


def _apply_solo_fear(
    positions: np.ndarray,
    directions: np.ndarray,
    is_pred: np.ndarray,
    prey_mask: np.ndarray,
    config: SimConfig,
    rng: np.random.Generator,
) -> None:
    """P6.1: Solo prey (no neighbours) near predators — flee only.

    Unlike _apply_fear_blending, these birds have no neighbour alignment
    to blend with.  The direction is set to flee_dir directly (with noise
    mixing from the existing noisy direction).
    """
    R_pred = config.vicsek_radius_predators
    width = config.width
    height = config.height
    depth = config.depth
    pred_idx = np.where(is_pred)[0]
    prey_indices = np.where(prey_mask)[0]

    if len(pred_idx) == 0:
        return

    pred_pos = positions[pred_idx]

    for pi in prey_indices:
        prey_pos = positions[pi]
        diffs = pred_pos - prey_pos
        for dim_idx, domain in enumerate([width, height, depth]):
            half = domain / 2.0
            col_diffs = diffs[:, dim_idx]
            col_diffs[col_diffs > half] -= domain
            col_diffs[col_diffs < -half] += domain
        dists = np.linalg.norm(diffs, axis=1)
        near_mask = dists < R_pred
        if not near_mask.any():
            continue

        # Flee direction away from nearby predators
        near_diffs = diffs[near_mask]
        flee_dir = -near_diffs.mean(axis=0)
        flee_norm = np.linalg.norm(flee_dir)
        if flee_norm < 1e-10:
            flee_dir = rng.normal(size=3).astype(np.float32)
            flee_norm = np.linalg.norm(flee_dir) + 1e-10
        flee_dir = flee_dir / flee_norm

        # Mix 70% flee + 30% existing noisy direction for organic feel
        existing = directions[pi]
        blended = 0.7 * flee_dir + 0.3 * existing
        b_norm = np.linalg.norm(blended)
        if b_norm > 1e-10:
            directions[pi] = blended / b_norm


def _apply_predator_hunting(
    positions: np.ndarray,
    directions: np.ndarray,
    pred_mask: np.ndarray,
    is_pred: np.ndarray,
    config: SimConfig,
    rng: np.random.Generator,
) -> None:
    """P6.2: Predator hunting — steer toward nearest prey.

    u_target = normalize(nearest_prey_pos - predator_pos)
    u_new = normalize(u_target + predator_noise_ratio * random_unit)
    Fallback: random walk if no prey in range.
    """
    R_pred = config.vicsek_radius_predators
    detect_r = config.vicsek_detect_ratio * R_pred
    noise_ratio = config.vicsek_predator_noise_ratio
    width = config.width
    height = config.height
    depth = config.depth
    prey_idx = np.where(~is_pred)[0]
    pred_indices = np.where(pred_mask)[0]

    if len(prey_idx) == 0:
        # No prey → predators random walk
        for pi in pred_indices:
            rand_dir = rng.normal(size=3).astype(np.float32)
            rand_norm = np.linalg.norm(rand_dir) + 1e-10
            directions[pi] = rand_dir / rand_norm
        return

    prey_pos = positions[prey_idx]

    for pi in pred_indices:
        pred_pos = positions[pi]
        diffs = prey_pos - pred_pos
        for dim_idx, domain in enumerate([width, height, depth]):
            half = domain / 2.0
            col_diffs = diffs[:, dim_idx]
            col_diffs[col_diffs > half] -= domain
            col_diffs[col_diffs < -half] += domain
        dists = np.linalg.norm(diffs, axis=1)
        near_mask = dists < detect_r

        if near_mask.any():
            nearest_idx = int(np.argmin(dists))
            target = diffs[nearest_idx]
            target_norm = np.linalg.norm(target)
            if target_norm > 1e-10:
                target = target / target_norm
            else:
                target = rng.normal(size=3).astype(np.float32)
                target = target / (np.linalg.norm(target) + 1e-10)
            # Add hunting noise
            noise = rng.normal(size=3).astype(np.float32)
            noise = noise / (np.linalg.norm(noise) + 1e-10)
            desired = target + noise_ratio * noise
            d_norm = np.linalg.norm(desired)
            if d_norm > 1e-10:
                directions[pi] = desired / d_norm
        else:
            # Fallback random walk
            rand_dir = rng.normal(size=3).astype(np.float32)
            rand_norm = np.linalg.norm(rand_dir) + 1e-10
            directions[pi] = rand_dir / rand_norm


def resolve_species_collisions(
    positions: np.ndarray,
    is_predator: np.ndarray,
    config: SimConfig,
    active: np.ndarray | None = None,
) -> int:
    """P6.3: Asymmetric position collisions.

    - Same-type at d < R_avoid: each moves (R_avoid-d)/2 along min-image n̂
    - Prey-predator at d < R_pred: prey takes FULL (R_pred-d), predator unmoved
    - Toroidal seam-crossing: min-image vectors used throughout.

    Args:
        positions: (N, 3) float32 array — mutated in place.
        is_predator: (N,) bool array.
        config: SimConfig with vicsek_radius_avoid and vicsek_radius_predators.
        active: (N,) bool mask — only active birds considered.

    Returns:
        Number of collision corrections applied.
    """
    R_avoid = config.vicsek_radius_avoid
    R_pred = config.vicsek_radius_predators
    width = config.width
    height = config.height
    depth = config.depth
    domains = np.array([width, height, depth], dtype=np.float32)

    if active is None:
        active_idx = np.arange(len(positions))
    else:
        active_idx = np.where(active)[0]

    if len(active_idx) < 2:
        return 0

    corrections = 0

    # Brute-force O(N²) — used for small N in tests; acceptable since
    # Vicsek mode typically runs small flocks.  Can be upgraded to
    # spatial-hash or KDTree for larger N later.
    for i_idx, i in enumerate(active_idx):
        for j in active_idx[i_idx + 1:]:
            # Min-image vector from i to j
            delta = positions[j] - positions[i]
            for dim in range(3):
                half = domains[dim] / 2.0
                if delta[dim] > half:
                    delta[dim] -= domains[dim]
                elif delta[dim] < -half:
                    delta[dim] += domains[dim]

            dist = np.linalg.norm(delta)
            if dist < 1e-10:
                continue

            n_hat = delta / dist  # unit vector from i to j

            same_type = is_predator[i] == is_predator[j]

            if same_type and dist < R_avoid:
                # Symmetric: each moves (R_avoid-d)/2 away from each other
                push = (R_avoid - dist) * 0.5
                positions[i] -= push * n_hat
                positions[j] += push * n_hat
                corrections += 1
            elif not same_type and dist < R_pred:
                # Asymmetric: prey moves full (R_pred-d), predator unmoved
                push = R_pred - dist
                if is_predator[i] and not is_predator[j]:
                    # i is predator, j is prey → push prey (j) away
                    positions[j] += push * n_hat
                elif is_predator[j] and not is_predator[i]:
                    # j is predator, i is prey → push prey (i) away
                    positions[i] -= push * n_hat
                corrections += 1

    return corrections


# Backward compatibility alias — tests import vicsek_forces directly
vicsek_forces: ForceFn = VicsekMode.compute  # type: ignore[assignment]
vicsek_forces.needs_index = True
