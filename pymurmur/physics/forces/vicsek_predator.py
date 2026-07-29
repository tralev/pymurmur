"""Vicsek mode — Phase 6 (P6.1–P6.3) predator-prey species dynamics.

  P6.1: Fear-weighted alignment blending for prey near predators
        (_apply_fear_blending, _apply_solo_fear).
  P6.2: Predator hunting strategy with nearest-prey pursuit
        (_apply_predator_hunting).
  P6.3: Asymmetric position collisions — same-type symmetric,
        prey-predator asymmetric, toroidal seam-crossing
        (resolve_species_collisions).

Split out of vicsek.py (file-size split) — VicsekMode's core P1.8
alignment/memory-term compute() stays in the original and calls back
into _apply_fear_blending/_apply_solo_fear/_apply_predator_hunting
here. resolve_species_collisions is called externally by
simulation/engine.py's P6.3 post-integrate step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...core.config import SimConfig

# P8: species-collision kernels — numba when available, numpy fallback.
try:
    from ._kernels import _HAS_NUMBA as _KERNELS_HAS_NUMBA  # noqa: F401
    from ._kernels import _numpy_species_collisions
    if _KERNELS_HAS_NUMBA:
        from ._kernels import _numba_species_collisions
    else:
        _numba_species_collisions = _numpy_species_collisions
except ImportError:
    _KERNELS_HAS_NUMBA = False

    def _numpy_species_collisions(
        positions: np.ndarray, is_predator: np.ndarray, active_idx: np.ndarray,
        r_avoid: float, r_pred: float,
        domain_w: float, domain_h: float, domain_d: float,
    ) -> int:
        """Inline numpy fallback — species collisions (identical logic)."""
        domains = np.array([domain_w, domain_h, domain_d], dtype=np.float32)
        corrections = 0
        for i_idx, i in enumerate(active_idx):
            for j in active_idx[i_idx + 1:]:
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
                n_hat = delta / dist
                same_type = is_predator[i] == is_predator[j]
                if same_type and dist < r_avoid:
                    push = (r_avoid - dist) * 0.5
                    positions[i] -= push * n_hat
                    positions[j] += push * n_hat
                    corrections += 1
                elif not same_type and dist < r_pred:
                    push = r_pred - dist
                    if is_predator[i] and not is_predator[j]:
                        positions[j] += push * n_hat
                    elif is_predator[j] and not is_predator[i]:
                        positions[i] -= push * n_hat
                    corrections += 1
        return corrections

    _numba_species_collisions = _numpy_species_collisions


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

    if active is None:
        active_idx = np.arange(len(positions))
    else:
        active_idx = np.where(active)[0]

    if len(active_idx) < 2:
        return 0

    # P8: this is an O(N^2) sequential (Gauss-Seidel-style) pairwise
    # correction — each pair's push mutates `positions` immediately, so
    # later pairs see already-corrected positions. That order-dependence
    # means it can't be batched into one vectorised numpy pass without
    # changing behaviour; dispatch to a numba-compiled version of the
    # exact same sequential loop when available (removes ~2 million
    # Python-level calls/step at N=2000 — the dominant cost of vicsek
    # mode), falling back to the pure-Python loop otherwise.
    use_numba = getattr(config.perf, 'use_numba', False)
    if use_numba and _KERNELS_HAS_NUMBA:
        return int(_numba_species_collisions(
            positions, is_predator, active_idx.astype(np.int64),
            float(R_avoid), float(R_pred),
            float(width), float(height), float(depth),
        ))
    return _numpy_species_collisions(
        positions, is_predator, active_idx,
        float(R_avoid), float(R_pred),
        float(width), float(height), float(depth),
    )
