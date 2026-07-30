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

Phase 6 predator-prey helper functions and resolve_species_collisions
moved to vicsek_predator.py (file-size split of this file).

Modularity pass 10/11: this mode computes a genuine per-bird target
speed (v0 for prey, vicsek_velocity_predator for predators), but
speed_mode="fixed" previously always renormalised to a flat config.v0
in flock.integrate() regardless -- nothing ever told it what this mode
actually computed, silently discarding the predator/prey distinction
in every real simulation run (unit tests calling compute() directly,
before integrate() runs, never caught it). _stash_target_speed() fixes
this by scattering the computed speed onto config._mode_target_speed,
which flock.integrate() now reads as the max_speed base.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..plugins.force_mode import ForceFn, ForceMode, register
from ..plugins.neighbor_selection import NEIGHBOR_SELECTOR_REGISTRY
from ._base import stash_target_speed as _stash_target_speed
from .vicsek_predator import (
    _apply_fear_blending,
    _apply_predator_hunting,
    _apply_solo_fear,
)

if TYPE_CHECKING:
    from ...core.config import SimConfig
    from ...core.types import SpatialIndex


@register("vicsek")
class VicsekMode(ForceMode):
    """Vicsek 1995 constant-speed angle coupling with memory term (P1.8)
    and predator-prey species dynamics (P6.1–P6.2)."""

    needs_index = True
    speed_mode = "fixed"  # D2: constant |v| = v0 (or v_pred), set directly below

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
            # S2.D1: All-predator flock → skip alignment/hunting interaction,
            # but still apply a pure random walk (spec) rather than freezing
            # velocities outright — a flock of only predators isn't inert.
            if n_prey == 0:
                v_pred = config.vicsek_velocity_predator
                rand_dirs = rng.normal(size=(n_active, 3)).astype(np.float32)
                rand_norms = np.linalg.norm(rand_dirs, axis=1, keepdims=True) + 1e-10
                velocities[active_idx] = (rand_dirs / rand_norms) * v_pred
                _stash_target_speed(
                    config, len(positions), active_idx,
                    np.full(n_active, v_pred, dtype=np.float32),
                )
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

            # §09/§11-style heading-blend inertia — this early-return path
            # has its own velocity assignment, separate from the "Finalise
            # velocities" section below, so it needs the same blend applied
            # here too for the feature to apply uniformly regardless of
            # flock size. 0.0 (default) skips this entirely.
            heading_inertia = config.vicsek.vicsek_heading_inertia
            if heading_inertia > 0.0:
                blended = old_dirs * heading_inertia + directions * (1.0 - heading_inertia)
                blended_norms = np.linalg.norm(blended, axis=1, keepdims=True) + 1e-10
                directions = blended / blended_norms

            velocities[active_idx] = directions * v0
            _stash_target_speed(
                config, len(positions), active_idx,
                np.full(n_active, v0, dtype=np.float32),
            )
            return

        # ── Neighbour alignment (standard Vicsek) ─────────────
        neighbour_dirs = np.zeros((n_active, 3), dtype=np.float32)

        vel_norms = np.linalg.norm(velocities[active_idx], axis=1)
        valid_mask = vel_norms > 1e-6

        unit_dirs = np.zeros((n_active, 3), dtype=np.float32)
        if valid_mask.any():
            unit_dirs[valid_mask] = (
                velocities[active_idx][valid_mask]
                / vel_norms[valid_mask, np.newaxis]
            )

        adj, nbr_counts = NEIGHBOR_SELECTOR_REGISTRY["ball_tree_radius"].select(
            positions, velocities, active, index, config, radius=radius,
        )
        if adj is not None:
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

        # §09/§11-style heading-blend inertia: blends the bird's prior
        # heading into the fully-finalized new direction, applying
        # uniformly regardless of which branch above produced `directions`
        # (plain Vicsek blend, fear-blend, solo-fear, predator-hunting).
        # Independent of the vicsek_couplage/vicsek_diffusion memory term
        # above. 0.0 (default) skips this entirely — byte-identical to
        # before this existed.
        heading_inertia = config.vicsek.vicsek_heading_inertia
        if heading_inertia > 0.0:
            blended = old_dirs * heading_inertia + directions * (1.0 - heading_inertia)
            blended_norms = np.linalg.norm(blended, axis=1, keepdims=True) + 1e-10
            directions = blended / blended_norms

        # ── Finalise velocities ───────────────────────────────
        speeds = np.full(n_active, v0, dtype=np.float32)
        if n_pred > 0:
            v_pred = config.vicsek_velocity_predator
            speeds[is_pred] = v_pred

        velocities[active_idx] = directions * speeds[:, np.newaxis]
        _stash_target_speed(config, len(positions), active_idx, speeds)




# Backward compatibility alias — tests import vicsek_forces directly
vicsek_forces: ForceFn = VicsekMode.compute  # type: ignore[assignment]
vicsek_forces.needs_index = True
