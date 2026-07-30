"""Pearce 2014 projection mode — 3D spherical-cap occlusion + alignment.

Computes delta per bird via occlusion.py, then blends with neighbour
alignment. Uses topological sigma for neighbour selection.

I1.3: Uses spherical_cap_occlusion_batched — all observers in one call,
zero Python object allocations in the hot path.

P2.2: Wrapped in ProjectionMode(ForceMode) with @register("projection").
Modularity pass 9: delta/alignment/heading_inertia/noise are registered
       as named ForceTerm entries in PROJECTION_TERMS and composed via
       composeForces() (physics/forces/_base.py) to build v_desired —
       the same S2.A5 contract field.py's terms use. The Reynolds
       subtraction, clamp, and steric stay outside the composed terms,
       matching field.py's own precedent of keeping post-processing
       separate from the composed terms.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._base import ForceTerm, composeForces
from ..occlusion import spherical_cap_occlusion_batched
from ..steric import steric_force  # P1.10: L0 atom import at module top (no cycle risk)
from ..plugins.force_mode import ForceFn, ForceMode, register
from ..plugins.neighbor_selection import NEIGHBOR_SELECTOR_REGISTRY

if TYPE_CHECKING:
    from ...core.config import SimConfig
    from ...core.types import SpatialIndex


# ── Modularity pass 9: projection-mode term composition (S2.A5 contract) ──
# The composable part of projection mode is building v_desired from
# delta/alignment/heading-inertia/noise (a genuine weighted sum); the
# Reynolds-subtract + clamp + steric stay outside composeForces, matching
# field.py's own precedent of keeping post-processing (its max_force
# clamp) separate from the composed terms. Faithful 1:1 extraction of
# the pre-existing v_desired-building expression, not a reformulation.

@dataclass
class ProjectionTermContext:
    """Shared per-frame context passed to every projection-mode ForceTerm."""

    delta: np.ndarray                  # (n_active, 3) — from occlusion
    align_dir: np.ndarray              # (n_active, 3)
    phi_p: float                       # coherence-gated
    phi_a: float                       # coherence-gated
    phi_n: float
    heading_inertia: float
    current_heading: np.ndarray | None  # (n_active, 3) or None if heading_inertia <= 0
    eta_dir: np.ndarray | None          # (n_active, 3) unit noise or None if phi_n <= 0


def _term_projection_delta(fx: ProjectionTermContext) -> np.ndarray:
    return fx.delta * fx.phi_p


def _term_projection_alignment(fx: ProjectionTermContext) -> np.ndarray:
    return fx.align_dir * fx.phi_a


def _term_projection_heading_inertia(fx: ProjectionTermContext) -> np.ndarray:
    """SS09/S11-style heading-blend inertia: a genuinely separate additive
    pull toward the bird's own current heading, independent of the
    phi_p+phi_a+phi_n partition of unity (v_desired was never actually
    constrained to unit length before the Reynolds subtraction, so this
    extends rather than breaks that invariant)."""
    if fx.current_heading is None:
        return np.zeros_like(fx.delta)
    return fx.current_heading * fx.heading_inertia


def _term_projection_noise(fx: ProjectionTermContext) -> np.ndarray:
    """S1.4: Pearce noise term — v proportional to phi_p*delta +
    phi_a*<v_hat>_sigma + phi_n*eta_hat, phi_n = 1 - phi_p - phi_a,
    eta_hat uniform on S^2. Keeps the flock from converging to perfect
    alignment."""
    if fx.eta_dir is None:
        return np.zeros_like(fx.delta)
    return fx.eta_dir * fx.phi_n


# Order matches the pre-extraction expression exactly: delta*phi_p +
# align_dir*phi_a (one Python expression), then optionally +=
# current_heading*heading_inertia, then optionally += eta_dir*phi_n —
# composeForces' left-to-right accumulation reproduces the same
# floating-point summation order bit-for-bit.
PROJECTION_TERMS: list[ForceTerm] = [
    ForceTerm("delta", fn=_term_projection_delta),
    ForceTerm("alignment", fn=_term_projection_alignment),
    ForceTerm("heading_inertia", fn=_term_projection_heading_inertia),
    ForceTerm("noise", fn=_term_projection_noise),
]


@register("projection")
class ProjectionMode(ForceMode):
    """Pearce 2014 hybrid projection forces — occlusion + alignment."""

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
        """Compute Pearce hybrid projection forces.

        1. Batch-collect topological sigma neighbours via spatial index
        2. Batch-compute delta and Theta via spherical_cap_occlusion_batched
        3. Blend: v_desired = phi_p * delta + phi_a * align_dir + phi_n * eta
           (S1.4: phi_n = 1 − phi_p − phi_a, eta uniform on S²)
        4. Clamp steering and apply
        """
        active_idx = np.where(active)[0]
        n_active = len(active_idx)
        if n_active == 0:
            return

        # max_visibility caps sigma for the occlusion step
        sigma = min(config.sigma, config.max_visibility)
        blind_cos = None
        if config.blind_deg > 0:
            blind_cos = np.cos(np.radians(config.blind_deg / 2))

        # --- Stage 1: collect neighbour indices for all active birds ---
        nbr_idx = NEIGHBOR_SELECTOR_REGISTRY["topological_visibility"].select(
            positions, velocities, active, index, config, sigma=sigma,
        )
        # nbr_idx: (n_active, sigma) int32 — some rows may have -1 sentinels

        # Find birds with at least one valid neighbour
        has_nbrs = (nbr_idx >= 0).any(axis=1)  # (n_active,)
        if not has_nbrs.any():
            return

        # --- Stage 2: gather neighbour positions/velocities in batch ---
        # Clamp -1 sentinels to 0 for safe gather (will be masked out)
        safe_idx = np.maximum(nbr_idx, 0)  # (n_active, sigma)
        nbr_pos = positions[safe_idx]       # (n_active, sigma, 3)
        nbr_vel = velocities[safe_idx]      # (n_active, sigma, 3)

        # --- Stage 3: batched occlusion ---
        valid_mask = nbr_idx >= 0  # (n_active, sigma) — exclude -1 sentinels
        delta, visible_mask, theta = spherical_cap_occlusion_batched(
            positions[active_idx],
            velocities[active_idx],
            nbr_pos,
            nbr_vel,
            boid_size=config.boid_size,
            blind_cos=blind_cos,
            anisotropy=config.anisotropy if config.refinements else 1.0,
            max_candidates=config.max_occlusion_neighbors,
            valid_mask=valid_mask,
            n_jobs=config.parallel_workers,
        )
        # delta:     (n_active, 3)
        # visible_mask: (n_active, sigma) bool
        # theta:     (n_active,)

        last_theta[active_idx] = theta

        # --- Stage 4: alignment direction from visible neighbours ---
        # Zero out invisible neighbour velocities
        vis_vel = nbr_vel * visible_mask[:, :, np.newaxis]  # (n_active, sigma, 3)
        n_visible = visible_mask.sum(axis=1, keepdims=True)  # (n_active, 1)

        v_avg = vis_vel.sum(axis=1) / np.maximum(n_visible, 1)  # (n_active, 3)
        v_norm = np.linalg.norm(v_avg, axis=1, keepdims=True)

        align_dir = np.zeros((n_active, 3), dtype=np.float32)
        valid = (v_norm.squeeze() > 1e-6)
        align_dir[valid] = v_avg[valid] / v_norm[valid]

        # --- Stage 5: blend and steer ---
        phi_p = config.projection.phi_p
        phi_a = config.phi_a

        # S2.B8: Coherence gate — reduce directional/positional pull for
        # small flocks, mirroring SpatialMode's align/coh gating. Runtime-
        # private field set by the ecology extension (see spatial.py).
        coherence = getattr(config, '_coherence_factor', 1.0)
        if coherence < 1.0:
            phi_p *= coherence
            phi_a *= coherence

        # 0.0 heading_inertia adds nothing — byte-identical to before this
        # existed. current_heading is only computed when needed, matching
        # the original conditional-skip (no wasted norm() calls).
        heading_inertia = config.projection.projection_heading_inertia
        current_heading = None
        if heading_inertia > 0.0:
            v_norms = np.linalg.norm(velocities[active_idx], axis=1, keepdims=True)
            safe_norms = np.where(v_norms > 1e-6, v_norms, 1.0)
            current_heading = np.where(
                v_norms > 1e-6, velocities[active_idx] / safe_norms, 0.0,
            )

        # S1.4: Pearce noise term — phi_n = 1 - phi_p - phi_a. eta_dir is
        # only drawn from rng when phi_n > 0, matching the original
        # conditional-skip exactly (determinism-critical: the rng call
        # sequence must stay identical to preserve golden trajectories).
        phi_n = max(0.0, 1.0 - phi_p - phi_a)
        eta_dir = None
        if phi_n > 0.0:
            eta = rng.normal(size=(n_active, 3)).astype(np.float32)
            eta_norms = np.linalg.norm(eta, axis=1, keepdims=True)
            eta_norms[eta_norms == 0] = 1.0
            eta_dir = eta / eta_norms

        fx = ProjectionTermContext(
            delta=delta,
            align_dir=align_dir,
            phi_p=phi_p,
            phi_a=phi_a,
            phi_n=phi_n,
            heading_inertia=heading_inertia,
            current_heading=current_heading,
            eta_dir=eta_dir,
        )
        v_desired = composeForces(fx, PROJECTION_TERMS, n=n_active)

        steering = v_desired - velocities[active_idx]

        steer_mag = np.linalg.norm(steering, axis=1)
        too_strong = steer_mag > config.max_force
        if too_strong.any():
            steering[too_strong] = (
                steering[too_strong] / steer_mag[too_strong, np.newaxis] * config.max_force
            )

        accelerations[active_idx] += steering

        # --- Stage 6: steric repulsion (per-bird, lightweight) ---
        if config.refinements and config.steric > 0:
            steric_visible_only = config.refinement.steric_visible_only
            for j, i in enumerate(active_idx):
                if steric_visible_only:
                    nbr_mask = (nbr_idx[j] >= 0) & visible_mask[j]
                else:
                    nbr_mask = nbr_idx[j] >= 0
                valid_nbrs = nbr_idx[j][nbr_mask]
                if len(valid_nbrs) > 0:
                    accelerations[i] += steric_force(
                        positions[i], positions[valid_nbrs], config.steric,
                        threshold=config.refinement.steric_radius,
                        max_force=config.max_force,
                    )


# Backward compatibility alias — tests import projection_forces directly
projection_forces: ForceFn = ProjectionMode.compute  # type: ignore[assignment]
projection_forces.needs_index = True


def _topological_neighbors_batch(
    positions: np.ndarray,
    index: SpatialIndex | None,
    active_idx: np.ndarray,
    sigma: int,
) -> np.ndarray:
    """Collect sigma nearest neighbours for all active birds.

    Returns (n_active, sigma) int32 with -1 sentinels where fewer than
    sigma neighbours exist.
    """
    n_active = len(active_idx)

    if index is None or not index.ready:
        return np.full((n_active, sigma), -1, dtype=np.int32)

    result = np.full((n_active, sigma), -1, dtype=np.int32)
    for j, i in enumerate(active_idx):
        nbrs = index.query_knn(positions[i], k=sigma)
        result[j, :len(nbrs)] = nbrs

    return result
