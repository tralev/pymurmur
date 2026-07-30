"""Modularity pass 10 — angle.py vectorization regression tests.

angle.py's per-bird Python loop was replaced with fully vectorised
numpy (see angle.py's module docstring for the RNG-order caveat: golden
trajectories were regenerated since the *draw order* changed, even
though the *math* didn't). This file cross-checks the vectorised
AngleMode.compute() against a hand-written per-bird reference
implementation of the pre-vectorization logic, for deterministic
scenarios (jitter_deg=0 and all birds already moving, so no rng call
happens in either implementation — isolating the math from the RNG
change entirely).

The reference below is a direct, un-optimised transcription of the
per-bird loop angle.py had before this pass (git history has the exact
original) — deliberately NOT sharing any code with the vectorised
version, so a bug shared between "reference" and "implementation"
can't hide from this check.
"""

from __future__ import annotations

from copy import copy

import numpy as np

from pymurmur.core.types import min_image, rotate_about, safe_normalize
from pymurmur.physics.forces.angle import AngleMode


def _reference_compute(
    positions, velocities, active, nbr_idx, config,
):
    """Per-bird reference matching angle.py's pre-vectorization loop
    exactly (jitter_deg assumed 0, no stationary birds -> no rng calls
    needed by either side)."""
    b = config.boid_size
    turn_rate = np.radians(config.angle.turn_rate)
    max_turn_rate = np.radians(config.angle.max_turn_rate)
    turn_threshold = np.radians(config.angle.turn_threshold)
    margin = config.boundary.boundary_margin
    base_speed = config.angle.base_speed
    sep_r = config.angle.sep_radius_bodies * b
    align_r = config.angle.align_radius_bodies * b
    range_r = config.angle.range_radius_bodies * b
    n_neighbors = config.angle.angle_neighbors
    speed_mode = config.angle.angle_speed_mode
    deficit_cap = float(n_neighbors * n_neighbors)
    border_mode = config.boundary.boundary_mode
    width, height, depth = config.width, config.height, config.depth
    dt = 1.0 / max(config.fps, 1)
    coherence = getattr(config, '_coherence_factor', 1.0)

    active_idx = np.where(active)[0]
    out_vel = velocities.copy()

    for j, global_i in enumerate(active_idx):
        pi = positions[global_i]
        vi = velocities[global_i]
        speed = np.linalg.norm(vi)
        assert speed >= 1e-6, "reference assumes no stationary birds"
        hdg = vi / speed

        nbrs = nbr_idx[j]
        nbrs = nbrs[nbrs > 0]

        target = None
        is_fleeing = False

        if len(nbrs) > 0:
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
                to_nbr = positions[nearest_idx] - pi
                if border_mode == "toroidal":
                    to_nbr = min_image(
                        to_nbr.reshape(1, 3),
                        np.array([width, height, depth], dtype=np.float32),
                    ).ravel()
                target = safe_normalize(-to_nbr)
            elif nearest_dist < align_r:
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
                centroid = nbr_pos.mean(axis=0)
                if border_mode == "toroidal":
                    to_centroid = centroid - pi
                    to_centroid = min_image(
                        to_centroid.reshape(1, 3),
                        np.array([width, height, depth], dtype=np.float32),
                    ).ravel()
                    centroid = pi + to_centroid
                target = safe_normalize(centroid - pi)

        n_nbrs = len(nbrs)
        deficit = n_neighbors - n_nbrs
        if deficit > 0:
            if speed_mode == "quadratic":
                new_speed = base_speed + min(deficit_cap, deficit * deficit)
            elif speed_mode == "softened":
                new_speed = base_speed + min(deficit_cap, deficit * deficit / 2.0)
            else:
                new_speed = base_speed + deficit * 5.0
        else:
            new_speed = base_speed

        edge_target = None
        if border_mode in ("margin", "sphere"):
            if border_mode == "sphere":
                radius = config.boundary.boundary_sphere_radius
                dist_from_center = np.linalg.norm(pi)
                if dist_from_center > radius - margin:
                    edge_target = safe_normalize(-pi)
                    edge_factor = float(1.0 - (radius - dist_from_center) / margin)
                    turn_rate_now = turn_rate + edge_factor * (max_turn_rate - turn_rate)
                else:
                    turn_rate_now = turn_rate
            else:
                turn_rate_now = turn_rate
                face_dist = float("inf")
                face_normal = np.zeros(3, dtype=np.float32)
                if pi[0] < margin:
                    d = pi[0]
                    if d < face_dist:
                        face_dist, face_normal = d, np.array([1.0, 0, 0], dtype=np.float32)
                if pi[0] > width - margin:
                    d = width - pi[0]
                    if d < face_dist:
                        face_dist, face_normal = d, np.array([-1.0, 0, 0], dtype=np.float32)
                if pi[1] < margin:
                    d = pi[1]
                    if d < face_dist:
                        face_dist, face_normal = d, np.array([0, 1.0, 0], dtype=np.float32)
                if pi[1] > height - margin:
                    d = height - pi[1]
                    if d < face_dist:
                        face_dist, face_normal = d, np.array([0, -1.0, 0], dtype=np.float32)
                if pi[2] < margin:
                    d = pi[2]
                    if d < face_dist:
                        face_dist, face_normal = d, np.array([0, 0, 1.0], dtype=np.float32)
                if pi[2] > depth - margin:
                    d = depth - pi[2]
                    if d < face_dist:
                        face_dist, face_normal = d, np.array([0, 0, -1.0], dtype=np.float32)
                if face_dist < margin:
                    edge_target = face_normal
                    edge_factor = float(1.0 - face_dist / margin)
                    turn_rate_now = turn_rate + edge_factor * (max_turn_rate - turn_rate)
        else:
            turn_rate_now = turn_rate

        if edge_target is not None:
            if target is not None:
                target = safe_normalize(target + edge_target)
            else:
                target = edge_target

        if target is not None:
            cos_phi = np.clip(np.dot(hdg, target), -1.0, 1.0)
            phi = np.arccos(cos_phi)
            if phi > turn_threshold:
                edge_only = edge_target is not None and target is edge_target
                gated_turn = turn_rate_now
                if not is_fleeing and not edge_only and coherence < 1.0:
                    gated_turn *= coherence
                axis = np.cross(hdg, target)
                axis_norm = np.linalg.norm(axis)
                if axis_norm < 1e-10:
                    axis = safe_normalize(np.cross(hdg, np.array([1.0, 0, 0], dtype=np.float32)))
                    if np.linalg.norm(axis) < 1e-10:
                        axis = np.array([0, 1.0, 0], dtype=np.float32)
                else:
                    axis = axis / axis_norm
                turn_angle = min(phi, gated_turn * dt)
                hdg = rotate_about(hdg, axis, turn_angle)

        out_vel[global_i] = hdg * new_speed

    return out_vel


def _run_both(cfg, positions, velocities, k_neighbors=7):
    """Run both the reference and the real (vectorised) AngleMode on
    the same synthetic scenario. jitter_deg=0 so neither draws rng."""
    from pymurmur.physics.flock import PhysicsFlock

    cfg = copy(cfg)
    cfg.angle.jitter_deg = 0.0
    n = len(positions)
    flock = PhysicsFlock(cfg)
    flock.positions[:n] = positions
    flock.velocities[:n] = velocities
    flock.active[:] = False
    flock.active[:n] = True
    flock.accelerations[:] = 0.0
    flock.get_index().rebuild(flock.positions, flock.active)

    active_idx = np.where(flock.active)[0]
    k = min(k_neighbors + 1, len(active_idx))
    tree_positions = flock.positions[active_idx]
    from scipy.spatial import cKDTree
    tree = cKDTree(tree_positions)
    _, compacted = tree.query(tree_positions, k=k, workers=-1)
    if compacted.ndim == 1:
        compacted = compacted.reshape(-1, 1)
    nbr_idx = np.zeros((len(active_idx), k - 1), dtype=np.int32)
    for j in range(len(active_idx)):
        row = compacted[j, 1:k]
        nbr_idx[j, :len(row)] = active_idx[row]

    ref_vel = _reference_compute(
        flock.positions, flock.velocities, flock.active, nbr_idx, cfg,
    )

    AngleMode.compute(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng, flock.last_theta, cfg,
    )
    vec_vel = flock.velocities.copy()

    return ref_vel[:n], vec_vel[:n]


def test_vectorized_matches_reference_flee_scenario(default_config):
    """Tight cluster -> every bird's nearest neighbour is within
    sep_radius -> flee branch for all."""
    cfg = copy(default_config)
    cfg.mode = "angle"
    cfg.boundary_mode = "toroidal"
    cfg.num_boids = 8

    rng = np.random.default_rng(1)
    positions = np.array([500, 350, 200], dtype=np.float32) + rng.normal(
        scale=2.0, size=(8, 3),
    ).astype(np.float32)
    velocities = rng.normal(size=(8, 3)).astype(np.float32)
    velocities /= np.linalg.norm(velocities, axis=1, keepdims=True)
    velocities *= 4.0

    ref, vec = _run_both(cfg, positions, velocities)
    np.testing.assert_allclose(ref, vec, atol=1e-4)


def test_vectorized_matches_reference_align_cohere_scenario(default_config):
    """Moderately spread cluster -> nearest neighbour lands in the
    align+cohere band for most birds."""
    cfg = copy(default_config)
    cfg.mode = "angle"
    cfg.boundary_mode = "toroidal"
    cfg.num_boids = 10

    rng = np.random.default_rng(2)
    positions = np.array([500, 350, 200], dtype=np.float32) + rng.normal(
        scale=25.0, size=(10, 3),
    ).astype(np.float32)
    velocities = rng.normal(size=(10, 3)).astype(np.float32)
    velocities /= np.linalg.norm(velocities, axis=1, keepdims=True)
    velocities *= 4.0

    ref, vec = _run_both(cfg, positions, velocities)
    np.testing.assert_allclose(ref, vec, atol=1e-4)


def test_vectorized_matches_reference_cohere_only_scenario(default_config):
    """Sparse cluster -> nearest neighbour lands in the cohere-only band."""
    cfg = copy(default_config)
    cfg.mode = "angle"
    cfg.boundary_mode = "toroidal"
    cfg.num_boids = 10

    rng = np.random.default_rng(3)
    positions = np.array([500, 350, 200], dtype=np.float32) + rng.normal(
        scale=60.0, size=(10, 3),
    ).astype(np.float32)
    velocities = rng.normal(size=(10, 3)).astype(np.float32)
    velocities /= np.linalg.norm(velocities, axis=1, keepdims=True)
    velocities *= 4.0

    ref, vec = _run_both(cfg, positions, velocities)
    np.testing.assert_allclose(ref, vec, atol=1e-4)


def test_vectorized_matches_reference_margin_edge_combined_with_neighbours(default_config):
    """Some birds near a cube face (edge target), combined with a
    tight-ish cluster (neighbour target) -> exercises the
    target+edge_target combination branch."""
    cfg = copy(default_config)
    cfg.mode = "angle"
    cfg.boundary_mode = "margin"
    cfg.boundary_margin = 50.0
    cfg.num_boids = 8

    rng = np.random.default_rng(4)
    positions = np.array([20.0, 350, 200], dtype=np.float32) + rng.normal(
        scale=15.0, size=(8, 3),
    ).astype(np.float32)
    positions[:, 0] = np.clip(positions[:, 0], 1.0, 45.0)  # keep near the x=0 face
    velocities = rng.normal(size=(8, 3)).astype(np.float32)
    velocities /= np.linalg.norm(velocities, axis=1, keepdims=True)
    velocities *= 4.0

    ref, vec = _run_both(cfg, positions, velocities)
    np.testing.assert_allclose(ref, vec, atol=1e-4)


def test_vectorized_matches_reference_sphere_edge_no_neighbours(default_config):
    """Widely separated birds near the sphere boundary -> edge-only
    target (no neighbour target), each bird effectively isolated.

    Heading is radially outward PLUS a small tangential perturbation,
    not exactly radial -- heading exactly opposite the (radially
    inward) edge target puts the Rodrigues rotation axis
    (cross(hdg, target)) exactly at the zero vector, an inherently
    ill-conditioned case (any perpendicular direction is equally valid
    mathematically) where the degenerate-axis fallback in *either* a
    scalar or vectorised implementation is free to round to a
    different, still-mathematically-valid axis -- not a vectorization
    bug, just not a meaningful thing to bit-compare. A small tangential
    component avoids the degeneracy while keeping the scenario
    (isolated bird near a sphere boundary) realistic."""
    cfg = copy(default_config)
    cfg.mode = "angle"
    cfg.boundary_mode = "sphere"
    cfg.boundary_sphere_radius = 200.0
    cfg.boundary_margin = 50.0
    cfg.num_boids = 6

    rng = np.random.default_rng(5)
    dirs = rng.normal(size=(6, 3)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    positions = dirs * 190.0  # near the R=200 boundary, widely separated angularly

    tangential = rng.normal(size=(6, 3)).astype(np.float32)
    tangential -= (np.sum(tangential * dirs, axis=1, keepdims=True)) * dirs  # project out radial component
    tangential /= np.linalg.norm(tangential, axis=1, keepdims=True)
    heading = dirs * 0.9 + tangential * 0.1  # mostly outward, not exactly
    heading /= np.linalg.norm(heading, axis=1, keepdims=True)
    velocities = heading * 4.0

    ref, vec = _run_both(cfg, positions, velocities)
    np.testing.assert_allclose(ref, vec, atol=1e-4)


def test_vectorized_matches_reference_quadratic_and_softened_speed_modes(default_config):
    """Isolated birds (deficit > 0) under quadratic/softened adaptive
    speed laws — exercises both non-default branches of Stage 3."""
    for mode in ("quadratic", "softened"):
        cfg = copy(default_config)
        cfg.mode = "angle"
        cfg.boundary_mode = "toroidal"
        cfg.angle.angle_speed_mode = mode
        cfg.num_boids = 5

        rng = np.random.default_rng(6)
        # Widely separated -> isolated (deficit = angle_neighbors > 0)
        positions = rng.uniform(100, 900, size=(5, 3)).astype(np.float32)
        velocities = rng.normal(size=(5, 3)).astype(np.float32)
        velocities /= np.linalg.norm(velocities, axis=1, keepdims=True)
        velocities *= 4.0

        ref, vec = _run_both(cfg, positions, velocities)
        np.testing.assert_allclose(ref, vec, atol=1e-4, err_msg=f"mode={mode}")
