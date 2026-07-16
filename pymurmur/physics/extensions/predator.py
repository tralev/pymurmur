"""Predator / Threat extension — full FSM + force bundle (P3.8–P3.9).

P3.8: Bounded panic + blackening
    panic = clamp(prox_i, 0, 1) · threat_strength
    boost = panic · (0.72 + wave_gain·0.18 + vacuole_strength·0.12)
    max_speed_i = v0 · (1 + min(1.35, boost))    [ceiling raise, NOT compound]
    black = 1 + blackening_gain · prox_i · 0.85
    sep_eff = separation · (2 − black)           [weaker near threat]
    coh_eff = cohesion · black                   [stronger near threat]

P3.9: Threat FSM + force bundle
    capture = max(0.18, threat_radius·0.72)·U
    pass_dist = (0.92 + threat_radius·2.6 + momentum·1.32)·U
    clear = pass_dist·(0.72 + momentum·0.16)
    Approach→egress: ||p_threat−center|| ≤ capture
    Egress→approach: ||p_threat−center|| > clear AND dot(dir, to_center̂) < −0.12
    Force bundle: push, wake, split, wave
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from ._base import Extension

if TYPE_CHECKING:
    from ...core.config import SimConfig
    from ..flock import PhysicsFlock
    from ._base import StepContext


def _rotate_toward(
    current: np.ndarray,
    target: np.ndarray,
    max_angle: float,
) -> np.ndarray:
    """Rodrigues rotation of current toward target, capped at max_angle (rad).

    Returns unit vector.  Falls back to any perpendicular axis when
    current is (anti-)parallel to target.
    """
    cur = current / max(np.linalg.norm(current), 1e-10)
    tgt = target / max(np.linalg.norm(target), 1e-10)

    cos_angle = np.clip(np.dot(cur, tgt), -1.0, 1.0)
    angle = np.arccos(cos_angle)

    if angle < 1e-10 or angle <= max_angle:
        return tgt.astype(np.float32)

    # Cap rotation
    rot_angle = max_angle

    # Rotation axis
    k = np.cross(cur, tgt)
    k_norm = np.linalg.norm(k)
    if k_norm < 1e-10:
        # Anti-parallel or parallel: pick any perpendicular axis
        if abs(cur[0]) < 0.9:
            k = np.cross(cur, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        else:
            k = np.cross(cur, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        k = k / max(np.linalg.norm(k), 1e-10)
    else:
        k = k / k_norm

    # Rodrigues: v_rot = v·cosθ + (k×v)·sinθ + k·(k·v)·(1−cosθ)
    cos_t = np.cos(rot_angle)
    sin_t = np.sin(rot_angle)
    v_rot = (cur * cos_t
             + np.cross(k, cur) * sin_t
             + k * np.dot(k, cur) * (1.0 - cos_t))

    return (v_rot / max(np.linalg.norm(v_rot), 1e-10)).astype(np.float32)


class Predator(Extension):
    """Autonomous threat agent with full approach/egress FSM (P3.8–P3.9).

    Publishes ctx.threat_prox (per-bird proximity in [0,1]) for
    panic/blackening consumption by field mode and other modes.
    """

    def __init__(self, config: SimConfig) -> None:
        self._pos = np.zeros(3, dtype=np.float32)
        self._vel = np.zeros(3, dtype=np.float32)
        self._dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self._turn_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self._phase: str = "approach"
        # Place predator at a random edge position
        self._pos = np.array([
            config.width * 0.2,
            config.height * 0.5,
            config.depth * 0.5,
        ], dtype=np.float32)

    def apply(self, flock: PhysicsFlock, ctx: StepContext) -> None:
        """Update threat state and apply force bundle to nearby birds."""
        active = flock.active
        n_active = active.sum()
        if n_active == 0:
            ctx.threat_prox = np.zeros(flock.N_capacity, dtype=np.float32)
            return

        cfg = ctx.config
        dt = ctx.dt
        center = np.mean(flock.positions[active], axis=0)

        # ── Config values ──
        threat_radius = getattr(cfg, 'predator_threat_radius', 12.0)
        threat_strength = getattr(cfg, 'predator_strength', 1.0)
        momentum = getattr(cfg, 'predator_momentum', 0.5)
        split_gain = getattr(cfg, 'predator_split_gain', 0.5)
        acceleration = getattr(cfg, 'predator_acceleration', 0.8)
        vacuole_strength = getattr(cfg, 'predator_vacuole_strength', 0.0)
        wave_gain = getattr(cfg, 'field_wave_gain', 0.5)

        # ── Unit scale ──
        unit_scale = getattr(cfg, 'field_unit_scale', None)
        U = float(unit_scale) if unit_scale is not None else (
            0.4 * min(cfg.width, cfg.height, cfg.depth)
        )

        # ── FSM distances ──
        capture_dist = max(0.18, threat_radius * 0.72) * U
        pass_dist = (0.92 + threat_radius * 2.6 + momentum * 1.32) * U
        clear_dist = pass_dist * (0.72 + momentum * 0.16)

        # Speed
        v0 = cfg.v0
        speed = 2.0 * v0 * (1.0 + 0.5 * momentum)

        # ── Phase transitions ──
        dist_to_center = np.linalg.norm(self._pos - center)
        to_center = center - self._pos
        to_center_dir = to_center / max(np.linalg.norm(to_center), 1e-6)

        if self._phase == "approach":
            if dist_to_center <= capture_dist:
                self._phase = "egress"
        else:  # egress
            dot_check = np.dot(self._dir, to_center_dir)
            if dist_to_center > clear_dist and dot_check < -0.12:
                self._phase = "approach"

        # ── Target selection ──
        # ── Turn rate: different for approach vs egress (P3.9) ──
        if self._phase == "approach":
            turn_rate = (0.54 + acceleration * 0.025) * (1.0 - momentum * 0.24)
        else:
            turn_rate = 0.42 * (1.0 - momentum * 0.24)
        max_turn = turn_rate * dt

        if self._phase == "approach":
            target_point = center
        else:
            # Egress: target beyond center with arc offset (P3.9)
            base_target = center + self._dir * pass_dist
            # Arc lift + drift
            broad = pass_dist * (0.24)  # orbit arc scale
            t = ctx.frame * ctx.dt
            lift = self._turn_axis * np.sin(t * 0.18 + 0.7) * broad
            drift = np.cross(self._turn_axis, self._dir)
            drift_norm = np.linalg.norm(drift)
            if drift_norm > 1e-6:
                drift = drift / drift_norm * np.cos(t * 0.13 + 1.4) * broad * 0.72
            else:
                drift = np.zeros(3, dtype=np.float32)
            target_point = base_target + lift.astype(np.float32) + drift.astype(np.float32)

        # ── Steer toward target ──
        desired_dir = target_point - self._pos
        desired_dir = desired_dir / max(np.linalg.norm(desired_dir), 1e-6)

        # Rotate heading toward desired direction (capped)
        self._dir = _rotate_toward(self._dir, desired_dir, max_turn)

        # ── Move ──
        self._vel = self._dir * speed
        self._pos += self._vel * dt

        # ── Force bundle on birds within threat radius ──
        threat_dist = threat_radius * U * 2.0  # influence radius
        active_idx = np.where(active)[0]
        positions = flock.positions[active_idx]
        threat_prox = np.zeros(flock.N_capacity, dtype=np.float32)

        to_threat = positions - self._pos
        d = np.linalg.norm(to_threat, axis=1)
        safe_d = np.maximum(d, 1e-6)
        away_dir = to_threat / safe_d[:, np.newaxis]

        within = d < threat_dist
        if not within.any():
            ctx.threat_prox = threat_prox
            return

        # ── Proximity ──
        prox = 1.0 - d / threat_dist
        prox = np.clip(prox, 0.0, 1.0)
        broad = np.sqrt(prox + 1e-6)

        # Populate threat_prox for ctx
        threat_prox[active_idx] = prox

        # ── Push (radial away from threat) ──
        push = away_dir * threat_strength * (2.5 + vacuole_strength * 1.7) * broad[:, np.newaxis]

        # ── Wake (drag along threat's path) ──
        v_threat_mag = np.linalg.norm(self._vel)
        wake_dir = away_dir - self._dir.reshape(1, 3) * 0.35
        wake_scale = min(1.8, v_threat_mag / max(v0, 1e-6)) * threat_strength * broad * 0.42
        wake = wake_dir * wake_scale[:, np.newaxis]

        # ── Split (horizontal tear, z-up) ──
        split = np.column_stack([
            -away_dir[:, 1] * 1.45,
            away_dir[:, 0] * 1.45,
            away_dir[:, 2] * 0.28,
        ]).astype(np.float32) * (split_gain * broad)[:, np.newaxis]

        # ── Wave (velocity-aligned perturbation) ──
        v_norms = np.linalg.norm(flock.velocities[active_idx], axis=1, keepdims=True)
        v_dirs = flock.velocities[active_idx] / np.maximum(v_norms, 1e-6)
        wave = v_dirs * (wave_gain * broad * 0.22)[:, np.newaxis]

        # ── Accumulate ──
        force_mask = active_idx[within]
        flock.accelerations[force_mask] += (
            push[within] + wake[within] + split[within] + wave[within]
        )

        # ── P3.8: Panic ceiling (ceiling raise, NOT compound multiply) ──
        panic = prox * threat_strength
        boost = panic * (0.72 + wave_gain * 0.18 + vacuole_strength * 0.12)
        speed_mult = 1.0 + np.minimum(1.35, boost)

        if flock.max_speed is None:
            flock.max_speed = np.full(flock.N_capacity, v0, dtype=np.float32)

        flock.max_speed[active_idx[within]] = np.maximum(
            flock.max_speed[active_idx[within]],
            v0 * speed_mult[within],
        )

        # ── P3.8: Blackening (weaken sep, strengthen coh near threat) ──
        blackening_gain = getattr(cfg, 'predator_blackening_gain', 0.6)
        black = 1.0 + blackening_gain * prox * 0.85
        # Store on config for field mode to read and modulate separation/cohesion
        cfg._threat_blackening = black.astype(np.float32)
        cfg._threat_active = active_idx[within].astype(np.int32)
        cfg._threat_present = True

        # ── Publish threat_prox ──
        ctx.threat_prox = threat_prox
