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

from typing import TYPE_CHECKING

import numpy as np

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

    @property
    def position(self) -> np.ndarray:
        """D7/S2.A8: current world-space threat position, for rendering."""
        return self._pos.copy()

    def apply(self, flock: PhysicsFlock, ctx: StepContext) -> None:
        """Update threat state and apply force bundle to nearby birds.

        C1: predator_mode selects the targeting strategy —
        "autonomous" (default) is the original approach/egress FSM,
        unchanged. "off" freezes the threat (no movement, no force) but
        keeps its state alive, unlike predator_enabled=False which never
        instantiates the extension. "orbit" always uses the egress
        arc-offset targeting, skipping the capture/egress FSM. "cursor"
        targets the viz-supplied `_cursor_world_pos` bridge directly when
        set, falling back to full autonomous behaviour when it isn't
        (e.g. headless runs with no viz layer).
        """
        active = flock.active
        n_active = active.sum()
        cfg = ctx.config
        if n_active == 0:
            ctx.threat_prox = np.zeros(flock.N_capacity, dtype=np.float32)
            cfg._threat_present = False
            return

        mode = cfg.predator.predator_mode

        if mode == "off":
            ctx.threat_prox = np.zeros(flock.N_capacity, dtype=np.float32)
            cfg._threat_present = False
            return

        dt = ctx.dt
        center = np.mean(flock.positions[active], axis=0)

        # ── Config values ──
        threat_radius = cfg.predator.predator_threat_radius
        threat_strength = cfg.predator.predator_strength
        momentum = cfg.predator.predator_momentum
        split_gain = cfg.predator.predator_split_gain
        acceleration = cfg.predator.predator_acceleration
        vacuole_strength = cfg.predator.predator_vacuole_strength
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

        # ── S2.A8: sign-aligned EMA turn axis ──
        # desired = normalize(dir × to_center̂); flip sign toward the
        # previous axis (the cross product's sign flips discontinuously
        # as the threat crosses the centre line, which would otherwise
        # jerk the egress arc's lift/drift direction each such crossing).
        raw_axis = np.cross(self._dir, to_center_dir)
        raw_axis_norm = np.linalg.norm(raw_axis)
        if raw_axis_norm > 1e-8:
            desired_axis = raw_axis / raw_axis_norm
            if np.dot(self._turn_axis, desired_axis) < 0.0:
                desired_axis = -desired_axis
            amt = 0.15  # per-frame EMA blend rate
            blended = self._turn_axis * (1.0 - amt) + desired_axis * amt
            blended_norm = np.linalg.norm(blended)
            if blended_norm > 1e-8:
                self._turn_axis = (blended / blended_norm).astype(np.float32)

        # C1: cursor mode targets the bridge position directly when live;
        # the FSM phase update is skipped in that case (phase is only used
        # for turn_rate below, and stays frozen at its last value).
        _raw_cursor = getattr(cfg, '_cursor_world_pos', None) if mode == "cursor" else None
        cursor_target: np.ndarray | None = (
            np.asarray(_raw_cursor, dtype=np.float32) if _raw_cursor is not None else None
        )

        if mode == "orbit":
            self._phase = "egress"  # C1: always orbit, never capture
        elif cursor_target is None:
            # "autonomous", and "cursor" without a live bridge (fallback).
            if self._phase == "approach":
                if dist_to_center <= capture_dist:
                    self._phase = "egress"
            else:  # egress
                dot_check = np.dot(self._dir, to_center_dir)
                if dist_to_center > clear_dist and dot_check < -0.12:
                    self._phase = "approach"

        # ── Turn rate: different for approach vs egress (P3.9) ──
        if self._phase == "approach":
            turn_rate = (0.54 + acceleration * 0.025) * (1.0 - momentum * 0.24)
            # S2.A8: steer-response multiplier — approach turns are far
            # more responsive than the raw turn_rate cap alone implies
            # (the threat commits hard to closing the distance).
            steer_response = 1.86 + (1.0 - momentum) * 0.48
        else:
            turn_rate = 0.42 * (1.0 - momentum * 0.24)
            # S2.A8: egress is comparatively sluggish to turn — it's
            # riding the arc out, not actively hunting.
            steer_response = 0.34 + (1.0 - momentum) * 0.44
        max_turn = turn_rate * steer_response * dt

        # ── Target selection ──
        if cursor_target is not None:
            target_point = cursor_target
        elif self._phase == "approach":
            target_point = center
        else:
            # Egress/orbit: target beyond center with arc offset (P3.9)
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
        desired_dir = desired_dir / max(float(np.linalg.norm(desired_dir)), 1e-6)

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
            cfg._threat_present = False
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
        blackening_gain = cfg.predator.predator_blackening_gain
        black = 1.0 + blackening_gain * prox * 0.85
        # Store on config for field mode to read and modulate separation/cohesion
        cfg._threat_blackening = black.astype(np.float32)
        cfg._threat_active = active_idx[within].astype(np.int32)
        cfg._threat_present = True

        # ── Publish threat_prox ──
        ctx.threat_prox = threat_prox
