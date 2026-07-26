"""Cross-field validation for the opt-in extension/plugin toggles.

Split out of config_validation.py (file-size split) — this was the
"Extensions cross-field" section of what used to be
_validate_refinements_angle_extensions, extracted verbatim as this
session's extension count grew (predator, speed_noise,
neighbor_adaptive_speed, dynamic_vision_range, boid_state_machine) and
pushed the parent file over the line-count guard. No behavior change —
same conditions, same messages, same order.
"""
from __future__ import annotations

from typing import Callable

_OkFn = Callable[[str], bool]


def _validate_plugin_extensions(cfg, ok: _OkFn) -> list[str]:
    """predator_mode/predator_enabled and every later *_enabled opt-in
    extension's own cross-field rules, gated the same way."""
    issues: list[str] = []

    if cfg.predator_mode not in cfg._VALID_PREDATOR_MODES:
        issues.append(
            f"predator_mode must be one of {cfg._VALID_PREDATOR_MODES}, "
            f"got {cfg.predator_mode!r}"
        )
    if cfg.predator_enabled:
        if (
            ok("predator_threat_radius")
            and cfg.predator_threat_radius <= 0
        ):
            issues.append(
                "predator_enabled=True but predator_threat_radius must be > 0"
            )
        if ok("predator_strength") and cfg.predator_strength <= 0:
            issues.append(
                "predator_enabled=True but predator_strength must be > 0"
            )
    if cfg.speed_noise_enabled:
        if ok("speed_noise_frequency") and cfg.speed_noise_frequency <= 0:
            issues.append(
                "speed_noise_enabled=True but speed_noise_frequency must be > 0"
            )
        if ok("speed_noise_min_mult") and cfg.speed_noise_min_mult < 0:
            issues.append(
                "speed_noise_enabled=True but speed_noise_min_mult must be >= 0"
            )
        if (
            ok("speed_noise_min_mult") and ok("speed_noise_max_mult")
            and cfg.speed_noise_min_mult > cfg.speed_noise_max_mult
        ):
            issues.append(
                "speed_noise_min_mult must be <= speed_noise_max_mult"
            )
    if cfg.neighbor_adaptive_speed_enabled:
        if (
            ok("neighbor_adaptive_speed_target")
            and cfg.neighbor_adaptive_speed_target < 1
        ):
            issues.append(
                "neighbor_adaptive_speed_enabled=True but "
                "neighbor_adaptive_speed_target must be >= 1"
            )
        if (
            ok("neighbor_adaptive_speed_radius")
            and cfg.neighbor_adaptive_speed_radius <= 0
        ):
            issues.append(
                "neighbor_adaptive_speed_enabled=True but "
                "neighbor_adaptive_speed_radius must be > 0"
            )
        # Reuses angle.py's speed-mode vocabulary (_VALID_ANGLE_SPEED_MODES)
        # since it's the same linear/quadratic/softened law.
        if cfg.neighbor_adaptive_speed_mode not in cfg._VALID_ANGLE_SPEED_MODES:
            issues.append(
                f"neighbor_adaptive_speed_mode must be one of "
                f"{cfg._VALID_ANGLE_SPEED_MODES}, "
                f"got {cfg.neighbor_adaptive_speed_mode!r}"
            )
    if cfg.dynamic_vision_range_enabled:
        if ok("dynamic_vision_range_ideal_count") and cfg.dynamic_vision_range_ideal_count <= 0:
            issues.append(
                "dynamic_vision_range_enabled=True but "
                "dynamic_vision_range_ideal_count must be > 0"
            )
        if ok("dynamic_vision_range_step") and cfg.dynamic_vision_range_step <= 0:
            issues.append(
                "dynamic_vision_range_enabled=True but "
                "dynamic_vision_range_step must be > 0"
            )
        if (
            ok("dynamic_vision_range_min_mult") and ok("dynamic_vision_range_max_mult")
            and cfg.dynamic_vision_range_min_mult > cfg.dynamic_vision_range_max_mult
        ):
            issues.append(
                "dynamic_vision_range_min_mult must be <= dynamic_vision_range_max_mult"
            )
        if ok("dynamic_vision_range_sample_k") and cfg.dynamic_vision_range_sample_k < 1:
            issues.append(
                "dynamic_vision_range_enabled=True but "
                "dynamic_vision_range_sample_k must be >= 1"
            )
    if cfg.boid_state_machine_enabled:
        if ok("boid_state_neighbor_radius") and cfg.boid_state_neighbor_radius <= 0:
            issues.append(
                "boid_state_machine_enabled=True but "
                "boid_state_neighbor_radius must be > 0"
            )
        if ok("boid_state_sample_k") and cfg.boid_state_sample_k < 1:
            issues.append(
                "boid_state_machine_enabled=True but boid_state_sample_k must be >= 1"
            )
        if (
            ok("boid_state_isolated_neighbor_threshold")
            and cfg.boid_state_isolated_neighbor_threshold < 0
        ):
            issues.append(
                "boid_state_isolated_neighbor_threshold must be >= 0"
            )
        if ok("boid_state_isolated_speed_mult") and cfg.boid_state_isolated_speed_mult < 0:
            issues.append("boid_state_isolated_speed_mult must be >= 0")
        if (
            ok("boid_state_crowded_neighbor_threshold")
            and cfg.boid_state_crowded_neighbor_threshold < 0
        ):
            issues.append(
                "boid_state_crowded_neighbor_threshold must be >= 0"
            )
        if ok("boid_state_crowded_speed_mult") and cfg.boid_state_crowded_speed_mult < 0:
            issues.append("boid_state_crowded_speed_mult must be >= 0")
        if (
            ok("boid_state_threatened_proximity_threshold")
            and not (0.0 <= cfg.boid_state_threatened_proximity_threshold <= 1.0)
        ):
            issues.append(
                "boid_state_threatened_proximity_threshold must be in [0, 1]"
            )
        if ok("boid_state_threatened_speed_mult") and cfg.boid_state_threatened_speed_mult < 0:
            issues.append("boid_state_threatened_speed_mult must be >= 0")

    return issues
