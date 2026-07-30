"""SimConfig YAML I/O — from_file/to_file, extracted as standalone
functions (config_cls/cfg passed explicitly rather than importing
SimConfig, avoiding a circular import with config.py).

Extracted from config.py (file-size split).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .config_field_map import _ALL_FIELD_NAMES, _FIELD_MAP, _NESTED_ONLY

# G5: Top-level YAML keys that are deliberately NOT SimConfig fields —
# consumed by a separate loader instead (e.g. analysis/evoflock.py's
# load_obstacle_scene reads `obstacles:` directly from the raw YAML).
# Exempted from from_file's unknown-key check by *name*, not by value
# shape, so a typo'd list key is still caught.
_NON_FIELD_TOP_LEVEL_LISTS: set[str] = {"obstacles"}


# Known tuple-typed fields that need YAML round-trip coercion.
# YAML loads sequences as Python lists; these fields expect tuples.
_TUPLE_FIELDS: set[str] = {
    "background_top", "background_bottom",
    "field_drift_direction", "ecology_roost",
    "influencer_target_freq_primary", "influencer_target_freq_secondary",
    "influencer_target_amp_primary", "influencer_target_amp_secondary",
    "influencer_target_phase_offsets",
}


def _coerce_tuples(cfg) -> None:
    """Post-load: cast list values to tuples for tuple-typed dataclass fields.

    YAML parses sequences as Python lists, but dataclass type hints expect
    tuples.  Walk each known tuple field and coerce list→tuple.
    Done after the config is fully constructed, so setattr delegation works.
    """
    for flat_name in _TUPLE_FIELDS:
        sub_attr, field_name = _FIELD_MAP[flat_name]
        sub_cfg = object.__getattribute__(cfg, sub_attr)
        val = getattr(sub_cfg, field_name)
        if isinstance(val, list):
            object.__setattr__(sub_cfg, field_name, tuple(val))


def load_config_from_file(config_cls, path: str | Path, strict: bool = True):
    """Load config from a YAML file. Nested keys are flattened.

    Args:
        path: YAML file path.
        strict: if True (default), unknown section keys raise ValueError
                naming the section and key (G5). Set False for configs
                that carry extra sections (e.g. evoflock GA parameters).

    Raises FileNotFoundError if path doesn't exist.
    Raises ValueError if strict=True and unknown keys are found.
    """
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = yaml.safe_load(path.read_text()) or {}
    flat: dict[str, Any] = {}

    # Flatten nested sections with section-aware key normalisation
    unknown_keys: list[tuple[str, str]] = []  # G5: track unknown for actionable errors
    for section_name, section_data in raw.items():
        if isinstance(section_data, dict):
            for key, value in section_data.items():
                # Normalise short keys: field/noise → field_noise,
                # wander/attractor_speed → wander_attractor_speed, etc.
                if not key.startswith(f"{section_name}_"):
                    prefixed = f"{section_name}_{key}"
                    if prefixed in _ALL_FIELD_NAMES:
                        key = prefixed
                # Special case: 'performance'/'metrics' → metrics_ prefix
                if section_name in ("performance", "metrics"):
                    if not key.startswith("metrics_"):
                        metrics_key = f"metrics_{key}"
                        if metrics_key in _ALL_FIELD_NAMES:
                            key = metrics_key
                # G5: Track unknown keys for actionable error messages
                if key in _ALL_FIELD_NAMES:
                    flat[key] = value
                else:
                    unknown_keys.append((section_name, key))
        elif isinstance(section_data, list):
            # Non-field top-level list, e.g. `obstacles:` (a scene spec
            # consumed separately by analysis/evoflock.py's
            # load_obstacle_scene, not a SimConfig field). Only the
            # known name is exempt — a typo'd list key (e.g.
            # `obstalces:`) is still a mistake worth surfacing.
            if section_name not in _NON_FIELD_TOP_LEVEL_LISTS:
                unknown_keys.append(("<top-level>", section_name))
        else:
            # G5: top-level (non-nested) scalar key, e.g. `mode: spatial`.
            # Must still be validated — an unrecognized one (typo'd
            # section/field name) was previously swallowed silently.
            if section_name in _ALL_FIELD_NAMES:
                flat[section_name] = section_data
            else:
                unknown_keys.append(("<top-level>", section_name))

    # G5: Actionable YAML errors — name the offending key AND section
    if strict and unknown_keys:
        lines = [f"  [{sec}] {key}" for sec, key in unknown_keys]
        raise ValueError(
            f"Unknown config keys in {path.name}:\n"
            + "\n".join(lines)
            + f"\n\nKnown fields (non-exhaustive): {sorted(_ALL_FIELD_NAMES)[:20]}..."
        )

    # Nested-only fields (flat shim retired) — route explicitly
    nested_vals = {
        key: flat.pop(key) for key in list(_NESTED_ONLY) if key in flat
    }

    # Filter to known fields only
    filtered = {k: v for k, v in flat.items() if k in _ALL_FIELD_NAMES}

    cfg = config_cls(**filtered)
    for key, value in nested_vals.items():
        sub_attr, field_name = _NESTED_ONLY[key]
        setattr(getattr(cfg, sub_attr), field_name, value)

    # YAML round-trip: coerce lists back to tuples for tuple-typed
    # dataclass fields (background_top/bottom, ecology_roost, etc.)
    _coerce_tuples(cfg)
    return cfg


def save_config_to_file(cfg, path: str | Path) -> None:
    """Write config to a YAML file. Round-trip preserves all fields."""
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {
        "domain": {"width": cfg.width, "height": cfg.height,
                   "depth": cfg.depth},
        "flock": {"num_boids": cfg.num_boids, "boid_size": cfg.boid_size,
                  "v0": cfg.v0, "max_force": cfg.max_force,
                  "dt_phys": cfg.dt_phys,
                  "speed_min_factor": cfg.speed_min_factor,
                  "n_predators": cfg.n_predators,
                  "visual_range": cfg.visual_range,
                  "velocity_damping": cfg.velocity_damping},
        "mode": cfg.mode,
        "projection": {"phi_p": cfg.projection.phi_p, "phi_a": cfg.phi_a,
                       "sigma": cfg.sigma,
                       "max_visibility": cfg.max_visibility,
                       "max_occlusion_neighbors": cfg.max_occlusion_neighbors,
                       "projection_heading_inertia": cfg.projection_heading_inertia},
        "spatial": {"separation_weight": cfg.separation_weight,
                    "alignment_weight": cfg.alignment_weight,
                    "cohesion_weight": cfg.cohesion_weight,
                    "noise_scale": cfg.noise_scale,
                    "noise_mode": cfg.noise_mode,
                    "acceleration_scale": cfg.acceleration_scale,
                    "influence_count": cfg.influence_count,
                    "speed_mode": cfg.speed_mode,
                    "flow_weight": cfg.flow_weight,
                    "neighbor_filter": cfg.neighbor_filter,
                    "separation_kernel": cfg.separation_kernel,
                    "separation_kernel_radius": cfg.separation_kernel_radius,
                    "cohesion_kernel": cfg.cohesion_kernel,
                    "alignment_kernel": cfg.alignment_kernel,
                    "kernel_zone_width": cfg.kernel_zone_width,
                    "alignment_radius_ratio": cfg.alignment_radius_ratio,
                    "separation_distance": cfg.separation_distance,
                    "max_dist_sep": cfg.max_dist_sep,
                    "max_dist_align": cfg.max_dist_align,
                    "max_dist_coh": cfg.max_dist_coh,
                    "angle_sep": cfg.angle_sep,
                    "angle_align": cfg.angle_align,
                    "angle_coh": cfg.angle_coh,
                    "coherence_factor": cfg.coherence_factor,
                    "w_fwd": cfg.w_fwd,
                    "predator_escape_factor": cfg.predator_escape_factor,
                    "predator_speed_boost": cfg.predator_speed_boost,
                    "predator_perception_boost": cfg.predator_perception_boost,
                    "predator_accel_boost": cfg.predator_accel_boost,
                    "jitter_separation": cfg.jitter_separation,
                    "jitter_cohesion": cfg.jitter_cohesion,
                    "jitter_alignment": cfg.jitter_alignment,
                    "static_avoid_weight": cfg.static_avoid_weight,
                    "predictive_avoid_weight": cfg.predictive_avoid_weight,
                    "fly_away_max_dist": cfg.fly_away_max_dist,
                    "min_time_to_collide": cfg.min_time_to_collide},
        "boundary": {"boundary_mode": cfg.boundary_mode,
                     "boundary_sphere_radius": cfg.boundary_sphere_radius,
                     "boundary_avoidance_factor": cfg.boundary_avoidance_factor,
                     "boundary_radius_factor": cfg.boundary_radius_factor,
                     "boundary_margin": cfg.boundary_margin},
        "refinements": {"refinements": cfg.refinements,
                        "steric": cfg.steric,
                        "blind_deg": cfg.blind_deg,
                        "anisotropy": cfg.anisotropy,
                        "steric_radius": cfg.steric_radius,
                        "steric_visible_only": cfg.steric_visible_only},
        "extensions": {"predator_enabled": cfg.predator_enabled,
                       "roosting_enabled": cfg.roosting_enabled,
                       "wander_enabled": cfg.wander_enabled,
                       "ripple_enabled": cfg.ripple_enabled,
                       "speed_noise_enabled": cfg.speed_noise_enabled,
                       "priority_stack_enabled": cfg.priority_stack_enabled,
                       "neighbor_adaptive_speed_enabled": cfg.neighbor_adaptive_speed_enabled,
                       "dynamic_vision_range_enabled": cfg.dynamic_vision_range_enabled,
                       "boid_state_machine_enabled": cfg.boid_state_machine_enabled},
        "predator": {"predator_threat_radius": cfg.predator_threat_radius,
                     "predator_strength": cfg.predator_strength,
                     "predator_momentum": cfg.predator_momentum,
                     "predator_split_gain": cfg.predator_split_gain,
                     "predator_acceleration": cfg.predator_acceleration,
                     "predator_vacuole_strength": cfg.predator_vacuole_strength,
                     "predator_blackening_gain": cfg.predator_blackening_gain,
                     "predator_mode": cfg.predator_mode},
        "ecology": {"ecology_roost": list(cfg.ecology_roost),
                    "ecology_critical_mass": cfg.ecology_critical_mass,
                    "ecology_dusk_width": cfg.ecology_dusk_width,
                    "ecology_seasonal_amplitude": cfg.ecology_seasonal_amplitude,
                    "ecology_temperature_boost": cfg.ecology_temperature_boost,
                    "ecology_predator_presence": cfg.ecology_predator_presence},
        "roost": {"roost_z_target": cfg.roost.z_target},
        "vicsek": {"vicsek_couplage": cfg.vicsek_couplage,
                   "vicsek_diffusion": cfg.vicsek_diffusion,
                   "vicsek_radius_influence": cfg.vicsek_radius_influence,
                   "vicsek_radius_avoid": cfg.vicsek_radius_avoid,
                   "vicsek_velocity": cfg.vicsek_velocity,
                   "vicsek_time_step": cfg.vicsek_time_step,
                   "vicsek_radius_predators": cfg.vicsek_radius_predators,
                   "vicsek_velocity_predator": cfg.vicsek_velocity_predator,
                   "vicsek_detect_ratio": cfg.vicsek_detect_ratio,
                   "vicsek_weight_afraid": cfg.vicsek_weight_afraid,
                   "vicsek_predator_noise_ratio": cfg.vicsek_predator_noise_ratio,
                   "vicsek_heading_inertia": cfg.vicsek_heading_inertia},
        "influencer": {"influencer_rank_exponent": cfg.influencer_rank_exponent,
                       "influencer_substeps": cfg.influencer_substeps,
                       "influencer_scale": cfg.influencer_scale,
                       "influencer_influence_mode": cfg.influencer_influence_mode,
                       "influencer_near_dist_sq": cfg.influencer_near_dist_sq,
                       "influencer_init_separation": cfg.influencer_init_separation,
                       "influencer_tick_rate": cfg.influencer_tick_rate,
                       "influencer_target_freq_primary": list(cfg.influencer_target_freq_primary),
                       "influencer_target_freq_secondary": list(cfg.influencer_target_freq_secondary),
                       "influencer_target_amp_primary": list(cfg.influencer_target_amp_primary),
                       "influencer_target_amp_secondary": list(cfg.influencer_target_amp_secondary),
                       "influencer_target_vert_offset": cfg.influencer_target_vert_offset,
                       "influencer_target_phase_offsets": list(cfg.influencer_target_phase_offsets),
                       "influencer_influence_min": cfg.influencer_influence_min,
                       "influencer_influence_max": cfg.influencer_influence_max,
                       "influencer_use_rank_override": cfg.influencer_use_rank_override,
                       "influencer_move_then_steer": cfg.influencer_move_then_steer,
                       "influencer_density_scaled_init": cfg.influencer_density_scaled_init,
                       "influencer_pilot_enabled": cfg.influencer_pilot_enabled,
                       "influencer_pilot_speed": cfg.influencer_pilot_speed},
        "field": {"field_separation": cfg.field_separation,
                  "field_alignment": cfg.field_alignment,
                  "field_cohesion": cfg.field_cohesion,
                  "field_flow": cfg.field_flow,
                  "field_chase_strength": cfg.field_chase_strength,
                  "field_noise": cfg.field_noise,
                  "field_target_pull": cfg.field_target_pull,
                  "field_drift_pull": cfg.field_drift_pull,
                  "field_drift_direction": list(cfg.field_drift_direction),
                  "field_shell_influence": cfg.field_shell_influence,
                  "field_tangent_pull": cfg.field_tangent_pull,
                  "field_wave_gain": cfg.field_wave_gain,
                  "field_ripple_trains": cfg.field_ripple_trains,
                  "field_inertia": cfg.field_inertia,
                  "field_shell_radius_base": cfg.field_shell_radius_base,
                  "field_inner_radius_factor": cfg.field_inner_radius_factor,
                  "field_num_groups": cfg.field_num_groups,
                  "field_leader_fraction": cfg.field_leader_fraction,
                  "field_flow_pull": cfg.field_flow_pull,
                  "field_unit_scale": cfg.field_unit_scale,
                  "disabled_terms": list(cfg.disabled_terms) if cfg.disabled_terms else []},
        "index": {"spatial_index": cfg.spatial_index,
                  "topological_cap": cfg.topological_cap,
                  "use_toroidal_distance": cfg.use_toroidal_distance},
        "performance": {"target_fps": cfg.target_fps,
                        "metrics_detail_level": cfg.metrics_detail_level,
                        "metrics_interval": cfg.metrics_interval,
                        "instance_buffer_chunk": cfg.instance_buffer_chunk,
                        "parallel_workers": cfg.parallel_workers,
                        "bird_mass_kg": cfg.bird_mass_kg,
                        "cruise_speed_ms": cfg.cruise_speed_ms,
                        "acc_peak_ms2": cfg.acc_peak_ms2,
                        "history_cap": cfg.history_cap,
                        "use_numba": cfg.use_numba,
                        "fastmath": cfg.fastmath,
                        "num_threads": cfg.num_threads,
                        "adaptive_quality": cfg.adaptive_quality,
                        "readout_smooth": cfg.readout_smooth},
        "visual": {"fps": cfg.fps,
                   "window_width": cfg.window_width,
                   "window_height": cfg.window_height,
                   "show_grid": cfg.show_grid,
                   "auto_rotate": cfg.auto_rotate,
                   "theme": cfg.theme,
                   "point_sprites": cfg.point_sprites,
                   "winged_mesh": cfg.winged_mesh,
                   "gradient_sky": cfg.gradient_sky,
                   "dual_view": cfg.dual_view,
                   "trails": cfg.trails,
                   "trail_length": cfg.trail_length,
                   "density_mode": cfg.density_mode,
                   "density_alpha": cfg.density_alpha,
                   "per_bird_color": cfg.per_bird_color,
                   "background_top": list(cfg.background_top),
                   "background_bottom": list(cfg.background_bottom),
                   "bird_mesh": cfg.bird_mesh,
                   "flap_period": cfg.flap_period,
                   "hud": cfg.hud},
        "capture": {"capture_width": cfg.capture_width,
                    "capture_prewarm": cfg.capture_prewarm,
                    "capture_sweep": cfg.capture_sweep,
                    "capture_scale": cfg.capture_scale,
                    "capture_height": cfg.capture_height,
                    "capture_frames": cfg.capture_frames,
                    "capture_every": cfg.capture_every,
                    "capture_fps": cfg.capture_fps,
                    "capture_output": cfg.capture_output,
                    "capture_metrics_csv": cfg.capture_metrics_csv,
                    "capture_metrics_json": cfg.capture_metrics_json,
                    "capture_with_viz": cfg.capture_with_viz,
                    "capture_mpl_fallback": cfg.capture_mpl_fallback,
                    "capture_mpl_dpi": cfg.capture_mpl_dpi},
        "wander": {"wander_attractor_speed": cfg.wander_attractor_speed,
                  "wander_attractor_radius": cfg.wander_attractor_radius},
        "speed_noise": {"speed_noise_frequency": cfg.speed_noise_frequency,
                        "speed_noise_min_mult": cfg.speed_noise_min_mult,
                        "speed_noise_max_mult": cfg.speed_noise_max_mult,
                        "speed_noise_power": cfg.speed_noise_power,
                        "speed_noise_time_scale": cfg.speed_noise_time_scale},
        "neighbor_adaptive_speed": {
            "neighbor_adaptive_speed_target": cfg.neighbor_adaptive_speed_target,
            "neighbor_adaptive_speed_radius": cfg.neighbor_adaptive_speed_radius,
            "neighbor_adaptive_speed_mode": cfg.neighbor_adaptive_speed_mode,
            "neighbor_adaptive_speed_linear_scale": cfg.neighbor_adaptive_speed_linear_scale,
        },
        "dynamic_vision_range": {
            "dynamic_vision_range_ideal_count": cfg.dynamic_vision_range_ideal_count,
            "dynamic_vision_range_step": cfg.dynamic_vision_range_step,
            "dynamic_vision_range_min_mult": cfg.dynamic_vision_range_min_mult,
            "dynamic_vision_range_max_mult": cfg.dynamic_vision_range_max_mult,
            "dynamic_vision_range_sample_k": cfg.dynamic_vision_range_sample_k,
        },
        "boid_state_machine": {
            "boid_state_neighbor_radius": cfg.boid_state_neighbor_radius,
            "boid_state_sample_k": cfg.boid_state_sample_k,
            "boid_state_isolated_neighbor_threshold": cfg.boid_state_isolated_neighbor_threshold,
            "boid_state_isolated_speed_mult": cfg.boid_state_isolated_speed_mult,
            "boid_state_crowded_neighbor_threshold": cfg.boid_state_crowded_neighbor_threshold,
            "boid_state_crowded_speed_mult": cfg.boid_state_crowded_speed_mult,
            "boid_state_threatened_proximity_threshold": cfg.boid_state_threatened_proximity_threshold,
            "boid_state_threatened_speed_mult": cfg.boid_state_threatened_speed_mult,
        },
        "angle": {"turn_rate": cfg.turn_rate,
                  "max_turn_rate": cfg.max_turn_rate,
                  "turn_threshold": cfg.turn_threshold,
                  "jitter_deg": cfg.jitter_deg,
                  "base_speed": cfg.base_speed,
                  "angle_neighbors": cfg.angle_neighbors,
                  "sep_radius_bodies": cfg.sep_radius_bodies,
                  "align_radius_bodies": cfg.align_radius_bodies,
                  "range_radius_bodies": cfg.range_radius_bodies,
                  "angle_speed_mode": cfg.angle_speed_mode},
        "marl": {"marl_velocity_cap": cfg.marl_velocity_cap,
                 "marl_rule_weight": cfg.marl_rule_weight,
                 "marl_separation_radius": cfg.marl_separation_radius,
                 "marl_action_scale": cfg.marl_action_scale,
                 "marl_episode_steps": cfg.marl_episode_steps,
                 "marl_reward_w_a": cfg.marl_reward_w_a,
                 "marl_reward_w_c": cfg.marl_reward_w_c,
                 "marl_reward_w_L": cfg.marl_reward_w_L,
                 "marl_reward_w_b": cfg.marl_reward_w_b,
                 "marl_reward_w_z": cfg.marl_reward_w_z},
        "seed": cfg.seed,
        "velocity_init": cfg.velocity_init,
    }

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
