"""Flat-name <-> sub-config field mapping for SimConfig's
backward-compatible flat attribute access (config.width, config.v0, ...).

Extracted from config.py (file-size split).
"""
from __future__ import annotations

_FIELD_MAP: dict[str, tuple[str, str]] = {
    # DomainConfig
    "width": ("_domain", "width"),
    "height": ("_domain", "height"),
    "depth": ("_domain", "depth"),
    # FlockConfig
    "num_boids": ("_flock", "num_boids"),
    "boid_size": ("_flock", "boid_size"),
    "v0": ("_flock", "v0"),
    "max_force": ("_flock", "max_force"),
    "visual_range": ("_flock", "visual_range"),
    "dt_phys": ("_flock", "dt_phys"),  # P8.10
    "speed_min_factor": ("_flock", "speed_min_factor"),  # P11.5
    "n_predators": ("_flock", "n_predators"),  # C4
    # BoundaryConfig
    "boundary_mode": ("_boundary", "boundary_mode"),
    "boundary_sphere_radius": ("_boundary", "boundary_sphere_radius"),
    "boundary_avoidance_factor": ("_boundary", "boundary_avoidance_factor"),
    "boundary_radius_factor": ("_boundary", "boundary_radius_factor"),
    "boundary_margin": ("_boundary", "boundary_margin"),  # C1
    # ProjectionConfig
    "phi_a": ("_projection", "phi_a"),
    "sigma": ("_projection", "sigma"),
    "max_visibility": ("_projection", "max_visibility"),  # C4
    "max_occlusion_neighbors": ("_projection", "max_occlusion_neighbors"),  # C4
    # SpatialConfig
    "separation_weight": ("_spatial", "separation_weight"),
    "alignment_weight": ("_spatial", "alignment_weight"),
    "cohesion_weight": ("_spatial", "cohesion_weight"),
    "noise_scale": ("_spatial", "noise_scale"),
    "acceleration_scale": ("_spatial", "acceleration_scale"),
    "influence_count": ("_spatial", "influence_count"),
    "predator_escape_factor": ("_spatial", "predator_escape_factor"),
    "predator_speed_boost": ("_spatial", "predator_speed_boost"),
    "predator_perception_boost": ("_spatial", "predator_perception_boost"),
    "predator_accel_boost": ("_spatial", "predator_accel_boost"),
    "jitter_separation": ("_spatial", "jitter_separation"),
    "jitter_cohesion": ("_spatial", "jitter_cohesion"),
    "jitter_alignment": ("_spatial", "jitter_alignment"),
    # SpatialConfig — new leaves
    "noise_mode": ("_spatial", "noise_mode"),
    "speed_mode": ("_spatial", "speed_mode"),
    "flow_weight": ("_spatial", "flow_weight"),
    "neighbor_filter": ("_spatial", "neighbor_filter"),
    "separation_kernel": ("_spatial", "separation_kernel"),
    "separation_kernel_radius": ("_spatial", "separation_kernel_radius"),
    "cohesion_kernel": ("_spatial", "cohesion_kernel"),
    "alignment_radius_ratio": ("_spatial", "alignment_radius_ratio"),
    "separation_distance": ("_spatial", "separation_distance"),
    "max_dist_sep": ("_spatial", "max_dist_sep"),
    "max_dist_align": ("_spatial", "max_dist_align"),
    "max_dist_coh": ("_spatial", "max_dist_coh"),
    "angle_sep": ("_spatial", "angle_sep"),
    "angle_align": ("_spatial", "angle_align"),
    "angle_coh": ("_spatial", "angle_coh"),
    "coherence_factor": ("_spatial", "coherence_factor"),
    "w_fwd": ("_spatial", "w_fwd"),
    # SpatialConfig — obstacle avoidance (EvoFlock genes)
    "static_avoid_weight": ("_spatial", "static_avoid_weight"),
    "predictive_avoid_weight": ("_spatial", "predictive_avoid_weight"),
    "fly_away_max_dist": ("_spatial", "fly_away_max_dist"),
    "min_time_to_collide": ("_spatial", "min_time_to_collide"),
    # FieldConfig
    "field_separation": ("_field", "field_separation"),
    "field_alignment": ("_field", "field_alignment"),
    "field_cohesion": ("_field", "field_cohesion"),
    "field_flow": ("_field", "field_flow"),
    "field_chase_strength": ("_field", "field_chase_strength"),
    "field_noise": ("_field", "field_noise"),
    "field_target_pull": ("_field", "field_target_pull"),
    "field_drift_pull": ("_field", "field_drift_pull"),
    "field_drift_direction": ("_field", "field_drift_direction"),
    "field_shell_influence": ("_field", "field_shell_influence"),
    "field_tangent_pull": ("_field", "field_tangent_pull"),
    "field_wave_gain": ("_field", "field_wave_gain"),
    "field_ripple_trains": ("_field", "field_ripple_trains"),
    "field_inertia": ("_field", "field_inertia"),
    "field_shell_radius_base": ("_field", "field_shell_radius_base"),
    "field_inner_radius_factor": ("_field", "field_inner_radius_factor"),
    "field_num_groups": ("_field", "field_num_groups"),
    "field_leader_fraction": ("_field", "field_leader_fraction"),
    # FieldConfig — new leaves
    "field_flow_pull": ("_field", "field_flow_pull"),
    "field_unit_scale": ("_field", "field_unit_scale"),
    "disabled_terms": ("_field", "disabled_terms"),
    # WanderConfig
    "wander_attractor_speed": ("_wander", "wander_attractor_speed"),
    "wander_attractor_radius": ("_wander", "wander_attractor_radius"),
    # SpeedNoiseConfig
    "speed_noise_frequency": ("_speed_noise", "speed_noise_frequency"),
    "speed_noise_min_mult": ("_speed_noise", "speed_noise_min_mult"),
    "speed_noise_max_mult": ("_speed_noise", "speed_noise_max_mult"),
    "speed_noise_power": ("_speed_noise", "speed_noise_power"),
    "speed_noise_time_scale": ("_speed_noise", "speed_noise_time_scale"),
    # VicsekConfig
    "vicsek_couplage": ("_vicsek", "vicsek_couplage"),
    "vicsek_diffusion": ("_vicsek", "vicsek_diffusion"),
    "vicsek_radius_influence": ("_vicsek", "vicsek_radius_influence"),
    "vicsek_radius_avoid": ("_vicsek", "vicsek_radius_avoid"),
    "vicsek_velocity": ("_vicsek", "vicsek_velocity"),
    "vicsek_time_step": ("_vicsek", "vicsek_time_step"),
    "vicsek_radius_predators": ("_vicsek", "vicsek_radius_predators"),
    "vicsek_velocity_predator": ("_vicsek", "vicsek_velocity_predator"),
    "vicsek_detect_ratio": ("_vicsek", "vicsek_detect_ratio"),
    "vicsek_weight_afraid": ("_vicsek", "vicsek_weight_afraid"),
    "vicsek_predator_noise_ratio": ("_vicsek", "vicsek_predator_noise_ratio"),
    # InfluencerConfig
    "influencer_rank_exponent": ("_influencer", "influencer_rank_exponent"),
    "influencer_substeps": ("_influencer", "influencer_substeps"),
    "influencer_scale": ("_influencer", "influencer_scale"),
    "influencer_influence_mode": ("_influencer", "influencer_influence_mode"),
    "influencer_near_dist_sq": ("_influencer", "influencer_near_dist_sq"),
    "influencer_init_separation": ("_influencer", "influencer_init_separation"),
    "influencer_tick_rate": ("_influencer", "influencer_tick_rate"),
    "influencer_target_freq_primary": ("_influencer", "influencer_target_freq_primary"),
    "influencer_target_freq_secondary": ("_influencer", "influencer_target_freq_secondary"),
    "influencer_target_amp_primary": ("_influencer", "influencer_target_amp_primary"),
    "influencer_target_amp_secondary": ("_influencer", "influencer_target_amp_secondary"),
    "influencer_target_vert_offset": ("_influencer", "influencer_target_vert_offset"),
    "influencer_target_phase_offsets": ("_influencer", "influencer_target_phase_offsets"),
    "influencer_influence_min": ("_influencer", "influencer_influence_min"),
    "influencer_influence_max": ("_influencer", "influencer_influence_max"),
    "influencer_use_rank_override": ("_influencer", "influencer_use_rank_override"),
    "influencer_move_then_steer": ("_influencer", "influencer_move_then_steer"),
    "influencer_density_scaled_init": ("_influencer", "influencer_density_scaled_init"),
    "influencer_pilot_enabled": ("_influencer", "influencer_pilot_enabled"),
    "influencer_pilot_speed": ("_influencer", "influencer_pilot_speed"),
    # AngleConfig
    "turn_rate": ("_angle", "turn_rate"),
    "max_turn_rate": ("_angle", "max_turn_rate"),
    "turn_threshold": ("_angle", "turn_threshold"),
    "jitter_deg": ("_angle", "jitter_deg"),
    "base_speed": ("_angle", "base_speed"),
    "angle_neighbors": ("_angle", "angle_neighbors"),
    "sep_radius_bodies": ("_angle", "sep_radius_bodies"),
    "align_radius_bodies": ("_angle", "align_radius_bodies"),
    "range_radius_bodies": ("_angle", "range_radius_bodies"),
    "angle_speed_mode": ("_angle", "angle_speed_mode"),
    # MarlConfig
    "marl_velocity_cap": ("_marl", "marl_velocity_cap"),
    "marl_rule_weight": ("_marl", "marl_rule_weight"),
    "marl_separation_radius": ("_marl", "marl_separation_radius"),
    "marl_action_scale": ("_marl", "marl_action_scale"),
    "marl_episode_steps": ("_marl", "marl_episode_steps"),
    "marl_reward_w_a": ("_marl", "marl_reward_w_a"),
    "marl_reward_w_c": ("_marl", "marl_reward_w_c"),
    "marl_reward_w_L": ("_marl", "marl_reward_w_L"),
    "marl_reward_w_b": ("_marl", "marl_reward_w_b"),
    "marl_reward_w_z": ("_marl", "marl_reward_w_z"),
    # IndexConfig
    "spatial_index": ("_index", "spatial_index"),
    "topological_cap": ("_index", "topological_cap"),
    "use_toroidal_distance": ("_index", "use_toroidal_distance"),
    # RefinementConfig
    "refinements": ("_refinement", "refinements"),
    "steric": ("_refinement", "steric"),
    "blind_deg": ("_refinement", "blind_deg"),
    "anisotropy": ("_refinement", "anisotropy"),
    "steric_radius": ("_refinement", "steric_radius"),
    "steric_visible_only": ("_refinement", "steric_visible_only"),
    # ExtensionConfig
    "predator_enabled": ("_extension", "predator_enabled"),
    "roosting_enabled": ("_extension", "roosting_enabled"),
    "wander_enabled": ("_extension", "wander_enabled"),
    "ripple_enabled": ("_extension", "ripple_enabled"),
    "speed_noise_enabled": ("_extension", "speed_noise_enabled"),
    "priority_stack_enabled": ("_extension", "priority_stack_enabled"),
    # PredatorConfig
    "predator_threat_radius": ("_predator", "predator_threat_radius"),
    "predator_strength": ("_predator", "predator_strength"),
    "predator_momentum": ("_predator", "predator_momentum"),
    "predator_split_gain": ("_predator", "predator_split_gain"),
    "predator_acceleration": ("_predator", "predator_acceleration"),
    "predator_vacuole_strength": ("_predator", "predator_vacuole_strength"),
    "predator_blackening_gain": ("_predator", "predator_blackening_gain"),
    "predator_mode": ("_predator", "predator_mode"),
    # EcologyConfig
    "ecology_roost": ("_ecology", "ecology_roost"),
    "ecology_critical_mass": ("_ecology", "ecology_critical_mass"),
    "ecology_dusk_width": ("_ecology", "ecology_dusk_width"),
    "ecology_seasonal_amplitude": ("_ecology", "ecology_seasonal_amplitude"),
    "ecology_temperature_boost": ("_ecology", "ecology_temperature_boost"),
    "ecology_predator_presence": ("_ecology", "ecology_predator_presence"),
    # RoostConfig
    "roost_z_target": ("_roost", "z_target"),
    # PerfConfig
    "metrics_detail_level": ("_perf", "metrics_detail_level"),
    "metrics_interval": ("_perf", "metrics_interval"),
    "instance_buffer_chunk": ("_perf", "instance_buffer_chunk"),
    "parallel_workers": ("_perf", "parallel_workers"),
    "bird_mass_kg": ("_perf", "bird_mass_kg"),
    "cruise_speed_ms": ("_perf", "cruise_speed_ms"),
    "acc_peak_ms2": ("_perf", "acc_peak_ms2"),
    "target_fps": ("_perf", "target_fps"),           # P8.6
    "history_cap": ("_perf", "history_cap"),         # D19
    # PerfConfig — new leaves
    "use_numba": ("_perf", "use_numba"),
    "fastmath": ("_perf", "fastmath"),
    "num_threads": ("_perf", "num_threads"),
    "adaptive_quality": ("_perf", "adaptive_quality"),
    "readout_smooth": ("_perf", "readout_smooth"),
    # VizConfig
    "fps": ("_viz", "fps"),
    "window_width": ("_viz", "window_width"),
    "window_height": ("_viz", "window_height"),
    "show_grid": ("_viz", "show_grid"),
    "auto_rotate": ("_viz", "auto_rotate"),
    "theme": ("_viz", "theme"),
    "point_sprites": ("_viz", "point_sprites"),     # P8.1
    "winged_mesh": ("_viz", "winged_mesh"),         # P8.4
    "gradient_sky": ("_viz", "gradient_sky"),       # P8.4
    "dual_view": ("_viz", "dual_view"),               # P8.8
    "trails": ("_viz", "trails"),                   # P8.3
    "trail_length": ("_viz", "trail_length"),       # P8.3
    "density_mode": ("_viz", "density_mode"),       # P8.11
    "density_alpha": ("_viz", "density_alpha"),     # P8.11
    # VizConfig — new leaves
    "per_bird_color": ("_viz", "per_bird_color"),
    "background_top": ("_viz", "background_top"),
    "background_bottom": ("_viz", "background_bottom"),
    "bird_mesh": ("_viz", "bird_mesh"),
    "flap_period": ("_viz", "flap_period"),
    "hud": ("_viz", "hud"),
    # CaptureConfig
    "capture_width": ("_capture", "capture_width"),
    "capture_height": ("_capture", "capture_height"),
    "capture_frames": ("_capture", "capture_frames"),
    "capture_every": ("_capture", "capture_every"),
    "capture_fps": ("_capture", "capture_fps"),
    "capture_output": ("_capture", "capture_output"),
    "capture_metrics_csv": ("_capture", "capture_metrics_csv"),
    "capture_metrics_json": ("_capture", "capture_metrics_json"),
    "capture_with_viz": ("_capture", "capture_with_viz"),
    "capture_prewarm": ("_capture", "capture_prewarm"),     # P8.7
    "capture_sweep": ("_capture", "capture_sweep"),         # P8.7
    "capture_scale": ("_capture", "capture_scale"),         # P8.7
    "capture_mpl_fallback": ("_capture", "capture_mpl_fallback"),  # P8.9
    "capture_mpl_dpi": ("_capture", "capture_mpl_dpi"),            # P8.9
    "capture_frame_cap": ("_capture", "capture_frame_cap"),          # D19
}

# Fields not in _FIELD_MAP are stored directly on SimConfig
_DIRECT_FIELDS: set[str] = {"mode", "seed", "position_init", "velocity_init"}

# Nested-only fields: no flat alias at all (shim fully retired).
# Access via sub-config only, e.g. config.projection.phi_p.
# from_file() routes their YAML keys to the sub-config explicitly.
_NESTED_ONLY: dict[str, tuple[str, str]] = {
    "phi_p": ("_projection", "phi_p"),
}

# All known field names (for YAML from_file filtering)
_ALL_FIELD_NAMES: set[str] = set(_FIELD_MAP.keys()) | _DIRECT_FIELDS | set(_NESTED_ONLY.keys())
