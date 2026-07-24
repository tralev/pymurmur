"""SimConfig cross-field validation.

Extracted from config.py (file-size split) — validate_config(cfg)
holds the full rule set previously inlined in SimConfig.validate();
that method is now a thin wrapper that raises ValueError if this
returns any issues. References cfg._VALID_MODES etc. (the small
class-level allowed-value sets stay on SimConfig itself, since
SimConfig._VALID_MODES is accessed directly by at least one test) --
this function is passed a SimConfig instance, not decoupled from it.
"""
from __future__ import annotations


def validate_config(cfg) -> list[str]:
    """Check cross-field consistency, returning a list of issue strings
    (empty if valid). SimConfig.validate() raises ValueError with these
    aggregated if non-empty -- call at engine creation time to catch
    misconfiguration early.
    """
    issues: list[str] = []

    # ── Type guards: catch non-numeric values early ───────
    _numeric_fields = (
        "width", "height", "depth",
        "num_boids", "boid_size", "v0", "max_force", "visual_range",
        "phi_a", "sigma",
        "separation_weight", "alignment_weight", "cohesion_weight",
        "noise_scale", "acceleration_scale",
        "influence_count",
        "predator_escape_factor", "predator_speed_boost",
        "predator_perception_boost",            "predator_accel_boost",
        "jitter_separation", "jitter_cohesion", "jitter_alignment",
        "steric", "blind_deg", "anisotropy",
        "parallel_workers", "metrics_interval", "metrics_detail_level",
        "bird_mass_kg", "cruise_speed_ms", "acc_peak_ms2",
        "topological_cap",
        "boundary_sphere_radius",
        "fps", "window_width", "window_height",
        "capture_width", "capture_height", "capture_frames",
        "capture_every", "capture_fps",
        "vicsek_couplage", "vicsek_diffusion",
        "vicsek_radius_influence", "vicsek_radius_avoid",
        "vicsek_velocity", "vicsek_time_step",
        "vicsek_radius_predators", "vicsek_velocity_predator",
        "vicsek_detect_ratio", "vicsek_weight_afraid",
        "vicsek_predator_noise_ratio",
        "vicsek_radius_predators", "vicsek_velocity_predator",
        "vicsek_detect_ratio", "vicsek_weight_afraid",
        "vicsek_predator_noise_ratio",
        "influencer_rank_exponent", "influencer_substeps",
        "influencer_scale",
        "influencer_near_dist_sq", "influencer_init_separation",
        "influencer_tick_rate", "influencer_pilot_speed",
        "influencer_influence_min", "influencer_influence_max",
        "influencer_target_vert_offset",
        "predator_threat_radius", "predator_strength",
        "predator_momentum", "predator_split_gain",
        "field_separation", "field_alignment", "field_cohesion",
        "field_flow", "field_chase_strength",
        "field_noise", "field_target_pull", "field_drift_pull",
        "field_shell_influence", "field_tangent_pull",
        "field_wave_gain", "field_inertia",
        "field_shell_radius_base", "field_ripple_trains",
        "field_inner_radius_factor", "field_leader_fraction",
        "field_num_groups",
        "wander_attractor_speed", "wander_attractor_radius",
        "boundary_avoidance_factor", "boundary_radius_factor",
        "acceleration_scale",
        "ecology_dusk_width", "ecology_seasonal_amplitude",
        "ecology_temperature_boost",
        "trail_length",
        # AngleConfig
        "turn_rate", "max_turn_rate", "turn_threshold",
        "jitter_deg", "base_speed", "angle_neighbors",
        "sep_radius_bodies", "align_radius_bodies",
        "range_radius_bodies",
        # MarlConfig
        "marl_velocity_cap", "marl_rule_weight",
        "marl_separation_radius", "marl_action_scale",
        "marl_episode_steps",
        "marl_reward_w_a", "marl_reward_w_c", "marl_reward_w_L",
        "marl_reward_w_b", "marl_reward_w_z",
        # PredatorConfig extras
        "predator_acceleration", "predator_vacuole_strength",
        "predator_blackening_gain",
        # New spatial leaves
        "flow_weight", "w_fwd",
        "readout_smooth",
        "max_dist_sep", "max_dist_align", "max_dist_coh",
        "angle_sep", "angle_align", "angle_coh", "coherence_factor",
        # New boundary leaves
        "boundary_margin",
        # New projection leaves
        "max_visibility", "max_occlusion_neighbors",
        # New flock leaves
        "n_predators",
        # New field leaves
        "field_flow_pull",
        # New perf leaves
        "num_threads",
        # New viz leaves
        "flap_period",
        # New spatial obstacle avoidance leaves
        "static_avoid_weight", "predictive_avoid_weight",
        "fly_away_max_dist", "min_time_to_collide",
        # New roost config leaves
        "roost_z_target",
    )
    _type_bad: set[str] = set()
    for fname in _numeric_fields:
        val = getattr(cfg, fname)
        if not isinstance(val, (int, float)):
            issues.append(
                f"{fname} must be numeric, got {type(val).__name__} {val!r}"
            )
            _type_bad.add(fname)

    def _ok(fname: str) -> bool:
        """True if this field passed the type guard (safe for comparisons)."""
        return fname not in _type_bad

    # ── Domain dimensions ──────────────────────────────────
    if _ok("width") and cfg.width <= 0:
        issues.append(f"domain.width must be > 0, got {cfg.width}")
    if _ok("height") and cfg.height <= 0:
        issues.append(f"domain.height must be > 0, got {cfg.height}")
    if _ok("depth") and cfg.depth <= 0:
        issues.append(f"domain.depth must be > 0, got {cfg.depth}")

    # ── Flock ─────────────────────────────────────────────
    if _ok("num_boids") and cfg.num_boids < 0:
        issues.append(f"num_boids must be >= 0, got {cfg.num_boids}")
    if _ok("boid_size") and cfg.boid_size <= 0:
        issues.append(f"boid_size must be > 0, got {cfg.boid_size}")
    if _ok("v0") and cfg.v0 < 0:
        issues.append(f"v0 must be >= 0, got {cfg.v0}")
    if _ok("max_force") and cfg.max_force < 0:
        issues.append(f"max_force must be >= 0, got {cfg.max_force}")
    if _ok("visual_range") and cfg.visual_range <= 0:
        issues.append(f"visual_range must be > 0, got {cfg.visual_range}")

    # ── Boundary ──────────────────────────────────────────
    if cfg.boundary_mode not in cfg._VALID_BOUNDARY_MODES:
        issues.append(
            f"boundary_mode must be one of {cfg._VALID_BOUNDARY_MODES}, "
            f"got {cfg.boundary_mode!r}"
        )
    if _ok("boundary_sphere_radius") and cfg.boundary_sphere_radius <= 0:
        issues.append(
            f"boundary_sphere_radius must be > 0, got {cfg.boundary_sphere_radius}"
        )

    # ── Direct fields ─────────────────────────────────────
    if cfg.position_init not in cfg._VALID_POSITION_INITS:
        issues.append(
            f"position_init must be one of {cfg._VALID_POSITION_INITS}, "
            f"got {cfg.position_init!r}"
        )
    if cfg.velocity_init not in cfg._VALID_VELOCITY_INITS:
        issues.append(
            f"velocity_init must be one of {cfg._VALID_VELOCITY_INITS}, "
            f"got {cfg.velocity_init!r}"
        )

    # ── Mode ──────────────────────────────────────────────
    if cfg.mode not in cfg._VALID_MODES:
        issues.append(
            f"mode must be one of {cfg._VALID_MODES}, got {cfg.mode!r}"
        )

    # ── Mode-specific constraints ─────────────────────────
    # Explicit phi_p validation (shim retired — read from sub-config)
    if not isinstance(cfg.projection.phi_p, (int, float)):
        issues.append(
            f"projection.phi_p must be numeric, "
            f"got {type(cfg.projection.phi_p).__name__} {cfg.projection.phi_p!r}"
        )

    if cfg.mode == "projection":
        if _ok("sigma") and cfg.sigma <= 0:
            issues.append(f"projection.sigma must be > 0, got {cfg.sigma}")
        if isinstance(cfg.projection.phi_p, (int, float)) and cfg.projection.phi_p < 0:
            issues.append(f"projection.phi_p must be >= 0, got {cfg.projection.phi_p}")
        if _ok("phi_a") and cfg.phi_a < 0:
            issues.append(f"projection.phi_a must be >= 0, got {cfg.phi_a}")

    if cfg.mode == "spatial":
        if _ok("separation_weight") and cfg.separation_weight < 0:
            issues.append(f"spatial.separation_weight >= 0, got {cfg.separation_weight}")
        if _ok("alignment_weight") and cfg.alignment_weight < 0:
            issues.append(f"spatial.alignment_weight >= 0, got {cfg.alignment_weight}")
        if _ok("cohesion_weight") and cfg.cohesion_weight < 0:
            issues.append(f"spatial.cohesion_weight >= 0, got {cfg.cohesion_weight}")
        if _ok("influence_count") and cfg.influence_count < 1:
            issues.append(f"spatial.influence_count must be >= 1, got {cfg.influence_count}")
        if _ok("noise_scale") and cfg.noise_scale < 0:
            issues.append(f"spatial.noise_scale >= 0, got {cfg.noise_scale}")

    if cfg.mode == "vicsek":
        if _ok("vicsek_couplage") and not (
            0.0 <= cfg.vicsek_couplage <= 1.0
        ):
            issues.append(
                f"vicsek_couplage must be in [0,1], got {cfg.vicsek_couplage}"
            )
        if _ok("vicsek_diffusion") and cfg.vicsek_diffusion < 0:
            issues.append(
                f"vicsek_diffusion must be >= 0, got {cfg.vicsek_diffusion}"
            )
        if (
            _ok("vicsek_radius_influence")
            and _ok("vicsek_radius_avoid")
            and cfg.vicsek_radius_influence <= cfg.vicsek_radius_avoid
        ):
            issues.append(
                f"vicsek_radius_influence ({cfg.vicsek_radius_influence}) "
                f"must be > vicsek_radius_avoid ({cfg.vicsek_radius_avoid})"
            )
        if _ok("vicsek_velocity") and cfg.vicsek_velocity <= 0:
            issues.append(
                f"vicsek_velocity must be > 0, got {cfg.vicsek_velocity}"
            )
        if _ok("vicsek_time_step") and cfg.vicsek_time_step <= 0:
            issues.append(
                f"vicsek_time_step must be > 0, got {cfg.vicsek_time_step}"
            )

    if cfg.mode == "influencer":
        if _ok("influencer_substeps") and cfg.influencer_substeps < 1:
            issues.append(
                f"influencer_substeps must be >= 1, got {cfg.influencer_substeps}"
            )
        if _ok("influencer_rank_exponent") and cfg.influencer_rank_exponent <= 0:
            issues.append(
                f"influencer_rank_exponent must be > 0, got {cfg.influencer_rank_exponent}"
            )
        if _ok("influencer_scale") and cfg.influencer_scale <= 0:
            issues.append(
                f"influencer_scale must be > 0, got {cfg.influencer_scale}"
            )
        if cfg.influencer_influence_mode not in {"rank", "distance"}:
            issues.append(
                f"influencer_influence_mode must be 'rank' or 'distance', "
                f"got {cfg.influencer_influence_mode!r}"
            )
        if _ok("influencer_near_dist_sq") and cfg.influencer_near_dist_sq <= 0:
            issues.append(
                f"influencer_near_dist_sq must be > 0, got {cfg.influencer_near_dist_sq}"
            )
        if _ok("influencer_init_separation") and cfg.influencer_init_separation <= 0:
            issues.append(
                f"influencer_init_separation must be > 0, got {cfg.influencer_init_separation}"
            )
        if _ok("influencer_tick_rate") and cfg.influencer_tick_rate <= 0:
            issues.append(
                f"influencer_tick_rate must be > 0, got {cfg.influencer_tick_rate}"
            )
        if not cfg.influencer_move_then_steer:
            issues.append(
                "influencer_move_then_steer=False is not supported — "
                "move-then-steer is the only implemented update order (S2.E2)"
            )
        if cfg.influencer_influence_min > cfg.influencer_influence_max:
            issues.append(
                f"influencer_influence_min ({cfg.influencer_influence_min}) must be "
                f"<= influencer_influence_max ({cfg.influencer_influence_max})"
            )
        for tup_name in (
            "influencer_target_freq_primary", "influencer_target_freq_secondary",
            "influencer_target_amp_primary", "influencer_target_amp_secondary",
            "influencer_target_phase_offsets",
        ):
            tup_val = getattr(cfg, tup_name)
            if len(tup_val) != 3:
                issues.append(f"{tup_name} must have exactly 3 elements, got {len(tup_val)}")
        if any(f == 0 for f in cfg.influencer_target_freq_primary) or any(
            f == 0 for f in cfg.influencer_target_freq_secondary
        ):
            issues.append("influencer_target_freq_primary/secondary entries must be nonzero")
        if _ok("influencer_pilot_speed") and cfg.influencer_pilot_speed <= 0:
            issues.append(
                f"influencer_pilot_speed must be > 0, got {cfg.influencer_pilot_speed}"
            )

    # ── Refinements ───────────────────────────────────────
    if _ok("blind_deg") and (cfg.blind_deg < 0 or cfg.blind_deg >= 360):
        issues.append(
            f"blind_deg must be in [0, 360), got {cfg.blind_deg}"
        )
    if _ok("anisotropy") and cfg.anisotropy < 1.0:
        issues.append(
            f"anisotropy must be >= 1.0 (body axis ratio a/b), got {cfg.anisotropy}"
        )
    if _ok("steric") and cfg.steric < 0:
        issues.append(f"steric must be >= 0, got {cfg.steric}")

    # ── Angle mode ──────────────────────────────────────────
    if cfg.angle_speed_mode not in cfg._VALID_ANGLE_SPEED_MODES:
        issues.append(
            f"angle_speed_mode must be one of {cfg._VALID_ANGLE_SPEED_MODES}, "
            f"got {cfg.angle_speed_mode!r}"
        )

    # ── Extensions cross-field ────────────────────────────
    if cfg.predator_mode not in cfg._VALID_PREDATOR_MODES:
        issues.append(
            f"predator_mode must be one of {cfg._VALID_PREDATOR_MODES}, "
            f"got {cfg.predator_mode!r}"
        )
    if cfg.predator_enabled:
        if (
            _ok("predator_threat_radius")
            and cfg.predator_threat_radius <= 0
        ):
            issues.append(
                "predator_enabled=True but predator_threat_radius must be > 0"
            )
        if _ok("predator_strength") and cfg.predator_strength <= 0:
            issues.append(
                "predator_enabled=True but predator_strength must be > 0"
            )

    # ── Spatial index ─────────────────────────────────────
    if cfg.spatial_index not in cfg._VALID_INDEX_TYPES:
        issues.append(
            f"spatial_index must be one of {cfg._VALID_INDEX_TYPES}, "
            f"got {cfg.spatial_index!r}"
        )
    if _ok("topological_cap") and cfg.topological_cap < 1:
        issues.append(f"topological_cap must be >= 1, got {cfg.topological_cap}")

    # ── Performance ───────────────────────────────────────
    if cfg.metrics_detail_level not in cfg._VALID_METRICS_LEVELS:
        issues.append(
            f"metrics_detail_level must be in {cfg._VALID_METRICS_LEVELS}, "
            f"got {cfg.metrics_detail_level}"
        )
    if _ok("metrics_interval") and cfg.metrics_interval < 1:
        issues.append(f"metrics_interval must be >= 1, got {cfg.metrics_interval}")
    if _ok("parallel_workers") and cfg.parallel_workers < -1:
        issues.append(
            f"parallel_workers must be >= -1, got {cfg.parallel_workers}"
        )
    # S2.B10: fastmath relaxes IEEE float semantics — only safe when
    # metrics aren't being exported for scientific analysis (detail
    # level 0 = visual-only runs). detail_level >= 1 requires
    # IEEE-precise kernels so IEEE-precise observables stay meaningful.
    if cfg.fastmath and cfg.metrics_detail_level > 0:
        issues.append(
            "perf.fastmath=True requires perf.metrics_detail_level == 0 "
            f"(visual-only runs) -- got metrics_detail_level={cfg.metrics_detail_level}. "
            "fastmath relaxes IEEE float semantics, which would make exported "
            "metrics non-reproducible."
        )

    # ── Visualization ─────────────────────────────────────
    if _ok("fps") and cfg.fps <= 0:
        issues.append(f"viz.fps must be > 0, got {cfg.fps}")
    if _ok("window_width") and cfg.window_width <= 0:
        issues.append(f"viz.window_width must be > 0, got {cfg.window_width}")
    if _ok("window_height") and cfg.window_height <= 0:
        issues.append(f"viz.window_height must be > 0, got {cfg.window_height}")
    if cfg.theme not in cfg._VALID_THEMES:
        issues.append(
            f"viz.theme must be one of {cfg._VALID_THEMES}, got {cfg.theme!r}"
        )

    # S4.4a: Validate bird_mesh
    if _ok("bird_mesh") and cfg.bird_mesh not in cfg._VALID_MESH_NAMES:
        issues.append(
            f"viz.bird_mesh must be one of {cfg._VALID_MESH_NAMES}, "
            f"got {cfg.bird_mesh!r}"
        )

    # ── Capture ───────────────────────────────────────────
    if _ok("capture_width") and cfg.capture_width <= 0:
        issues.append(f"capture_width must be > 0, got {cfg.capture_width}")
    if _ok("capture_height") and cfg.capture_height <= 0:
        issues.append(f"capture_height must be > 0, got {cfg.capture_height}")
    if _ok("capture_frames") and cfg.capture_frames < 1:
        issues.append(f"capture_frames must be >= 1, got {cfg.capture_frames}")
    if _ok("capture_every") and cfg.capture_every < 1:
        issues.append(f"capture_every must be >= 1, got {cfg.capture_every}")
    if _ok("capture_fps") and cfg.capture_fps <= 0:
        issues.append(f"capture_fps must be > 0, got {cfg.capture_fps}")

    # ── Angle mode ────────────────────────────────────────
    if cfg.mode == "angle":
        if _ok("turn_rate") and cfg.turn_rate <= 0:
            issues.append(f"angle.turn_rate must be > 0, got {cfg.turn_rate}")
        if _ok("max_turn_rate") and cfg.max_turn_rate <= 0:
            issues.append(f"angle.max_turn_rate must be > 0, got {cfg.max_turn_rate}")
        if _ok("turn_threshold") and cfg.turn_threshold < 0:
            issues.append(f"angle.turn_threshold must be >= 0, got {cfg.turn_threshold}")
        if _ok("base_speed") and cfg.base_speed <= 0:
            issues.append(f"angle.base_speed must be > 0, got {cfg.base_speed}")
        if _ok("angle_neighbors") and cfg.angle_neighbors < 1:
            issues.append(f"angle.angle_neighbors must be >= 1, got {cfg.angle_neighbors}")

    # ── MARL mode ─────────────────────────────────────────
    if cfg.mode == "marl":
        if _ok("marl_velocity_cap") and cfg.marl_velocity_cap <= 0:
            issues.append(f"marl.velocity_cap must be > 0, got {cfg.marl_velocity_cap}")
        if _ok("marl_rule_weight") and cfg.marl_rule_weight < 0:
            issues.append(f"marl.rule_weight must be >= 0, got {cfg.marl_rule_weight}")
        if _ok("marl_separation_radius") and cfg.marl_separation_radius <= 0:
            issues.append(
                f"marl.separation_radius must be > 0, got {cfg.marl_separation_radius}"
            )
        if _ok("marl_action_scale") and cfg.marl_action_scale <= 0:
            issues.append(
                f"marl.action_scale must be > 0, got {cfg.marl_action_scale}"
            )
        if _ok("marl_episode_steps") and cfg.marl_episode_steps < 1:
            issues.append(
                f"marl.episode_steps must be >= 1, got {cfg.marl_episode_steps}"
            )

    # ── Ecology cross-field ───────────────────────────────
    if cfg.roosting_enabled:
        rx, ry, rz = cfg.ecology_roost
        domain_ok = (
            _ok("width") and 0 <= rx <= cfg.width
            and _ok("height") and 0 <= ry <= cfg.height
            and _ok("depth") and 0 <= rz <= cfg.depth
        )
        if not domain_ok:
            issues.append(
                f"roosting_enabled=True but ecology_roost {cfg.ecology_roost} "
                f"is outside domain bounds ({cfg.width}x{cfg.height}x{cfg.depth})"
            )


    return issues
