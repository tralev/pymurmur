"""Unit tests for SimConfig.validate() — cross-field validation rules."""

import pytest

from pymurmur.core.config import SimConfig


def _assert_valid(cfg: SimConfig):
    """Helper: assert config validates without error."""
    cfg.validate()


def _assert_invalid(cfg: SimConfig, fragment: str = ""):
    """Helper: assert config raises ValueError, optionally containing fragment."""
    with pytest.raises(ValueError) as exc:
        cfg.validate()
    if fragment:
        assert fragment in str(exc.value), (
            f"Expected '{fragment}' in error, got: {exc.value}"
        )


# ── Domain dimension rules ────────────────────────────────────────

def test_default_config_validates():
    """Default SimConfig() passes validation."""
    _assert_valid(SimConfig())


def test_negative_width_rejected():
    _assert_invalid(SimConfig(width=-1), "domain.width")


def test_zero_height_rejected():
    _assert_invalid(SimConfig(height=0), "domain.height")


def test_negative_depth_rejected():
    _assert_invalid(SimConfig(depth=-0.5), "domain.depth")


# ── Flock rules ────────────────────────────────────────────────────

def test_negative_num_boids_rejected():
    _assert_invalid(SimConfig(num_boids=-5), "num_boids")


def test_zero_boid_size_rejected():
    _assert_invalid(SimConfig(boid_size=0), "boid_size")


def test_negative_v0_rejected():
    _assert_invalid(SimConfig(v0=-1), "v0")


def test_negative_max_force_rejected():
    _assert_invalid(SimConfig(max_force=-0.1), "max_force")


def test_zero_visual_range_rejected():
    _assert_invalid(SimConfig(visual_range=0), "visual_range")


# ── Boundary rules ─────────────────────────────────────────────────

def test_unknown_boundary_mode_rejected():
    _assert_invalid(SimConfig(boundary_mode="reflective"), "boundary_mode")


def test_all_valid_boundary_modes_accepted():
    for mode in ("toroidal", "open", "margin", "sphere"):
        _assert_valid(SimConfig(boundary_mode=mode))


def test_negative_sphere_radius_rejected():
    _assert_invalid(SimConfig(boundary_sphere_radius=-1), "boundary_sphere_radius")


# ── Mode rules ─────────────────────────────────────────────────────

def test_unknown_mode_rejected():
    _assert_invalid(SimConfig(mode="foo_invalid"), "mode")


def test_all_valid_modes_accepted():
    for mode in ("projection", "spatial", "field", "vicsek", "influencer", "angle"):
        _assert_valid(SimConfig(mode=mode))


# ── Mode-specific: projection ──────────────────────────────────────

def test_projection_sigma_zero_rejected():
    _assert_invalid(SimConfig(mode="projection", sigma=0), "sigma")


def test_projection_negative_phi_p_rejected():
    """projection.phi_p < 0 rejected (shim retired — uses sub-config access)."""
    cfg = SimConfig(mode="projection")
    cfg.projection.phi_p = -0.01
    _assert_invalid(cfg, "phi_p")


def test_projection_negative_phi_a_rejected():
    _assert_invalid(SimConfig(mode="projection", phi_a=-0.01), "phi_a")


def test_projection_defaults_valid():
    _assert_valid(SimConfig(mode="projection"))


# ── Mode-specific: spatial ────────────────────────────────────────

def test_spatial_negative_weights_rejected():
    _assert_invalid(SimConfig(mode="spatial", separation_weight=-1), "separation_weight")
    _assert_invalid(SimConfig(mode="spatial", alignment_weight=-1), "alignment_weight")
    _assert_invalid(SimConfig(mode="spatial", cohesion_weight=-1), "cohesion_weight")
    _assert_invalid(SimConfig(mode="spatial", noise_scale=-1), "noise_scale")


def test_spatial_defaults_valid():
    _assert_valid(SimConfig(mode="spatial"))


# ── Mode-specific: vicsek ─────────────────────────────────────────

def test_vicsek_couplage_out_of_range_rejected():
    _assert_invalid(SimConfig(mode="vicsek", vicsek_couplage=1.5), "vicsek_couplage")
    _assert_invalid(SimConfig(mode="vicsek", vicsek_couplage=-0.1), "vicsek_couplage")


def test_vicsek_couplage_boundary_accepted():
    _assert_valid(SimConfig(mode="vicsek", vicsek_couplage=0.0))
    _assert_valid(SimConfig(mode="vicsek", vicsek_couplage=1.0))


def test_vicsek_negative_diffusion_rejected():
    _assert_invalid(SimConfig(mode="vicsek", vicsek_diffusion=-0.1), "vicsek_diffusion")


def test_vicsek_influence_not_greater_than_avoid_rejected():
    _assert_invalid(
        SimConfig(mode="vicsek", vicsek_radius_influence=1.0, vicsek_radius_avoid=2.0),
        "vicsek_radius_influence",
    )
    _assert_invalid(
        SimConfig(mode="vicsek", vicsek_radius_influence=1.0, vicsek_radius_avoid=1.0),
        "vicsek_radius_influence",
    )


def test_vicsek_zero_velocity_rejected():
    _assert_invalid(SimConfig(mode="vicsek", vicsek_velocity=0), "vicsek_velocity")


def test_vicsek_zero_time_step_rejected():
    _assert_invalid(SimConfig(mode="vicsek", vicsek_time_step=0), "vicsek_time_step")


def test_vicsek_defaults_valid():
    _assert_valid(SimConfig(mode="vicsek"))


# ── Mode-specific: influencer ─────────────────────────────────────

def test_influencer_zero_substeps_rejected():
    _assert_invalid(SimConfig(mode="influencer", influencer_substeps=0), "influencer_substeps")


def test_influencer_negative_rank_exponent_rejected():
    _assert_invalid(
        SimConfig(mode="influencer", influencer_rank_exponent=-0.1),
        "influencer_rank_exponent",
    )


# ── Refinements ────────────────────────────────────────────────────

def test_blind_deg_out_of_range_rejected():
    _assert_invalid(SimConfig(blind_deg=-5), "blind_deg")
    _assert_invalid(SimConfig(blind_deg=360), "blind_deg")
    _assert_invalid(SimConfig(blind_deg=400), "blind_deg")


def test_blind_deg_boundary_accepted():
    _assert_valid(SimConfig(blind_deg=0))
    _assert_valid(SimConfig(blind_deg=359))
    _assert_valid(SimConfig(blind_deg=180))


def test_anisotropy_below_one_rejected():
    _assert_invalid(SimConfig(anisotropy=0.5), "anisotropy")


def test_anisotropy_exactly_one_accepted():
    _assert_valid(SimConfig(anisotropy=1.0))


def test_negative_steric_rejected():
    _assert_invalid(SimConfig(steric=-0.1), "steric")


def test_steric_zero_accepted():
    _assert_valid(SimConfig(steric=0.0))


# ── Extensions cross-field ─────────────────────────────────────────

def test_predator_enabled_without_radius_rejected():
    _assert_invalid(
        SimConfig(predator_enabled=True, predator_threat_radius=0),
        "predator_threat_radius",
    )


def test_predator_enabled_without_strength_rejected():
    _assert_invalid(
        SimConfig(predator_enabled=True, predator_strength=0),
        "predator_strength",
    )


def test_predator_disabled_fine_with_zero_params():
    """When predator_enabled=False, zero radius/strength is fine (not checked)."""
    _assert_valid(SimConfig(predator_enabled=False, predator_threat_radius=0, predator_strength=0))


def test_predator_enabled_with_valid_params_accepted():
    _assert_valid(SimConfig(predator_enabled=True, predator_threat_radius=10, predator_strength=1.0))


# ── Spatial index ──────────────────────────────────────────────────

def test_unknown_spatial_index_rejected():
    _assert_invalid(SimConfig(spatial_index="rtree"), "spatial_index")


def test_all_valid_index_types_accepted():
    for idx_type in ("auto", "hash_grid", "kdtree", "none"):
        _assert_valid(SimConfig(spatial_index=idx_type))


def test_zero_topological_cap_rejected():
    _assert_invalid(SimConfig(topological_cap=0), "topological_cap")


# ── Performance ────────────────────────────────────────────────────

def test_invalid_metrics_detail_level_rejected():
    _assert_invalid(SimConfig(metrics_detail_level=3), "metrics_detail_level")
    _assert_invalid(SimConfig(metrics_detail_level=-1), "metrics_detail_level")


def test_all_valid_metrics_levels_accepted():
    for level in (0, 1, 2):
        _assert_valid(SimConfig(metrics_detail_level=level))


def test_zero_metrics_interval_rejected():
    _assert_invalid(SimConfig(metrics_interval=0), "metrics_interval")


def test_negative_parallel_workers_rejected():
    _assert_invalid(SimConfig(parallel_workers=-2), "parallel_workers")


def test_parallel_workers_minus_one_accepted():
    _assert_valid(SimConfig(parallel_workers=-1))


# ── Visualization ──────────────────────────────────────────────────

def test_zero_fps_rejected():
    _assert_invalid(SimConfig(fps=0), "fps")


def test_negative_window_dimensions_rejected():
    _assert_invalid(SimConfig(window_width=0), "window_width")
    _assert_invalid(SimConfig(window_height=-1), "window_height")


def test_unknown_theme_rejected():
    _assert_invalid(SimConfig(theme="neon"), "theme")


def test_all_valid_themes_accepted():
    for theme in ("ink", "inverse", "paper", "graphite"):
        _assert_valid(SimConfig(theme=theme))


# ── Capture ────────────────────────────────────────────────────────

def test_zero_capture_dimensions_rejected():
    _assert_invalid(SimConfig(capture_width=0), "capture_width")
    _assert_invalid(SimConfig(capture_height=-1), "capture_height")


def test_zero_capture_frames_rejected():
    _assert_invalid(SimConfig(capture_frames=0), "capture_frames")


def test_zero_capture_every_rejected():
    _assert_invalid(SimConfig(capture_every=0), "capture_every")


def test_zero_capture_fps_rejected():
    _assert_invalid(SimConfig(capture_fps=0), "capture_fps")


# ── Ecology cross-field ────────────────────────────────────────────

def test_roosting_enabled_with_out_of_bounds_roost_rejected():
    _assert_invalid(
        SimConfig(
            roosting_enabled=True,
            width=100, height=100, depth=100,
            ecology_roost=(200, 50, 50),
        ),
        "ecology_roost",
    )


def test_roosting_disabled_with_out_of_bounds_roost_accepted():
    """roosting_enabled=False skips roost position check."""
    _assert_valid(
        SimConfig(
            roosting_enabled=False,
            width=100, height=100, depth=100,
            ecology_roost=(200, 50, 50),
        )
    )


def test_roosting_enabled_with_valid_roost_accepted():
    _assert_valid(
        SimConfig(
            roosting_enabled=True,
            width=1000, height=700, depth=400,
            ecology_roost=(500, 350, 200),
        )
    )


# ── Direct fields ──────────────────────────────────────────────────

def test_invalid_position_init_rejected():
    _assert_invalid(SimConfig(position_init="invalid"), "position_init")


def test_all_valid_position_inits_accepted():
    for init in ("box", "random", "sphere"):
        _assert_valid(SimConfig(position_init=init))


def test_velocity_init_drift_alias_accepted():
    """C3: "drift" is a valid velocity_init alias for "blob"."""
    _assert_valid(SimConfig(velocity_init="drift"))


# ── Type guard rules ──────────────────────────────────────────────

def test_numeric_field_with_list_rejected():
    """Numeric fields that somehow got a list value are caught by type guards."""
    cfg = SimConfig()
    object.__setattr__(cfg._flock, "max_force", [0, 100])
    _assert_invalid(cfg, "max_force must be numeric")


# ── Multiple issues aggregated ─────────────────────────────────────

def test_multiple_issues_aggregated_in_one_error():
    """When multiple rules fail, all issues appear in the single ValueError."""
    with pytest.raises(ValueError) as exc:
        SimConfig(width=-1, height=0, mode="bogus", fps=0).validate()

    msg = str(exc.value)
    assert "4 issue" in msg
    assert "width" in msg
    assert "height" in msg
    assert "mode" in msg
    assert "fps" in msg


# ── Engine integration ─────────────────────────────────────────────

def test_engine_creation_validates_config():
    """SimulationEngine creation calls validate() — invalid config raises."""
    from pymurmur.simulation.engine import SimulationEngine

    with pytest.raises(ValueError):
        SimulationEngine(SimConfig(width=-1))


# ── S2.B10: fastmath-vs-metrics-detail-level policy ─────────────────

def test_fastmath_with_metrics_detail_raises():
    """S2.B10: fastmath=True with metrics_detail_level>0 fails validation
    -- fastmath breaks IEEE reproducibility, so it must not be combined
    with scientific metric export."""
    cfg = SimConfig()
    cfg.fastmath = True
    cfg.metrics_detail_level = 1
    with pytest.raises(ValueError, match="fastmath"):
        cfg.validate()


def test_fastmath_with_metrics_off_is_valid():
    """S2.B10: fastmath=True is fine when metrics_detail_level==0 (visual-only)."""
    cfg = SimConfig()
    cfg.fastmath = True
    cfg.metrics_detail_level = 0
    cfg.validate()  # must not raise


def test_fastmath_default_off_is_valid_at_any_detail_level():
    """S2.B10: default fastmath=False never triggers the policy check."""
    cfg = SimConfig()
    assert cfg.fastmath is False
    cfg.metrics_detail_level = 2
    cfg.validate()  # must not raise
