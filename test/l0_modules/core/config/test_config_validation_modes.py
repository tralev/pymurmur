"""Unit tests for SimConfig.validate() — mode-specific cross-field
validation rules (projection, spatial, velocity_damping, vicsek,
influencer).

Split out of test_config_validation.py (file-size split) — domain,
flock, boundary, mode-selection, refinements, extensions, spatial
index, performance, viz, capture, ecology, and aggregation tests stay
in the original.
"""

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


def test_projection_heading_inertia_out_of_range_rejected():
    _assert_invalid(
        SimConfig(mode="projection", projection_heading_inertia=1.5),
        "projection_heading_inertia",
    )
    _assert_invalid(
        SimConfig(mode="projection", projection_heading_inertia=-0.1),
        "projection_heading_inertia",
    )


def test_projection_heading_inertia_boundary_accepted():
    _assert_valid(SimConfig(mode="projection", projection_heading_inertia=0.0))
    _assert_valid(SimConfig(mode="projection", projection_heading_inertia=1.0))


# ── Mode-specific: spatial ────────────────────────────────────────

def test_spatial_negative_weights_rejected():
    _assert_invalid(SimConfig(mode="spatial", separation_weight=-1), "separation_weight")
    _assert_invalid(SimConfig(mode="spatial", alignment_weight=-1), "alignment_weight")
    _assert_invalid(SimConfig(mode="spatial", cohesion_weight=-1), "cohesion_weight")
    _assert_invalid(SimConfig(mode="spatial", noise_scale=-1), "noise_scale")


def test_spatial_defaults_valid():
    _assert_valid(SimConfig(mode="spatial"))


def test_unknown_separation_kernel_rejected():
    _assert_invalid(SimConfig(mode="spatial", separation_kernel="bogus"), "separation_kernel")


def test_all_valid_separation_kernels_accepted():
    for kernel in SimConfig._VALID_SEPARATION_KERNELS:
        _assert_valid(SimConfig(mode="spatial", separation_kernel=kernel))


def test_unknown_cohesion_kernel_rejected():
    _assert_invalid(SimConfig(mode="spatial", cohesion_kernel="bogus"), "cohesion_kernel")


def test_all_valid_cohesion_kernels_accepted():
    for kernel in SimConfig._VALID_COHESION_KERNELS:
        _assert_valid(SimConfig(mode="spatial", cohesion_kernel=kernel))


def test_negative_separation_kernel_radius_rejected():
    _assert_invalid(
        SimConfig(mode="spatial", separation_kernel_radius=-1),
        "separation_kernel_radius",
    )


def test_unknown_alignment_kernel_rejected():
    _assert_invalid(SimConfig(mode="spatial", alignment_kernel="bogus"), "alignment_kernel")


def test_all_valid_alignment_kernels_accepted():
    for kernel in SimConfig._VALID_ALIGNMENT_KERNELS:
        _assert_valid(SimConfig(mode="spatial", alignment_kernel=kernel))


def test_zero_kernel_zone_width_rejected():
    _assert_invalid(SimConfig(mode="spatial", kernel_zone_width=0), "kernel_zone_width")


def test_negative_kernel_zone_width_rejected():
    _assert_invalid(SimConfig(mode="spatial", kernel_zone_width=-1), "kernel_zone_width")


# ── velocity_damping (mode-agnostic, FlockConfig) ──────────────────

def test_negative_velocity_damping_rejected():
    _assert_invalid(SimConfig(velocity_damping=-0.1), "velocity_damping")


def test_zero_velocity_damping_accepted():
    _assert_valid(SimConfig(velocity_damping=0.0))


def test_positive_velocity_damping_accepted():
    _assert_valid(SimConfig(velocity_damping=0.5))


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


def test_vicsek_heading_inertia_out_of_range_rejected():
    _assert_invalid(
        SimConfig(mode="vicsek", vicsek_heading_inertia=1.5), "vicsek_heading_inertia",
    )
    _assert_invalid(
        SimConfig(mode="vicsek", vicsek_heading_inertia=-0.1), "vicsek_heading_inertia",
    )


def test_vicsek_heading_inertia_boundary_accepted():
    _assert_valid(SimConfig(mode="vicsek", vicsek_heading_inertia=0.0))
    _assert_valid(SimConfig(mode="vicsek", vicsek_heading_inertia=1.0))


# ── Mode-specific: influencer ─────────────────────────────────────

def test_influencer_zero_substeps_rejected():
    _assert_invalid(SimConfig(mode="influencer", influencer_substeps=0), "influencer_substeps")


def test_influencer_negative_rank_exponent_rejected():
    _assert_invalid(
        SimConfig(mode="influencer", influencer_rank_exponent=-0.1),
        "influencer_rank_exponent",
    )

