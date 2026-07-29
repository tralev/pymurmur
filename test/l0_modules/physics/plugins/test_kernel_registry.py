"""Unit tests for physics.plugins.kernel_registry — SEPARATION_KERNEL_REGISTRY,
ALIGNMENT_KERNEL_REGISTRY, COHESION_KERNEL_REGISTRY, and KernelInfo metadata,
mirroring test_boundary_registry.py's shape (registry-membership assertions
for a modularity-pass registry).

Modularity pass 7: formalises the separation/alignment/cohesion kernel
string-dispatch (previously if/elif chains + separate boolean frozensets
in force_kernels.py) behind one KernelInfo-per-kernel registry. These
tests verify registry contents and metadata flags; the kernel math itself
is covered by test_force_kernels.py.
"""

from __future__ import annotations

from pymurmur.physics.plugins.kernel_registry import (
    ALIGNMENT_KERNEL_REGISTRY,
    COHESION_KERNEL_REGISTRY,
    KernelInfo,
    SEPARATION_KERNEL_REGISTRY,
)


class TestSeparationKernelRegistry:
    def test_all_eleven_kernels_registered(self):
        assert set(SEPARATION_KERNEL_REGISTRY.keys()) == {
            "sum", "mean", "unit", "exp", "linear_ramp", "asymptotic",
            "velocity_weighted", "cosine_zone", "linear", "nearest_only",
            "bell_zone",
        }

    def test_entries_are_kernel_info(self):
        for info in SEPARATION_KERNEL_REGISTRY.values():
            assert isinstance(info, KernelInfo)
            assert callable(info.fn)

    def test_needs_radius_flags(self):
        needs_radius = {
            name for name, info in SEPARATION_KERNEL_REGISTRY.items()
            if info.needs_radius
        }
        assert needs_radius == {"exp", "linear_ramp", "asymptotic", "bell_zone"}

    def test_needs_zone_width_flags(self):
        needs_zone_width = {
            name for name, info in SEPARATION_KERNEL_REGISTRY.items()
            if info.needs_zone_width
        }
        assert needs_zone_width == {"bell_zone"}

    def test_needs_closing_speed_flags(self):
        needs_closing_speed = {
            name for name, info in SEPARATION_KERNEL_REGISTRY.items()
            if info.needs_closing_speed
        }
        assert needs_closing_speed == {"velocity_weighted"}

    def test_needs_heading_flags(self):
        needs_heading = {
            name for name, info in SEPARATION_KERNEL_REGISTRY.items()
            if info.needs_heading
        }
        assert needs_heading == {"cosine_zone"}

    def test_default_kernel_info_has_no_flags(self):
        """sum/mean/unit/linear/nearest_only need nothing beyond diffs/dists/close."""
        for name in ("sum", "mean", "unit", "linear", "nearest_only"):
            info = SEPARATION_KERNEL_REGISTRY[name]
            assert not info.needs_radius
            assert not info.needs_zone_width
            assert not info.needs_closing_speed
            assert not info.needs_heading


class TestAlignmentKernelRegistry:
    def test_all_four_kernels_registered(self):
        assert set(ALIGNMENT_KERNEL_REGISTRY.keys()) == {
            "unweighted", "fov_weighted", "circular_mean_2d", "bell_zone",
        }

    def test_needs_heading_flags(self):
        needs_heading = {
            name for name, info in ALIGNMENT_KERNEL_REGISTRY.items()
            if info.needs_heading
        }
        assert needs_heading == {"fov_weighted"}

    def test_needs_radius_and_zone_width_flags(self):
        for flagged in (
            {name for name, info in ALIGNMENT_KERNEL_REGISTRY.items() if info.needs_radius},
            {name for name, info in ALIGNMENT_KERNEL_REGISTRY.items() if info.needs_zone_width},
        ):
            assert flagged == {"bell_zone"}


class TestCohesionKernelRegistry:
    def test_all_three_kernels_registered(self):
        assert set(COHESION_KERNEL_REGISTRY.keys()) == {
            "unweighted", "inverse_distance", "bell_zone",
        }

    def test_needs_radius_and_zone_width_flags(self):
        for flagged in (
            {name for name, info in COHESION_KERNEL_REGISTRY.items() if info.needs_radius},
            {name for name, info in COHESION_KERNEL_REGISTRY.items() if info.needs_zone_width},
        ):
            assert flagged == {"bell_zone"}
