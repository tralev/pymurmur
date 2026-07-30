"""SeparationKernel ABC, SEPARATION_KERNEL_REGISTRY, and @register decorator.

Modularity pass 7: formalises the string-to-function dispatch in
_base.py's _dispatch_separation_kernel() as a registry, mirroring
physics/forces/_mode.py's ForceMode/MODE_REGISTRY pattern.

Comparison.md's Separation taxonomy lists 12 techniques across the
surveyed implementations; this codebase implements 10 of them plus
2 composite kernels. Each kernel function in force_kernels.py is
registered here with its name, parameter requirements, and the
kernel function itself.

Alignment and cohesion kernel dispatches follow the same pattern
via ALIGNMENT_KERNEL_REGISTRY and COHESION_KERNEL_REGISTRY.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

# ── Kernel descriptor ─────────────────────────────────────────────


@dataclass
class KernelInfo:
    """Metadata for a registered kernel function."""

    fn: Callable[..., np.ndarray]
    """The kernel function from force_kernels.py."""

    needs_radius: bool = False
    """True if kernel needs a `radius` kwarg (exp, linear_ramp, etc.)."""

    needs_zone_width: bool = False
    """True if kernel needs a `zone_width` kwarg (bell_zone only)."""

    needs_closing_speed: bool = False
    """True if kernel needs `closing_speed` kwarg (velocity_weighted)."""

    needs_heading: bool = False
    """True if kernel needs `heading` kwarg (cosine_zone, fov_weighted)."""

    needs_neighbor_vel: bool = False
    """True for alignment kernels that need neighbor velocities."""


SEPARATION_KERNEL_REGISTRY: dict[str, KernelInfo] = {}
ALIGNMENT_KERNEL_REGISTRY: dict[str, KernelInfo] = {}
COHESION_KERNEL_REGISTRY: dict[str, KernelInfo] = {}


def register_separation(name: str, **kwargs: Any):
    """Decorator to register a separation kernel.

    Usage::

        @register_separation("sum")
        def kernel_sum(diffs, dists, close):
            ...
    """

    def decorator(fn):
        SEPARATION_KERNEL_REGISTRY[name] = KernelInfo(fn=fn, **kwargs)
        return fn

    return decorator


def register_alignment(name: str, **kwargs: Any):
    """Decorator to register an alignment kernel."""

    def decorator(fn):
        ALIGNMENT_KERNEL_REGISTRY[name] = KernelInfo(fn=fn, **kwargs)
        return fn

    return decorator


def register_cohesion(name: str, **kwargs: Any):
    """Decorator to register a cohesion kernel."""

    def decorator(fn):
        COHESION_KERNEL_REGISTRY[name] = KernelInfo(fn=fn, **kwargs)
        return fn

    return decorator


# ── Separation kernels ────────────────────────────────────────────

@register_separation("sum")
def _sep_sum(diffs, dists, close):
    from ..forces.force_kernels import kernel_sum
    return kernel_sum(diffs, dists, close)


@register_separation("mean")
def _sep_mean(diffs, dists, close):
    from ..forces.force_kernels import kernel_mean
    return kernel_mean(diffs, dists, close)


@register_separation("unit")
def _sep_unit(diffs, dists, close):
    from ..forces.force_kernels import kernel_unit
    return kernel_unit(diffs, dists, close)


@register_separation("exp", needs_radius=True)
def _sep_exp(diffs, dists, close, radius):
    from ..forces.force_kernels import kernel_exp
    return kernel_exp(diffs, dists, close, radius)


@register_separation("linear_ramp", needs_radius=True)
def _sep_linear_ramp(diffs, dists, close, radius):
    from ..forces.force_kernels import kernel_linear_ramp
    return kernel_linear_ramp(diffs, dists, close, radius)


@register_separation("asymptotic", needs_radius=True)
def _sep_asymptotic(diffs, dists, close, radius):
    from ..forces.force_kernels import kernel_asymptotic
    return kernel_asymptotic(diffs, dists, close, radius)


@register_separation("velocity_weighted", needs_closing_speed=True)
def _sep_velocity_weighted(diffs, dists, close, closing_speed):
    from ..forces.force_kernels import kernel_velocity_weighted
    return kernel_velocity_weighted(diffs, dists, close, closing_speed)


@register_separation("cosine_zone", needs_heading=True)
def _sep_cosine_zone(diffs, dists, close, heading):
    from ..forces.force_kernels import kernel_cosine_zone
    return kernel_cosine_zone(diffs, dists, close, heading)


@register_separation("linear")
def _sep_linear(diffs, dists, close):
    from ..forces.force_kernels import kernel_linear
    return kernel_linear(diffs, dists, close)


@register_separation("nearest_only")
def _sep_nearest_only(diffs, dists, close):
    from ..forces.force_kernels import kernel_nearest_only
    return kernel_nearest_only(diffs, dists, close)


@register_separation("bell_zone", needs_radius=True, needs_zone_width=True)
def _sep_bell_zone(diffs, dists, close, radius, zone_width):
    from ..forces.force_kernels import kernel_bell_zone
    return kernel_bell_zone(diffs, dists, close, radius, zone_width)


# ── Alignment kernels ─────────────────────────────────────────────

@register_alignment("unweighted")
def _align_unweighted(diffs, dists, close, neighbor_vel):
    return np.mean(neighbor_vel, axis=-2)


@register_alignment("fov_weighted", needs_heading=True)
def _align_fov_weighted(diffs, dists, close, neighbor_vel, heading, fov_min):
    from ..forces.force_kernels import kernel_fov_weighted
    return kernel_fov_weighted(diffs, dists, close, heading, neighbor_vel, fov_min)


@register_alignment("spherical_mean")
def _align_spherical_mean(diffs, dists, close, neighbor_vel):
    from ..forces.force_kernels import kernel_spherical_mean_alignment
    return kernel_spherical_mean_alignment(diffs, dists, close, neighbor_vel)


@register_alignment("bell_zone", needs_radius=True, needs_zone_width=True)
def _align_bell_zone(diffs, dists, close, neighbor_vel, radius, zone_width):
    from ..forces.force_kernels import kernel_bell_zone_alignment
    return kernel_bell_zone_alignment(diffs, dists, close, neighbor_vel, radius, zone_width)


# ── Cohesion kernels ──────────────────────────────────────────────

@register_cohesion("unweighted")
def _coh_unweighted(diffs):
    from ..forces.force_kernels import kernel_unweighted
    return kernel_unweighted(diffs)


@register_cohesion("inverse_distance")
def _coh_inverse_distance(diffs, dists, close):
    from ..forces.force_kernels import kernel_inverse_distance
    return kernel_inverse_distance(diffs, dists, close)


@register_cohesion("bell_zone", needs_radius=True, needs_zone_width=True)
def _coh_bell_zone(diffs, dists, close, radius, zone_width):
    from ..forces.force_kernels import kernel_bell_zone_cohesion
    return kernel_bell_zone_cohesion(diffs, dists, close, radius, zone_width)
