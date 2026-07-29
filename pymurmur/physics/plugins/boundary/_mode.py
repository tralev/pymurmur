"""BoundaryMode ABC, BOUNDARY_REGISTRY, and @register decorator.

Modularity pass: formalises the 5 boundary strategies (previously a
plain if/elif in boid.py::_apply_boundary) behind a registry, mirroring
physics/forces/_mode.py's proven ForceMode/MODE_REGISTRY/@register
pattern exactly.

Each strategy module registers its BoundaryMode subclass via
@register("name"). boid.py's _apply_boundary() dispatches through
BOUNDARY_REGISTRY instead of a hardcoded if/elif chain.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import numpy as np

BOUNDARY_REGISTRY: dict[str, type["BoundaryMode"]] = {}


class BoundaryMode(ABC):
    """Protocol for boundary strategies.

    Each subclass overrides apply() with the shared boundary signature.

    Usage::

        @register("margin")
        class MarginBoundary(BoundaryMode):
            @staticmethod
            def apply(positions, velocities, active, width, height, depth,
                      sphere_radius, avoidance_factor, center=None):
                ...  # boundary enforcement
    """

    #: Human-readable name, set by @register for introspection/error messages.
    name: ClassVar[str] = ""

    @staticmethod
    @abstractmethod
    def apply(
        positions: "np.ndarray",
        velocities: "np.ndarray",
        active: "np.ndarray",
        width: float,
        height: float,
        depth: float,
        sphere_radius: float,
        avoidance_factor: float,
        center: "np.ndarray | None" = None,
    ) -> None:
        """Enforce boundary conditions on active birds, mutating in place.

        Args:
            positions: (N, 3) float32 — mutated by hard-boundary strategies
            velocities: (N, 3) float32 — mutated by soft-boundary strategies
            active: (N,) bool
            width, height, depth: domain bounds
            sphere_radius: radius for sphere/sphere_soft strategies
            avoidance_factor: strategy-specific push strength
            center: (3,) float32 — sphere strategies' centre (defaults to
                origin if None; boid.py's integrate() defaults this to the
                domain centre before calling _apply_boundary)
        """
        ...


def register(name: str):
    """Decorator to register a BoundaryMode subclass in BOUNDARY_REGISTRY.

    Usage::

        @register("margin")
        class MarginBoundary(BoundaryMode):
            ...
    """

    def decorator(cls: type[BoundaryMode]) -> type[BoundaryMode]:
        cls.name = name
        BOUNDARY_REGISTRY[name] = cls
        return cls

    return decorator
