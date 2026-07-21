"""ForceMode ABC, MODE_REGISTRY, and @register decorator.

P2.2 — formalises the 5 force modes behind a protocol with class-level
metadata (needs_index).

Each mode module registers its ForceMode subclass via @register("name").
The __init__.py dispatch uses MODE_REGISTRY instead of a hardcoded dict.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

    from ...core.config import SimConfig
    from ...core.types import SpatialIndex

MODE_REGISTRY: dict[str, type["ForceMode"]] = {}


class ForceMode(ABC):
    """Protocol for force modes.

    Each subclass overrides compute() with the 8-arg force signature
    and sets class-level metadata flags (needs_index).

    Usage::

        @register("projection")
        class ProjectionMode(ForceMode):
            needs_index: ClassVar[bool] = True

            @staticmethod
            def compute(positions, velocities, accelerations,
                        active, index, rng, last_theta, config):
                ...  # force computation
    """

    needs_index: ClassVar[bool] = False

    @staticmethod
    @abstractmethod
    def compute(
        positions: "np.ndarray",
        velocities: "np.ndarray",
        accelerations: "np.ndarray",
        active: "np.ndarray",
        index: "SpatialIndex | None",
        rng: "np.random.Generator",
        last_theta: "np.ndarray",
        config: "SimConfig",
    ) -> None:
        """Compute forces and mutate arrays in place.

        Args:
            positions: (N, 3) float32 — read-only
            velocities: (N, 3) float32 — vicsek mutates; others read-only
            accelerations: (N, 3) float32 — mutate in place
            active: (N,) bool
            index: spatial index or None
            rng: numpy random generator
            last_theta: (N,) float32 — occlusion angle (projection only)
            config: SimConfig
        """
        ...


@runtime_checkable
class ForceFn(Protocol):
    """Protocol for backward-compat force function aliases.

    Each force module exposes a plain-function alias like
    ``spatial_forces = SpatialMode.compute`` with an extra
    ``.needs_index`` attribute.  This Protocol lets mypy
    verify that the attribute exists instead of reporting
    ``attr-defined`` on a bare Callable.
    """

    needs_index: bool

    def __call__(
        self,
        positions: "np.ndarray",
        velocities: "np.ndarray",
        accelerations: "np.ndarray",
        active: "np.ndarray",
        index: "SpatialIndex | None",
        rng: "np.random.Generator",
        last_theta: "np.ndarray",
        config: "SimConfig",
    ) -> None:
        ...


def register(name: str):
    """Decorator to register a ForceMode subclass in MODE_REGISTRY.

    Usage::

        @register("projection")
        class ProjectionMode(ForceMode):
            ...
    """

    def decorator(cls: type[ForceMode]) -> type[ForceMode]:
        MODE_REGISTRY[name] = cls
        return cls

    return decorator
