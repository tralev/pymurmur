"""ObstacleAvoidanceStrategy ABC, OBSTACLE_AVOIDANCE_REGISTRY, and
@register decorator.

Modularity pass 3: formalises pymurmur's one obstacle-avoidance strategy
(SDF-gradient static fly-away + linear time-to-collision predictive
steering, ObstacleScene.avoidance_accel()) behind a registry, mirroring
physics/forces/_mode.py's ForceMode/MODE_REGISTRY pattern and
physics/boundary/_mode.py's BoundaryMode pattern.

Comparison.md's taxonomy lists 7 detection/response mechanisms across
the surveyed implementations; this codebase implements exactly 1
(SDF-gradient based, not raycast-based like most of the other 6). This
pass registers that one existing strategy and stops there — building
the other 6 is genuinely new physics/feature work (different geometry
representations, real design and tuning validation), not extraction,
and deliberately out of scope here (same scaffolding-only precedent as
the boundary and neighbor-selection registries).

ObstacleScene.avoidance_accel() itself is untouched — this module wraps
it, it does not replace it. Direct calls to avoidance_accel() (existing
tests, any future caller) keep working exactly as before.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from ..obstacles import ObstacleScene

OBSTACLE_AVOIDANCE_REGISTRY: dict[str, type["ObstacleAvoidanceStrategy"]] = {}


class ObstacleAvoidanceStrategy(ABC):
    """Protocol for obstacle-avoidance strategies.

    Usage::

        @register("sdf_ttc")
        class SDFTTCStrategy(ObstacleAvoidanceStrategy):
            @staticmethod
            def compute_accel(scene, positions, velocities, **kwargs):
                ...  # detection + response
    """

    @staticmethod
    @abstractmethod
    def compute_accel(
        scene: "ObstacleScene",
        positions: "np.ndarray",
        velocities: "np.ndarray",
        **kwargs: Any,
    ) -> "np.ndarray":
        """Compute avoidance acceleration for the given positions/velocities.

        Args:
            scene: the ObstacleScene holding avoidance-relevant geometry/state.
            positions: (N, 3) float32
            velocities: (N, 3) float32
            **kwargs: strategy-specific weights (e.g. static_weight,
                predictive_weight, fly_away_max_dist, min_time_to_collide
                for "sdf_ttc" — a future raycast-based strategy might take
                a probe count instead).

        Returns:
            (N, 3) float32 acceleration.
        """
        ...


def register(name: str):
    """Decorator to register an ObstacleAvoidanceStrategy subclass in
    OBSTACLE_AVOIDANCE_REGISTRY.

    Usage::

        @register("sdf_ttc")
        class SDFTTCStrategy(ObstacleAvoidanceStrategy):
            ...
    """

    def decorator(cls: type[ObstacleAvoidanceStrategy]) -> type[ObstacleAvoidanceStrategy]:
        OBSTACLE_AVOIDANCE_REGISTRY[name] = cls
        return cls

    return decorator


@register("sdf_ttc")
class SDFTTCStrategy(ObstacleAvoidanceStrategy):
    """Wraps ObstacleScene.avoidance_accel() unchanged — SDF-gradient
    static fly-away + linear time-to-collision predictive steering.
    The only strategy this codebase implements today.

    kwargs: static_weight, predictive_weight, fly_away_max_dist,
    min_time_to_collide (all default 0.0, matching avoidance_accel()'s
    own defaults).
    """

    @staticmethod
    def compute_accel(scene, positions, velocities, **kwargs):
        return scene.avoidance_accel(positions, velocities, **kwargs)
