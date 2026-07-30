"""NoiseStrategy ABC, NOISE_STRATEGY_REGISTRY, and @register decorator.

Modularity pass 8: extracts the 5-way if/elif noise-mode chain from
spatial.py::compute() (additive/maxwellian/none/seed_sinusoidal/velocity)
behind a registry, mirroring physics/plugins/speed_model.py's
SpeedModel/SPEED_MODEL_REGISTRY pattern exactly.

Pure extraction, not a redesign — each strategy below is a faithful
1:1 port of its former if/elif branch, including its non-uniform side
effects: some strategies return a genuine (N, 3) force contribution to
be added to accelerations (additive, seed_sinusoidal); others mutate
velocities directly and return zeros (maxwellian); "velocity" stashes
a one-shot side-channel array on config for flock.integrate() to
consume (S2.B2's post-clamp velocity-domain noise) and also returns
zeros. apply()'s (N, 3) return value is always the accelerations-space
contribution — never None — so composeForces()-style callers can treat
every strategy uniformly even though their side effects differ.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...core.config import SimConfig

NOISE_STRATEGY_REGISTRY: dict[str, type["NoiseStrategy"]] = {}


class NoiseStrategy(ABC):
    """Protocol for spatial-mode noise strategies.

    Each subclass overrides apply() with the shared noise-injection
    signature.

    Usage::

        @register("additive")
        class AdditiveNoiseStrategy(NoiseStrategy):
            @staticmethod
            def apply(positions, velocities, active, n_active, config, rng):
                ...  # return an (N, 3) accelerations-space contribution
    """

    @staticmethod
    @abstractmethod
    def apply(
        positions: "np.ndarray",
        velocities: "np.ndarray",
        active: "np.ndarray",
        n_active: int,
        config: "SimConfig",
        rng: "np.random.Generator",
    ) -> "np.ndarray":
        """Inject this strategy's noise, mutating velocities/config as needed.

        Args:
            positions: (N, 3) float32 — read-only
            velocities: (N, 3) float32 — mutated in place by strategies
                whose noise acts in velocity-space (maxwellian)
            active: (N,) bool
            n_active: active.sum(), precomputed by the caller
            config: SimConfig — strategies read config.noise_scale and
                may set config._spatial_velocity_noise (one-shot,
                cleared to None by every strategy except "velocity")
            rng: numpy random generator

        Returns:
            (N, 3) float32 — accelerations-space contribution (zeros
            for strategies whose effect happens outside the
            accumulate-then-clamp force pipeline).
        """
        ...


def register(name: str):
    """Decorator to register a NoiseStrategy subclass in NOISE_STRATEGY_REGISTRY.

    Usage::

        @register("additive")
        class AdditiveNoiseStrategy(NoiseStrategy):
            ...
    """

    def decorator(cls: type[NoiseStrategy]) -> type[NoiseStrategy]:
        NOISE_STRATEGY_REGISTRY[name] = cls
        return cls

    return decorator


# ── Registered strategies ─────────────────────────────────────────

@register("none")
class NoneNoiseStrategy(NoiseStrategy):
    """No noise — the accelerations-space contribution is all-zero."""

    @staticmethod
    def apply(positions, velocities, active, n_active, config, rng):
        config._spatial_velocity_noise = None
        return np.zeros((len(positions), 3), dtype=np.float32)


@register("velocity")
class VelocityNoiseStrategy(NoiseStrategy):
    """S2.B2: (U^3 - 0.5)*noise_scale added directly to velocity, after
    v+=a and before the final speed clamp (spec pipeline order), not to
    accelerations. Stashed on config for flock.integrate() to consume
    and clear (one-shot) — this strategy's own apply() always returns
    zeros for the accelerations-space contribution.
    """

    @staticmethod
    def apply(positions, velocities, active, n_active, config, rng):
        vel_noise = np.zeros((len(positions), 3), dtype=np.float32)
        u = rng.uniform(0.0, 1.0, (n_active, 3)).astype(np.float32)
        vel_noise[active] = (u ** 3 - 0.5) * config.noise_scale
        config._spatial_velocity_noise = vel_noise
        return np.zeros((len(positions), 3), dtype=np.float32)


@register("maxwellian")
class MaxwellianNoiseStrategy(NoiseStrategy):
    """Maxwellian: isotropic velocity perturbation scaled by
    noise_scale, applied directly to velocities (not accelerations).
    Returns zeros for the accelerations-space contribution.
    """

    @staticmethod
    def apply(positions, velocities, active, n_active, config, rng):
        from ..forces._base import noise_force

        config._spatial_velocity_noise = None
        noise_full = np.zeros((len(positions), 3), dtype=np.float32)
        noise_full[active] = noise_force(n_active, 1.0, rng)
        if n_active > 0:
            velocities[active] += noise_full[active] * config.noise_scale * 0.1
        return np.zeros((len(positions), 3), dtype=np.float32)


@register("seed_sinusoidal")
class SeedSinusoidalNoiseStrategy(NoiseStrategy):
    """S2.B11: deterministic per-bird sinusoids (seed_noise3, L0 atom,
    +-0.18/axis) instead of a seeded-rng draw -- same (seeds, t) always
    gives the same noise, independent of rng call order elsewhere in
    the pipeline.
    """

    @staticmethod
    def apply(positions, velocities, active, n_active, config, rng):
        from ...core.types import seed_noise3

        config._spatial_velocity_noise = None
        noise_full = np.zeros((len(positions), 3), dtype=np.float32)
        active_idx_noise = np.where(active)[0]
        seeds = np.arange(len(active_idx_noise), dtype=np.float32)
        t = getattr(config, '_field_time', 0.0)
        noise_full[active_idx_noise] = seed_noise3(seeds, t) * (
            config.noise_scale / 0.18
        )
        return noise_full


@register("additive")
class AdditiveNoiseStrategy(NoiseStrategy):
    """Default: isotropic random-direction noise, magnitude =
    noise_scale, added as a genuine accelerations-space contribution.
    """

    @staticmethod
    def apply(positions, velocities, active, n_active, config, rng):
        from ..forces._base import noise_force

        config._spatial_velocity_noise = None
        noise = noise_force(n_active, config.noise_scale, rng)
        noise_full = np.zeros((len(positions), 3), dtype=np.float32)
        noise_full[active] = noise
        return noise_full
