"""ExtensionManager — assembles enabled extensions and applies them in pre_step()."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._base import StepContext  # noqa: F401  # used in type hints with annotations future
from .ecology import Ecology
from .predator import Predator
from .ripple import Ripple
from .wander import Wander

if TYPE_CHECKING:
    from ...core.config import SimConfig
    from ..flock import PhysicsFlock


class ExtensionManager:
    """Manages pluggable behavioural extensions.

    Instantiate once; call pre_step() before each simulation step.
    Extensions are lazily created on first enable and dropped on disable
    — T/K toggles take effect immediately without a reset (I5.3).

    Ecology runs first (advances day, sets predator presence), then
    Predator is conditionally applied based on predator_present(day).
    """

    def __init__(self, config: SimConfig) -> None:
        self._predator: Predator | None = None
        self._ecology: Ecology | None = None
        self._wander: Wander | None = None
        self._ripple: Ripple | None = None

        # Lazy-init from initial config
        cfg = config
        if cfg.predator_enabled:
            self._predator = Predator(cfg)
        if cfg.roosting_enabled:
            self._ecology = Ecology(cfg)
        if cfg.wander_enabled:
            self._wander = Wander()
        if cfg.ripple_enabled:
            self._ripple = Ripple()

    def pre_step(self, flock: PhysicsFlock, ctx: StepContext) -> None:
        """Run all enabled extensions before force computation.

        Checks config.*_enabled flags each frame (I5.3) — lazy-creates
        extensions on first enable, drops on disable.

        Ecology runs first to advance the day and update predator_present.
        Predator only runs when ecology says predator is present (or when
        ecology is not enabled, in which case predator is always active).
        """
        cfg = ctx.config

        # ── Lazy lifecycle: check config flags each frame (I5.3) ──
        if cfg.roosting_enabled:
            if self._ecology is None:
                self._ecology = Ecology(cfg)
        else:
            self._ecology = None

        if cfg.predator_enabled:
            if self._predator is None:
                self._predator = Predator(cfg)
        else:
            self._predator = None

        if cfg.wander_enabled:
            if self._wander is None:
                self._wander = Wander()
        else:
            self._wander = None

        if cfg.ripple_enabled:
            if self._ripple is None:
                self._ripple = Ripple()
        else:
            self._ripple = None

        eco = self._ecology
        pred = self._predator

        # Run ecology first (advances day, sets predator_active flag)
        if eco is not None:
            eco.apply(flock, ctx)
            # P4.8: Expose coherence factor on flock (single source of truth).
            # Engine bridges flock.coherence_factor → config._coherence_factor
            # before force computation so spatial mode can read it.
            flock.coherence_factor = eco.coherence_factor
        else:
            flock.coherence_factor = 1.0

        # Run predator only if present (or ecology not enabled → always present)
        if pred is not None:
            if eco is None or eco.predator_active:  # I5.4: public attr
                pred.apply(flock, ctx)

        # Run remaining extensions (wander, ripple)
        if self._wander is not None:
            self._wander.apply(flock, ctx)
        if self._ripple is not None:
            self._ripple.apply(flock, ctx)

    @property
    def count(self) -> int:
        return sum(1 for e in (self._ecology, self._predator,
                                self._wander, self._ripple) if e is not None)

    @property
    def predator_position(self):
        """D7/S2.A8: threat marker position (np.ndarray) for rendering,
        or None when no predator extension is active."""
        return self._predator.position if self._predator is not None else None
