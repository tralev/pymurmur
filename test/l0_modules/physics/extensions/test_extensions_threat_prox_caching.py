"""Extensions — threat_prox contract (I5.4), Ecology N_active caching (I5.3).

Split out of test_extensions.py (file-size split).
"""

import numpy as np

from pymurmur.physics.extensions import ExtensionManager
from pymurmur.physics.extensions._base import StepContext
from pymurmur.physics.extensions.ecology import Ecology
from pymurmur.physics.flock import PhysicsFlock
from test.l0_modules.physics.extensions.test_extensions import _make_ctx

# ── threat_prox contract (I5.4) ───────────────────────────────────

class TestThreatProx:
    """M15-M18: threat_prox array published by Predator extension."""

    @staticmethod
    def _mk_ctx(flock, config, threat_prox=None):
        """Create a StepContext, optionally with a pre-set threat_prox."""
        return StepContext(
            frame=0, dt=1.0 / 60.0, rng=flock.rng,
            center=flock.center, config=config,
            threat_prox=threat_prox,
        )

    def test_threat_prox_none_when_predator_disabled(self, default_config):
        """M15: ctx.threat_prox stays None when predator is not enabled."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.predator_enabled = False
        cfg.roosting_enabled = False

        mgr = ExtensionManager(cfg)
        flock = PhysicsFlock(cfg)
        ctx = self._mk_ctx(flock, cfg)

        mgr.pre_step(flock, ctx)

        # No predator → ctx.threat_prox must still be None
        assert ctx.threat_prox is None, (
            f"threat_prox must be None when predator is disabled, "
            f"got {type(ctx.threat_prox)}"
        )

    def test_threat_prox_not_none_when_predator_enabled(self, default_config):
        """M16: ctx.threat_prox is set to an array when predator runs.

        Predator.apply() publishes threat_prox for downstream consumers.
        """
        cfg = default_config
        cfg.num_boids = 10
        cfg.predator_enabled = True
        cfg.roosting_enabled = False  # no ecology to gate predator

        mgr = ExtensionManager(cfg)
        flock = PhysicsFlock(cfg)
        ctx = self._mk_ctx(flock, cfg)

        mgr.pre_step(flock, ctx)

        assert ctx.threat_prox is not None, (
            "threat_prox must be set by Predator.apply()"
        )

    def test_threat_prox_has_correct_structure(self, default_config):
        """M17: ctx.threat_prox is an N_capacity float32 array with [0,1] values.

        The array has one entry per capacity slot; inactive slots stay at 0.
        Values are in [0, 1] where 1 = at predator position, 0 = at radius edge.
        """
        cfg = default_config
        cfg.num_boids = 10
        cfg.predator_enabled = True
        cfg.predator_threat_radius = 200.0
        cfg.roosting_enabled = False

        mgr = ExtensionManager(cfg)
        flock = PhysicsFlock(cfg)
        ctx = self._mk_ctx(flock, cfg)

        mgr.pre_step(flock, ctx)

        tp = ctx.threat_prox
        assert tp is not None
        assert isinstance(tp, np.ndarray), (
            f"threat_prox must be ndarray, got {type(tp)}"
        )
        assert tp.dtype == np.float32, (
            f"threat_prox dtype must be float32, got {tp.dtype}"
        )
        assert tp.shape == (flock.N_capacity,), (
            f"threat_prox shape must be (N_capacity,), got {tp.shape}"
        )
        # Values must be in [0, 1]
        assert np.all(tp >= 0.0), "threat_prox values must be >= 0"
        assert np.all(tp <= 1.0), "threat_prox values must be <= 1"
        # At least one active bird should have non-zero threat if predator is near
        assert np.isfinite(tp).all(), "threat_prox must be finite"

    def test_threat_prox_none_in_context_when_predator_gated_by_ecology(
        self, default_config
    ):
        """M18: ctx.threat_prox stays None when ecology gates predator off.

        If ecology.predator_active is False, predator's apply() is never
        called, so ctx.threat_prox remains None.
        """
        cfg = default_config
        cfg.num_boids = 10
        cfg.predator_enabled = True
        cfg.roosting_enabled = True  # ecology enabled → can gate predator

        mgr = ExtensionManager(cfg)
        flock = PhysicsFlock(cfg)

        # Force ecology to signal no predator
        mgr._ecology.predator_active = False

        ctx = self._mk_ctx(flock, cfg)
        mgr.pre_step(flock, ctx)

        assert ctx.threat_prox is None, (
            "threat_prox must be None when ecology gates predator off"
        )

    def test_threat_prox_present_when_ecology_allows_predator(
        self, default_config
    ):
        """When ecology.predator_active is True, predator runs → threat_prox set."""
        cfg = default_config
        cfg.num_boids = 10
        cfg.predator_enabled = True
        cfg.roosting_enabled = True

        mgr = ExtensionManager(cfg)
        flock = PhysicsFlock(cfg)

        # Force ecology to signal predator IS active
        mgr._ecology.predator_active = True

        ctx = self._mk_ctx(flock, cfg)
        mgr.pre_step(flock, ctx)

        assert ctx.threat_prox is not None, (
            "threat_prox must be set when ecology allows predator"
        )
        assert isinstance(ctx.threat_prox, np.ndarray)


# ── Ecology N_active caching (I5.3) ───────────────────────────────

class TestEcologyCaching:
    """M7: Ecology must recompute N_active each frame, not cache it.

    The dusk roost pull uses a smoothstep mass_factor based on N_active
    relative to critical_mass. If N_active is cached and not invalidated
    after add_boids/remove_boids, the wrong mass_factor is used.
    """

    def test_ecology_dusk_mass_factor_responds_to_n_active(self, default_config):
        """M7: Same Ecology instance responds to changing N_active.

        Uses a single Ecology instance with one flock — apply to small
        flock, add birds, apply again. If N_active were cached, the
        second apply would use the stale small-flock value, producing
        the same force both times.
        """
        cfg = default_config
        cfg.ecology_critical_mass = 500

        # S2.B8: gate window is [0.4,1.2]x critical_mass = [200,600] — start
        # inside the window (250, dampened but nonzero) rather than below it
        # (which now gates fully to 0, see coherence_gate).
        cfg.num_boids = 250
        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0

        eco = Ecology(cfg)
        eco._day = 172.0 + 0.82  # dusk window
        eco._day_dt = 0  # freeze time

        eco.apply(flock, _make_ctx(flock, cfg))
        force_small = float(
            np.linalg.norm(
                flock.accelerations[flock.active], axis=1
            ).mean()
        )
        assert force_small > 0, "Small flock should get some roost pull"

        # Add 400 birds — N_active goes from 250 → 650, above the gate window
        flock.accelerations[:] = 0.0  # reset forces
        flock.add_boids(400, cfg)

        # Same Ecology, same dusk time, now much larger flock
        eco.apply(flock, _make_ctx(flock, cfg))
        force_large = float(
            np.linalg.norm(
                flock.accelerations[flock.active], axis=1
            ).mean()
        )

        # Force must increase — mass_factor went from dampened (~0.043) → 1.0
        assert force_large > force_small * 3.0, (
            f"Same Ecology instance must respond to N_active change: "
            f"large flock force={force_large:.6f}, "
            f"small flock force={force_small:.6f}. "
            f"If N_active were cached, both would be equal."
        )

    def test_ecology_n_active_recomputed_after_add_boids(
        self, default_config
    ):
        """M7: After add_boids, next apply() uses the new N_active.

        If Ecology cached N_active from a previous frame, the mass_factor
        would stay dampened even after adding birds.
        """
        cfg = default_config
        cfg.ecology_critical_mass = 500

        # S2.B8: gate window is [0.4,1.2]x critical_mass = [200,600]. Start
        # inside the window (below it now gates fully to 0, so force_before
        # would be exactly 0 and the "still get some force" assertion below
        # would be meaningless).
        cfg.num_boids = 250
        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0

        eco = Ecology(cfg)
        eco._day = 172.0 + 0.82  # dusk
        eco._day_dt = 0

        # First apply: small flock → dampened force
        eco.apply(flock, _make_ctx(flock, cfg))
        force_before = float(
            np.linalg.norm(
                flock.accelerations[flock.active], axis=1
            ).mean()
        )
        assert force_before > 0, "Small flock should still get some force"

        # Add many birds — N_active changes from 250 → 750, above the gate window
        flock.accelerations[:] = 0.0  # reset forces
        flock.add_boids(500, cfg)

        # Second apply: flock is now much larger → should get stronger pull
        eco.apply(flock, _make_ctx(flock, cfg))
        force_after = float(
            np.linalg.norm(
                flock.accelerations[flock.active], axis=1
            ).mean()
        )

        # Force must increase because mass_factor went from dampened → near 1.0
        assert force_after > force_before, (
            f"After adding 500 birds (N_active={flock.N_active}), "
            f"force ({force_after:.6f}) must exceed "
            f"pre-add force ({force_before:.6f}). "
            f"If N_active were cached, forces would be equal."
        )

    def test_ecology_n_active_zero_birds_handled(self, default_config):
        """M7: Ecology handles zero active birds without division by zero."""
        cfg = default_config
        cfg.num_boids = 0  # empty flock
        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0

        eco = Ecology(cfg)
        eco._day = 172.0 + 0.82
        eco._day_dt = 0

        # Should not crash — n_active=0 → t=0 → mass_factor=0 → forces unchanged
        eco.apply(flock, _make_ctx(flock, cfg))

        # Forces must remain finite (no NaN from division by zero)
        assert np.isfinite(flock.accelerations).all(), (
            "Ecology must handle zero active birds without producing NaN"
        )


