"""P2.10 — ForceTerm composition tests.

Verifies the ForceTerm dataclass and composeForces reducer:
- Default values
- Runtime toggle (enabled=False skips term)
- Gain multiplier
- Linearity (composeForces(a+b) = composeForces(a) + composeForces(b))
- Empty terms list
- None fn
"""

import numpy as np

from pymurmur.physics.forces._base import ForceTerm, composeForces
from pymurmur.core.config import SimConfig
from pymurmur.physics.extensions._base import StepContext
from pymurmur.physics.flock import PhysicsFlock


def term_constant(val: float = 1.0):
    """Return a term fn that produces a constant force per bird."""
    def fn(flock, ctx, cfg):
        N = len(flock.positions)
        F = np.zeros((N, 3), dtype=np.float32)
        F[flock.active, 0] = val  # push along +X
        return F
    return fn


def test_force_term_defaults():
    """ForceTerm has correct defaults."""
    fn = term_constant()
    t = ForceTerm("test", fn=fn)
    assert t.name == "test"
    assert t.enabled is True
    assert t.gain == 1.0
    assert t.fn is fn


def test_force_term_disabled():
    """Disabled term contributes zero force."""
    cfg = SimConfig()
    cfg.num_boids = 5
    flock = PhysicsFlock(cfg)

    ctx = StepContext(frame=0, dt=1 / 60, rng=flock.rng, center=None, config=cfg)

    terms = [
        ForceTerm("push", gain=1.0, fn=term_constant(5.0)),
    ]
    # Enabled → force should be non-zero
    F = composeForces(flock, ctx, cfg, terms)
    assert np.allclose(F[flock.active, 0], 5.0)

    # Disabled → force should be zero
    terms[0].enabled = False
    F = composeForces(flock, ctx, cfg, terms)
    assert np.allclose(F, 0.0)


def test_force_term_gain():
    """Gain multiplier scales the force contribution."""
    cfg = SimConfig()
    cfg.num_boids = 5
    flock = PhysicsFlock(cfg)

    ctx = StepContext(frame=0, dt=1 / 60, rng=flock.rng, center=None, config=cfg)

    terms = [
        ForceTerm("push", gain=2.5, fn=term_constant(3.0)),
    ]
    F = composeForces(flock, ctx, cfg, terms)
    # 3.0 * 2.5 = 7.5
    assert np.allclose(F[flock.active, 0], 7.5)


def test_compose_forces_linear():
    """composeForces(a+b) = composeForces(a) + composeForces(b)."""
    cfg = SimConfig()
    cfg.num_boids = 5
    flock1 = PhysicsFlock(cfg)
    flock2 = PhysicsFlock(cfg)
    flock_combined = PhysicsFlock(cfg)

    ctx = StepContext(frame=0, dt=1 / 60, rng=flock1.rng, center=None, config=cfg)

    t_a = ForceTerm("a", gain=1.0, fn=term_constant(2.0))
    t_b = ForceTerm("b", gain=1.0, fn=term_constant(3.0))

    F_a = composeForces(flock1, ctx, cfg, [t_a])
    F_b = composeForces(flock2, ctx, cfg, [t_b])
    F_ab = composeForces(flock_combined, ctx, cfg, [t_a, t_b])

    np.testing.assert_allclose(F_ab, F_a + F_b)


def test_compose_forces_empty():
    """Empty terms list returns zeros."""
    cfg = SimConfig()
    cfg.num_boids = 5
    flock = PhysicsFlock(cfg)

    ctx = StepContext(frame=0, dt=1 / 60, rng=flock.rng, center=None, config=cfg)

    F = composeForces(flock, ctx, cfg, [])
    assert np.allclose(F, 0.0)


def test_compose_forces_none_fn():
    """ForceTerm with fn=None contributes nothing."""
    cfg = SimConfig()
    cfg.num_boids = 5
    flock = PhysicsFlock(cfg)

    ctx = StepContext(frame=0, dt=1 / 60, rng=flock.rng, center=None, config=cfg)

    terms = [
        ForceTerm("broken", gain=100.0, fn=None),
    ]
    F = composeForces(flock, ctx, cfg, terms)
    assert np.allclose(F, 0.0)


def test_compose_forces_inactive_unchanged():
    """Inactive birds get zero force while active ones get contributions."""
    cfg = SimConfig()
    cfg.num_boids = 10
    flock = PhysicsFlock(cfg)
    flock.active[5:] = False  # deactivate half

    ctx = StepContext(frame=0, dt=1 / 60, rng=flock.rng, center=None, config=cfg)

    terms = [
        ForceTerm("push", gain=1.0, fn=term_constant(7.0)),
    ]
    F = composeForces(flock, ctx, cfg, terms)

    # Active birds get the constant force
    assert np.allclose(F[flock.active, 0], 7.0)
    # Inactive birds get zero
    assert np.allclose(F[~flock.active], 0.0)
