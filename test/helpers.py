"""Shared test helpers — importable from any test file."""


def _step_flock(flock, config, dt=1.0 / 60.0):
    """Step a PhysicsFlock — orchestrates rebuild → forces → integrate.

    Replaces the removed flock.step() (I4.2). The engine now owns
    force orchestration, but unit tests still need this helper.
    """
    from pymurmur.physics.forces import compute_all_forces, mode_needs_index
    if flock._index is not None and mode_needs_index(config.mode):
        flock._index.rebuild(flock.positions, flock.active)
    compute_all_forces(flock, config)
    flock.integrate(config, dt)


def _call_force(fn, flock, cfg):
    """Call a force function with the standard 8-arg signature unpacked from flock.

    Usage: `_call_force(field_forces, flock, cfg)` instead of
    `field_forces(flock.positions, flock.velocities, flock.accelerations,
    flock.active, flock.get_index(), flock.rng, flock.last_theta, cfg)`
    """
    return fn(
        flock.positions, flock.velocities, flock.accelerations,
        flock.active, flock.get_index(), flock.rng, flock.last_theta, cfg,
    )
