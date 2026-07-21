"""Independent L0 function tests for P3.7 ripple envelopes.

Tests the _smoothstep helper, 3-train staggering, envelope bounds,
moving Lissajous origins, radial+twist forces, and envelope sum export.
"""

import numpy as np

from pymurmur.core.config import SimConfig
from pymurmur.physics.extensions._base import StepContext
from pymurmur.physics.extensions.ripple import Ripple, _smoothstep
from pymurmur.physics.flock import PhysicsFlock


def _make_ctx(flock, config, frame=0, dt=1.0 / 60.0):
    return StepContext(
        frame=frame, dt=dt, rng=flock.rng,
        center=flock.center, config=config,
    )


class TestRippleSmoothstep:
    """P3.7: Smoothstep envelope helper."""

    def test_smoothstep_zero_below_edge0(self):
        assert _smoothstep(0.6, 1.7, 0.3) == 0.0

    def test_smoothstep_one_above_edge1(self):
        assert _smoothstep(0.6, 1.7, 2.0) == 1.0

    def test_smoothstep_monotone(self):
        vals = [_smoothstep(0.6, 1.7, t) for t in np.linspace(0.5, 2.0, 20)]
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1], f"Not monotone at index {i}"


class TestRippleThreeTrains:
    """P3.7: 3-train staggered ripple behavior."""

    def test_ripple_apply_runs_without_error(self):
        cfg = SimConfig()
        cfg.num_boids = 20
        cfg.mode = "field"
        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        r = Ripple()
        r.apply(flock, _make_ctx(flock, cfg))
        assert np.isfinite(flock.accelerations).all()

    def test_ripple_envelope_exported_to_config(self):
        """P3.7/D10: envelope exported as per-bird (N_capacity,) float32 array."""
        cfg = SimConfig()
        cfg.num_boids = 20
        cfg.mode = "field"
        flock = PhysicsFlock(cfg)
        r = Ripple()
        r.apply(flock, _make_ctx(flock, cfg))
        assert hasattr(cfg, '_ripple_envelope_sum'), "Envelope must be exported"
        env = cfg._ripple_envelope_sum
        assert isinstance(env, np.ndarray), "D10: envelope must be per-bird array"
        assert env.shape == (len(flock.positions),)
        assert env.dtype == np.float32

    def test_ripple_produces_forces_at_later_times(self):
        """After sufficient time, ripple trains produce forces."""
        cfg = SimConfig()
        cfg.num_boids = 50
        cfg.mode = "field"
        cfg.field_flow = 1.0
        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        r = Ripple()
        # Advance to t=15 so all 3 trains are within their 28s cycle
        r._t = 15.0
        r.apply(flock, _make_ctx(flock, cfg))
        acc_mags = np.linalg.norm(flock.accelerations[flock.active], axis=1)
        # At least some birds should feel ripple forces at this time
        assert np.any(acc_mags > 1e-6), "Ripple should produce forces at t=15"

    def test_ripple_zero_active(self):
        """Ripple handles zero active birds."""
        cfg = SimConfig()
        cfg.num_boids = 10
        flock = PhysicsFlock(cfg)
        flock.active[:] = False
        r = Ripple()
        r.apply(flock, _make_ctx(flock, cfg))
        # Should not crash; D10: per-bird envelope defaults to all-zero
        assert np.all(cfg._ripple_envelope_sum == 0.0)

    def test_ripple_envelope_zeros_at_t_zero(self):
        """At t=0, no train has had time to start → envelope sum is 0."""
        cfg = SimConfig()
        cfg.num_boids = 20
        cfg.mode = "field"
        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        r = Ripple()
        r._t = 0.0
        r.apply(flock, _make_ctx(flock, cfg))
        # All trains: t-offset < 0, so env=0 → no forces
        assert np.all(cfg._ripple_envelope_sum == 0.0)
        assert np.allclose(flock.accelerations[flock.active], 0.0)


class TestRippleTrainsConfig:
    """C3: field_ripple_trains config wiring."""

    def test_field_ripple_trains_changes_number_of_trains(self):
        """C3: field_ripple_trains=1 vs 5 produces different force patterns."""
        cfg_1 = SimConfig()
        cfg_1.num_boids = 50
        cfg_1.mode = "field"
        cfg_1.field_ripple_trains = 1

        cfg_5 = SimConfig()
        cfg_5.num_boids = 50
        cfg_5.mode = "field"
        cfg_5.field_ripple_trains = 5

        flock_1 = PhysicsFlock(cfg_1)
        flock_1.accelerations[:] = 0.0
        r1 = Ripple()
        r1._t = 15.0  # all trains active
        r1.apply(flock_1, _make_ctx(flock_1, cfg_1))

        flock_5 = PhysicsFlock(cfg_5)
        flock_5.accelerations[:] = 0.0
        r5 = Ripple()
        r5._t = 15.0  # all trains active
        r5.apply(flock_5, _make_ctx(flock_5, cfg_5))

        # Different number of trains → different force patterns
        # (5-train envelope sums are different from 1-train)
        env_1 = cfg_1._ripple_envelope_sum
        env_5 = cfg_5._ripple_envelope_sum
        assert not np.allclose(env_1, env_5, atol=1e-4), (
            "field_ripple_trains=1 vs 5 must produce different ripple envelopes"
        )

    def test_field_ripple_trains_minimum_one(self):
        """C3: field_ripple_trains=0 or negative → clamps to 1 train."""
        cfg = SimConfig()
        cfg.num_boids = 20
        cfg.mode = "field"
        cfg.field_ripple_trains = 0  # should clamp to 1

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        r = Ripple()
        r._t = 15.0
        r.apply(flock, _make_ctx(flock, cfg))
        # Should not crash — minimum 1 train
        assert np.isfinite(flock.accelerations).all()
