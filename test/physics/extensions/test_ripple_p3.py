"""Independent L0 function tests for P3.7 ripple envelopes.

Tests the _smoothstep helper, 3-train staggering, envelope bounds,
moving Lissajous origins, radial+twist forces, and envelope sum export.
"""

import numpy as np
import pytest

from pymurmur.core.config import SimConfig
from pymurmur.physics.extensions.ripple import Ripple, _smoothstep
from pymurmur.physics.flock import PhysicsFlock
from pymurmur.physics.extensions._base import StepContext


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
        """P3.7: ripple_envelope_sum is exported to config._ripple_envelope_sum."""
        cfg = SimConfig()
        cfg.num_boids = 20
        cfg.mode = "field"
        flock = PhysicsFlock(cfg)
        r = Ripple()
        r.apply(flock, _make_ctx(flock, cfg))
        assert hasattr(cfg, '_ripple_envelope_sum'), "Envelope sum must be exported"
        assert isinstance(cfg._ripple_envelope_sum, float)

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
        # Should not crash, envelope sum defaults to 0
        assert cfg._ripple_envelope_sum == 0.0

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
        assert cfg._ripple_envelope_sum == 0.0
        assert np.allclose(flock.accelerations[flock.active], 0.0)
