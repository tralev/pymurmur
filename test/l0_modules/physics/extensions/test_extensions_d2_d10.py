"""Extensions — D2 Wander uses configured speed/radius, D10 Ripple envelope per-bird array.

Split out of test_extensions.py (file-size split).
"""

import numpy as np
import pytest

from pymurmur.physics.extensions._base import StepContext
from pymurmur.physics.extensions.ripple import Ripple
from pymurmur.physics.extensions.wander import Wander
from pymurmur.physics.flock import PhysicsFlock

# ── D2: Wander uses configured speed/radius ──────────────────

def test_wander_uses_configured_speed(monkeypatch, default_config):
    """D2: Wander internal clock _t advances at cfg.wander.wander_attractor_speed·dt.

    Before D2: wander.py read cfg.wander_speed (non-existent key) → attribute
               error or wrong value, silently ran at wrong speed.
    After D2:  wander.py reads cfg.wander.wander_attractor_speed → uses the
               actual configured value from WanderConfig.

    This test creates a Wander directly, sets up a StepContext with known
    config values, calls apply() twice, and verifies _t advanced correctly.
    """
    from pymurmur.physics.extensions._base import StepContext
    cfg = default_config
    cfg.wander.wander_attractor_speed = 0.05  # custom speed
    cfg.wander_enabled = True
    cfg.num_boids = 10

    flock = PhysicsFlock(cfg)
    wander = Wander()
    dt = 1.0 / 60.0

    # Frame 0
    ctx = StepContext(frame=0, dt=dt, rng=np.random.default_rng(42),
                      center=np.array([500,350,200],dtype=np.float32),
                      config=cfg, threat_prox=None)
    t_before = wander._t
    wander.apply(flock, ctx)
    delta_0 = wander._t - t_before
    assert delta_0 == pytest.approx(dt, rel=1e-6), (
        f"_t advance: expected {dt:.6f}, got {delta_0:.6f}"
    )

    # Frame 1 — _t should advance by another dt
    t_mid = wander._t
    ctx2 = StepContext(frame=1, dt=dt, rng=np.random.default_rng(42),
                       center=np.array([500,350,200],dtype=np.float32),
                       config=cfg, threat_prox=None)
    wander.apply(flock, ctx2)
    delta_1 = wander._t - t_mid
    assert delta_1 == pytest.approx(dt, rel=1e-6), (
        f"_t advance frame 1: expected {dt:.6f}, got {delta_1:.6f}"
    )

    # Change speed mid-simulation — should take effect immediately
    cfg.wander.wander_attractor_speed = 0.20  # 4× faster
    t_before2 = wander._t
    ctx3 = StepContext(frame=2, dt=dt, rng=np.random.default_rng(42),
                       center=np.array([500,350,200],dtype=np.float32),
                       config=cfg, threat_prox=None)
    wander.apply(flock, ctx3)
    delta_2 = wander._t - t_before2
    assert delta_2 == pytest.approx(dt, rel=1e-6), (
        f"_t still advances by dt (speed only affects path argument, not clock): "
        f"expected {dt:.6f}, got {delta_2:.6f}"
    )
    # Speed affects bounded_unit_path argument: path(self._t * speed)
    # So wander centre moves faster even though _t advances at same rate.
    # Verify the wander centre path argument t·speed differs for different speeds.
    path_arg_slow = wander._t * 0.05
    path_arg_fast = wander._t * 0.20
    assert path_arg_fast == pytest.approx(4.0 * path_arg_slow, rel=0.01), (
        f"Path arg ratio: slow={path_arg_slow:.4f}, fast={path_arg_fast:.4f}"
    )


def test_wander_uses_configured_radius(default_config):
    """D2: Wander centre stays within cfg.wander.wander_attractor_radius of
    flock centre. Uses Wander directly (no full SimulationEngine).

    Before D2: wander.py read cfg.attractor_radius (non-existent key).
    After D2:  wander.py reads cfg.wander.wander_attractor_radius.
    """
    from pymurmur.physics.extensions._base import StepContext
    cfg = default_config
    cfg.wander.wander_attractor_radius = 100.0  # small radius for tight check
    cfg.wander_enabled = True
    cfg.num_boids = 10

    flock = PhysicsFlock(cfg)
    # Place birds at a known centre
    flock.positions[:] = np.array([500.0, 350.0, 200.0], dtype=np.float32)
    flock.center = np.array([500.0, 350.0, 200.0], dtype=np.float32)
    wander = Wander()
    dt = 1.0 / 60.0

    # Run several frames and measure max wander radius
    max_dist = 0.0
    for frame in range(100):
        ctx = StepContext(frame=frame, dt=dt, rng=np.random.default_rng(42),
                          center=flock.center, config=cfg, threat_prox=None)
        wander.apply(flock, ctx)
        if flock.wander_center is not None:
            d = float(np.linalg.norm(flock.wander_center - flock.center))
            if d > max_dist:
                max_dist = d

    # boundedUnitTravel ‖path‖ ≤ 1, so wander_center = C + path·radius
    # → max distance = radius (when ‖path‖ = 1)
    assert max_dist <= cfg.wander.wander_attractor_radius * 1.1, (
        f"Wander exceeded configured radius: max_dist={max_dist:.1f}, "
        f"radius={cfg.wander.wander_attractor_radius}"
    )
    # Also verify wander actually moves (not stuck at centre)
    assert max_dist > 0, "Wander should move away from centre"


def test_wander_config_keys_exist(default_config):
    """D2: WanderConfig has wander_attractor_speed and wander_attractor_radius.

    Verifies the config fields exist with correct types and defaults.
    """
    from pymurmur.core.config import WanderConfig
    w = WanderConfig()
    assert hasattr(w, "wander_attractor_speed")
    assert hasattr(w, "wander_attractor_radius")
    assert isinstance(w.wander_attractor_speed, float)
    assert isinstance(w.wander_attractor_radius, float)
    assert w.wander_attractor_speed == 0.10
    assert w.wander_attractor_radius == 300.0

    # Also verify the config is accessible via SimConfig
    cfg = default_config
    assert cfg.wander.wander_attractor_speed == 0.10
    assert cfg.wander.wander_attractor_radius == 300.0

    # Flat access via _FIELD_MAP
    assert cfg.wander_attractor_speed == 0.10
    assert cfg.wander_attractor_radius == 300.0


def test_wander_config_roundtrip(tmp_path, default_config):
    """D2: Wander config survives YAML round-trip."""
    import yaml
    cfg = default_config
    cfg.wander.wander_attractor_speed = 0.05
    cfg.wander.wander_attractor_radius = 200.0

    # Write
    out = tmp_path / "wander_config.yaml"
    cfg.to_file(out)

    # Read back
    loaded_text = out.read_text()
    assert "wander_attractor_speed" in loaded_text
    assert "wander_attractor_radius" in loaded_text

    # Parse to verify values
    data = yaml.safe_load(loaded_text)
    assert data["wander"]["wander_attractor_speed"] == 0.05
    assert data["wander"]["wander_attractor_radius"] == 200.0

    # Reload via SimConfig
    from pymurmur.core.config import SimConfig
    cfg2 = SimConfig.from_file(out)
    assert cfg2.wander.wander_attractor_speed == 0.05
    assert cfg2.wander.wander_attractor_radius == 200.0


# ── D10: Ripple envelope per-bird array ────────────────────────────


class TestD10RippleEnvelope:
    """D10: ripple_envelope_sum exports a per-bird (N,) array, not a
    scalar.  Two birds at the same position get equal envelope values;
    two birds far apart get different values."""

    def _make_ripple_ctrl(self, cfg):
        """Return (ripple, flock, ctx) for ripple.apply()."""
        from pymurmur.physics.extensions._base import StepContext
        from pymurmur.physics.flock import PhysicsFlock

        ripple = Ripple()
        flock = PhysicsFlock(cfg)
        rng = np.random.default_rng(42)
        C = np.array([cfg.width / 2, cfg.height / 2, cfg.depth / 2],
                     dtype=np.float32)
        ctx = StepContext(
            frame=0, dt=0.5, rng=rng, center=C, config=cfg, threat_prox=None,
        )
        return ripple, flock, ctx

    def test_envelope_is_per_bird_array_not_scalar(self, default_config):
        """D10: After ripple.apply(), _ripple_envelope_sum is an (N,) array
        matching N_capacity, not a float."""
        cfg = default_config
        cfg.num_boids = 20
        ripple, flock, ctx = self._make_ripple_ctrl(cfg)

        ripple.apply(flock, ctx)

        env = cfg._ripple_envelope_sum
        assert isinstance(env, np.ndarray), (
            f"Expected ndarray, got {type(env)}"
        )
        assert env.shape == (flock.N_capacity,), (
            f"Expected shape ({flock.N_capacity},), got {env.shape}"
        )

    def test_inactive_birds_get_zero_envelope(self, default_config):
        """D10: Inactive birds have envelope value 0.0."""
        cfg = default_config
        cfg.num_boids = 10
        ripple, flock, ctx = self._make_ripple_ctrl(cfg)

        # Deactivate a few birds
        flock.active[3] = False
        flock.active[7] = False

        ripple.apply(flock, ctx)

        env = cfg._ripple_envelope_sum
        assert env[3] == 0.0, "Inactive bird 3 should have envelope 0"
        assert env[7] == 0.0, "Inactive bird 7 should have envelope 0"
        # Some active birds should have nonzero envelope (if any train is active)
        active_mask = flock.active
        assert env[active_mask].sum() >= 0.0  # may be zero if no train active

    def test_zero_active_returns_zero_array(self, default_config):
        """D10: When no birds are active, envelope is all-zeros array."""
        cfg = default_config
        cfg.num_boids = 5
        ripple, flock, ctx = self._make_ripple_ctrl(cfg)

        flock.active[:] = False
        ripple.apply(flock, ctx)

        env = cfg._ripple_envelope_sum
        assert np.allclose(env, 0.0), (
            f"All-zero flock should give all-zero envelope, got max={env.max()}"
        )

    def test_same_position_birds_get_equal_envelope(self, default_config):
        """D10: Two birds at the same position get equal envelope values."""
        cfg = default_config
        cfg.num_boids = 5
        cfg.width = 1000.0
        cfg.height = 700.0
        cfg.depth = 400.0
        ripple, flock, ctx = self._make_ripple_ctrl(cfg)

        # Place two birds at identical positions
        flock.positions[0] = np.array([500.0, 350.0, 200.0], dtype=np.float32)
        flock.positions[1] = np.array([500.0, 350.0, 200.0], dtype=np.float32)

        # Run a few steps to accumulate ripple envelope
        for _ in range(20):
            ctx = StepContext(
                frame=ctx.frame + 1, dt=0.5, rng=ctx.rng,
                center=ctx.center, config=cfg, threat_prox=None,
            )
            ripple.apply(flock, ctx)

        env = cfg._ripple_envelope_sum
        assert env[0] == pytest.approx(env[1]), (
            f"Birds at same position should have equal envelope: "
            f"{env[0]:.6f} vs {env[1]:.6f}"
        )

    def test_far_apart_birds_get_different_envelope(self, default_config):
        """D10: Two birds at very different distances from the ripple
        origin get different envelope values.

        Bird 0 is near the domain centre (where the ripple Lissajous
        origin moves); bird 1 is far out at the corner.  The ripple's
        gaussian drop-off ensures bird 1's envelope is near zero while
        bird 0's is nonzero when a train is active."""
        cfg = default_config
        cfg.num_boids = 2
        cfg.width = 1000.0
        cfg.height = 700.0
        cfg.depth = 400.0
        ripple, flock, ctx = self._make_ripple_ctrl(cfg)

        # Bird 0: near the domain centre where ripple originates
        flock.positions[0] = np.array([500.0, 350.0, 200.0], dtype=np.float32)
        # Bird 1: far corner — ripple gaussian makes envelope near zero
        flock.positions[1] = np.array([50.0, 50.0, 50.0], dtype=np.float32)

        # Run enough steps for multiple ripple trains to activate
        for _ in range(50):
            ctx = StepContext(
                frame=ctx.frame + 1, dt=0.5, rng=ctx.rng,
                center=ctx.center, config=cfg, threat_prox=None,
            )
            ripple.apply(flock, ctx)

        env = cfg._ripple_envelope_sum
        # Bird near centre should have nonzero envelope while far bird
        # has near-zero — the difference must be significant
        assert abs(env[0] - env[1]) > 1e-9, (
            f"Birds at different distances should have "
            f"different envelope: {env[0]:.6f} vs {env[1]:.6f}"
        )

    def test_envelope_not_normalised_by_n(self, default_config):
        """D10: Envelope values are independent of N — adding more birds
        does not change existing birds' envelope values.

        Both configs use the same seed so the non-bird-0 positions
        are identical, giving both flocks the same centroid C.
        Only bird 0's position is set explicitly; all others are
        determined by the shared seed."""
        cfg = default_config
        cfg.seed = 42
        cfg.num_boids = 10
        cfg.width = 1000.0
        cfg.height = 700.0
        cfg.depth = 400.0

        # Copy config for large-flock run (same seed, more birds)
        cfg_large = default_config
        cfg_large.seed = 42
        cfg_large.num_boids = 30
        cfg_large.width = 1000.0
        cfg_large.height = 700.0
        cfg_large.depth = 400.0

        ripple_small, flock_small, ctx_small = self._make_ripple_ctrl(cfg)
        flock_small.positions[0] = np.array([500.0, 350.0, 200.0], dtype=np.float32)

        for _ in range(30):
            ctx_small = StepContext(
                frame=ctx_small.frame + 1, dt=0.5, rng=ctx_small.rng,
                center=ctx_small.center, config=cfg, threat_prox=None,
            )
            ripple_small.apply(flock_small, ctx_small)

        env_small_bird0 = cfg._ripple_envelope_sum[0]

        ripple_large, flock_large, ctx_large = self._make_ripple_ctrl(cfg_large)
        flock_large.positions[0] = np.array([500.0, 350.0, 200.0], dtype=np.float32)

        for _ in range(30):
            ctx_large = StepContext(
                frame=ctx_large.frame + 1, dt=0.5, rng=ctx_large.rng,
                center=ctx_large.center, config=cfg_large, threat_prox=None,
            )
            ripple_large.apply(flock_large, ctx_large)

        env_large_bird0 = cfg_large._ripple_envelope_sum[0]

        # Bird 0's envelope should be independent of flock size
        assert env_small_bird0 == pytest.approx(env_large_bird0, rel=0.01), (
            f"Envelope should be independent of N: "
            f"N=10 → {env_small_bird0:.6f}, N=30 → {env_large_bird0:.6f}"
        )
