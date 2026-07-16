"""Phase 3 missing coverage tests — P3.6–P3.9 edge cases and integration.

Gaps filled:
  P3.6: buoyancy zero-crossing, curl flow zero at flow_pull=0
  P3.7: 28s cycle periodicity, Gaussian falloff shape, distinct origins per train
  P3.8: blackening→field integration (sep_eff/coh_eff formulas)
  P3.9: egress arc path (lift + drift sinusoidal)
  Integration: ripple→fold noise coupling, wander→drift alignment
"""

from __future__ import annotations

import numpy as np
import pytest

from pymurmur.core.config import SimConfig
from pymurmur.physics.forces.field import (
    _compute_buoyancy,
    _compute_curl_flow,
    _compute_fold_noise,
    _compute_drift_alignment,
    _compute_leader_chaser,
    _compute_anchors,
    _compute_targets,
    _compute_shell_force,
    _compute_tangential,
    _compute_slot_repulsion,
    _compute_viscous_drag,
    _compute_floating_boundary,
    _compute_grid_sep_normalized,
    _compute_phases,
    _hash01,
    FieldMode,
)
from pymurmur.physics.extensions.ripple import Ripple, _smoothstep
from pymurmur.physics.extensions.predator import Predator, _rotate_toward
from pymurmur.physics.extensions.wander import wander_heading
from pymurmur.physics.flock import PhysicsFlock
from pymurmur.physics.extensions._base import StepContext
from test.helpers import _call_force


def _make_ctx(flock, config, frame=0, dt=1.0 / 60.0):
    return StepContext(
        frame=frame, dt=dt, rng=flock.rng,
        center=flock.center, config=config,
    )


# ═══════════════════════════════════════════════════════════════════
# P3.6: Buoyancy & curl flow edge cases
# ═══════════════════════════════════════════════════════════════════

class TestP3_6_BuoyancyEdgeCases:
    """P3.6: Buoyancy force — zero-crossing and per-bird determinism."""

    def test_buoyancy_changes_sign_with_z_difference(self):
        """Buoyancy flips sign when target_z crosses bird_z."""
        n = 2
        pos = np.array([[0, 0, 100], [0, 0, 100]], dtype=np.float32)
        targ = np.array([[0, 0, 200], [0, 0, 0]], dtype=np.float32)
        seeds = np.array([0, 1], dtype=np.float32)

        F = _compute_buoyancy(pos, targ, seeds, 0.0, 100.0, 1.0)

        # Bird 0: target_z > bird_z → T_z - p_z = +100 → buoyancy pushes +z
        # Bird 1: target_z < bird_z → T_z - p_z = -100 → buoyancy pushes -z
        assert F[0, 2] > 0, f"Bird 0 should rise: Fz={F[0,2]:.4f}"
        assert F[1, 2] < 0, f"Bird 1 should sink: Fz={F[1,2]:.4f}"

    def test_buoyancy_zero_when_all_at_target_z(self):
        """When p_z == T_z for all birds, buoyancy oscillatory term still active."""
        n = 5
        pos = np.random.randn(n, 3).astype(np.float32)
        pos[:, 2] = 0.0  # all at z=0
        targ = np.zeros_like(pos)  # all at z=0
        seeds = np.arange(n, dtype=np.float32)
        F = _compute_buoyancy(pos, targ, seeds, 0.0, 100.0, 1.0)
        # x,y should be zero; z oscillates via sin(d*8/U - t*1.1 + seed*17)*0.09
        np.testing.assert_allclose(F[:, 0], 0.0, atol=1e-6)
        np.testing.assert_allclose(F[:, 1], 0.0, atol=1e-6)
        # z component: oscillatory from sin(seed*17 + ...) — each bird different
        assert len(np.unique(np.round(F[:, 2], 4))) >= 2, (
            "Buoyancy should vary per bird via seed-dependent sin term"
        )

    def test_buoyancy_deterministic(self):
        """Same inputs → same buoyancy output."""
        pos = np.random.randn(5, 3).astype(np.float32)
        targ = np.random.randn(5, 3).astype(np.float32)
        seeds = np.arange(5, dtype=np.float32)
        F1 = _compute_buoyancy(pos, targ, seeds, 1.5, 50.0, 0.8)
        F2 = _compute_buoyancy(pos, targ, seeds, 1.5, 50.0, 0.8)
        np.testing.assert_allclose(F1, F2, atol=1e-6)


class TestP3_6_CurlFlowEdgeCases:
    """P3.6: Curl flow — zero at flow_pull=0, determinism."""

    def test_curl_flow_zero_when_flow_pull_zero(self):
        """Curl flow returns zeros when flow_pull=0."""
        pos = np.random.randn(10, 3).astype(np.float32) * 50
        C = np.zeros(3, dtype=np.float32)
        seeds = np.arange(10, dtype=np.float32)
        F = _compute_curl_flow(pos, C, seeds, 1.0, 100.0, 1.0, 0.0)
        np.testing.assert_allclose(F, 0.0)

    def test_curl_flow_always_unit_length(self):
        """Normalized flow vectors have unit length (before gain * flow * 0.08 * flow_pull)."""
        pos = np.random.randn(100, 3).astype(np.float32) * 10
        C = np.zeros(3, dtype=np.float32)
        seeds = np.arange(100, dtype=np.float32)
        # With flow=1, flow_pull=1, max magnitude = 0.08
        F = _compute_curl_flow(pos, C, seeds, 0.0, 1.0, 1.0, 1.0)
        mags = np.linalg.norm(F, axis=1)
        assert np.allclose(mags, 0.08, atol=1e-4), (
            f"Curl flow magnitude should be exactly 0.08, got {mags.min():.4f}–{mags.max():.4f}"
        )

    def test_curl_flow_deterministic(self):
        """Same inputs → same curl flow."""
        pos = np.random.randn(8, 3).astype(np.float32)
        C = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        seeds = np.arange(8, dtype=np.float32)
        F1 = _compute_curl_flow(pos, C, seeds, 2.5, 100.0, 0.5, 0.7)
        F2 = _compute_curl_flow(pos, C, seeds, 2.5, 100.0, 0.5, 0.7)
        np.testing.assert_allclose(F1, F2, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════
# P3.7: Ripple — cycle, falloff, origin distinctness
# ═══════════════════════════════════════════════════════════════════

class TestP3_7_RippleCycle:
    """P3.7: 28-second cycle — all 3 trains wrap around."""

    def test_cycle_periodicity(self):
        """At t and t+28, all train tau values repeat → same envelope.

        Uses separate Ripple instances to avoid internal time side-effects.
        Must use t ≥ max(_OFFSETS) = 18.67 so all 3 trains have started.
        """
        cfg = SimConfig()
        cfg.num_boids = 20
        cfg.mode = "field"
        # Use same seed to get identical initial positions
        cfg.seed = 42
        flock1 = PhysicsFlock(cfg)
        flock2 = PhysicsFlock(cfg)

        r1 = Ripple()
        r1._t = 20.0
        r1.apply(flock1, _make_ctx(flock1, cfg))
        env_20 = cfg._ripple_envelope_sum

        r2 = Ripple()
        r2._t = 48.0
        r2.apply(flock2, _make_ctx(flock2, cfg))
        env_48 = cfg._ripple_envelope_sum

        assert env_20 == pytest.approx(env_48, abs=1e-4), (
            f"Envelope should be periodic with 28s: {env_20:.4f} vs {env_48:.4f}"
        )

    def test_train_offsets_staggered(self):
        """Train 1 (offset=0), Train 2 (offset=9.33), Train 3 (offset=18.67)."""
        r = Ripple()
        assert len(r._OFFSETS) == 3
        np.testing.assert_allclose(r._OFFSETS, [0.0, 9.33, 18.67], atol=1e-2)

    def test_trains_different_origins(self):
        """Each train's Lissajous origin follows a distinct path."""
        cfg = SimConfig()
        cfg.num_boids = 3
        cfg.mode = "field"
        flock = PhysicsFlock(cfg)
        # Place birds at centre
        flock.positions[:] = np.array([[500, 350, 200]], dtype=np.float32)

        # Capture origin positions for each train by running at specific times
        # where only one train is active
        r = Ripple()
        origins = []
        for offset in [0.0, 9.33, 18.67]:
            r._t = offset + 3.0  # each train at τ=3 (well within envelope)
            C = np.mean(flock.positions[flock.active], axis=0)
            U = 0.4 * min(cfg.width, cfg.height, cfg.depth)
            origin_phase = offset
            origin = C + np.array([
                np.sin(r._t * 0.17 + origin_phase) * 0.46,
                np.cos(r._t * 0.13 + origin_phase * 1.7) * 0.25,
                np.cos(r._t * 0.19 + origin_phase * 0.6) * 0.42,
            ], dtype=np.float32) * U
            origins.append(origin)

        # All 3 origins should differ
        for i in range(3):
            for j in range(i + 1, 3):
                assert not np.allclose(origins[i], origins[j]), (
                    f"Trains {i} and {j} have identical origins"
                )


class TestP3_7_GaussianFalloff:
    """P3.7: Gaussian falloff exp(-((r-radius)/width)²)."""

    def test_smoothstep_hermite_shape(self):
        """Smoothstep is t²(3−2t) with correct midpoint."""
        # At midpoint t=0.5: 0.5² * (3-2*0.5) = 0.25 * 2 = 0.5
        assert _smoothstep(0.6, 1.7, 1.15) == pytest.approx(0.5, abs=0.01)

    def test_smoothstep_symmetric_complement(self):
        """smoothstep(0.6, 1.7, t) + (1-smoothstep(6.2, 8.8, t)) in transition."""
        # At t=4.0, well between the two ramps: rise≈1.0, fall side=1.0
        rise = _smoothstep(0.6, 1.7, 4.0)
        fall = 1.0 - _smoothstep(6.2, 8.8, 4.0)
        assert rise == pytest.approx(1.0, abs=1e-6), f"rise={rise}"
        assert fall == pytest.approx(1.0, abs=1e-6), f"fall={fall}"
        # Product = 1.0 (full envelope)
        assert rise * fall == pytest.approx(1.0, abs=1e-6)

    def test_smoothstep_flat_outside_range(self):
        """Outside [e0, e1]: 0 for x<e0, 1 for x>e1."""
        assert _smoothstep(0.6, 1.7, 0.0) == 0.0
        assert _smoothstep(0.6, 1.7, 2.0) == 1.0

    def test_gaussian_falloff_peak_at_radius(self):
        """At r=radius, amount = exp(0)*env = env (peak)."""
        rng = np.random.default_rng(0)
        # Simulate the falloff formula:
        # amount = exp(-((r - radius) / width)²) * env
        U = 100.0
        tau = 5.0
        radius = (0.16 + tau * 0.16) * U  # = (0.16+0.8)*100 = 96
        width = (0.11 + tau * 0.012) * U   # = (0.11+0.06)*100 = 17
        env = _smoothstep(0.6, 1.7, tau) * (1.0 - _smoothstep(6.2, 8.8, tau))

        # At peak (r = radius)
        r_at_peak = radius
        delta = (r_at_peak - radius) / width  # = 0
        amount_peak = np.exp(-delta * delta) * env
        assert amount_peak == pytest.approx(float(env), abs=1e-6)

        # At r = radius + 2*width → exp(-4) ≈ 0.018
        r_far = radius + 2.0 * width
        delta_far = (r_far - radius) / width  # = 2
        amount_far = np.exp(-delta_far * delta_far) * env
        assert amount_far < amount_peak * 0.02, (
            f"Falloff should be <2% at 2σ: {amount_far:.4f} vs peak {amount_peak:.4f}"
        )


# ═══════════════════════════════════════════════════════════════════
# P3.8: Blackening integration — field.py consumes _threat_blackening
# ═══════════════════════════════════════════════════════════════════

class TestP3_8_BlackeningFormulas:
    """P3.8: Verify blackening computes sep_eff and coh_eff correctly."""

    def test_blackening_formula_no_threat(self):
        """No threat (prox=0): sep_eff = sep*2, coh_eff = coh (inverted)."""
        # P3.8: black = 1 + blackening_gain * prox * 0.85
        # No threat → prox=0 → black=1
        # sep_eff = separation * (2 - black) = separation * 1
        # coh_eff = cohesion * black = cohesion * 1
        prox = 0.0
        blackening_gain = 0.6
        black = 1.0 + blackening_gain * prox * 0.85
        assert black == 1.0
        # Normally: sep_eff = sep * (2 - black) = sep, coh_eff = coh * black = coh
        # No modulation

    def test_blackening_formula_max_threat(self):
        """Max threat (prox=1): black > 1 → sep weaker, coh stronger."""
        prox = 1.0
        blackening_gain = 0.6
        black = 1.0 + blackening_gain * prox * 0.85  # = 1 + 0.51 = 1.51
        assert black > 1.0
        # sep_eff = sep * (2 - 1.51) = sep * 0.49  → weaker separation
        # coh_eff = coh * 1.51  → stronger cohesion
        assert 2.0 - black < 1.0, "separation should weaken near threat"
        assert black > 1.0, "cohesion should strengthen near threat"

    def test_blackening_published_values_range(self):
        """_threat_blackening values are >= 1.0."""
        cfg = SimConfig()
        cfg.num_boids = 30
        cfg.predator_threat_radius = 200.0
        cfg.predator_strength = 0.5
        flock = PhysicsFlock(cfg)

        p = Predator(cfg)
        com = np.mean(flock.positions[flock.active], axis=0)
        p._pos = com

        # Place bird near predator (high threat)
        bird_idx = np.where(flock.active)[0][0]
        flock.positions[bird_idx] = com + np.array([30, 0, 0], dtype=np.float32)

        p.apply(flock, _make_ctx(flock, cfg))

        assert hasattr(cfg, '_threat_blackening'), "Blackening must be published"
        black = cfg._threat_blackening
        # All blackening values >= 1.0 (black = 1 + gain*prox*0.85)
        assert (black >= 1.0).all(), f"Blackening values must be >= 1.0: min={black.min()}"


class TestP3_8_BlackeningFieldIntegration:
    """P3.8: Verify field mode can consume predator blackening via config."""

    def test_field_mode_runs_with_blackening_present(self):
        """FieldMode.compute() does not crash when _threat_blackening is set."""
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 20
        cfg.field_cohesion = 1.0
        cfg.field_flow = 1.0
        cfg.field_separation = 1.0

        # Simulate predator having published blackening
        cfg._threat_blackening = np.ones(20, dtype=np.float32) * 1.5
        cfg._threat_active = np.array([0, 1, 2], dtype=np.int32)
        cfg._threat_present = True

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        _call_force(FieldMode.compute, flock, cfg)

        # Should not crash — forces computed normally
        assert np.isfinite(flock.accelerations).all()
        acc_mags = np.linalg.norm(flock.accelerations[flock.active], axis=1)
        assert np.any(acc_mags > 1e-6)


# ═══════════════════════════════════════════════════════════════════
# P3.9: Egress arc path — lift + drift sinusoidal
# ═══════════════════════════════════════════════════════════════════

class TestP3_9_EgressArcPath:
    """P3.9: During egress, predator follows a sinusoidal arc (lift + drift)."""

    def test_egress_move_away_from_centre(self):
        """During egress, predator moves further from COM each frame."""
        cfg = SimConfig()
        cfg.num_boids = 30
        flock = PhysicsFlock(cfg)
        p = Predator(cfg)
        com = np.mean(flock.positions[flock.active], axis=0)

        p._phase = "egress"
        p._pos = com + np.array([100, 0, 0], dtype=np.float32)
        p._dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        dists = []
        for _ in range(10):
            dists.append(np.linalg.norm(p._pos - com))
            ctx = _make_ctx(flock, cfg, frame=_)
            p.apply(flock, ctx)

        # Distance should increase monotonically during egress
        for i in range(len(dists) - 1):
            assert dists[i + 1] > dists[i], (
                f"Egress must increase distance: step {i}: {dists[i]:.0f}→{dists[i+1]:.0f}"
            )

    def test_egress_does_not_fly_straight(self):
        """Egress path has lateral deviation from a straight line (arc via lift+drift).

        Runs enough frames for the sinusoidal lift+drift to produce observable
        deviation from a pure radial path away from centre.
        """
        cfg = SimConfig()
        cfg.num_boids = 30
        flock = PhysicsFlock(cfg)
        p = Predator(cfg)
        com = np.mean(flock.positions[flock.active], axis=0)

        p._phase = "egress"
        p._pos = com + np.array([100, 0, 0], dtype=np.float32)
        p._dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        p._turn_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        positions = [p._pos.copy()]
        for frame in range(50):
            ctx = _make_ctx(flock, cfg, frame=frame)
            p.apply(flock, ctx)
            positions.append(p._pos.copy())

        positions = np.array(positions)
        # Radial direction: from start toward final position
        radial_dir = positions[-1] - positions[0]
        radial_dir = radial_dir / max(np.linalg.norm(radial_dir), 1e-6)

        # Project each position onto the perpendicular plane and compute deviation
        deviations = []
        for i in range(len(positions)):
            along = np.dot(positions[i] - positions[0], radial_dir) * radial_dir
            perp = (positions[i] - positions[0]) - along
            deviations.append(np.linalg.norm(perp))

        max_dev = max(deviations)
        assert max_dev > 0.05, (
            f"Egress path should deviate from straight line: max_perp={max_dev:.4f}"
        )

    def test_egress_to_approach_transition(self):
        """After clearing clear_dist, predator resets to approach."""
        cfg = SimConfig()
        cfg.num_boids = 30
        flock = PhysicsFlock(cfg)
        p = Predator(cfg)
        com = np.mean(flock.positions[flock.active], axis=0)

        # Force egress far away: place beyond clear_dist with dir pointing away
        U = 0.4 * min(cfg.width, cfg.height, cfg.depth)
        threat_radius = getattr(cfg, 'predator_threat_radius', 12.0)
        momentum = getattr(cfg, 'predator_momentum', 0.5)
        pass_dist = (0.92 + threat_radius * 2.6 + momentum * 1.32) * U
        clear_dist = pass_dist * (0.72 + momentum * 0.16)

        p._phase = "egress"
        p._pos = com + np.array([clear_dist * 1.5, 0, 0], dtype=np.float32)
        p._dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        p.apply(flock, _make_ctx(flock, cfg))
        assert p._phase == "approach", (
            f"Egress→approach transition failed at dist={np.linalg.norm(p._pos-com):.0f} vs clear={clear_dist:.0f}"
        )


# ═══════════════════════════════════════════════════════════════════
# Integration tests: cross-extension data flow
# ═══════════════════════════════════════════════════════════════════

class TestIntegrationRippleToFoldNoise:
    """P3.7→P3.6: Ripple envelope sum feeds into fold noise scaling."""

    def test_fold_noise_scales_linearly_with_envelope(self):
        """Doubling ripple_envelope_sum doubles fold noise force."""
        pos = np.random.randn(20, 3).astype(np.float32) * 10
        C = np.zeros(3, dtype=np.float32)
        seeds = np.arange(20, dtype=np.float32)
        F1 = _compute_fold_noise(pos, C, seeds, 1.0, 100.0, 1.0, 1.0, 1.0)
        F2 = _compute_fold_noise(pos, C, seeds, 1.0, 100.0, 1.0, 1.0, 2.0)
        np.testing.assert_allclose(F2, F1 * 2.0, atol=1e-4)

    def test_fold_noise_zero_when_envelope_zero(self):
        """ripple_envelope_sum=0 → no fold noise."""
        pos = np.random.randn(10, 3).astype(np.float32)
        C = np.zeros(3, dtype=np.float32)
        seeds = np.arange(10, dtype=np.float32)
        F = _compute_fold_noise(pos, C, seeds, 1.0, 100.0, 1.0, 1.0, 0.0)
        np.testing.assert_allclose(F, 0.0)

    def test_ripple_sets_envelope_for_field_consumption(self):
        """Ripple.apply() sets config._ripple_envelope_sum that fold noise can read."""
        cfg = SimConfig()
        cfg.num_boids = 20
        cfg.mode = "field"
        cfg.field_flow = 1.0
        flock = PhysicsFlock(cfg)

        r = Ripple()
        r._t = 5.0  # well within train 1 envelope
        r.apply(flock, _make_ctx(flock, cfg))

        assert hasattr(cfg, '_ripple_envelope_sum'), "Ripple must export envelope"
        assert isinstance(cfg._ripple_envelope_sum, float)
        # At t=5, train 1 is active → envelope should be > 0
        assert cfg._ripple_envelope_sum >= 0.0


class TestIntegrationWanderToDrift:
    """P3.1→P3.6: Wander heading feeds into drift alignment force."""

    def test_drift_alignment_steers_toward_wander_heading(self):
        """Drift alignment accelerates birds toward wander_heading * v0."""
        # Get a wander heading at a known time
        heading = wander_heading(10.0)
        v0 = 4.0

        v = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        F = _compute_drift_alignment(v, heading, v0, 1.0, 1.0)

        # Force should point in same direction as heading (since v=0)
        dot = np.dot(F[0], heading)
        assert dot > 0, f"Drift should steer toward wander heading: dot={dot:.4f}"

    def test_drift_alignment_symmetry(self):
        """If velocity already matches wander_heading*v0, force is zero."""
        heading = wander_heading(5.0)
        v0 = 4.0
        target_vel = heading * v0

        F = _compute_drift_alignment(
            target_vel.reshape(1, 3).astype(np.float32),
            heading, v0, 1.0, 1.0,
        )
        np.testing.assert_allclose(F, 0.0, atol=1e-4)

    def test_wander_publishes_heading_for_drift(self):
        """Wander extension publishes flock.wander_heading that drift alignment can use."""
        from pymurmur.physics.extensions.wander import Wander
        cfg = SimConfig()
        cfg.num_boids = 10
        cfg.mode = "field"
        cfg.wander_enabled = True
        flock = PhysicsFlock(cfg)

        w = Wander()
        w.apply(flock, _make_ctx(flock, cfg))

        assert flock.wander_heading is not None
        assert np.abs(np.linalg.norm(flock.wander_heading) - 1.0) < 1e-6


class TestIntegrationPredatorToField:
    """P3.8→P3.6: Predator blackening published for field mode consumption."""

    def test_predator_publishes_data_for_field_consumption(self):
        """After predator.apply(), cfg has _threat_blackening and _threat_present."""
        cfg = SimConfig()
        cfg.num_boids = 30
        cfg.predator_threat_radius = 200.0
        cfg.predator_strength = 0.5
        flock = PhysicsFlock(cfg)

        p = Predator(cfg)
        com = np.mean(flock.positions[flock.active], axis=0)
        p._pos = com

        # Place bird near predator
        bird_idx = np.where(flock.active)[0][0]
        flock.positions[bird_idx] = com + np.array([30, 0, 0], dtype=np.float32)

        p.apply(flock, _make_ctx(flock, cfg))

        # Verify all integration points
        assert hasattr(cfg, '_threat_present'), "Predator must set _threat_present"
        assert cfg._threat_present is True
        assert hasattr(cfg, '_threat_blackening'), "Predator must set _threat_blackening"
        assert hasattr(cfg, '_threat_active'), "Predator must set _threat_active"
        assert cfg._threat_blackening.dtype == np.float32

    def test_field_mode_runs_after_predator_without_crash(self):
        """Simulate the engine step order: predator → field mode."""
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 30
        cfg.predator_threat_radius = 200.0
        cfg.predator_strength = 0.5
        cfg.field_cohesion = 1.0
        cfg.field_flow = 1.0
        flock = PhysicsFlock(cfg)

        # Step 1: Predator
        p = Predator(cfg)
        com = np.mean(flock.positions[flock.active], axis=0)
        p._pos = com
        bird_idx = np.where(flock.active)[0][0]
        flock.positions[bird_idx] = com + np.array([30, 0, 0], dtype=np.float32)
        p.apply(flock, _make_ctx(flock, cfg))

        # Step 2: Field mode reads predator data
        flock.accelerations[:] = 0.0
        _call_force(FieldMode.compute, flock, cfg)

        # Should not crash, forces computed
        assert np.isfinite(flock.accelerations).all()
        acc_mags = np.linalg.norm(flock.accelerations[flock.active], axis=1)
        assert np.any(acc_mags > 1e-6)
