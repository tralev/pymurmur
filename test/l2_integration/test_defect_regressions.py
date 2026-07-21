"""Whole-system regression guards for the D1–D21 defect fixes (roadmap_deepseek.md).

Each defect has per-module tests next to the code it touched. This module
tests the defect *contracts through the full engine*, because every
regression found after the fixes landed was invisible at module level and
only surfaced when the pieces ran together:

- D10's per-bird ripple envelope crashed field mode's fold noise with a
  (N,3)×(N,) broadcast error — only when ripple and field ran in one step.
- D11's owns_positions honouring froze influencer birds — the mode never
  moved positions itself, so nothing did.
- D1's boundary centre used the EMA centroid tracker at flock level — the
  sphere followed the birds instead of bounding them, while the level-0
  integrate() tests (domain-centre default) stayed green.
- D12's globally-applied field_inertia softened the hard speed band that
  the P4 acceptance fuzz pins for spatial mode.

Rule of thumb for this file: build a real SimulationEngine, step it, and
assert on observable state. No mocking of the seams under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymurmur.core.config import SimConfig
from pymurmur.physics.forces import MODE_REGISTRY
from pymurmur.simulation.engine import SimulationEngine

ALL_MODES = sorted(MODE_REGISTRY.keys())


def _make_engine(mode: str, n: int = 20, seed: int | None = 1,
                 **overrides) -> SimulationEngine:
    cfg = SimConfig()
    cfg.seed = seed
    cfg.mode = mode
    cfg.num_boids = n
    for key, val in overrides.items():
        setattr(cfg, key, val)
    return SimulationEngine(cfg)


# ═══════════════════════════════════════════════════════════════════
# D11 (+ D12, D21): every mode advances positions and stays finite
# ═══════════════════════════════════════════════════════════════════


class TestAllModesAdvance:
    """D11 whole-system: whatever a mode declares (owns_positions or not),
    positions must actually advance when the engine steps. Guards against
    honouring an ownership flag no one implements."""

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_positions_advance_and_stay_finite(self, mode):
        eng = _make_engine(mode)
        p0 = eng.flock.positions.copy()
        for _ in range(5):
            eng.step(1.0 / 60.0)
        assert np.isfinite(eng.flock.positions).all(), f"{mode}: NaN/Inf positions"
        assert np.isfinite(eng.flock.velocities).all(), f"{mode}: NaN/Inf velocities"
        moved = np.linalg.norm(eng.flock.positions - p0, axis=1)
        assert moved[eng.flock.active].max() > 1e-3, (
            f"{mode}: no bird moved in 5 frames — position pipeline is dead "
            f"(owns_positions honoured but not implemented?)"
        )

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_owns_positions_modes_move_their_birds(self, mode):
        """D11: a mode claiming owns_positions must mutate positions in
        compute() — integrate() is called with move=False for it."""
        cls = MODE_REGISTRY[mode]
        if not getattr(cls, "owns_positions", False):
            pytest.skip(f"{mode} does not own positions")
        eng = _make_engine(mode)
        p0 = eng.flock.positions.copy()
        eng.step(1.0 / 60.0)
        assert not np.allclose(p0, eng.flock.positions), (
            f"{mode}: owns_positions=True but compute() left positions "
            f"untouched and integrate(move=False) skipped them"
        )


# ═══════════════════════════════════════════════════════════════════
# D10: per-bird ripple envelope survives the full field pipeline
# ═══════════════════════════════════════════════════════════════════


class TestRippleFieldPipeline:
    """D10 whole-system: ripple exports a per-bird (N_capacity,) envelope
    and field-mode fold noise must broadcast it — at any N."""

    @pytest.mark.parametrize("n", [10, 200])
    def test_field_with_ripple_steps_cleanly(self, n):
        eng = _make_engine("field", n=n, ripple_enabled=True)
        for _ in range(10):
            eng.step(1.0 / 60.0)
        env = getattr(eng.config, "_ripple_envelope_sum", None)
        assert isinstance(env, np.ndarray), (
            "D10: ripple must export a per-bird envelope array"
        )
        assert env.shape == (len(eng.flock.positions),)
        assert np.isfinite(eng.flock.last_accelerations).all()

    def test_envelope_padded_for_inactive_birds(self):
        eng = _make_engine("field", n=30, ripple_enabled=True)
        eng.flock.active[20:] = False
        for _ in range(5):
            eng.step(1.0 / 60.0)
        env = eng.config._ripple_envelope_sum
        assert np.all(env[~eng.flock.active] == 0.0), (
            "D10: inactive birds must have zero envelope"
        )


# ═══════════════════════════════════════════════════════════════════
# D1: sphere boundary is domain-centred at engine level
# ═══════════════════════════════════════════════════════════════════


class TestSphereBoundaryDomainCentred:
    """D1 whole-system: the boundary sphere must be centred on the domain
    centre C — not the origin, and not the EMA flock-centroid tracker
    (which would make the boundary follow the flock instead of bounding it)."""

    @pytest.mark.parametrize("mode", ["spatial", "projection"])
    def test_hard_sphere_contains_flock_around_C(self, mode):
        eng = _make_engine(mode, n=30, boundary_mode="sphere",
                           boundary_sphere_radius=150.0)
        C = np.array([eng.config.width / 2, eng.config.height / 2,
                      eng.config.depth / 2], dtype=np.float32)
        for _ in range(30):
            eng.step(1.0 / 60.0)
            dists = np.linalg.norm(
                eng.flock.positions[eng.flock.active] - C, axis=1
            )
            assert (dists <= 150.0 + 1.0).all(), (
                f"{mode}: bird escaped the domain-centred sphere "
                f"(max {dists.max():.1f} > 150) — boundary centre drifted?"
            )

    def test_boundary_does_not_follow_centroid(self):
        """Even when the EMA centroid tracker has drifted far from C
        (flock packed in a corner), the sphere still bounds around C."""
        eng = _make_engine("spatial", n=20, boundary_mode="sphere",
                           boundary_sphere_radius=150.0)
        # Pack the flock into a corner so update_center() drags the EMA
        # tracker away from the domain centre.
        eng.flock.positions[:] = eng.flock.positions * 0.1  # near origin corner
        C = np.array([eng.config.width / 2, eng.config.height / 2,
                      eng.config.depth / 2], dtype=np.float32)
        for _ in range(30):
            eng.step(1.0 / 60.0)
        dists = np.linalg.norm(
            eng.flock.positions[eng.flock.active] - C, axis=1
        )
        assert (dists <= 150.0 + 1.0).all(), (
            "D1: sphere boundary followed the flock centroid instead of "
            "staying centred on the domain centre"
        )


# ═══════════════════════════════════════════════════════════════════
# D12: field_inertia must not soften other modes' speed band
# ═══════════════════════════════════════════════════════════════════


class TestSpeedBandHardOutsideFieldMode:
    """D12 whole-system: the raw/clamped inertia lerp is a field-mode
    feature. In band-clamped modes the speed band is a hard contract
    (P4 acceptance fuzz pins 10k frames; this is the fast guard)."""

    def test_spatial_band_holds_with_default_inertia(self):
        eng = _make_engine("spatial", n=15, v0=4.0, visual_range=200.0)
        v0 = eng.config.v0
        for frame in range(60):
            eng.step(1.0 / 60.0)
            speeds = np.linalg.norm(
                eng.flock.velocities[eng.flock.active], axis=1
            )
            assert (speeds >= 0.3 * v0 - 0.01).all(), (
                f"frame {frame}: speed floor broken "
                f"({speeds.min():.2f} < {0.3 * v0:.2f}) — inertia lerp "
                f"leaking into non-field mode?"
            )
            assert (speeds <= v0 + 0.01).all(), (
                f"frame {frame}: speed cap broken ({speeds.max():.2f} > {v0:.2f})"
            )


# ═══════════════════════════════════════════════════════════════════
# D6: seed semantics end-to-end
# ═══════════════════════════════════════════════════════════════════


class TestSeedSemanticsEndToEnd:
    """D6 whole-system: seed=0 is deterministic, seed=None is fresh
    entropy — observed through full engine trajectories, not just rng
    construction."""

    def _trajectory(self, seed, steps=10):
        eng = _make_engine("spatial", n=15, seed=seed)
        for _ in range(steps):
            eng.step(1.0 / 60.0)
        return eng.flock.positions.copy()

    def test_seed_zero_is_deterministic(self):
        np.testing.assert_array_equal(self._trajectory(0), self._trajectory(0))

    def test_seed_none_gives_fresh_entropy(self):
        assert not np.allclose(self._trajectory(None), self._trajectory(None)), (
            "seed=None produced identical runs — 0/None conflation is back"
        )

    def test_seed_zero_differs_from_seed_one(self):
        assert not np.allclose(self._trajectory(0), self._trajectory(1))


# ═══════════════════════════════════════════════════════════════════
# D18: metrics read real (pre-zeroing) forces through the engine
# ═══════════════════════════════════════════════════════════════════


class TestMetricsSeeRealForces:
    """D18 whole-system: after engine.step(), collected metrics must show
    non-zero force/power (accelerations are zeroed at the end of
    integrate(); metrics read the pre-zeroing stash)."""

    def test_force_avg_nonzero_after_steps(self):
        eng = _make_engine("spatial", n=30)
        for _frame in range(5):
            eng.step(1.0 / 60.0)
        eng.metrics.collect(eng.flock, frame=5)
        snap = eng.metrics.snapshot()
        assert np.isfinite(snap.force_avg)
        assert snap.force_avg > 0.0, (
            "force_avg == 0 after stepping a live flock — metrics are "
            "reading the zeroed accelerations again (D18)"
        )


# ═══════════════════════════════════════════════════════════════════
# D19: history accumulators bounded through the engine loop
# ═══════════════════════════════════════════════════════════════════


class TestHistoryCapsThroughEngine:
    """D19 whole-system: stepping the engine far past history_cap keeps
    the metrics history ring-buffered."""

    def test_metrics_history_bounded(self):
        cfg = SimConfig()
        cfg.seed = 1
        cfg.mode = "spatial"
        cfg.num_boids = 10
        cfg.history_cap = 5
        eng = SimulationEngine(cfg)
        for frame in range(12):
            eng.step(1.0 / 60.0)
            eng.metrics.collect(eng.flock, frame=frame)
        assert len(eng.metrics._history) <= 5, (
            f"metrics history grew to {len(eng.metrics._history)} "
            f"despite history_cap=5"
        )


# ═══════════════════════════════════════════════════════════════════
# D8: steric clamp engaged at production defaults, in the engine
# ═══════════════════════════════════════════════════════════════════


class TestStericClampInProduction:
    """D8 whole-system: two overlapping birds in projection mode with the
    production max_force default must not produce exploding forces."""

    def test_overlapping_pair_forces_bounded(self):
        eng = _make_engine("projection", n=10)
        # Force two birds to near-overlap: raw 1/d² would be ~10⁴.
        eng.flock.positions[1] = eng.flock.positions[0] + np.float32(0.01)
        eng.step(1.0 / 60.0)
        acc = eng.flock.last_accelerations[eng.flock.active]
        assert np.isfinite(acc).all()
        mags = np.linalg.norm(acc, axis=1)
        assert mags.max() < 5.0, (
            f"force {mags.max():.1f} at d≈0.01 — steric clamp not engaged "
            f"at production max_force (D8)"
        )


# ═══════════════════════════════════════════════════════════════════
# D3: clear-all pipeline through the engine command queue
# ═══════════════════════════════════════════════════════════════════


class TestClearAllPipeline:
    """D3 whole-system: enqueue_clear() → next step drains the command
    and deactivates every bird."""

    def test_enqueue_clear_deactivates_all(self):
        eng = _make_engine("spatial", n=20)
        eng.step(1.0 / 60.0)
        assert eng.flock.active.any()
        eng.enqueue_clear()
        eng.step(1.0 / 60.0)
        assert not eng.flock.active.any(), (
            "clear-all command did not deactivate the flock"
        )
